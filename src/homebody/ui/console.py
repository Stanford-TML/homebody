"""Viser displays supplied frames; callbacks only enqueue session commands."""
import socket
import sys
import time
import traceback
from dataclasses import dataclass, replace
from html import escape
from itertools import pairwise
from queue import Empty, Queue
from threading import Condition, Event, Lock, Thread

import numpy as np

from homebody.session.settings import Topdown
from homebody.skills import navigate, pick, place
from homebody.skills.contract import SkillFailure
from homebody.ui.selection import (
    Pen,
    Selection,
    decision_topdown,
    draw_path,
)
from homebody.ui.theme import (
    BRAND,
    FACING,
    GOAL,
    HEADING,
    PANEL_WIDTH,
    ROBOT,
    ROUTE,
    STANCE,
    STYLE,
    header,
    note,
    section,
)
from homebody.vlm.observations import depth_preview, label_preview, map_preview

AWAITING = "Top-down: awaiting a decision"
STATUS = {"model_done": "Model reports done", "step_limit": "Step limit reached",
          "provider_error": "Provider error", "cancelled": "Cancelled", "error": "Error",
          "interrupted": "Interrupted"}


def claim(host, port):
    """Raise OSError while another server listens on HOST:PORT. TIME_WAIT connections do not count."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))


def blank(height, width):
    return np.full((height, width, 3), 235, dtype=np.uint8)


def thought_bubble(message="", *, thinking=False, model=""):
    """The agent bubble's HTML with the model's decision text."""
    title = "Agent" + (f": {model}" if model else "") + (" Thinking about the next step…" if thinking else "")
    text = message or ("" if thinking else "Ready for your task.")
    return (f'<div class="hb-bubble" role="status" aria-live="polite">'
            f'<div class="hb-title">{escape(title)}</div>'
            f'<div class="hb-message">{escape(text)}</div></div>')


def stage_progress(skill, stages, result=None, *, durations=None, simulated=None):
    """The stage panel's HTML for the reached stages and, once finished, the result."""
    count = len(stages)
    title = f"{skill.capitalize()}: {stages[-1]}" if stages else skill.capitalize()
    label = f"Stage {count}" if result is None else f"Result: {result.code}"
    if durations is not None:
        label += f" · {sum(durations):.1f}s elapsed"
        if simulated is not None:
            label += f" ({simulated:.1f}s sim)"
    recent = list(zip(stages, [None] * count if durations is None else durations, strict=True))[-5:]
    last_class = "is-current" if result is None else "is-done" if result.code == "OK" else "is-failed"
    steps = []
    for index, (name, duration) in enumerate(recent):
        css_class = last_class if index == len(recent) - 1 else "is-done"
        clock = f'<span class="hb-step-time">{duration:.1f}s</span>' if duration is not None else ""
        steps.append(f'<li class="{css_class}">{escape(name)}{clock}</li>')
    steps = "".join(steps)
    history = f'<ol class="hb-steps">{steps}</ol>' if steps else ""
    message = f'<div class="hb-stage-count">{escape(result.message)}</div>' if result is not None else ""
    return (f'<div class="hb-stage{" hb-result" if result is not None else ""}" role="status" aria-live="polite">'
            f'<div class="hb-stage-head"><strong>{escape(title)}</strong>'
            f'<span class="hb-stage-count">{escape(label)}</span></div>{history}{message}</div>')


@dataclass(frozen=True)
class Progress:
    """The stage panel's state, replaced whole so the render thread reads a consistent snapshot."""
    skill: str = ""
    stages: tuple = ()
    times: tuple = ()
    result: object = None
    finished_at: float | None = None
    sim_start: float | None = None
    sim_now: float | None = None


@dataclass(frozen=True)
class Command:
    name: str
    text: str = ""


class Controls:
    def __init__(self):
        self.queue = Queue()
        self.cancel = Event()
        self.closing = Event()

    def submit(self, text: str):
        if text.strip():
            self.queue.put(Command("task", text.strip()))

    def stop(self):
        """Cancel the running task and drop tasks that have not started."""
        self.cancel.set()
        with self.queue.mutex:
            kept = [command for command in self.queue.queue if command.name != "task"]
            self.queue.queue.clear()
            self.queue.queue.extend(kept)
        self.queue.put(Command("stop"))

    def reset(self):
        self.cancel.set()
        self.queue.put(Command("reset"))

    def next(self):
        try:
            return self.queue.get_nowait()
        except Empty:
            return None


def palm_path(arm, frame, plan):
    """World palm positions a prepared pick or place follows, from the frame it was planned from."""
    start = frame.palm(arm, plan.side)[:3, 3]
    if isinstance(plan, pick.Plan):
        waypoints = [(frame.world_torso @ arm.forward(q)[0])[:3, 3] for q in plan.approach_joints]
        return np.array([start, *waypoints, plan.pre[:3, 3], plan.palm[:3, 3]])
    return np.array([start, plan.pre[:3, 3], plan.palm[:3, 3]])


class Overlays:
    """What the cockpit draws over observations, derived from decisions and skill events. A prepared plan is drawn against the newest observation at its event."""
    def __init__(self, topdown_settings, arms=None):
        self.topdown_settings, self.arms = topdown_settings, arms or {}
        self.epoch = self.latest = None
        self.clear()

    def clear(self):
        self.decision = self.captured = self.selection = self.palm_path = self.resolved = None
        self.marks = {}

    def observe(self, frame):
        """Track the newest observation; True when a new epoch invalidated every overlay."""
        stale = frame.epoch != self.epoch
        if stale:
            self.clear()
            self.epoch = frame.epoch
        self.latest = frame
        return stale

    def select(self, decision, frame):
        self.clear()
        self.epoch, self.decision, self.captured = frame.epoch, decision, frame
        self.selection = Selection.from_decision(decision, frame)
        if decision.skill == "navigate":
            try:
                goal, facing, _, _ = navigate.goal_arguments(decision.arguments, frame.navigation)
            except SkillFailure:
                return
            self.marks = {"goal": goal} if facing is None else {"goal": goal, "facing": facing}

    def prepared(self, plan):
        """Add a prepared plan's overlays; returns the observation they are drawn against."""
        frame = self.captured if self.latest is None else self.latest
        route = getattr(plan, "drive_plan", plan)
        if isinstance(route, navigate.Plan):
            points = route.waypoints
            if route.departure_goal is not None:
                points = [frame.base_pose[:2], route.departure_goal, *points]
            yaw = float(frame.base_pose[2]) if route.yaw is None else route.yaw
            self.marks = {**self.marks, "route": points, "stance": [*route.goal, yaw]}
        if isinstance(plan, (pick.Plan, place.Plan)):
            arm = self.arms.get(plan.side)
            self.palm_path = None if arm is None else palm_path(arm, frame, plan)
        if isinstance(plan, place.Plan):
            self.resolved = plan.point
        return frame

    def returning(self, event):
        """Clear the approach path once the arm starts its return."""
        self.palm_path = None

    def ego_rgb(self, frame):
        """RGB with the selected segment, its click and the palm path."""
        selection, path = self.selection, self.palm_path  # clear() may run on another thread
        rgb = frame.rgb if selection is None else selection.overlay(frame, frame.rgb)
        if path is not None:
            rgb = draw_path(frame, rgb, path)
        return rgb

    def ego(self, frame):
        """Replay view: RGB, depth and segmentation from the same observation."""
        return self.ego_rgb(frame), depth_preview(frame.depth), label_preview(frame.labels)

    def map(self, frame):
        """The agent's map with the decision's marks, and the robot drawn over them."""
        pen, world_to_pixel = Pen(map_preview(frame)), np.linalg.inv(frame.navigation.T_map_px)

        def px(xy):
            point = world_to_pixel @ np.r_[np.asarray(xy, dtype=float)[:2], 1.]
            return point[:2] / point[2]

        def pose(xy_yaw, colour):
            x, y, yaw = xy_yaw
            pen.pose(px((x, y)), px((x + HEADING * np.cos(yaw), y + HEADING * np.sin(yaw))), colour)

        marks = self.marks
        if len(marks.get("route", ())) > 1:
            pen.path([px(point) for point in marks["route"]], ROUTE)
            for point in marks["route"][1:]:
                pen.dot(px(point), ROUTE)
        for name, colour in (("goal", GOAL), ("facing", FACING)):
            if name in marks:
                pen.ring(px(marks[name]), colour)
        if "stance" in marks:
            pose(marks["stance"], STANCE)
        pose(frame.base_pose, ROBOT)
        return pen.pixels()

    def topdown(self):
        """The decision's captured top-down: the requested release ring, the resolved dot."""
        return decision_topdown(self.decision, self.captured, self.topdown_settings, self.resolved)


class Console:
    def __init__(self, controls: Controls, *, host="127.0.0.1", port=8080, initial_task="",
                 scene_name="Simulation", model_name="", topdown_settings: Topdown | None = None,
                 render_in_background=False):
        """With RENDER_IN_BACKGROUND the live view and map are drawn on a UI thread from the newest frame."""
        import viser

        claim(host, port)
        self.server = viser.ViserServer(host=host, port=port)
        self.server.gui.configure_theme(titlebar_content=None,
                                        control_width="large", dark_mode=False,
                                        brand_color=BRAND,
                                        show_logo=False, show_share_button=False)
        self.server.gui.main_panel.dock_right()
        self.server.gui.main_panel.set_width(PANEL_WIDTH)
        self.server.gui.add_html(STYLE)
        self.server.gui.add_html(header(scene_name))
        self.scene_view = None
        self.model_name = model_name
        self.overlays = Overlays(topdown_settings if topdown_settings is not None else Topdown())
        self._detail_skill = None
        self.server.gui.add_html(section("Ego view"))
        self.ego = self.server.gui.add_image(blank(480, 640), label="Live camera", format="jpeg")
        self.task = self.server.gui.add_text("Instruction", initial_value=initial_task, multiline=True)
        self.submit_button = self.server.gui.add_button("Start task", icon="player-play", disabled=True)
        self.stop_button = self.server.gui.add_button("Stop", color="gray", icon="player-stop")
        self.status = self.server.gui.add_markdown("Loading the simulation. You can edit the instruction.")
        self.server.gui.add_html(section("Agent thoughts"))
        self.thought = self.server.gui.add_html(thought_bubble("Waiting for the simulation to load.",
                                                              model=self.model_name))
        self._bubble_message = ""
        self.stage = self.server.gui.add_html("")
        self._progress = Progress()
        self.explanation = self.server.gui.add_html("")
        with self.server.gui.add_folder("Details", expand_by_default=False):
            self.click = self.server.gui.add_image(blank(480, 640), label="Decision click", format="jpeg",
                                                   visible=False)
            self.map = self.server.gui.add_image(blank(240, 960), label="Navigation map", visible=False)
            self.topdown = self.server.gui.add_image(blank(240, 960), label=AWAITING, visible=False)
        self.controls, self._starting = controls, Lock()
        self.submit_button.on_click(self._start)
        self.stop_button.on_click(lambda _: controls.stop())
        self._pending, self._latest, self._closing = Condition(), None, False
        self._render_failed = False
        self._renderer = Thread(target=self._render_newest, name="console-render", daemon=True) \
            if render_in_background else None
        if self._renderer is not None:
            self._renderer.start()

    def _start(self, _):
        """One task per click: a second click before the session reports back is ignored."""
        with self._starting:
            if self.submit_button.disabled or not self.task.value.strip():
                return
            self.submit_button.disabled = True
        self.controls.submit(self.task.value)

    @property
    def arms(self):
        return self.overlays.arms

    @arms.setter
    def arms(self, arms):
        self.overlays.arms = arms

    def show(self, frame):
        if self.overlays.observe(frame):
            self.clear_selection()
        if self._renderer is None:
            self.render(frame)
            return
        with self._pending:
            self._latest = frame
            self._pending.notify()

    def _render_newest(self):
        """Draw the newest frame shown since the last drawing, skipping any shown meanwhile."""
        while True:
            with self._pending:
                self._pending.wait_for(lambda: self._latest is not None or self._closing)
                if self._closing:
                    return
                frame, self._latest = self._latest, None
            try:
                self.render(frame)
            except Exception:  # noqa: BLE001 -- one bad frame must not freeze the live view
                if not self._render_failed:
                    self._render_failed = True
                    print("console: rendering a frame failed; later failures are not printed",
                          file=sys.stderr)
                    traceback.print_exc()

    def render(self, frame):
        self._refresh_stage(simulation_time=frame.time)
        self.ego.image = self.overlays.ego_rgb(frame)
        self.ego.label = "Live camera"
        if self._detail_skill == "navigate":
            self.map.image = self.overlays.map(frame)
            self.map.label = f"Navigation map: {frame.time:.1f} s"

    def clear_selection(self):
        self.overlays.clear()
        self._detail_skill = None
        self.click.visible = False
        self.map.visible = False
        self.topdown.visible = False
        if self.scene_view is not None:
            self.scene_view.clear_selection()

    def clear_view(self):
        """Blank every view for a scene reset."""
        self.clear_selection()
        self.ego.image = np.zeros_like(self.ego.image)
        self.ego.label = "Live view: waiting for fresh observations"
        self.map.image = np.zeros_like(self.map.image)
        self.map.label = "Map: waiting for fresh observations"
        self.topdown.image = np.zeros_like(self.topdown.image)
        self.topdown.label = AWAITING
        self.stage.content = self.explanation.content = ""
        self._progress = Progress()
        self.status.content = "Resetting the scene"
        self.submit_button.disabled = True

    def select(self, decision, frame):
        """Show a decision on its captured frame. The click view stays until the next decision."""
        self.clear_selection()
        self.overlays.select(decision, frame)
        self._detail_skill = decision.skill
        self.map.visible = decision.skill == "navigate"
        self.topdown.visible = decision.skill == "place"
        self._progress = Progress()
        self.stage.content = ""
        self.render(frame)
        if self.topdown.visible:
            self.topdown.image = self.overlays.topdown()
            self.topdown.label = f"Place target: {frame.time:.1f} s"
        selection = self.overlays.selection
        if decision.skill == "pick" and selection is not None:
            self.click.image = selection.overlay(frame, frame.rgb)
            self.click.label = f"Selected segment: {frame.time:.1f} s"
            self.click.visible = True
        if self.scene_view is not None and selection is not None:
            self.scene_view.show_selection(selection.point)

    def skill_event(self, event):
        kind = event["type"]
        if kind == "skill_stage":
            now, sim = time.monotonic(), event.get("simulation_time")
            progress = self._progress
            self._progress = replace(
                progress, skill=event["skill"], stages=(*progress.stages, event["stage"]),
                times=(*progress.times, now),
                sim_start=sim if progress.sim_start is None else progress.sim_start,
                sim_now=progress.sim_now if sim is None else max(progress.sim_now or sim, sim))
            self._refresh_stage(now=now)
        elif kind == "skill_prepared":
            self.prepared(event["plan"])
        elif kind == "arm_return_prepared":
            self.overlays.returning(event)
            if self.overlays.latest is not None:
                self.ego.image = self.overlays.ego_rgb(self.overlays.latest)
            if self.scene_view is not None:
                self.scene_view.clear_arm_path()
        elif kind == "skill_finished":
            progress, now = self._progress, time.monotonic()
            self._progress = replace(progress, result=event["result"], finished_at=now,
                                     sim_now=event.get("simulation_time", progress.sim_now))
            self._refresh_stage(now=now)

    def _refresh_stage(self, *, now=None, simulation_time=None):
        """Redraw the stage panel from one snapshot of the progress."""
        progress = self._progress
        if not progress.stages:
            return
        sim_now = progress.sim_now
        if simulation_time is not None and progress.result is None:
            sim_now = max(sim_now or simulation_time, simulation_time)
        now = time.monotonic() if now is None else now
        if progress.finished_at is not None:
            now = progress.finished_at
        durations = [max(0., end - start) for start, end in pairwise(progress.times)]
        durations.append(max(0., now - progress.times[-1]))
        simulated = (None if progress.sim_start is None or sim_now is None else
                     max(0., sim_now - progress.sim_start))
        skill = progress.result.skill if progress.result is not None else progress.skill
        if progress is not self._progress:  # a reset or new decision replaced it meanwhile
            return
        self.stage.content = stage_progress(skill, progress.stages, progress.result,
                                            durations=durations, simulated=simulated)

    def prepared(self, plan):
        """Draw a prepared plan: route and stance, and the palm's reference path, also in 3D."""
        overlays = self.overlays
        self.render(overlays.prepared(plan))
        if self.topdown.visible and isinstance(plan, place.Plan):
            self.topdown.image = overlays.topdown()
        if self.scene_view is None:
            return
        if isinstance(plan, (navigate.Plan, pick.ApproachPlan)):
            self.scene_view.show_route(overlays.marks["route"], overlays.marks["stance"][2])
        if isinstance(plan, (pick.Plan, place.Plan)) and overlays.palm_path is not None:
            self.scene_view.show_arm_path(overlays.palm_path)

    def update(self, status: str, text: str = ""):
        self.status.content = STATUS.get(status, status)
        if status == "Thinking":
            self.ego.label = "Last camera frame (agent thinking)"
        elif status == "Executing":
            self.ego.label = "Last camera frame"
        elif status == "Ready":
            self.ego.label = "Live camera"
        if status == "Executing":
            self._bubble_message = text
        elif status == "Ready":
            self._bubble_message = ""
        if status == "Ready":
            self.stage.content = ""
            self.clear_selection()
        self.thought.content = thought_bubble(self._bubble_message, thinking=status == "Thinking",
                                              model=self.model_name)
        self.explanation.content = "" if status == "Executing" else note(text, done=status == "model_done")
        self.submit_button.disabled = status in ("Thinking", "Executing", "Reset required")

    def close(self):
        if self._renderer is not None:
            with self._pending:
                self._closing = True
                self._pending.notify()
            self._renderer.join()
        self.server.stop()

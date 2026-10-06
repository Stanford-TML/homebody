"""Scripted physics check: run a sequence of pick-and-place transfers in one simulation.

Each step of a --sequence file (configs/continuous_four.json is the reference) stands
facing an object, picks it, optionally carries it, then places or drops it. No VLM is
involved, and scoring reads only the recorded physics against each step's evaluator_target.
"""
import argparse
import faulthandler
import json
import runpy
import shutil
import sys
import time
import uuid
from collections import Counter
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PICK_NUDGES = (0., .04, -.04)
SETTLE_TICKS, SETTLE_TICK = 25, .1
SUPPORT_STRIDE = 5
PROGRESS_PERIOD = 2.0
PLACE_REFUSALS = {"SURFACE_CHANGED", "COLLISION", "TARGET_LOST"}
TARGET_KEYS = {"bounds", "support_contacts", "interior_halfspaces"}


@dataclass(frozen=True)
class Step:
    """One transfer: navigate to `goal`, pick `label` with `hand`, carry to `carry_goal`, then
    place on a visible support or drop at `release_map_xy_m`.

    Positions are ws_map metres and goals are [x, y, yaw]. `fixture_id` and
    `evaluator_target` are for scoring only.
    """
    fixture_id: str
    label: int | str
    hand: str
    goal: list
    facing: list
    via_goals: list = field(default_factory=list)
    carry_via_goals: list = field(default_factory=list)
    carry_goal: list | None = None
    carry_facing: list | None = None
    destination_label: int | str | None = None
    release_map_xy_m: list | None = None
    release_delta_z_m: float = 0.
    evaluator_target: dict | None = None

    def __post_init__(self):
        drop = self.release_map_xy_m is not None
        checks = {
            "fixture_id": (isinstance(self.fixture_id, str) and self.fixture_id != ""
                           and all(c.isalnum() or c in "_-" for c in self.fixture_id)),
            "label": positive_label(self.label),
            "hand": self.hand in ("auto", "left", "right"),
            "goal": finite(self.goal, 3),
            "facing": finite(self.facing, 2),
            "via_goals": waypoints(self.via_goals),
            "carry_via_goals": waypoints(self.carry_via_goals),
            "carry_goal": self.carry_goal is None or finite(self.carry_goal, 3),
            "carry_facing": self.carry_facing is None or finite(self.carry_facing, 2),
            "destination_label": (self.destination_label is None
                                  or positive_label(self.destination_label)
                                  and self.destination_label != self.label),
            "release_map_xy_m": (not drop or finite(self.release_map_xy_m, 2)
                                 and self.destination_label is not None
                                 and self.evaluator_target is not None),
            "release_delta_z_m": (finite([self.release_delta_z_m], 1)
                                  and abs(self.release_delta_z_m) <= .3
                                  and (drop or self.release_delta_z_m == 0)),
            "evaluator_target": (self.evaluator_target is None
                                 or valid_target(self.evaluator_target)),
        }
        invalid = [name for name, valid in checks.items() if not valid]
        if invalid:
            raise ValueError(f"Step {self.fixture_id!r} has invalid {', '.join(invalid)}")


def finite(value, size):
    """A list of `size` finite numbers, booleans excluded."""
    return (isinstance(value, list) and len(value) == size
            and all(type(v) in (int, float) and abs(v) <= sys.float_info.max for v in value))


def positive_label(value):
    """A scene object's name or its positive segmentation label."""
    return (type(value) is int and value > 0) or (isinstance(value, str) and value != "")


def resolved(step, label_ids):
    """STEP with its object and destination named by their segmentation labels."""
    names = {"label": step.label, "destination_label": step.destination_label}
    unknown = [value for value in names.values() if isinstance(value, str) and value not in label_ids]
    if unknown:
        raise ValueError(f"The scene has no object named {unknown}")
    return replace(step, **{key: label_ids[value] if isinstance(value, str) else value
                            for key, value in names.items()})


def waypoints(value):
    """At most eight [x, y] or [x, y, yaw] poses."""
    return (isinstance(value, list) and len(value) <= 8
            and all(finite(pose, 2) or finite(pose, 3) for pose in value))


def valid_target(value):
    """Ordered finite bounds, named support contacts and optional interior halfspaces."""
    if not (isinstance(value, dict)
            and {"bounds", "support_contacts"} <= value.keys() <= TARGET_KEYS):
        return False
    bounds, supports = value["bounds"], value["support_contacts"]
    planes = value.get("interior_halfspaces")
    return (isinstance(bounds, list) and len(bounds) == 2 and all(finite(row, 3) for row in bounds)
            and all(low < high for low, high in zip(*bounds))
            and isinstance(supports, list) and len(supports) > 0
            and all(isinstance(name, str) and name != "" for name in supports)
            and ("interior_halfspaces" not in value
                 or isinstance(planes, list) and 1 <= len(planes) <= 32
                 and all(finite(plane, 4) and any(plane[:3]) for plane in planes)))


def load_steps(path, self_targets=False):
    """Read and validate a sequence file. Only SELF_TARGETS admits a step without an
    evaluator_target."""
    spec = json.loads(Path(path).read_text())
    if not (isinstance(spec, dict) and spec.keys() == {"schema_version", "steps"}
            and type(spec["schema_version"]) is int and spec["schema_version"] == 1
            and isinstance(spec["steps"], list) and spec["steps"]):
        raise ValueError("A sequence needs schema_version 1 and a nonempty steps list")
    known = {item.name for item in fields(Step)}
    required = {item.name for item in fields(Step)
                if item.default is MISSING and item.default_factory is MISSING}
    steps = []
    for index, raw in enumerate(spec["steps"]):
        if not isinstance(raw, dict) or not required <= raw.keys() <= known:
            raise ValueError(f"Step {index} needs {sorted(required)} and may add "
                             f"{sorted(known - required)}")
        steps.append(Step(**raw))
    if len({step.fixture_id for step in steps}) < len(steps):
        raise ValueError("Steps need distinct fixture_id values")
    unscored = [step.fixture_id for step in steps if step.evaluator_target is None]
    if unscored and not self_targets:
        raise ValueError(f"Steps {unscored} have no evaluator_target; --self-targets would score each "
                         "around the support the skill chooses, which cannot catch a wrong support")
    return steps


def run_sequence(steps, session, cycles):
    """Run every step in one epoch and return False at the first incomplete cycle."""
    port, context = session.port, session.context
    epoch = port.epoch
    for index, step in enumerate(steps):
        context.check(epoch)
        session.begin_cycle(index, step)
        cycle = {"index": index, "fixture_id": step.fixture_id, "start_epoch": port.epoch,
                 "start_time": port.time, "completed": False}
        cycles.append(cycle)
        try:
            cycle["completed"] = run_cycle(step, session)
            context.check(epoch)
        finally:
            cycle.update(end_epoch=port.epoch, end_time=port.time)
            session.record("cycle.json", cycle)
            session.record("final_carried.json", context.carried)
            session.end_cycle(cycle)
        if not cycle["completed"]:
            return False
    return True


def run_cycle(step, session):
    """Navigate, pick, carry and place one step's object; False if the transfer stops early."""
    port, execute = session.port, session.execute
    for index, via in enumerate(step.via_goals):
        if not execute("navigate", navigate_arguments(via), port.snapshot(),
                       evidence_name=f"via_navigate_{index}"):
            return False
    side = pick_object(step, session)
    if side is None:
        return False
    for index, via in enumerate(step.carry_via_goals):
        if not execute("navigate", navigate_arguments(via), port.snapshot(),
                       evidence_name=f"loaded_via_navigate_{index}"):
            return False
    if step.carry_goal is not None and not execute(
            "navigate", navigate_arguments(step.carry_goal, step.carry_facing), port.snapshot(),
            evidence_name="loaded_navigate"):
        return False
    return place_object(step, side, session)


def pick_object(step, session):
    """Stand facing the object and pick it, retrying with nudged stances, and return the picking
    side, or None."""
    port, execute = session.port, session.execute
    for attempt, nudge in enumerate(PICK_NUDGES):
        suffix = f"_retry_{attempt}" if attempt else ""
        frame = port.snapshot()
        away = np.subtract(step.facing, frame.base_pose[:2])
        facing = (np.asarray(step.facing) + nudge * away / np.linalg.norm(away)).tolist()
        if not execute("navigate", navigate_arguments(step.goal, facing, step.hand), frame,
                       evidence_name="navigate" + suffix):
            return None
        frame = port.snapshot()
        click = target_click(frame, step.label)
        if click is None:
            session.record(f"pick_selection{suffix}.json", {"label": step.label, "visible": False,
                                                            "frame": frame.frame_id})
            session.outcome.update(status="target_not_visible", first_failure={
                "evidence_name": "pick" + suffix, "code": "TARGET_NOT_VISIBLE",
                "message": "The object was not visible from the stance; no pick was requested"})
            continue
        u, v = click
        side = step.hand if step.hand != "auto" else ("left" if u < frame.rgb.shape[1] / 2 else "right")
        session.record("pick_selection.json", {
            "label": step.label, "pixel": [u, v], "frame": frame.frame_id,
            "selection": "explicit fixture identity selects current visible pixels only"})
        if execute("pick", {"point_normalized_1000": normalized(frame, u, v), "hand": side}, frame,
                   evidence_name="pick" + suffix):
            return side
        if session.context.carried:
            return None
    return None


def place_object(step, side, session):
    """Place on a visible support or drop at the step's release point, recording target.json
    first and standing still afterwards while the object settles."""
    from homebody.skills.contract import Planning

    port, context, record = session.port, session.context, session.record
    frame = port.snapshot()
    context.check(frame.epoch)
    palm = frame.palm(context.arms[side], side)
    if step.release_map_xy_m is None:
        planning = Planning(context.arms, context.jaw_samples, context.settings, context.carried,
                            context.cancelled)
        _, u, v, label, point = support_click(frame, step.label, palm, side, planning,
                                              session.place_candidates, record,
                                              step.destination_label)
        arguments = {"point_normalized_1000": normalized(frame, u, v), "hand": side}
        record("place_selection.json", {"label": label, "pixel": [u, v],
                                        "visible_depth_point": point, "frame": frame.frame_id})
        target = step.evaluator_target or support_target(point)
    else:
        frame.target_points(step.destination_label)
        arguments = drop_arguments(frame, palm, step.release_map_xy_m, side, step.release_delta_z_m)
        record("place_selection.json", {"label": step.destination_label, "frame": frame.frame_id,
                                        "static_release_map_xy_m": step.release_map_xy_m,
                                        "arguments": arguments, "mode": "fixture_drop"})
        target = step.evaluator_target
    record("target.json", {"schema_version": 1, "frame": "ws_map",
                           "objects": {step.fixture_id: target}})
    if not session.execute("place", arguments, frame):
        return False
    for _ in range(SETTLE_TICKS):
        context.check(frame.epoch)
        port.base_velocity(frame.epoch, 0., 0., 0.)
        context.advance(frame.epoch, SETTLE_TICK)
    return True


def navigate_arguments(goal, facing=None, hand="auto"):
    """Arguments for a snapped navigation to `goal`."""
    arguments = {"goal_xy_m": goal[:2], "snap": True}
    if facing is not None:
        arguments["facing_xy_m"] = facing
    elif len(goal) == 3:
        arguments["yaw"] = goal[2]
    if hand != "auto":
        arguments["hand"] = hand
    return arguments


def normalized(frame, u, v):
    return [min(1000., (u + .25) * 1000 / (frame.rgb.shape[1] - 1)),
            min(1000., (v + .25) * 1000 / (frame.rgb.shape[0] - 1))]


def target_click(frame, label):
    """The visible pixel of `label` nearest its median, or None when too few pixels have depth."""
    v, u = np.nonzero((frame.labels == label) & np.isfinite(frame.depth) & (frame.depth > 0))
    if len(u) < 12:
        return None
    center = np.median(np.column_stack((u, v)), axis=0)
    index = np.argmin(np.linalg.norm(np.column_stack((u, v)) - center, axis=1))
    return int(u[index]), int(v[index])


def support_options(frame, target_label, palm, destination_label=None):
    """Visible upward-facing points of other segments as (distance, u, v, label, point),
    nearest the palm first."""
    options = []
    for v in range(3, frame.depth.shape[0] - 3, SUPPORT_STRIDE):
        for u in range(3, frame.depth.shape[1] - 3, SUPPORT_STRIDE):
            label = int(frame.labels[v, u])
            if label <= 0 or label == target_label or destination_label not in (None, label):
                continue
            if any(frame.labels[y, x] != label for x, y in ((u + 2, v), (u, v + 2))):
                continue
            z = frame.depth[v:v + 3, u:u + 3]
            if not np.isfinite(z).all() or np.any(z <= 0):
                continue
            point = frame.point(u, v)
            normal = np.cross(frame.point(u + 2, v) - point, frame.point(u, v + 2) - point)
            size = np.linalg.norm(normal)
            if size > 1e-8 and abs(normal[2]) / size > .90:
                options.append((float(np.linalg.norm(point[:2] - palm[:2, 3])), u, v, label, point.tolist()))
    return sorted(options)


def support_click(frame, target_label, palm, side, planning, max_candidates, record,
                  destination_label=None):
    """The first support option the place planner accepts. Raises when none does."""
    from homebody.motion.carry import RecoveryPlan
    from homebody.primitives.observations import ObservationError
    from homebody.skills import place
    from homebody.skills.contract import SkillFailure

    options = support_options(frame, target_label, palm, destination_label)
    report = {"frame": frame.frame_id, "sample_stride_pixels": SUPPORT_STRIDE,
              "candidate_limit": max_candidates, "destination_label": destination_label,
              "sampled_candidates": len(options), "attempts": []}
    try:
        for option in options[:max_candidates]:
            distance, u, v, label, point = option
            args = {"point_normalized_1000": normalized(frame, u, v), "hand": side}
            attempt = {"pixel": [u, v], "label": label, "visible_depth_point": point,
                       "distance_from_palm_xy_m": distance, "arguments": args,
                       "status": "preparing"}
            report["attempts"].append(attempt)
            try:
                if planning.cancelled():
                    raise SkillFailure("CANCELLED", "Cancelled support selection")
                plan = place.prepare(args, frame, frame, planning)
                if planning.cancelled():
                    raise SkillFailure("CANCELLED", "Cancelled during support preparation")
            except ObservationError as error:
                attempt.update(status="TARGET_LOST", message=str(error))
                continue
            except SkillFailure as error:
                attempt.update(status=error.code, message=str(error))
                if error.code not in PLACE_REFUSALS:
                    raise
                continue
            if isinstance(plan, RecoveryPlan):
                attempt.update(status="HOLD_UNCERTAIN",
                               message="Carry requires fresh recovery before placement")
                raise SkillFailure("HOLD_UNCERTAIN", attempt["message"])
            attempt.update(status="prepared", prepared_support_point=plan.point.tolist())
            return option
        counts = dict(Counter(row["status"] for row in report["attempts"]))
        raise ValueError(f"No sampled support passed place preparation: sampled={len(options)}, "
                         f"tried={len(report['attempts'])}, refusals={counts}")
    finally:
        record("place_candidates.json", report)


def drop_arguments(frame, palm, map_xy, side, delta_z=0.):
    """Place arguments that drop at map point MAP_XY at the palm's torso height."""
    torso_palm = np.linalg.solve(frame.world_torso, palm[:, 3])
    xy = np.linalg.solve(frame.world_torso[:2, :2], np.asarray(map_xy)
                         - frame.world_torso[:2, 3] - frame.world_torso[:2, 2] * torso_palm[2])
    return {"x_ahead_m": float(xy[0]), "y_left_m": float(xy[1]),
            "delta_z_m": delta_z, "hand": side}


def support_target(point):
    """The --self-targets scoring bounds around the chosen support point."""
    x, y, z = point
    return {"bounds": [[x - .12, y - .12, z - .03], [x + .12, y + .12, z + .35]]}


class ScriptedSession:
    """One port, one skill runner and the evidence for a whole sequence."""

    def __init__(self, backend, recorder, task, outcome, root, settings, args, start):
        from homebody.helpers.kinematics import Arm
        from homebody.session.session import ObservedBackend
        from homebody.skills.contract import Context, Runner

        self.backend, self.recorder, self.task, self.outcome = backend, recorder, task, outcome
        self.visuals, self.place_candidates = not args.no_visuals, args.place_candidates
        self.start = start
        self.cycle, self.step, self.directory = 0, None, task
        self.last_frame = self.last_progress = self.last_hand = -1.
        self.last_grip, self.closure_views = {}, 0
        backend.set_evaluation_sink(self.sample_physics)
        self.recording = settings.recording
        self.port = RecordingPort(ObservedBackend(backend, self.publish, self.recording.publish_period),
                                  recorder, self.view_closure)
        self.arms = {side: Arm(root / "assets/robot/g1_29dof_with_hand.urdf", side)
                     for side in ("left", "right")}
        calibration = json.loads((root / "assets/robot/calibration.json").read_text())
        self.context = Context(self.port, self.port, self.arms, settings,
                               jaw_samples=calibration["jaw_samples"], emit=self.emit,
                               cancelled=lambda: time.monotonic() - start > args.wall_timeout)
        self.runner = Runner(self.context)

    def execute(self, name, arguments, frame, *, evidence_name=None):
        """Run one skill from `frame` and record it; return whether it succeeded."""
        evidence_name = evidence_name or name
        if self.visuals:
            self.recorder.frame(frame)
        self.recorder.event("scripted_request", skill=name, arguments=arguments,
                            frame=frame.frame_id, evidence_name=evidence_name, cycle=self.cycle)
        result = self.runner.run(name, arguments, frame)
        summary = asdict(result)
        self.outcome["steps"].append({**summary, "evidence_name": evidence_name,
                                      "cycle": self.cycle})
        if self.visuals:
            self.recorder.frame(self.port.snapshot())
        self.record(f"{evidence_name}_result.json", summary)
        print(f"RESULT {evidence_name} {json.dumps(summary)}", flush=True)
        if not result.ok:
            self.outcome.update(status="skill_failed", first_failure=summary)
        return result.ok

    def record(self, name, value):
        write_json(self.directory / name, value)

    def begin_cycle(self, index, step):
        self.cycle, self.step = index, step
        self.directory = cycle_directory(self.task, index, step)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.recorder.event("diagnostic_cycle", phase="started", index=index,
                            fixture_id=step.fixture_id, epoch=self.port.epoch, time=self.port.time)
        self.sample_physics()

    def end_cycle(self, cycle):
        self.recorder.event("diagnostic_cycle", phase="finished", **cycle)
        self.record("final_physics.json", self.sample_physics())

    def sample_physics(self, state=None):
        """Record and return a physics row with every hand contact."""
        row = {**(self.backend.evaluation_state() if state is None else state),
               "hand_contacts_detailed": hand_contacts(self.backend)}
        self.recorder.physics(row)
        return row

    def publish(self, frame):
        """Save frames, shoulder and room views at their periods and print progress."""
        r = self.recording
        if self.visuals and frame.time - self.last_hand >= r.shoulder_period:
            side = "right" if self.step is None or self.step.hand == "auto" else self.step.hand
            self.recorder.observer(self.backend.shoulder_frame(side), epoch=frame.epoch,
                                   simulation_time=frame.time, stream="shoulder",
                                   fps=1 / r.shoulder_period)
            self.last_hand = frame.time
        if self.visuals and frame.time - self.last_frame >= r.frame_period:
            self.recorder.frame(frame)
            self.last_frame = frame.time
        if frame.time - self.last_progress >= PROGRESS_PERIOD:
            if self.visuals:
                self.recorder.observer(self.backend.observer_frame(), epoch=frame.epoch,
                                       simulation_time=frame.time, stream="room",
                                       fps=1 / PROGRESS_PERIOD)
            print(f"OBS sim={frame.time:.2f} base={frame.base_pose.tolist()} "
                  f"wall={time.monotonic() - self.start:.1f}s", flush=True)
            self.last_progress = frame.time

    def emit(self, event):
        if event["type"] == "observation":
            frame = event["frame"]
            self.recorder.event("observation", time=frame.time, base=frame.base_pose,
                                frame=frame.frame_id, hands={side: {
                                    "closure": hand.closure, "gap": hand.closure_gap,
                                    "effort": hand.effort_fraction, "joints": hand.joints,
                                    "finger_effort_fraction": hand.finger_effort_fraction,
                                    "finger_closure_gap": hand.finger_closure_gap}
                                    for side, hand in frame.hands.items()})
            return
        self.recorder.event("skill", event=event)
        if event["type"] == "skill_stage" and self.visuals:
            self.recorder.frame(self.port.snapshot())

    def view_closure(self, side, closure):
        """Save the palm pose and visible target at each new nonzero grip (visuals only)."""
        from homebody.primitives.observations import ObservationError

        changed = closure > 0 and self.last_grip.get(side) != closure
        self.last_grip[side] = closure
        if not (self.visuals and changed):
            return
        frame = self.port.snapshot()
        self.recorder.frame(frame)
        palm = frame.palm(self.arms[side], side)
        try:
            cloud, missing = frame.target_points(self.step.label), None
        except ObservationError as error:
            cloud, missing = np.empty((0, 3)), str(error)
        path = self.directory / f"closure_view_{self.closure_views:03d}.npz"
        np.savez_compressed(path, world_palm=palm, target_cloud_world=cloud, rgb=frame.rgb,
                            depth=frame.depth, labels=frame.labels, intrinsics=frame.intrinsics,
                            world_camera=frame.world_camera)
        self.recorder.event("closure_view", time=frame.time, frame=frame.frame_id, side=side,
                            commanded_closure=closure, world_palm=palm,
                            visible_target_points=len(cloud), missing_evidence=missing,
                            path=path.relative_to(self.recorder.path))
        self.closure_views += 1

    def close(self):
        """Stop and close the backend and return the operations that failed."""
        backend, task = self.backend, self.task
        operations = {
            "final_carried": lambda: write_json(task / "final_carried.json", self.context.carried),
            "stop": lambda: backend.stop(backend.epoch),
            "final_state": lambda: write_json(task / "final_physics.json", self.sample_physics()),
            "detach": lambda: backend.set_evaluation_sink(None),
            "close": backend.close}
        errors = []
        for name, operation in operations.items():
            try:
                operation()
            except Exception as error:  # noqa: BLE001
                errors.append(f"{name}: {type(error).__name__}: {error}")
        return errors


class RecordingPort:
    """The observed backend with each base, arm and grip command logged."""

    def __init__(self, observed, recorder, on_grip):
        self.observed, self.recorder, self.on_grip = observed, recorder, on_grip

    def __getattr__(self, name):
        return getattr(self.observed, name)

    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        self.recorder.event("base_command", time=self.time, epoch=epoch,
                            velocity=[forward, lateral, yaw_rate])
        self.observed.base_velocity(epoch, forward, lateral, yaw_rate)

    def arm_target(self, epoch, side, joints):
        self.recorder.event("arm_command", time=self.time, epoch=epoch, side=side, joints=joints)
        self.observed.arm_target(epoch, side, joints)

    def grip(self, epoch, side, closure):
        self.on_grip(side, closure)
        self.recorder.event("grip_command", time=self.time, epoch=epoch, side=side, closure=closure)
        self.observed.grip(epoch, side, closure)


def hand_contacts(backend):
    """Every contact touching a robot hand, with its signed distance and contact force."""
    import mujoco

    model, data, contacts = backend.model, backend.data, []
    for index, contact in enumerate(data.contact):
        geoms = (contact.geom1, contact.geom2)
        bodies = [model.body(model.geom_bodyid[geom]).name for geom in geoms]
        if not any("_hand_" in body or body.endswith("_rubber_hand") for body in bodies):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        contacts.append({"geoms": [model.geom(geom).name for geom in geoms], "bodies": bodies,
                         "position": contact.pos.copy(), "distance": float(contact.dist),
                         "contact_force": force})
    return contacts


def cycle_directory(task, index, step):
    return task / "cycles" / f"{index:03d}-{step.fixture_id}"


def write_json(path, value):
    """The recorder's JSON writer, imported late so setup() can put the snapshot first on sys.path."""
    from homebody.session.recorder import write_json as write

    write(path, value)


def cycle_evidence(rows, fixture_id, lift_required, required_seconds=.3):
    """Post-run pick evidence for one object: verified by REQUIRED_SECONDS of one interval
    lifted LIFT_REQUIRED metres, in hand and unsupported."""
    timeline, lifts, invalid = [], [], set()
    for row, prior in zip(rows, [None, *rows]):
        if prior is not None and (row["time"] <= prior["time"] or row["epoch"] != prior["epoch"]):
            invalid.add("nonmonotonic time or reset")
        if not row["physics_valid"] or not row["robot_upright"] or row["warnings"]:
            invalid.add("invalid physics, fall or warning")
        state = row["objects"][fixture_id]
        lowest = min(point[2] for point in state["corners"])
        rise = lowest - (timeline[0]["lowest_corner"] if timeline else lowest)
        timeline.append({"time": row["time"], "lowest_corner": lowest, "rise": rise,
                         "hand_contacts": state["hand_contacts"],
                         "support_contacts": state["support_contacts"]})
        lifts.append((row["time"], row["epoch"], bool(
            rise >= lift_required and state["hand_contacts"] and not state["support_contacts"])))
    intervals = lifted_intervals(lifts)
    longest = max((end - start for start, end in intervals), default=0.)
    detailed = [row["hand_contacts_detailed"] for row in rows if "hand_contacts_detailed" in row]
    contacts = [contact for sample in detailed for contact in sample]

    def deepest(selected):
        return max((max(0., -contact["distance"]) for contact in selected), default=0.) if detailed else None
    return {"schema_version": 1, "fixture_id": fixture_id,
            "physical_pick_verified": bool(rows and not invalid and longest >= required_seconds - 1e-9),
            "lift_required_metres": lift_required,
            "required_unsupported_hand_contact_seconds": required_seconds,
            "max_qualifying_interval_seconds": longest, "qualifying_intervals": intervals,
            "max_lowest_corner_rise": max((x["rise"] for x in timeline), default=0.),
            "contact_penetration_sample_count": len(detailed),
            "max_hand_contact_penetration": deepest(contacts),
            "max_fixture_hand_contact_penetration": deepest(
                contact for contact in contacts if fixture_id in contact["bodies"]),
            "invalid_reasons": sorted(invalid), "support_height_timeline": timeline}


def lifted_intervals(lifts, gap=.2):
    """[first, last] time of each run of lifted samples within one epoch and GAP seconds."""
    intervals = []
    for (now, epoch, lifted), prior in zip(lifts, [None, *lifts]):
        if lifted and prior is not None and prior[2] and prior[1] == epoch and 0 < now - prior[0] <= gap:
            intervals[-1][1] = now
        elif lifted:
            intervals.append([now, now])
    return intervals


def sequence_evidence(rows, steps, cycles, lift_required):
    """Pick evidence for every step, each scored within its own cycle."""
    reports = []
    for index, step in enumerate(steps):
        cycle = cycles[index] if index < len(cycles) else None
        samples = ([] if cycle is None else [row for row in rows
                   if cycle["start_time"] <= row["time"] <= cycle["end_time"]])
        evidence = cycle_evidence(samples, step.fixture_id, lift_required)
        evidence.update(index=index, completed=bool(cycle and cycle["completed"]))
        reports.append(evidence)
    continuous = bool(cycles and len({c["start_epoch"] for c in cycles}
                                    | {c["end_epoch"] for c in cycles}) == 1)
    return {"schema_version": 1, "cycles": reports, "continuous_epoch": continuous,
            "physical_pick_verified": bool(continuous and len(cycles) == len(steps)
                and all(row["completed"] and row["physical_pick_verified"] for row in reports))}


def score(attempt, task, steps, outcome, lift_required):
    """Score the recorded physics into result.json and return the physics rows."""
    rows = [json.loads(line) for line in (task / "physics.jsonl").read_text().splitlines()]
    evidence = sequence_evidence(rows, steps, outcome["cycles"], lift_required)
    write_json(attempt / "cycle_evidence.json", evidence)
    outcome["independent_physical_pick_verified"] = evidence["physical_pick_verified"]
    targets = {}
    for index, (step, cycle) in enumerate(zip(steps, evidence["cycles"])):
        directory = cycle_directory(task, index, step)
        if (directory / "target.json").exists():
            targets.update(json.loads((directory / "target.json").read_text())["objects"])
        if directory.exists():
            write_json(directory / "cycle_evidence.json", cycle)
    outcome["all_targets_recorded"] = len(targets) == len(steps)
    if targets:
        target = {"schema_version": 1, "frame": "ws_map", "objects": targets}
        write_json(task / "target.json", target)
        scorer = runpy.run_path(str(attempt / "evaluate.py"))["score_attempt"]
        outcome["local_placement_score"] = placement = scorer(rows, target)
        if not outcome["all_targets_recorded"]:
            placement["success"] = False
            placement["invalid_reasons"].append("missing_sequence_targets")
    write_json(attempt / "result.json", outcome)
    return rows


def report(outcome, rows, steps):
    placement = outcome.get("local_placement_score", {})
    print(f"PHYSICS_SCORE picks={outcome.get('independent_physical_pick_verified')} "
          f"placements={placement.get('success')} "
          f"objects={placement.get('objects', {})} "
          f"invalid={placement.get('invalid_reasons', [])}", flush=True)
    if rows:
        for step in steps:
            state = rows[-1]["objects"][step.fixture_id]
            corners = np.asarray(state["corners"])
            print(f"FINAL_OBJECT {step.fixture_id} center={state['position']} "
                  f"bounds={[corners.min(axis=0).tolist(), corners.max(axis=0).tolist()]} "
                  f"support={state['support_contacts']}", flush=True)
    print(f"FINISHED {outcome['status']} wall={outcome['wall_seconds']:.2f}s", flush=True)


def parse_args(argv):
    """Parse the command line and load the sequence and return (args, steps)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True,
                        help="JSON transfer steps, run in order in one simulation without resets")
    parser.add_argument("--output", type=Path,
                        help="Folder for the evidence directory (default ROOT/runs/checks)")
    parser.add_argument("--wall-timeout", type=float, default=600,
                        help="Cancel the run after this many wall-clock seconds")
    parser.add_argument("--no-visuals", action="store_true",
                        help="Skip saved video and observer renders; "
                             "retain sensor observations and physics scoring")
    parser.add_argument("--place-candidates", type=int, default=256,
                        help="Maximum distance-ranked visible supports checked by the "
                             "ordinary place planner")
    parser.add_argument("--width", type=int, help="Camera width (default: the settings' value)")
    parser.add_argument("--height", type=int, help="Camera height (default: the settings' value)")
    parser.add_argument("--settings", type=Path,
                        help="TOML of only the values to change, over configs/simulation.toml and "
                             "the scene's; copied into the evidence")
    parser.add_argument("--source", type=Path,
                        help="homebody package to snapshot and run (default ROOT/src/homebody)")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Release folder holding assets, configs and evaluation")
    parser.add_argument("--scene", type=Path, help="Scene package directory (default: the packaged scene)")
    parser.add_argument("--self-targets", action="store_true",
                        help="Admit steps without an evaluator_target: such a place is scored within "
                             "24 cm of the support the skill itself chose, so a wrong support passes")
    args = parser.parse_args(argv)
    if not (np.isfinite(args.wall_timeout) and args.wall_timeout > 0
            and min(args.place_candidates, args.width or 1, args.height or 1) > 0):
        parser.error("--wall-timeout, --place-candidates, --width and --height must be positive")
    try:
        return args, load_steps(args.sequence, args.self_targets)
    except (OSError, ValueError) as error:
        parser.error(f"{args.sequence}: {error}")


def setup(args):
    """Create the evidence directory with a code snapshot and return (root, attempt, settings, scene)."""
    root = args.root.resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    attempt = (args.output or root / "runs/checks") / f"manipulation_{stamp}_{uuid.uuid4().hex[:8]}"
    attempt.mkdir(parents=True)
    source = args.source.resolve() if args.source else root / "src/homebody"
    override = args.settings.resolve() if args.settings else None
    shutil.copytree(source, attempt / "source/homebody",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(Path(__file__), attempt / "check_manipulation.py")
    shutil.copyfile(root / "evaluation/evaluate.py", attempt / "evaluate.py")
    shutil.copyfile(args.sequence, attempt / "sequence.json")
    shutil.copyfile(root / "configs/simulation.toml", attempt / "simulation.toml")
    if override:
        shutil.copyfile(override, attempt / "settings_override.toml")
    sys.path[:0] = [str(attempt / "source"), str(root / "vendor")]
    from homebody.backends.scene import DEFAULT_SCENE
    from homebody.session.recorder import revision, source_digest
    from homebody.session.settings import Settings

    scene = args.scene.resolve() if args.scene else root / DEFAULT_SCENE
    settings = Settings.load(attempt / "simulation.toml", scene=scene,
                             override=attempt / "settings_override.toml" if override else None)
    settings = replace(settings, camera=replace(settings.camera, width=args.width or settings.camera.width,
                                                height=args.height or settings.camera.height))
    write_json(attempt / "fixture.json", {
        "kind": "scripted perfect-rendered-depth-and-segmentation manipulation diagnostic",
        "source_snapshot_origin": str(source), "revision": revision(root),
        "source_sha256": source_digest(attempt / "source/homebody"),
        "settings_override": str(override) if override else None,
        "vlm_used": False, "sequence": json.loads((attempt / "sequence.json").read_text()),
        "self_targets": args.self_targets,
        "place_candidate_limit": args.place_candidates, "settings": asdict(settings),
        "wall_timeout": args.wall_timeout, "saved_visuals": not args.no_visuals, "scene": str(scene),
        "camera": [settings.camera.width, settings.camera.height]})
    return root, attempt, settings, scene


def run_attempt(args, root, attempt, settings, steps, scene):
    """Run the sequence on one backend, then close it and score whatever was recorded."""
    from homebody.backends.mujoco import MujocoBackend
    from homebody.session.recorder import Recorder

    recorder = Recorder(root, settings, attempt)
    task = recorder.start_task(
        f"Scripted {len(steps)} consecutive transfers in one unchanged simulation epoch", None)
    start = time.monotonic()
    print(f"EVIDENCE {attempt}", flush=True)
    outcome = {"status": "error", "steps": [], "cycles": [], "vlm_used": False}
    session = None
    with (attempt / "stack.txt").open("w") as stack:
        faulthandler.enable(file=stack)
        try:
            backend = MujocoBackend(root, settings, scene=scene)
            print(f"INITIALIZED wall={time.monotonic() - start:.2f}s", flush=True)
            state = backend.evaluation_state()
            absent = sorted({step.fixture_id for step in steps} - state["objects"].keys())
            if absent:
                raise ValueError(f"The scene package moves no object named {absent}")
            steps = [resolved(step, backend.label_ids) for step in steps]
            session = ScriptedSession(backend, recorder, task, outcome, root, settings, args, start)
            if run_sequence(steps, session, outcome["cycles"]):
                outcome["status"] = "skills_completed"
        except BaseException as error:
            outcome.update(status="error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            cleanup_errors = session.close() if session is not None else []
            outcome["wall_seconds"] = time.monotonic() - start
            if cleanup_errors:
                outcome.update(status="error", cleanup_errors=cleanup_errors)
            recorder.end_task(outcome)
            rows = score(attempt, task, steps, outcome, settings.grasp.lift_evidence)
            recorder.close()
            faulthandler.disable()
            report(outcome, rows, steps)
    return outcome


def passed(outcome):
    """Whether every skill completed, every pick was verified and the placement scored."""
    return (outcome["status"] == "skills_completed"
            and outcome.get("independent_physical_pick_verified") is True
            and outcome.get("local_placement_score", {}).get("success") is True)


def main(argv=None):
    args, steps = parse_args(argv)
    root, attempt, settings, scene = setup(args)
    outcome = run_attempt(args, root, attempt, settings, steps, scene)
    return 0 if passed(outcome) else 1


if __name__ == "__main__":
    raise SystemExit(main())

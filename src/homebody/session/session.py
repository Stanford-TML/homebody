"""One physics owner, one decision, one skill result; operator tasks can continue."""
import json
import math
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from homebody.primitives.actions import PhysicsInvalid
from homebody.skills.contract import COMMON_FAILURES, Context, Runner
from homebody.skills.registry import PROMPTS, REGISTRY
from homebody.vlm.observations import annotations, model_observation, rounded
from homebody.vlm.provider import Cancelled, ProviderBusy, ProviderError


def prompted(result: dict) -> dict:
    """A skill result without its traceback detail, as the prompted history keeps it."""
    details = {key: value for key, value in result.get("details", {}).items() if key != "traceback"}
    return {**result, "details": details}


def hand_status(side: str, carry: dict | None) -> str:
    """The prompt's account of one hand, from the carry record its last pick left."""
    if carry is None:
        return f"the {side} hand is free"
    if carry["state"] == "verified":
        return (f"the {side} hand HOLDS the object from its last pick, not placed yet; "
                "only a place by that hand opens it")
    return f"the {side} hand's grasp is uncertain; repeat pick on its target to recheck it"


REAL_TIME_SLACK = 0.5  # s behind the wall clock before the pace restarts


class ObservedBackend:
    """Expose only the declared observation and action methods to skills. With `real_time` the physics is paced to the wall clock."""
    def __init__(self, backend, publish: Callable, publish_period: float, scene_update=None,
                 real_time=False):
        self.__backend = backend
        self._publish = publish
        self._publish_period = publish_period
        self._scene_update = scene_update
        self._last = float("-inf")
        self._real_time, self._anchor = real_time, None

    @property
    def epoch(self):
        return self.__backend.epoch

    @property
    def time(self):
        return self.__backend.time

    def update_scene(self):
        if self._scene_update is not None:
            self._scene_update()

    def snapshot(self):
        frame = self.__backend.snapshot()
        self.update_scene()
        self._publish(frame)
        self._last = frame.time
        return frame

    def measure(self):
        return self.__backend.measure()

    def base_velocity(self, epoch, forward, lateral, yaw_rate):
        self.__backend.base_velocity(epoch, forward, lateral, yaw_rate)

    def arm_target(self, epoch, side, joints):
        self.__backend.arm_target(epoch, side, joints)

    def grip(self, epoch, side, closure):
        self.__backend.grip(epoch, side, closure)

    def stop(self, epoch):
        self.__backend.stop(epoch)

    def advance(self, epoch, seconds):
        try:
            self.__backend.advance(epoch, seconds)
        finally:
            self.update_scene()
        if self._real_time:
            self._pace()
        if self.time - self._last >= self._publish_period or self.time < self._last:
            self.snapshot()


    def _pace(self):
        wall, sim = time.monotonic(), self.__backend.time
        if self._anchor is None or wall - self._anchor[0] > sim - self._anchor[1] + REAL_TIME_SLACK:
            self._anchor = (wall, sim)
            return
        ahead = (sim - self._anchor[1]) - (wall - self._anchor[0])
        if ahead > 0:
            time.sleep(ahead)


class Session:
    def __init__(self, backend, provider, settings, recorder, controls, *, arms, jaw_samples,
                 ui=None, semantic_context=None, observer_frame=None, shoulder_frame=None,
                 scene_update=None, real_time=False):
        self.backend, self.provider = backend, provider
        self.settings, self.recorder = settings, recorder
        self.controls, self.ui = controls, ui
        self.semantic_context = semantic_context or {}
        self.observer_frame, self.shoulder_frame, self.view_side = observer_frame, shoulder_frame, "right"
        self._observer_epoch = self._frame_epoch = None
        self._observer_time = self._frame_time = self._view_time = float("-inf")
        self.observed = ObservedBackend(backend, self.publish, settings.recording.publish_period,
                                        scene_update, real_time)
        self.context = Context(observations=self.observed, actions=self.observed, arms=arms,
                               settings=settings, jaw_samples=jaw_samples,
                               cancelled=controls.cancel.is_set, emit=self.skill_event)
        self.runner = Runner(self.context)
        self.history = []
        self.status = "Ready"
        self._physics_fault = None

    @property
    def requires_reset(self):
        return self._physics_fault is not None

    def physics_failed(self, error):
        fault = self.backend.epoch, str(error)
        if fault != self._physics_fault:
            self._physics_fault = fault
            self.recorder.event("physics_invalid", epoch=fault[0], reason=fault[1],
                                simulation_seconds=self.backend.time)
            self.backend.stop(self.backend.epoch)
        self.update("Reset required", f"{fault[1]} Restart the console to continue.")

    def update(self, status, text=""):
        self.status = status
        if self.ui is not None:
            self.ui.update(status, text)

    def publish(self, frame):
        """Show every frame; record it and the operator views only while a task records."""
        if self.ui is not None:
            self.ui.show(frame)
        if not self.recorder.recording:
            return
        r = self.settings.recording
        if frame.epoch != self._frame_epoch or frame.time - self._frame_time >= r.frame_period:
            self.recorder.frame(frame)
            self._frame_epoch, self._frame_time = frame.epoch, frame.time
        if self.observer_frame is not None and (frame.epoch != self._observer_epoch or
                                               frame.time - self._observer_time >= r.room_period):
            self.recorder.observer(self.observer_frame(), epoch=frame.epoch,
                                   simulation_time=frame.time, fps=1 / r.room_period)
            self._observer_epoch, self._observer_time = frame.epoch, frame.time
        if self.shoulder_frame is not None and frame.time - self._view_time >= r.shoulder_period:
            self.recorder.observer(self.shoulder_frame(self.view_side), epoch=frame.epoch,
                                   simulation_time=frame.time, stream="shoulder",
                                   fps=1 / r.shoulder_period)
            self._view_time = frame.time

    def skill_event(self, event):
        if event["type"] == "observation":
            return
        self.recorder.event("skill", simulation_time=self.backend.time, event=event)
        if self.ui is not None:
            self.ui.skill_event({**event, "simulation_time": self.backend.time})

    def heartbeat(self):
        """Step valid physics while idle/thinking; invalid generations wait for reset."""
        if self.requires_reset:
            time.sleep(.02)
            return
        start = time.monotonic()
        try:
            self.backend.base_velocity(self.backend.epoch, 0.0, 0.0, 0.0)
            if self.status == "Thinking":  # step without publishing frames
                self.backend.advance(self.backend.epoch, 0.02)
                self.observed.update_scene()
            else:
                self.observed.advance(self.backend.epoch, 0.02)
        except PhysicsInvalid as error:
            self.physics_failed(error)
            raise
        time.sleep(max(0, 0.02 - (time.monotonic() - start)))

    def prompt(self, task, frame):
        """The turn's prompt: instructions, skills, scene annotations, hands, task, observation and recent steps."""
        settings = vars(self.settings)
        instructions = (Path(__file__).parents[1] / "vlm" / "agent_prompt.md").read_text()
        skills = "\n".join(f"{skill.PROMPT.format_map(settings)} Refusals: {', '.join(dict.fromkeys(skill.FAILURES))}."
                           for skill in REGISTRY.values())
        hands = "; ".join(hand_status(side, self.context.carried.get(side)) for side in ("left", "right"))
        steps = "\n".join(json.dumps(rounded({key: entry[key] for key in ("operator_task", "base_pose", "decision", "result")
                                              if key in entry})) for entry in self.history[-24:])
        return "\n\n".join((
            instructions.format_map(settings).strip(),
            f"Skills:\n{skills}\nAny skill may also refuse with: {', '.join(COMMON_FAILURES)}.",
            "Initial-scene memory:\n" + annotations(self.semantic_context),
            f"Hands now: {hands}.",
            "Current task:\n" + task,
            "Current observation:\n" + json.dumps(model_observation(frame)),
            "Recent steps, oldest first:\n" + (steps or "none")))

    def decide(self, prompt, images, names, attempt):
        """One decision. A busy service is asked again under attempt-retryN."""
        agent = self.settings.agent
        for retry in range(agent.busy_retries + 1):
            try:
                return self.provider.generate(prompt, images=images, names=names,
                                              attempt=attempt if not retry else attempt.with_name(f"{attempt.name}-retry{retry}"),
                                              tick=self.heartbeat, cancelled=self.controls.cancel.is_set)
            except ProviderBusy as busy:
                if retry == agent.busy_retries:
                    raise
                self.update("Thinking", f"{busy} Asking again in {agent.busy_wait * (retry + 1):g} s.")
                until = time.monotonic() + agent.busy_wait * (retry + 1)
                while time.monotonic() < until and not self.controls.cancel.is_set():
                    self.heartbeat()
                if self.controls.cancel.is_set():
                    raise Cancelled("Operator cancelled while waiting to ask again")
        raise AssertionError("unreachable")

    def run_task(self, task: str) -> dict:
        task_dir = self.recorder.start_task(task, self.backend.epoch)
        started_wall, started_sim = time.monotonic(), self.backend.time
        outcome = {"status": "error", "steps": 0, "model_claim": None}
        self.history.append({"operator_task": task})
        try:
            if self.requires_reset:
                raise PhysicsInvalid(self._physics_fault[1])
            self.backend.set_evaluation_sink(self.recorder.physics)
            names = tuple(PROMPTS) + ("done",)
            for step in range(self.settings.agent.max_steps):
                if self.controls.cancel.is_set():
                    raise Cancelled("Task stopped by operator")
                frame = self.observed.snapshot()
                images = self.recorder.agent_images(frame)
                prompt = self.prompt(task, frame)
                self.update("Thinking")
                decision = self.decide(prompt, images, names, task_dir / "decisions" / f"step-{step:03d}")
                if self.controls.cancel.is_set() or frame.epoch != self.backend.epoch:
                    raise Cancelled("Decision invalidated before execution")
                if decision.skill not in names:
                    raise ProviderError(f"Unsupported skill: {decision.skill}")
                self.recorder.event("decision", step=step, epoch=frame.epoch, sequence=frame.sequence,
                                    simulation_time=self.backend.time, decision=decision)
                outcome["steps"] = step + 1
                self.update("Executing", decision.text)
                if self.ui is not None:
                    self.ui.select(decision, frame)
                if decision.skill == "done":
                    summary = decision.arguments.get("summary", decision.text)
                    if not isinstance(summary, str):
                        raise ProviderError("done.summary must be text")
                    outcome.update(status="model_done", model_claim=summary)
                    break
                if decision.arguments.get("hand") in ("left", "right"):
                    self.view_side = decision.arguments["hand"]
                result = asdict(self.runner.run(decision.skill, decision.arguments, frame))
                self.history.append({"base_pose": frame.base_pose.tolist(), "decision": asdict(decision),
                                     "epoch": frame.epoch,
                                     "frame_sequence": frame.sequence, "result": prompted(result)})
                self.recorder.event("result", step=step, simulation_time=self.backend.time, result=result)
                if result["code"] == "PHYSICS_INVALID":
                    raise PhysicsInvalid(result["message"])
            else:
                outcome["status"] = "step_limit"
        except PhysicsInvalid as exc:
            self.physics_failed(exc)
            outcome.update(status="physics_invalid", reason=str(exc), requires_reset=True)
        except Cancelled as exc:
            outcome.update(status="cancelled", reason=str(exc))
        except ProviderError as exc:
            outcome.update(status="provider_error", reason=str(exc))
        except KeyboardInterrupt:
            outcome.update(status="interrupted", reason="KeyboardInterrupt")
            raise
        except Exception as exc:
            outcome.update(status="error", reason=f"{type(exc).__name__}: {exc}",
                           traceback=traceback.format_exc())
            raise
        finally:
            cleanup_errors = []
            try:
                self._halt(outcome["status"])
            except Exception as exc:  # noqa: BLE001 -- Cleanup must retain terminal evidence.
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
            try:
                self.recorder.physics(self.backend.evaluation_state())
            except Exception as exc:  # noqa: BLE001 -- Preserve failure if final sensors are lost.
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
            self.backend.set_evaluation_sink(None)
            if cleanup_errors:
                outcome.update(status="error", status_before_cleanup=outcome["status"],
                               cleanup_error="; ".join(cleanup_errors))
            outcome.update(wall_seconds=time.monotonic() - started_wall,
                           simulation_seconds=self.backend.time - started_sim)
            self.recorder.end_task(outcome)
            if not self.requires_reset:
                self.update(outcome["status"], outcome.get("reason") or outcome.get("model_claim") or "")
        return outcome

    def _halt(self, status):
        """Stop the robot after a task; a finished task keeps its arm targets."""
        if self.requires_reset:
            return
        if status == "model_done":
            epoch, tick = self.backend.epoch, self.settings.motion.tick
            self.backend.base_velocity(epoch, 0.0, 0.0, 0.0)
            for _ in range(math.ceil(self.settings.agent.done_settle_seconds / tick)):
                self.observed.advance(epoch, tick)
        else:
            self.backend.stop(self.backend.epoch)

    def run(self):
        """The idle/task loop. UI threads never step or reset the backend."""
        self.observed.snapshot()
        while not self.controls.closing.is_set():
            command = self.controls.next()
            if command is None:
                try:
                    self.heartbeat()
                except PhysicsInvalid:
                    pass
                continue
            if command.name == "task":
                self.run_task(command.text)
            elif command.name == "stop":
                if not self.requires_reset:
                    self.backend.stop(self.backend.epoch)
                    self.update("Stopped", "Enter another task when ready.")
                self.controls.cancel.clear()
            elif command.name == "reset":
                self.backend.set_evaluation_sink(None)
                if self.ui is not None:
                    self.ui.clear_view()
                try:
                    self.backend.reset()
                    self.observed.snapshot()
                except PhysicsInvalid as error:
                    self.observed.update_scene()
                    self.physics_failed(error)
                    continue
                self._physics_fault = None
                self.context.carried.clear()
                self.history.clear()
                self.controls.cancel.clear()
                self.recorder.event("scene_reset", epoch=self.backend.epoch)
                self.update("Ready", "Scene reset. Previous plans were invalidated.")

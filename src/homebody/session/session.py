"""One physics owner, one decision, one skill result; operator tasks can continue."""
import json
import math
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from homebody.primitives.actions import PhysicsInvalid
from homebody.primitives.observations import ObservationError
from homebody.skills.contract import COMMON_FAILURES, Context, Runner
from homebody.skills.manipulation_policy import ManipulationPolicy
from homebody.skills.registry import REGISTRY
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


def measured_hand_status(side, carry, frame, settings):
    """Describe the frozen hand evidence without treating an absent ledger as empty."""
    if carry is not None:
        return hand_status(side, carry)
    hand = frame.hands.get(side)
    owner_fault = ((frame.sensor_metadata or {}).get("command_owner") or {}).get("fault")
    if (hand is None or owner_fault or not math.isfinite(hand.closure)
            or not 0 <= hand.closure <= 1):
        return f"the {side} hand's occupancy is unknown; current hand evidence is unavailable or invalid"
    if hand.closure > settings.placement.open_closure_limit or hand.retained(settings.grasp):
        return (f"the {side} hand may hold an untracked object; it is closed or has retention evidence. "
                "Do not assume it is empty; verify it from current observations or end the task with done and say why")
    return f"the {side} hand is measured open, with no tracked object or retention evidence"


REAL_TIME_SLACK = 0.5  # s behind the wall clock before the pace restarts
_DEFAULT_EVALUATOR = object()


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
                 scene_update=None, real_time=False, registry=None, memory=None,
                 navigation=None, manipulation=None, target_selector=None,
                 navigation_evidence=None, evaluator=_DEFAULT_EVALUATOR,
                 persistent=False, readiness=None, target_retention=None):
        self.backend, self.provider = backend, provider
        self.settings, self.recorder = settings, recorder
        self.controls, self.ui = controls, ui
        self.registry = REGISTRY if registry is None else registry
        self.evaluator = backend if evaluator is _DEFAULT_EVALUATOR else evaluator
        self.memory, self.persistent, self.readiness = memory, persistent, readiness
        self.target_retention = target_retention
        self.recalled, self._memory_context = [], []
        self.semantic_context = semantic_context or {}
        self.observer_frame, self.shoulder_frame, self.view_side = observer_frame, shoulder_frame, "right"
        self._observer_epoch = self._frame_epoch = None
        self._observer_time = self._frame_time = self._view_time = float("-inf")
        self.observed = ObservedBackend(backend, self.publish, settings.recording.publish_period,
                                        scene_update, real_time)
        self.context = Context(observations=self.observed, actions=self.observed, arms=arms,
                               settings=settings, jaw_samples=jaw_samples,
                               cancelled=controls.cancel.is_set, emit=self.skill_event,
                               memory=memory, navigation=navigation,
                               manipulation=manipulation or ManipulationPolicy(),
                               target_selector=target_selector, navigation_evidence=navigation_evidence,
                               paused=controls.pause_requested.is_set if persistent else None)
        self.runner = Runner(self.context, self.registry)
        self.history = []
        self.status = "Ready"
        self._physics_fault = None
        self._task_active = False
        self.retention = None
        if persistent:
            from .retention import RetentionGuard
            self.retention = RetentionGuard(
                backend, recorder, controls, carried=lambda: self.context.carried,
                update=self.update, task_active=lambda: self._task_active,
                target_retention=target_retention)

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
            self._workflow_state()

    def _workflow_state(self):
        if self.persistent and self.ui is not None:
            self.ui.workflow_state({"carried": self.context.carried.copy(),
                                    "memory": self.memory.status() if self.memory else None,
                                    "task_active": self._task_active,
                                    "paused": self.controls.paused.is_set(),
                                    "recording": str(self.recorder.path),
                                    "retention": self.retention.status() if self.retention else None,
                                    "readiness": self.readiness() if self.readiness else None})

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
            if self.persistent and not self._task_active and self.status == "Waiting for sensors":
                self.update("Ready", "Fresh sensor evidence restored. Ready for your task.")
        except ObservationError as error:
            if not self.persistent or self._task_active:
                raise
            self.backend.stop(self.backend.epoch)
            if self.status != "Waiting for sensors":
                self.recorder.event("idle_observation_unavailable", error=str(error))
            self.update("Waiting for sensors", str(error))
        except PhysicsInvalid as error:
            self.physics_failed(error)
            raise
        time.sleep(max(0, 0.02 - (time.monotonic() - start)))

    def prompt(self, task, frame):
        """The turn's prompt: instructions, skills, scene annotations, hands, task, observation and recent steps."""
        settings = vars(self.settings)
        instructions = (Path(__file__).parents[1] / "vlm" / "agent_prompt.md").read_text()
        def description(skill):
            prompt = (getattr(skill, "REAL_PROMPT", skill.PROMPT)
                      if self.context.manipulation.target_or_refuse else skill.PROMPT)
            failures = (skill.FAILURES if self.context.manipulation.target_or_refuse
                        else getattr(skill, "PROMPT_FAILURES", skill.FAILURES))
            if self.context.navigation is not None and hasattr(skill, "GRASP_FAILURES"):
                failures = skill.GRASP_FAILURES + self.context.navigation.FAILURES
            return f"{prompt.format_map(settings)} Refusals: {', '.join(dict.fromkeys(failures))}."
        skills = "\n".join(description(skill) for skill in self.registry.values())
        hands = "; ".join(measured_hand_status(side, self.context.carried.get(side), frame, self.settings)
                          if self.context.manipulation.target_or_refuse or self.persistent
                          else hand_status(side, self.context.carried.get(side))
                          for side in ("left", "right"))
        steps = "\n".join(json.dumps(rounded({key: entry[key] for key in ("operator_task", "base_pose", "decision", "result")
                                              if key in entry})) for entry in self.history[-24:])
        parts = [
            instructions.format_map(settings).strip(),
            f"Skills:\n{skills}\nAny skill may also refuse with: {', '.join(COMMON_FAILURES)}.",
            "Initial-scene memory:\n" + annotations(self.semantic_context),
            f"Hands now: {hands}.",
            "Current task:\n" + task,
            "Current observation:\n" + json.dumps(model_observation(frame)),
            "Recent steps, oldest first:\n" + (steps or "none")]
        if self.memory is not None:
            parts.append("Persistent photo memory: remember stores this current view without motion. "
                         "Recall retrieves historical views near a requested map pose, not live evidence. "
                         "Use current calibrated observations to verify an object before walking or picking. "
                         "An empty recall result is valid; change the query or continue from current observations.\n"
                         "Historical images appended after the four current images:\n" +
                         json.dumps(self._memory_context))
        if self.context.manipulation.target_or_refuse:
            parts.append("Task contract: carry one object at a time and place it at the requested target; "
                         "if that is not possible, say why instead of placing it elsewhere.")
        if self.persistent:
            parts.append("STOP holds the robot and preserves grip and memory. End session requires an "
                         "empty carry state after verified placement or operator-confirmed handover.")
        return "\n\n".join(parts)

    def images(self, frame):
        images = tuple(self.recorder.agent_images(frame))
        self._memory_context = []
        if self.memory is None:
            return images
        from homebody.memory import MemoryError
        attachments, gallery = [], []
        for hit in self.recalled:
            try:
                rgb = self.memory.load_image(hit)
                path = self.recorder.historical_image(hit, rgb)
            except (MemoryError, OSError) as error:
                self.recorder.event("memory_attachment_refused", reason=str(error), record_id=hit.get("record_id"))
                self._memory_context.append({"unavailable": str(error), "historical": True})
                continue
            metadata = {key: hit[key] for key in ("record_id", "pose", "time", "frame_id", "content") if key in hit}
            metadata.update(historical=True, image_number=len(images) + len(attachments) + 1,
                            sha256=hit["image"]["sha256"])
            self._memory_context.append(metadata)
            attachments.append(path)
            gallery.append({**metadata, "rgb": rgb})
        if self.persistent and self.ui is not None:
            self.ui.recalled(gallery)
        return images + tuple(attachments)

    def _wait_paused(self):
        if not self.persistent or not self.controls.pause_requested.is_set():
            return
        self.backend.stop(self.backend.epoch)
        self.recorder.event("paused", epoch=self.backend.epoch, time=self.backend.time)
        self.update("Paused", "Position and grip held. Resume uses fresh observations and a new decision.")
        while self.controls.paused.is_set() and not self.controls.cancel.is_set():
            self.heartbeat()
        self.controls.pause_requested.clear()
        if self.controls.cancel.is_set():
            raise Cancelled("Task stopped while paused")
        self.recorder.event("resumed", epoch=self.backend.epoch, time=self.backend.time)

    def _decision_context(self, task, names, attempt):
        interruption = 0
        while True:
            self._wait_paused()
            frame = self.observed.snapshot()
            images = self.images(frame)
            prompt = self.prompt(task, frame)
            self.update("Thinking")
            try:
                request = attempt if not interruption else attempt.with_name(f"{attempt.name}-resume{interruption}")
                decision = self.decide(prompt, images, names, request)
            except Cancelled:
                if self.persistent and self.controls.pause_requested.is_set() and not self.controls.cancel.is_set():
                    interruption += 1
                    continue
                raise
            if self.persistent and self.controls.pause_requested.is_set():
                self.recorder.event("decision_discarded", reason="pause", decision=decision)
                interruption += 1
                continue
            return frame, decision

    def decide(self, prompt, images, names, attempt):
        """One decision. A busy service is asked again under attempt-retryN."""
        agent = self.settings.agent
        for retry in range(agent.busy_retries + 1):
            try:
                return self.provider.generate(prompt, images=images, names=names,
                                              attempt=attempt if not retry else attempt.with_name(f"{attempt.name}-retry{retry}"),
                                              tick=self.heartbeat, cancelled=lambda: self.controls.cancel.is_set() or
                                              (self.persistent and self.controls.pause_requested.is_set()))
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
        if self.retention is not None:
            self.retention.begin_task()
        task_dir = self.recorder.start_task(task, self.backend.epoch)
        self._task_active = True
        started_wall, started_sim = time.monotonic(), self.backend.time
        outcome = {"status": "error", "steps": 0, "model_claim": None}
        self.history.append({"operator_task": task})
        try:
            if self.requires_reset:
                raise PhysicsInvalid(self._physics_fault[1])
            if self.evaluator is not None:
                self.evaluator.set_evaluation_sink(self.recorder.physics)
            names = tuple(self.registry) + ("done",)
            for step in range(self.settings.agent.max_steps):
                if self.controls.cancel.is_set():
                    raise Cancelled("Task stopped by operator")
                frame, decision = self._decision_context(task, names, task_dir / "decisions" / f"step-{step:03d}")
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
                if self.target_retention is not None:
                    self.target_retention(self.context.carried)
                if decision.skill == "recall" and result["code"] == "OK":
                    self.recalled = result["details"].get("hits", []) if decision.arguments.get("op", "retrieve") == "retrieve" else []
                self._workflow_state()
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
                if self.evaluator is not None:
                    self.recorder.physics(self.evaluator.evaluation_state())
            except Exception as exc:  # noqa: BLE001 -- Preserve failure if final sensors are lost.
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
            if self.evaluator is not None:
                self.evaluator.set_evaluation_sink(None)
            if cleanup_errors:
                outcome.update(status="error", status_before_cleanup=outcome["status"],
                               cleanup_error="; ".join(cleanup_errors))
            outcome.update(wall_seconds=time.monotonic() - started_wall,
                           simulation_seconds=self.backend.time - started_sim)
            if self.persistent:
                outcome.update(carried=self.context.carried.copy(), independently_verified=False,
                               session_retained=True)
            try:
                self.recorder.end_task(outcome)
            finally:
                self._task_active = False
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
        if self.persistent:
            self.heartbeat()
        else:
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
                with self.controls.task_lock:
                    if not self.controls.end_requested.is_set():
                        self.controls.cancel.clear()
                self.controls.paused.clear()
                self.controls.pause_requested.clear()
            elif command.name == "end" and self.persistent:
                self.end_session()
            elif command.name == "handover" and self.persistent:
                self.confirm_handover(command.confirmation)
            elif command.name == "reset":
                if self.persistent:
                    self.update("Stopped", "REAL sessions cannot reset the robot or map through the console.")
                    continue
                self.evaluator.set_evaluation_sink(None)
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

    def end_session(self):
        """A voluntary close preserves a carried object until placement or explicit handover."""
        return self.retention.end()

    def confirm_handover(self, confirmation):
        return self.retention.handover(confirmation)

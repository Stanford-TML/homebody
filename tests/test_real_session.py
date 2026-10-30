"""REAL session contracts through the shared provider/Runner loop; no hardware effects."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_session import Backend, ImageOnlyRecorder, Provider

from homebody.memory import MapBinding, NamedMemory
from homebody.session.config import Agent
from homebody.session.session import Session
from homebody.session.settings import Settings
from homebody.skills import recall, remember
from homebody.skills.manipulation_policy import REAL_MANIPULATION
from homebody.ui.console import Controls
from homebody.vlm.provider import Decision


class LiveContract(Backend):
    """An explicit real-shaped fake has no scorer, reset or gripper actuation methods."""
    color = 67

    def snapshot(self):
        frame = super().snapshot()
        return replace(frame, rgb=np.full_like(frame.rgb, self.color))

    def set_evaluation_sink(self, sink):
        raise AssertionError("REAL has no simulator evaluator")

    def evaluation_state(self):
        raise AssertionError("REAL has no privileged physics state")

    def reset(self):
        raise AssertionError("REAL task controls cannot reset the body or map")


@pytest.fixture
def build(tmp_path):
    recorders = []

    def make(provider, *, memory=None, registry=None, controls=None):
        backend = LiveContract()
        settings = Settings(agent=Agent(done_settle_seconds=0., max_steps=6))
        recorder = ImageOnlyRecorder(tmp_path, settings, tmp_path / "runs", backend="real")
        recorders.append(recorder)
        controls = controls or Controls()
        return Session(backend, provider, settings, recorder, controls, arms={}, jaw_samples={},
                       registry=registry or {"remember": remember, "recall": recall}, memory=memory,
                       manipulation=REAL_MANIPULATION, evaluator=None, persistent=True)

    yield make
    for recorder in recorders:
        recorder.close()


def store(tmp_path):
    frame = LiveContract().snapshot()
    binding = MapBinding.from_navigation("room", "a" * 64, "map", frame.navigation)
    return NamedMemory.open(tmp_path / "memory", "lab", binding)


def events(session):
    return [json.loads(line) for line in (session.recorder.path / "events.jsonl").read_text().splitlines()]


def test_stop_retains_carry_memory_and_allows_later_task(build, tmp_path):
    controls = Controls()
    provider = Provider([Decision("Late", "done", {})], controls=controls, ticks=0)
    session = build(provider, memory=store(tmp_path), controls=controls)
    carried = {"state": "verified", "label": 1}
    session.context.carried["right"] = carried
    result = session.run_task("Carry the cup")
    assert result["status"] == "cancelled" and result["session_retained"]
    assert session.context.carried["right"] == carried and not controls.closing.is_set()
    assert not session.end_session()
    assert session.context.carried and not controls.closing.is_set()
    controls.cancel.clear()
    session.provider = Provider([Decision("Retain hold", "done", {})], ticks=0)
    assert session.run_task("Keep holding")["status"] == "model_done"
    assert session.context.carried
    assert session.confirm_handover(session.retention.status()["handover_token"])
    assert not session.context.carried and session.end_session()
    assert controls.closing.is_set()
    assert any(row["kind"] == "operator_handover" for row in events(session))


@pytest.mark.parametrize("sensor_error", [False, True])
def test_idle_end_requires_open_hand_evidence_or_explicit_handover(build, sensor_error):
    session = build(Provider([], ticks=0))
    def probe():
        if sensor_error:
            raise RuntimeError("Hand sensors unavailable")
        return {"retained_or_unknown": True, "reason": "untracked_closed_hand",
                "hands": {"right": {"closure": .6}}}
    session.backend.retention_status = probe
    assert not session.context.carried
    assert not session.end_session()
    assert not session.controls.closing.is_set()
    assert session.confirm_handover(session.retention.status()["handover_token"])
    assert not session.context.carried  # operator acknowledgment is not a fabricated object record
    assert session.end_session()


def test_pause_during_skill_holds_then_redecides_fresh(build):
    controls = Controls()

    def execute(plan, ctx):
        controls.pause()
        controls.resume()
        ctx.check(plan)
        raise AssertionError("A paused action must not continue")

    skill = SimpleNamespace(NEEDS=("base_velocity",), FAILURES=(), PROMPT="test: act",
                            prepare=lambda args, selected, current, planning: current.epoch,
                            execute=execute)
    provider = Provider([Decision("Act", "test", {}), Decision("Fresh", "done", {})], ticks=0)
    session = build(provider, controls=controls, registry={"test": skill})
    assert session.run_task("Act")["status"] == "model_done"
    result = next(row["result"] for row in events(session) if row["kind"] == "result")
    assert result["code"] == "PAUSED"
    assert session.backend.stops >= 1


def test_idle_real_session_holds_on_sensor_gap_and_recovers_without_model_calls(build):
    from homebody.primitives.observations import ObservationError
    provider = Provider([])
    session = build(provider)
    original = session.observed.advance
    def gap(*args):
        raise ObservationError('RGB-D expired')
    session.observed.advance = gap
    session.heartbeat()
    assert session.status == 'Waiting for sensors'
    assert len([r for r in events(session) if r['kind'] == 'idle_observation_unavailable']) == 1
    session.heartbeat()
    assert len([r for r in events(session) if r['kind'] == 'idle_observation_unavailable']) == 1
    session.observed.advance = original
    session.heartbeat()
    assert session.status == 'Ready' and not provider.prompts


NAVIGATE_REFUSALS = "NO_ROUTE, STALL, TIMEOUT, ARRIVAL_NOT_VERIFIED, CARRY_CLEARED, HOLD_UNCERTAIN"
PICK_REFUSALS = ("HAND_OCCUPIED, CALIBRATION_MISSING, NO_GRASP, UNREACHABLE, TARGET_MOVED, ARM_NOT_REACHED, "
                 "LIFT_NOT_VERIFIED, CARRY_CLEARED, HOLD_UNCERTAIN, RETRY_REQUIRED, NO_ROUTE, STALL, TIMEOUT, "
                 "ARRIVAL_NOT_VERIFIED")
PLACE_REFUSALS = ("NOT_HOLDING, SURFACE_CHANGED, ARM_NOT_REACHED, RELEASE_NOT_VERIFIED, OUTCOME_UNKNOWN, "
                  "PLACEMENT_NOT_VERIFIED, CARRY_CLEARED, HOLD_UNCERTAIN, RETRY_REQUIRED")


def refusals(session):
    prompt = session.prompt("Inspect", session.backend.snapshot())
    return [line.split(" Refusals: ", 1)[1] for line in prompt.splitlines() if " Refusals: " in line]


def test_prompt_refusals_follow_the_manipulation_policy(build):
    """Each policy's prompt lists exactly the codes its skills can return, in a fixed order."""
    from homebody.skills.registry import task_registry
    sim = Session(Backend(), Provider([]), Settings(), SimpleNamespace(), Controls(),
                  arms={}, jaw_samples={})
    assert refusals(sim) == [NAVIGATE_REFUSALS + ".", PICK_REFUSALS + ".", PLACE_REFUSALS + "."]
    real = build(Provider([]), registry=task_registry(memory=True))
    memory = "MEMORY_UNAVAILABLE, MEMORY_INVALID, MEMORY_IO."
    from homebody.skills import pick
    from homebody.skills.navigate_real import task as navigate_real
    assert set(navigate_real.FAILURES) <= set(pick.FAILURES)  # pick's walk-in can return them
    assert refusals(real) == [NAVIGATE_REFUSALS + ", FOOTPRINT_UNAVAILABLE.",
                              PICK_REFUSALS + ", FOOTPRINT_UNAVAILABLE.",
                              PLACE_REFUSALS + ", UNREACHABLE.", memory, memory]

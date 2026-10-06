"""Typed failures become declared codes; anything else is an error that propagates after
exactly one terminal event; the robot is parked and stopped the same way either way."""
from types import SimpleNamespace

import pytest
from test_skills import FakeBackend

from homebody.primitives.actions import ActionRejected, PhysicsInvalid
from homebody.primitives.observations import ObservationError
from homebody.skills.contract import Context, Result, Runner, SkillFailure
from homebody.skills.registry import REGISTRY


def finished(events):
    return [event["result"] for event in events if event["type"] == "skill_finished"]


def fake_skill(monkeypatch, execute, abort=None):
    skill = SimpleNamespace(NEEDS=(), FAILURES=("BROKEN",),
                            prepare=lambda arguments, selected, current, planning: "plan",
                            execute=execute)
    if abort is not None:
        skill.abort = abort
    monkeypatch.setitem(REGISTRY, "fake", skill)
    return skill


def run(backend, events, name="fake"):
    return Runner(Context(backend, backend, {}, emit=events.append)).run(name, {}, backend.frame)


def raising(error):
    def execute(plan, ctx):
        raise error
    return execute


def test_an_error_while_preparing_reports_once_and_sends_nothing(monkeypatch):
    skill = fake_skill(monkeypatch, raising(SkillFailure("BROKEN", "unused")))
    skill.prepare = lambda *args: 1 / 0
    backend, events = FakeBackend(), []
    with pytest.raises(ZeroDivisionError):
        run(backend, events)
    (result,) = finished(events)
    assert result.code == "ERROR" and result.message == "ZeroDivisionError"
    assert "ZeroDivisionError" in result.details["traceback"]
    assert not backend.commands and backend.stopped == 0


@pytest.mark.parametrize("failure", ["raise", "return"])
def test_an_undeclared_code_is_an_error_that_still_stops_the_robot(monkeypatch, failure):
    aborts = []
    execute = (raising(SkillFailure("MADE_UP", "not declared")) if failure == "raise"
               else lambda plan, ctx: Result("fake", "MADE_UP", "not declared"))
    fake_skill(monkeypatch, execute, abort=lambda plan, ctx: aborts.append(plan))
    backend, events = FakeBackend(), []
    with pytest.raises(RuntimeError, match="undeclared"):
        run(backend, events)
    (result,) = finished(events)
    assert result.code == "ERROR" and result.message == "RuntimeError"
    assert backend.stopped == 1 and not aborts


def test_a_declared_failure_aborts_then_stops_and_keeps_its_code(monkeypatch):
    def abort(plan, ctx):
        raise SkillFailure("CANCELLED", "operator stopped the park")
    fake_skill(monkeypatch, raising(SkillFailure("BROKEN", "the skill's own reason")), abort)
    backend, events = FakeBackend(), []
    result = run(backend, events)
    assert result.code == "BROKEN"
    assert result.message == "the skill's own reason; abort: operator stopped the park"
    assert finished(events) == [result] and backend.stopped == 1


@pytest.mark.parametrize("error,code,aborts_arm", [
    (SkillFailure("CANCELLED", "cancelled"), "CANCELLED", False),
    (ActionRejected("refused"), "ACTION_REJECTED", True),
    (PhysicsInvalid("exploded"), "PHYSICS_INVALID", False),
    (ObservationError("Selected target is no longer sufficiently visible"), "TARGET_LOST", True)])
def test_failures_during_execution_abort_except_cancellation_and_invalid_physics(
        monkeypatch, error, code, aborts_arm):
    aborts = []
    fake_skill(monkeypatch, raising(error), abort=lambda plan, ctx: aborts.append(plan))
    backend, events = FakeBackend(), []
    result = run(backend, events)
    assert result.code == code and bool(aborts) == aborts_arm
    assert finished(events) == [result] and backend.stopped == 1


def test_a_bad_click_before_execution_is_an_invalid_argument(monkeypatch):
    def prepare(*args):
        raise ObservationError("Click is background")
    fake_skill(monkeypatch, raising(AssertionError("never executed"))).prepare = prepare
    result = run(FakeBackend(), [])
    assert (result.code, result.message) == ("INVALID_ARGUMENT", "Click is background")


@pytest.mark.parametrize("outcome", ["OK", "BROKEN"])
def test_a_rejected_stop_is_its_own_failure_only_after_success(monkeypatch, outcome):
    class RefusingBackend(FakeBackend):
        def base_velocity(self, epoch, forward, lateral, yaw_rate):
            raise ActionRejected("base refused")

        def stop(self, epoch):
            raise ActionRejected("stop refused")

    execute = (lambda plan, ctx: Result("fake", "OK", "done")) if outcome == "OK" else raising(
        SkillFailure("BROKEN", "failed"))
    fake_skill(monkeypatch, execute)
    backend, events = RefusingBackend(), []
    result = run(backend, events)
    assert result.code == ("ACTION_REJECTED" if outcome == "OK" else "BROKEN")
    assert result.message.endswith("refused") and "stop rejected" in result.message
    assert finished(events) == [result]


def test_unknown_skill_is_an_invalid_argument_without_actuation():
    backend, events = FakeBackend(), []
    result = run(backend, events, name="levitate")
    assert result.code == "INVALID_ARGUMENT" and finished(events) == [result]
    assert not backend.commands and backend.stopped == 0

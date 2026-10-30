"""A verified empty hand backs clear before strict placement resets the arm."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import PlanningArm
from test_skills import FakeBackend, frame

from homebody.motion.carry import CarryState
from homebody.primitives.observations import HandState
from homebody.skills import place
from homebody.skills.contract import Context, Runner, SkillFailure
from homebody.skills.manipulation_policy import REAL_MANIPULATION
from homebody.skills.registry import task_registry


@pytest.fixture
def placement(monkeypatch):
    current = frame()
    hands = dict(current.hands)
    hands["right"] = HandState(np.zeros(7), .8, np.ones(3), np.ones(3))
    current = replace(current, hands=hands, labels=np.zeros_like(current.labels))
    robot = FakeBackend(current)
    robot.grip = lambda *args: None
    arms = {side: PlanningArm() for side in ("left", "right")}
    record = {"label": 1, "state": CarryState.VERIFIED,
              "palm_to_payload": np.eye(4), "payload_half": np.full(3, .05)}
    events, order = [], []
    ctx = Context(robot, robot, arms, carried={"right": record},
                  manipulation=REAL_MANIPULATION, emit=events.append)
    palm = current.palm(arms["right"], "right")
    plan = place.Plan(current.epoch, "right", 1, palm[:3, 3].copy(),
                      np.zeros(7), palm.copy(), palm.copy(), support=False)
    dock = place.DockPlan(current.epoch, "right", 1, None, plan.point[:2].copy(), 0., np.eye(4))
    case = SimpleNamespace(ctx=ctx, robot=robot, record=record, plan=plan, dock=dock,
                           order=order, events=events, refusal=None, before_back=None)

    def open_hand(ctx, epoch, side, closure, seconds):
        order.append("open")
        hands = dict(robot.frame.hands)
        hands[side] = HandState(np.zeros(7), 0., np.zeros(3), np.zeros(3))
        robot.frame = replace(robot.frame, hands=hands)
        return robot.frame

    def back_out(ctx, epoch):
        if case.before_back is not None:
            case.before_back()
        ctx.check(epoch)
        order.append(("back", dict(ctx.carried)))
        return ctx.observe(epoch).base_pose, case.refusal

    def servo(*args, **kwargs):
        if "open" in order:
            order.append("retract")

    monkeypatch.setattr(place, "prepare", lambda *args: case.dock)
    monkeypatch.setattr(place, "dock_in", lambda *args: case.plan)
    monkeypatch.setattr(place, "continuing", lambda *args, **kwargs: True)
    monkeypatch.setattr(place, "move_joints", lambda *args: None)
    monkeypatch.setattr(place, "servo", servo)
    monkeypatch.setattr(place, "close_or_open", open_hand)
    monkeypatch.setattr(place, "servo_unchecked", lambda *args: order.append("retract"))
    monkeypatch.setattr(place, "return_arm", lambda *args: order.append("reset"))
    ctx.navigation = SimpleNamespace(back_out=back_out)
    return case


def run_placement(case):
    result = Runner(case.ctx, task_registry()).run(
        "place", {}, case.robot.frame)
    terminal = [event for event in case.events if event["type"] == "skill_finished"]
    assert len(terminal) == 1 and terminal[0]["result"] is result
    return result


@pytest.mark.parametrize("docked", [False, True])
def test_runner_strict_release_withdraws_empty_before_reset(placement, docked):
    case = placement
    if not docked:
        case.dock = case.plan
    result = run_placement(case)
    assert result.ok
    assert case.order == ["open", "retract", ("back", {}), "reset"]
    assert case.ctx.carried == {}
    assert case.robot.time == pytest.approx(case.ctx.settings.placement.settle_count * case.ctx.settings.placement.tick)


def test_post_release_verification_failure_holds_instead_of_resetting(placement, monkeypatch):
    case = placement

    def fail(*args, **kwargs):
        raise SkillFailure("PLACEMENT_NOT_VERIFIED", "visible target has not settled")

    monkeypatch.setattr(place, "verify_settled", fail)
    result = run_placement(case)
    assert result.code == "PLACEMENT_NOT_VERIFIED"
    assert "reset" not in case.order
    assert not any(isinstance(item, tuple) and item[0] == "back" for item in case.order)
    assert case.record["state"] == CarryState.UNCERTAIN
    assert case.robot.stopped == 1


@pytest.mark.parametrize("docked", [False, True])
@pytest.mark.parametrize("failure", ["CANCELLED", "STALE_EPOCH"])
def test_runner_interruption_during_withdrawal_never_resets(placement, failure, docked):
    case = placement
    if not docked:
        case.dock = case.plan

    def interrupt():
        if failure == "CANCELLED":
            case.ctx.cancelled = lambda: True
        else:
            case.robot.frame = replace(case.robot.frame, epoch=case.robot.epoch + 1)

    case.before_back = interrupt
    result = run_placement(case)
    assert result.code == failure
    assert case.order == ["open", "retract"]
    assert case.ctx.carried == {}


def test_unreachable_withdrawal_refuses_before_hand_opens(placement, monkeypatch):
    case = placement
    monkeypatch.setattr(case.ctx.arms["right"], "solve", lambda *args: None)
    monkeypatch.setattr(place, "compact_carry", lambda *args: None)
    result = run_placement(case)
    assert result.code == "UNREACHABLE"
    assert case.order == []
    assert case.ctx.carried["right"] is case.record
    assert case.record["state"] == CarryState.VERIFIED
    assert result.details["released"] is False
    assert result.details["phase"] == "withdrawal_preflight"

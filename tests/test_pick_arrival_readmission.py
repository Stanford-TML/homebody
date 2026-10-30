"""A failed final stance verification may admit a REAL grasp, never navigation arrival."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import PlanningArm
from test_pick_approach_abort import PickBackend
from test_real_navigation import evidence

from homebody.skills import pick
from homebody.skills.contract import Context, Result, Runner, SkillFailure
from homebody.skills.manipulation_policy import REAL_MANIPULATION
from homebody.skills.navigate_real import task
from homebody.skills.registry import task_registry


def refusal(code="ARRIVAL_NOT_VERIFIED", *, final=True):
    details = {"distance_m": .123, "arrival_distance_m": .1}
    if final:
        details["verification_phase"] = "settled_pose"
    return SkillFailure(code, "navigation did not verify the requested stance", details)


def setup_pick(monkeypatch, *, real=True, error=None):
    robot, events, resets, plans, grasps = PickBackend(), [], [], [], []
    drive = SimpleNamespace(epoch=1, radius_m=.30, execution_radius_m=.205, frame_id="test-room", map_hash="test-map-v1", generation=(1, 1),
                            payload_signature=task.payload_signature({}))
    approach = pick.ApproachPlan(1, "right", 1, drive)
    prepared = pick.Plan(1, "right", 1, (), np.eye(4), np.eye(4),
                         robot.frame.target_points(1).mean(axis=0), robot.frame.target_points(1))
    failure = error if error is not None else refusal()

    def navigate(plan, ctx):
        raise failure

    navigation = SimpleNamespace(execute=navigate, check_held_footprint=task.check_held_footprint,
                                 refine=lambda *args: pytest.fail("No further base refinement"))
    kwargs = {"manipulation": REAL_MANIPULATION} if real else {}
    ctx = Context(robot, robot, {side: PlanningArm() for side in ("left", "right")},
                  navigation=navigation, navigation_evidence=lambda: evidence(robot),
                  emit=events.append, paused=lambda: False, **kwargs)
    monkeypatch.setattr(pick, "prepare", lambda *args: approach)
    monkeypatch.setattr(pick, "reset_arm", lambda ctx, epoch, side: resets.append(side) or True)

    def plan(side, label, frame, planning):
        plans.append((side, label, frame, planning))
        return prepared

    def grasp(plan, ctx):
        grasps.append(plan)
        return Result("pick", "LIFT_NOT_VERIFIED", "execution still needs physical hold evidence")

    monkeypatch.setattr(pick, "plan_target", plan)
    monkeypatch.setattr(pick, "grasp", grasp)
    runner = Runner(ctx, task_registry())
    return SimpleNamespace(robot=robot, ctx=ctx, runner=runner, events=events, resets=resets,
                           plans=plans, grasps=grasps, prepared=prepared, drive=drive, failure=failure)


def run(case):
    return case.runner.run("pick", {}, case.robot.frame)


def terminal(events):
    return [row["result"] for row in events if row["type"] == "skill_finished"]


def test_final_refusal_uses_one_fresh_complete_plan_without_claiming_arrival(monkeypatch):
    case = setup_pick(monkeypatch)
    result = run(case)
    assert result.code == "LIFT_NOT_VERIFIED"
    assert len(case.plans) == 1 and case.grasps == [case.prepared]
    side, label, frame, _ = case.plans[0]
    assert (side, label) == ("right", 1) and frame.time > 0 and frame.sequence > 1
    admitted = [row for row in case.events if row.get("navigation_arrived") is False]
    assert len(admitted) == 1
    assert admitted[0]["navigation_failure"] == {
        "code": case.failure.code, "message": str(case.failure), "details": case.failure.details}
    assert admitted[0]["held_footprint"]["map_hash"] == case.drive.map_hash
    assert not case.resets and not case.robot.arm_commands and not case.robot.grip_commands
    assert not any(any(command) for command in case.robot.commands)
    assert not any(row.get("result", {}).get("arrived") for row in case.events
                   if isinstance(row.get("result"), dict))
    assert terminal(case.events) == [result]


def test_held_circle_blocks_even_when_a_grasp_planner_would_accept(monkeypatch):
    case = setup_pick(monkeypatch)
    occupied = case.robot.frame.navigation.occupied.copy()
    occupied[10, 10] = True  # blocks the held floor/body circle
    case.robot.frame = replace(case.robot.frame,
                               navigation=replace(case.robot.frame.navigation, occupied=occupied))
    result = run(case)
    assert result.code == "NO_ROUTE" and not case.plans and not case.grasps
    assert result.details["navigation_failure"]["code"] == "ARRIVAL_NOT_VERIFIED"
    assert not case.resets


@pytest.mark.parametrize("code", ["NO_GRASP", "CALIBRATION_MISSING", "HAND_OCCUPIED"])
def test_complete_grasp_preparation_refusal_never_opens_or_moves_hand(monkeypatch, code):
    case = setup_pick(monkeypatch)

    def refused(*args):
        raise SkillFailure(code, "existing grasp gate refused", {"blocker": "approach clearance"})

    monkeypatch.setattr(pick, "plan_target", refused)
    result = run(case)
    assert result.code == code and result.details["blocker"] == "approach clearance"
    assert result.details["navigation_failure"]["code"] == "ARRIVAL_NOT_VERIFIED"
    assert not case.grasps and not case.resets
    assert not case.robot.arm_commands and not case.robot.grip_commands

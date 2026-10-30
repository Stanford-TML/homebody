"""Measured docking selection and bounded carry lifting before any base movement."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import PlanningArm
from test_skills import FakeBackend, frame

from homebody.motion.carry import CarryState
from homebody.primitives.observations import ARM, HandState
from homebody.skills import place
from homebody.skills.contract import Context, SkillFailure
from homebody.skills.manipulation_policy import REAL_MANIPULATION


def lift_scene(monkeypatch, *, side="right", needed=.03, undertravel=0., gains=None):
    current = frame()
    hands = dict(current.hands)
    hands[side] = HandState(np.zeros(7), .8, np.ones(3), np.ones(3))
    robot = FakeBackend(replace(current, hands=hands))
    arm = PlanningArm()
    record = {"label": 1, "state": CarryState.VERIFIED,
              "palm_to_payload": np.eye(4), "payload_half": np.full(3, .05)}
    torso = np.eye(4)
    torso[:3, 3] = [7., -4., .2]
    plan = place.DockPlan(current.epoch, side, 1, None, np.array([.72, .14]), .03, torso)
    scene = SimpleNamespace(robot=robot, plan=plan, commands=[], floors=[], docks=[], releases=[],
                            openings=[], cancelled=False, retained=True, after_lift_blocked=False,
                            servo_failure=None, after_servo=lambda: None, release=object())
    robot.grip = lambda *args: scene.openings.append(args)
    initial_z = current.palm(arm, side)[2, 3]
    goal_z = initial_z + needed

    def distance(candidate, context, observed, selected, lift, turn):
        assert candidate is plan and context is scene.ctx
        assert selected is plan.torso
        return {0: .16, 10: .12, 20: .10, 30: .08}[round(abs(np.degrees(turn)))]

    def carry_lift(candidate, context, observed, selected, ahead, turn):
        assert candidate is plan and selected is plan.torso
        return max(0., goal_z - observed.palm(arm, side)[2, 3])

    def floor(context, observed, ahead, turn):
        angle = round(abs(np.degrees(turn)))
        scene.floors.append((observed.sequence, angle))
        if observed.sequence > current.sequence:
            return angle == 30 and not scene.after_lift_blocked
        return angle in (20, 30)

    def servo(context, epoch, hand, target, exclude_labels):
        assert context is scene.ctx and epoch == plan.epoch and hand == side
        assert exclude_labels == (plan.label,)
        context.check(epoch)
        if scene.servo_failure is not None:
            raise scene.servo_failure
        before = robot.frame.palm(arm, side)
        commanded = target[2, 3] - before[2, 3]
        scene.commands.append(commanded)
        gain = gains[len(scene.commands) - 1] if gains is not None else max(0., commanded - undertravel)
        measured = before.copy()
        measured[2, 3] += gain
        joints = robot.frame.joints.copy()
        joints[ARM[side]] = arm.solve(measured, robot.frame.arm_joints(side))
        robot.frame = replace(robot.frame, joints=joints, sequence=robot.frame.sequence + 1,
                              time=robot.frame.time + .1)
        scene.after_servo()

    def dock(context, epoch, ahead, turn):
        scene.docks.append((ahead, turn, robot.frame.sequence))
        robot.frame = replace(robot.frame, sequence=robot.frame.sequence + 1,
                              base_pose=np.array([2., 3., turn]))
        return robot.frame.base_pose, None

    def plan_release(hand, surface, point, height, selected, observed, planning):
        scene.releases.append((hand, surface, point, height, selected, observed))
        return scene.release

    ctx = Context(robot, robot, {side: arm}, carried={side: record},
                  manipulation=REAL_MANIPULATION, cancelled=lambda: scene.cancelled,
                  navigation=SimpleNamespace(dock_floor_clear=floor, dock=dock))
    scene.ctx, scene.record = ctx, record
    scene.remaining_lift = lambda: max(0., goal_z - robot.frame.palm(arm, side)[2, 3])
    monkeypatch.setattr(place, "docking_distance", distance)
    monkeypatch.setattr(place, "carry_lift", carry_lift)
    monkeypatch.setattr(place, "idle_clear", lambda *args: True)
    monkeypatch.setattr(place, "servo", servo)
    monkeypatch.setattr(place, "continuing", lambda *args: scene.retained)
    monkeypatch.setattr(place, "plan_release", plan_release)
    return scene


def assert_retained_without_docking(scene):
    assert scene.docks == [] and scene.releases == [] and scene.openings == []
    assert scene.ctx.carried[scene.plan.side] is scene.record
    assert scene.record["state"] == CarryState.VERIFIED
    assert scene.robot.frame.hands[scene.plan.side].closure == .8


@pytest.mark.parametrize("side, expected_turn", [("right", 30.), ("left", -30.)])
def test_docking_reselects_after_measured_lift_and_keeps_exact_target(monkeypatch, side, expected_turn):
    scene = lift_scene(monkeypatch, side=side)
    point, torso = scene.plan.point.copy(), scene.plan.torso.copy()

    assert place.dock_in(scene.plan, scene.ctx) is scene.release

    assert len(scene.commands) == 1
    assert scene.docks == [(pytest.approx(.08), pytest.approx(np.radians(expected_turn)), 2)]
    assert [angle for sequence, angle in scene.floors if sequence == 1] == [0, 10, 20]
    assert [angle for sequence, angle in scene.floors if sequence == 2] == [0, 10, 20, 30]
    hand, surface, selected_point, height, selected_torso, observed = scene.releases[0]
    assert hand == side and surface is scene.plan.surface and height == .03
    assert selected_point is scene.plan.point and selected_torso is scene.plan.torso
    assert observed is scene.robot.frame and observed.sequence == 3
    np.testing.assert_array_equal(scene.plan.point, point)
    np.testing.assert_array_equal(scene.plan.torso, torso)
    assert scene.openings == []


@pytest.mark.parametrize("gains, remaining", [([0., 0., 0.], .03), ([.008, .008, .008], .006)])
def test_three_lifts_without_measured_clearance_refuse_before_docking(monkeypatch, gains, remaining):
    scene = lift_scene(monkeypatch, gains=gains)

    with pytest.raises(SkillFailure) as failure:
        place.dock_in(scene.plan, scene.ctx)

    assert failure.value.code == "UNREACHABLE"
    assert len(scene.commands) == 3
    assert scene.remaining_lift() == pytest.approx(remaining)
    assert_retained_without_docking(scene)


def test_no_candidate_after_lift_refuses_without_moving_the_base(monkeypatch):
    scene = lift_scene(monkeypatch)
    scene.after_lift_blocked = True

    with pytest.raises(SkillFailure) as failure:
        place.dock_in(scene.plan, scene.ctx)

    assert failure.value.code == "UNREACHABLE"
    assert len(scene.commands) == 1
    assert_retained_without_docking(scene)


@pytest.mark.parametrize("change, code", [("cancelled", "CANCELLED"), ("retained", "HOLD_UNCERTAIN")])
def test_lift_cancellation_and_retention_failure_propagate_without_docking(monkeypatch, change, code):
    scene = lift_scene(monkeypatch)
    scene.after_servo = lambda: setattr(scene, change, change == "cancelled")

    with pytest.raises(SkillFailure) as failure:
        place.dock_in(scene.plan, scene.ctx)

    assert failure.value.code == code
    assert len(scene.commands) == 1
    assert_retained_without_docking(scene)

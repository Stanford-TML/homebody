"""Empty-arm departure folding uses observed map/depth geometry, never smaller margins."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_real_navigation import evidence
from test_skills import FakeBackend, frame

from homebody.helpers.geometry import inverse
from homebody.helpers.kinematics import Arm
from homebody.helpers.navigation import pixel
from homebody.motion.carry import robot_boxes, sweep_radius
from homebody.primitives.actions import ActionRejected
from homebody.primitives.observations import ARM, NavigationMap
from homebody.session.settings import Settings
from homebody.skills.contract import Context, Runner, SkillFailure
from homebody.skills.manipulation_policy import REAL_MANIPULATION
from homebody.skills.navigate_real import posture, task
from homebody.skills.registry import task_registry


def navigation(occupied=()):
    grid = np.zeros((80, 80), dtype=bool)
    for row, col in occupied:
        grid[row, col] = True
    return NavigationMap(grid, np.array([[.05, 0., -2.], [0., .05, -2.], [0., 0., 1.]]), .05)


def box(x, y, half=.02, yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([x, y, .8]), np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]), np.array([half, .02, .02])


def measured_pose(frame, side, q):
    joints = frame.joints.copy()
    joints[ARM[side]] = q
    return replace(frame, joints=joints)


@pytest.fixture(scope="module")
def arms():
    urdf = Path(__file__).resolve().parents[1] / "assets/robot/g1_29dof_with_hand.urdf"
    return {side: Arm(urdf, side) for side in ("left", "right")}


class FoldingRobot(FakeBackend):
    def __init__(self, settings):
        initial = frame()
        joints = initial.joints.copy()
        for side in ARM:
            joints[ARM[side]] = settings.placement.rest_arm(side)
        torso = np.eye(4)
        torso[2, 3] = .8
        super().__init__(replace(initial, joints=joints, base_pose=np.zeros(3), world_torso=torso,
                                 depth=np.full((6, 8), 10.), navigation=navigation()))
        self.arm_commands = []
        self.jam = False

    def arm_target(self, epoch, side, joints):
        super().arm_target(epoch, side, joints)
        self.arm_commands.append((side, np.asarray(joints).copy()))
        if not self.jam:
            self.frame = measured_pose(self.frame, side, joints)


class WalkingFoldingRobot(FoldingRobot):
    """The fake primitive moves its camera/torso with the measured base pose."""

    def advance(self, epoch, seconds):
        def base(pose):
            x, y, yaw = pose
            c, s = np.cos(yaw), np.sin(yaw)
            return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]])
        before = self.frame
        super().advance(epoch, seconds)
        delta = base(self.frame.base_pose) @ inverse(base(before.base_pose))
        self.frame = replace(self.frame, world_torso=delta @ before.world_torso,
                             world_camera=delta @ before.world_camera)


def task_setup(arms, *, blocked=True, obstacle=(37, 48)):
    settings = Settings()
    robot = WalkingFoldingRobot(settings)
    robot.frame = replace(robot.frame, navigation=navigation([obstacle] if blocked else []))
    events = []
    ctx = Context(robot, robot, arms, settings=settings, navigation=task,
                  manipulation=REAL_MANIPULATION, navigation_evidence=lambda: evidence(robot),
                  emit=events.append)
    return robot, ctx, Runner(ctx, task_registry()), events


def test_safe_goal_from_blocked_start_folds_before_actual_checked_navigation(arms):
    from homebody.motion.exploration import gait

    robot, ctx, runner, events = task_setup(arms)
    goal = [-.3, 0.]
    attempt = runner.prepare("navigate", {"goal_xy_m": goal, "snap": False}, robot.frame)
    assert not attempt.finished
    bridge = task._Bridge(ctx.planning(), robot.frame)
    checker = gait.occupancy_grid(bridge.artifact(), attempt.plan.radius_m, exact_footprint=True)[0]
    assert not checker.is_safe(*robot.frame.base_pose[:2])
    assert checker.is_safe(*goal)
    assert not robot.arm_commands and not robot.commands
    result = runner.execute(attempt)
    assert result.ok, result
    stages = [row for row in events if row.get("type") == "navigation"]
    kinds = [row["kind"] for row in stages]
    assert kinds.count("posture.prepared") == kinds.count("posture.settled") == 1
    assert kinds.index("posture.settled") < kinds.index("drive.request")
    settled = next(row["details"] for row in stages if row["kind"] == "posture.settled")
    assert settled["arrived"] is False
    assert np.allclose(settled["pose"], [0., 0., 0.])
    assert settled["planning_radius_m"] < attempt.plan.radius_m
    assert settled["remaining_s"] < ctx.settings.motion.drive_timeout
    assert result.details["goal_xy"] == goal
    assert result.details["navigation"]["goal_xy"] == goal
    assert np.linalg.norm(robot.frame.base_pose[:2] - goal) <= ctx.settings.motion.arrival_distance
    assert robot.time < ctx.settings.motion.drive_timeout
    assert len([row for row in events if row["type"] == "skill_finished"]) == 1


def setup(arms):
    settings = Settings()
    robot = FoldingRobot(settings)
    events = []
    ctx = Context(robot, robot, arms, settings=settings,
                  navigation_evidence=lambda: evidence(robot), emit=events.append)
    return robot, ctx, events


def admitted(ctx, limit=.375):
    return lambda current: sweep_radius(current, ctx.arms, {}, ctx.settings.motion.robot_radius) < limit


@pytest.mark.parametrize("condition", ["carried", "closed", "invalid_closure", "extended", "already_clear"])
def test_ineligible_posture_never_prepares_or_moves(arms, condition):
    robot, ctx, _ = setup(arms)
    if condition == "carried":
        ctx.carried["left"] = {"state": "verified"}
    elif condition in ("closed", "invalid_closure"):
        hands = dict(robot.frame.hands)
        hands["right"] = replace(hands["right"], closure=.9 if condition == "closed" else -.1)
        robot.frame = replace(robot.frame, hands=hands)
    elif condition == "extended":
        q = robot.frame.arm_joints("right").copy()
        q[3] += .7
        robot.frame = measured_pose(robot.frame, "right", q)
    plan = posture.prepare(robot.frame, ctx.planning(), evidence(robot),
                           admitted=lambda _: condition == "already_clear")
    assert plan is None and not robot.arm_commands


def test_observed_depth_collision_is_not_overridden_by_free_navigation_cells(arms):
    robot, ctx, _ = setup(arms)
    point = robot_boxes(robot.frame, arms, {})[-3][0]
    camera = np.eye(4)
    camera[:3, 3] = point - [0., 0., 1.]
    robot.frame = replace(robot.frame, depth=np.ones((6, 8)), world_camera=camera,
                          intrinsics=np.array([[500., 0., 4.], [0., 500., 3.], [0., 0., 1.]]))
    assert not posture._path_clear(robot.frame, ctx.planning(), evidence(robot), "right",
                                   robot.frame.arm_joints("right"))


@pytest.mark.parametrize("change", ["map", "generation", "closed", "obstacle", "hold"])
def test_changed_execution_evidence_prevents_arm_commands(arms, change):
    robot, ctx, _ = setup(arms)
    plan = posture.prepare(robot.frame, ctx.planning(), evidence(robot), admitted=admitted(ctx))
    sample = evidence(robot)
    if change == "map":
        sample = replace(sample, map_hash="new-map")
    elif change == "generation":
        sample = replace(sample, generation=(1, 2))
    elif change == "closed":
        hands = dict(robot.frame.hands)
        hands["left"] = replace(hands["left"], closure=.9)
        robot.frame = replace(robot.frame, hands=hands)
    elif change == "obstacle":
        point = robot_boxes(robot.frame, arms, {})[-3][0][:2]
        sample = replace(sample, navigation=navigation([pixel(sample.navigation, point)]))
    else:
        def failed_hold(_):
            raise ActionRejected("test hold failure")
        robot.stop = failed_hold
    with pytest.raises((SkillFailure, ActionRejected)):
        posture.execute(ctx, plan, validate_evidence=lambda: sample)
    assert not robot.arm_commands


def test_cancellation_is_bounded_without_base_motion(arms):
    robot, ctx, _ = setup(arms)
    plan = posture.prepare(robot.frame, ctx.planning(), evidence(robot), admitted=admitted(ctx))
    ctx.cancelled = lambda: robot.time >= .04
    with pytest.raises(SkillFailure, match="cancelled"):
        posture.execute(ctx, plan, validate_evidence=lambda: evidence(robot))
    assert not robot.commands
    assert len(robot.arm_commands) <= 4


def test_jammed_arm_times_out_without_claiming_settled(arms):
    robot, ctx, events = setup(arms)
    plan = posture.prepare(robot.frame, ctx.planning(), evidence(robot), admitted=admitted(ctx))
    robot.jam = True
    ctx.settings = replace(ctx.settings, motion=replace(ctx.settings.motion, arm_timeout=.08))
    with pytest.raises(SkillFailure, match="did not settle"):
        posture.execute(ctx, plan, validate_evidence=lambda: evidence(robot))
    assert not any(event["type"] == "navigation_posture_settled" for event in events)
    assert not robot.commands


@pytest.mark.parametrize("missing", ["depth", "singular_map", "unregistered", "nonmobile"])
def test_unavailable_geometry_cannot_authorize_an_empty_fold(arms, missing):
    robot, ctx, _ = setup(arms)
    sample = evidence(robot)
    if missing == "depth":
        robot.frame = replace(robot.frame, depth=np.full(robot.frame.depth.shape, np.nan))
    elif missing == "singular_map":
        sample = replace(sample, navigation=replace(sample.navigation, T_map_px=np.zeros((3, 3))))
    else:
        sample = replace(sample, **{"registered" if missing == "unregistered" else "mobile": False})
    assert posture.prepare(robot.frame, ctx.planning(), sample, admitted=admitted(ctx)) is None
    assert not robot.arm_commands and not robot.commands

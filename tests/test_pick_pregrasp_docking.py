"""Pre-grasp docking through Runner and the checked navigation owner."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import PlanningArm
from test_pick_approach_abort import PickBackend, terminals
from test_real_navigation import evidence

from homebody.primitives.actions import ActionRejected
from homebody.skills import pick
from homebody.skills.contract import Context, Result, Runner
from homebody.skills.manipulation_policy import REAL_MANIPULATION
from homebody.skills.navigate_real import task as navigation

ARGS = {"hand": "right", "point_normalized_1000": [500, 500]}


def setup(monkeypatch):
    robot = PickBackend()
    camera = np.eye(4)
    camera[:2, 3] = [1.58, 1.]
    robot.frame = replace(robot.frame, world_camera=camera)
    jaw = np.eye(4)
    jaw[0, 3] = .1
    events, calls, resets = [], [], []
    ctx = Context(robot, robot, {side: PlanningArm() for side in ("left", "right")},
                  jaw_samples={"right": [{"palm_to_jaw": jaw}]}, navigation=navigation,
                  manipulation=REAL_MANIPULATION, navigation_evidence=lambda: evidence(robot),
                  emit=events.append)
    monkeypatch.setattr(navigation, "robot_boxes", lambda current, *a: [
        (np.r_[current.base_pose[:2], 1.], np.eye(3), np.full(3, .03))])
    monkeypatch.setattr(navigation, "robot_part_names", lambda *a: ["right test link"])
    monkeypatch.setattr(navigation, "prepare", lambda *a, **k: SimpleNamespace(goal=np.array([1., 1.]), yaw=0.))
    monkeypatch.setattr(navigation, "execute", lambda *a: pytest.fail("No ordinary walk in DockPlan"))
    monkeypatch.setattr(navigation, "refine", lambda *a: pytest.fail("No retry/refinement"))
    floor, dock = navigation.dock_floor_clear, navigation.dock

    def check_floor(*args):
        calls.append(("floor", args[2], args[3]))
        return floor(*args)

    def checked_dock(*args):
        calls.append(("dock", args[2], args[3]))
        result = dock(*args)
        calls.append(("settled", robot.frame))
        return result

    def plan(side, label, current, planning):
        calls.append(("plan", current))
        assert pick.in_grasp_range(*pick.grasp_offset(side, label, current, planning)[1:], planning.settings.grasp)
        return SimpleNamespace(epoch=current.epoch, side=side, label=label)

    monkeypatch.setattr(navigation, "dock_floor_clear", check_floor)
    monkeypatch.setattr(navigation, "dock", checked_dock)
    monkeypatch.setattr(pick, "plan_target", plan)
    monkeypatch.setattr(pick, "grasp", lambda p, c: calls.append(("grasp",)) or Result("pick", "OK", "test grasp boundary"))
    monkeypatch.setattr(pick, "reset_arm", lambda c, e, s: resets.append(s) or True)
    return robot, ctx, Runner(ctx, {"pick": pick}), events, calls, resets


def assert_pregrasp_failure(robot, events, calls, resets, result):
    assert not result.ok and result.details["grasp_started"] is False
    assert result.details["pick_phase"] == "dock_in"
    assert not any(c[0] in ("plan", "grasp") for c in calls)
    assert not resets and not robot.arm_commands and not robot.grip_commands
    assert not robot.velocity.any() and terminals(events) == [result]


def test_runner_docks_once_then_plans_from_new_settled_frame(monkeypatch):
    robot, _ctx, runner, events, calls, resets = setup(monkeypatch)
    attempt = runner.prepare("pick", ARGS, robot.frame)
    assert not attempt.finished, attempt.result
    assert isinstance(attempt.plan, pick.DockPlan)
    assert not attempt.plan.points.flags.writeable
    result = runner.execute(attempt)
    assert result.ok, result
    assert [c[0] for c in calls] == ["floor", "dock", "settled", "plan", "grasp"]
    assert calls[0][1:] == calls[1][1:] and calls[0][2] == 0.
    assert calls[3][1].time > calls[2][1].time
    assert calls[3][1].sequence > calls[2][1].sequence
    assert any(any(c) for c in robot.commands)
    assert not resets and terminals(events) == [result]


@pytest.mark.parametrize("fault", ["unobserved", "occupied", "cancel", "stale", "deadline", "postclamp", "hold"])
def test_actual_checked_docking_dispatch_retains_refusal_gates(monkeypatch, fault):
    robot, ctx, runner, events, calls, resets = setup(monkeypatch)
    original_check, original_select, original_clamp = navigation._Bridge._dock_check, navigation._dock_command, navigation.clamp_lower
    checks, runtimes = [], []
    selected = False
    stale = False
    ctx.navigation_evidence = lambda: evidence(robot, pose_time=robot.time - 1.) if stale else evidence(robot)
    def check(bridge, vx, vy, wz):
        if bridge.docking and (vx or vy or wz):
            checks.append((vx, vy, wz))
            assert bridge._dock_floor_checker(bridge.sample()).radius_baked == ctx.settings.motion.robot_radius + ctx.settings.collision.cloud_margin
            if fault in ("unobserved", "occupied") or fault == "postclamp" and selected:
                raise navigation.MapObstacleRefused("component refuses", details={"evidence": fault})
        return original_check(bridge, vx, vy, wz)
    def select(bridge, deadline, *command):
        nonlocal selected, stale
        runtimes.append(deadline)
        if fault == "deadline":
            deadline._end = robot.time
        value = original_select(bridge, deadline, *command)
        selected = True
        if fault == "cancel":
            ctx.cancelled = lambda: True
        elif fault == "stale":
            stale = True
            robot.frame = replace(robot.frame, time=robot.time + 1.)
        return value
    def clamp(command, observer):
        result = original_clamp(command, observer)
        return replace(result, vx=result.vx * 1.2) if fault == "postclamp" and selected else result
    if fault == "hold":
        original_velocity = robot.base_velocity
        def velocity(epoch, *command):
            if not any(command):
                raise ActionRejected("hold rejected")
            return original_velocity(epoch, *command)
        monkeypatch.setattr(robot, "base_velocity", velocity)
    monkeypatch.setattr(navigation._Bridge, "_dock_check", check)
    monkeypatch.setattr(navigation, "_dock_command", select)
    monkeypatch.setattr(navigation, "clamp_lower", clamp)
    result = runner.run("pick", ARGS, robot.frame)
    assert_pregrasp_failure(robot, events, calls, resets, result)
    assert result.code == {"cancel": "CANCELLED", "stale": "STALE_OBSERVATION", "deadline": "TIMEOUT"}.get(fault, "ACTION_REJECTED")
    assert len({id(d) for d in runtimes}) == 1
    if fault != "hold":
        assert not any(any(c) for c in robot.commands)
    if fault == "postclamp":
        assert checks[-1][0] == pytest.approx(checks[0][0] * 1.2)


@pytest.mark.parametrize("fault", ["cancel", "epoch", "stale", "ownership"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_execution_revalidates_context_before_motion_and_grasp(monkeypatch, fault, when):
    robot, ctx, runner, events, calls, resets = setup(monkeypatch)
    attempt = runner.prepare("pick", ARGS, robot.frame)
    assert not attempt.finished, attempt.result
    def change():
        if fault == "cancel":
            ctx.cancelled = lambda: True
        elif fault == "epoch":
            robot.frame = replace(robot.frame, epoch=2)
        elif fault == "stale":
            old = robot.frame
            robot.frame = replace(old, time=old.time + 1.)
            monkeypatch.setattr(robot, "snapshot", lambda: old)
        else:
            ctx.carried["left"] = {"label": 9, "state": "uncertain"}
    if when == "before":
        change()
    else:
        original = navigation.dock
        def changed(*args):
            result = original(*args)
            change()
            return result
        monkeypatch.setattr(navigation, "dock", changed)
    result = runner.execute(attempt)
    assert result.code == {"cancel": "CANCELLED", "epoch": "STALE_EPOCH",
                           "stale": "STALE_OBSERVATION", "ownership": "UNREACHABLE"}[fault]
    assert not any(c[0] in ("plan", "grasp") for c in calls)
    assert not resets and not robot.arm_commands and not robot.grip_commands
    assert terminals(events) == [result]
    if when == "before":
        assert not any(any(c) for c in robot.commands)

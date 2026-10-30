"""Close docking may select one shorter checked command before any gait tick."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import PlanningArm
from test_real_navigation import context
from test_skills import FakeBackend, frame

from homebody.primitives.navigation.errors import MapObstacleRefused
from homebody.skills.contract import SkillFailure
from homebody.skills.navigate_real import task as navigation


def docking_scene(*, wall=None, delta=(.12, 0.), tick=.004, stall=.004):
    occupied = np.zeros((500, 500), bool)
    if wall is not None:
        occupied[:, round(wall / .005)] = True
    current = frame()
    current = replace(current, navigation=replace(current.navigation, occupied=occupied,
                      resolution=.005, T_map_px=np.diag([.005, .005, 1.])))
    robot = FakeBackend(current)
    ctx, _, events = context(robot)
    arm = PlanningArm()
    arm.chain = ()
    arm.collision = SimpleNamespace(
        boxes=lambda *_: [(np.array([1., 1., 1.]), np.eye(3), np.full(3, .02))],
        hand_boxes=lambda *_: [], finger_joints=(), bounds={})
    ctx.arms = {"right": arm}
    ctx.settings = replace(ctx.settings,
        motion=replace(ctx.settings.motion, robot_radius=.20, min_close_speed=.22,
                       final_speed=.25, tick=tick, stall_timeout=stall),
        collision=replace(ctx.settings.collision, cloud_margin=.005))
    bridge = navigation._Bridge(ctx.planning(), robot.frame, ctx)
    bridge.docking = True
    live = bridge.context()
    target = robot.frame.base_pose[:2] + delta
    return robot, ctx, bridge, live, target, events


def record_checks(monkeypatch, bridge):
    checks, original = [], bridge._motion_check

    def checked(*command):
        checks.append(command)
        return original(*command)

    monkeypatch.setattr(bridge, "_motion_check", checked)
    return checks


@pytest.mark.parametrize("delta,wall,expected", [
    ((.12, 0.), 1.295, (.153, 0., 0.)),
    ((.12, .06), 1.285, (.153, .0765, 0.)),
])
def test_shorter_same_direction_sweep_is_selected_before_any_write(delta, wall, expected):
    robot, ctx, bridge, live, target, events = docking_scene(wall=wall, delta=delta)
    original_target = target.copy()
    with pytest.raises(MapObstacleRefused):
        direction = np.asarray(delta) / np.linalg.norm(delta)
        bridge._motion_check(*(direction * .22), 0.)
    bridge._motion_check(*expected)
    assert robot.commands == []
    navigation._walk_to(bridge, live, target, 0., ctx.settings.motion)
    np.testing.assert_allclose(robot.commands, [expected], atol=1e-12)
    np.testing.assert_array_equal(target, original_target)
    assert any(event.get("kind") == "dock.slowed" for event in events)


def test_both_sweeps_refuse_without_actuation_and_keep_dock_failure_path(monkeypatch):
    robot, ctx, bridge, live, target, _ = docking_scene(wall=1.27)
    checks = record_checks(monkeypatch, bridge)
    refused = navigation._dock_walk(bridge, live, target, 0., ctx.settings.motion)
    assert refused is not None
    np.testing.assert_allclose(checks, [(.22, 0., 0.), (.153, 0., 0.)])
    np.testing.assert_allclose(refused["command"], [.153, 0., 0.])
    assert robot.commands == []
    assert bridge.docking is False


@pytest.mark.parametrize("code", ["STALE_OBSERVATION", "CANCELLED", "STALE_EPOCH"])
def test_non_geometric_failure_never_selects_a_smaller_command(monkeypatch, code):
    robot, ctx, bridge, live, target, _ = docking_scene()
    calls = []

    def fail(*command):
        calls.append(command)
        raise SkillFailure(code, "Injected sensor, control, or carry invalidation")

    monkeypatch.setattr(bridge, "_motion_check", fail)
    with pytest.raises(SkillFailure) as failure:
        navigation._walk_to(bridge, live, target, 0., ctx.settings.motion)
    assert failure.value.code == code
    assert calls == [(.22, 0., 0.)]
    assert robot.commands == []

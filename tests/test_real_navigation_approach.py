"""Loaded REAL approach uses the checked aim/close path in one action."""
from dataclasses import replace

import numpy as np
import pytest
from test_real_navigation import loaded_navigation

from homebody.helpers.raster import world_to_px
from homebody.primitives.navigation.errors import MapObstacleRefused
from homebody.skills.navigate_real import task as navigation

GOAL = np.array([3., 3.])
YAW = .5


def request():
    return {"goal_xy_m": GOAL.tolist(), "yaw": YAW, "snap": False}


def test_blocked_original_goal_cannot_be_replaced_by_clear_approach():
    robot, _ctx, runner, _ = loaded_navigation(compactable=True)
    nav = robot.frame.navigation
    occupied = nav.occupied.copy()
    px = np.rint(world_to_px(nav.T_map_px, GOAL)).astype(int)[0]
    occupied[px[1], px[0]] = True
    robot.frame = replace(robot.frame, navigation=replace(nav, occupied=occupied))
    attempt = runner.prepare("navigate", request(), robot.frame)
    assert attempt.finished and not attempt.result.ok
    assert robot.commands == []


def test_run_in_aims_before_closing_with_original_runtime_and_result(monkeypatch):
    robot, ctx, runner, events = loaded_navigation(compactable=True)
    runtimes, starts = [], []
    original_runtime, original_stance = navigation._navigation_runtime, navigation._stance_to

    def runtime(*args, **kwargs):
        result = original_runtime(*args, **kwargs)
        runtimes.append(result)
        return result

    def stance(bridge, live, target, yaw, motion, **kwargs):
        starts.append((robot.frame.base_pose.copy(), len(robot.commands)))
        assert live.deadline is runtimes[0][1].deadline
        assert live.gate is runtimes[0][1].gate
        np.testing.assert_array_equal(target, GOAL)
        assert yaw == YAW
        return original_stance(bridge, live, target, yaw, motion, **kwargs)

    monkeypatch.setattr(navigation, "_navigation_runtime", runtime)
    monkeypatch.setattr(navigation, "_stance_to", stance)
    result = runner.run("navigate", request(), robot.frame)
    assert result.ok, result
    assert len(runtimes) == len(starts) == 1
    assert np.linalg.norm(starts[0][0][:2] - GOAL) > .25
    commands = robot.commands[starts[0][1]:]
    turning = [i for i, command in enumerate(commands) if abs(command[2]) > .01]
    translating = [i for i, command in enumerate(commands) if np.linalg.norm(command[:2]) > .01]
    assert turning and translating and turning[0] < translating[0]
    assert result.details["goal_xy"] == GOAL.tolist()
    assert np.linalg.norm(robot.frame.base_pose[:2] - GOAL) <= ctx.settings.motion.arrival_distance
    assert abs(robot.frame.base_pose[2] - YAW) <= ctx.settings.motion.arrival_angle
    assert sum(e.get("kind") == "drive.request" for e in events) == 1
    assert sum(e["type"] == "skill_started" for e in events) == 1
    assert sum(e["type"] == "skill_finished" for e in events) == 1


@pytest.mark.parametrize("fault", ["occupied", "unobserved", "deadline", "cancel"])
def test_planned_approach_preserves_command_refusal_and_action_bounds(monkeypatch, fault):
    robot, ctx, runner, events = loaded_navigation(compactable=True)
    original_stance, original_check = navigation._stance_to, navigation._Bridge._dock_check
    starts = []

    def stance(bridge, live, *args, **kwargs):
        starts.append(len(robot.commands))
        if fault == "deadline":
            live.deadline._end = robot.time
        return original_stance(bridge, live, *args, **kwargs)

    def check(bridge, *command):
        if starts and fault in ("occupied", "unobserved"):
            raise MapObstacleRefused("component evidence refusal", details={"evidence": fault})
        return original_check(bridge, *command)

    monkeypatch.setattr(navigation, "_stance_to", stance)
    monkeypatch.setattr(navigation._Bridge, "_dock_check", check)
    ctx.cancelled = lambda: bool(starts) and fault == "cancel"
    result = runner.run("navigate", request(), robot.frame)
    assert starts and not result.ok
    assert result.code == {"deadline": "TIMEOUT", "cancel": "CANCELLED"}.get(fault, "ACTION_REJECTED")
    assert not any(any(c) for c in robot.commands[starts[0]:])
    assert sum(e["type"] == "skill_finished" for e in events) == 1
    assert np.allclose(robot.velocity, 0.)

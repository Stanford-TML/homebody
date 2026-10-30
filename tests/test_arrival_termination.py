"""Ordinary arrival stops before unsent motion, then certifies the current held state."""
from dataclasses import replace

import numpy as np
import pytest
from test_real_navigation import evidence, loaded_navigation, marked

from homebody.skills.navigate_real import task as navigation


def arrival_run(monkeypatch, fault=None, *, already=False):
    robot, ctx, runner, events = loaded_navigation()
    goal, yaw = np.array([2.1, 3.]), .5
    state = {"crossed": None, "zero": None, "stationary": [], "settles": []}
    stance, advance = navigation._stance_to, robot.advance
    check, dock = navigation._Bridge._motion_check, navigation._Bridge._dock_check
    hold = navigation.gait.hold

    def crossing_pose():
        distance = .101 if fault == "outside" else .0947
        angle = .101 if fault == "heading" else .0695
        robot.frame = replace(robot.frame, base_pose=np.r_[goal + [distance, 0.], yaw + angle])

    def start(bridge, live, target, heading, motion, **kwargs):
        state.update(active=True, live=live)
        robot.frame = replace(robot.frame, base_pose=np.r_[goal + [.12, 0.], yaw + .04])
        if already:
            crossing_pose()
            state["crossed"] = len(robot.commands)
        return stance(bridge, live, target, heading, motion, **kwargs)

    def step(epoch, seconds):
        advance(epoch, seconds)
        if not state.get("active"):
            return
        if state["crossed"] is None and np.any(robot.velocity):
            crossing_pose()
            state["crossed"] = len(robot.commands)
        elif state["crossed"] is not None and state["zero"] is None and not np.any(robot.velocity):
            state["zero"] = len(robot.commands) - 1
            if fault == "rebound":
                robot.frame = replace(robot.frame, base_pose=np.r_[goal + [.2, 0.], yaw])
            elif fault == "base":
                robot.frame = replace(robot.frame, navigation=marked(
                    robot.frame.navigation, np.r_[robot.frame.base_pose[:2], 0.][None]))
            elif fault == "deadline":
                state["live"].deadline._end = robot.time
            elif fault == "retention":
                hands = dict(robot.frame.hands)
                hands["right"] = replace(hands["right"], finger_effort_fraction=np.zeros(3))
                robot.frame = replace(robot.frame, hands=hands, labels=np.zeros_like(robot.frame.labels))
            elif fault == "payload":
                ctx.carried["right"]["payload_half"] = np.array([.7, .1, .1])

    def motion(bridge, vx, vy, wz):
        if state["crossed"] is not None and (vx or vy or wz):
            raise navigation.MapObstacleRefused("Unsent prospective sweep", details={"witness": "prospective"})
        return check(bridge, vx, vy, wz)

    def stationary(bridge, vx, vy, wz):
        if state["zero"] is not None and not (vx or vy or wz):
            state["stationary"].append((robot.time, bridge._measure.time))
            if fault == "component":
                raise navigation.MapObstacleRefused("Current idle finger blocked", details={"witness": "idle_finger"})
            result = dock(bridge, vx, vy, wz)
            if fault == "geometry_deadline":
                state["live"].deadline._end = robot.time
            elif fault == "geometry_age":
                robot.frame = replace(robot.frame, time=robot.time + 1.)
            elif fault == "ray_expiry":
                bridge._dock_ray_expiry = robot.time - .001
            return result
        return dock(bridge, vx, vy, wz)

    def settled(live, seconds, vx, vy, wz, **kwargs):
        if state.get("active") and seconds == ctx.settings.motion.settle_seconds and not (vx or vy or wz):
            state["settles"].append(robot.time)
        return hold(live, seconds, vx, vy, wz, **kwargs)

    ctx.cancelled = lambda: state["zero"] is not None and fault == "cancel"
    ctx.navigation_evidence = lambda: evidence(robot, **(
        {"pose_time": -1.} if state["zero"] is not None and fault == "stale" else
        {"generation": (1, 2)} if state["zero"] is not None and fault == "generation" else {}))
    monkeypatch.setattr(navigation, "_stance_to", start)
    monkeypatch.setattr(navigation._Bridge, "_motion_check", motion)
    monkeypatch.setattr(navigation._Bridge, "_dock_check", stationary)
    monkeypatch.setattr(navigation.gait, "hold", settled)
    robot.advance = step
    result = runner.run("navigate", {"goal_xy_m": goal.tolist(), "yaw": yaw, "snap": False}, robot.frame)
    assert state["crossed"] is not None
    assert not any(any(command) for command in robot.commands[state["crossed"]:])
    assert sum(e.get("type") == "skill_finished" for e in events) == 1
    assert np.allclose(robot.velocity, 0.)
    return result, state, events


@pytest.mark.parametrize("already", [False, True])
def test_arrival_joint_bounds_stop_before_unsent_sweep_then_verify(monkeypatch, already):
    result, state, events = arrival_run(monkeypatch, already=already)
    assert result.ok, result
    assert state["zero"] is not None
    assert len(state["settles"]) == 1
    assert state["stationary"]
    assert all(now == measured for now, measured in state["stationary"])
    assert sum(e.get("kind") == "drive.arrival_correction" for e in events) == 1


@pytest.mark.parametrize("fault,code", [("outside", "ACTION_REJECTED"), ("heading", "ACTION_REJECTED"),
    ("rebound", "ACTION_REJECTED"), ("base", "ACTION_REJECTED"),
    ("component", "ACTION_REJECTED"), ("cancel", "CANCELLED"), ("stale", "STALE_OBSERVATION"),
    ("generation", "STALE_EPOCH"), ("deadline", "TIMEOUT"), ("geometry_deadline", "TIMEOUT"),
    ("geometry_age", "STALE_OBSERVATION"), ("ray_expiry", "STALE_OBSERVATION"),
    ("retention", "HOLD_UNCERTAIN"), ("payload", "STALE_EPOCH")])
def test_stop_crossing_never_hides_current_failure_or_settled_miss(monkeypatch, fault, code):
    result, _state, _events = arrival_run(monkeypatch, fault)
    assert result.code == code, result
    if fault == "component":
        assert result.details["witness"] == "idle_finger"
    elif fault in ("outside", "heading", "rebound"):
        assert result.details["witness"] == "prospective"

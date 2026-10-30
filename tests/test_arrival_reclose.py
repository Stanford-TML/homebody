"""Checked ordinary arrival can reclose a bounded miss after zero-command settling."""
from dataclasses import replace

import numpy as np
import pytest
from test_real_navigation import evidence, loaded_navigation, marked

from homebody.skills.navigate_real import task as navigation


def reclose_run(monkeypatch, settled, fault=None):
    robot, ctx, runner, events = loaded_navigation()
    goal, yaw = np.array([2.1, 3.]), .5
    state = {"settles": [], "commands": [], "deadlines": [], "active": False}
    stance, hold = navigation._stance_to, navigation.gait.hold
    motion_check = navigation._Bridge._motion_check

    def pose(distance, angle):
        robot.frame = replace(robot.frame, base_pose=np.r_[goal + [distance, 0.], yaw + angle])

    def start(bridge, live, target, heading, motion, **kwargs):
        state["active"] = True
        state["deadline"] = live.deadline
        pose(.0995, .0687)
        return stance(bridge, live, target, heading, motion, **kwargs)

    def settling(live, seconds, vx, vy, wz, **kwargs):
        if not state["active"]:
            return hold(live, seconds, vx, vy, wz, **kwargs)
        state["deadlines"].append(live.deadline)
        is_settle = seconds == ctx.settings.motion.settle_seconds and not (vx or vy or wz)
        result = hold(live, seconds, vx, vy, wz, **kwargs)
        if is_settle:
            index = len(state["settles"])
            state["settles"].append(robot.time)
            state["commands"].append(len(robot.commands))
            pose(*settled[min(index, len(settled) - 1)])
            if fault == "deadline":
                live.deadline._end = robot.time
            elif fault == "base":
                robot.frame = replace(robot.frame, navigation=marked(
                    robot.frame.navigation, np.r_[robot.frame.base_pose[:2], 0.][None]))
            elif fault == "retention":
                hands = dict(robot.frame.hands)
                hands["right"] = replace(hands["right"], finger_effort_fraction=np.zeros(3))
                robot.frame = replace(robot.frame, hands=hands, labels=np.zeros_like(robot.frame.labels))
        return result

    def check(bridge, vx, vy, wz):
        if state["settles"] and fault == "prospective" and (vx or vy or wz):
            raise navigation.MapObstacleRefused("Blocked reclose", details={"witness": "reclose"})
        return motion_check(bridge, vx, vy, wz)

    ctx.cancelled = lambda: bool(state["settles"]) and fault == "cancel"
    ctx.navigation_evidence = lambda: evidence(robot, **(
        {"pose_time": -1.} if state["settles"] and fault == "stale" else {}))
    monkeypatch.setattr(navigation, "_stance_to", start)
    monkeypatch.setattr(navigation.gait, "hold", settling)
    monkeypatch.setattr(navigation._Bridge, "_motion_check", check)
    result = runner.run("navigate", {"goal_xy_m": goal.tolist(), "yaw": yaw, "snap": False}, robot.frame)
    assert all(deadline is state["deadline"] for deadline in state["deadlines"])
    assert sum(e.get("type") == "skill_finished" for e in events) == 1
    assert np.allclose(robot.velocity, 0.)
    return result, state, robot, events


def test_settled_drift_recloses_through_checked_commands_before_success(monkeypatch):
    result, state, robot, events = reclose_run(monkeypatch, [(.1077, .0528), (.099, .05)])
    assert result.ok, result
    assert len(state["settles"]) == 2
    assert any(any(c) for c in robot.commands[state["commands"][0]:state["commands"][1]])
    assert sum(e.get("kind") == "drive.arrival_correction" for e in events) == 1


@pytest.mark.parametrize("fault,code", [("prospective", "ACTION_REJECTED"), ("base", "ACTION_REJECTED"),
    ("cancel", "CANCELLED"), ("stale", "STALE_OBSERVATION"), ("deadline", "TIMEOUT"),
    ("retention", "HOLD_UNCERTAIN")])
def test_reclose_never_retries_or_hides_checked_failure(monkeypatch, fault, code):
    result, state, robot, _events = reclose_run(monkeypatch, [(.1077, .0528)], fault)
    assert result.code == code, result
    assert len(state["settles"]) == 1
    if fault != "retention":
        assert not any(any(c) for c in robot.commands[state["commands"][0]:])
    else:
        # The retention policy may admit motion during its detection grace period;
        # it must end this cycle before another settle.
        assert "Navigation requires current verified carry evidence" in result.message
    if fault == "prospective":
        assert result.details["witness"] == "reclose"

"""Stop an unsent stance slice once the joint distance and heading predicate holds, then verify settling."""
from dataclasses import replace

import numpy as np
from test_navigation_facing_correction import FacingDrift
from test_real_navigation import context, loaded_navigation

from homebody.helpers.raster import world_to_px
from homebody.skills.navigate_real import task as navigation


def crossing_run(monkeypatch, *, loaded=False, fault=None, heading_sign=1):
    if loaded:
        robot, ctx, runner, events = loaded_navigation()
        goal, yaw = np.array([2.1, 3.]), 1.2
    else:
        robot = FacingDrift()
        ctx, runner, events = context(robot)
        goal, yaw = np.array([1.6, 1.]), .5
    original_stance, original_advance = navigation._stance_to, robot.advance
    original_check = navigation._Bridge._motion_check
    state = {"active": False, "crossed": None, "zero": None, "refusals": 0}
    original_evidence = ctx.navigation_evidence
    ctx.navigation_evidence = lambda: (replace(original_evidence(), pose_time=-1.)
        if ((fault == "stale" and state["crossed"] is not None)
            or (fault == "after_stop_stale" and state["zero"] is not None)) else original_evidence())

    def stance(bridge, live, target, heading, motion, **kwargs):
        assert np.allclose(target, goal)
        pose = np.r_[goal + [.033003, 0.], yaw + heading_sign * .034653]
        robot.frame = replace(robot.frame, base_pose=pose)
        state.update(active=True, live=live)
        return original_stance(bridge, live, target, heading, motion, arrival=False)

    def advance(epoch, seconds):
        original_advance(epoch, seconds)
        if not state["active"]:
            return
        if state["crossed"] is None and np.linalg.norm(robot.velocity[:2]) > .01:
            distance = .031 if fault == "outside" else .029722687
            angle = .051 if fault == "not_facing" else .01130957
            robot.frame = replace(robot.frame,
                base_pose=np.r_[goal + [distance, 0.], yaw + heading_sign * angle])
            state["crossed"] = len(robot.commands)
            state["crossed_at"] = robot.time
            if fault == "current_collision":
                nav = robot.frame.navigation
                x, y = np.rint(world_to_px(nav.T_map_px, robot.frame.base_pose[:2])).astype(int)[0]
                occupied = nav.occupied.copy()
                occupied[y, x] = True
                robot.frame = replace(robot.frame, navigation=replace(nav, occupied=occupied))
            elif fault == "deadline":
                live = state["live"]
                live.deadline._end = robot.time
        elif state["crossed"] is not None and state["zero"] is None and not np.any(robot.velocity):
            state["zero"] = len(robot.commands) - 1
            if fault == "rebound":
                robot.frame = replace(robot.frame, base_pose=np.r_[goal + [.2, 0.], yaw + .14])
            elif fault == "after_stop_deadline":
                state["live"].deadline._end = robot.time
            elif fault == "after_stop_collision":
                nav = robot.frame.navigation
                x, y = np.rint(world_to_px(nav.T_map_px, robot.frame.base_pose[:2])).astype(int)[0]
                occupied = nav.occupied.copy()
                occupied[y, x] = True
                robot.frame = replace(robot.frame, navigation=replace(nav, occupied=occupied))

    def check(bridge, vx, vy, wz):
        if state["crossed"] is not None and (vx or vy or wz) and fault != "rebound":
            state["refusals"] += 1
            raise navigation.MapObstacleRefused("next prospective stance command refused",
                                                 details={"witness": "unsent_motion"})
        return original_check(bridge, vx, vy, wz)

    monkeypatch.setattr(navigation, "_stance_to", stance)
    monkeypatch.setattr(navigation._Bridge, "_motion_check", check)
    robot.advance = advance
    if fault == "cancel":
        ctx.cancelled = lambda: state["crossed"] is not None
    elif fault == "after_stop_cancel":
        ctx.cancelled = lambda: state["zero"] is not None
    result = runner.run("navigate", {"goal_xy_m": goal.tolist(), "yaw": yaw, "snap": False}, robot.frame)
    assert state["crossed"] is not None
    assert sum(e.get("kind") == "drive.arrival_correction" for e in events) == 1
    assert sum(e.get("type") == "skill_finished" for e in events) == 1
    assert np.allclose(robot.velocity, 0.)
    return result, robot, ctx, state


def test_crossing_latches_stop_and_settled_rebound_is_not_success(monkeypatch):
    result, robot, _ctx, state = crossing_run(monkeypatch, fault="rebound")
    assert state["zero"] is not None
    assert not any(any(c) for c in robot.commands[state["zero"]:])
    assert result.code == "ARRIVAL_NOT_VERIFIED", result
    assert result.details["verification_phase"] == "settled_pose"

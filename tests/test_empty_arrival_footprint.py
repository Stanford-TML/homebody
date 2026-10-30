"""Empty waypoint completion retains planning room through the actual Runner."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_real_navigation import context, evidence, marked
from test_skills import FakeBackend, frame

from homebody.primitives.observations import ARM, NavigationMap
from homebody.skills.navigate_real import task as navigation


@pytest.mark.parametrize("fault", ["planning", "clear", "stale", "epoch", "cancel",
                                  "body_stale", "body_epoch", "geometry", "deadline"])
def test_settled_empty_arrival_requires_current_planning_footprint(monkeypatch, fault):
    initial = frame()
    initial = replace(initial, navigation=NavigationMap(
        np.zeros((250, 250), bool), np.diag([.01, .01, 1.]), .01))
    robot = FakeBackend(initial)
    ctx, runner, events = context(robot)
    # Keep this terminal fault fixture on the no-run-in facing path.
    ctx.settings = replace(ctx.settings, motion=replace(ctx.settings.motion, run_in_distance=.61))
    settled = []
    if fault == "geometry":
        ctx.arms["right"] = SimpleNamespace(collision=SimpleNamespace(
            boxes=lambda arm, q: [] if not q[0] else [(np.array([q[0], 0., 0.]), np.eye(3), np.full(3, .005))],
            hand_boxes=lambda *args: []))

    def emit(event):
        events.append(event)
        if event.get("kind") == "drive.facing.final_turn_settled" and not settled:
            # Safe accepted goal [1.6,1], within-tolerance settled pose [1.56,1].
            # The new occupied cell leaves .305m current clearance and .345m
            # goal clearance: execution .288m is clear but planning .318m is not.
            nav = robot.frame.navigation
            if fault in ("planning", "geometry"):
                nav = marked(nav, np.array([[1.20 if fault == "geometry" else 1.25, 1., 0.]]))
            robot.frame = replace(robot.frame, base_pose=[1.56, 1., .5], navigation=nav)
            settled.append(len(robot.commands))

    ctx.emit = emit
    ctx.cancelled = lambda: bool(settled) and fault == "cancel"
    ctx.navigation_evidence = lambda: evidence(robot, **(
        {"pose_time": robot.time - 1.} if settled and fault == "stale" else
        {"epoch": 2} if settled and fault == "epoch" else {}))
    sample = navigation._Bridge.sample
    terminal_reads = []

    def terminal_sample(bridge, force=False):
        result = sample(bridge, force=force)
        if force and settled:
            terminal_reads.append(result)
        if force and settled and fault in ("body_stale", "body_epoch", "geometry"):
            measured = robot.frame
            if fault == "geometry":
                joints = measured.joints.copy()
                joints[ARM["right"].start] = .33
                torso = measured.world_torso.copy()
                torso[:2, 3] = measured.base_pose[:2]
                measured = replace(measured, joints=joints, world_torso=torso)
            else:
                measured = replace(measured, **({"time": measured.time - 1.} if fault == "body_stale"
                                                else {"epoch": measured.epoch + 1}))
            # Keep acquisition ID fixed: current measured geometry must win over
            # prepared/cached radius. The Context itself validates body age/epoch.
            monkeypatch.setattr(robot, "measure", lambda: measured)
        return result

    monkeypatch.setattr(navigation._Bridge, "sample", terminal_sample)
    grid = navigation.gait.occupancy_grid

    def checked_grid(*args, **kwargs):
        result = grid(*args, **kwargs)
        if terminal_reads and fault == "deadline":
            robot.frame = replace(robot.frame, time=robot.time + ctx.settings.motion.drive_timeout)
        return result

    monkeypatch.setattr(navigation.gait, "occupancy_grid", checked_grid)
    result = runner.run("navigate", {"x": 1.6, "y": 1., "yaw": .5, "snap": False}, robot.frame)
    assert settled
    assert sum(e["type"] == "skill_finished" for e in events) == 1
    assert not any(any(c) for c in robot.commands[settled[0]:])
    assert not any(e.get("kind") == "drive.arrival_correction" for e in events)
    assert np.allclose(robot.velocity, 0.)
    if fault in ("planning", "geometry"):
        assert result.code == "ARRIVAL_NOT_VERIFIED", result
        assert result.details["verification_phase"] == "settled_footprint"
        check = result.details["footprint"]
        assert not check["current"]["safe"] and check["goal"]["safe"]
        assert check["execution"]["safe"]
        assert result.details["carried_hands"] == []
        assert result.details["distance_m"] < ctx.settings.motion.arrival_distance
        if fault == "geometry":
            assert result.details["planning_radius_m"] > .36
    elif fault == "clear":
        assert result.ok, result
    else:
        assert result.code == {"stale": "STALE_OBSERVATION", "epoch": "STALE_OBSERVATION",
                               "body_stale": "STALE_OBSERVATION", "body_epoch": "STALE_OBSERVATION",
                               "deadline": "TIMEOUT",
                               "cancel": "CANCELLED"}[fault], result

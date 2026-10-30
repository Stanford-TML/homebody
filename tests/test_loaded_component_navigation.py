"""Loaded transport routes the base while checking the measured parts at each command."""
from dataclasses import replace
from xml.etree.ElementTree import fromstring

import numpy as np
import pytest
from test_real_navigation import evidence, loaded_navigation
from test_volume_columns import surface

from homebody.primitives.navigation.errors import MapObstacleRefused
from homebody.skills.contract import SkillFailure
from homebody.skills.navigate_real import task as navigation


def component_scene(observation, *, finger=False):
    robot, ctx, runner, events = loaded_navigation(compactable=True)
    ctx.carried["right"]["payload_half"] = np.full(3, .05)
    ctx.carried["right"]["palm_to_payload"][0, 3] = .5
    if finger:
        arm = ctx.arms["right"]
        arm.collision.hand_boxes = lambda *_: [(np.array([.5, .20, 1.]), np.eye(3), np.full(3, .03))]
        arm.collision.finger_joints = (fromstring('<joint><child link="right_test_finger"/></joint>'),)
        arm.collision.bounds = {"right_test_finger": object()}
    occupied = robot.frame.navigation.occupied.copy()
    occupied[:, 21] = True  # Base at x=1.5 is clear; extended parts approach x=2.1.
    robot.frame = replace(robot.frame, navigation=replace(robot.frame.navigation, occupied=occupied))
    if observation == "missing":
        points = np.empty((0, 3))
    else:
        points = surface((2.01, 2.24), (2.8, 3.4), .4)
        if observation == "blocked":
            points = np.vstack((points, *(surface((2.09, 2.24), (2.8, 3.4), z)
                                         for z in np.arange(.90, 1.11, .01))))
    ctx.navigation_evidence = lambda: evidence(robot, obstacles=points)
    return robot, ctx, runner, events


def test_extended_load_keeps_original_posture_and_base_planning_circle():
    robot, ctx, runner, events = loaded_navigation()
    joints = robot.frame.joints.copy()
    attachment = ctx.carried["right"]["palm_to_payload"].copy()
    arms = []
    robot.arm_target = lambda *args: arms.append(args)
    result = runner.run("navigate", {"goal_xy_m": [2.1, 3.], "snap": False}, robot.frame)
    assert result.ok, result
    base = max(.24, ctx.settings.motion.robot_radius) + ctx.settings.collision.cloud_margin
    assert result.details["execution_radius_m"] == base
    assert navigation.sweep_radius(robot.frame, ctx.arms, ctx.carried, base) > base + .20
    np.testing.assert_array_equal(robot.frame.joints, joints)
    np.testing.assert_array_equal(ctx.carried["right"]["palm_to_payload"], attachment)
    assert arms == []
    assert sum(e.get("type") == "skill_started" for e in events) == 1
    assert sum(e.get("type") == "skill_finished" for e in events) == 1


@pytest.mark.parametrize("observation", ["missing", "blocked"])
@pytest.mark.parametrize("finger", [False, True])
@pytest.mark.parametrize("docking", [False, True])
def test_component_command_refuses_uncleared_parts_outside_base_circle(observation, finger, docking):
    robot, ctx, _runner, _events = component_scene(observation, finger=finger)
    ctx.settings = replace(ctx.settings, motion=replace(ctx.settings.motion, robot_radius=.20),
                           collision=replace(ctx.settings.collision, cloud_margin=.005))
    bridge = navigation._Bridge(ctx.planning(), robot.frame, ctx)
    bridge.docking = docking
    with pytest.raises(MapObstacleRefused, match="Observed columns") as failure:
        bridge._motion_check(.20, 0., 0.)
    assert failure.value.details["uncleared_points"] > 0
    assert failure.value.details["witness_part"] in ("right payload", "right right_test_finger")
    assert failure.value.details["evidence"] == ("unobserved" if observation == "missing" else "occupied")
    if observation == "missing":
        assert not failure.value.details["observed_beneath"]
    assert robot.commands == []


def calibrated_component_scene(monkeypatch, observation="missing"):
    robot, ctx, _runner, _events = component_scene(observation)
    camera = np.eye(4)
    camera[:3, :3] = [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]
    camera[:3, 3] = [1., 3., 1.]
    meta = {"registered_pose": True, "navigation_generation": (1, 1),
            "map_frame": "test-room", "map_sha256": "test-map-v1",
            "calibration_hash": "test-calibration", "camera_identity": "test-camera",
            "depth_time": robot.time, "depth_rays_from_world_camera": True}
    robot.frame = replace(robot.frame, world_camera=camera,
                          depth=np.full_like(robot.frame.depth, 3.), sensor_metadata=meta)
    observed, original = [], ctx.observe

    def observe(epoch):
        result = original(epoch)
        observed.append(result)
        return result

    monkeypatch.setattr(ctx, "observe", observe)
    return robot, ctx, observed


@pytest.mark.parametrize("change", ["occluded", "invalid", "masked", "outside", "missing_calibration",
                                   "wrong_frame", "wrong_map", "wrong_generation", "wrong_calibration",
                                   "stale", "future", "frame_stale", "frame_future", "epoch"])
def test_unknown_component_sweep_keeps_refusal_without_usable_current_depth(monkeypatch, change):
    robot, ctx, observed = calibrated_component_scene(monkeypatch)
    bridge = navigation._Bridge(ctx.planning(), robot.frame, ctx)
    bridge.columns(bridge.sample(), robot.frame)
    captured = robot.frame
    meta = dict(captured.sensor_metadata)
    if change in ("occluded", "invalid"):
        captured = replace(captured, depth=np.full_like(captured.depth, .5 if change == "occluded" else np.nan))
    elif change == "masked":
        captured = replace(captured, labels=np.full_like(captured.labels, -1))
    elif change == "outside":
        camera = captured.world_camera.copy()
        camera[1, 3] += 5.
        captured = replace(captured, world_camera=camera)
    elif change == "epoch":
        captured = replace(captured, epoch=captured.epoch + 1)
    else:
        if change == "missing_calibration":
            meta.pop("calibration_hash")
        elif change == "wrong_frame":
            meta["map_frame"] = "another-room"
        elif change == "wrong_map":
            meta["map_sha256"] = "another-map"
        elif change == "wrong_generation":
            meta["navigation_generation"] = (1, 2)
        elif change == "wrong_calibration":
            meta["calibration_hash"] = "another-calibration"
        elif change == "stale":
            meta["depth_time"] = captured.time - ctx.settings.motion.observation_age - .01
        elif change == "future":
            meta["depth_time"] = captured.time + .1
        elif change == "frame_stale":
            captured = replace(captured, time=captured.time - ctx.settings.motion.observation_age - .01)
        elif change == "frame_future":
            captured = replace(captured, time=captured.time + .1)
        captured = replace(captured, sensor_metadata=meta)
    bridge._column_frames[-1] = captured
    with pytest.raises(MapObstacleRefused, match="Observed columns") as failure:
        bridge._motion_check(.20, 0., 0.)
    assert not failure.value.details["observed_at_height"]
    assert failure.value.details["uncleared_points"] > 0
    assert len(observed) == 1 and robot.commands == []


def test_loaded_component_refusal_allows_one_retreat_then_terminates_held(monkeypatch):
    robot, ctx, runner, events = component_scene("missing")
    initial_pose = robot.frame.base_pose.copy()
    poses = [initial_pose]
    advance = robot.advance

    def record_pose(epoch, dt):
        advance(epoch, dt)
        poses.append(robot.frame.base_pose.copy())

    monkeypatch.setattr(robot, "advance", record_pose)
    result = runner.run("navigate", {"goal_xy_m": [1.7, 3.], "snap": False}, robot.frame)
    assert result.code == "ACTION_REJECTED", result
    assert result.details["evidence"] == "unobserved"
    assert result.details["witness_part"] == "right payload"
    assert 2 < result.details["attempts"] <= 4
    assert "right" in ctx.carried and robot.frame.hands["right"].closure > .15
    assert ctx.carried["right"]["state"] == "verified"
    moving = [command for command in robot.commands if any(command)]
    assert moving and moving[0][0] < 0.
    assert all(vy == wz == 0. for _, vy, wz in moving)
    poses = np.asarray(poses)
    assert poses[:, 0].min() < initial_pose[0] - .5
    assert poses[:, 0].max() <= initial_pose[0] + 1e-9
    recoveries = [e for e in events if e.get("kind") == "recovery.complete"]
    assert len(recoveries) == 1
    assert recoveries[0]["details"]["pose"][0] < initial_pose[0]
    retries = [e["details"]["kind"] for e in events if e.get("kind") == "drive.retry"]
    assert retries.count("after_checked_retreat") == 1
    assert retries[-1] == "same_goal_replan"
    assert np.allclose(robot.velocity, 0.)
    assert sum(e.get("type") == "skill_started" for e in events) == 1
    assert sum(e.get("type") == "skill_finished" for e in events) == 1


def dock_floor_scene(column):
    robot, ctx, _runner, events = loaded_navigation(compactable=True)
    ctx.settings = replace(ctx.settings, motion=replace(ctx.settings.motion, robot_radius=.20),
                           collision=replace(ctx.settings.collision, cloud_margin=.005))
    occupied = np.zeros((400, 400), dtype=bool)
    occupied[:, column] = True
    robot.frame = replace(robot.frame, navigation=replace(
        robot.frame.navigation, occupied=occupied, resolution=.01,
        T_map_px=np.diag([.01, .01, 1.])))
    return robot, ctx, events


@pytest.mark.parametrize("column,ahead", [(185, .2), (180, .6)])
def test_dock_floor_query_rejects_blocked_endpoint_or_crossing(column, ahead):
    robot, ctx, _events = dock_floor_scene(column)
    bridge = navigation._Bridge(ctx.planning(), robot.frame, ctx)
    checker = navigation.gait.occupancy_grid(bridge.artifact(), .205, exact_footprint=True)[0]
    assert checker.is_safe(*robot.frame.base_pose[:2])
    if ahead == .6:
        assert checker.is_safe(2.1, 3.)  # Both endpoints clear; the middle crosses the wall.
    else:
        assert not checker.is_safe(1.7, 3.)
    assert not navigation.dock_floor_clear(ctx, robot.frame, ahead, 0.)
    assert robot.commands == []


@pytest.mark.parametrize("change", [{"registered": False}, {"pose_time": -1.},
                                   {"obstacle_time": -1.}, {"epoch": 2}])
def test_dock_floor_query_rejects_stale_or_wrong_epoch_navigation_evidence(change):
    robot, ctx, _events = dock_floor_scene(192)
    ctx.navigation_evidence = lambda: evidence(robot, **change)
    with pytest.raises(SkillFailure) as failure:
        navigation.dock_floor_clear(ctx, robot.frame, .20, 0.)
    assert failure.value.code == "STALE_OBSERVATION"
    assert robot.commands == []

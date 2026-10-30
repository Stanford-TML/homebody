"""Withdrawal closes a fixed world destination without imposing a docking heading."""
from dataclasses import replace
from types import SimpleNamespace
from xml.etree.ElementTree import fromstring

import numpy as np
import pytest
from test_real_navigation import loaded_navigation

from homebody.skills.contract import SkillFailure
from homebody.skills.navigate_real import task as navigation


def withdrawal_scene(monkeypatch, drift=.08):
    start = np.array([1.5, 1.5, .4])
    backward = -np.array([np.cos(start[2]), np.sin(start[2])])
    robot, ctx, _, events = loaded_navigation(compactable=True)
    ctx.carried.clear()
    collision = ctx.arms["right"].collision
    collision.hand_boxes = lambda *_: [(np.array([.3, 0., 1.]), np.eye(3), np.full(3, .03))]
    collision.finger_joints = (fromstring('<joint><child link="right_test_finger"/></joint>'),)
    collision.bounds = {"right_test_finger": object()}
    torso = robot.frame.world_torso.copy()
    torso[:2, 3] = start[:2]
    robot.frame = replace(robot.frame, base_pose=start, world_torso=torso,
                          hands={side: replace(hand, closure=0.) for side, hand in robot.frame.hands.items()})
    state = SimpleNamespace(shifted_at=None, final_blocked=False, targets=[], checked=[])
    advance = robot.advance

    def step(epoch, seconds):
        advance(epoch, seconds)
        if state.shifted_at is None and np.linalg.norm(robot.velocity[:2]) > .01:
            pose = robot.frame.base_pose.copy()
            pose[2] += drift
            robot.frame = replace(robot.frame, base_pose=pose)
            state.shifted_at = len(robot.commands)

    robot.advance = step
    grid, walk, check = navigation.gait.occupancy_grid, navigation._dock_walk, navigation._Bridge._motion_check
    radius = navigation.footprint(robot.frame, ctx.planning())[0]

    def footprint_grid(artifact, requested_radius, **kwargs):
        if "obstacles" in kwargs:
            return grid(artifact, requested_radius, **kwargs)
        required = .12 if requested_radius <= radius + 1e-9 else .30
        return SimpleNamespace(is_safe=lambda x, y: not state.final_blocked and
            np.dot(np.array([x, y]) - start[:2], backward) >= required - 1e-9), None

    def record_walk(bridge, live, target, yaw, motion):
        state.targets.append((target.copy(), yaw))
        return walk(bridge, live, target, yaw, motion)

    def record_check(bridge, vx, vy, wz):
        state.checked.append((vx, vy, wz))
        return check(bridge, vx, vy, wz)

    monkeypatch.setattr(navigation.gait, "occupancy_grid", footprint_grid)
    monkeypatch.setattr(navigation, "_dock_walk", record_walk)
    monkeypatch.setattr(navigation._Bridge, "_motion_check", record_check)
    return robot, ctx, events, state, start, backward


@pytest.mark.parametrize("drift", [-.08, .08])
def test_withdrawal_yaw_drift_preserves_world_target_without_turning(monkeypatch, drift):
    robot, ctx, events, state, start, backward = withdrawal_scene(monkeypatch, drift)
    pose, refused = navigation.back_out(ctx, robot.epoch)
    assert refused is None, refused
    assert state.shifted_at is not None
    assert len(state.targets) == 1
    target, _ = state.targets[0]
    np.testing.assert_allclose(target, start[:2] + .30 * backward)
    assert np.linalg.norm(pose[:2] - target) <= ctx.settings.motion.stop_distance
    after_drift = robot.commands[state.shifted_at:]
    assert any(np.linalg.norm(command[:2]) > .01 for command in after_drift)
    assert all(command[2] == 0. for command in after_drift), "Withdrawal must not impose the initial yaw"
    assert pose[2] == pytest.approx(start[2] + drift)
    assert state.checked and all(command[2] == 0. for command in state.checked)
    assert sum(event.get("kind") == "dock.departure_checked" for event in events) == 1
    np.testing.assert_array_equal(robot.velocity, np.zeros(3))


@pytest.mark.parametrize("drift", [-.08, .08])
def test_dock_in_corrects_heading_drift(monkeypatch, drift):
    robot, ctx, _, state, start, backward = withdrawal_scene(monkeypatch, drift)
    pose, refused = navigation.dock(ctx, robot.epoch, .30)
    assert refused is None, refused
    assert len(state.targets) == 1
    target, yaw = state.targets[0]
    np.testing.assert_allclose(target, start[:2] - .30 * backward)
    assert yaw == pytest.approx(start[2])
    assert any(command[2] * drift < 0. for command in robot.commands[state.shifted_at:])
    assert abs(pose[2] - start[2]) <= ctx.settings.motion.arrival_angle / 2
    np.testing.assert_array_equal(robot.velocity, np.zeros(3))


@pytest.mark.parametrize("fault,code", [("cancel", "CANCELLED"), ("stale", "STALE_OBSERVATION")])
def test_withdrawal_stops_when_current_authority_or_localization_changes(monkeypatch, fault, code):
    robot, ctx, _, state, _, _ = withdrawal_scene(monkeypatch)
    revoked_at = []
    if fault == "cancel":
        ctx.cancelled = lambda: state.shifted_at is not None
    else:
        read = ctx.navigation_evidence
        def current_evidence():
            if state.shifted_at is not None:
                revoked_at.append(len(robot.commands))
            return replace(read(), registered=state.shifted_at is None)

        ctx.navigation_evidence = current_evidence
    with pytest.raises(SkillFailure) as failure:
        navigation.back_out(ctx, robot.epoch)
    assert failure.value.code == code
    assert state.shifted_at is not None
    stopped_at = state.shifted_at if fault == "cancel" else revoked_at[0]
    assert not any(any(command) for command in robot.commands[stopped_at:])
    np.testing.assert_array_equal(robot.velocity, np.zeros(3))

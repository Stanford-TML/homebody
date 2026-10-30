"""Ordinary approach refusal stops at once without preempting a later checked action."""
from dataclasses import replace

import numpy as np
import pytest
from test_loaded_component_navigation import component_scene
from test_real_navigation import evidence, loaded_navigation
from test_skills import frame

from homebody.primitives.navigation.errors import MapObstacleRefused
from homebody.skills.navigate_real import task


def refused_empty_approach():
    robot, ctx, runner, events = loaded_navigation(compactable=True)
    carried, hands = dict(ctx.carried), robot.frame.hands
    robot.frame = replace(robot.frame, hands=frame().hands)
    ctx.carried.clear()
    advance = robot.advance
    correction, injected = [], []

    def emit(event):
        events.append(event)
        if event.get('kind') == 'drive.arrival_correction':
            correction.append(event)

    def drift(epoch, seconds):
        advance(epoch, seconds)
        if correction and not injected and np.any(robot.velocity):
            # Measured motion crosses the wall despite the admitted command.
            nav = robot.frame.navigation
            occupied = nav.occupied.copy()
            occupied[:, 18] = True
            torso = robot.frame.world_torso.copy()
            torso[:2, 3] = [1.5, 3.]
            robot.frame = replace(robot.frame, base_pose=np.array([1.5, 3., 0.]),
                                  world_torso=torso, navigation=replace(nav, occupied=occupied))
            injected.append(len(robot.commands))

    ctx.emit, robot.advance = emit, drift
    first = runner.run('navigate', {'goal_xy_m': [2.7, 3.], 'yaw': 0., 'snap': False}, robot.frame)
    assert injected and first.code == 'ACTION_REJECTED', first
    assert first.details['arrived'] is False
    assert first.details['recovery_check']['failed_stage'] == 'sweep'
    assert not first.details['recovery_check']['sweep']['safe']
    assert not any(any(c) for c in robot.commands[injected[0]:])
    assert np.allclose(robot.velocity, 0.)
    # A subsequent grasp changes ownership; fresh observations remain mandatory.
    ctx.carried.update(carried)
    robot.frame = replace(robot.frame, hands=hands)
    robot.advance = advance
    return robot, ctx, runner, events


@pytest.mark.parametrize('fault', ['blocked', 'missing', 'stale', 'cancel'])
def test_later_loaded_action_still_refuses_current_evidence_or_authority(monkeypatch, fault):
    robot, ctx, runner, events = refused_empty_approach()
    assert not ctx.navigation_state._escape_attempts
    if fault in ('blocked', 'missing'):
        # A fresh acquisition sees an occupied column ahead of an extended finger.
        template, current, _, _ = component_scene(fault, finger=True)
        robot.frame = replace(robot.frame, navigation=template.frame.navigation)
        ctx.arms, ctx.carried = current.arms, current.carried
        points = current.navigation_evidence().obstacles
        ctx.navigation_evidence = lambda: evidence(robot, obstacles=points)
    elif fault == 'stale':
        ctx.navigation_evidence = lambda: evidence(robot, pose_time=-10.)
    else:
        ctx.cancelled = lambda: True
    start = len(robot.commands)
    refused_at, admitted = [], []
    original = task._Bridge._motion_check
    base_velocity = robot.base_velocity

    def checked(bridge, *command):
        admitted.clear()
        try:
            result = original(bridge, *command)
        except MapObstacleRefused:
            refused_at.append(len(robot.commands))
            raise
        admitted.append(command)
        return result

    def command(epoch, vx, vy, wz):
        if any((vx, vy, wz)):
            assert admitted, 'A nonzero command must pass the current component guard'
            np.testing.assert_allclose(admitted[-1], [vx, vy, wz])
        base_velocity(epoch, vx, vy, wz)

    monkeypatch.setattr(task._Bridge, '_motion_check', checked)
    monkeypatch.setattr(robot, 'base_velocity', command)
    event_start = len(events)
    second = runner.run('navigate', {'goal_xy_m': [1.7, 3.], 'snap': False}, robot.frame)
    assert not second.ok, second
    if fault in ('blocked', 'missing'):
        assert second.code == 'ACTION_REJECTED'
        assert second.details['evidence'] == ('occupied' if fault == 'blocked' else 'unobserved')
    else:
        assert second.code == ('STALE_OBSERVATION' if fault == 'stale' else 'CANCELLED')
    if fault in ('blocked', 'missing'):
        assert refused_at
        # Prospective refusal can select another checked recovery direction.
        assert not any(any(c) for c in robot.commands[refused_at[-1]:])
    else:
        assert not any(any(c) for c in robot.commands[start:])
    assert sum(e.get('type') == 'skill_finished' for e in events) == 2
    assert not any(e.get('kind') == 'drive.arrival_confirmed' for e in events[event_start:])
    assert np.allclose(robot.velocity, 0.)

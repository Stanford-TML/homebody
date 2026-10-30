"""Loaded facing holds its stance through the actual checked task controller."""
from dataclasses import replace

import numpy as np
import pytest
from test_real_navigation import loaded_navigation

from homebody.helpers.navigation import pixel
from homebody.skills.navigate_real import task as navigation


def loaded_drift():
    robot, ctx, runner, events = loaded_navigation()
    advance = robot.advance
    robot.facing = False
    robot.turning_commands = []
    robot.fault = None
    robot.fault_at = None

    def step(epoch, seconds):
        advance(epoch, seconds)
        if not robot.facing or abs(robot.velocity[2]) <= .01:
            return
        robot.turning_commands.append(robot.velocity.copy())
        pose = robot.frame.base_pose.copy()
        pose[1] += .04 * seconds
        robot.frame = replace(robot.frame, base_pose=pose)
        if robot.fault is not None and robot.fault_at is None:
            robot.fault_at = len(robot.commands)
            robot.fault(robot, ctx)

    def emit(event):
        events.append(event)
        if (event.get('kind') == 'drive.arrival_correction' or
                (event.get('kind') == 'base.action' and
                 event.get('details', {}).get('tag') == 'drive.facing.final_turn')):
            robot.facing = True

    robot.advance, ctx.emit = step, emit
    return robot, ctx, runner, events


def run(runner, robot):
    return runner.run('navigate', {'goal_xy_m': [2.1, 3.], 'yaw': 1.2, 'snap': False}, robot.frame)


def test_loaded_facing_rejects_yaw_only_drift_and_uses_checked_xy_feedback(monkeypatch):
    robot, ctx, runner, events = loaded_drift()
    monkeypatch.setattr(navigation.gait, 'final_turns', lambda *a, **k:
                        pytest.fail('Loaded facing must not first drift through a yaw-only servo'))
    result = run(runner, robot)
    assert result.ok, result
    assert robot.turning_commands and any(np.linalg.norm(c[:2]) > .005 for c in robot.turning_commands)
    assert np.linalg.norm(robot.frame.base_pose[:2] - [2.1, 3.]) <= ctx.settings.motion.arrival_distance
    assert abs(robot.frame.base_pose[2] - 1.2) <= ctx.settings.motion.arrival_angle
    assert sum(e.get('kind') == 'drive.arrival_correction' for e in events) == 1
    assert sum(e['type'] == 'skill_started' for e in events) == 1
    assert sum(e['type'] == 'skill_finished' for e in events) == 1
    assert sum(e.get('kind') == 'drive.request' for e in events) == 1
    assert np.allclose(robot.velocity, 0.)


@pytest.mark.parametrize('fault,code', [('swept_obstacle', 'ACTION_REJECTED'),
                                       ('off_centre', 'ACTION_REJECTED'),
                                       ('goal', 'ACTION_REJECTED'), ('retention', 'HOLD_UNCERTAIN')])
def test_loaded_facing_measured_failure_stops_without_retry_or_arrival(fault, code):
    robot, ctx, runner, events = loaded_drift()

    def fail(robot, ctx):
        if fault != 'retention':
            # Measured sweep: at the pose or 0.1 m against +y drift (leaving would be egress).
            # The goal lies beyond the actual sweep and must fail goal admission separately.
            xy = {'swept_obstacle': robot.frame.base_pose[:2],
                  'off_centre': robot.frame.base_pose[:2] - [0., .1],
                  'goal': np.array([2.1, 3.])}[fault]
            occupied = robot.frame.navigation.occupied.copy()
            occupied[pixel(robot.frame.navigation, xy)] = True
            robot.frame = replace(robot.frame, navigation=replace(robot.frame.navigation, occupied=occupied))
        else:
            # Remove both supported current-hold observations, preserving the ownership record.
            hands = dict(robot.frame.hands)
            hands['right'] = replace(hands['right'], finger_effort_fraction=np.zeros(3))
            robot.frame = replace(robot.frame, hands=hands, labels=np.zeros_like(robot.frame.labels))

    robot.fault = fail
    result = run(runner, robot)
    assert result.code == code, result
    assert robot.fault_at is not None
    assert not any(any(c) for c in robot.commands[robot.fault_at:])
    assert sum(e.get('kind') == 'drive.arrival_correction' for e in events) == 1
    assert sum(e['type'] == 'skill_finished' for e in events) == 1
    assert 'right' in ctx.carried and np.allclose(robot.velocity, 0.)
    if fault in ('swept_obstacle', 'off_centre'):
        assert not ctx.navigation_state._escape_attempts
        check = result.details['recovery_check']
        assert check['failed_stage'] == 'sweep'
        assert check['allow_egress'] is False and check['sweep']['safe'] is False
        assert result.details['arrived'] is False
        refused = [e['details'] for e in events if e.get('kind') == 'drive.arrival_correction_refused']
        assert len(refused) == 1 and refused[0]['recovery_check'] == check
        assert not any(e.get('kind') == 'recovery.memory' for e in events)
    elif fault == 'goal':
        assert not ctx.navigation_state._escape_attempts
        assert result.details['start']['safe']
        assert not result.details['goal']['safe']
        assert result.details['segment'] is None
        assert result.details['arrived'] is False


def test_loaded_blocked_facing_start_is_not_egress(monkeypatch):
    robot, _ctx, runner, events = loaded_drift()
    original = navigation._correct_settled_arrival
    blocked_at = []

    def correct(plan, ctx, bridge, live, navigator, initial):
        occupied = robot.frame.navigation.occupied.copy()
        occupied[pixel(robot.frame.navigation, robot.frame.base_pose[:2])] = True
        robot.frame = replace(robot.frame, navigation=replace(robot.frame.navigation, occupied=occupied))
        blocked_at.append(len(robot.commands))
        return original(plan, ctx, bridge, live, navigator, initial)

    monkeypatch.setattr(navigation, '_correct_settled_arrival', correct)
    result = run(runner, robot)
    assert result.code == 'ARRIVAL_NOT_VERIFIED', result
    assert len(blocked_at) == 1
    assert not any(any(c) for c in robot.commands[blocked_at[0]:])
    skipped = result.details['correction']['skipped']
    assert skipped['reason'] == 'strict_correction_unavailable' and skipped['start']['safe'] is False
    assert not any(e.get('kind') == 'recovery.request' for e in events)

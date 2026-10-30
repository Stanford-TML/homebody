"""Empty-hand task approaches align at the prepared run-in before their final close."""
from dataclasses import replace

import numpy as np
import pytest
from test_real_navigation import context
from test_skills import FakeBackend

from homebody.helpers.navigation import pixel
from homebody.skills.navigate_real import task

ARGS = {'goal_xy_m': [2.5, 2.], 'yaw': 1.2, 'snap': False}


def setup(monkeypatch, *, forbid_late_turn=True):
    robot = FakeBackend()
    ctx, runner, events = context(robot)
    corrected, runtimes = [], []
    runtime = task._navigation_runtime

    def record_runtime(*args, **kwargs):
        value = runtime(*args, **kwargs)
        runtimes.append(value)
        return value

    monkeypatch.setattr(task, '_navigation_runtime', record_runtime)
    if forbid_late_turn:
        monkeypatch.setattr(task.gait, 'final_turns', lambda *a, **kw:
                            pytest.fail('A prepared run-in must not use a separate late yaw phase'))
    emit = ctx.emit

    def record(event):
        emit(event)
        if event.get('kind') == 'drive.arrival_correction':
            corrected.append((robot.frame.base_pose.copy(), len(robot.commands)))

    ctx.emit = record
    return robot, ctx, runner, events, corrected, runtimes


def test_empty_runner_uses_run_in_then_aligns_before_close(monkeypatch):
    robot, ctx, runner, events, corrected, _ = setup(monkeypatch)
    commands = []
    base_velocity = robot.base_velocity

    def record(epoch, vx, vy, wz):
        if corrected:
            commands.append((robot.frame.base_pose.copy(), (vx, vy, wz)))
        return base_velocity(epoch, vx, vy, wz)

    robot.base_velocity = record
    attempt = runner.prepare('navigate', ARGS, robot.frame)
    assert not attempt.finished and attempt.plan.run_in is not None
    goal = attempt.plan.goal.copy()
    result = runner.execute(attempt)
    assert result.ok, result
    assert len(corrected) == 1
    assert np.linalg.norm(corrected[0][0][:2] - goal) > ctx.settings.motion.arrival_distance
    forward = [(pose, command) for pose, command in commands if command[0] > .05]
    assert forward and all(abs(pose[2] - ARGS['yaw']) <= ctx.settings.motion.arrival_angle for pose, _ in forward)
    assert np.linalg.norm(robot.frame.base_pose[:2] - goal) <= ctx.settings.motion.arrival_distance
    assert abs(robot.frame.base_pose[2] - ARGS['yaw']) <= ctx.settings.motion.arrival_angle
    assert sum(e.get('kind') == 'drive.request' for e in events) == 1
    assert sum(e.get('type') == 'skill_finished' for e in events) == 1
    assert np.allclose(robot.velocity, 0.)


@pytest.mark.parametrize('obstacle', ['connector', 'goal'])
def test_changed_connector_or_goal_refuses_before_final_close(monkeypatch, obstacle):
    robot, _ctx, runner, _events, _corrected, _ = setup(monkeypatch)
    original = task._correct_settled_arrival
    injected = []

    def block(plan, ctx, bridge, live, navigator, initial):
        point = plan.goal if obstacle == 'goal' else (plan.goal + initial.base_pose[:2]) / 2
        occupied = robot.frame.navigation.occupied.copy()
        occupied[pixel(robot.frame.navigation, point)] = True
        robot.frame = replace(robot.frame, navigation=replace(robot.frame.navigation, occupied=occupied))
        injected.append(len(robot.commands))
        return original(plan, ctx, bridge, live, navigator, initial)

    monkeypatch.setattr(task, '_correct_settled_arrival', block)
    result = runner.run('navigate', ARGS, robot.frame)
    assert injected and result.code == 'ARRIVAL_NOT_VERIFIED', result
    assert not any(any(c) for c in robot.commands[injected[0]:])
    assert 'skipped' in result.details['correction']
    assert not result.details['correction']['skipped']['segment' if obstacle == 'connector' else 'goal']['safe']

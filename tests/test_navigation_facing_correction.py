"""Bounded final-facing correction through the actual REAL Runner/controller."""

from dataclasses import replace

import numpy as np
import pytest
from test_real_navigation import context as navigation_context
from test_skills import FakeBackend


def context(robot):
    ctx, runner, events = navigation_context(robot)
    # Exercise the no-run-in facing fallback, away from the .6 m float boundary.
    ctx.settings = replace(ctx.settings, motion=replace(ctx.settings.motion, run_in_distance=.61))
    return ctx, runner, events


class FacingDrift(FakeBackend):
    def __init__(self, drift=.18):
        super().__init__()
        self.drift = drift
        self.shifted = False

    def advance(self, epoch, seconds):
        super().advance(epoch, seconds)
        if not self.shifted and abs(self.velocity[2]) > .01:
            pose = self.frame.base_pose.copy()
            pose[1] += self.drift
            self.frame = replace(self.frame, base_pose=pose)
            self.shifted = True


def test_facing_drift_is_corrected_without_relaxing_arrival():
    robot = FacingDrift()
    ctx, runner, events = context(robot)
    correction_start = []

    def emit(event):
        events.append(event)
        if event.get("kind") == "drive.arrival_correction":
            correction_start.append((len(robot.commands), robot.frame.base_pose[2]))

    ctx.emit = emit
    result = runner.run("navigate", {"x": 1.6, "y": 1., "yaw": .5}, robot.frame)
    assert result.ok, result
    correction = result.details["correction"]
    assert correction["before_distance_m"] > ctx.settings.motion.arrival_distance
    assert correction["after_distance_m"] <= ctx.settings.motion.arrival_distance
    assert len([e for e in events if e.get("kind") == "drive.arrival_correction"]) == 1
    assert len([e for e in events if e["type"] == "skill_finished"]) == 1
    command_start, heading = correction_start[0]
    assert abs(heading - .5) <= ctx.settings.motion.arrival_angle
    assert any(np.linalg.norm(command[:2]) > 0. and command[2] * (.5 - heading) > 0.
               for command in robot.commands[command_start:])
    assert abs(robot.frame.base_pose[2] - .5) < abs(heading - .5)
    assert np.allclose(robot.velocity, 0)


@pytest.mark.parametrize("fault", ["cancel", "stalled", "obstacle"])
def test_correction_rechecks_authority_progress_and_obstacles(fault):
    robot = FacingDrift()
    ctx, runner, events = context(robot)
    correcting = False

    def emit(event):
        nonlocal correcting
        events.append(event)
        if event.get("kind") != "drive.arrival_correction":
            return
        correcting = True
        if fault == "stalled":
            robot.freeze_pose = True
        elif fault == "obstacle":
            occupied = robot.frame.navigation.occupied.copy()
            occupied[10:13, 13:19] = True
            robot.frame = replace(robot.frame,
                                  navigation=replace(robot.frame.navigation, occupied=occupied))

    ctx.emit = emit
    ctx.cancelled = lambda: correcting and fault == "cancel"
    result = runner.run("navigate", {"x": 1.6, "y": 1., "yaw": .5}, robot.frame)
    assert correcting
    assert not result.ok
    assert result.code == {"cancel": "CANCELLED", "stalled": "ARRIVAL_NOT_VERIFIED",
                           "obstacle": "ACTION_REJECTED"}[fault]
    assert robot.time < 20
    assert np.allclose(robot.velocity, 0)
    assert len([e for e in events if e["type"] == "skill_finished"]) == 1

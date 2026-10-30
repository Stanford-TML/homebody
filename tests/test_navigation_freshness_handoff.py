"""Held replanning must refresh the measurements its controller actually consumes."""
from dataclasses import replace

import pytest
from test_real_navigation import context, evidence
from test_skills import FakeBackend

from homebody.motion.exploration import localizer
from homebody.skills.contract import SkillFailure
from homebody.skills.navigate_real import task


def lagged_bridge(lag=.26, until=float("inf")):
    robot = FakeBackend()
    ctx, _, events = context(robot)
    original = robot.measure
    sample = evidence(robot)
    ctx.navigation_evidence = lambda: replace(sample, epoch=robot.epoch,
        time=robot.time, pose_time=robot.time, obstacle_time=robot.time, map_time=robot.time,
        base_pose=robot.frame.base_pose)
    robot.measure = lambda: replace(original(), time=robot.time-lag) if robot.time < until else original()
    bridge = task._Bridge(ctx.planning(), robot.frame, ctx)
    live = bridge.context()
    return robot, ctx, bridge, live, events


def test_persistent_staleness_reports_packet_age_and_bounded_hold():
    robot, _, bridge, live, _ = lagged_bridge()
    with pytest.raises(SkillFailure) as caught:
        bridge.fresh(live, robot.epoch)
    assert caught.value.code == "STALE_OBSERVATION"
    assert caught.value.details["age_s"] == pytest.approx(.26)
    assert caught.value.details["max_age_s"] == .25
    assert caught.value.details["waited_s"] == pytest.approx(robot.time)
    assert localizer.FRESH_WAIT_S <= robot.time <= localizer.FRESH_WAIT_S + .004
    assert not any(any(command) for command in robot.commands)

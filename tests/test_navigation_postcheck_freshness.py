"""A checked command must still be current and authorized when it is sent."""
from dataclasses import replace

import pytest
from test_real_navigation import context, evidence
from test_skills import FakeBackend, frame

from homebody.primitives.navigation.actions.lower_body import LowerBody
from homebody.skills.contract import SkillFailure
from homebody.skills.navigate_real import task as navigation


class WallClockBackend(FakeBackend):
    """Sensor packets can remain fixed while CPU work consumes real elapsed time."""

    def __init__(self):
        super().__init__(frame(time=10.))
        self.clock = 10.

    @property
    def time(self):
        return self.clock


def checked_bridge():
    robot = WallClockBackend()
    ctx, _, _ = context(robot)
    sample = evidence(robot)
    reads = []

    def navigation_evidence():
        reads.append(sample)
        return sample

    ctx.navigation_evidence = navigation_evidence
    bridge = navigation._Bridge(ctx.planning(), robot.frame, ctx)
    return robot, ctx, bridge, reads


@pytest.mark.parametrize("expired", ["localization", "obstacles", "map", "measured_geometry"])
def test_elapsed_geometry_check_cannot_send_using_expired_proof(monkeypatch, expired):
    robot, ctx, bridge, reads = checked_bridge()
    if expired == "measured_geometry":
        robot.frame = replace(robot.frame,
                              time=robot.time - ctx.settings.motion.observation_age + .01)
    else:
        stamp, limit = {"localization": ("pose_time", .3),
                        "obstacles": ("obstacle_time", .5),
                        "map": ("map_time", 3.)}[expired]
        original_read = ctx.navigation_evidence

        def aging_navigation_evidence():
            return replace(original_read(), **{stamp: 10. - limit + .01})

        bridge.planning = replace(bridge.planning, navigation_evidence=aging_navigation_evidence)
    original_check = bridge._motion_check
    checks = []

    def expensive_check(*command):
        original_check(*command)
        checks.append(bridge._sample)
        robot.clock += .02

    monkeypatch.setattr(bridge, "_motion_check", expensive_check)
    failure = None
    try:
        bridge.step("lower_body", LowerBody(vx=.2, vy=0., wz=0.))
    except SkillFailure as error:
        failure = error
    assert checks, "The actual motion check must complete before the clock advances"
    assert not any(any(command) for command in robot.commands)
    assert failure is not None and failure.code == "STALE_OBSERVATION"
    assert len(reads) == 1, "Do not replace the expired proof with unchecked fresh geometry"


@pytest.mark.parametrize("control,code", [("cancel", "CANCELLED"), ("pause", "PAUSED")])
def test_control_change_during_geometry_check_prevents_nonzero_write(monkeypatch, control, code):
    robot, ctx, bridge, _ = checked_bridge()
    requested = []
    if control == "cancel":
        ctx.cancelled = lambda: bool(requested)
    else:
        ctx.paused = lambda: bool(requested)
    original_check = bridge._motion_check

    def interrupted_check(*command):
        original_check(*command)
        requested.append(True)

    monkeypatch.setattr(bridge, "_motion_check", interrupted_check)
    failure = None
    try:
        bridge.step("lower_body", LowerBody(vx=.2, vy=0., wz=0.))
    except SkillFailure as error:
        failure = error
    assert requested
    assert not any(any(command) for command in robot.commands)
    assert failure is not None and failure.code == code

"""Manipulation stance control holds heading while closing through the REAL Runner."""

import numpy as np
import pytest
from test_navigation_facing_correction import FacingDrift
from test_real_navigation import context

from homebody.skills.navigate_real import task as navigation


@pytest.fixture(autouse=True)
def strict_refinement(monkeypatch):
    """Force the precision close used by pick refinement."""
    original = navigation._stance_to

    def stance(bridge, live, target, yaw, motion, **kwargs):
        return original(bridge, live, target, yaw, motion, arrival=False)

    monkeypatch.setattr(navigation, "_stance_to", stance)


@pytest.mark.parametrize("evidence", ["occupied", "unobserved"])
def test_close_mixed_command_refusal_stops_without_straight_fallback(monkeypatch, evidence):
    robot = FacingDrift()
    ctx, runner, events = context(robot)
    original_check, original_emit = navigation._Bridge._motion_check, ctx.emit
    closing, refused_at = [], []

    def emit(event):
        original_emit(event)
        if event.get("kind") == "drive.arrival_correction":
            closing.append(True)

    def check(bridge, vx, vy, wz):
        if closing and (vx or vy) and wz:
            refused_at.append(len(robot.commands))
            raise navigation.MapObstacleRefused("combined sweep refused", details={"witness": evidence})
        return original_check(bridge, vx, vy, wz)

    ctx.emit = emit
    monkeypatch.setattr(navigation._Bridge, "_motion_check", check)
    result = runner.run("navigate", {"x": 1.6, "y": 1., "yaw": .5}, robot.frame)

    assert result.code == "ACTION_REJECTED", result
    assert result.details["witness"] == evidence
    assert len(refused_at) == 1
    assert not any(any(command) for command in robot.commands[refused_at[0]:])
    assert sum(event.get("kind") == "drive.arrival_correction" for event in events) == 1
    assert sum(event.get("type") == "skill_finished" for event in events) == 1
    assert np.allclose(robot.velocity, 0.)

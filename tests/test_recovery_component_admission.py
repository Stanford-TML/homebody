"""Recovery restoration uses the same checked REAL command owner as dispatch."""

import numpy as np
import pytest
from test_loaded_component_navigation import component_scene
from test_real_navigation import loaded_navigation, observed_low_wall

from homebody.primitives.navigation.actions.lower_body import LowerBody
from homebody.primitives.navigation.errors import (
    MapObstacleRefused,
)
from homebody.skills.navigate_real import task


def test_runner_recovers_blocked_start_then_follows_oblique_continuation():
    robot, ctx, runner, events = loaded_navigation(compactable=True)
    observed_low_wall(robot, ctx)
    goal = [1., 3.1]

    result = runner.run("navigate", {"goal_xy_m": goal, "snap": False}, robot.frame)

    assert result.ok, result
    kinds = [event.get("kind") for event in events]
    assert kinds.index("recovery.complete") < kinds.index("recovery.continuation")
    restored = next(event["details"] for event in events
                    if event.get("kind") == "recovery.continuation")
    assert restored["outcome"] == "route_restored"
    assert restored["goal_xy"] == goal
    assert np.linalg.norm(robot.frame.base_pose[:2] - goal) <= ctx.settings.motion.arrival_distance
    assert np.allclose(robot.velocity, 0.)
    assert sum(event.get("type") == "skill_finished" for event in events) == 1


@pytest.mark.parametrize("observation", ["missing", "blocked", "below"])
def test_actual_component_probe_matches_dispatch_without_actuation(observation):
    robot, ctx, _, _ = component_scene(observation, finger=True)
    bridge = task._Bridge(ctx.planning(), robot.frame, ctx)
    before = robot.time
    refusal = bridge.motion_refusal(.20, 0., 0.)
    assert robot.commands == [] and robot.time == before and bridge._turn_anchor is None
    assert not bridge.docking
    command = LowerBody(vx=.20, vy=0., wz=0.)
    if observation == "below":
        assert refusal is None
        bridge.step("lower_body", command)
        assert robot.commands[-1] == (.20, 0., 0.)
    else:
        assert isinstance(refusal, MapObstacleRefused)
        assert refusal.details["evidence"] == ("unobserved" if observation == "missing" else "occupied")
        with pytest.raises(MapObstacleRefused) as actual:
            bridge.step("lower_body", command)
        assert actual.value.details == refusal.details
        assert robot.commands == [] and robot.time == before

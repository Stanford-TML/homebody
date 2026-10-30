"""Column margin must be finite and nonnegative, and zero padding refuses actual occupied history hits."""
from dataclasses import replace

import numpy as np
import pytest
from test_loaded_component_navigation import calibrated_component_scene

from homebody.helpers import volume
from homebody.primitives.navigation.errors import MapObstacleRefused
from homebody.session.settings import Settings
from homebody.skills.navigate_real import task


@pytest.mark.parametrize("margin", [-.001, float("inf"), -float("inf"), float("nan")])
def test_column_margin_rejects_negative_or_nonfinite(margin):
    settings = Settings()
    with pytest.raises(ValueError, match="collision.column_margin"):
        replace(settings, collision=replace(settings.collision, column_margin=margin))


def bridge_with_history(monkeypatch, point):
    robot, ctx, observed = calibrated_component_scene(monkeypatch)
    assert ctx.settings.collision.column_margin == 0.
    robot.frame = replace(robot.frame, time=1., depth=np.full_like(robot.frame.depth, np.nan),
                          sensor_metadata=dict(robot.frame.sensor_metadata, depth_time=1.))
    occupied = robot.frame.navigation.occupied.copy()
    occupied[:, 20] = True
    robot.frame = replace(robot.frame, navigation=replace(robot.frame.navigation, occupied=occupied))
    ctx.navigation_history = volume.ColumnHistory(seconds=ctx.settings.collision.column_seconds)
    points = np.empty((0, 3)) if point is None else np.array([point])
    ctx.navigation_history.add(.8, task._column_binding(robot.frame), points,
                               np.full(len(points), 99, dtype=int))
    return robot, ctx, task._Bridge(ctx.planning(), robot.frame, ctx), observed, points


@pytest.mark.parametrize("vx", [0., -.15])
def test_default_bridge_refuses_actual_history_intersection(monkeypatch, vx):
    robot, ctx, bridge, observed, points = bridge_with_history(monkeypatch, [2.025, 3., 1.])
    box = task.robot_boxes(robot.frame, ctx.arms, ctx.carried)[-1]
    assert volume.in_box(points, box).all()
    with pytest.raises(MapObstacleRefused, match="Observed columns") as refusal:
        bridge._dock_check(vx, 0., 0.)
    assert refusal.value.details["evidence"] == "occupied"
    assert refusal.value.details["witness_part"] == "right payload"
    assert len(observed) == 1 and robot.commands == []

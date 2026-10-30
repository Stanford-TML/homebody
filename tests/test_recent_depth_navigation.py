"""Recent registered depth may clear unknown swept parts within the live freshness window."""
from dataclasses import replace

import pytest
from test_loaded_component_navigation import calibrated_component_scene

from homebody.primitives.navigation.errors import MapObstacleRefused
from homebody.skills.contract import SkillFailure
from homebody.skills.navigate_real import task as navigation


def current_frame(robot, time, *, outside=False, metadata=None, **changes):
    camera = robot.frame.world_camera.copy()
    camera[1, 3] = 8. if outside else 3.
    meta = dict(robot.frame.sensor_metadata, depth_time=time)
    meta.update(metadata or {})
    robot.frame = replace(robot.frame, time=time, sequence=robot.frame.sequence + 1,
                          world_camera=camera, sensor_metadata=meta, **changes)


def scene(monkeypatch, observation="missing"):
    robot, ctx, observed = calibrated_component_scene(monkeypatch, observation)
    current_frame(robot, 1.)
    bridge = navigation._Bridge(ctx.planning(), robot.frame, ctx)
    return robot, ctx, observed, bridge


def check(bridge):
    bridge.sample(force=True)
    bridge._motion_check(.20, 0., 0.)


def test_previous_depth_clears_after_latest_view_loses_sweep_until_age_limit(monkeypatch):
    robot, ctx, observed, bridge = scene(monkeypatch)
    check(bridge)
    for time in (1.12, 1.24):
        current_frame(robot, time, outside=True)
        check(bridge)

    assert len(observed) == 3
    assert observed[-1].world_camera[1, 3] == 8.
    assert ctx.settings.motion.observation_age == .25
    current_frame(robot, 1.251, outside=True)
    with pytest.raises(MapObstacleRefused, match="Observed columns") as failure:
        check(bridge)
    assert failure.value.details["uncleared_points"] > 0
    assert not failure.value.details["observed_at_height"]
    assert len(observed) == 3 and robot.commands == []


@pytest.mark.parametrize("metadata", [
    {"calibration_hash": None},
    {"registered_pose": False},
    {"map_frame": "other-room"},
    {"map_sha256": "other-map"},
    {"navigation_generation": (1, 2)},
    {"calibration_hash": "other-calibration"},
    {"camera_identity": "other-camera"},
    {"depth_rays_from_world_camera": False},
    {"depth_rays_from_world_camera": None},
    {"depth_time": .80},
    {"depth_time": 1.20},
    {"depth_time": 0.},
])
def test_latest_provenance_change_prevents_older_depth_from_clearing_sweep(monkeypatch, metadata):
    robot, _ctx, observed, bridge = scene(monkeypatch)
    check(bridge)
    current_frame(robot, 1.12, outside=True, metadata=metadata)

    with pytest.raises((MapObstacleRefused, SkillFailure)):
        check(bridge)

    assert len(observed) == 2 and robot.commands == []


def test_observed_component_height_vetoes_all_retained_free_depth(monkeypatch):
    robot, ctx, observed, bridge = scene(monkeypatch)
    check(bridge)
    blocked, blocked_ctx, _blocked_observed = calibrated_component_scene(monkeypatch, "blocked")
    blocked_evidence = blocked_ctx.navigation_evidence().obstacles
    original = ctx.navigation_evidence
    ctx.navigation_evidence = lambda: replace(original(), obstacles=blocked_evidence)
    bridge.planning = ctx.planning()
    current_frame(robot, 1.12, outside=True)

    with pytest.raises(MapObstacleRefused, match="Observed columns") as failure:
        check(bridge)

    assert failure.value.details["observed_at_height"]
    assert len(observed) == 2 and robot.commands == [] and blocked.commands == []

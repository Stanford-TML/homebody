"""Replay actual SuperOdom ingestion into a nonidentity deployment-map frame."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from homebody.primitives.localization.map_binding import MapBinding, sha256
from homebody.primitives.localization.mapped import SuperOdomMappedLocalization
from homebody.primitives.localization.superodom import SuperOdomLocalization
from homebody.primitives.observations import ObservationError


def replay(tmp_path):
    clock = SimpleNamespace(now=100.)
    occupancy = np.zeros((100, 100), dtype=bool)
    path = tmp_path / "map.npz"
    np.savez(path, occupancy=occupancy, T_map_px=[[.1, 0, -5], [0, .1, -5], [0, 0, 1]])
    binding = MapBinding.load(path, expected_hash=sha256(path), frame_id="mapped-room")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", .4).as_matrix()
    T[:3, 3] = [2, -1, .5]
    source_pose = np.eye(4)
    source_pose[:3, :3] = Rotation.from_euler("z", .2).as_matrix()
    source_pose[:3, 3] = [-1, .3, .15]
    doc = {"map_sha256": binding.sha256, "map_frame": binding.frame_id,
           "source_frame": "ws_map", "units": "m", "provenance": "test measured alignment",
           "anchor_odom_stamp_s": 100., "anchor_T_source_lidar": source_pose.tolist(),
           "T_map_source": T.tolist()}
    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps(doc))
    source = SuperOdomLocalization("unused", start=False)
    source._clock = lambda: clock.now
    source._take_imu(100., 100.)
    source._take_health(True, 100.)
    source._take_odometry(100., source_pose, 100.)
    adapter = SuperOdomMappedLocalization("unused", binding, alignment, source=source,
                                          clock=lambda: clock.now)
    # Floor and obstacle are distant from the robot's own-arm filter and origin.
    points = np.array([[1., .7, -.9], [1.5, -.8, -.4], [2., .3, .1]])
    for i in range(1, 8):
        clock.now = 100. + i * .1
        source._take_imu(clock.now, clock.now)
        source._take_health(True, clock.now)
        source._take_odometry(clock.now, source_pose, clock.now)
        source._on_scan(points, clock.now, received_at=clock.now)
    return adapter, source, clock, source_pose, points, T


def test_map_pose_and_cloud_share_fixed_registration_not_scan_grounding(tmp_path):
    adapter, source, clock, pose, points, transform = replay(tmp_path)
    registered = adapter.pose_at(clock.now)
    actual, captured = adapter.obstacles()
    np.testing.assert_allclose(registered.T_map_lidar, transform @ pose)
    np.testing.assert_allclose(actual, points @ transform[:3, :3].T + transform[:3, 3])
    assert captured == clock.now == registered.captured_at
    assert registered.source_stamp == clock.now
    assert registered.generation == 1
    # Do not inherit the start-frame floor's artificially zeroed Z.
    assert not np.allclose(actual[:, 2], source._last_points[:, 2])
    adapter.close()


@pytest.mark.parametrize("defect", ["frame_reset", "health", "imu_stale", "odom_reversed", "unmatched_capture"])
def test_bad_localization_never_becomes_healthy_navigation(tmp_path, defect):
    adapter, source, clock, pose, _, _ = replay(tmp_path)
    if defect == "frame_reset":
        source.frame_reset = True
    elif defect == "health":
        source._take_health(False, clock.now)
    elif defect == "imu_stale":
        source._imu_reference = (clock.now - 1, clock.now - 1)
    elif defect == "odom_reversed":
        source._take_odometry(99., pose, clock.now)
    with pytest.raises(ObservationError):
        adapter.pose_at(clock.now + .2 if defect == "unmatched_capture" else None)

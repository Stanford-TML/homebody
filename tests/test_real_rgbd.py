"""Measured RGB-D contracts: capture times, color optical pose, pairing and bound depth identity."""
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.primitives.cameras.rgbd import CalibratedRGBD as RGBD
from homebody.primitives.cameras.verified_depth import VerifiedDepthFrame
from homebody.primitives.observations import ObservationError

SERIAL = "123456789012"
CALIBRATION = "a" * 64


def CalibratedRGBD(*args, **kwargs):
    return RGBD(*args, camera_serial=SERIAL, stereo_calibration_sha256=CALIBRATION, **kwargs)


class Clock:
    def __init__(self):
        self.value = 100.
    def __call__(self):
        return self.value
    def sleep(self, seconds):
        self.value += seconds


def calibration(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({"T_cam_base": np.eye(4).tolist(), "provenance": "fixture measured extrinsic"}))
    return path


def clients(stamp=100., depth_stamp=100., geometry=None):
    color = SimpleNamespace(image=np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8), captured_at=stamp, generation=7,
                            camera_serial=SERIAL, capture_timestamp_basis="realsense_global_time",
                            sdk_receive_realtime_ns=int(stamp * 1e9) + 1000000)
    geometry = geometry or {"K_ir": np.eye(3).tolist(), "color_intr": [1, 1, 0, 0, 3, 2],
                            "ir2c_R": np.eye(3).tolist(), "ir2c_t": [0, 0, 0]}
    c = SimpleNamespace(wait=lambda _: color, geometry=geometry, close=lambda: None)
    d = VerifiedDepthFrame(np.full((2, 3), 1000, dtype=np.uint16), 9, int(depth_stamp * 1e9), 0, 0,
                           hashlib.sha256(SERIAL.encode()).hexdigest(), CALIBRATION, "b" * 64)
    return c, SimpleNamespace(read_latest=lambda: d, close=lambda: None)


def test_capture_keeps_rgb_depth_times_and_color_optical_pose(tmp_path):
    clock = Clock()
    c, d = clients(stamp=99.98, depth_stamp=99.96)
    c.geometry["ir2c_t"] = [.1, 0, 0]
    capture = CalibratedRGBD(c, d, calibration(tmp_path), clock=clock,
                             monotonic=clock, sleep=clock.sleep).snapshot()
    assert capture.color_time == 99.98
    assert capture.depth_time == 99.96
    assert capture.assembled_at == clock()
    np.testing.assert_allclose(capture.T_torso_camera[:3, 3], [-.1, 0, 0])
    np.testing.assert_array_equal(capture.rgb[0, 0], [30, 20, 10])
    assert not capture.rgb.flags.writeable
    assert capture.sensor_identity["camera_serial"] == SERIAL
    assert capture.sensor_identity["depth_frame_id"] == 9
    assert capture.sensor_identity["color_timestamp_basis"] == "realsense_global_time"
    with pytest.raises(TypeError):
        capture.sensor_identity["camera_serial"] = "another"


@pytest.mark.parametrize("color_time,depth_time", [(99., 99.), (100.1, 100.), (100., 99.8)])
def test_stale_future_or_unpaired_capture_is_refused(tmp_path, color_time, depth_time):
    clock = Clock()
    c, d = clients(color_time, depth_time)
    with pytest.raises(ObservationError, match="No fresh"):
        CalibratedRGBD(c, d, calibration(tmp_path), wait_s=.02, clock=clock,
                      monotonic=clock, sleep=clock.sleep).snapshot()


@pytest.mark.parametrize("field,value", [("camera_sha256", None), ("camera_sha256", "c" * 64),
    ("calibration_sha256", None), ("calibration_sha256", "c" * 64)])
def test_unbound_or_swapped_depth_refused(tmp_path, field, value):
    clock = Clock()
    c, d = clients()
    frame = replace(d.read_latest(), **{field: value})
    d.read_latest = lambda: frame
    with pytest.raises(ObservationError, match="Depth"):
        CalibratedRGBD(c, d, calibration(tmp_path), clock=clock, monotonic=clock, sleep=clock.sleep).snapshot()

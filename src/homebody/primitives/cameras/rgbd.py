"""Capture-associated RGB-D over the HSCOLOR1 and HMDPTH01 transports."""
import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from homebody.primitives.localization.map_binding import rigid, sha256
from homebody.primitives.observations import ObservationError, frozen_array, frozen_metadata


@dataclass(frozen=True)
class RGBDCapture:
    sequence: int
    captured_at: float
    color_time: float
    depth_time: float
    assembled_at: float
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    T_torso_camera: np.ndarray
    calibration_hash: str
    sensor_identity: object = None

    def __post_init__(self):
        height, width = np.shape(self.depth)
        for name, shape in (("rgb", (height, width, 3)), ("depth", (height, width)),
                            ("intrinsics", (3, 3)), ("T_torso_camera", (4, 4))):
            object.__setattr__(self, name, frozen_array(getattr(self, name), shape))
        if self.sensor_identity is not None:
            object.__setattr__(self, "sensor_identity", frozen_metadata(self.sensor_identity))


def align_depth(depth, K_depth, K_color, T_color_depth, shape):
    """Project valid optical-axis depth to RGB pixels; nearest measured surface wins."""
    depth = np.asarray(depth, dtype=float)
    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[rows, cols]
    points = np.array(((cols - K_depth[0, 2]) * z / K_depth[0, 0],
                       (rows - K_depth[1, 2]) * z / K_depth[1, 1], z))
    points = T_color_depth[:3, :3] @ points
    points += T_color_depth[:3, 3, None]
    h, w = shape
    output = np.full(h * w, np.inf, dtype=np.float32)
    if points.shape[1]:
        positive = points[2] > 0
        z = np.where(positive, points[2], 1.)
        u = np.rint(K_color[0, 0] * points[0] / z + K_color[0, 2]).astype(int)
        v = np.rint(K_color[1, 1] * points[1] / z + K_color[1, 2]).astype(int)
        valid = positive & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        np.minimum.at(output, v[valid] * w + u[valid], points[2, valid].astype(np.float32))
    output[~np.isfinite(output)] = np.nan
    return output.reshape(h, w)


def intrinsics(value, name):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])):
        raise ObservationError(f"{name} has invalid optical intrinsics")
    return matrix


def validate_head_calibration(path, *, root=None):
    path = Path(path)
    doc = json.loads(path.read_text())
    if not doc.get("measured_source") and not doc.get("provenance"):
        raise ObservationError("Head camera calibration requires measured provenance")
    rigid(doc["T_cam_base"], "head optical extrinsic")
    scale, bias = float(doc.get("depth_scale", 1.)), float(doc.get("depth_bias_m", 0.))
    if not np.isfinite([scale, bias]).all() or scale <= 0:
        raise ObservationError("Invalid calibrated metric depth scale/bias")
    if doc.get("urdf_sha256"):
        model = (Path(root) if root else path.parent.parent.parent) / doc["urdf"]
        if not model.is_file() or sha256(model) != doc["urdf_sha256"]:
            raise ObservationError("Camera calibration's referenced URDF hash differs")
    return doc


class CalibratedRGBD:
    """Calibrated RGB-D pairs from running HSCOLOR1 and HMDPTH01 producers; starts no service."""
    def __init__(self, color, depth, calibration, *, camera_serial, stereo_calibration_sha256,
                 root=None, max_age_s=.5, pair_skew_s=.1,
                 wait_s=2.5, clock=time.time, monotonic=time.monotonic, sleep=time.sleep):
        self.color, self.depth = color, depth
        self.clock, self.monotonic, self.sleep = clock, monotonic, sleep
        self.max_age_s, self.pair_skew_s, self.wait_s = max_age_s, pair_skew_s, wait_s
        self.camera_serial = camera_serial
        self.camera_sha256 = hashlib.sha256(camera_serial.encode()).hexdigest()
        self.stereo_calibration_sha256 = stereo_calibration_sha256
        path = Path(calibration)
        doc = validate_head_calibration(path, root=root)
        # T_cam_base maps left-IR optical axes to the torso.
        self.T_torso_depth = rigid(doc["T_cam_base"], "head optical extrinsic")
        self.calibration_hash = sha256(path)
        self.depth_scale = float(doc.get("depth_scale", 1.))
        self.depth_bias = float(doc.get("depth_bias_m", 0.))
        self._colors = deque(maxlen=32)
        self._depths = deque(maxlen=32)
        self._last_depth_id = None
        self._sequence = 0
        self._pair = None
        self._capture = None

    def snapshot(self):
        deadline = self.monotonic() + self.wait_s
        while True:
            color = self.color.wait(0.)
            if color is not None:
                self._colors.append((color, dict(self.color.geometry or {})))
            depth = self.depth.read_latest()
            if depth is not None:
                key = (depth.frame_id, depth.camera_capture_time_ns)
                if key != self._last_depth_id:
                    self._depths.append(depth)
                    self._last_depth_id = key
            now = self.clock()
            candidates = [(c, g, d) for c, g in self._colors for d in self._depths
                          if 0 <= now - c.captured_at <= self.max_age_s
                          and 0 <= now - d.camera_capture_time_ns / 1e9 <= self.max_age_s
                          and abs(c.captured_at - d.camera_capture_time_ns / 1e9) <= self.pair_skew_s]
            if candidates:
                color, geometry, depth = max(candidates, key=lambda pair: min(
                    pair[0].captured_at, pair[2].camera_capture_time_ns / 1e9))
                return self._assemble(color, geometry, depth)
            if self.monotonic() >= deadline:
                raise ObservationError("No fresh capture-associated calibrated RGB-D pair; "
                                       "an RGB-only iPhone stream cannot supply metric depth")
            self.sleep(.005)

    def _assemble(self, color, geometry, depth):
        if getattr(color, "camera_serial", None) != self.camera_serial:
            raise ObservationError("RGB camera identity differs from the calibrated head camera")
        if getattr(color, "capture_timestamp_basis", None) not in {
                "realsense_global_time", "realsense_system_time"}:
            raise ObservationError("RGB capture time has no verified RealSense clock domain")
        if getattr(depth, "camera_sha256", None) != self.camera_sha256:
            raise ObservationError("Depth has no matching calibrated head camera identity")
        if getattr(depth, "calibration_sha256", None) != self.stereo_calibration_sha256:
            raise ObservationError("Depth has no matching stereo calibration identity")
        identity = (color.generation, color.captured_at, depth.frame_id, depth.camera_capture_time_ns)
        if identity == self._pair:
            return self._capture
        try:
            K_depth = intrinsics(np.asarray(geometry["K_ir"]).reshape(3, 3), "depth")
            fx, fy, cx, cy, width, height = geometry["color_intr"]
            K_color = intrinsics([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "RGB")
            T_color_depth = np.eye(4)
            T_color_depth[:3, :3] = np.asarray(geometry["ir2c_R"]).reshape(3, 3)
            T_color_depth[:3, 3] = np.asarray(geometry["ir2c_t"]).reshape(3)
            T_color_depth = rigid(T_color_depth, "depth to RGB calibration")
        except (KeyError, ValueError, TypeError) as exc:
            raise ObservationError(f"HSCOLOR1 lacks complete RGB-D calibration: {exc}") from exc
        if color.image.shape != (int(height), int(width), 3):
            raise ObservationError("RGB dimensions differ from its calibrated intrinsics")
        native = depth.depth_mm.astype(np.float32) / 1000. * self.depth_scale + self.depth_bias
        native[depth.depth_mm == 0] = np.nan
        aligned = align_depth(native, K_depth, K_color, T_color_depth, color.image.shape[:2])
        self._sequence += 1
        self._capture = RGBDCapture(self._sequence, color.captured_at, color.captured_at,
                                   depth.camera_capture_time_ns / 1e9, self.clock(),
                                   color.image[..., ::-1], aligned, K_color,
                                   self.T_torso_depth @ np.linalg.inv(T_color_depth),
                                   self.calibration_hash,
                                   {"camera_serial": self.camera_serial,
                                    "camera_sha256": depth.camera_sha256,
                                    "stereo_calibration_sha256": depth.calibration_sha256,
                                    "depth_engine_sha256": depth.engine_sha256,
                                    "depth_frame_id": depth.frame_id,
                                    "depth_identity_basis": "FFSTEL01 exact frame_id and capture_time_ns",
                                    "color_timestamp_basis": color.capture_timestamp_basis,
                                    "color_sdk_receive_realtime_ns": getattr(color, "sdk_receive_realtime_ns", None)})
        self._pair = identity
        return self._capture

    def close(self):
        try:
            self.color.close()
        finally:
            self.depth.close()

"""Bind HMDPTH01 depth to its exact FFSTEL01 camera/calibration evidence."""
import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass

import zmq

from homebody.primitives.observations import ObservationError

from .metric_depth import LatestMetricDepthClient, MetricDepthFrame


@dataclass(frozen=True)
class VerifiedDepthFrame(MetricDepthFrame):
    camera_sha256: str
    calibration_sha256: str
    engine_sha256: str


class VerifiedMetricDepth:
    """Bounded join; neither a matching frame counter alone nor receipt time establishes identity."""
    def __init__(self, depth, telemetry, *, camera_serial, calibration_sha256):
        self.depth, self.telemetry = depth, telemetry
        self.camera_sha256 = hashlib.sha256(camera_serial.encode()).hexdigest()
        self.calibration_sha256 = calibration_sha256
        self._records = OrderedDict()

    def _record(self, payload):
        try:
            record = json.loads(payload)
            if not isinstance(record, dict) or record.get("schema") != "FFSTEL01":
                raise ValueError("Expected FFSTEL01 identity evidence")
            frame, capture = record["frame_id"], record["capture_time_ns"]
            if type(frame) is not int or frame < 0 or type(capture) is not int or capture <= 0:
                raise ValueError("Invalid depth frame/capture identity")
            for name in ("camera_sha256", "calibration_sha256", "engine_sha256"):
                if not isinstance(record.get(name), str) or re.fullmatch(r"[0-9a-f]{64}", record[name]) is None:
                    raise ValueError("Missing or invalid depth " + name)
            identity = (frame, capture)
            previous = self._records.get(identity)
            if previous is not None and any(previous[name] != record[name] for name in
                                            ("camera_sha256", "calibration_sha256", "engine_sha256")):
                raise ValueError("Conflicting depth provenance for one capture")
            self._records[identity] = record
            while len(self._records) > 64:
                self._records.popitem(last=False)
        except (KeyError, TypeError, ValueError) as exc:
            raise ObservationError(f"Depth identity evidence refused: {exc}") from exc

    def read_latest(self):
        for _ in range(64):
            payload = self.telemetry.read_latest()
            if payload is None:
                break
            self._record(payload)
        depth = self.depth.read_latest()
        if depth is None:
            return None
        record = self._records.get((depth.frame_id, depth.camera_capture_time_ns))
        if record is None:
            return None  # RGB-D's bounded wait handles delayed/missing sideband data.
        if record["camera_sha256"] != self.camera_sha256:
            raise ObservationError("Depth camera identity differs from the calibrated head camera")
        if record["calibration_sha256"] != self.calibration_sha256:
            raise ObservationError("Depth stereo calibration identity differs")
        return VerifiedDepthFrame(depth.depth_mm, depth.frame_id, depth.camera_capture_time_ns,
                                  depth.completion_monotonic_ns, depth.ffs_latency_ns,
                                  record["camera_sha256"], record["calibration_sha256"], record["engine_sha256"])

    def close(self):
        try:
            self.depth.close()
        finally:
            self.telemetry.close()


class _Telemetry:
    def __init__(self, endpoint):
        self.socket = zmq.Context.instance().socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, 64)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        try:
            self.socket.connect(endpoint)
        except BaseException:
            self.socket.close(0)
            raise

    def read_latest(self):
        try:
            return self.socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None

    def close(self):
        self.socket.close(0)


def connect_verified_depth(config):
    depth = LatestMetricDepthClient(endpoint=config.depth_endpoint, shape=config.depth_shape)
    try:
        telemetry = _Telemetry(config.depth_telemetry_endpoint)
    except BaseException:
        depth.close()
        raise
    return VerifiedMetricDepth(depth, telemetry, camera_serial=config.head_camera_serial,
                               calibration_sha256=config.stereo_calibration_sha256)

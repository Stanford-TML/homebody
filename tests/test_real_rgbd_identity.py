"""Metric depth is refused when its telemetry has a wrong or missing camera, calibration or engine identity."""
import hashlib
import json
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.primitives.cameras.metric_depth import MetricDepthFrame
from homebody.primitives.cameras.verified_depth import VerifiedMetricDepth
from homebody.primitives.observations import ObservationError

SERIAL = "123456789012"
CAMERA = hashlib.sha256(SERIAL.encode()).hexdigest()
CALIBRATION = "a" * 64


def metadata(frame=1, stamp=100000000000, **overrides):
    return json.dumps({"schema": "FFSTEL01", "frame_id": frame, "capture_time_ns": stamp,
                       "camera_sha256": CAMERA, "calibration_sha256": CALIBRATION,
                       "engine_sha256": "b" * 64, **overrides}).encode()


def fixture():
    messages = deque()
    frame = MetricDepthFrame(np.ones((2, 3), np.uint16), 1, 100000000000, 4, 5)
    depth = SimpleNamespace(read_latest=lambda: frame, close=lambda: None)
    telemetry = SimpleNamespace(read_latest=lambda: messages.popleft() if messages else None,
                                close=lambda: None)
    return VerifiedMetricDepth(depth, telemetry, camera_serial=SERIAL,
                               calibration_sha256=CALIBRATION), messages


@pytest.mark.parametrize("override", [{"camera_sha256": "c" * 64},
    {"calibration_sha256": "c" * 64}, {"camera_sha256": None}, {"engine_sha256": None}])
def test_wrong_or_missing_identity_refuses(override):
    client, messages = fixture()
    messages.append(metadata(**override))
    with pytest.raises(ObservationError):
        client.read_latest()

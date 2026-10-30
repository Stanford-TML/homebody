"""Sensor provenance travels with frozen frames into the recording."""
import json
from dataclasses import replace

import pytest
from test_real_backend import backend
from test_session import ImageOnlyRecorder

from homebody.session.settings import Settings


def test_real_frame_metadata_is_deeply_frozen_and_matches_recorded_capture(tmp_path):
    robot, _, clock = backend(tmp_path)
    frame = robot.snapshot()
    assert frame.sensor_metadata["hand_time_basis"].startswith("DDS receipt")
    assert frame.sensor_metadata["registered_pose"] is True
    with pytest.raises(TypeError):
        frame.sensor_metadata["hand_times"]["left"] = 9.
    source = {"nested": {"capture": 1.}}
    copied = replace(frame, sensor_metadata=source)
    source["nested"]["capture"] = 2.
    assert copied.sensor_metadata["nested"]["capture"] == 1.

    recorder = ImageOnlyRecorder(tmp_path, Settings(), tmp_path / "runs", backend="real")
    rgb, _, _ = recorder.frame(frame)
    capture = robot.ports.camera.snapshot()
    clock.sleep(.04)
    robot.ports.camera.snapshot = lambda: replace(capture, sequence=2, captured_at=clock(),
                                                 color_time=clock(), depth_time=clock())
    newer = robot.snapshot()
    assert newer.sensor_metadata["color_time"] != frame.sensor_metadata["color_time"]
    recorder.close()
    state = json.loads((rgb.parent / "state.json").read_text())
    events = [json.loads(line) for line in (recorder.path / "events.jsonl").read_text().splitlines()]
    saved = state["sensor_metadata"]
    assert saved == next(row["sensor_metadata"] for row in events if row["kind"] == "frame")
    for key in ("color_time", "depth_time", "pose_time", "body_time", "hand_times",
                "obstacle_time", "camera_assembled_at", "source_odom_stamp", "map_frame",
                "map_sha256", "calibration_hash", "pose_time_basis", "hand_time_basis"):
        assert saved[key] == frame.sensor_metadata[key]
    assert "captured sensor timestamps" in json.loads((recorder.path / "provenance.json").read_text())["video"]

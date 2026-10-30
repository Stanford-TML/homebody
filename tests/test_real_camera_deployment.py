"""Producer closure, offline deployment and consumer protocol compatibility."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from homebody.primitives.cameras import deployment

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/homebody/primitives/cameras/deployment.py"


def measured_calibration(tmp_path):
    intrinsics = {"width": 848, "height": 480, "fx": 400., "fy": 400.,
                  "ppx": 424., "ppy": 240., "distortion_model": "brown_conrady",
                  "distortion_coefficients": [0.] * 5}
    value = {"schema_version": 1, "camera_name": "Intel RealSense D435I",
             "camera_serial": "123456789012", "firmware_version": "fixture",
             "width": 848, "height": 480, "fps": 60, "pixel_format": "Y8",
             "left_stream_index": 1, "right_stream_index": 2,
             "left_intrinsics": intrinsics, "right_intrinsics": intrinsics,
             "left_to_right_rotation": np.eye(3).ravel().tolist(),
             "left_to_right_translation_m": [-.05, 0., 0.]}
    path = tmp_path / "measured.json"
    path.write_text(json.dumps(value))
    return path


def test_pinned_closure_hashes_match():
    deployment.verify_closure()


def test_help_and_offline_plan_do_not_import_hardware_or_start_processes(tmp_path):
    code = """
import runpy, sys
def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.exec', 'socket.connect', 'socket.bind'):
        raise RuntimeError('unexpected effect: ' + event)
    if event == 'import' and args[0].split('.')[0] in ('pyrealsense2', 'torch', 'tensorrt', 'zmq'):
        raise RuntimeError('unexpected runtime import: ' + args[0])
sys.addaudithook(audit)
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    for arguments in (["--help"], ["check", "color", "--calibration", str(measured_calibration(tmp_path))]):
        result = subprocess.run([sys.executable, "-c", code, str(SCRIPT), *arguments],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr


def test_server_requires_verified_engine_and_relay_binds_identities(tmp_path):
    cal = measured_calibration(tmp_path)
    args = deployment.parser().parse_args(["plan", "server", "--calibration", str(cal)])
    with pytest.raises(ValueError, match="explicit path"):
        deployment.make_plan(args)
    engine = tmp_path / "fixture.engine"
    engine.write_bytes(b"fixture only; not an executable engine")
    args.engine, args.engine_sha256 = engine, "0" * 64
    with pytest.raises(ValueError, match="differs"):
        deployment.make_plan(args)
    args.engine_sha256 = deployment.digest(engine)
    plan = deployment.make_plan(args)
    assert plan["environment"]["PYTHONPATH"] == str(ROOT / "vendor/mjlab_depth")
    assert "not performed" in plan["runtime_validation"]
    args.stage = "relay"
    command = deployment.make_plan(args)["command"]
    assert command[command.index("--expected-camera-sha256") + 1] == hashlib.sha256(b"123456789012").hexdigest()
    assert "--expected-calibration-sha256" in command
    assert args.engine_sha256 in command

"""Explicit foreground head-camera producer stages; plan/check are offline.

Usable as a standalone Python 3.8 script on the camera host. The pinned stereo
package runs in an isolated subprocess namespace, separate from locomotion.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VENDOR = ROOT / "vendor/mjlab_depth"
SHA256 = re.compile(r"[0-9a-f]{64}")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_closure():
    manifest = json.loads((VENDOR / "CLOSURE.json").read_text())
    for name, expected in manifest["files"].items():
        path = (VENDOR / name).resolve()
        if VENDOR.resolve() not in path.parents or digest(path) != expected:
            raise ValueError("Pinned stereo source differs: " + name)


def calibration(path):
    # calibration.py imports only the stdlib; loading it never discovers or opens a camera.
    spec = importlib.util.spec_from_file_location(
        "_homebody_stereo_calibration", VENDOR / "holosoma_inference/fast_stereo/calibration.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    record = module.load_calibration(Path(path))
    return record, module.calibration_digest(record)


def require_file(path, expected, kind):
    if not path or not expected or SHA256.fullmatch(expected) is None:
        raise ValueError(kind + " requires an explicit path and lowercase SHA-256")
    if digest(path) != expected:
        raise ValueError(kind + " SHA-256 differs")
    return str(Path(path).resolve())


def make_plan(args):
    verify_closure()
    command = [args.python]
    environment = {"PYTHONPATH": str(VENDOR)}
    modules = []
    serial = ""
    record = None
    if args.stage != "build":
        if not args.calibration:
            raise ValueError("Every producer stage requires --calibration for this physical camera")
        record, calibration_hash = calibration(args.calibration)
        serial = record.camera_serial
    if args.stage == "capture":
        command += ["-m", "holosoma_inference.fast_stereo.g1_capture",
                    "--calibration", str(args.calibration.resolve()), "--duration-s", "86400",
                    "--endpoint", args.stereo_bind]
        if args.jpeg_quality is not None:
            command += ["--jpeg-quality", str(args.jpeg_quality)]
        if args.output_rate_hz is not None:
            command += ["--output-rate-hz", str(args.output_rate_hz)]
        modules = ["pyrealsense2", "numpy", "cv2", "zmq"]
    elif args.stage == "color":
        command += [str(Path(__file__).with_name("color_publisher.py")), "--serial", serial,
                    "--endpoint", args.color_bind, "--ir-width", str(record.width),
                    "--width", str(record.width), "--height", str(record.height)]
        modules = ["pyrealsense2", "numpy", "cv2", "zmq"]
    elif args.stage == "server":
        engine = require_file(args.engine, args.engine_sha256, "TensorRT engine")
        command += ["-m", "holosoma_inference.runtime.ffs_stream_server", "--engine", engine,
                    "--calibration", str(args.calibration.resolve()),
                    "--stereo-endpoint", args.stereo_endpoint,
                    "--depth-endpoint", args.ffs_depth_endpoint,
                    "--telemetry-endpoint", args.telemetry_endpoint]
        modules = ["torch", "tensorrt", "numpy", "cv2", "zmq"]
    elif args.stage == "relay":
        if not args.engine_sha256 or SHA256.fullmatch(args.engine_sha256) is None:
            raise ValueError("Relay requires the explicit --engine-sha256 expected from its server")
        command += ["-m", "holosoma_inference.run_depth_vision",
                    "--source-depth", args.ffs_depth_endpoint,
                    "--source-telemetry", args.telemetry_endpoint, "--output", args.depth_bind,
                    "--width", str(record.width), "--height", str(record.height),
                    "--expected-engine-sha256", args.engine_sha256,
                    "--expected-camera-sha256", hashlib.sha256(serial.encode()).hexdigest(),
                    "--expected-calibration-sha256", calibration_hash]
        modules = ["numpy", "zmq"]
    else:
        onnx = require_file(args.onnx, args.onnx_sha256, "ONNX export")
        data = Path(onnx + ".data")
        if data.exists() or args.onnx_data_sha256:
            require_file(data, args.onnx_data_sha256, "ONNX external tensor data")
        if not args.output or args.output.exists():
            raise ValueError("Build requires --output pointing to a new engine file")
        command += [str(VENDOR / "build_ffs_engine.py"), "--onnx", onnx,
                    "--output", str(args.output.resolve())]
        if args.fp32:
            command.append("--fp32")
        modules = ["tensorrt"]
    return {"stage": args.stage, "command": command, "environment": environment,
            "camera_serial": serial, "required_modules": modules,
            "runtime_validation": "not performed; compatible CUDA/TensorRT/GPU engine required",
            "hardware_validation": "not performed"}


def ownership_lock(stage, serial):
    """Keep a per-camera/stage lease across exec; never evict an existing owner."""
    import fcntl
    directory = Path(os.environ.get("XDG_RUNTIME_DIR", str(Path.home() / ".cache")))
    directory = directory / "homebody/cameras"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (stage + "-" + (serial or "engine-build") + ".lock")
    descriptor = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError("Producer stage already owned: " + str(path)) from exc
    os.set_inheritable(descriptor, True)
    return descriptor


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("plan", "check", "run"))
    result.add_argument("stage", choices=("capture", "color", "server", "relay", "build"))
    result.add_argument("--python", default=sys.executable, help="interpreter on this host")
    result.add_argument("--calibration", type=Path, help="measured stereo calibration for this serial")
    result.add_argument("--engine", type=Path)
    result.add_argument("--engine-sha256")
    result.add_argument("--onnx", type=Path)
    result.add_argument("--onnx-sha256")
    result.add_argument("--onnx-data-sha256", help="required when ONNX uses its adjacent .data file")
    result.add_argument("--output", type=Path)
    result.add_argument("--fp32", action="store_true")
    result.add_argument("--stereo-bind", default="tcp://0.0.0.0:55562")
    result.add_argument("--stereo-endpoint", default="tcp://127.0.0.1:55562")
    result.add_argument("--color-bind", default="tcp://0.0.0.0:55566")
    result.add_argument("--ffs-depth-endpoint", default="tcp://127.0.0.1:55565")
    result.add_argument("--telemetry-endpoint", default="tcp://127.0.0.1:55564")
    result.add_argument("--depth-bind", default="tcp://127.0.0.1:55563")
    result.add_argument("--jpeg-quality", type=int, choices=range(1, 101), metavar="1..100")
    result.add_argument("--output-rate-hz", type=float)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        plan = make_plan(args)
        print(json.dumps(plan, indent=2), flush=True)
        if args.mode in ("plan", "check"):
            print("Offline source/asset check only. Install required_modules separately; "
                  "camera availability, clocks and engine compatibility remain unverified.")
            return 0
        descriptor = ownership_lock(args.stage, plan["camera_serial"])
        environment = dict(os.environ)
        environment.update(plan["environment"])
        print("Starting explicitly requested foreground stage: " +
              shlex.join(plan["command"]), flush=True)
        try:
            os.execvpe(plan["command"][0], plan["command"], environment)
        finally:
            os.close(descriptor)
    except (OSError, ValueError, RuntimeError) as exc:
        print("Camera deployment refused: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

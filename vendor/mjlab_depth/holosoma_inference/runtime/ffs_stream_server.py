# Modified for HomeBody; the changes are recorded in vendor/mjlab_depth/CLOSURE.json.
"""Fast-FoundationStereo stream server for the G1 head stereo camera.

Consumes the G1's native ``HSSPAIR1`` capture stream
(``holosoma_inference.fast_stereo``) and serves metric depth:

    G1 capture (fast_stereo.g1_capture; PUSH bind :55562, atomic Y8 or JPEG pair packets)
    -> this server: fast_stereo LatestStereoPairReceiver (drain to newest)
       + TensorRT FFS at the 384x288 profile
    -> HSDPTH01 full-resolution uint16 millimetre depth on --depth-endpoint
       + FFSTEL01 JSON telemetry on --telemetry-endpoint (PUB, latest-only)
    -> the relay (run_depth_vision.py) joins the two by frame ID into
       the single HMDPTH01 stream that depth consumers read.

The calibration JSON is the same file the capture validates against
(identical on the capture and inference hosts); its digest must match
the ``calibration_sha256`` every stereo header carries, so a camera or
calibration swap fails loudly instead of silently changing the geometry.

Run (in an environment with tensorrt and torch)::

    python -m holosoma_inference.runtime.ffs_stream_server \
        --engine <fast_foundationstereo.engine> \
        --calibration <stereo-calibration.json> \
        --stereo-endpoint tcp://192.168.123.164:55562
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from holosoma_inference.fast_stereo import calibration as fs_calibration
from holosoma_inference.fast_stereo import pair_protocol as fs_pair_protocol
from holosoma_inference.runtime.metric_depth import pack_ffs_depth

DEFAULT_STEREO_ENDPOINT = "tcp://192.168.123.164:55562"
DEFAULT_DEPTH_ENDPOINT = "tcp://127.0.0.1:55565"
DEFAULT_TELEMETRY_ENDPOINT = "tcp://127.0.0.1:55564"


@dataclass(frozen=True)
class StereoFrame:
    """One identity-checked stereo frameset ready for inference."""

    left: np.ndarray
    right: np.ndarray
    frame_id: int
    capture_time_ns: int
    camera_sha256: str
    calibration_sha256: str
    fx_px: float
    baseline_m: float


def _decode_image(payload, pixel_format: int, shape: tuple[int, int]) -> np.ndarray:
    """Decode one raw Y8 or JPEG-encoded stereo image."""
    if pixel_format == fs_pair_protocol.PIXEL_FORMAT_JPEG:
        import cv2

        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != tuple(shape):
            raise ValueError(f"JPEG stereo payload did not decode to {shape}")
        return image
    return np.frombuffer(payload, dtype=np.uint8).reshape(shape)


def frame_from_pair(pair, calibration_record) -> StereoFrame:
    """Build a StereoFrame from one fast_stereo ``CachedStereoPair``.

    The header's calibration digest must equal the local calibration file's
    digest -- the file supplies fx/baseline, so a mismatch means the depth
    would be computed with the wrong geometry.
    """

    header = pair.header
    calibration_sha256 = header.calibration_sha256.hex()
    expected = fs_calibration.calibration_digest(calibration_record)
    if calibration_sha256 != expected:
        raise ValueError(
            f"stereo header calibration {calibration_sha256[:12]}… does not match "
            f"local calibration file {expected[:12]}… -- wrong camera or stale file"
        )
    height, width = header.height, header.width
    left = _decode_image(pair.left, header.pixel_format, (height, width))
    right = _decode_image(pair.right, header.pixel_format, (height, width))
    return StereoFrame(
        left=left,
        right=right,
        frame_id=header.sequence,
        capture_time_ns=header.capture_realtime_ns,
        camera_sha256=header.camera_serial_sha256.hex(),
        calibration_sha256=calibration_sha256,
        fx_px=float(calibration_record.left_intrinsics.fx),
        baseline_m=fs_calibration.stereo_baseline_m(calibration_record),
    )


def disparity_to_depth_mm(
    disparity_engine: np.ndarray,
    *,
    out_shape: tuple[int, int],
    fx_native_px: float,
    baseline_m: float,
    max_depth_m: float = 20.0,
) -> np.ndarray:
    """Engine-resolution disparity -> full-resolution uint16 millimetre depth.

    With the stretch-resize geometry, convert to metric depth AT ENGINE
    RESOLUTION with the width-scaled focal length, mark invalid there, then
    NEAREST-upsample the uint16 image. Upsampling depth with nearest (not
    disparity with bilinear) keeps invalid pixels exactly invalid -- a bilinear
    disparity upsample bleeds invalid zeros into valid neighbours and turns
    edge pixels into wrong finite depths. A disparity is invalid when
    non-finite, non-positive, or right-image-invisible (the match x - d would
    fall off the right image).
    """

    import cv2

    out_h, out_w = out_shape
    disp = np.asarray(disparity_engine, dtype=np.float32)
    if disp.ndim != 2:
        raise ValueError(f"disparity must be 2-D, got {disp.shape}")
    eng_h, eng_w = disp.shape
    fx_engine = fx_native_px * (eng_w / float(out_w))
    x_coords = np.arange(eng_w, dtype=np.float32)[None, :]
    invalid = ~np.isfinite(disp) | (disp <= 0.0) | (x_coords - disp < 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_m = fx_engine * baseline_m / disp
    depth_m[invalid | ~np.isfinite(depth_m)] = 0.0
    np.clip(depth_m, 0.0, max_depth_m, out=depth_m)
    depth_mm = (depth_m * 1000.0).round().astype(np.uint16)
    if (eng_h, eng_w) != (out_h, out_w):
        depth_mm = cv2.resize(depth_mm, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(depth_mm)


def build_telemetry(
    frame: StereoFrame,
    *,
    publish_time_ns: int,
    host_to_device_ms: float,
    infer_depth_ms: float,
    device_to_host_ms: float,
    engine_sha256: str,
) -> bytes:
    """One FFSTEL01 record, shaped exactly for the relay's parse_telemetry."""

    record = {
        "schema": "FFSTEL01",
        "frame_id": frame.frame_id,
        "capture_time_ns": frame.capture_time_ns,
        "publish_time_ns": publish_time_ns,
        "host_to_device_ms": max(0.0, float(host_to_device_ms)),
        "infer_depth_ms": max(0.0, float(infer_depth_ms)),
        "device_to_host_ms": max(0.0, float(device_to_host_ms)),
        "engine_sha256": engine_sha256,
        "camera_sha256": frame.camera_sha256,
        "calibration_sha256": frame.calibration_sha256,
    }
    return json.dumps(record).encode("utf-8")


def process_stereo_frame(
    frame: StereoFrame,
    *,
    infer: Callable[[np.ndarray, np.ndarray], np.ndarray],
    engine_sha256: str,
    out_shape: tuple[int, int],
    timings_ms: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[bytes, bytes]:
    """Run one frame through FFS and pack the (HSDPTH01, FFSTEL01) pair."""

    disparity = infer(frame.left, frame.right)
    depth_mm = disparity_to_depth_mm(
        disparity,
        out_shape=out_shape,
        fx_native_px=frame.fx_px,
        baseline_m=frame.baseline_m,
    )
    publish_time_ns = max(time.time_ns(), frame.capture_time_ns)
    depth_packet = pack_ffs_depth(
        depth_mm,
        frame_id=frame.frame_id,
        capture_time_ns=frame.capture_time_ns,
        publish_time_ns=publish_time_ns,
    )
    h2d, infer_ms, d2h = timings_ms
    telemetry_packet = build_telemetry(
        frame,
        publish_time_ns=publish_time_ns,
        host_to_device_ms=h2d,
        infer_depth_ms=infer_ms,
        device_to_host_ms=d2h,
        engine_sha256=engine_sha256,
    )
    return depth_packet, telemetry_packet


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="Fast-FoundationStereo .engine")
    parser.add_argument("--calibration", required=True, type=Path,
                        help="Stereo calibration JSON that the capture validates against (fx/baseline + identity)")
    parser.add_argument("--stereo-endpoint", default=DEFAULT_STEREO_ENDPOINT,
                        help="G1 capture HSSPAIR1 endpoint (the robot binds; we connect)")
    parser.add_argument("--depth-endpoint", default=DEFAULT_DEPTH_ENDPOINT)
    parser.add_argument("--telemetry-endpoint", default=DEFAULT_TELEMETRY_ENDPOINT)
    parser.add_argument("--status-every-s", type=float, default=5.0)
    args = parser.parse_args()

    import zmq

    from holosoma_inference.fast_stereo.publisher import LatestStereoPairReceiver
    from holosoma_inference.runtime.trt_ffs_runner import TrtFfsRunner

    calibration_record = fs_calibration.load_calibration(args.calibration)
    runner = TrtFfsRunner(args.engine)
    engine_sha256 = file_sha256(args.engine)
    print(
        f"FFS server | engine {args.engine} ({runner.height}x{runner.width}) "
        f"sha256 {engine_sha256[:12]}… | camera {calibration_record.camera_serial} "
        f"fx={calibration_record.left_intrinsics.fx:.2f} "
        f"baseline={fs_calibration.stereo_baseline_m(calibration_record):.4f} | "
        f"stereo {args.stereo_endpoint} -> depth {args.depth_endpoint} "
        f"+ telemetry {args.telemetry_endpoint}",
        flush=True,
    )

    receiver = LatestStereoPairReceiver(args.stereo_endpoint)
    context = zmq.Context()
    depth_pub = context.socket(zmq.PUB)
    telemetry_pub = context.socket(zmq.PUB)
    for socket, endpoint in ((depth_pub, args.depth_endpoint), (telemetry_pub, args.telemetry_endpoint)):
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(endpoint)

    processed = 0
    window = 0
    rejected = 0
    last_status = time.monotonic()
    out_shape: tuple[int, int] | None = None
    try:
        while True:
            pair = receiver.receive_latest(timeout_ms=1000)
            if pair is None:
                now = time.monotonic()
                if now - last_status >= args.status_every_s:
                    print(
                        f"FFS server | waiting for stereo pairs on {args.stereo_endpoint} "
                        f"(received {receiver.counters.received}, "
                        f"gaps {receiver.counters.sequence_gaps}, "
                        f"malformed {receiver.counters.malformed})",
                        flush=True,
                    )
                    last_status = now
                continue
            try:
                frame = frame_from_pair(pair, calibration_record)
            except ValueError as exc:
                rejected += 1
                print(f"FFS server rejected stereo pair: {exc}", flush=True)
                continue
            if out_shape is None:
                out_shape = frame.left.shape
                print(f"FFS server locked output shape {out_shape}", flush=True)

            t0 = time.perf_counter()
            depth_packet, telemetry_packet = process_stereo_frame(
                frame,
                infer=runner,
                engine_sha256=engine_sha256,
                out_shape=out_shape,
                timings_ms=runner.last_timings_ms,
            )
            depth_pub.send(depth_packet)
            telemetry_pub.send(telemetry_packet)
            processed += 1
            window += 1

            now = time.monotonic()
            if now - last_status >= args.status_every_s:
                dt = now - last_status
                print(
                    f"FFS server | {window / dt:.1f} Hz | total {processed} "
                    f"rejected {rejected} | receiver: received {receiver.counters.received} "
                    f"gaps {receiver.counters.sequence_gaps} "
                    f"replaced {receiver.counters.cache_replaced} | "
                    f"last cycle {(time.perf_counter() - t0) * 1000.0:.1f} ms",
                    flush=True,
                )
                window = 0
                last_status = now
    except KeyboardInterrupt:
        print("FFS server stopped", flush=True)
    finally:
        receiver.close()
        for socket in (depth_pub, telemetry_pub):
            socket.close(0)
        context.term()


if __name__ == "__main__":
    main()

# Modified for HomeBody; the changes are recorded in vendor/mjlab_depth/CLOSURE.json.
"""Non-actuating G1 D435i IR1/IR2 capture and HSSPAIR1 publisher."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from holosoma_inference.fast_stereo import (
    calibration,
    native_depth_recorder,
    pair_protocol,
    pair_recorder,
)
from holosoma_inference.fast_stereo.capture_core import (
    CaptureCore,
    FrameObservation,
    OutputCadenceSelector,
)
from holosoma_inference.fast_stereo.publisher import DEFAULT_BIND_ENDPOINT, StereoPairPublisher

METADATA_FIELDS = (
    "frame_timestamp",
    "sensor_timestamp",
    "time_of_arrival",
    "backend_timestamp",
    "actual_exposure",
    "frame_counter",
)


class G1CaptureError(RuntimeError):
    """Raised when the camera runtime cannot satisfy the frozen capture contract."""


def timestamp_domain_to_protocol(domain: Any, rs_module: Any) -> int:
    """Map only the three documented RealSense timestamp domains."""

    if domain == rs_module.timestamp_domain.hardware_clock:
        return pair_protocol.TIMESTAMP_DOMAIN_HARDWARE_CLOCK
    if domain == rs_module.timestamp_domain.system_time:
        return pair_protocol.TIMESTAMP_DOMAIN_SYSTEM_TIME
    if domain == rs_module.timestamp_domain.global_time:
        return pair_protocol.TIMESTAMP_DOMAIN_GLOBAL_TIME
    raise G1CaptureError(f"unsupported RealSense timestamp domain {domain!r}")


def frame_observation_from_realsense(frame: Any, *, side: str, rs_module: Any) -> FrameObservation:
    """Copy one Y8 frame while retaining its true device/global timestamp and metadata."""

    if side not in {"left", "right"}:
        raise G1CaptureError(f"frame side must be left or right, got {side!r}")
    if frame.profile.format() != rs_module.format.y8:
        raise G1CaptureError(f"{side} frame must be Y8, got {frame.profile.format()!r}")
    width = frame.get_width()
    height = frame.get_height()
    stride_bytes = frame.get_stride_in_bytes()
    bits_per_pixel = frame.get_bits_per_pixel()
    if bits_per_pixel != 8:
        raise G1CaptureError(f"{side} Y8 frame must have 8 bits per pixel, got {bits_per_pixel}")
    if stride_bytes != width:
        raise G1CaptureError(f"{side} Y8 stride must equal width, got stride={stride_bytes} width={width}")
    timestamp_ms = frame.get_timestamp()
    if not isinstance(timestamp_ms, (int, float)) or not math.isfinite(timestamp_ms) or timestamp_ms <= 0.0:
        raise G1CaptureError(f"{side} frame has invalid SDK timestamp {timestamp_ms!r}")
    timestamp_ns = round(timestamp_ms * 1_000_000.0)
    domain = timestamp_domain_to_protocol(frame.get_frame_timestamp_domain(), rs_module)

    metadata: Dict[str, Any] = {
        "stride_bytes": stride_bytes,
        "bits_per_pixel": bits_per_pixel,
    }
    for name in METADATA_FIELDS:
        metadata_key = getattr(rs_module.frame_metadata_value, name)
        if frame.supports_frame_metadata(metadata_key):
            metadata[name] = frame.get_frame_metadata(metadata_key)

    return FrameObservation(
        side=side,
        frame_number=frame.get_frame_number(),
        width=width,
        height=height,
        pixel_format=pair_protocol.PIXEL_FORMAT_Y8,
        timestamp_domain=domain,
        timestamp_ns=timestamp_ns,
        payload=bytes(frame.get_data()),
        metadata=metadata,
    )


def native_depth_record_from_realsense(
    frame: Any,
    *,
    pair_header: pair_protocol.StereoPairHeader,
    rs_module: Any,
) -> native_depth_recorder.NativeDepthRecord:
    """Copy one Z16 frame only when it is truly paired with the IR observation."""

    if frame.profile.format() != rs_module.format.z16:
        raise G1CaptureError(f"native depth frame must be Z16, got {frame.profile.format()!r}")
    width = frame.get_width()
    height = frame.get_height()
    stride_bytes = frame.get_stride_in_bytes()
    bits_per_pixel = frame.get_bits_per_pixel()
    if bits_per_pixel != 16:
        raise G1CaptureError(
            f"native Z16 frame must have 16 bits per pixel, got {bits_per_pixel}"
        )
    if stride_bytes != width * 2:
        raise G1CaptureError(
            f"native Z16 stride must equal twice the width, got stride={stride_bytes} width={width}"
        )
    if (width, height) != (pair_header.width, pair_header.height):
        raise G1CaptureError(
            "native depth shape must match the paired infrared frame shape"
        )
    frame_number = frame.get_frame_number()
    if not isinstance(frame_number, int) or isinstance(frame_number, bool) or frame_number < 0:
        raise G1CaptureError(f"native depth frame number must be a non-negative integer, got {frame_number!r}")
    timestamp_ms = frame.get_timestamp()
    if not isinstance(timestamp_ms, (int, float)) or not math.isfinite(timestamp_ms) or timestamp_ms <= 0.0:
        raise G1CaptureError(f"native depth frame has invalid SDK timestamp {timestamp_ms!r}")
    timestamp_ns = round(timestamp_ms * 1_000_000.0)
    timestamp_domain = timestamp_domain_to_protocol(frame.get_frame_timestamp_domain(), rs_module)
    if timestamp_domain != pair_protocol.TIMESTAMP_DOMAIN_GLOBAL_TIME:
        raise G1CaptureError("native depth timestamp must use RealSense global time")
    timestamp_skew_ns = abs(timestamp_ns - pair_header.camera_timestamp_ns)
    if timestamp_skew_ns > native_depth_recorder.MAX_NATIVE_PAIR_SKEW_NS:
        raise G1CaptureError(
            f"native-to-infrared timestamp skew {timestamp_skew_ns} exceeds 0.5 ms"
        )
    payload = bytes(frame.get_data())
    expected_size = width * height * 2
    if len(payload) != expected_size:
        raise G1CaptureError(
            f"native Z16 payload size {len(payload)} does not match expected payload size {expected_size}"
        )

    metadata: Dict[str, Any] = {
        "stride_bytes": stride_bytes,
        "bits_per_pixel": bits_per_pixel,
    }
    for name in METADATA_FIELDS:
        metadata_key = getattr(rs_module.frame_metadata_value, name)
        if frame.supports_frame_metadata(metadata_key):
            metadata[name] = frame.get_frame_metadata(metadata_key)
    return native_depth_recorder.NativeDepthRecord(
        sequence=pair_header.sequence,
        pair_camera_frame_number=pair_header.camera_frame_number,
        pair_camera_timestamp_ns=pair_header.camera_timestamp_ns,
        frame_number=frame_number,
        timestamp_ns=timestamp_ns,
        timestamp_domain=timestamp_domain,
        height=height,
        width=width,
        payload=payload,
        metadata=metadata,
    )


def paired_recording_rate_passes(
    frame_count: int,
    elapsed_s: float,
    *,
    minimum_hz: float = 55.0,
) -> bool:
    """Return whether the paired IR1/IR2/Z16 recording reached the minimum frame rate."""

    if frame_count < 0:
        raise G1CaptureError("paired frame count cannot be negative")
    if not math.isfinite(elapsed_s) or elapsed_s <= 0.0:
        raise G1CaptureError("paired recording elapsed time must be positive and finite")
    if not math.isfinite(minimum_hz) or minimum_hz <= 0.0:
        raise G1CaptureError("minimum paired recording rate must be positive and finite")
    return frame_count / elapsed_s >= minimum_hz


def output_cadence_allows_frameset(
    left: Optional[FrameObservation],  # noqa: UP045 - G1 uses Python 3.8
    right: Optional[FrameObservation],  # noqa: UP045 - G1 uses Python 3.8
    selector: OutputCadenceSelector,
) -> bool:
    """Throttle only matched pairs so invalid framesets still reach validation."""

    if left is None or right is None or left.frame_number != right.frame_number:
        return True
    return selector.select()


def make_output_cadence_selector(
    *,
    source_rate_hz: float,
    output_rate_hz: Optional[float],  # noqa: UP045 - G1 uses Python 3.8
) -> tuple[OutputCadenceSelector, str]:
    """Build the selector while preserving whether native cadence was requested."""

    selection_mode = "native" if output_rate_hz is None else "fractional"
    target_rate_hz = source_rate_hz if output_rate_hz is None else output_rate_hz
    return (
        OutputCadenceSelector(
            source_rate_hz=source_rate_hz,
            target_rate_hz=target_rate_hz,
        ),
        selection_mode,
    )


def output_cadence_summary(
    *,
    selector: OutputCadenceSelector,
    selection_mode: str,
    source_framesets_received: int,
    pairs_sent: int,
    transport_drops: int,
) -> Dict[str, Any]:
    """Render the acquisition, selection, and transport counts without ambiguity."""

    if selection_mode not in {"native", "fractional"}:
        raise G1CaptureError(f"unsupported selection mode: {selection_mode}")
    counters = (
        source_framesets_received,
        pairs_sent,
        transport_drops,
    )
    if any(value < 0 for value in counters):
        raise G1CaptureError("output cadence counters cannot be negative")
    return {
        "selection_mode": selection_mode,
        "source_rate_hz": selector.source_rate_hz,
        "target_rate_hz": selector.target_rate_hz,
        "source_framesets_received": source_framesets_received,
        "pairs_evaluated": selector.source_frames_seen,
        "pairs_selected": selector.output_frames_selected,
        "pairs_skipped": selector.frames_skipped,
        "pairs_sent": pairs_sent,
        "transport_drops": transport_drops,
    }


def native_depth_profile_from_realsense_device(
    device: Any,
    rs_module: Any,
    *,
    expected_calibration: Optional[calibration.CalibrationRecord] = None,  # noqa: UP045
) -> Dict[str, Any]:
    """Discover the exact Z16 profile and its projection into the left IR frame."""

    if expected_calibration is None:
        width = calibration.PROFILE_WIDTH
        height = calibration.PROFILE_HEIGHT
        fps = calibration.PROFILE_FPS
        left_stream_index = calibration.LEFT_STREAM_INDEX
    else:
        calibration.validate_calibration(expected_calibration)
        width = expected_calibration.width
        height = expected_calibration.height
        fps = expected_calibration.fps
        left_stream_index = expected_calibration.left_stream_index

    depth_profiles = []
    left_profiles = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            try:
                video = profile.as_video_stream_profile()
            except (AttributeError, RuntimeError):
                continue
            if (
                profile.fps() != fps
                or video.width() != width
                or video.height() != height
            ):
                continue
            if profile.stream_type() == rs_module.stream.depth and profile.format() == rs_module.format.z16:
                depth_profiles.append(profile)
            elif (
                profile.stream_type() == rs_module.stream.infrared
                and profile.format() == rs_module.format.y8
                and profile.stream_index() == left_stream_index
            ):
                left_profiles.append(profile)
    if len(depth_profiles) != 1 or len(left_profiles) != 1:
        raise G1CaptureError(
            f"paired recording requires exactly one {width}x{height}@{fps}Hz "
            "Z16 depth profile "
            "and exactly one matching left Y8 profile; "
            f"found depth={len(depth_profiles)} left={len(left_profiles)}"
        )

    depth_profile = depth_profiles[0]
    left_profile = left_profiles[0]
    intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
    extrinsics = depth_profile.get_extrinsics_to(left_profile)
    distortion_model = str(intrinsics.model)
    if distortion_model.startswith("distortion."):
        distortion_model = distortion_model[len("distortion.") :]
    depth_scale_m = float(device.first_depth_sensor().get_depth_scale())
    if not math.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise G1CaptureError(f"native depth scale must be positive and finite, got {depth_scale_m}")
    return {
        "stream": "depth",
        "format": "z16",
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fps": depth_profile.fps(),
        "stream_index": depth_profile.stream_index(),
        "intrinsics": {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "model": distortion_model,
            "coeffs": list(intrinsics.coeffs),
        },
        "depth_to_left_ir": {
            "rotation": list(extrinsics.rotation),
            "translation_m": list(extrinsics.translation),
        },
        "depth_scale_m": depth_scale_m,
    }


def enable_global_time_and_read_options(device: Any, rs_module: Any) -> Dict[str, Any]:
    """Enable global time on the stereo module and snapshot control-relevant options."""

    stereo_sensor = None
    for sensor in device.query_sensors():
        if sensor.get_info(rs_module.camera_info.name) == "Stereo Module":
            stereo_sensor = sensor
            break
    if stereo_sensor is None:
        raise G1CaptureError("D435i Stereo Module sensor was not found")
    if not stereo_sensor.supports(rs_module.option.global_time_enabled):
        raise G1CaptureError("D435i Stereo Module does not support global time")
    stereo_sensor.set_option(rs_module.option.global_time_enabled, 1.0)

    result: Dict[str, Any] = {"sensor_name": "Stereo Module"}
    for name in (
        "global_time_enabled",
        "emitter_enabled",
        "laser_power",
        "enable_auto_exposure",
        "exposure",
        "gain",
    ):
        option = getattr(rs_module.option, name)
        if stereo_sensor.supports(option):
            result[name] = stereo_sensor.get_option(option)
    if result.get("global_time_enabled") != 1.0:
        raise G1CaptureError(f"global time did not remain enabled: {result!r}")
    return result


def _select_device(rs_module: Any, serial: str) -> tuple[Any, Any]:
    context = rs_module.context()
    devices = [
        device
        for device in context.query_devices()
        if device.get_info(rs_module.camera_info.serial_number) == serial
    ]
    if len(devices) != 1:
        raise G1CaptureError(f"expected exactly one D435i serial {serial}, found {len(devices)}")
    return context, devices[0]


def configure_capture_streams(
    config: Any,
    rs_module: Any,
    expected_calibration: calibration.CalibrationRecord,
    *,
    include_depth: bool,
) -> None:
    """Configure exactly the streams named by one reviewed calibration."""

    calibration.validate_calibration(expected_calibration)
    config.enable_device(expected_calibration.camera_serial)
    config.enable_stream(
        rs_module.stream.infrared,
        expected_calibration.left_stream_index,
        expected_calibration.width,
        expected_calibration.height,
        rs_module.format.y8,
        expected_calibration.fps,
    )
    config.enable_stream(
        rs_module.stream.infrared,
        expected_calibration.right_stream_index,
        expected_calibration.width,
        expected_calibration.height,
        rs_module.format.y8,
        expected_calibration.fps,
    )
    if include_depth:
        config.enable_stream(
            rs_module.stream.depth,
            expected_calibration.width,
            expected_calibration.height,
            rs_module.format.z16,
            expected_calibration.fps,
        )


def capture_core_from_calibration(
    expected_calibration: calibration.CalibrationRecord,
    calibration_sha256: bytes,
    jpeg_quality: Optional[int] = None,  # noqa: UP045 - G1 uses Python 3.8
) -> CaptureCore:
    """Build the HSSPAIR1 validator from the same reviewed camera profile."""

    calibration.validate_calibration(expected_calibration)
    return CaptureCore(
        width=expected_calibration.width,
        height=expected_calibration.height,
        expected_calibration_sha256=calibration_sha256,
        camera_serial=expected_calibration.camera_serial,
        jpeg_quality=jpeg_quality,
    )


def _metadata_record(
    pair: Any,
    sent: bool,
    native_record: Optional[native_depth_recorder.NativeDepthRecord] = None,  # noqa: UP045
) -> Dict[str, Any]:
    header = asdict(pair.header)
    header["camera_serial_sha256"] = pair.header.camera_serial_sha256.hex()
    header["calibration_sha256"] = pair.header.calibration_sha256.hex()
    result = {
        "header": header,
        "left_metadata": dict(pair.left_metadata),
        "right_metadata": dict(pair.right_metadata),
        "transport_sent": sent,
    }
    if native_record is not None:
        native = asdict(native_record)
        native.pop("payload")
        native["metadata"] = dict(native_record.metadata)
        result["native_depth"] = native
    return result


def write_local_paired_evidence(
    stereo_writer: Any,
    native_writer: Any,
    pair: Any,
    native_record: native_depth_recorder.NativeDepthRecord,
    *,
    record_realtime_ns: int,
    record_monotonic_ns: int,
) -> None:
    """Write the same-frameset IR pair and native depth locally, independent of best-effort transport."""

    if native_record.sequence != pair.header.sequence:
        raise G1CaptureError("native and infrared local records must share one pair sequence")
    if record_realtime_ns <= 0 or record_monotonic_ns <= 0:
        raise G1CaptureError("local paired-record timestamps must be positive")
    stereo_writer.write_pair(
        pair,
        receive_realtime_ns=record_realtime_ns,
        receive_monotonic_ns=record_monotonic_ns,
    )
    native_writer.write_frame(native_record)


def run_capture(
    *,
    rs_module: Any,
    expected_calibration: calibration.CalibrationRecord,
    endpoint: str,
    duration_s: float,
    warmup_frames: int,
    metadata_jsonl: Optional[Path],  # noqa: UP045 - G1 uses Python 3.8
    paired_record_dir: Optional[Path] = None,  # noqa: UP045 - G1 uses Python 3.8
    minimum_paired_hz: float = 55.0,
    output_rate_hz: Optional[float] = None,  # noqa: UP045 - G1 uses Python 3.8
    jpeg_quality: Optional[int] = None,  # noqa: UP045 - G1 uses Python 3.8
) -> Dict[str, Any]:
    """Run bounded stereo capture, optionally recording same-frameset native Z16."""

    if duration_s <= 0.0:
        raise G1CaptureError("capture duration must be positive")
    if warmup_frames < 0:
        raise G1CaptureError("warmup frame count cannot be negative")
    if not math.isfinite(minimum_paired_hz) or minimum_paired_hz <= 0.0:
        raise G1CaptureError("minimum paired recording rate must be positive and finite")
    try:
        output_selector, selection_mode = make_output_cadence_selector(
            source_rate_hz=float(expected_calibration.fps),
            output_rate_hz=output_rate_hz,
        )
    except ValueError as exc:
        raise G1CaptureError(f"invalid output cadence: {exc}") from exc
    if (
        paired_record_dir is not None
        and minimum_paired_hz > output_selector.target_rate_hz
    ):
        raise G1CaptureError(
            "minimum paired recording rate cannot exceed the selected output rate"
        )
    if paired_record_dir is not None:
        if paired_record_dir.exists() and any(paired_record_dir.iterdir()):
            raise G1CaptureError(f"paired recording directory is not empty: {paired_record_dir}")
        paired_record_dir.mkdir(parents=True, exist_ok=True)
    context, device = _select_device(rs_module, expected_calibration.camera_serial)
    discovered = calibration.calibration_from_realsense_device(
        device,
        rs_module,
        expected=expected_calibration,
    )
    calibration.require_exact_calibration_match(discovered, expected_calibration)
    camera_options = enable_global_time_and_read_options(device, rs_module)
    calibration_sha256 = bytes.fromhex(calibration.calibration_digest(discovered))
    native_profile = (
        native_depth_profile_from_realsense_device(
            device,
            rs_module,
            expected_calibration=expected_calibration,
        )
        if paired_record_dir is not None
        else None
    )

    config = rs_module.config()
    configure_capture_streams(
        config,
        rs_module,
        expected_calibration,
        include_depth=paired_record_dir is not None,
    )
    pipeline = rs_module.pipeline(context)
    core = capture_core_from_calibration(
        expected_calibration,
        calibration_sha256,
        jpeg_quality=jpeg_quality,
    )
    metadata_stream = None
    stereo_writer = None
    native_writer = None
    started = False
    start_monotonic_ns = 0
    paired_frames_recorded = 0
    source_framesets_received = 0
    paired_drops: Dict[str, int] = {}
    last_native_error = ""
    try:
        pipeline.start(config)
        started = True
        for _ in range(warmup_frames):
            pipeline.wait_for_frames(2_000)
        start_monotonic_ns = time.monotonic_ns()
        deadline_ns = start_monotonic_ns + round(duration_s * 1_000_000_000)
        if metadata_jsonl is not None:
            metadata_jsonl.parent.mkdir(parents=True, exist_ok=True)
            metadata_stream = metadata_jsonl.open("w", encoding="utf-8")
        if paired_record_dir is not None:
            stereo_writer = pair_recorder.PairChunkWriter(paired_record_dir / "stereo")
            native_writer = native_depth_recorder.NativeDepthChunkWriter(
                paired_record_dir / "native",
                camera_serial_sha256=core.camera_serial_sha256.hex(),
                infrared_calibration_sha256=calibration_sha256.hex(),
                native_profile=native_profile,
            )
        with StereoPairPublisher(endpoint) as transport:
            while time.monotonic_ns() < deadline_ns:
                try:
                    frameset = pipeline.wait_for_frames(1_000)
                except RuntimeError as exc:
                    if "arrive" in str(exc).lower() or "timeout" in str(exc).lower():
                        core.process_frameset(
                            None,
                            None,
                            acquisition_monotonic_ns=time.monotonic_ns(),
                            publish_realtime_ns=time.time_ns(),
                            calibration_sha256=calibration_sha256,
                        )
                        continue
                    core.record_camera_disconnect(str(exc))
                    raise G1CaptureError(f"camera disconnected or failed: {exc}") from exc
                acquisition_monotonic_ns = time.monotonic_ns()
                source_framesets_received += 1
                left_frame = frameset.get_infrared_frame(
                    expected_calibration.left_stream_index
                )
                right_frame = frameset.get_infrared_frame(
                    expected_calibration.right_stream_index
                )
                native_frame = frameset.get_depth_frame() if paired_record_dir is not None else None
                left = (
                    frame_observation_from_realsense(left_frame, side="left", rs_module=rs_module)
                    if left_frame
                    else None
                )
                right = (
                    frame_observation_from_realsense(right_frame, side="right", rs_module=rs_module)
                    if right_frame
                    else None
                )
                if not output_cadence_allows_frameset(
                    left,
                    right,
                    output_selector,
                ):
                    continue
                pair = core.process_frameset(
                    left,
                    right,
                    acquisition_monotonic_ns=acquisition_monotonic_ns,
                    publish_realtime_ns=time.time_ns(),
                    calibration_sha256=calibration_sha256,
                )
                if pair is None:
                    continue
                native_record = None
                if paired_record_dir is not None:
                    if not native_frame:
                        paired_drops["missing_native_depth"] = paired_drops.get(
                            "missing_native_depth", 0
                        ) + 1
                        continue
                    try:
                        native_record = native_depth_record_from_realsense(
                            native_frame,
                            pair_header=pair.header,
                            rs_module=rs_module,
                        )
                    except G1CaptureError as exc:
                        paired_drops["invalid_native_depth"] = paired_drops.get(
                            "invalid_native_depth", 0
                        ) + 1
                        last_native_error = str(exc)
                        continue
                sent = transport.send_pair(pair.header, pair.left, pair.right)
                if paired_record_dir is not None:
                    if not sent:
                        paired_drops["transport_not_sent"] = paired_drops.get(
                            "transport_not_sent", 0
                        ) + 1
                    write_local_paired_evidence(
                        stereo_writer,
                        native_writer,
                        pair,
                        native_record,
                        record_realtime_ns=time.time_ns(),
                        record_monotonic_ns=time.monotonic_ns(),
                    )
                    paired_frames_recorded += 1
                if metadata_stream is not None:
                    metadata_stream.write(
                        json.dumps(
                            _metadata_record(pair, sent, native_record),
                            sort_keys=True,
                        )
                        + "\n"
                    )
            elapsed_s = (time.monotonic_ns() - start_monotonic_ns) / 1_000_000_000.0
            publisher_counters = asdict(transport.counters)
    finally:
        if native_writer is not None:
            native_writer.close()
        if stereo_writer is not None:
            stereo_writer.close()
        if metadata_stream is not None:
            metadata_stream.close()
        if started:
            active_exception = sys.exc_info()[0] is not None
            try:
                pipeline.stop()
            except RuntimeError:
                if not active_exception:
                    raise

    paired_enabled = paired_record_dir is not None
    paired_rate_hz = paired_frames_recorded / elapsed_s if paired_enabled else None
    paired_passed = (
        paired_recording_rate_passes(
            paired_frames_recorded,
            elapsed_s,
            minimum_hz=minimum_paired_hz,
        )
        if paired_enabled
        else None
    )
    return {
        "schema_version": 1,
        "camera_serial": discovered.camera_serial,
        "calibration_sha256": calibration.calibration_digest(discovered),
        "camera_options": camera_options,
        "warmup_frames": warmup_frames,
        "requested_duration_s": duration_s,
        "elapsed_s": elapsed_s,
        "capture": asdict(core.counters),
        "output_cadence": output_cadence_summary(
            selector=output_selector,
            selection_mode=selection_mode,
            source_framesets_received=source_framesets_received,
            pairs_sent=publisher_counters["sent"],
            transport_drops=publisher_counters["send_dropped"],
        ),
        "publisher": publisher_counters,
        "paired_recording": {
            "enabled": paired_enabled,
            "output_dir": str(paired_record_dir) if paired_enabled else None,
            "native_profile": native_profile,
            "frames_recorded": paired_frames_recorded,
            "recorded_hz": paired_rate_hz,
            "minimum_hz": minimum_paired_hz,
            "passed": paired_passed,
            "drops": paired_drops,
            "last_native_error": last_native_error,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: UP045 - G1 uses Python 3.8
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_BIND_ENDPOINT)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--warmup-frames", type=int, default=240)
    parser.add_argument("--metadata-jsonl", type=Path)
    parser.add_argument(
        "--paired-record-dir",
        type=Path,
        help="record exact same-frameset IR1, IR2, and native Z16 under this new directory",
    )
    parser.add_argument("--minimum-paired-hz", type=float, default=55.0)
    parser.add_argument(
        "--output-rate-hz",
        type=float,
        help="publish a deterministic fractional subset at this rate; default is the camera rate",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        help="JPEG-encode each Y8 image at this quality for a slow link; default raw",
    )
    args = parser.parse_args(argv)
    try:
        import pyrealsense2 as rs  # noqa: PLC0415

        expected = calibration.load_calibration(args.calibration)
        result = run_capture(
            rs_module=rs,
            expected_calibration=expected,
            endpoint=args.endpoint,
            duration_s=args.duration_s,
            warmup_frames=args.warmup_frames,
            metadata_jsonl=args.metadata_jsonl,
            paired_record_dir=args.paired_record_dir,
            minimum_paired_hz=args.minimum_paired_hz,
            output_rate_hz=args.output_rate_hz,
            jpeg_quality=args.jpeg_quality,
        )
    except (
        calibration.CalibrationError,
        G1CaptureError,
        native_depth_recorder.NativeDepthRecorderError,
        pair_recorder.PairRecorderError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"G1 fast-stereo capture failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    paired_passed = result["paired_recording"]["passed"]
    return 0 if result["capture"]["pairs_valid"] > 0 and paired_passed is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())

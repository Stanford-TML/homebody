"""Canonical calibration contract for the G1's D435i stereo pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

AUDITED_CAMERA_NAME = "Intel RealSense D435I"
DEFAULT_CAMERA_SERIAL = "337322071497"
PROFILE_WIDTH = 640
PROFILE_HEIGHT = 480
PROFILE_FPS = 60
PROFILE_PIXEL_FORMAT = "Y8"
LEFT_STREAM_INDEX = 1
RIGHT_STREAM_INDEX = 2
APPROVED_PROFILES = (
    (640, 480, 60, "Y8"),
    (848, 480, 60, "Y8"),
)


class CalibrationError(ValueError):
    """Raised when calibration cannot satisfy the frozen stereo contract."""


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole parameters for one rectified infrared stream."""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    distortion_model: str
    distortion_coefficients: Tuple[float, ...]


@dataclass(frozen=True)
class CalibrationRecord:
    """All calibration and profile fields that affect metric stereo depth."""

    schema_version: int
    camera_name: str
    camera_serial: str
    firmware_version: str
    width: int
    height: int
    fps: int
    pixel_format: str
    left_stream_index: int
    right_stream_index: int
    left_intrinsics: CameraIntrinsics
    right_intrinsics: CameraIntrinsics
    left_to_right_rotation: Tuple[float, ...]
    left_to_right_translation_m: Tuple[float, ...]


def _finite(values: Tuple[float, ...]) -> bool:
    return all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)


def _validate_intrinsics(name: str, intrinsics: CameraIntrinsics, record: CalibrationRecord) -> None:
    if (intrinsics.width, intrinsics.height) != (record.width, record.height):
        raise CalibrationError(
            f"{name} intrinsics dimensions {(intrinsics.width, intrinsics.height)} "
            f"do not match profile {(record.width, record.height)}"
        )
    values = (intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
    if not _finite(values) or intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise CalibrationError(f"{name} intrinsics must contain finite positive focal lengths")
    if not 0.0 <= intrinsics.ppx < intrinsics.width or not 0.0 <= intrinsics.ppy < intrinsics.height:
        raise CalibrationError(f"{name} intrinsics principal point is outside the image")
    if intrinsics.distortion_model != "brown_conrady":
        raise CalibrationError(f"{name} distortion model must be brown_conrady")
    if len(intrinsics.distortion_coefficients) != 5 or not _finite(intrinsics.distortion_coefficients):
        raise CalibrationError(f"{name} distortion coefficients must be five finite values")
    if any(abs(value) > 1e-9 for value in intrinsics.distortion_coefficients):
        raise CalibrationError(f"{name} distortion must be zero for the rectified stereo contract")


def _determinant_3x3(rotation: Tuple[float, ...]) -> float:
    return (
        rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7])
        - rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6])
        + rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6])
    )


def _validate_rotation(rotation: Tuple[float, ...]) -> None:
    if len(rotation) != 9 or not _finite(rotation):
        raise CalibrationError("left-to-right rotation must contain nine finite values")
    for row in range(3):
        for other_row in range(3):
            dot = sum(rotation[row * 3 + col] * rotation[other_row * 3 + col] for col in range(3))
            target = 1.0 if row == other_row else 0.0
            if abs(dot - target) > 1e-5:
                raise CalibrationError("left-to-right rotation is not orthonormal")
    if abs(_determinant_3x3(rotation) - 1.0) > 1e-5:
        raise CalibrationError("left-to-right rotation must have determinant +1")


def validate_calibration(
    record: CalibrationRecord,
    *,
    expected_serial: Optional[str] = None,  # noqa: UP045 - G1 uses Python 3.8
) -> None:
    """Reject records that differ from the approved rectified stereo contract."""

    if record.schema_version != 1:
        raise CalibrationError(f"unsupported calibration schema {record.schema_version}")
    if record.camera_name != AUDITED_CAMERA_NAME:
        raise CalibrationError(f"unexpected camera name {record.camera_name!r}")
    if not record.camera_serial.isdigit() or len(record.camera_serial) != 12:
        raise CalibrationError(
            f"camera serial must be a 12-digit RealSense serial, got {record.camera_serial!r}"
        )
    if expected_serial is not None and record.camera_serial != expected_serial:
        raise CalibrationError(
            f"camera serial {record.camera_serial!r} does not match expected serial {expected_serial!r}"
        )
    if not record.firmware_version:
        raise CalibrationError("firmware version is required")
    profile = (record.width, record.height, record.fps, record.pixel_format)
    if profile not in APPROVED_PROFILES:
        raise CalibrationError(
            f"stereo profile {profile!r} is not an approved profile "
            f"{APPROVED_PROFILES!r}"
        )
    if (record.left_stream_index, record.right_stream_index) != (LEFT_STREAM_INDEX, RIGHT_STREAM_INDEX):
        raise CalibrationError(
            "stereo stream indices must be left=1 and right=2, "
            f"got left={record.left_stream_index} right={record.right_stream_index}"
        )
    _validate_intrinsics("left", record.left_intrinsics, record)
    _validate_intrinsics("right", record.right_intrinsics, record)
    _validate_rotation(record.left_to_right_rotation)
    translation = record.left_to_right_translation_m
    if len(translation) != 3 or not _finite(translation):
        raise CalibrationError("left-to-right translation must contain three finite values")
    x, y, z = translation
    if not -0.06 < x < -0.04:
        raise CalibrationError(f"left-to-right baseline must be -x and near 0.05 m, got x={x}")
    if abs(y) > 0.002 or abs(z) > 0.002:
        raise CalibrationError(f"left-to-right translation must lie on the x axis, got {(x, y, z)!r}")


def stereo_baseline_m(record: CalibrationRecord) -> float:
    """Return the validated positive baseline used by depth = fx * baseline / disparity."""

    validate_calibration(record)
    return -record.left_to_right_translation_m[0]


def calibration_to_dict(record: CalibrationRecord) -> Dict[str, Any]:
    """Convert a record to the only supported JSON representation."""

    result = asdict(record)
    result["left_intrinsics"]["distortion_coefficients"] = list(
        record.left_intrinsics.distortion_coefficients
    )
    result["right_intrinsics"]["distortion_coefficients"] = list(
        record.right_intrinsics.distortion_coefficients
    )
    result["left_to_right_rotation"] = list(record.left_to_right_rotation)
    result["left_to_right_translation_m"] = list(record.left_to_right_translation_m)
    return result


def _intrinsics_from_dict(data: Mapping[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=data["width"],
        height=data["height"],
        fx=data["fx"],
        fy=data["fy"],
        ppx=data["ppx"],
        ppy=data["ppy"],
        distortion_model=data["distortion_model"],
        distortion_coefficients=tuple(data["distortion_coefficients"]),
    )


def calibration_from_dict(data: Mapping[str, Any]) -> CalibrationRecord:
    """Build and validate a record without supplying defaults for missing fields."""

    try:
        record = CalibrationRecord(
            schema_version=data["schema_version"],
            camera_name=data["camera_name"],
            camera_serial=data["camera_serial"],
            firmware_version=data["firmware_version"],
            width=data["width"],
            height=data["height"],
            fps=data["fps"],
            pixel_format=data["pixel_format"],
            left_stream_index=data["left_stream_index"],
            right_stream_index=data["right_stream_index"],
            left_intrinsics=_intrinsics_from_dict(data["left_intrinsics"]),
            right_intrinsics=_intrinsics_from_dict(data["right_intrinsics"]),
            left_to_right_rotation=tuple(data["left_to_right_rotation"]),
            left_to_right_translation_m=tuple(data["left_to_right_translation_m"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"missing calibration field or invalid structure: {exc}") from exc
    validate_calibration(record)
    return record


def canonical_json_bytes(record: CalibrationRecord) -> bytes:
    """Return deterministic UTF-8 JSON bytes used as the packet calibration identity."""

    validate_calibration(record)
    try:
        text = json.dumps(
            calibration_to_dict(record),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"calibration is not canonically serializable: {exc}") from exc
    return (text + "\n").encode("utf-8")


def calibration_digest(record: CalibrationRecord) -> str:
    """Return the lowercase SHA-256 hex digest of the canonical record."""

    return hashlib.sha256(canonical_json_bytes(record)).hexdigest()


def load_calibration(path: Path) -> CalibrationRecord:
    """Load and validate one calibration JSON file."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot load calibration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError("calibration JSON root must be an object")
    return calibration_from_dict(data)


def _realsense_intrinsics(profile: Any) -> CameraIntrinsics:
    intrinsics = profile.as_video_stream_profile().get_intrinsics()
    distortion_model = str(intrinsics.model)
    if distortion_model.startswith("distortion."):
        distortion_model = distortion_model[len("distortion.") :]
    return CameraIntrinsics(
        width=intrinsics.width,
        height=intrinsics.height,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        ppx=intrinsics.ppx,
        ppy=intrinsics.ppy,
        distortion_model=distortion_model,
        distortion_coefficients=tuple(intrinsics.coeffs),
    )


def calibration_from_realsense_device(
    device: Any,
    rs_module: Any,
    *,
    expected: Optional[CalibrationRecord] = None,  # noqa: UP045 - G1 uses Python 3.8
) -> CalibrationRecord:
    """Read one exact approved IR profile from an already selected device."""

    camera_name = device.get_info(rs_module.camera_info.name)
    camera_serial = device.get_info(rs_module.camera_info.serial_number)
    firmware_version = device.get_info(rs_module.camera_info.firmware_version)
    expected_camera_serial = (
        expected.camera_serial if expected is not None else DEFAULT_CAMERA_SERIAL
    )
    if camera_serial != expected_camera_serial:
        raise CalibrationError(
            f"camera serial {camera_serial!r} does not match expected serial "
            f"{expected_camera_serial!r}"
        )
    if expected is None:
        width = PROFILE_WIDTH
        height = PROFILE_HEIGHT
        fps = PROFILE_FPS
        pixel_format = PROFILE_PIXEL_FORMAT
        left_stream_index = LEFT_STREAM_INDEX
        right_stream_index = RIGHT_STREAM_INDEX
    else:
        validate_calibration(expected, expected_serial=expected_camera_serial)
        width = expected.width
        height = expected.height
        fps = expected.fps
        pixel_format = expected.pixel_format
        left_stream_index = expected.left_stream_index
        right_stream_index = expected.right_stream_index

    matching_profiles: Dict[int, Any] = {}
    duplicate_indices = set()
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            try:
                video = profile.as_video_stream_profile()
            except (AttributeError, RuntimeError):
                continue
            if (
                profile.stream_type() != rs_module.stream.infrared
                or profile.format() != rs_module.format.y8
                or profile.fps() != fps
                or video.width() != width
                or video.height() != height
                or profile.stream_index()
                not in (left_stream_index, right_stream_index)
            ):
                continue
            index = profile.stream_index()
            if index in matching_profiles:
                duplicate_indices.add(index)
            matching_profiles[index] = profile

    expected_indices = {left_stream_index, right_stream_index}
    if set(matching_profiles) != expected_indices or duplicate_indices:
        raise CalibrationError(
            "device must expose exactly one left and one right "
            f"{width}x{height}@{fps}Hz {pixel_format} profile; "
            f"found indices={sorted(matching_profiles)} duplicates={sorted(duplicate_indices)}"
        )

    left_profile = matching_profiles[left_stream_index]
    right_profile = matching_profiles[right_stream_index]
    extrinsics = left_profile.get_extrinsics_to(right_profile)
    record = CalibrationRecord(
        schema_version=1,
        camera_name=camera_name,
        camera_serial=camera_serial,
        firmware_version=firmware_version,
        width=width,
        height=height,
        fps=fps,
        pixel_format=pixel_format,
        left_stream_index=left_stream_index,
        right_stream_index=right_stream_index,
        left_intrinsics=_realsense_intrinsics(left_profile),
        right_intrinsics=_realsense_intrinsics(right_profile),
        left_to_right_rotation=tuple(extrinsics.rotation),
        left_to_right_translation_m=tuple(extrinsics.translation),
    )
    validate_calibration(record, expected_serial=expected_camera_serial)
    return record


def discover_realsense_calibration(
    rs_module: Any,
    *,
    expected: Optional[CalibrationRecord] = None,  # noqa: UP045 - G1 uses Python 3.8
) -> CalibrationRecord:
    """Select the camera named by the expected record and read its stereo calibration."""

    expected_camera_serial = (
        expected.camera_serial if expected is not None else DEFAULT_CAMERA_SERIAL
    )
    devices = [
        device
        for device in rs_module.context().query_devices()
        if device.get_info(rs_module.camera_info.serial_number) == expected_camera_serial
    ]
    if len(devices) != 1:
        raise CalibrationError(
            f"expected exactly one RealSense with serial {expected_camera_serial}, "
            f"found {len(devices)}"
        )
    return calibration_from_realsense_device(
        devices[0],
        rs_module,
        expected=expected,
    )


def require_exact_calibration_match(discovered: CalibrationRecord, expected: CalibrationRecord) -> None:
    """Reject any field drift between live discovery and the reviewed calibration file."""

    if canonical_json_bytes(discovered) != canonical_json_bytes(expected):
        raise CalibrationError(
            "discovered calibration does not match expected calibration: "
            f"discovered={calibration_digest(discovered)} expected={calibration_digest(expected)}"
        )


def main(
    argv: Optional[Sequence[str]] = None,  # noqa: UP045 - G1 uses Python 3.8
    *,
    rs_module: Any = None,
) -> int:
    """Discover, verify, and print the live camera's canonical calibration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True, help="reviewed calibration JSON")
    args = parser.parse_args(argv)

    try:
        expected = load_calibration(args.expected)
        if rs_module is None:
            try:
                import pyrealsense2 as rs_module  # type: ignore[no-redef]  # noqa: PLC0415
            except ImportError as exc:
                raise CalibrationError(f"pyrealsense2 is unavailable: {exc}") from exc
        discovered = discover_realsense_calibration(rs_module, expected=expected)
        require_exact_calibration_match(discovered, expected)
    except CalibrationError as exc:
        print(f"calibration discovery failed: {exc}", file=sys.stderr)
        return 2

    print(canonical_json_bytes(discovered).decode("utf-8"), end="")
    print(f"calibration_sha256={calibration_digest(discovered)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

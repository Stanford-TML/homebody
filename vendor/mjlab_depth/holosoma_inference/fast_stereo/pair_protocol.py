# Modified for HomeBody; the changes are recorded in vendor/mjlab_depth/CLOSURE.json.
"""Version-one wire contract for one atomic rectified Y8 stereo pair."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Sequence, Union

MAGIC = b"HSSPAIR1"
VERSION = 1
PIXEL_FORMAT_Y8 = 1
# JPEG-encoded Y8 for bandwidth-constrained links; payload sizes are bounded by height * width.
PIXEL_FORMAT_JPEG = 2

TIMESTAMP_DOMAIN_UNKNOWN = 0
TIMESTAMP_DOMAIN_HARDWARE_CLOCK = 1
TIMESTAMP_DOMAIN_SYSTEM_TIME = 2
TIMESTAMP_DOMAIN_GLOBAL_TIME = 3

_SUPPORTED_TIMESTAMP_DOMAINS = {
    TIMESTAMP_DOMAIN_UNKNOWN,
    TIMESTAMP_DOMAIN_HARDWARE_CLOCK,
    TIMESTAMP_DOMAIN_SYSTEM_TIME,
    TIMESTAMP_DOMAIN_GLOBAL_TIME,
}
_HEADER = struct.Struct("<8sHHIQQIIHHQQQQQQQII32s32s20x")
HEADER_SIZE = _HEADER.size
_RESERVED_OFFSET = 172
_UINT32_MAX = (1 << 32) - 1

BytesLike = Union[bytes, bytearray, memoryview]


class ProtocolError(ValueError):
    """Raised when a stereo packet violates the frozen wire contract."""


@dataclass(frozen=True)
class StereoPairHeader:
    """Decoded fields from an `HSSPAIR1` fixed-size header."""

    flags: int
    sequence: int
    camera_frame_number: int
    height: int
    width: int
    pixel_format: int
    timestamp_domain: int
    camera_timestamp_ns: int
    left_sensor_timestamp_ns: int
    right_sensor_timestamp_ns: int
    capture_realtime_ns: int
    acquisition_monotonic_ns: int
    publish_realtime_ns: int
    pair_skew_ns: int
    left_payload_size: int
    right_payload_size: int
    camera_serial_sha256: bytes
    calibration_sha256: bytes


@dataclass(frozen=True)
class DecodedStereoPair:
    """A validated header and byte views for one complete stereo pair."""

    header: StereoPairHeader
    left: memoryview
    right: memoryview


def _require_unsigned(name: str, value: int, bits: int, *, allow_zero: bool = True) -> None:
    lower = 0 if allow_zero else 1
    upper = (1 << bits) - 1
    if not isinstance(value, int) or not lower <= value <= upper:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ProtocolError(f"{name} must be a {qualifier} uint{bits}, got {value!r}")


def _as_byte_view(name: str, payload: BytesLike) -> memoryview:
    try:
        view = memoryview(payload)
    except TypeError as exc:
        raise ProtocolError(f"{name} payload must expose a contiguous byte buffer") from exc
    if not view.c_contiguous:
        raise ProtocolError(f"{name} payload must be C-contiguous")
    try:
        return view.cast("B")
    except TypeError as exc:
        raise ProtocolError(f"{name} payload cannot be viewed as bytes") from exc


def _validate_header(header: StereoPairHeader) -> None:
    _require_unsigned("flags", header.flags, 32)
    _require_unsigned("sequence", header.sequence, 64)
    _require_unsigned("camera_frame_number", header.camera_frame_number, 64)
    if (
        not isinstance(header.height, int)
        or isinstance(header.height, bool)
        or not isinstance(header.width, int)
        or isinstance(header.width, bool)
        or header.height <= 0
        or header.width <= 0
        or header.height > _UINT32_MAX
        or header.width > _UINT32_MAX
    ):
        raise ProtocolError(
            "image dimensions must be positive uint32 values, "
            f"got height={header.height!r} width={header.width!r}"
        )
    if header.pixel_format not in (PIXEL_FORMAT_Y8, PIXEL_FORMAT_JPEG):
        raise ProtocolError(f"unsupported pixel format {header.pixel_format}")
    if header.timestamp_domain not in _SUPPORTED_TIMESTAMP_DOMAINS:
        raise ProtocolError(f"unsupported timestamp domain {header.timestamp_domain}")
    _require_unsigned("camera_timestamp_ns", header.camera_timestamp_ns, 64)
    _require_unsigned("left_sensor_timestamp_ns", header.left_sensor_timestamp_ns, 64)
    _require_unsigned("right_sensor_timestamp_ns", header.right_sensor_timestamp_ns, 64)
    _require_unsigned("capture_realtime_ns", header.capture_realtime_ns, 64)
    _require_unsigned("acquisition_monotonic_ns", header.acquisition_monotonic_ns, 64, allow_zero=False)
    _require_unsigned("publish_realtime_ns", header.publish_realtime_ns, 64, allow_zero=False)
    _require_unsigned("pair_skew_ns", header.pair_skew_ns, 64)
    _require_unsigned("left_payload_size", header.left_payload_size, 32, allow_zero=False)
    _require_unsigned("right_payload_size", header.right_payload_size, 32, allow_zero=False)
    if len(header.camera_serial_sha256) != 32:
        raise ProtocolError("camera_serial_sha256 must be exactly 32 bytes")
    if len(header.calibration_sha256) != 32:
        raise ProtocolError("calibration_sha256 must be exactly 32 bytes")
    expected_payload_size = header.height * header.width
    if header.pixel_format == PIXEL_FORMAT_JPEG:
        for name, size in (("left", header.left_payload_size), ("right", header.right_payload_size)):
            if size > expected_payload_size:
                raise ProtocolError(
                    f"{name} JPEG payload size {size} exceeds the raw image size {expected_payload_size}"
                )
        return
    left_size_invalid = header.left_payload_size != expected_payload_size
    right_size_invalid = header.right_payload_size != expected_payload_size
    if left_size_invalid and right_size_invalid:
        raise ProtocolError(
            "Y8 payload size must equal height * width for both images "
            f"({expected_payload_size}), got left={header.left_payload_size} "
            f"right={header.right_payload_size}"
        )
    if left_size_invalid:
        raise ProtocolError(
            "left payload size must equal height * width for Y8 "
            f"({expected_payload_size}), got {header.left_payload_size}"
        )
    if right_size_invalid:
        raise ProtocolError(
            "right payload size must equal height * width for Y8 "
            f"({expected_payload_size}), got {header.right_payload_size}"
        )


def encode_header(header: StereoPairHeader) -> bytes:
    """Validate and serialize one fixed-size little-endian header."""

    _validate_header(header)
    return _HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_SIZE,
        header.flags,
        header.sequence,
        header.camera_frame_number,
        header.height,
        header.width,
        header.pixel_format,
        header.timestamp_domain,
        header.camera_timestamp_ns,
        header.left_sensor_timestamp_ns,
        header.right_sensor_timestamp_ns,
        header.capture_realtime_ns,
        header.acquisition_monotonic_ns,
        header.publish_realtime_ns,
        header.pair_skew_ns,
        header.left_payload_size,
        header.right_payload_size,
        header.camera_serial_sha256,
        header.calibration_sha256,
    )


def decode_header(data: BytesLike) -> StereoPairHeader:
    """Validate and decode one complete version-one header."""

    raw = _as_byte_view("header", data)
    if len(raw) != HEADER_SIZE:
        raise ProtocolError(f"header size must be {HEADER_SIZE} bytes, got {len(raw)}")
    if any(raw[_RESERVED_OFFSET:]):
        raise ProtocolError("reserved header bytes must be zero")
    (
        magic,
        version,
        header_size,
        flags,
        sequence,
        camera_frame_number,
        height,
        width,
        pixel_format,
        timestamp_domain,
        camera_timestamp_ns,
        left_sensor_timestamp_ns,
        right_sensor_timestamp_ns,
        capture_realtime_ns,
        acquisition_monotonic_ns,
        publish_realtime_ns,
        pair_skew_ns,
        left_payload_size,
        right_payload_size,
        camera_serial_sha256,
        calibration_sha256,
    ) = _HEADER.unpack(raw)
    if magic != MAGIC:
        raise ProtocolError(f"invalid stereo packet magic {magic!r}")
    if version != VERSION:
        raise ProtocolError(f"unsupported stereo packet version {version}")
    if header_size != HEADER_SIZE:
        raise ProtocolError(f"header size field must be {HEADER_SIZE}, got {header_size}")
    header = StereoPairHeader(
        flags=flags,
        sequence=sequence,
        camera_frame_number=camera_frame_number,
        height=height,
        width=width,
        pixel_format=pixel_format,
        timestamp_domain=timestamp_domain,
        camera_timestamp_ns=camera_timestamp_ns,
        left_sensor_timestamp_ns=left_sensor_timestamp_ns,
        right_sensor_timestamp_ns=right_sensor_timestamp_ns,
        capture_realtime_ns=capture_realtime_ns,
        acquisition_monotonic_ns=acquisition_monotonic_ns,
        publish_realtime_ns=publish_realtime_ns,
        pair_skew_ns=pair_skew_ns,
        left_payload_size=left_payload_size,
        right_payload_size=right_payload_size,
        camera_serial_sha256=camera_serial_sha256,
        calibration_sha256=calibration_sha256,
    )
    _validate_header(header)
    return header


def encode_multipart(
    header: StereoPairHeader,
    left: BytesLike,
    right: BytesLike,
) -> tuple[bytes, memoryview, memoryview]:
    """Return the exact three ZMQ parts for one validated atomic pair."""

    header_bytes = encode_header(header)
    left_view = _as_byte_view("left", left)
    right_view = _as_byte_view("right", right)
    if len(left_view) != header.left_payload_size:
        raise ProtocolError(
            f"left payload length is {len(left_view)}, header declares {header.left_payload_size}"
        )
    if len(right_view) != header.right_payload_size:
        raise ProtocolError(
            f"right payload length is {len(right_view)}, header declares {header.right_payload_size}"
        )
    return header_bytes, left_view, right_view


def decode_multipart(
    parts: Sequence[BytesLike],
    *,
    previous_sequence: Optional[int] = None,  # noqa: UP045 - wire module also runs on G1 Python 3.8
) -> DecodedStereoPair:
    """Validate a complete multipart message and optionally enforce sequence order."""

    if len(parts) != 3:
        raise ProtocolError(f"stereo packet must have exactly three parts, got {len(parts)}")
    header = decode_header(parts[0])
    if previous_sequence is not None and header.sequence <= previous_sequence:
        raise ProtocolError(
            f"sequence {header.sequence} must be strictly greater than previous sequence {previous_sequence}"
        )
    left = _as_byte_view("left", parts[1])
    right = _as_byte_view("right", parts[2])
    if len(left) != header.left_payload_size:
        raise ProtocolError(f"left payload length is {len(left)}, header declares {header.left_payload_size}")
    if len(right) != header.right_payload_size:
        raise ProtocolError(f"right payload length is {len(right)}, header declares {header.right_payload_size}")
    return DecodedStereoPair(header=header, left=left, right=right)

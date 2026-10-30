# Modified for HomeBody; the changes are recorded in vendor/mjlab_depth/CLOSURE.json.
"""Camera-independent validation from one D435i frameset to one HSSPAIR1 pair."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from holosoma_inference.fast_stereo import pair_protocol

MAX_PAIR_SKEW_NS = 500_000


@dataclass(frozen=True)
class FrameObservation:
    """SDK-neutral fields extracted from exactly one infrared frame."""

    side: str
    frame_number: int
    width: int
    height: int
    pixel_format: int
    timestamp_domain: int
    timestamp_ns: int
    payload: pair_protocol.BytesLike
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CapturedPair:
    """One validated, receiver-safe pair and its diagnostic metadata."""

    header: pair_protocol.StereoPairHeader
    left: bytes
    right: bytes
    left_metadata: Mapping[str, Any]
    right_metadata: Mapping[str, Any]


@dataclass
class CaptureCounters:
    framesets_seen: int = 0
    pairs_valid: int = 0
    camera_disconnects: int = 0
    last_disconnect_error: str = ""
    drops: Dict[str, int] = field(default_factory=dict)


@dataclass
class OutputCadenceSelector:
    """Select a deterministic fractional-rate subset without sleeping or buffering."""

    source_rate_hz: float
    target_rate_hz: float
    source_frames_seen: int = field(init=False, default=0)
    output_frames_selected: int = field(init=False, default=0)
    frames_skipped: int = field(init=False, default=0)
    _phase: float = field(init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.source_rate_hz) or self.source_rate_hz <= 0.0:
            raise ValueError("source rate must be positive and finite")
        if not math.isfinite(self.target_rate_hz) or self.target_rate_hz <= 0.0:
            raise ValueError("target rate must be positive and finite")
        if self.target_rate_hz > self.source_rate_hz:
            raise ValueError("target rate cannot exceed source rate")
        # Seed the accumulator so the first valid source pair is selected.
        self._phase = self.source_rate_hz - self.target_rate_hz

    def select(self) -> bool:
        """Return whether the next source frame belongs to the output cadence."""

        self.source_frames_seen += 1
        self._phase += self.target_rate_hz
        if self._phase >= self.source_rate_hz:
            self._phase -= self.source_rate_hz
            self.output_frames_selected += 1
            return True
        self.frames_skipped += 1
        return False


class CaptureCore:
    """Fail-closed validator with one unambiguous reason per dropped frameset."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        expected_calibration_sha256: bytes,
        camera_serial: str,
        maximum_pair_skew_ns: int = MAX_PAIR_SKEW_NS,
        jpeg_quality: Optional[int] = None,  # noqa: UP045 - G1 uses Python 3.8
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("capture dimensions must be positive")
        if len(expected_calibration_sha256) != 32:
            raise ValueError("expected calibration SHA-256 must contain 32 bytes")
        if maximum_pair_skew_ns < 0:
            raise ValueError("maximum pair skew must be non-negative")
        if jpeg_quality is not None and not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("JPEG quality must be within 1..100")
        self.width = width
        self.height = height
        self.expected_calibration_sha256 = expected_calibration_sha256
        self.camera_serial_sha256 = hashlib.sha256(camera_serial.encode("utf-8")).digest()
        self.maximum_pair_skew_ns = maximum_pair_skew_ns
        self.jpeg_quality = None if jpeg_quality is None else int(jpeg_quality)
        self.counters = CaptureCounters()
        self._next_sequence = 0
        self._last_camera_frame: Optional[int] = None  # noqa: UP045 - G1 uses Python 3.8
        self._active_timestamp_domain: Optional[int] = None  # noqa: UP045 - G1 uses Python 3.8

    def _drop(self, reason: str) -> None:
        self.counters.drops[reason] = self.counters.drops.get(reason, 0) + 1

    @staticmethod
    def _payload_bytes(frame: FrameObservation) -> Optional[bytes]:  # noqa: UP045 - G1 uses Python 3.8
        try:
            view = memoryview(frame.payload)
        except TypeError:
            return None
        if not view.c_contiguous:
            return None
        try:
            return view.cast("B").tobytes()
        except TypeError:
            return None

    def process_frameset(
        self,
        left: Optional[FrameObservation],  # noqa: UP045 - G1 uses Python 3.8
        right: Optional[FrameObservation],  # noqa: UP045 - G1 uses Python 3.8
        *,
        acquisition_monotonic_ns: int,
        publish_realtime_ns: int,
        calibration_sha256: bytes,
    ) -> Optional[CapturedPair]:  # noqa: UP045 - G1 uses Python 3.8
        """Return a complete pair or drop both members with exactly one reason."""

        self.counters.framesets_seen += 1
        if left is None and right is None:
            self._drop("frameset_timeout")
            return None

        sequence = self._next_sequence
        self._next_sequence += 1
        if left is None or right is None:
            self._drop("missing_pair_member")
            return None
        if calibration_sha256 != self.expected_calibration_sha256:
            self._drop("calibration_hash_mismatch")
            return None
        if (left.width, left.height) != (self.width, self.height) or (
            right.width,
            right.height,
        ) != (self.width, self.height):
            self._drop("unexpected_shape")
            return None
        if left.pixel_format != pair_protocol.PIXEL_FORMAT_Y8 or right.pixel_format != pair_protocol.PIXEL_FORMAT_Y8:
            self._drop("unexpected_format")
            return None
        left_bytes = self._payload_bytes(left)
        right_bytes = self._payload_bytes(right)
        expected_payload_size = self.width * self.height
        if (
            left_bytes is None
            or right_bytes is None
            or len(left_bytes) != expected_payload_size
            or len(right_bytes) != expected_payload_size
        ):
            self._drop("payload_size_mismatch")
            return None
        if left.frame_number != right.frame_number:
            self._drop("frame_number_mismatch")
            return None
        if self._last_camera_frame is not None and left.frame_number <= self._last_camera_frame:
            self._drop("nonmonotonic_camera_frame")
            return None
        if left.timestamp_domain != right.timestamp_domain:
            self._drop("timestamp_domain_mismatch")
            return None
        domain = left.timestamp_domain
        if self._active_timestamp_domain is not None and domain != self._active_timestamp_domain:
            self._drop("timestamp_domain_changed")
            return None
        if domain != pair_protocol.TIMESTAMP_DOMAIN_GLOBAL_TIME:
            self._drop("timestamp_domain_not_global")
            return None
        if left.timestamp_ns <= 0 or right.timestamp_ns <= 0:
            self._drop("invalid_frame_timestamp")
            return None
        pair_skew_ns = abs(left.timestamp_ns - right.timestamp_ns)
        if pair_skew_ns > self.maximum_pair_skew_ns:
            self._drop("pair_skew_exceeds_0.5ms")
            return None
        if acquisition_monotonic_ns <= 0 or publish_realtime_ns <= 0:
            self._drop("invalid_host_timestamp")
            return None

        pixel_format = pair_protocol.PIXEL_FORMAT_Y8
        if self.jpeg_quality is not None:
            left_bytes = self._jpeg(left_bytes)
            right_bytes = self._jpeg(right_bytes)
            pixel_format = pair_protocol.PIXEL_FORMAT_JPEG
        camera_timestamp_ns = (left.timestamp_ns + right.timestamp_ns) // 2
        header = pair_protocol.StereoPairHeader(
            flags=0,
            sequence=sequence,
            camera_frame_number=left.frame_number,
            height=self.height,
            width=self.width,
            pixel_format=pixel_format,
            timestamp_domain=domain,
            camera_timestamp_ns=camera_timestamp_ns,
            left_sensor_timestamp_ns=left.timestamp_ns,
            right_sensor_timestamp_ns=right.timestamp_ns,
            capture_realtime_ns=camera_timestamp_ns,
            acquisition_monotonic_ns=acquisition_monotonic_ns,
            publish_realtime_ns=publish_realtime_ns,
            pair_skew_ns=pair_skew_ns,
            left_payload_size=len(left_bytes),
            right_payload_size=len(right_bytes),
            camera_serial_sha256=self.camera_serial_sha256,
            calibration_sha256=calibration_sha256,
        )
        pair_protocol.encode_header(header)
        self._last_camera_frame = left.frame_number
        self._active_timestamp_domain = domain
        self.counters.pairs_valid += 1
        return CapturedPair(
            header=header,
            left=left_bytes,
            right=right_bytes,
            left_metadata=dict(left.metadata),
            right_metadata=dict(right.metadata),
        )

    def _jpeg(self, payload: bytes) -> bytes:
        """One Y8 image -> JPEG at self.jpeg_quality; ffs_stream_server decodes it."""
        import cv2  # noqa: PLC0415 - only an encoding capture needs it
        import numpy as np  # noqa: PLC0415

        image = np.frombuffer(payload, dtype=np.uint8).reshape(self.height, self.width)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding of a stereo image failed")
        return encoded.tobytes()

    def record_camera_disconnect(self, error: str) -> None:
        self.counters.camera_disconnects += 1
        self.counters.last_disconnect_error = error

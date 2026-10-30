# Modified for HomeBody; the changes are recorded in vendor/mjlab_depth/CLOSURE.json.
"""Latest-only HMDPTH01 metric-depth wire contract.

The relay receives the Fast-FoundationStereo ``HSDPTH01`` and ``FFSTEL01``
streams, joins them, and republishes one ``HMDPTH01`` packet with frame
identity, same-host monotonic completion time, producer-measured FFS latency,
and a C-order uint16 millimetre image (zero means invalid).
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import zmq

FFS_DEPTH_MAGIC = b"HSDPTH01"
METRIC_DEPTH_MAGIC = b"HMDPTH01"
PACKET_VERSION = 1
DTYPE_UINT16 = 1

_FFS_DEPTH_HEADER = struct.Struct("<8sBBHQQQIII")
_METRIC_DEPTH_HEADER = struct.Struct("<8sBBHQQQQIII")
_UINT64_MAX = (1 << 64) - 1
_UINT32_MAX = (1 << 32) - 1


@dataclass(frozen=True)
class FFSDepthPacket:
    """One validated packet emitted by the FFS stream server."""

    depth_mm: np.ndarray
    frame_id: int
    capture_time_ns: int
    publish_time_ns: int


@dataclass(frozen=True)
class MetricDepthFrame:
    """Newest metric-depth result owned by the caller."""

    depth_mm: np.ndarray
    frame_id: int
    camera_capture_time_ns: int
    completion_monotonic_ns: int
    ffs_latency_ns: int

    @property
    def ffs_latency_ms(self) -> float:
        return self.ffs_latency_ns / 1_000_000.0

    def result_age_ms(self, now_ns: int | None = None) -> float:
        """Age since the relay completed this FFS result, on the monotonic clock."""

        observed_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        return max(0, observed_ns - self.completion_monotonic_ns) / 1_000_000.0

    def capture_age_ms(self, now_ns: int | None = None) -> float:
        """Camera-capture-to-now age on the producer/consumer wall clock.

        A future capture timestamp means the wall clocks are not comparable; expose
        that as NaN instead of silently reporting a plausible zero age.
        """

        observed_ns = time.time_ns() if now_ns is None else int(now_ns)
        delta_ns = observed_ns - self.camera_capture_time_ns
        return float("nan") if delta_ns < 0 else delta_ns / 1_000_000.0


def _validate_uint(value: int, *, name: str, maximum: int, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    converted = int(value)
    lower = 1 if positive else 0
    if not lower <= converted <= maximum:
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must fit {qualifier}unsigned range")
    return converted


def _validate_shape(shape: tuple[int, int], *, name: str) -> tuple[int, int]:
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError(f"{name} must contain two dimensions")
    return (
        _validate_uint(shape[0], name=f"{name} height", maximum=_UINT32_MAX, positive=True),
        _validate_uint(shape[1], name=f"{name} width", maximum=_UINT32_MAX, positive=True),
    )


def _validate_depth(depth_mm: np.ndarray) -> tuple[int, int, bytes]:
    if not isinstance(depth_mm, np.ndarray):
        raise TypeError("depth must be a numpy array")
    if depth_mm.dtype != np.uint16:
        raise TypeError(f"depth must be uint16 millimetres, got {depth_mm.dtype}")
    if depth_mm.ndim != 2:
        raise ValueError(f"depth must be 2-D, got {depth_mm.shape}")
    if not depth_mm.flags.c_contiguous:
        raise ValueError("depth must be C-contiguous")
    height, width = _validate_shape(depth_mm.shape, name="depth shape")
    pixels = depth_mm.tobytes(order="C")
    if len(pixels) > _UINT32_MAX:
        raise ValueError("depth payload is too large")
    return height, width, pixels


def _decode_pixels(
    payload: bytes,
    *,
    header_size: int,
    payload_size: int,
    shape: tuple[int, int],
) -> np.ndarray:
    expected_size = shape[0] * shape[1] * np.dtype(np.uint16).itemsize
    if payload_size != expected_size:
        raise ValueError(f"depth payload is {payload_size} bytes, expected {expected_size}")
    if len(payload) != header_size + expected_size:
        raise ValueError(f"depth message is {len(payload)} bytes, expected {header_size + expected_size}")
    return np.frombuffer(payload, dtype=np.uint16, offset=header_size).reshape(shape).copy(order="C")


def pack_ffs_depth(
    depth_mm: np.ndarray,
    *,
    frame_id: int,
    capture_time_ns: int,
    publish_time_ns: int,
) -> bytes:
    """Pack one producer-side ``HSDPTH01`` packet (inverse of ``unpack_ffs_depth``).

    Used by the in-repo FFS stream server; kept next to the unpacker so the
    two sides of the wire cannot drift apart.
    """

    height, width, pixels = _validate_depth(depth_mm)
    frame_id = _validate_uint(frame_id, name="frame_id", maximum=_UINT64_MAX)
    capture_ns = _validate_uint(
        capture_time_ns, name="capture_time_ns", maximum=_UINT64_MAX, positive=True
    )
    publish_ns = _validate_uint(
        publish_time_ns, name="publish_time_ns", maximum=_UINT64_MAX, positive=True
    )
    if publish_ns < capture_ns:
        raise ValueError("publish_time_ns precedes capture_time_ns")
    return _FFS_DEPTH_HEADER.pack(
        FFS_DEPTH_MAGIC,
        PACKET_VERSION,
        DTYPE_UINT16,
        0,
        frame_id,
        capture_ns,
        publish_ns,
        height,
        width,
        len(pixels),
    ) + pixels


def unpack_ffs_depth(payload: bytes, *, expected_shape: tuple[int, int]) -> FFSDepthPacket:
    """Validate and decode one ``HSDPTH01`` packet from the FFS stream server."""

    expected = _validate_shape(expected_shape, name="expected shape")
    if not isinstance(payload, bytes):
        raise TypeError("depth payload must be bytes")
    if len(payload) < _FFS_DEPTH_HEADER.size:
        raise ValueError("FFS depth message is too short")
    magic, version, dtype, flags, frame_id, capture_ns, publish_ns, height, width, payload_size = (
        _FFS_DEPTH_HEADER.unpack_from(payload)
    )
    if magic != FFS_DEPTH_MAGIC or version != PACKET_VERSION or dtype != DTYPE_UINT16 or flags != 0:
        raise ValueError("unsupported FFS metric-depth header")
    _validate_uint(frame_id, name="frame_id", maximum=_UINT64_MAX)
    _validate_uint(capture_ns, name="capture_time_ns", maximum=_UINT64_MAX, positive=True)
    _validate_uint(publish_ns, name="publish_time_ns", maximum=_UINT64_MAX, positive=True)
    if publish_ns < capture_ns:
        raise ValueError("FFS publish time precedes capture time")
    if (height, width) != expected:
        raise ValueError(f"FFS depth shape {(height, width)} does not match {expected}")
    return FFSDepthPacket(
        depth_mm=_decode_pixels(
            payload,
            header_size=_FFS_DEPTH_HEADER.size,
            payload_size=payload_size,
            shape=expected,
        ),
        frame_id=frame_id,
        capture_time_ns=capture_ns,
        publish_time_ns=publish_ns,
    )


def pack_metric_depth(frame: MetricDepthFrame) -> bytes:
    """Pack one ``HMDPTH01`` packet."""

    if not isinstance(frame, MetricDepthFrame):
        raise TypeError("frame must be MetricDepthFrame")
    height, width, pixels = _validate_depth(frame.depth_mm)
    frame_id = _validate_uint(frame.frame_id, name="frame_id", maximum=_UINT64_MAX)
    capture_ns = _validate_uint(
        frame.camera_capture_time_ns,
        name="camera_capture_time_ns",
        maximum=_UINT64_MAX,
        positive=True,
    )
    completion_ns = _validate_uint(
        frame.completion_monotonic_ns,
        name="completion_monotonic_ns",
        maximum=_UINT64_MAX,
        positive=True,
    )
    latency_ns = _validate_uint(frame.ffs_latency_ns, name="ffs_latency_ns", maximum=_UINT64_MAX)
    return _METRIC_DEPTH_HEADER.pack(
        METRIC_DEPTH_MAGIC,
        PACKET_VERSION,
        DTYPE_UINT16,
        0,
        frame_id,
        capture_ns,
        completion_ns,
        latency_ns,
        height,
        width,
        len(pixels),
    ) + pixels


def unpack_metric_depth(payload: bytes, *, expected_shape: tuple[int, int]) -> MetricDepthFrame:
    """Validate and decode one ``HMDPTH01`` packet."""

    expected = _validate_shape(expected_shape, name="expected shape")
    if not isinstance(payload, bytes):
        raise TypeError("depth payload must be bytes")
    if len(payload) < _METRIC_DEPTH_HEADER.size:
        raise ValueError("metric-depth message is too short")
    (
        magic,
        version,
        dtype,
        flags,
        frame_id,
        capture_ns,
        completion_ns,
        latency_ns,
        height,
        width,
        payload_size,
    ) = _METRIC_DEPTH_HEADER.unpack_from(payload)
    if magic != METRIC_DEPTH_MAGIC or version != PACKET_VERSION or dtype != DTYPE_UINT16 or flags != 0:
        raise ValueError("unsupported metric-depth header")
    _validate_uint(frame_id, name="frame_id", maximum=_UINT64_MAX)
    _validate_uint(capture_ns, name="camera_capture_time_ns", maximum=_UINT64_MAX, positive=True)
    _validate_uint(completion_ns, name="completion_monotonic_ns", maximum=_UINT64_MAX, positive=True)
    _validate_uint(latency_ns, name="ffs_latency_ns", maximum=_UINT64_MAX)
    if (height, width) != expected:
        raise ValueError(f"metric-depth shape {(height, width)} does not match {expected}")
    return MetricDepthFrame(
        depth_mm=_decode_pixels(
            payload,
            header_size=_METRIC_DEPTH_HEADER.size,
            payload_size=payload_size,
            shape=expected,
        ),
        frame_id=frame_id,
        camera_capture_time_ns=capture_ns,
        completion_monotonic_ns=completion_ns,
        ffs_latency_ns=latency_ns,
    )


class LatestMetricDepthClient:
    """Non-blocking HWM-one subscriber that retains the newest valid frame."""

    def __init__(
        self,
        *,
        endpoint: str = "tcp://127.0.0.1:55563",
        shape: tuple[int, int] = (480, 848),
        min_valid_fraction: float = 0.25,
        context: zmq.Context | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.shape = _validate_shape(shape, name="metric-depth shape")
        self.min_valid_fraction = float(min_valid_fraction)
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in [0, 1]")
        self._context = context or zmq.Context()
        self._owns_context = context is None
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)
        self._latest: MetricDepthFrame | None = None
        self.last_error: Exception | None = None

    def _accept_payload(self, payload: bytes) -> bool:
        try:
            candidate = unpack_metric_depth(payload, expected_shape=self.shape)
            if self._latest is not None and candidate.frame_id <= self._latest.frame_id:
                raise ValueError("metric-depth frame IDs must increase")
            valid_fraction = float(np.count_nonzero(candidate.depth_mm)) / float(candidate.depth_mm.size)
            if valid_fraction < self.min_valid_fraction:
                raise ValueError(
                    "metric-depth valid fraction "
                    f"{valid_fraction:.4f} is below required {self.min_valid_fraction:.4f}"
                )
        except (TypeError, ValueError) as exc:
            self.last_error = exc
            return False
        self._latest = candidate
        self.last_error = None
        return True

    def _poll_once(self) -> None:
        try:
            payload = self._socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return
        self._accept_payload(payload)

    def read_latest(self) -> MetricDepthFrame | None:
        """Return an owned copy of the latest valid result without consuming it."""

        self._poll_once()
        if self._latest is None:
            return None
        latest = self._latest
        return MetricDepthFrame(
            depth_mm=latest.depth_mm.copy(order="C"),
            frame_id=latest.frame_id,
            camera_capture_time_ns=latest.camera_capture_time_ns,
            completion_monotonic_ns=latest.completion_monotonic_ns,
            ffs_latency_ns=latest.ffs_latency_ns,
        )

    def require_ready(self, *, timeout_s: float = 5.0) -> MetricDepthFrame:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            latest = self.read_latest()
            if latest is not None:
                return latest
            time.sleep(0.001)
        detail = f": {self.last_error}" if self.last_error is not None else ""
        raise RuntimeError(f"metric-depth readiness timed out{detail}")

    def close(self) -> None:
        self._socket.close(linger=0)
        if self._owns_context:
            self._context.term()


__all__ = [
    "FFSDepthPacket",
    "LatestMetricDepthClient",
    "MetricDepthFrame",
    "pack_metric_depth",
    "unpack_ffs_depth",
    "unpack_metric_depth",
]

"""Latest-only HMDPTH01 metric-depth wire contract and client.

The relay joins Fast-FoundationStereo ``HSDPTH01`` depth and ``FFSTEL01``
telemetry into one ``HMDPTH01`` packet. Each packet carries frame identity,
camera capture time, same-host monotonic completion time, measured FFS latency,
and a C-order uint16 millimetre image; zero depth means invalid.
The client validates frame ordering and the minimum valid-depth fraction.
Source attribution, the pinned reference hash and the modifications are in
THIRD_PARTY_NOTICES.md.
"""

# ruff: noqa: TRY004 -- preserve the pinned wire-contract exception types
from __future__ import annotations

import struct
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import zmq

METRIC_DEPTH_MAGIC = b"HMDPTH01"
PACKET_VERSION = 1
DTYPE_UINT16 = 1

_METRIC_DEPTH_HEADER = struct.Struct("<8sBBHQQQQIII")
_UINT64_MAX = (1 << 64) - 1
_UINT32_MAX = (1 << 32) - 1


@dataclass(frozen=True)
class MetricDepthFrame:
    """Newest metric-depth result owned by the caller."""

    depth_mm: np.ndarray
    frame_id: int
    camera_capture_time_ns: int
    completion_monotonic_ns: int
    ffs_latency_ns: int


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


def _newer(candidate: MetricDepthFrame, latest: MetricDepthFrame) -> bool:
    """True for a later frame id, or for a later camera capture time, which accepts a
    producer that restarted its frame ids."""
    if candidate.frame_id > latest.frame_id:
        return True
    return candidate.camera_capture_time_ns > latest.camera_capture_time_ns


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
            if self._latest is not None and not _newer(candidate, self._latest):
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


    def close(self) -> None:
        self._socket.close(linger=0)
        if self._owns_context:
            self._context.term()


__all__ = [
    "LatestMetricDepthClient",
    "MetricDepthFrame",
    "unpack_metric_depth",
]

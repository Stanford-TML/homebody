"""Atomic HWM-one transport for raw G1 infrared stereo pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import zmq

from holosoma_inference.fast_stereo import pair_protocol

DEFAULT_BIND_ENDPOINT = "tcp://0.0.0.0:55562"
DEFAULT_CONNECT_ENDPOINT = "tcp://192.168.123.164:55562"


class TransportError(RuntimeError):
    """Raised when transport setup or sequence invariants fail."""


@dataclass
class PublisherCounters:
    attempted: int = 0
    sent: int = 0
    send_dropped: int = 0


@dataclass
class ReceiverCounters:
    received: int = 0
    malformed: int = 0
    cache_replaced: int = 0
    sequence_gaps: int = 0


@dataclass(frozen=True)
class CachedStereoPair:
    """One receiver-owned pair whose bytes outlive the ZMQ frame objects."""

    header: pair_protocol.StereoPairHeader
    left: bytes
    right: bytes


class StereoPairPublisher:
    """A single-consumer PUSH socket with explicit nonblocking backpressure."""

    def __init__(
        self,
        endpoint: str = DEFAULT_BIND_ENDPOINT,
        *,
        context: Any = None,
    ) -> None:
        self._owns_context = context is None
        self._context = context if context is not None else zmq.Context()
        self._socket = self._context.socket(zmq.PUSH)
        self._closed = False
        self._last_sequence: Optional[int] = None  # noqa: UP045 - G1 uses Python 3.8
        self.counters = PublisherCounters()
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IMMEDIATE, 1)
        try:
            self._socket.bind(endpoint)
        except zmq.ZMQError as exc:
            self._socket.close(linger=0)
            if self._owns_context:
                self._context.term()
            raise TransportError(f"cannot bind stereo publisher to {endpoint}: {exc}") from exc

    def send_pair(
        self,
        header: pair_protocol.StereoPairHeader,
        left: pair_protocol.BytesLike,
        right: pair_protocol.BytesLike,
    ) -> bool:
        """Attempt one atomic send; return false only for explicit HWM backpressure."""

        if self._closed:
            raise TransportError("cannot send on a closed stereo publisher")
        parts = pair_protocol.encode_multipart(header, left, right)
        if self._last_sequence is not None and header.sequence <= self._last_sequence:
            raise TransportError(
                f"sequence {header.sequence} must be strictly greater than previous sequence {self._last_sequence}"
            )
        self._last_sequence = header.sequence
        self.counters.attempted += 1
        try:
            self._socket.send_multipart(list(parts), flags=zmq.NOBLOCK, copy=True)
        except zmq.Again:
            self.counters.send_dropped += 1
            return False
        self.counters.sent += 1
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._socket.close(linger=0)
        if self._owns_context:
            self._context.term()
        self._closed = True

    def __enter__(self) -> StereoPairPublisher:  # noqa: PYI034 - Python 3.8 has no typing.Self
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class LatestStereoPairReceiver:
    """A PULL receiver that drains pending data into one receiver-owned cache."""

    def __init__(
        self,
        endpoint: str = DEFAULT_CONNECT_ENDPOINT,
        *,
        context: Any = None,
    ) -> None:
        self._owns_context = context is None
        self._context = context if context is not None else zmq.Context()
        self._socket = self._context.socket(zmq.PULL)
        self._closed = False
        self._last_sequence: Optional[int] = None  # noqa: UP045 - G1 uses Python 3.8
        self.cached_pair: Optional[CachedStereoPair] = None  # noqa: UP045 - G1 uses Python 3.8
        self.counters = ReceiverCounters()
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        try:
            self._socket.connect(endpoint)
        except zmq.ZMQError as exc:
            self._socket.close(linger=0)
            if self._owns_context:
                self._context.term()
            raise TransportError(f"cannot connect stereo receiver to {endpoint}: {exc}") from exc

    def receive_latest(self, *, timeout_ms: int = 0) -> Optional[CachedStereoPair]:  # noqa: UP045
        """Drain all currently pending packets and return only a newly received latest pair."""

        if self._closed:
            raise TransportError("cannot receive on a closed stereo receiver")
        if not self._socket.poll(timeout_ms, zmq.POLLIN):
            return None

        newest = self.cached_pair
        received_new = False
        while self._socket.poll(0, zmq.POLLIN):
            try:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
                decoded = pair_protocol.decode_multipart(parts, previous_sequence=self._last_sequence)
            except (zmq.Again, pair_protocol.ProtocolError, TypeError):
                self.counters.malformed += 1
                continue

            sequence = decoded.header.sequence
            if self._last_sequence is not None and sequence > self._last_sequence + 1:
                self.counters.sequence_gaps += sequence - self._last_sequence - 1
            self._last_sequence = sequence
            if newest is not None:
                self.counters.cache_replaced += 1
            newest = CachedStereoPair(
                header=decoded.header,
                left=decoded.left.tobytes(),
                right=decoded.right.tobytes(),
            )
            self.counters.received += 1
            received_new = True

        if received_new:
            self.cached_pair = newest
            return newest
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._socket.close(linger=0)
        if self._owns_context:
            self._context.term()
        self._closed = True

    def __enter__(self) -> LatestStereoPairReceiver:  # noqa: PYI034 - Python 3.8 has no typing.Self
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

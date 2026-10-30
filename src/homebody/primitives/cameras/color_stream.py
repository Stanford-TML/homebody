"""Colour stream subscriber (HSCOLOR1): one JPEG per frame, newest wins.

Each message is one zmq part (``b"HSCOLOR1" | uint32 LE header length | header JSON | JPEG``);
the header carries the sequence, capture time, timestamp kind and camera name, and for the head
camera its geometry (K_ir, baseline, colour intrinsics, IR1->colour extrinsic).
"""
from __future__ import annotations

import json
import struct
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import zmq


@dataclass(frozen=True)
class ColorFrame:
    """One decoded RGB receipt with source and receiver time metadata."""
    image: np.ndarray
    captured_at: float
    generation: int
    capture_timestamp_basis: str | None = None
    camera_serial: str | None = None
    sdk_receive_realtime_ns: int | None = None


MAGIC = b"HSCOLOR1"
_HEADER_LEN = struct.Struct("<I")


class ColorStreamError(RuntimeError):
    """A message that is not HSCOLOR1, or one whose JPEG does not decode."""


@dataclass(frozen=True)
class ColorMessage:
    header: dict
    jpeg: bytes


def pack(header: dict, jpeg: bytes) -> bytes:
    """The publisher's serializer, mirrored here so the wire format is tested end to end."""
    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return MAGIC + _HEADER_LEN.pack(len(body)) + body + bytes(jpeg)


def unpack(payload: bytes) -> ColorMessage:
    if len(payload) < len(MAGIC) + _HEADER_LEN.size or payload[:len(MAGIC)] != MAGIC:
        raise ColorStreamError("not an HSCOLOR1 message")
    (length,) = _HEADER_LEN.unpack_from(payload, len(MAGIC))
    start = len(MAGIC) + _HEADER_LEN.size
    if len(payload) < start + length + 2:
        raise ColorStreamError("HSCOLOR1 message is truncated")
    header = json.loads(payload[start:start + length].decode("utf-8"))
    return ColorMessage(header=header, jpeg=bytes(payload[start + length:]))


def decode(message: ColorMessage) -> ColorFrame:
    image = cv2.imdecode(np.frombuffer(message.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ColorStreamError("HSCOLOR1 JPEG did not decode")
    header = message.header
    return ColorFrame(image=image, captured_at=float(header["capture_realtime_ns"]) / 1e9,
                      generation=int(header["sequence"]),
                      capture_timestamp_basis=header.get("capture_timestamp_basis"),
                      camera_serial=header.get("camera_serial"),
                      sdk_receive_realtime_ns=header.get("sdk_receive_realtime_ns"))


class LatestColorClient:
    """A conflating SUB socket: wait() returns the newest frame, or None after its timeout."""

    def __init__(self, endpoint: str, *, context: zmq.Context | None = None):
        self.endpoint = endpoint
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        self._lock = threading.Lock()
        self.geometry: dict | None = None
        self.last_error: str = ""
        self.received = 0

    def wait(self, timeout_s: float) -> ColorFrame | None:
        """The next message within timeout_s, decoded, or None."""
        with self._lock:
            if not self._poller.poll(max(0.0, timeout_s) * 1000.0):
                return None
            return self._accept(self._socket.recv(copy=True))

    def _accept(self, payload: bytes) -> ColorFrame | None:
        try:
            message = unpack(payload)
            frame = decode(message)
        except (ColorStreamError, KeyError, ValueError) as exc:
            self.last_error = str(exc)
            return None
        self.geometry = dict(message.header.get("geometry") or {})
        self.received += 1
        return frame

    def close(self) -> None:
        with self._lock:
            self._socket.close(0)

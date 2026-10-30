"""Join the FFS depth and telemetry outputs into one minimal latest stream."""

from __future__ import annotations

import json
import math
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import zmq

from holosoma_inference.runtime.metric_depth import MetricDepthFrame, pack_metric_depth, unpack_ffs_depth


@dataclass(frozen=True)
class RelayStatus:
    frame_id: int
    completion_monotonic_ns: int
    ffs_latency_ms: float
    invalid_fraction: float
    valid_min_m: float
    valid_max_m: float


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IDENTITY_FIELDS = ("engine_sha256", "camera_sha256", "calibration_sha256")


def _require_number(record: dict[str, object], name: str) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"FFS telemetry {name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"FFS telemetry {name} must be finite and nonnegative")
    return converted


def parse_telemetry(
    payload: bytes,
    *,
    expected_identities: dict[str, str] | None = None,
) -> dict[str, object]:
    if not isinstance(payload, bytes):
        raise TypeError("FFS telemetry payload must be bytes")
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("FFS telemetry is not valid JSON") from exc
    if not isinstance(record, dict) or record.get("schema") != "FFSTEL01":
        raise ValueError("FFS telemetry schema must be FFSTEL01")
    for name in ("frame_id", "capture_time_ns", "publish_time_ns"):
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"FFS telemetry {name} must be a nonnegative integer")
    if int(record["capture_time_ns"]) <= 0:
        raise ValueError("FFS telemetry capture_time_ns must be positive")
    if int(record["publish_time_ns"]) < int(record["capture_time_ns"]):
        raise ValueError("FFS telemetry publish_time_ns precedes capture_time_ns")
    for name in ("host_to_device_ms", "infer_depth_ms", "device_to_host_ms"):
        _require_number(record, name)
    for name in _IDENTITY_FIELDS:
        value = record.get(name)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"FFS telemetry {name} must be a lowercase SHA-256")
        if expected_identities is not None and value != expected_identities[name]:
            raise ValueError(
                f"FFS telemetry {name} mismatch: expected {expected_identities[name]}, got {value}"
            )
    return record


class DepthVisionRelay:
    """Bounded identity join whose output callback receives HMDPTH01 bytes."""

    def __init__(
        self,
        *,
        shape: tuple[int, int] = (480, 848),
        publish: Callable[[bytes], None],
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        max_unmatched: int = 8,
        expected_identities: dict[str, str] | None = None,
    ) -> None:
        if expected_identities is not None:
            missing = set(_IDENTITY_FIELDS).difference(expected_identities)
            if missing:
                raise ValueError(f"missing expected FFS identities: {sorted(missing)}")
            invalid = {
                name: value
                for name, value in expected_identities.items()
                if name in _IDENTITY_FIELDS
                and (not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None)
            }
            if invalid:
                raise ValueError(f"invalid expected FFS identities: {sorted(invalid)}")
        self.shape = shape
        self._publish = publish
        self._monotonic_ns = monotonic_ns
        self._max_unmatched = max_unmatched
        self._expected_identities = expected_identities
        self._depth: dict[int, object] = {}
        self._telemetry: dict[int, dict[str, object]] = {}
        self._last_frame_id: int | None = None
        self.status: RelayStatus | None = None
        self.last_error: Exception | None = None
        self.joined_count = 0
        self.rejected_count = 0

    def _bound(self, records: dict[int, object]) -> None:
        while len(records) > self._max_unmatched:
            records.pop(next(iter(records)))

    def ingest_depth(self, payload: bytes) -> bool:
        try:
            packet = unpack_ffs_depth(payload, expected_shape=self.shape)
        except (TypeError, ValueError) as exc:
            self.last_error = exc
            self.rejected_count += 1
            return False
        self._depth[packet.frame_id] = packet
        self._bound(self._depth)
        self._join(packet.frame_id)
        return True

    def ingest_telemetry(self, payload: bytes) -> bool:
        try:
            record = parse_telemetry(payload, expected_identities=self._expected_identities)
            frame_id = int(record["frame_id"])
        except (TypeError, ValueError) as exc:
            self.last_error = exc
            self.rejected_count += 1
            return False
        self._telemetry[frame_id] = record
        self._bound(self._telemetry)
        self._join(frame_id)
        return True

    def _join(self, frame_id: int) -> None:
        packet = self._depth.get(frame_id)
        telemetry = self._telemetry.get(frame_id)
        if packet is None or telemetry is None:
            return
        del self._depth[frame_id]
        del self._telemetry[frame_id]
        try:
            if int(telemetry["capture_time_ns"]) != packet.capture_time_ns:
                raise ValueError("FFS depth/telemetry capture identity mismatch")
            if int(telemetry["publish_time_ns"]) != packet.publish_time_ns:
                raise ValueError("FFS depth/telemetry publish identity mismatch")
            if self._last_frame_id is not None and frame_id <= self._last_frame_id:
                raise ValueError("joined FFS frame IDs must increase")
            ffs_latency_ms = sum(
                _require_number(telemetry, name)
                for name in ("host_to_device_ms", "infer_depth_ms", "device_to_host_ms")
            )
            completion_ns = int(self._monotonic_ns())
            output = MetricDepthFrame(
                depth_mm=np.ascontiguousarray(packet.depth_mm),
                frame_id=frame_id,
                camera_capture_time_ns=packet.capture_time_ns,
                completion_monotonic_ns=completion_ns,
                ffs_latency_ns=round(ffs_latency_ms * 1_000_000.0),
            )
            self._publish(pack_metric_depth(output))
            valid = packet.depth_mm > 0
            invalid_fraction = 1.0 - float(np.count_nonzero(valid)) / float(valid.size)
            if np.any(valid):
                valid_min_m = float(packet.depth_mm[valid].min()) / 1000.0
                valid_max_m = float(packet.depth_mm[valid].max()) / 1000.0
            else:
                valid_min_m = float("nan")
                valid_max_m = float("nan")
        except (TypeError, ValueError) as exc:
            self.last_error = exc
            self.rejected_count += 1
            return
        self._last_frame_id = frame_id
        self.joined_count += 1
        self.status = RelayStatus(
            frame_id,
            completion_ns,
            ffs_latency_ms,
            invalid_fraction,
            valid_min_m,
            valid_max_m,
        )
        self.last_error = None


def run_relay(
    *,
    source_depth_endpoint: str,
    source_telemetry_endpoint: str,
    output_endpoint: str,
    shape: tuple[int, int],
    expected_identities: dict[str, str],
    ready_file: Path | None = None,
    startup_timeout_s: float = 15.0,
) -> None:
    """Run the foreground relay until interrupted."""

    context = zmq.Context()
    depth = context.socket(zmq.SUB)
    telemetry = context.socket(zmq.SUB)
    output = context.socket(zmq.PUB)
    for socket, endpoint in (
        (depth, source_depth_endpoint),
        (telemetry, source_telemetry_endpoint),
    ):
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.connect(endpoint)
    output.setsockopt(zmq.SNDHWM, 1)
    output.setsockopt(zmq.LINGER, 0)
    output.bind(output_endpoint)
    if ready_file is not None:
        ready_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = ready_file.with_name(f".{ready_file.name}.{time.monotonic_ns()}.tmp")
        temporary.write_text(f"{output_endpoint}\n", encoding="utf-8")
        temporary.replace(ready_file)
    relay = DepthVisionRelay(
        shape=shape,
        publish=output.send,
        expected_identities=expected_identities,
    )
    poller = zmq.Poller()
    poller.register(depth, zmq.POLLIN)
    poller.register(telemetry, zmq.POLLIN)
    last_report = time.monotonic()
    last_counted_frame: int | None = None
    frames_since_report = 0
    recent_ffs_ms: deque[float] = deque(maxlen=240)
    startup_deadline = time.monotonic() + float(startup_timeout_s)
    depth_packet_count = 0
    telemetry_packet_count = 0
    try:
        while True:
            events = dict(poller.poll(timeout=100))
            if depth in events:
                depth_packet_count += 1
                relay.ingest_depth(depth.recv())
            if telemetry in events:
                telemetry_packet_count += 1
                relay.ingest_telemetry(telemetry.recv())
            now = time.monotonic()
            if relay.status is not None and relay.status.frame_id != last_counted_frame:
                last_counted_frame = relay.status.frame_id
                frames_since_report += 1
                recent_ffs_ms.append(relay.status.ffs_latency_ms)
            if now - last_report >= 1.0:
                if relay.status is None:
                    detail = f" error={relay.last_error}" if relay.last_error else ""
                    print(
                        "VISION waiting for FFS | "
                        f"depth={depth_packet_count} telemetry={telemetry_packet_count} "
                        f"rejected={relay.rejected_count}{detail}",
                        flush=True,
                    )
                else:
                    age_ms = max(0, time.monotonic_ns() - relay.status.completion_monotonic_ns) / 1e6
                    rate_hz = frames_since_report / max(now - last_report, 1e-9)
                    ffs_p95_ms = float(np.percentile(recent_ffs_ms, 95)) if recent_ffs_ms else float("nan")
                    print(
                        "VISION ready | "
                        f"{rate_hz:.1f} Hz | FFS p95 {ffs_p95_ms:.2f} ms | frame {relay.status.frame_id} | "
                        f"age {age_ms:.1f} ms | invalid {100.0 * relay.status.invalid_fraction:.1f}% | "
                        f"valid {relay.status.valid_min_m:.2f}-{relay.status.valid_max_m:.2f} m | "
                        f"joined {relay.joined_count} rejected {relay.rejected_count}",
                        flush=True,
                    )
                frames_since_report = 0
                last_report = now
            if relay.status is None and now >= startup_deadline:
                detail = f"; last_error={relay.last_error}" if relay.last_error else ""
                raise RuntimeError(
                    "no joined depth/telemetry frame before startup timeout: "
                    f"depth_packets={depth_packet_count} "
                    f"telemetry_packets={telemetry_packet_count} "
                    f"rejected={relay.rejected_count}{detail}"
                )
    finally:
        if ready_file is not None:
            ready_file.unlink(missing_ok=True)
        for socket in (depth, telemetry, output):
            socket.close(linger=0)
        context.term()


__all__ = ["DepthVisionRelay", "RelayStatus", "parse_telemetry", "run_relay"]

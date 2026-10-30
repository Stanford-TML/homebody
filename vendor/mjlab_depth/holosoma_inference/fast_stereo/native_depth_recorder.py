"""Checksummed native D435i Z16 records paired to stereo-pair sequences."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Mapping, Optional

from holosoma_inference.fast_stereo import pair_protocol

MAX_NATIVE_PAIR_SKEW_NS = 500_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class NativeDepthRecorderError(RuntimeError):
    """Raised when paired native depth cannot be recorded or verified safely."""


@dataclass(frozen=True)
class NativeDepthRecord:
    """One native Z16 image joined to one validated HSSPAIR1 sequence."""

    sequence: int
    pair_camera_frame_number: int
    pair_camera_timestamp_ns: int
    frame_number: int
    timestamp_ns: int
    timestamp_domain: int
    height: int
    width: int
    payload: bytes
    metadata: Mapping[str, Any]


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise NativeDepthRecorderError(f"{name} must be a 64-character lowercase SHA-256")
    return value


def _canonical_profile(profile: Mapping[str, Any]) -> tuple[Dict[str, Any], bytes]:
    try:
        canonical = json.dumps(
            dict(profile),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        decoded = json.loads(canonical)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise NativeDepthRecorderError(f"native profile is not canonical JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise NativeDepthRecorderError("native profile must be a JSON object")
    if decoded.get("format") != "z16":
        raise NativeDepthRecorderError("native profile format must be z16")
    for name in ("width", "height", "fps"):
        value = decoded.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise NativeDepthRecorderError(f"native profile {name} must be a positive integer")
    return decoded, canonical


def _payload_bytes(payload: object) -> bytes:
    try:
        view = memoryview(payload)
    except TypeError as exc:
        raise NativeDepthRecorderError("native Z16 payload is not bytes-like") from exc
    if not view.c_contiguous:
        raise NativeDepthRecorderError("native Z16 payload must be C-contiguous")
    try:
        return view.cast("B").tobytes()
    except TypeError as exc:
        raise NativeDepthRecorderError("native Z16 payload cannot be viewed as bytes") from exc


def _validate_record(
    record: NativeDepthRecord,
    *,
    expected_width: int,
    expected_height: int,
    previous_sequence: Optional[int],  # noqa: UP045 - G1 uses Python 3.8
) -> bytes:
    if not isinstance(record.sequence, int) or isinstance(record.sequence, bool) or record.sequence < 0:
        raise NativeDepthRecorderError("paired sequence must be a non-negative integer")
    if previous_sequence is not None and record.sequence <= previous_sequence:
        raise NativeDepthRecorderError(
            f"sequence {record.sequence} must be strictly greater than previous sequence {previous_sequence}"
        )
    for name, frame_number in (
        ("paired infrared frame number", record.pair_camera_frame_number),
        ("native depth frame number", record.frame_number),
    ):
        if not isinstance(frame_number, int) or isinstance(frame_number, bool) or frame_number < 0:
            raise NativeDepthRecorderError(f"{name} must be a non-negative integer")
    if record.timestamp_domain != pair_protocol.TIMESTAMP_DOMAIN_GLOBAL_TIME:
        raise NativeDepthRecorderError("native depth timestamp must use RealSense global time")
    if record.timestamp_ns <= 0 or record.pair_camera_timestamp_ns <= 0:
        raise NativeDepthRecorderError("native and paired timestamps must be positive")
    timestamp_skew_ns = abs(record.timestamp_ns - record.pair_camera_timestamp_ns)
    if timestamp_skew_ns > MAX_NATIVE_PAIR_SKEW_NS:
        raise NativeDepthRecorderError(
            f"native-to-infrared timestamp skew {timestamp_skew_ns} exceeds 0.5 ms"
        )
    if (record.width, record.height) != (expected_width, expected_height):
        raise NativeDepthRecorderError(
            "native frame shape does not match the frozen native stream profile"
        )
    raw = _payload_bytes(record.payload)
    expected_size = expected_width * expected_height * 2
    if len(raw) != expected_size:
        raise NativeDepthRecorderError(
            f"native Z16 payload size {len(raw)} does not match expected payload size {expected_size}"
        )
    try:
        json.dumps(dict(record.metadata), allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise NativeDepthRecorderError(f"native frame metadata is not JSON serializable: {exc}") from exc
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_child(root: Path, filename: object) -> Path:
    if not isinstance(filename, str):
        raise NativeDepthRecorderError("native chunk filename must be a string")
    pure = PurePosixPath(filename)
    if pure.is_absolute() or len(pure.parts) != 1 or pure.name != filename:
        raise NativeDepthRecorderError(f"unsafe native chunk filename: {filename!r}")
    return root / filename


class NativeDepthChunkWriter:
    """Write uncompressed Z16 frames in bounded, source-bound chunks."""

    def __init__(
        self,
        output_dir: Path,
        *,
        frames_per_chunk: int = 60,
        camera_serial_sha256: str,
        infrared_calibration_sha256: str,
        native_profile: Mapping[str, Any],
    ) -> None:
        if frames_per_chunk < 1:
            raise NativeDepthRecorderError("frames_per_chunk must be positive")
        self.camera_serial_sha256 = _require_sha256(
            "camera serial SHA-256", camera_serial_sha256
        )
        self.infrared_calibration_sha256 = _require_sha256(
            "infrared calibration SHA-256", infrared_calibration_sha256
        )
        self.native_profile, canonical_profile = _canonical_profile(native_profile)
        self.native_profile_sha256 = hashlib.sha256(canonical_profile).hexdigest()
        self.expected_width = int(self.native_profile["width"])
        self.expected_height = int(self.native_profile["height"])
        self.output_dir = output_dir
        self.frames_per_chunk = frames_per_chunk
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(self.output_dir.iterdir()):
            raise NativeDepthRecorderError(
                f"native recording output directory is not empty: {self.output_dir}"
            )

        self._manifest_path = self.output_dir / "manifest.json"
        self._index_path = self.output_dir / "index.jsonl"
        self._index_partial_path = self.output_dir / "index.jsonl.partial"
        self._index_stream = self._index_partial_path.open("x", encoding="utf-8")
        self._chunk_stream: Optional[Any] = None  # noqa: UP045 - G1 uses Python 3.8
        self._chunk_partial_path: Optional[Path] = None  # noqa: UP045 - G1 uses Python 3.8
        self._chunk_hasher: Optional[Any] = None  # noqa: UP045 - G1 uses Python 3.8
        self._chunk_index = 0
        self._chunk_frame_count = 0
        self._chunk_first_sequence = 0
        self._chunk_last_sequence = 0
        self._frame_count = 0
        self._last_sequence: Optional[int] = None  # noqa: UP045 - G1 uses Python 3.8
        self._chunks: List[Dict[str, Any]] = []
        self._closed = False

    def _begin_chunk(self) -> None:
        filename = f"chunk-{self._chunk_index:05d}.z16"
        self._chunk_partial_path = self.output_dir / f"{filename}.partial"
        self._chunk_stream = self._chunk_partial_path.open("xb")
        self._chunk_hasher = hashlib.sha256()
        self._chunk_frame_count = 0

    def _finish_chunk(self) -> None:
        if self._chunk_stream is None:
            return
        self._chunk_stream.flush()
        os.fsync(self._chunk_stream.fileno())
        size_bytes = self._chunk_stream.tell()
        self._chunk_stream.close()
        filename = f"chunk-{self._chunk_index:05d}.z16"
        self._chunk_partial_path.replace(self.output_dir / filename)
        self._chunks.append(
            {
                "filename": filename,
                "frame_count": self._chunk_frame_count,
                "first_sequence": self._chunk_first_sequence,
                "last_sequence": self._chunk_last_sequence,
                "size_bytes": size_bytes,
                "sha256": self._chunk_hasher.hexdigest(),
            }
        )
        self._chunk_index += 1
        self._chunk_stream = None
        self._chunk_partial_path = None
        self._chunk_hasher = None

    def write_frame(self, record: NativeDepthRecord) -> None:
        if self._closed:
            raise NativeDepthRecorderError("cannot write to a closed native depth recorder")
        raw = _validate_record(
            record,
            expected_width=self.expected_width,
            expected_height=self.expected_height,
            previous_sequence=self._last_sequence,
        )
        if self._chunk_stream is None:
            self._begin_chunk()
        offset_bytes = self._chunk_stream.tell()
        self._chunk_stream.write(raw)
        self._chunk_hasher.update(raw)
        if self._chunk_frame_count == 0:
            self._chunk_first_sequence = record.sequence
        self._chunk_frame_count += 1
        self._chunk_last_sequence = record.sequence
        self._frame_count += 1
        self._last_sequence = record.sequence
        self._index_stream.write(
            json.dumps(
                {
                    "chunk": self._chunk_index,
                    "offset_bytes": offset_bytes,
                    "record_size_bytes": len(raw),
                    "payload_sha256": hashlib.sha256(raw).hexdigest(),
                    "sequence": record.sequence,
                    "pair_camera_frame_number": record.pair_camera_frame_number,
                    "pair_camera_timestamp_ns": record.pair_camera_timestamp_ns,
                    "frame_number": record.frame_number,
                    "timestamp_ns": record.timestamp_ns,
                    "timestamp_domain": record.timestamp_domain,
                    "height": record.height,
                    "width": record.width,
                    "metadata": dict(record.metadata),
                },
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
        )
        if self._chunk_frame_count >= self.frames_per_chunk:
            self._finish_chunk()

    def close(self) -> None:
        if self._closed:
            return
        self._finish_chunk()
        self._index_stream.flush()
        os.fsync(self._index_stream.fileno())
        self._index_stream.close()
        self._index_partial_path.replace(self._index_path)
        manifest = {
            "schema_version": 1,
            "format": "concatenated_z16_le",
            "frames_per_chunk": self.frames_per_chunk,
            "frame_count": self._frame_count,
            "camera_serial_sha256": self.camera_serial_sha256,
            "infrared_calibration_sha256": self.infrared_calibration_sha256,
            "native_profile": self.native_profile,
            "native_profile_sha256": self.native_profile_sha256,
            "index_sha256": _sha256_file(self._index_path),
            "chunks": self._chunks,
        }
        partial_manifest = self.output_dir / "manifest.json.partial"
        with partial_manifest.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        partial_manifest.replace(self._manifest_path)
        self._closed = True

    def __enter__(self) -> NativeDepthChunkWriter:  # noqa: PYI034 - Python 3.8 has no typing.Self
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _load_manifest(recording_dir: Path) -> Dict[str, Any]:
    try:
        manifest = json.loads((recording_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeDepthRecorderError(f"cannot load native depth manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise NativeDepthRecorderError("native depth manifest has an unsupported schema")
    if manifest.get("format") != "concatenated_z16_le":
        raise NativeDepthRecorderError("native depth manifest has an unsupported format")
    _require_sha256("camera serial SHA-256", manifest.get("camera_serial_sha256"))
    _require_sha256(
        "infrared calibration SHA-256", manifest.get("infrared_calibration_sha256")
    )
    profile, canonical_profile = _canonical_profile(manifest.get("native_profile", {}))
    profile_sha256 = _require_sha256(
        "native profile SHA-256", manifest.get("native_profile_sha256")
    )
    if hashlib.sha256(canonical_profile).hexdigest() != profile_sha256:
        raise NativeDepthRecorderError("native profile SHA-256 does not match manifest")
    manifest["native_profile"] = profile
    return manifest


def iter_recorded_native_depth(recording_dir: Path) -> Iterator[NativeDepthRecord]:
    """Verify every identity and digest before yielding native Z16 frames."""

    manifest = _load_manifest(recording_dir)
    index_path = recording_dir / "index.jsonl"
    if _sha256_file(index_path) != _require_sha256("index SHA-256", manifest.get("index_sha256")):
        raise NativeDepthRecorderError("native depth index SHA-256 does not match manifest")
    try:
        index_rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeDepthRecorderError(f"cannot load native depth index: {exc}") from exc

    chunks = {}
    for chunk_index, chunk in enumerate(manifest.get("chunks", [])):
        if not isinstance(chunk, dict):
            raise NativeDepthRecorderError("native chunk manifest entry must be an object")
        path = _safe_child(recording_dir, chunk.get("filename"))
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise NativeDepthRecorderError(f"cannot read native chunk {path.name}: {exc}") from exc
        if len(payload) != chunk.get("size_bytes"):
            raise NativeDepthRecorderError(f"native chunk {path.name} size does not match manifest")
        expected_sha256 = _require_sha256(
            f"native chunk {path.name} SHA-256", chunk.get("sha256")
        )
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise NativeDepthRecorderError(
                f"native chunk {path.name} SHA-256 does not match manifest"
            )
        chunks[chunk_index] = payload

    previous_sequence: Optional[int] = None  # noqa: UP045 - G1 uses Python 3.8
    profile = manifest["native_profile"]
    for row in index_rows:
        try:
            chunk_payload = chunks[row["chunk"]]
            offset = int(row["offset_bytes"])
            size = int(row["record_size_bytes"])
            raw = chunk_payload[offset : offset + size]
            if len(raw) != size:
                raise NativeDepthRecorderError(
                    f"native frame payload is truncated at sequence {row.get('sequence')}"
                )
            if hashlib.sha256(raw).hexdigest() != _require_sha256(
                "native frame payload SHA-256", row.get("payload_sha256")
            ):
                raise NativeDepthRecorderError(
                    f"native frame payload SHA-256 mismatch at sequence {row.get('sequence')}"
                )
            record = NativeDepthRecord(
                sequence=int(row["sequence"]),
                pair_camera_frame_number=int(row["pair_camera_frame_number"]),
                pair_camera_timestamp_ns=int(row["pair_camera_timestamp_ns"]),
                frame_number=int(row["frame_number"]),
                timestamp_ns=int(row["timestamp_ns"]),
                timestamp_domain=int(row["timestamp_domain"]),
                height=int(row["height"]),
                width=int(row["width"]),
                payload=raw,
                metadata=dict(row["metadata"]),
            )
        except (KeyError, TypeError, ValueError, NativeDepthRecorderError) as exc:
            if isinstance(exc, NativeDepthRecorderError):
                raise
            raise NativeDepthRecorderError(f"invalid native depth index row {row!r}: {exc}") from exc
        _validate_record(
            record,
            expected_width=int(profile["width"]),
            expected_height=int(profile["height"]),
            previous_sequence=previous_sequence,
        )
        previous_sequence = record.sequence
        yield record
    if len(index_rows) != manifest.get("frame_count"):
        raise NativeDepthRecorderError("native depth index count does not match manifest")

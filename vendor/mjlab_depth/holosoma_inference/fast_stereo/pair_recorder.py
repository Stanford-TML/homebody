"""Bounded, checksummed recording and replay for atomic HSSPAIR1 pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from holosoma_inference.fast_stereo import pair_protocol
from holosoma_inference.fast_stereo.publisher import (
    DEFAULT_CONNECT_ENDPOINT,
    CachedStereoPair,
    LatestStereoPairReceiver,
)


class PairRecorderError(RuntimeError):
    """Raised when a stereo-pair recording cannot be safely written or replayed."""


class PairChunkWriter:
    """Write fixed HSSPAIR1 records into bounded chunks with an immutable index."""

    def __init__(self, output_dir: Path, *, pairs_per_chunk: int = 60) -> None:
        if pairs_per_chunk < 1:
            raise PairRecorderError("pairs_per_chunk must be positive")
        self.output_dir = output_dir
        self.pairs_per_chunk = pairs_per_chunk
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.output_dir / "manifest.json"
        self._index_path = self.output_dir / "index.jsonl"
        if self._manifest_path.exists() or self._index_path.exists() or list(self.output_dir.glob("chunk-*")):
            raise PairRecorderError(f"recording output directory is not empty: {self.output_dir}")
        self._index_stream = self._index_path.open("w", encoding="utf-8")
        self._chunk_stream: Optional[Any] = None  # noqa: UP045 - G1 uses Python 3.8
        self._chunk_partial_path: Optional[Path] = None  # noqa: UP045 - G1 uses Python 3.8
        self._chunk_hasher: Optional[Any] = None  # noqa: UP045 - G1 uses Python 3.8
        self._chunk_index = 0
        self._chunk_pair_count = 0
        self._chunk_first_sequence = 0
        self._chunk_last_sequence = 0
        self._pair_count = 0
        self._last_sequence: Optional[int] = None  # noqa: UP045 - G1 uses Python 3.8
        self._camera_serial_sha256: Optional[str] = None  # noqa: UP045 - G1 uses Python 3.8
        self._calibration_sha256: Optional[str] = None  # noqa: UP045 - G1 uses Python 3.8
        self._chunks: List[Dict[str, Any]] = []
        self._closed = False

    def _begin_chunk(self) -> None:
        filename = f"chunk-{self._chunk_index:05d}.hsspair"
        self._chunk_partial_path = self.output_dir / f"{filename}.partial"
        self._chunk_stream = self._chunk_partial_path.open("wb")
        self._chunk_hasher = hashlib.sha256()
        self._chunk_pair_count = 0

    def _finish_chunk(self) -> None:
        if self._chunk_stream is None:
            return
        self._chunk_stream.flush()
        os.fsync(self._chunk_stream.fileno())
        size_bytes = self._chunk_stream.tell()
        self._chunk_stream.close()
        filename = f"chunk-{self._chunk_index:05d}.hsspair"
        final_path = self.output_dir / filename
        self._chunk_partial_path.replace(final_path)
        self._chunks.append(
            {
                "filename": filename,
                "pair_count": self._chunk_pair_count,
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

    def write_pair(
        self,
        pair: CachedStereoPair,
        *,
        receive_realtime_ns: int,
        receive_monotonic_ns: int,
    ) -> None:
        if self._closed:
            raise PairRecorderError("cannot write to a closed pair recorder")
        sequence = pair.header.sequence
        if self._last_sequence is not None and sequence <= self._last_sequence:
            raise PairRecorderError(
                f"sequence {sequence} must be strictly greater than previous sequence {self._last_sequence}"
            )
        if receive_realtime_ns <= 0 or receive_monotonic_ns <= 0:
            raise PairRecorderError("receive timestamps must be positive")
        parts = pair_protocol.encode_multipart(pair.header, pair.left, pair.right)
        camera_hash = pair.header.camera_serial_sha256.hex()
        calibration_hash = pair.header.calibration_sha256.hex()
        if self._camera_serial_sha256 is None:
            self._camera_serial_sha256 = camera_hash
            self._calibration_sha256 = calibration_hash
        elif (
            camera_hash != self._camera_serial_sha256
            or calibration_hash != self._calibration_sha256
        ):
            raise PairRecorderError("camera or calibration identity changed within recording")

        if self._chunk_stream is None:
            self._begin_chunk()
        offset_bytes = self._chunk_stream.tell()
        record_size_bytes = sum(len(part) for part in parts)
        for part in parts:
            raw = bytes(part)
            self._chunk_stream.write(raw)
            self._chunk_hasher.update(raw)
        if self._chunk_pair_count == 0:
            self._chunk_first_sequence = sequence
        self._chunk_pair_count += 1
        self._chunk_last_sequence = sequence
        self._pair_count += 1
        self._last_sequence = sequence
        self._index_stream.write(
            json.dumps(
                {
                    "chunk": self._chunk_index,
                    "offset_bytes": offset_bytes,
                    "record_size_bytes": record_size_bytes,
                    "sequence": sequence,
                    "camera_frame_number": pair.header.camera_frame_number,
                    "capture_realtime_ns": pair.header.capture_realtime_ns,
                    "receive_realtime_ns": receive_realtime_ns,
                    "receive_monotonic_ns": receive_monotonic_ns,
                },
                sort_keys=True,
            )
            + "\n"
        )
        if self._chunk_pair_count >= self.pairs_per_chunk:
            self._finish_chunk()

    def close(self) -> None:
        if self._closed:
            return
        self._finish_chunk()
        self._index_stream.flush()
        os.fsync(self._index_stream.fileno())
        self._index_stream.close()
        manifest = {
            "schema_version": 1,
            "format": "concatenated_hsspair1_header_left_right",
            "header_size_bytes": pair_protocol.HEADER_SIZE,
            "pairs_per_chunk": self.pairs_per_chunk,
            "pair_count": self._pair_count,
            "camera_serial_sha256": self._camera_serial_sha256,
            "calibration_sha256": self._calibration_sha256,
            "chunks": self._chunks,
        }
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._closed = True

    def __enter__(self) -> PairChunkWriter:  # noqa: PYI034 - Python 3.8 has no typing.Self
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _load_manifest(recording_dir: Path) -> Dict[str, Any]:
    try:
        manifest = json.loads((recording_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairRecorderError(f"cannot load pair manifest: {exc}") from exc
    if manifest.get("schema_version") != 1 or manifest.get("header_size_bytes") != pair_protocol.HEADER_SIZE:
        raise PairRecorderError("recording manifest has an unsupported schema or header size")
    return manifest


def iter_recorded_pairs(recording_dir: Path) -> Iterator[CachedStereoPair]:
    """Verify all chunk hashes, then replay records in index order."""

    manifest = _load_manifest(recording_dir)
    try:
        index_rows = [
            json.loads(line)
            for line in (recording_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise PairRecorderError(f"cannot load pair index: {exc}") from exc
    chunks = {index: chunk for index, chunk in enumerate(manifest["chunks"])}
    chunk_payloads = {}
    for chunk_index, chunk in chunks.items():
        try:
            payload = (recording_dir / chunk["filename"]).read_bytes()
        except OSError as exc:
            raise PairRecorderError(f"cannot read chunk {chunk['filename']}: {exc}") from exc
        if len(payload) != chunk["size_bytes"]:
            raise PairRecorderError(f"chunk {chunk['filename']} size does not match manifest")
        if hashlib.sha256(payload).hexdigest() != chunk["sha256"]:
            raise PairRecorderError(f"chunk {chunk['filename']} SHA-256 does not match manifest")
        chunk_payloads[chunk_index] = payload

    previous_sequence = None
    for row in index_rows:
        try:
            payload = chunk_payloads[row["chunk"]]
            offset = row["offset_bytes"]
            header_bytes = payload[offset : offset + pair_protocol.HEADER_SIZE]
            header = pair_protocol.decode_header(header_bytes)
            left_start = offset + pair_protocol.HEADER_SIZE
            right_start = left_start + header.left_payload_size
            end = right_start + header.right_payload_size
            if end - offset != row["record_size_bytes"]:
                raise PairRecorderError(f"record size mismatch at sequence {row['sequence']}")
            decoded = pair_protocol.decode_multipart(
                (header_bytes, payload[left_start:right_start], payload[right_start:end]),
                previous_sequence=previous_sequence,
            )
        except (KeyError, IndexError, pair_protocol.ProtocolError) as exc:
            raise PairRecorderError(f"invalid pair index row {row!r}: {exc}") from exc
        if decoded.header.sequence != row["sequence"]:
            raise PairRecorderError("index sequence does not match recorded header")
        previous_sequence = decoded.header.sequence
        yield CachedStereoPair(
            header=decoded.header,
            left=decoded.left.tobytes(),
            right=decoded.right.tobytes(),
        )
    if len(index_rows) != manifest["pair_count"]:
        raise PairRecorderError("pair index count does not match manifest")


def record_pairs(
    endpoint: str,
    *,
    duration_s: float,
    output_dir: Path,
    pairs_per_chunk: int,
) -> Dict[str, Any]:
    if duration_s <= 0.0:
        raise PairRecorderError("recording duration must be positive")
    receiver = LatestStereoPairReceiver(endpoint)
    start_ns = time.monotonic_ns()
    deadline_ns = start_ns + round(duration_s * 1_000_000_000)
    try:
        with PairChunkWriter(output_dir, pairs_per_chunk=pairs_per_chunk) as writer:
            while time.monotonic_ns() < deadline_ns:
                pair = receiver.receive_latest(timeout_ms=10)
                if pair is None:
                    continue
                writer.write_pair(
                    pair,
                    receive_realtime_ns=time.time_ns(),
                    receive_monotonic_ns=time.monotonic_ns(),
                )
    finally:
        receiver.close()
    return _load_manifest(output_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: UP045 - G1 uses Python 3.8
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_CONNECT_ENDPOINT)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-chunk", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        manifest = record_pairs(
            args.endpoint,
            duration_s=args.duration_s,
            output_dir=args.output_dir,
            pairs_per_chunk=args.pairs_per_chunk,
        )
    except (PairRecorderError, OSError) as exc:
        print(f"pair recording failed: {exc}")
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["pair_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

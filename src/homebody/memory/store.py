"""Atomic named photo memory with verified portable assets and persistent removal history.

Spatial ranking uses a nearest-view score and a nearest-score absence gate.
Records and removals share one atomic state file, so a failed update never exposes half
of a transaction. Unreferenced assets from an interrupted write remain recoverable.
"""
import copy
import fcntl
import hashlib
import io
import json
import math
import os
import re
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .binding import MapBinding, MemoryError

SCHEMA = "homebody-photo-memory/v1"
STATE = "memory.json"
DEFAULT_MAP_DIAGONAL_M = math.hypot(5, 5)


def _name(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value) is None:
        raise MemoryError("Memory name must contain only letters, digits, underscores and hyphens")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, indent=2).encode() + b"\n"


def _atomic_write(path, data):
    """Replace one file only after its complete contents have reached the filesystem."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _pose(value):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise MemoryError("Memory pose must be three finite map coordinates [x, y, yaw]") from error
    if array.shape != (3,) or not np.isfinite(array).all():
        raise MemoryError("Memory pose must be three finite map coordinates [x, y, yaw]")
    return array.tolist()


def _k(value):
    if type(value) is not int or not 1 <= value <= 8:
        raise MemoryError("recall k must be an integer from 1 to 8")
    return value


class NamedMemory:
    """One named store. Opening it requires the expected deployment map and frame."""

    def __init__(self, path, binding):
        self.path, self.binding = Path(path), binding
        self._mutex = threading.RLock()

    @classmethod
    def open(cls, root, name, binding, *, create=True, map_diagonal_m=DEFAULT_MAP_DIAGONAL_M,
             radius_m=2.0):
        if not isinstance(binding, MapBinding):
            raise MemoryError("An explicit MapBinding is required")
        root = Path(root).expanduser().resolve()
        path = root / _name(name)
        if path.is_symlink():
            raise MemoryError("A named memory directory cannot be a symlink")
        if not path.exists() and not create:
            raise MemoryError(f"Memory set does not exist: {name}")
        path.mkdir(parents=True, exist_ok=True)
        store = cls(path, binding)
        with store._locked():
            if not (path / STATE).exists():
                if any(item.name != ".lock" for item in path.iterdir()):
                    raise MemoryError("Memory metadata is missing; existing files retained for recovery")
                if not create:
                    raise MemoryError("Memory metadata is missing")
                if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in (map_diagonal_m, radius_m)):
                    raise MemoryError("Retrieval map diagonal and radius must be positive and finite")
                store._write({"schema": SCHEMA, "binding": asdict(binding),
                              "retrieval": {"map_diagonal_m": map_diagonal_m,
                                            "radius_m": radius_m, "yaw_weight": 0.30},
                              "records": [], "removals": []})
            store._read()
        return store

    @contextmanager
    def _locked(self):
        with self._mutex:
            if (self.path / ".lock").is_symlink():
                raise MemoryError("Memory lock cannot be a symlink")
            with (self.path / ".lock").open("a+b") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def _write(self, state):
        _atomic_write(self.path / STATE, _json(state))

    def _asset_path(self, reference):
        try:
            name, digest = reference["path"], reference["sha256"]
            size = reference["size_bytes"]
        except (KeyError, TypeError) as error:
            raise MemoryError("Invalid memory asset reference") from error
        if (not isinstance(name, str) or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or type(size) is not int or size < 0
                or name not in (f"assets/{digest}.png", f"assets/{digest}.npy")):
            raise MemoryError("Unsafe or malformed memory asset path")
        path = self.path / name
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise MemoryError(f"Missing or unsafe memory asset: {name}")
        if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise MemoryError(f"Memory asset integrity check failed: {name}")
        return path

    def _read(self):
        path = self.path / STATE
        if path.is_symlink():
            raise MemoryError("Memory metadata cannot be a symlink")
        try:
            state = json.loads(path.read_text())
            self._validate(state)
        except (OSError, ValueError, TypeError, KeyError) as error:
            if isinstance(error, MemoryError):
                raise
            raise MemoryError(f"Invalid memory metadata: {error}") from error
        return state

    def _validate(self, state):
        if state["schema"] != SCHEMA or MapBinding(**state["binding"]) != self.binding:
            raise MemoryError("Memory map/frame binding does not match this deployment")
        retrieval = state["retrieval"]
        if (any(type(retrieval[key]) not in (int, float)
                or not math.isfinite(retrieval[key]) or retrieval[key] <= 0
                for key in ("map_diagonal_m", "radius_m")) or retrieval["yaw_weight"] != 0.30):
            raise MemoryError("Invalid spatial memory retrieval configuration")
        if not isinstance(state["records"], list) or not isinstance(state["removals"], list):
            raise MemoryError("Memory records and removals must be lists")
        ids = set()
        for record in state["records"]:
            if not isinstance(record["id"], str) or record["id"] in ids:
                raise MemoryError("Duplicate or invalid memory record identity")
            ids.add(record["id"])
            _pose(record["pose"])
            if (not isinstance(record["content"], str) or not math.isfinite(record["time"])
                    or record["time"] < 0 or len(record["frame_id"]) != 2
                    or any(type(v) is not int or v < 0 for v in record["frame_id"])):
                raise MemoryError("Invalid memory record provenance")
            image = record["image"]
            if image["media_type"] != "image/png" or not image["path"].endswith(".png"):
                raise MemoryError("Invalid photograph media type")
            image_path = self._asset_path(image)
            try:
                with Image.open(image_path) as photograph:
                    if photograph.format != "PNG" or photograph.mode != "RGB":
                        raise MemoryError("Remembered photographs must be RGB PNG images")
                    image_shape = (photograph.height, photograph.width)
                    photograph.verify()
            except OSError as error:
                raise MemoryError("Remembered photograph is not readable") from error
            if record.get("depth") is not None:
                depth_ref = record["depth"]
                if (depth_ref["media_type"] != "application/x-npy"
                        or not depth_ref["path"].endswith(".npy")):
                    raise MemoryError("Invalid depth media type")
                depth_path = self._asset_path(depth_ref)
                try:
                    depth = np.load(depth_path, allow_pickle=False)
                    if not isinstance(depth, np.ndarray):
                        depth.close()
                        raise MemoryError("Remembered depth must be a single numeric array")
                    if (depth.shape != image_shape or depth.dtype.kind != "f"
                            or np.isinf(depth).any()):
                        raise MemoryError("Invalid remembered metric depth")
                except (ValueError, OSError) as error:
                    raise MemoryError("Remembered depth is not readable metric evidence") from error
                for key, shape in (("intrinsics", (3, 3)), ("world_camera", (4, 4))):
                    value = np.asarray(record["camera"][key])
                    if value.shape != shape or not np.isfinite(value).all():
                        raise MemoryError("Invalid memory camera geometry")
        removed = set()
        for removal in state["removals"]:
            identity = removal["record_id"]
            if identity not in ids or identity in removed:
                raise MemoryError("Invalid removal ledger")
            removed.add(identity)

    def _put(self, data, suffix, media_type):
        digest = hashlib.sha256(data).hexdigest()
        relative = f"assets/{digest}{suffix}"
        folder = self.path / "assets"
        if folder.is_symlink():
            raise MemoryError("Memory assets directory cannot be a symlink")
        folder.mkdir(exist_ok=True)
        reference = {"path": relative, "sha256": digest, "size_bytes": len(data),
                     "media_type": media_type}
        path = self.path / relative
        if path.exists() or path.is_symlink():
            self._asset_path(reference)
        else:
            _atomic_write(path, data)
        return reference

    def remember(self, frame, content=""):
        """Persist the current localized photograph and its optional calibrated depth."""
        self.binding.check_navigation(frame.navigation)
        pose = _pose(frame.base_pose)
        rgb = np.asarray(frame.rgb)
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size
                or not isinstance(content, str)):
            raise MemoryError("Remember requires an RGB uint8 image and a text description")
        if (not math.isfinite(frame.time) or frame.time < 0
                or any(type(v) is not int or v < 0 for v in frame.frame_id)):
            raise MemoryError("Remember requires valid capture time and frame identity")
        image = io.BytesIO()
        Image.fromarray(rgb).save(image, format="PNG")
        depth = getattr(frame, "depth", None)
        camera = None
        depth_bytes = None
        if depth is not None:
            depth = np.asarray(depth, dtype=np.float32)
            if depth.shape != rgb.shape[:2] or np.isinf(depth).any():
                raise MemoryError("Depth must share the image grid and use NaN for invalid readings")
            camera = {}
            for key, shape in (("intrinsics", (3, 3)), ("world_camera", (4, 4))):
                value = np.asarray(getattr(frame, key), dtype=float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise MemoryError("Remember requires valid calibrated camera geometry")
                camera[key] = value.tolist()
            depth_bytes = io.BytesIO()
            np.save(depth_bytes, depth, allow_pickle=False)
        with self._locked():
            state = self._read()
            record = {"id": uuid.uuid4().hex, "pose": pose, "time": float(frame.time),
                      "frame_id": list(frame.frame_id), "content": content,
                      "image": self._put(image.getvalue(), ".png", "image/png"),
                      "depth": None, "camera": camera}
            if depth_bytes is not None:
                record["depth"] = self._put(depth_bytes.getvalue(), ".npy", "application/x-npy")
            state["records"].append(record)
            self._write(state)
        return self._hit(record, 0.0)

    @staticmethod
    def _active(state):
        removed = {item["record_id"] for item in state["removals"]}
        return [record for record in state["records"] if record["id"] not in removed]

    def _hit(self, record, score):
        return {"record_id": record["id"], "score": score, "pose": list(record["pose"]),
                "pose_xy": record["pose"][:2], "yaw": record["pose"][2],
                "time": record["time"], "frame_id": list(record["frame_id"]),
                "content": record["content"], "historical": True,
                "image_path": str(self._asset_path(record["image"])),
                "image": copy.deepcopy(record["image"]), "depth": copy.deepcopy(record["depth"]),
                "camera": copy.deepcopy(record["camera"]), "map_binding": asdict(self.binding)}

    def recall(self, pose=None, *, k=1, op="retrieve"):
        """Retrieve spatial photos or remove those same hits; empty/remote queries are valid."""
        _k(k)
        if op not in ("retrieve", "remove"):
            raise MemoryError("recall op must be retrieve or remove")
        pose = None if pose is None else _pose(pose)
        with self._locked():
            state = self._read()
            records = self._active(state)
            result = {"absent": True, "reason": "No stored photos near this viewpoint",
                      "representation": "keyframe_nn", "operation": op, "hits": []}
            if pose is None:
                result["reason"] = "Spatial photo recall requires a map pose; text search is unavailable"
                return result
            config = state["retrieval"]
            def score(record):
                yaw_error = abs((pose[2] - record["pose"][2] + math.pi) % (2 * math.pi) - math.pi)
                return (math.dist(pose[:2], record["pose"][:2])
                        + config["yaw_weight"] * config["map_diagonal_m"] * yaw_error / math.pi)
            ranked = sorted(((score(record), index, record) for index, record in enumerate(records)))
            if not ranked or ranked[0][0] > config["radius_m"]:
                return result
            hits = [self._hit(record, value) for value, _, record in ranked[:k]]
            if op == "remove":
                state["removals"].extend({"record_id": hit["record_id"], "query_pose": pose}
                                         for hit in hits)
                self._write(state)
            return {**result, "absent": False, "reason": "Historical photos; verify current presence",
                    "hits": hits}

    def records(self):
        with self._locked():
            return copy.deepcopy(self._active(self._read()))

    def load_image(self, hit):
        """Verified RGB pixels for a frozen recall hit, never an arbitrary supplied file path."""
        with self._locked():
            state = self._read()
            if not isinstance(hit, dict) or hit.get("map_binding") != asdict(self.binding):
                raise MemoryError("Recalled photograph belongs to a different map/frame binding")
            record = next((item for item in self._active(state)
                           if item["id"] == hit.get("record_id")), None)
            if record is None or hit.get("image") != record["image"]:
                raise MemoryError("Recalled photograph was removed or its reference changed")
            with Image.open(self._asset_path(record["image"])) as image:
                rgb = np.array(image)
            rgb.setflags(write=False)
            return rgb

    def status(self):
        with self._locked():
            state = self._read()
            return {"name": self.path.name, "path": str(self.path),
                    "binding": asdict(self.binding), "records": len(self._active(state)),
                    "removed": len(state["removals"])}

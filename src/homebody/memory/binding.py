"""Identity of the deployment map and coordinate frame behind remembered poses."""
import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np


class MemoryError(ValueError):
    """A memory operation refused because its data or binding is invalid."""


def _geometry(navigation):
    occupied = np.asarray(navigation.occupied, dtype=np.uint8)
    transform = np.asarray(navigation.T_map_px, dtype="<f8")
    resolution = float(navigation.resolution)
    if (occupied.ndim != 2 or not occupied.size or transform.shape != (3, 3)
            or not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-12
            or not np.isfinite(resolution) or resolution <= 0):
        raise MemoryError("Memory requires a valid mapped navigation frame")
    return occupied, transform, resolution


def geometry_digest(navigation):
    """Hash the grid and its metric coordinates independently of transient obstacles."""
    occupied, transform, resolution = _geometry(navigation)
    digest = hashlib.sha256()
    digest.update(json.dumps([list(occupied.shape), resolution]).encode())
    digest.update(transform.tobytes())
    return digest.hexdigest()


def navigation_digest(navigation):
    """Hash occupancy, pixel-to-world transform and resolution, including array shapes."""
    occupied, transform, resolution = _geometry(navigation)
    digest = hashlib.sha256()
    digest.update(json.dumps([list(occupied.shape), resolution]).encode())
    digest.update(occupied.tobytes())
    digest.update(transform.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class MapBinding:
    """An explicit map asset, named metric frame, and navigation raster identity."""
    scene_id: str
    scene_hash: str
    frame_id: str
    navigation_hash: str
    geometry_hash: str | None = None

    def __post_init__(self):
        for field in ("scene_id", "frame_id"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise MemoryError(f"Memory requires an explicit {field}")
        for field in ("scene_hash", "navigation_hash"):
            value = getattr(self, field)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise MemoryError(f"Memory requires a SHA256 {field}")
        if self.geometry_hash is not None and (not isinstance(self.geometry_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.geometry_hash) is None):
            raise MemoryError("Memory requires a SHA256 geometry_hash")

    @classmethod
    def from_navigation(cls, scene_id, scene_hash, frame_id, navigation):
        source_hash = getattr(navigation, "source_hash", None)
        if source_hash is not None and navigation.frame_id != frame_id:
            raise MemoryError("Navigation frame does not match the memory frame")
        return cls(scene_id, scene_hash, frame_id, source_hash or navigation_digest(navigation),
                   geometry_digest(navigation))

    def check_navigation(self, navigation):
        source_hash = getattr(navigation, "source_hash", None)
        if source_hash is not None:
            if (source_hash != self.navigation_hash or navigation.frame_id != self.frame_id
                    or self.geometry_hash is None or geometry_digest(navigation) != self.geometry_hash):
                raise MemoryError("Current navigation map/frame differs from the memory map binding")
        elif navigation_digest(navigation) != self.navigation_hash:
            raise MemoryError("Current navigation map differs from the memory map binding")

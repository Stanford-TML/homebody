"""Validated map products and explicit localization-to-map registrations."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from homebody.primitives.observations import NavigationMap, ObservationError, frozen_array


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rigid(value, name="transform"):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise ObservationError(f"{name} must be a right-handed rigid transform in metres")
    return frozen_array(matrix)


@dataclass(frozen=True)
class MapBinding:
    path: Path
    sha256: str
    frame_id: str
    navigation: NavigationMap
    raster: np.ndarray

    @classmethod
    def load(cls, path, *, expected_hash, frame_id):
        path = Path(path).resolve()
        actual = sha256(path)
        if not expected_hash or actual != expected_hash:
            raise ObservationError("Map hash differs from the deployment binding")
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ObservationError("A named deployment map frame is required")
        with np.load(path, allow_pickle=False) as data:
            occupancy = np.array(data["occupancy"])
            matrix = np.array(data["T_map_px"], dtype=float)
            if occupancy.ndim != 2 or occupancy.dtype != np.dtype(bool):
                raise ObservationError("Map occupancy must be boolean, True means blocked")
            if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[2], [0, 0, 1], atol=1e-9)):
                raise ObservationError("T_map_px must map raster pixels to map metres")
            gram = matrix[:2, :2].T @ matrix[:2, :2]
            scale = float(np.linalg.norm(matrix[:2, 0]))
            if scale <= 0 or not np.allclose(gram, np.eye(2) * scale**2, rtol=1e-5, atol=1e-10):
                raise ObservationError("Navigation requires square, orthogonal metric pixels")
            raster = (np.array(data["raster"]) if "raster" in data else
                      np.repeat(np.where(occupancy, 0, 255).astype(np.uint8)[..., None], 3, axis=2))
        if raster.shape != (*occupancy.shape, 3) or raster.dtype != np.uint8:
            raise ObservationError("Map raster must be an RGB uint8 image matching occupancy")
        semantics = path.parent / "semantics.json"
        if semantics.is_file():
            declared = json.loads(semantics.read_text()).get("coordinate_frame")
            if isinstance(declared, dict):
                if (declared.get("units", "m") not in {"m", "metres", "meters"}
                        or declared.get("up_axis", "Z").upper() != "Z"
                        or declared.get("handedness", "right") != "right"):
                    raise ObservationError("Deployment maps require right-handed metre Z-up geometry")
                declared = declared.get("name", declared.get("id"))
            if declared and declared != frame_id:
                raise ObservationError("Scene semantics and deployment name different map frames")
        task = path.parent / "task_scene.json"
        if task.is_file():
            named = json.loads(task.read_text()).get("frame")
            if named and named != frame_id:
                raise ObservationError("Task scene and deployment name different map frames")
        return cls(path, actual, frame_id, NavigationMap(occupancy, matrix, scale,
                                                       source_hash=actual, frame_id=frame_id), frozen_array(raster))

    def with_obstacles(self, points, *, floor_height_m, minimum_z=.10, maximum_z=1.80):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ObservationError("Registered obstacle cloud must be finite Nx3")
        if not np.isfinite(floor_height_m):
            raise ObservationError("Obstacle filtering requires the measured map floor height")
        heights = points[:, 2] - floor_height_m
        points = points[(heights >= minimum_z) & (heights <= maximum_z)]
        raster = self.navigation.occupied.copy()
        if len(points):
            pixels = np.column_stack((points[:, :2], np.ones(len(points)))) @ np.linalg.inv(
                self.navigation.T_map_px).T
            uv = np.rint(pixels[:, :2]).astype(int)
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < raster.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < raster.shape[0])
            raster[uv[inside, 1], uv[inside, 0]] = True
        return NavigationMap(raster, self.navigation.T_map_px, self.navigation.resolution,
                             source_hash=self.sha256, frame_id=self.frame_id)


@dataclass(frozen=True)
class Alignment:
    """A session-specific registration, never an inferred identity transform."""
    map_hash: str
    map_frame: str
    anchor_stamp: float
    anchor_pose: np.ndarray
    T_map_source: np.ndarray
    provenance: str

    @classmethod
    def load(cls, path, binding):
        doc = json.loads(Path(path).read_text())
        if doc.get("map_sha256") != binding.sha256 or doc.get("map_frame") != binding.frame_id:
            raise ObservationError("Localization alignment belongs to another map")
        if doc.get("source_frame") != "ws_map":
            raise ObservationError("SuperOdom alignment must name the raw ws_map frame")
        if not doc.get("provenance") or doc.get("units") != "m":
            raise ObservationError("Alignment needs measured provenance and metre units")
        stamp = float(doc["anchor_odom_stamp_s"])
        if not np.isfinite(stamp) or stamp <= 0:
            raise ObservationError("Alignment needs a current registered odometry anchor stamp")
        return cls(binding.sha256, binding.frame_id, stamp,
                   rigid(doc["anchor_T_source_lidar"], "registered anchor"),
                   rigid(doc["T_map_source"], "map registration"), str(doc["provenance"]))

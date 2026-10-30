"""Captured navigation evidence for the shared checked navigation skill."""
from dataclasses import dataclass

import numpy as np

from homebody.helpers.geometry import inverse, points_in
from homebody.primitives.observations import NavigationMap, RobotState, frozen_array


@dataclass(frozen=True)
class NavigationEvidence:
    """Measured navigation state in one explicitly bound map frame.

    Times use Robot.time. Map occupancy includes the mapped structure and current
    obstacles. Registered localization and obstacle acquisition retain separate ages.
    """
    epoch: int
    time: float
    pose_time: float
    obstacle_time: float
    map_time: float
    base_pose: np.ndarray
    navigation: NavigationMap
    obstacles: np.ndarray
    localization_quality: float
    frame_id: str
    map_hash: str
    generation: object
    source: str
    registered: bool
    mobile: bool
    static_navigation: NavigationMap | None = None
    floor_height_m: float = 0.0
    floor_plane: tuple[float, float, float, float] | None = None

    obstacle_state: RobotState | None = None
    obstacle_labels: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "base_pose", frozen_array(self.base_pose, (3,)))
        points = frozen_array(self.obstacles)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Navigation obstacles must be finite Nx3 map-frame points")
        object.__setattr__(self, "obstacles", points)
        if self.obstacle_state is not None and (
                not isinstance(self.obstacle_state, RobotState) or
                not np.isfinite(self.obstacle_state.time) or self.obstacle_state.epoch != self.epoch or
                abs(self.obstacle_state.time - self.obstacle_time) > 1e-6):
            raise ValueError("Obstacle acquisition state must match its epoch and timestamp")
        if self.obstacle_labels is not None:
            labels = frozen_array(self.obstacle_labels, (len(points),))
            if (self.obstacle_state is None or not np.isfinite(labels).all()
                    or not np.equal(labels, np.rint(labels)).all()):
                raise ValueError("Obstacle identities require integer labels and acquisition state")
            object.__setattr__(self, "obstacle_labels", labels)
        if not np.isfinite(self.base_pose).all():
            raise ValueError("Navigation pose must be finite")
        if not all(np.isfinite(value) for value in (
                self.time, self.pose_time, self.obstacle_time, self.map_time,
                self.localization_quality, self.floor_height_m)):
            raise ValueError("Navigation timestamps and quality must be finite")
        if not 0 <= self.localization_quality <= 1:
            raise ValueError("Localization quality must lie in [0,1]")
        if self.floor_plane is not None:
            plane = np.asarray(self.floor_plane, dtype=float)
            if (plane.shape != (4,) or not np.isfinite(plane).all()
                    or not np.isclose(np.linalg.norm(plane[:3]), 1.) or plane[2] <= 0):
                raise ValueError("The floor plane requires a unit upward normal and finite offset")
            object.__setattr__(self, "floor_plane", tuple(float(value) for value in plane))
        if not self.frame_id or not self.map_hash or not self.source:
            raise ValueError("Navigation frame, map hash and source must be explicit")

    def heights_above_floor(self, points=None):
        points = self.obstacles if points is None else np.asarray(points)
        if self.floor_plane is None:
            return points[:, 2] - self.floor_height_m
        normal, offset = np.asarray(self.floor_plane[:3]), self.floor_plane[3]
        return points @ normal + offset


def depth_associated_labels(points, frame, tolerance):
    """Tracked RGB-D identities only where registered scan and optical depth agree."""
    camera = points_in(inverse(frame.world_camera), points)
    labels = np.zeros(len(points), dtype=int)
    visible = np.flatnonzero(np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0))
    projected = camera[visible] @ frame.intrinsics.T
    pixels = np.rint(projected[:, :2] / projected[:, 2, None]).astype(int)
    height, width = frame.depth.shape
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    visible, pixels = visible[inside], pixels[inside]
    u, v = pixels.T
    depth = frame.depth[v, u]
    matched = np.isfinite(depth) & (depth > 0) & (np.abs(depth - camera[visible, 2]) <= tolerance)
    labels[visible[matched]] = np.maximum(frame.labels[v[matched], u[matched]], 0)
    return labels

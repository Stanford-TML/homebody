"""Tracking a target's translation from its visible surfaces, and arm occlusion evidence."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .grasp import Shape, visible_shape

MIN_MATCHED_POINTS = 12
MATCHED_CAPTURE_SHARE = 0.15
MATCHED_VIEW_SHARE = 0.45
ELONGATION_RATIO = 1.5
HEADING_TURN_LIMIT = np.deg2rad(20)
BROAD_SPAN_TOLERANCES = 2
BROAD_SPAN_COVERAGE = 0.5
OCCLUSION_BOX_MARGIN = 0.005
NEARER_DEPTH = 0.01
RAY_DEPTH_SLACK = 0.03


@dataclass(frozen=True)
class TargetDecision:
    mode: str
    shift: np.ndarray
    current_residual: float | None
    reference_residual: float | None
    fit_residual: float | None
    displacement: float


def _cloud(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Target points must be a finite N by 3 array")
    return points


@dataclass(frozen=True)
class ArmOcclusion:
    measured_fraction: float
    explained_missing_fraction: float
    exposed_missing_fraction: float


def _live_depth_along(reference, frame):
    """Each reference point's camera depth, the live depth on its pixel, and whether a
    surface is measured in front of it."""
    camera = frame.world_camera[:3, 3]
    in_camera = (reference - camera) @ frame.world_camera[:3, :3]
    z = in_camera[:, 2]
    safe_z = np.where(z > 0, z, 1.)
    u = np.rint(frame.intrinsics[0, 0] * in_camera[:, 0] / safe_z + frame.intrinsics[0, 2]).astype(int)
    v = np.rint(frame.intrinsics[1, 1] * in_camera[:, 1] / safe_z + frame.intrinsics[1, 2]).astype(int)
    height, width = frame.depth.shape
    inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    observed = frame.depth[np.clip(v, 0, height - 1), np.clip(u, 0, width - 1)]
    nearer = inside & np.isfinite(observed) & (observed > 0) & (observed < z - NEARER_DEPTH)
    return z, observed, nearer


def _slab_crossing(origin, rays, half):
    """Parametric entry and exit of each ray through the box-frame box of half extents HALF."""
    enter = np.full(len(rays), -np.inf)
    leave = np.full(len(rays), np.inf)
    for axis in range(3):
        parallel = np.abs(rays[:, axis]) < 1e-9
        denominator = np.where(parallel, 1., rays[:, axis])
        low = (-half[axis] - origin[axis]) / denominator
        high = (half[axis] - origin[axis]) / denominator
        enter = np.maximum(enter, np.where(parallel, -np.inf, np.minimum(low, high)))
        leave = np.minimum(leave, np.where(parallel, np.inf, np.maximum(low, high)))
        leave = np.where(parallel & (np.abs(origin[axis]) > half[axis]), -np.inf, leave)
    return enter, leave


def arm_occlusion_evidence(reference, current, frame, boxes, tolerance):
    """How much of the missing target is explained by the arm's torso-frame BOXES in front of it."""
    camera = frame.world_camera[:3, 3]
    z, observed, nearer = _live_depth_along(reference, frame)
    measured = np.zeros(len(reference), dtype=bool)
    for box in boxes:
        local_center, local_rotation, half = box[:3]
        center = local_center @ frame.world_torso[:3, :3].T + frame.world_torso[:3, 3]
        rotation = frame.world_torso[:3, :3] @ local_rotation
        enter, leave = _slab_crossing((camera - center) @ rotation, (reference - camera) @ rotation,
                                      half + OCCLUSION_BOX_MARGIN)
        intersects = (leave >= np.maximum(enter, 0.)) & (enter < 1.)
        measured |= (intersects & nearer &
                     (observed >= np.maximum(enter, 0.) * z - RAY_DEPTH_SLACK) &
                     (observed <= np.minimum(leave, 1.) * z + RAY_DEPTH_SLACK))
    missing = (cKDTree(current).query(reference)[0] > tolerance if len(current)
               else np.ones(len(reference), dtype=bool))
    missing_count = max(1, int(np.count_nonzero(missing)))
    return ArmOcclusion(float(np.mean(measured)),
                        float(np.count_nonzero(missing & measured) / missing_count),
                        float(np.count_nonzero(missing & ~nearer) / missing_count))


def _heading(points):
    """Direction of an elongated footprint in map XY; None when it has no long axis."""
    values, vectors = np.linalg.eigh(np.cov(points[:, :2], rowvar=False))
    return vectors[:, -1] if values[-1] > ELONGATION_RATIO * values[0] else None


def _same_heading(reference, current):
    before, after = _heading(reference), _heading(current)
    return before is None or after is None or abs(before @ after) >= np.cos(HEADING_TURN_LIMIT)


class TargetTrack:
    def __init__(self, points, shape_quantiles, tolerance, capture_limit, original_center=None):
        self._points = _cloud(points).copy()
        if not len(self._points):
            raise ValueError("Target tracking requires captured visible points")
        low, high = shape_quantiles
        if not (0 <= low < high <= 1 and np.isfinite(tolerance) and tolerance > 0 and
                np.isfinite(capture_limit) and capture_limit > 0):
            raise ValueError("Invalid target tracking bounds")
        self.quantiles = (low, high)
        self.tolerance, self.capture_limit = tolerance, capture_limit
        self._shape = visible_shape(self._points, self.quantiles)
        self._origin = np.array(self._shape.center if original_center is None else original_center,
                                dtype=float, copy=True)
        if self._origin.shape != (3,) or not np.isfinite(self._origin).all():
            raise ValueError("Original target center must be a finite 3-vector")
        self._shift = np.zeros(3)

    @property
    def shift(self):
        return self._shift.copy()

    @property
    def last_supported_points(self):
        return self._points + self._shift

    @property
    def last_supported_shape(self):
        return Shape(self._shape.center + self._shift, self._shape.lower + self._shift,
                     self._shape.upper + self._shift)

    def _residuals(self, current, reference):
        forward = cKDTree(reference).query(current)[0]
        backward = cKDTree(current).query(reference)[0]
        return tuple(float(np.quantile(distances, self.quantiles[1]))
                     for distances in (forward, backward))

    def _expanded_view_support(self, current, reference):
        """Whether enough distinct matches spread across the capture to support it when
        face visibility has changed."""
        distances, indices = cKDTree(current).query(reference)
        matched = np.unique(indices[distances <= self.tolerance])
        enough = max(MIN_MATCHED_POINTS, int(np.ceil(MATCHED_CAPTURE_SHARE * len(reference))))
        if len(matched) < enough or len(matched) < MATCHED_VIEW_SHARE * len(current):
            return False
        return _same_heading(reference, current) and self._spread_over_broad_extents(reference, current[matched])

    def _spread_over_broad_extents(self, reference, matched):
        spans = np.quantile(reference, self.quantiles[1], axis=0) - np.quantile(
            reference, self.quantiles[0], axis=0)
        for axis in np.argsort(spans)[-2:]:
            if (spans[axis] > BROAD_SPAN_TOLERANCES * self.tolerance and
                    np.ptp(matched[:, axis]) < BROAD_SPAN_COVERAGE * spans[axis]):
                return False
        return True

    def update(self, points):
        """Classify the current points as supported, partial, uncertain or moved, updating
        the shift when a translation of the capture fits."""
        current = _cloud(points)
        displacement = float(np.linalg.norm(self.last_supported_shape.center - self._origin))
        if displacement > self.capture_limit:
            return TargetDecision("moved", self.shift, None, None, None, displacement)
        if not len(current):
            return TargetDecision("uncertain", self.shift, None, None, None, displacement)
        forward, backward = self._residuals(current, self.last_supported_points)
        if forward <= self.tolerance or backward <= self.tolerance:
            mode = "supported" if max(forward, backward) <= self.tolerance else "partial"
            return TargetDecision(mode, self.shift, forward, backward, None, displacement)
        if self._expanded_view_support(current, self.last_supported_points):
            return TargetDecision("partial", self.shift, forward, backward, None, displacement)
        proposal = visible_shape(current, self.quantiles).center - self._shape.center
        fit = max(self._residuals(current, self._points + proposal))
        if fit > self.tolerance:
            return TargetDecision("uncertain", self.shift, forward, backward, fit, displacement)
        displacement = float(np.linalg.norm(self._shape.center + proposal - self._origin))
        if displacement > self.capture_limit:
            return TargetDecision("moved", self.shift, forward, backward, fit, displacement)
        self._shift = proposal
        return TargetDecision("supported", self.shift, forward, backward, fit, displacement)

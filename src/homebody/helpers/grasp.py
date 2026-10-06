"""Grasp hypotheses over an object's visible depth points."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .geometry import inverse, transform


@dataclass(frozen=True)
class Shape:
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def visible_shape(points, quantiles):
    lower, upper = np.quantile(points, quantiles, axis=0)
    return Shape((lower + upper) / 2, lower, upper)


def face_angle(xy):
    """Orientation of the footprint's minimum-area bounding rectangle, None when degenerate."""
    try:
        hull = xy[ConvexHull(xy).vertices]
    except QhullError:
        return None
    edges = np.diff(np.vstack((hull, hull[:1])), axis=0)
    best = None
    for angle in np.arctan2(edges[:, 1], edges[:, 0]):
        c, s = np.cos(angle), np.sin(angle)
        turned = hull @ np.array([[c, s], [-s, c]]).T
        area = float(np.ptp(turned[:, 0]) * np.ptp(turned[:, 1]))
        if best is None or area < best[0]:
            best = (area, angle)
    return best[1]


def closing_angles(points, count):
    """Yaw hypotheses: the footprint's principal axes, its bounding rectangle's faces and a
    uniform fan of COUNT."""
    xy = points[:, :2] - np.mean(points[:, :2], axis=0)
    _, axes = np.linalg.eigh(xy.T @ xy)
    principal = np.arctan2(axes[1, 0], axes[0, 0])
    face = face_angle(xy)
    quarter_turns = np.arange(4) * np.pi / 2
    angles = np.r_[principal + quarter_turns,
                   () if face is None else face + quarter_turns,
                   np.linspace(0, 2 * np.pi, count, endpoint=False)]
    return np.unique(np.mod(angles, 2 * np.pi))


def _closing_directions(points, count, aperture_limits):
    """Horizontal unit closing directions along which the visible width fits a jaw."""
    for angle in closing_angles(points, count):
        closing = np.array([np.cos(angle), np.sin(angle), 0.0])
        if aperture_limits[0] <= np.ptp(points @ closing) <= aperture_limits[1]:
            yield closing


def candidates(points, palm_jaw, aperture_limits, settings):
    """Yield (rank, pre-grasp palm, palm, shape) over the visible points, one per insertion
    tier and fitting closing direction."""
    shape = visible_shape(points, settings.shape_quantiles)
    insertion = min(settings.depth_max, (shape.upper[2] - shape.lower[2]) * settings.depth_fraction)
    down = np.array([0.0, 0.0, -1.0])
    low, high = settings.shape_quantiles
    directions = list(_closing_directions(points, settings.yaw_samples, aperture_limits))
    for tier, scale in enumerate(settings.insertion_depth_scales):
        point = shape.center.copy()
        point[2] = shape.upper[2] - insertion * scale
        band = points[np.abs(points[:, 2] - point[2]) <= settings.pad_band]
        band = band if len(band) >= settings.pad_band_points else points
        for closing in directions:
            across = np.cross(down, closing)
            spread = np.quantile(band @ across, (low, high))
            rank = tier + (0 if spread[1] - spread[0] >= settings.finger_spread
                           else len(settings.insertion_depth_scales))
            centred = point + across * (spread.mean() - point @ across)
            centred += closing * (np.quantile(band @ closing, (low, high)).mean() - centred @ closing)
            rotation = np.column_stack((closing, across, down))
            palm = transform(centred, rotation) @ inverse(palm_jaw)
            pre = palm.copy()
            pre[2, 3] += settings.approach_height
            yield rank, pre, palm, shape


def payload_offset(carried, palm_rotation):
    """The payload centre's offset from the palm and its vertical half extent, for PALM_ROTATION."""
    pose = carried["palm_to_payload"]
    center_offset = palm_rotation @ pose[:3, 3]
    rotation = palm_rotation @ pose[:3, :3]
    return center_offset, float(carried["payload_half"] @ np.abs(rotation[2]))

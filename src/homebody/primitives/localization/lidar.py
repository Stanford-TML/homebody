"""LiDAR self filtering in the registered and exploration start frames."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from homebody.real2sim.exploration.settings import current

FLOOR_BAND_M = 0.10

LIDAR_HEIGHT_M = 1.20  # lidar height above the floor
RING_STOP_M = 0.35  # a lidar point this close in the direction of the move stops the move
RING_BAND_M = (0.25, 1.10)
# Pinned SuperOdom skips registration below squared range 0.01 m^2 but publishes
# those points at the map origin; ORIGIN_DROP_M is that 10 cm radius.
ORIGIN_DROP_M = 0.10
BODY_COLUMN_Z_M = (0.75, LIDAR_HEIGHT_M + 0.15)
ARM_BOX_M = ((-0.35, 0.45), 0.42, (0.40, 1.35))
SELF_LEARN_R_M = 0.45  # near points only: the ring's reach
SELF_LEARN_Z_M = (0.60, 1.35)  # arm and pack heights: legs beside the robot are never learned
SELF_LEARN_WINDOW = 20  # scans (~2 s)
SELF_LEARN_MIN = 16  # hit in this many of the window
SELF_LEARN_MOVE_M = 0.30  # the base must have travelled this far across the window


class SelfLearner:
    """Learns the robot's own cells: a near body-frame cell hit in SELF_LEARN_MIN of the last
    SELF_LEARN_WINDOW scans while the base travelled SELF_LEARN_MOVE_M is masked from then on."""

    def __init__(self):
        self.window: deque = deque(maxlen=SELF_LEARN_WINDOW)  # (cell keys, base xy)
        self.cells: set = set()

    @staticmethod
    def keys(bx, by, z) -> np.ndarray:
        cell = 0.05
        return (
            np.round(bx / cell).astype(np.int64) * 1_000_000
            + np.round(by / cell).astype(np.int64) * 1_000
            + np.round(z / 0.1).astype(np.int64)
        )

    def observe(self, bx, by, z, base_xy) -> None:
        near = (np.hypot(bx, by) < SELF_LEARN_R_M) & (z >= SELF_LEARN_Z_M[0])
        near &= z <= SELF_LEARN_Z_M[1]
        self.window.append((set(self.keys(bx[near], by[near], z[near]).tolist()), base_xy))
        if len(self.window) < SELF_LEARN_WINDOW:
            return
        (x0, y0), (x1, y1) = self.window[0][1], self.window[-1][1]
        if math.hypot(x1 - x0, y1 - y0) < SELF_LEARN_MOVE_M:
            return
        counts: dict = {}
        for cells, _ in self.window:
            for c in cells:
                counts[c] = counts.get(c, 0) + 1
        persistent = {c for c, n in counts.items() if n >= SELF_LEARN_MIN}
        # a (y, z) row spanning many x cells is a wall walked along, not the robot
        rows: dict = {}  # y key -> x keys at any height
        for c in persistent:
            rows.setdefault((c % 1_000_000) // 1_000, set()).add(c // 1_000_000)
        line = {y for y, xs in rows.items() if max(xs) - min(xs) >= 4}  # >= 0.2 m along x
        self.cells |= {c for c in persistent if (c % 1_000_000) // 1_000 not in line}

    def mask(self, bx, by, z) -> np.ndarray:
        """True where a point sits in a learned self cell."""
        if not self.cells or not len(bx):
            return np.zeros(len(bx), bool)
        return np.isin(self.keys(bx, by, z), np.fromiter(self.cells, np.int64))


def ring_ahead_points(points: np.ndarray, x: float, y: float, heading: float, within: float):
    """The body-band lidar points within `within` m in the half-plane the base moves into."""
    if points is None or not len(points):
        return np.zeros((0, 3))
    body = points[(points[:, 2] >= RING_BAND_M[0]) & (points[:, 2] <= RING_BAND_M[1])]
    if not len(body):
        return body
    dx, dy = body[:, 0] - x, body[:, 1] - y
    ahead = (dx * math.cos(heading) + dy * math.sin(heading) > 0.0) & (np.hypot(dx, dy) < within)
    return body[ahead]


def drop_self_points(pts: np.ndarray, T: np.ndarray, floor_z: float) -> np.ndarray:
    """A registered scan without SuperOdom's origin cluster and the robot's own body: the head
    column and the arm box. `pts` are ws_map points, `T` the lidar's ws_map pose (4x4, x forward),
    `floor_z` the floor's ws_map z."""
    origin = np.einsum("ij,ij->i", pts, pts) < ORIGIN_DROP_M**2
    body = (pts - T[:3, 3]) @ T[:3, :3]  # ws_map -> the lidar's frame
    z = pts[:, 2] - floor_z
    column = np.hypot(body[:, 0], body[:, 1]) < current().lidar.body_column_m
    column &= z >= BODY_COLUMN_Z_M[0]
    column &= z < BODY_COLUMN_Z_M[1]
    (x0, x1), half, (z0, z1) = ARM_BOX_M
    arms = (body[:, 0] >= x0) & (body[:, 0] <= x1) & (np.abs(body[:, 1]) <= half)
    arms &= (z >= z0) & (z <= z1)
    return pts[~(origin | column | arms)]


def ring_ahead(points: np.ndarray, x: float, y: float, heading: float) -> float:
    """The nearest body-band lidar point in the half-plane the base is moving into (start
    frame, `heading` = the move's direction), or inf: walking away from something close is
    allowed: a spin that drifted next to an obstacle must still be able to leave it."""
    if points is None or not len(points):
        return math.inf
    body = points[(points[:, 2] >= RING_BAND_M[0]) & (points[:, 2] <= RING_BAND_M[1])]
    if not len(body):
        return math.inf
    dx, dy = body[:, 0] - x, body[:, 1] - y
    ahead = dx * math.cos(heading) + dy * math.sin(heading) > 0.0
    return float(np.min(np.hypot(dx[ahead], dy[ahead]))) if ahead.any() else math.inf


def ring_min(points: np.ndarray, x: float, y: float) -> float:
    if points is None or not len(points):
        return math.inf
    body = points[(points[:, 2] >= RING_BAND_M[0]) & (points[:, 2] <= RING_BAND_M[1])]
    if not len(body):
        return math.inf
    return float(np.min(np.hypot(body[:, 0] - x, body[:, 1] - y)))

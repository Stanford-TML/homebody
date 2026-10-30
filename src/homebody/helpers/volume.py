"""Column evidence for the arm, finger and payload boxes a planar base motion carries over
occupied map cells. The boxes are grown by a margin and sampled; a sample is clear when observed
returns in its column lie beneath it and no nearby return intersects the expanded part.
The space over a seen support is admitted although no sensor sees it from the stance, while
intersecting returns or a column with nothing seen beneath refuse. Registered depth may certify
free space before a measured surface; it never overrides observed occupied heights."""
from collections import deque
from itertools import pairwise

import numpy as np

from .geometry import inverse, points_in

_KEY = 1 << 21  # bin columns per row in a packed bin key


def depth_clears(points, world_camera, intrinsics, depth, valid, margin):
    """Whether each world point lies before all four surrounding valid depth pixels.

    Compare optical depth in the calibrated camera frame. An incomplete pixel neighborhood,
    invalid return, or foreground surface supplies no free-space evidence. This may fill a
    missing-column gap, but must not override occupied columns. Callers establish direct
    rays from this optical origin, frame freshness and the supplied clearance margin;
    endpoints reprojected from another camera cannot certify these rays.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    depth, valid = np.asarray(depth), np.asarray(valid, bool)
    clear = np.zeros(len(points), bool)
    selected = np.flatnonzero(np.isfinite(points).all(axis=1))
    camera = points_in(inverse(world_camera), points[selected])
    forward = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
    selected, camera = selected[forward], camera[forward]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        projected = camera @ np.asarray(intrinsics).T
        uv = projected[:, :2] / projected[:, 2, None]
    height, width = depth.shape
    inside = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
              & (uv[:, 0] < width - 1) & (uv[:, 1] < height - 1))
    selected, camera = selected[inside], camera[inside]
    u, v = np.floor(uv[inside]).astype(int).T
    rows, cols = v[:, None] + (0, 0, 1, 1), u[:, None] + (0, 1, 0, 1)
    depths = depth[rows, cols]
    observed = (valid[rows, cols] & np.isfinite(depths) & (depths > 0)).all(axis=1)
    clear[selected] = observed & (depths.min(axis=1) > camera[:, 2] + margin)
    return clear


class ColumnHistory:
    """Bounded acquisition-frame points; callers own projection and self filtering.

    Retain distinct heights rather than replacing old obstacle hits with newer support.
    Verified held labels retire superseded positions through their acquisition time.
    """

    def __init__(self, seconds=10., period=.1, max_frames=101):
        if (not np.isfinite([seconds, period]).all() or seconds <= 0 or period < 0
                or type(max_frames) is not int or max_frames < 1):
            raise ValueError("History requires positive retention/capacity and nonnegative period")
        self.seconds, self.period, self.max_frames = seconds, period, max_frames
        self.revision = 0
        self._binding, self._bound = None, False
        self._add_time = self._query_time = None
        self._frames = deque()
        self._retired = {}

    @property
    def binding(self):
        return self._binding

    def _clock(self, time, binding, *, query):
        if not np.isfinite(time):
            raise ValueError("History timestamp must be finite")
        previous = self._query_time if query else self._add_time
        if not self._bound or binding != self._binding or (previous is not None and time < previous):
            self._frames.clear()
            self._retired.clear()
            self._binding, self._bound = binding, True
            self._add_time = self._query_time = None
            self.revision += 1
        if query:
            self._query_time = time
        else:
            self._add_time = time
        expired = False
        while self._frames and time - self._frames[0][0] > self.seconds:
            self._frames.popleft()
            expired = True
        if expired:
            self.revision += 1
        self._retired = {label: acquired for label, acquired in self._retired.items()
                         if time - acquired <= self.seconds}

    def add(self, time, binding, points, labels, *, held_labels=()):
        points, labels = np.asarray(points, float), np.asarray(labels)
        held = np.asarray(tuple(held_labels))
        if (points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),)
                or not np.isfinite(points).all() or not np.isfinite(labels).all()
                or not np.equal(labels, np.rint(labels)).all()
                or held.ndim != 1 or not np.isfinite(held).all()
                or not np.equal(held, np.rint(held)).all() or np.any(held <= 0)):
            raise ValueError("History requires finite Nx3 points and one integer label per point")
        self._clock(time, binding, query=False)
        if len(held):
            # Older frames were already retired; only scan newly superseded observations.
            for index, (acquired, previous, ids) in enumerate(self._frames):
                superseded = [label for label in held if acquired > self._retired.get(label, -np.inf)]
                if not superseded:
                    continue
                keep = ~np.isin(ids, superseded)
                if not keep.all():
                    self._frames[index] = acquired, previous[keep], ids[keep]
                    self.revision += 1
            self._retired.update((int(label), time) for label in held)
        if self._frames and time - self._frames[-1][0] + 1e-9 < self.period:
            return False
        # Reused held depth must stay retired even after the carry record is removed.
        keep = ~np.isin(labels, [label for label, acquired in self._retired.items() if time <= acquired])
        self._frames.append((time, points[keep].copy(), labels[keep].copy()))
        while len(self._frames) > self.max_frames:
            self._frames.popleft()
        self.revision += 1
        return True

    def forget(self, label, before):
        """Drop LABEL's returns acquired at or before BEFORE, e.g. a released carry's own returns."""
        frames = deque()
        for time, points, labels in self._frames:
            keep = (labels != label) | (time > before)
            frames.append((time, points[keep], labels[keep]))
        self._frames = frames
        self.revision += 1

    def points(self, now, binding, excluded_labels=()):
        self._clock(now, binding, query=True)
        if not self._frames:
            return np.empty((0, 3))
        return np.vstack([points[~np.isin(labels, excluded_labels)]
                          for _, points, labels in self._frames])


def box_points(boxes, margin, spacing):
    """Grid points at most SPACING apart filling each (center, rotation, half) box grown by MARGIN,
    and the index of the box each belongs to."""
    clouds, owners = [], []
    for index, (center, rotation, half) in enumerate(boxes):
        extent = np.asarray(half, float) + margin
        axes = [np.linspace(-e, e, max(2, int(np.ceil(2 * e / spacing)) + 1)) for e in extent]
        local = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        clouds.append(local @ np.asarray(rotation, float).T + center)
        owners.append(np.full(len(local), index))
    return np.vstack(clouds), np.concatenate(owners)


def in_box(points, box, margin=0.):
    center, rotation, half = box
    local = (points - center) @ np.asarray(rotation, float)
    return np.all(np.abs(local) <= np.asarray(half, float) + margin, axis=1)


def outside(points, boxes, margin):
    """Whether each point lies outside every box grown by MARGIN."""
    points = np.asarray(points)
    boxes = tuple(boxes)
    keep = np.ones(len(points), bool)
    bounds = []
    for center, rotation, half in boxes:
        center, rotation = np.asarray(center), np.asarray(rotation)
        half = np.asarray(half) + margin
        if np.any(half < 0):
            continue
        if not all(np.isfinite(value).all() for value in (center, rotation, half)):
            bounds = None
            break
        try:
            extent = np.abs(np.linalg.inv(rotation)).T @ half
        except np.linalg.LinAlgError:
            bounds = None
            break
        if not np.isfinite(extent).all():
            bounds = None
            break
        # Only a broad-phase filter: rounding slack never replaces the exact box test.
        pad = 1e-9 * (1 + np.abs(center) + extent)
        bounds.append((center - extent - pad, center + extent + pad))
    if bounds is None:
        selected = np.arange(len(points))
    elif not bounds:
        return keep
    else:
        lower = np.min([bound[0] for bound in bounds], axis=0)
        upper = np.max([bound[1] for bound in bounds], axis=0)
        selected = np.flatnonzero(np.all((points >= lower) & (points <= upper), axis=1))
    for box in boxes:
        keep[selected] &= ~in_box(points[selected], box, margin)
    return keep


def carried(points, pose, step):
    """POINTS moved with the base at POSE (x, y, yaw) through the base-frame STEP (ahead, left,
    turn about the base)."""
    x, y, yaw = pose
    ahead, left, turn = step
    c, s, ct, st = np.cos(yaw), np.sin(yaw), np.cos(turn), np.sin(turn)
    moved = np.array(points, float)
    moved[:, :2] = ((moved[:, :2] - (x, y)) @ np.array([[ct, st], [-st, ct]])
                    + (x + c * ahead - s * left, y + s * ahead + c * left))
    return moved


class Columns:
    """Observed returns binned into CELL-wide columns of the map plane; CELL is the spacing the
    boxes are sampled at, so half of it is the height within which a return meets a sample."""

    def __init__(self, hits, cell):
        hits = np.asarray(hits, float).reshape(-1, 3)
        self.hits = hits
        self.cell, self.height = cell, cell / 2 + 1e-3
        keys = self._keys(np.floor(hits[:, :2] / cell).astype(np.int64))
        order = np.argsort(keys, kind="stable")
        self._order = order
        keys = keys[order]
        self._starts = (np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
                        if len(keys) else np.empty(0, dtype=np.intp))
        self._ends = np.r_[self._starts[1:], len(keys)] if len(keys) else self._starts
        self.keys = keys[self._starts]
        self._heights = hits[order, 2]
        self._neighbors = {}

    @staticmethod
    def _keys(bins):
        return (bins[:, 0] + _KEY // 2) * _KEY + bins[:, 1] + _KEY // 2

    def box_hits(self, box, margin, offsets=((0., 0., 0.),)):
        """Broad-phase returns for a fixed-orientation box along ordered world offsets.

        Reuse the column index; callers still apply their exact continuous geometry.
        Unbounded or singular queries leave filtering entirely to that geometry.
        """
        if not len(self.hits):
            return self.hits
        center, rotation, half = (np.asarray(value, float) for value in box)
        offsets = np.asarray(offsets, float)
        try:
            extent = np.abs(np.linalg.inv(rotation)).T @ (half + margin)
        except np.linalg.LinAlgError:
            return self.hits
        pad = 1e-9 * (1 + np.abs(center) + extent + np.abs(offsets).max(axis=0))
        lower = center + offsets.min(axis=0) - extent - pad
        upper = center + offsets.max(axis=0) + extent + pad
        bounds = np.floor(np.array([lower[:2], upper[:2]]) / self.cell)
        if (not np.isfinite(bounds).all() or not np.isfinite([lower, upper]).all()
                or np.any(np.abs(bounds) >= _KEY // 2)
                or bounds[1, 0] - bounds[0, 0] > len(self.keys)):
            return self.hits
        lo, hi = bounds.astype(np.int64)
        x = np.arange(lo[0], hi[0] + 1)
        begin = self._keys(np.column_stack((x, np.full(len(x), lo[1]))))
        end = self._keys(np.column_stack((x, np.full(len(x), hi[1]))))
        starts, ends = np.searchsorted(self.keys, begin), np.searchsorted(self.keys, end, side="right")
        chunks = [self._order[self._starts[a]:self._ends[b - 1]]
                  for a, b in zip(starts, ends) if a < b]
        selected = self.hits[np.concatenate(chunks)] if chunks else self.hits[:0]
        return selected[np.all((selected >= lower) & (selected <= upper), axis=1)]

    def _near(self, bin_xy):
        key = int(self._keys(np.asarray(bin_xy)[None])[0])
        if key not in self._neighbors:
            neighbors = [key + dx * _KEY + dy for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
            neighbors = [neighbor for neighbor in neighbors if -(1 << 63) <= neighbor < (1 << 63)]
            positions = np.searchsorted(self.keys, neighbors)
            found = [self._heights[self._starts[index]:self._ends[index]]
                     for neighbor, index in zip(neighbors, positions)
                     if index < len(self.keys) and self.keys[index] == neighbor]
            self._neighbors[key] = np.sort(np.concatenate(found)) if found else np.empty(0)
        return self._neighbors[key]

    def evidence(self, points):
        """Per-point (seen_at, beneath) flags from neighboring columns."""
        points = np.asarray(points, float).reshape(-1, 3)
        seen_at, beneath = np.zeros(len(points), bool), np.zeros(len(points), bool)
        bins = np.floor(points[:, :2] / self.cell).astype(np.int64)
        unique, inverse = np.unique(bins, axis=0, return_inverse=True)
        for index, bin_xy in enumerate(unique):
            members = np.flatnonzero(inverse.ravel() == index)
            heights = self._near(bin_xy)
            if not len(heights):
                continue
            z = points[members, 2]
            at = np.searchsorted(heights, z)
            # The closest height lies on either side of the insertion point.
            seen_at[members] = np.abs(heights[np.maximum(at - 1, 0)] - z) <= self.height
            seen_at[members] |= np.abs(heights[np.minimum(at, len(heights) - 1)] - z) <= self.height
            beneath[members] = heights[0] < z - self.height
        return seen_at, beneath


def _translation_hits(points, box, offsets, margin, *, witness=False):
    """Any hit (or its first actual return) in a translated grown box along a polyline."""
    center, rotation, half = box
    half = np.asarray(half) + margin
    extent = np.abs(np.linalg.inv(rotation)).T @ half
    lower = center + offsets.min(axis=0) - extent - 1e-12
    upper = center + offsets.max(axis=0) + extent + 1e-12
    points = points[np.all((points >= lower) & (points <= upper), axis=1)]
    if not len(points):
        return None if witness else False
    local = (points - center) @ rotation
    offsets = offsets @ rotation
    if len(offsets) == 1:
        offsets = np.repeat(offsets, 2, axis=0)
    for start, end in pairwise(offsets):
        point, velocity = local - start, start - end
        enter, leave = np.zeros(len(point)), np.ones(len(point))
        valid = np.ones(len(point), bool)
        for axis in range(3):
            if velocity[axis] == 0.:
                valid &= np.abs(point[:, axis]) <= half[axis] + 1e-12
            else:
                a = (-half[axis] - point[:, axis]) / velocity[axis]
                b = (half[axis] - point[:, axis]) / velocity[axis]
                enter = np.maximum(enter, np.minimum(a, b))
                leave = np.minimum(leave, np.maximum(a, b))
        # Closed slabs include tangency; the slack is only floating-point roundoff.
        hit = np.flatnonzero(valid & (enter <= leave + 1e-12))
        if len(hit):
            if not witness:
                return True
            selected = points[hit]
            # Column storage order must not change the reported physical return.
            order = np.lexsort((selected[:, 2], selected[:, 1], selected[:, 0]))
            return selected[order[0]]
    return None if witness else False


def _rotation_hits(columns, box, pose, steps, margin, *, occupied=None, witness=False):
    """Refine chord overlaps; unresolved contact or exhausted work stays blocked."""
    center, rotation, half = map(np.asarray, box)
    radius = np.linalg.norm(np.abs(np.linalg.inv(rotation).T) @ (half + margin))
    center_radius = np.linalg.norm(center[:2] - np.asarray(pose)[:2])
    local_scale = np.linalg.norm(rotation, axis=0).max()
    last_hit = None
    def query(points, fixed, offsets, padding):
        if witness:
            return _translation_hits(points, fixed, offsets, padding, witness=True)
        return _translation_hits(points, fixed, offsets, padding)

    for interval in pairwise(steps):
        pending = [interval]
        # Bound work per interval of STEPS. This is not a clearance tolerance:
        # any overlap still unresolved after the budget continues to refuse.
        for _ in range(64):
            if not pending:
                break
            a, b = pending.pop()
            midpoint = np.asarray(a) / 2 + np.asarray(b) / 2
            delta = abs(b[2] - a[2])
            c, s = np.cos(midpoint[2]), np.sin(midpoint[2])
            turn = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            centers = np.array([carried(center[None], pose, step)[0] for step in (a, b)])
            error = (radius * delta / 2 + center_radius * delta**2 / 8) * local_scale
            fixed, offsets, padding = (centers[0], turn @ rotation, half), centers - centers[0], margin + error
            points = columns.box_hits(fixed, padding, offsets)
            if occupied is not None:
                points = points[occupied(points[:, :2])]
            hit = query(points, fixed, offsets, padding)
            if (hit is None) if witness else not hit:
                continue
            last_hit = hit
            exact = (carried(center[None], pose, midpoint)[0], turn @ rotation, half)
            exact_hit = query(points, exact, np.zeros((1, 3)), margin)
            if (exact_hit is not None) if witness else exact_hit:
                return exact_hit
            # The slab owner's roundoff scale only ends refinement; it cannot
            # turn an unresolved overlap into evidence of clearance.
            if error <= 1e-12 or midpoint[2] in (a[2], b[2]):
                return last_hit if witness else True
            pending.extend(((a, midpoint), (midpoint, b)))
        if pending:
            return last_hit if witness else True
    return None if witness else False


def swept_uncleared(boxes, pose, steps, occupied, columns, *, margin, spacing):
    """Points of BOXES grown by MARGIN, carried through the base-frame STEPS from POSE, over
    OCCUPIED(xy) map cells that the observed COLUMNS do not clear, and the index of the box each
    belongs to. Missing support inside a part's starting volume is exempt. Its initial margin
    is also exempt when motion does not decrease the distance to that world point. These
    exemptions never override observed hits or admit newly entered unknown volume.
    Neighbor-column hits receive a continuous box test for translation, or conservative chord
    enclosures for rotation; ordered STEPS must cover the command and its stopping travel."""
    boxes, steps = tuple(boxes), np.asarray(tuple(steps), float).reshape(-1, 3)
    translation = len(steps) and np.all(steps[:, 2] == steps[0, 2])
    # Positive returns must not depend on a component sample landing in their map cell.
    for index, box in enumerate(boxes):
        if translation:
            center, rotation, half = box
            c, s = np.cos(steps[0, 2]), np.sin(steps[0, 2])
            turn = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            centers = np.array([carried(np.asarray(center)[None], pose, row)[0] for row in steps])
            initial = (centers[0], turn @ rotation, half)
            offsets = centers - centers[0]
            hits = columns.box_hits(initial, margin, offsets)
            hits = hits[occupied(hits[:, :2])]
            hit = _translation_hits(hits, initial, offsets, margin, witness=True)
        else:
            hit = _rotation_hits(columns, box, pose, steps, margin,
                                 occupied=occupied, witness=True)
        if hit is not None:
            return hit[None], np.array([index])
    start, owners = box_points(boxes, margin, spacing)
    lifts = np.linspace(-margin, margin, max(2, int(np.ceil(2 * margin / spacing)) + 1))
    def distance(points, box):
        center, rotation, half = box
        delta = np.abs((points - center) @ rotation) - half
        return np.linalg.norm(np.maximum(delta, 0.), axis=1) + np.minimum(delta.max(axis=1), 0.)

    original_distance = np.empty(len(start))
    for index, box in enumerate(boxes):
        own = owners == index
        original_distance[own] = distance(start[own], box)
    flagged, parts = [], []
    for step in steps:
        points = carried(start, pose, step)
        kept = occupied(points[:, :2])
        points, owner, sample_distance = points[kept], owners[kept], original_distance[kept]
        if not len(points):
            continue
        exempt = np.zeros(len(points), bool)
        for index in np.unique(owner):
            box = boxes[index]
            own = np.flatnonzero(owner == index)
            for lift in lifts:
                exempt[own] |= in_box(points[own] - (0., 0., lift), box, 1e-9)
            # A rigidly carried sample keeps its distance to its moving part. Compare
            # against the initial part so a new command cannot creep into its old margin.
            inherited = in_box(points[own], box, margin + 1e-9)
            exempt[own] |= inherited & (sample_distance[own] >= distance(points[own], box) - 1e-9)
        _seen_at, beneath = columns.evidence(points)
        blocked = ~beneath & ~exempt
        flagged.append(points[blocked])
        parts.append(owner[blocked])
    if not flagged:
        return np.empty((0, 3)), np.empty(0, int)
    return np.vstack(flagged), np.concatenate(parts)

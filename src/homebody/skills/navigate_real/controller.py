"""Checked base navigation to a 2-D map goal (metres, radians).

`prepare` plans a collision-checked route over the inflated occupancy grid (grid A*, RRT
fallback) and resamples its waypoints; `execute` walks them with velocity commands that re-read
the localized pose every poll, turns in place when a waypoint leaves the heading band, then
closes the final gap and takes the arrival heading."""

from __future__ import annotations

import functools
import hashlib
import itertools
import math
import time
from dataclasses import dataclass, field
from dataclasses import replace as _replace

import numpy as np
from scipy import ndimage

from homebody.helpers.geometry import wrap as wrap_angle
from homebody.helpers.raster import world_to_px
from homebody.motion.exploration import localizer
from homebody.motion.exploration.carry import _carry_check, _held_snapshot
from homebody.motion.exploration.gait import (
    GOAL_RADIUS_M,
    LATERAL_SPEED,
    LOC_QUALITY_FLOOR,
    MIN_BURST_S,
    MIN_WALK_GAP_M,
    TICK_HZ,
    TURN_GATE_RAD,
    WALK_SPEED,
    WZ_MIN_ACCEPT,
    announce,
    final_turns,
    finite,
    held_move,
    hold,
    motion_words,
    occupancy_grid,
    pose,
    require_xy,
    step_base,
    turn_speed,
)
from homebody.motion.exploration.gait import burst as _gait_burst
from homebody.motion.exploration.nav import (
    GridCollisionChecker,
    _grid_segment_free,
    path_len,
    plan_grid_astar,
    plan_path,
)
from homebody.motion.exploration.stage import stage
from homebody.primitives.navigation.actions.lower_body import LowerBody
from homebody.primitives.navigation.contracts import DriveFailure, Needs, StaleObservation
from homebody.primitives.navigation.envelope import validate_radius_m
from homebody.primitives.navigation.observations.gripper_state import UNKNOWN
from homebody.real2sim.exploration.settings import current

NEEDS = Needs(
    obs=("base_pose", "map", "loc_quality", "gripper_state", "qpos", "lidar_obstacles"),
    actions=("lower_body",),
    models=(),
)


PLAN_SEEDS: tuple[int, ...] = (0, 17, 41)  # planner seeds, tried in order
DEFAULT_RADIUS_M = 0.33  # route check radius; the parked arms reach 0.31 m
EXPLORATION_CLEARANCE_MARGIN_M = 0.15  # soft preference outside the hard footprint
INTERMEDIATE_RADIUS_M = DEFAULT_RADIUS_M + 0.30  # widest corridor: parked hands plus drift
CORRIDOR_RUNGS = 3  # narrower corridors tried below the widest
CORRIDOR_CENTRE_CAP_M = 0.30  # most one centring pass moves a waypoint
CENTRE_PASSES = 6  # most centring passes
CENTRE_LENGTH_SLACK = 0.18  # fraction centring may lengthen the route
CARRY_SWEEP_RADIUS_M = DEFAULT_RADIUS_M  # sweep radius while carrying
LOC_DRIFT_M = 0.10  # localizer drift added to a transit stance
TRANSIT_STANCE_M = GOAL_RADIUS_M + LOC_DRIFT_M  # footprint a passing stance is judged at
STALL_PROGRESS_M = 0.03   # a gap that closes by less than this has made no progress
STALL_WINDOW_S = 5.0      # no progress for this long is a stall
TURN_WALK_GAP_S = 0.5     # the base stands this long between a turn and a walk
WAYPOINT_FINAL_M = 2.5  # the route's last stretch, sampled at the final spacing
WAYPOINT_FINAL_SPACING_M = 0.5
WAYPOINT_ALIGN_RAD = math.radians(7.0)  # heading gate bounds of a waypoint leg (`_align_gate`)
WAYPOINT_ALIGN_MAX_RAD = math.radians(60.0)
TURN_DRIFT_CAP_M = 0.20       # the translation a turn in place may cost
LOC_RECOVERY_WAIT_S = 3.0     # the longest wait for the localizer before planning
START_BACKUP_M = 0.60         # the backward walk that clears a blocked start
BLIND_ESCAPE_M = 0.20         # the reverse taken when no escape ray is checked free
NO_ROUTE_PATIENCE_S = 5.0     # a stance with no route for this long is left, not waited on
NO_PROGRESS_MAX_S = 90.0      # the replan wait ends after this long without moving
ESCAPE_DIRECTIONS_RAD = tuple(math.radians(v) for v in (180, 0))  # straight back, then ahead


def _align_gate(gap_m: float, tolerance_m: float | None = None) -> float:
    """The heading gate for a leg to land inside the tolerance: `asin(tol / gap)`, clamped to
    [WAYPOINT_ALIGN_RAD, WAYPOINT_ALIGN_MAX_RAD]."""
    gap = float(gap_m)
    tolerance = current().drive.waypoint_tol_m if tolerance_m is None else tolerance_m
    if gap <= tolerance:
        return WAYPOINT_ALIGN_MAX_RAD
    return min(WAYPOINT_ALIGN_MAX_RAD,
               max(WAYPOINT_ALIGN_RAD, math.asin(min(1.0, tolerance / gap))))



def _turn_clearance_safe(checker: GridCollisionChecker, x: float, y: float,
                         drift_m: float | None = None) -> bool:
    """Whether a turn in place may translate anywhere inside its drift disc and stay safe."""
    drift_m = current().drive.final_turn_drift_m if drift_m is None else drift_m
    cell = max(float(checker.cell), 1e-6)
    rings = max(1, math.ceil(float(drift_m) / cell))
    for i in range(-rings, rings + 1):
        for j in range(-rings, rings + 1):
            dx, dy = i * cell, j * cell
            if (dx * dx + dy * dy <= float(drift_m) ** 2 + 1e-12
                    and not checker.is_safe(x + dx, y + dy)):
                return False
    return True


def _checked_escape(checker: GridCollisionChecker, x: float, y: float, yaw: float,
                    want_m: float = START_BACKUP_M) -> tuple[float, float] | None:
    """(distance, body angle) of the ray with the longest free stretch, straight back first,
    then ahead; None when no ray reaches a free cell within `want_m`."""
    step = max(float(checker.cell), 0.02)
    best = None   # (free_run, -exit_distance, offset)
    for offset in ESCAPE_DIRECTIONS_RAD:
        heading = yaw + offset
        exit_d, free_to = None, 0.0
        for i in range(1, math.ceil(want_m / step) + 1):
            d = min(i * step, want_m)
            px, py = x + d * math.cos(heading), y + d * math.sin(heading)
            if not checker.contains(px, py):
                break
            if checker.is_safe(px, py):
                exit_d = d if exit_d is None else exit_d
                free_to = d
            elif exit_d is not None:
                break   # geometry again past free space
        if exit_d is None:
            continue
        key = (free_to - exit_d, -exit_d, offset)
        if best is None or key[:2] > best[:2]:
            best = key
    if best is None:
        return None
    run, neg_exit, offset = best
    return (float(run - neg_exit), float(offset))   # to the end of the free stretch


def _nearest_free(checker, point, radius: float):
    """The nearest safe point to `point` at `radius` within 2 m, at the grid cell, or None."""
    x, y = float(point[0]), float(point[1])
    if checker.is_safe(x, y, radius):
        return (x, y)
    found = checker.nearest_safe(x, y, radius, 2.0, float(checker.cell) or 0.1)
    return None if found is None else (float(found[0]), float(found[1]))


GOAL_SNAP_SLACK_M = 0.10   # how much farther than the nearest free cell a snapped goal may sit


def _nearest_free_toward(checker, point, base_xy) -> tuple[float, float] | None:
    """The free cell closest to the base among those within GOAL_SNAP_SLACK_M of the nearest
    free cell to `point`."""
    px, py = float(point[0]), float(point[1])
    if checker.is_safe(px, py):       # a free request is kept as is
        return (px, py)
    nearest = _nearest_free(checker, point, 0.0)
    if nearest is None:
        return None
    reach = math.hypot(nearest[0] - px, nearest[1] - py) + GOAL_SNAP_SLACK_M
    grid = np.asarray(checker.grid, dtype=bool)
    cell = float(checker.cell)
    i0 = max(0, int((px - reach - checker.xmin) / cell))
    i1 = min(checker.nx - 1, int((px + reach - checker.xmin) / cell) + 1)
    j0 = max(0, int((py - reach - checker.ymin) / cell))
    j1 = min(checker.ny - 1, int((py + reach - checker.ymin) / cell) + 1)
    ii, jj = np.nonzero(grid[i0:i1 + 1, j0:j1 + 1])
    if ii.size == 0:
        return (float(nearest[0]), float(nearest[1]))
    xs, ys = checker.xmin + (ii + i0) * cell, checker.ymin + (jj + j0) * cell
    near = np.hypot(xs - px, ys - py) <= reach
    if not near.any():
        return (float(nearest[0]), float(nearest[1]))
    xs, ys = xs[near], ys[near]
    k = int(np.argmin(np.hypot(xs - float(base_xy[0]), ys - float(base_xy[1]))))
    return (float(xs[k]), float(ys[k]))


GOAL_BACKOFF_MAX_M = 1.20
GOAL_BACKOFF_STEP_M = 0.05


def _usable_goal(art, goal_xy, facing_xy, radius_m: float, base_xy=None):
    """The stance for a goal inside its own inflation: the nearest free cell unless it puts the
    target behind the stance, else a back-off along the facing ray."""
    if facing_xy is None:
        return None
    checker, _ = occupancy_grid(art, GOAL_RADIUS_M)  # endpoint radius
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    if checker.is_safe(gx, gy):
        return None
    nearest = (_nearest_free(checker, (gx, gy), GOAL_RADIUS_M) if base_xy is None
               else _nearest_free_toward(checker, (gx, gy), base_xy))
    if nearest is not None:
        fx, fy = float(facing_xy[0]), float(facing_xy[1])
        before = (fx - gx, fy - gy)
        after = (fx - nearest[0], fy - nearest[1])
        if before[0] * after[0] + before[1] * after[1] > 0.0:
            return (float(nearest[0]), float(nearest[1]))
    return _back_off_along_facing(art, goal_xy, facing_xy, radius_m)


BOUNDARY_STEP_M = 0.02  # step of the walk out to the target's padded boundary


def _on_padded_boundary(checker, target_xy, goal_xy, cap_m: float):
    """The first clear point walking from `target_xy` toward `goal_xy` within `cap_m`, or None."""
    tx, ty = (float(v) for v in target_xy)
    gx, gy = (float(v) for v in goal_xy)
    span = math.hypot(gx - tx, gy - ty)
    if span <= 1e-6:
        return None
    ux, uy = (gx - tx) / span, (gy - ty) / span
    reach = min(span, float(cap_m))
    steps = max(1, int(reach / BOUNDARY_STEP_M))
    for k in range(1, steps + 1):
        d = k * BOUNDARY_STEP_M
        px, py = tx + ux * d, ty + uy * d
        safe = (_grid_segment_free(checker, (px, py), (px, py))
                if isinstance(checker, GridCollisionChecker) else checker.is_safe(px, py))
        if safe:
            return (px, py)
    return None


STANCE_REACH_M = 0.42    # the arm's forward reach
STANCE_SEARCH_M = 1.5    # how far round a marker its nearest free cell is looked for
STANCE_MAX_SWING_RAD = math.radians(90.0)  # preferred band either side of the clicked bearing


def _reachable_stance(checker, target_xy, goal_xy, reach_m: float):
    """The free cell within `reach_m` of the target nearest the clicked goal, clicked side first;
    with no goal or an empty band, the free cell nearest the target; None when there is none."""
    tx, ty = (float(v) for v in target_xy)
    reach = float(reach_m)
    if reach <= GOAL_RADIUS_M:
        return None
    grid = np.asarray(checker.grid, dtype=bool)
    cell = float(checker.cell)
    edge = (reach if goal_xy is not None else STANCE_SEARCH_M) + cell  # no goal: wider search
    i0 = max(0, int((tx - edge - checker.xmin) / cell))
    i1 = min(checker.nx - 1, int((tx + edge - checker.xmin) / cell) + 1)
    j0 = max(0, int((ty - edge - checker.ymin) / cell))
    j1 = min(checker.ny - 1, int((ty + edge - checker.ymin) / cell) + 1)
    if i1 < i0 or j1 < j0:
        return None
    ii, jj = np.nonzero(grid[i0:i1 + 1, j0:j1 + 1])
    if ii.size == 0:
        return None
    xs = checker.xmin + (ii + i0) * cell
    ys = checker.ymin + (jj + j0) * cell
    span = np.hypot(xs - tx, ys - ty)
    band = (span >= GOAL_RADIUS_M) & (span <= reach)
    if goal_xy is None or not band.any():
        outer = span >= GOAL_RADIUS_M  # the free cell closest to the target
        if not outer.any():
            return None
        k = int(np.argmin(np.where(outer, span, np.inf)))
        return (float(xs[k]), float(ys[k]))
    xs, ys = xs[band], ys[band]
    bearing = math.atan2(float(goal_xy[1]) - ty, float(goal_xy[0]) - tx)
    off = np.abs((np.arctan2(ys - ty, xs - tx) - bearing + np.pi) % (2.0 * np.pi) - np.pi)
    near = np.hypot(xs - float(goal_xy[0]), ys - float(goal_xy[1]))
    chosen_side = off <= STANCE_MAX_SWING_RAD
    ranked = np.where(chosen_side, near, np.inf) if chosen_side.any() else near
    k = int(np.argmin(ranked))
    return (float(xs[k]), float(ys[k]))


FACE_WINDOW_M = 2.00       # the target is looked for this far around the faced point
FACE_BAND_M = 0.08         # the near face is sampled this deep
FACE_MAX_SPAN_M = 2.50     # wider than this is a wall, not a face, and is left alone
FACE_MAX_SHIFT_M = 0.75    # the most the stance slides along the face


def _face_centre(art, faced, goal):
    """The middle and outward normal of the face of the component under `faced` that the click
    approaches, from its principal axes; or None."""
    fx, fy = float(faced[0]), float(faced[1])
    ux, uy = float(goal[0]) - fx, float(goal[1]) - fy
    span = math.hypot(ux, uy)
    if span <= 1e-6:
        return None
    ux, uy = ux / span, uy / span
    raw, _ = occupancy_grid(art, 0.0)
    blocked = ~np.asarray(raw.grid, dtype=bool)
    cell = float(raw.cell)
    i0 = max(0, int((fx - FACE_WINDOW_M - raw.xmin) / cell))
    i1 = min(raw.nx - 1, int((fx + FACE_WINDOW_M - raw.xmin) / cell))
    j0 = max(0, int((fy - FACE_WINDOW_M - raw.ymin) / cell))
    j1 = min(raw.ny - 1, int((fy + FACE_WINDOW_M - raw.ymin) / cell))
    window = blocked[i0:i1 + 1, j0:j1 + 1]
    if not window.any():
        return None
    labels, _ = ndimage.label(window, structure=np.ones((3, 3), dtype=bool))
    ii, jj = np.nonzero(window)
    xs, ys = raw.xmin + (ii + i0) * cell, raw.ymin + (jj + j0) * cell
    seed = int(np.argmin(np.hypot(xs - fx, ys - fy)))
    keep = labels[ii, jj] == labels[ii[seed], jj[seed]]
    pts = np.stack((xs[keep], ys[keep]), axis=1)
    if pts.shape[0] < 4:
        return None
    mean = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - mean).T))
    normals = [sign * vecs[:, k] for k in (0, 1) for sign in (1.0, -1.0)]
    normal = max(normals, key=lambda v: float(v[0] * ux + v[1] * uy))
    nx, ny = float(normal[0]), float(normal[1])
    along = (pts[:, 0] - mean[0]) * nx + (pts[:, 1] - mean[1]) * ny
    across = -(pts[:, 0] - mean[0]) * ny + (pts[:, 1] - mean[1]) * nx
    near = along >= float(along.max()) - max(FACE_BAND_M, 3.0 * cell)
    lo, hi = float(across[near].min()), float(across[near].max())
    middle = 0.5 * (lo + hi)
    pointed = -(fx - mean[0]) * ny + (fy - mean[1]) * nx
    if hi - lo > FACE_MAX_SPAN_M or abs(middle - pointed) > FACE_MAX_SHIFT_M:
        return None
    reach = float(along[near].max())
    return ((float(mean[0] + nx * reach - ny * middle),
             float(mean[1] + ny * reach + nx * middle)), (nx, ny))


def _back_off_along_facing(art, goal_xy, facing_xy, radius_m: float):
    """Slide a goal that sits on an obstacle straight back along its facing ray, or None."""
    if facing_xy is None:
        return None
    checker, _ = occupancy_grid(art, float(radius_m))
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    if checker.is_safe(gx, gy) and _turn_clearance_safe(checker, gx, gy):
        return None
    fx, fy = float(facing_xy[0]), float(facing_xy[1])
    away = math.hypot(gx - fx, gy - fy)
    if away < 1e-6:
        return None
    ux, uy = (gx - fx) / away, (gy - fy) / away
    steps = int(GOAL_BACKOFF_MAX_M / GOAL_BACKOFF_STEP_M)
    for i in range(1, steps + 1):
        d = i * GOAL_BACKOFF_STEP_M
        px, py = gx + ux * d, gy + uy * d
        if checker.is_safe(px, py) and _turn_clearance_safe(checker, px, py):
            return (px, py)
    return None


@dataclass(frozen=True)
class _Route:
    """One planning outcome: `code` is "" on success, else START_BLOCKED, GOAL_BLOCKED or
    NO_PATH, all reported as NO_COLLISION_FREE_PATH."""

    path: list[tuple[float, float]] | None
    code: str
    requested_start: tuple[float, float]
    requested_goal: tuple[float, float]
    start: tuple[float, float] | None
    goal: tuple[float, float] | None
    seeds_tried: tuple[int, ...]
    reason: str
    diagnostic: str
    checker: GridCollisionChecker | None = field(repr=False)
    landing: GridCollisionChecker | None = field(repr=False)  # endpoint radius: where it may stand
    transit: GridCollisionChecker | None = field(repr=False)  # body plus drift: where it may pass
    faced: tuple[float, float] | None = None  # the point to face; the stance walks out from it
    clearance_margin_m: float = 0.

    def __post_init__(self) -> None:
        for name in ("requested_start", "requested_goal", "start", "goal", "faced"):
            point = getattr(self, name)
            if point is not None:
                object.__setattr__(self, name, (float(point[0]), float(point[1])))
        if self.path is not None:
            object.__setattr__(self, "path",
                               [(float(x), float(y)) for x, y in self.path])

    @property
    def facing(self) -> float:
        """The arrival heading: toward the facing point, else along snapped -> requested goal,
        else the direction of travel."""
        if self.goal is None:
            return 0.0
        if self.faced is not None:
            dx = self.faced[0] - self.goal[0]
            dy = self.faced[1] - self.goal[1]
            if math.hypot(dx, dy) > 1e-9:
                return math.atan2(dy, dx)
        dx = self.requested_goal[0] - self.goal[0]
        dy = self.requested_goal[1] - self.goal[1]
        if math.hypot(dx, dy) <= 1e-9:
            dx = self.goal[0] - self.requested_start[0]
            dy = self.goal[1] - self.requested_start[1]
        return math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-9 else 0.0


def _without_grids(route: _Route) -> _Route:
    """The route without its grids; `execute` rebuilds them from the map it reads."""
    return _replace(route, checker=None, landing=None, transit=None)


def _with_grids(route: _Route, art, radius_m: float) -> _Route:
    """The route's three checkers, rebuilt from `art`."""
    if route.checker is not None:
        return route
    checker, _ = occupancy_grid(art, float(radius_m))
    landing, _ = occupancy_grid(art, GOAL_RADIUS_M)
    transit, _ = occupancy_grid(art, TRANSIT_STANCE_M)
    return _replace(route, checker=checker, landing=landing, transit=transit)


def _line_free(checker, a, b) -> bool:
    """Whether the straight segment a->b stays on cells the checker calls safe."""
    if isinstance(checker, GridCollisionChecker):
        return _grid_segment_free(checker, a, b)
    ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    n = max(2, int(math.hypot(bx - ax, by - ay) / max(float(checker.cell) * 0.5, 1e-3)) + 1)
    return all(checker.is_safe(ax + (bx - ax) * k / (n - 1), ay + (by - ay) * k / (n - 1))
               for k in range(n))


def _with_samples(pts, spacing: float) -> list:
    """`pts` with a point every `spacing` along each longer leg, endpoints kept."""
    out = [tuple(float(v) for v in pts[0])]
    for a, b in itertools.pairwise(pts):
        leg = math.dist(a, b)
        n = math.floor(leg / spacing)
        for k in range(1, n + 1):
            f = k * spacing / leg
            if leg - k * spacing < 1e-6:
                break
            out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
        out.append(tuple(float(v) for v in b))
    return out


def _sparse(path, checkers):
    """The corners the corridor forces: from each kept point, the farthest later point a straight
    line reaches on safe cells."""
    pts = [(float(a), float(b)) for a, b in path]
    if len(pts) <= 2:
        return pts
    kept, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not any(_line_free(c, pts[i], pts[j]) for c in checkers):
            j -= 1
        kept.append(pts[j])
        i = j
    return kept


def _clearance_of(point, rungs) -> float:
    """The widest corridor radius `point` still stands clear in, 0.0 in none of them."""
    for radius, checker in rungs:
        if checker.is_safe(float(point[0]), float(point[1]), 0.0):
            return float(radius)
    return 0.0


def _padding_field(art):
    """(the raw grid, metres from each free cell to the nearest mapped obstacle)."""
    raw, _bounds = occupancy_grid(art, 0.0)
    free = np.asarray(raw.grid, dtype=bool)
    # no obstacle at all: infinite padding, the map edge is not a wall
    if free.all():
        return raw, np.full(free.shape, np.inf)
    return raw, ndimage.distance_transform_edt(free) * float(raw.cell)


def _padding_at(pad, x: float, y: float) -> float:
    """The padding under (x, y) in metres; off the grid is no padding at all."""
    raw, field = pad
    i = round((float(x) - raw.xmin) / raw.cell)
    j = round((float(y) - raw.ymin) / raw.cell)
    if not (0 <= i < raw.nx and 0 <= j < raw.ny):
        return 0.0
    return float(field[i, j])


def _route_len(pts) -> float:
    return sum(math.dist(a, b) for a, b in itertools.pairwise(pts))


def _centred(pts, rungs, legs, pad=None):
    """The intermediate waypoints moved across the route up the padding ridge, pass after pass
    until none moves, within `CENTRE_LENGTH_SLACK` and with both legs walkable."""
    out = [tuple(float(v) for v in p) for p in pts]
    if len(out) < 3:
        return out
    score = ((lambda p: _padding_at(pad, p[0], p[1])) if pad is not None
             else (lambda p: _clearance_of(p, rungs)))
    floor = 0.5 * float(pad[0].cell) if pad is not None else CORRIDOR_CENTRE_CAP_M / 3.0
    budget = _route_len(out) * (1.0 + CENTRE_LENGTH_SLACK)
    walkable = lambda a, b: any(_line_free(c, a, b) for _r, c in rungs + legs)
    for _pass in range(CENTRE_PASSES):
        shifted = False
        for i in range(1, len(out) - 1):
            here, best_clear = out[i], score(out[i])
            step, best = CORRIDOR_CENTRE_CAP_M, here
            chord = (out[i + 1][0] - out[i - 1][0], out[i + 1][1] - out[i - 1][1])
            span = math.hypot(*chord)
            if span < 1e-6:
                continue
            across = (-chord[1] / span, chord[0] / span)
            while step >= floor:
                gained = False
                for sign in (1.0, -1.0):
                    cand = (best[0] + sign * step * across[0], best[1] + sign * step * across[1])
                    if score(cand) <= best_clear:
                        continue
                    grown = (math.dist(out[i - 1], cand) + math.dist(cand, out[i + 1])
                             - math.dist(out[i - 1], here) - math.dist(here, out[i + 1]))
                    if _route_len(out) + grown > budget:
                        continue
                    if walkable(out[i - 1], cand) and walkable(cand, out[i + 1]):
                        best, best_clear, gained = cand, score(cand), True
                if not gained:
                    step *= 0.5
            if best is not here:
                out[i], shifted = best, True
        if not shifted:
            break
    return out


def _plan_route(art, start_xy, goal_xy, radius_m: float,
                goal_radius_m: float | None = None,
                start_radius_m: float | None = None, snap: bool = True,
                facing_xy=None, face_centre: bool = True,
                preferred_clearance: bool = True, clearance_margin_m: float = 0.,
                cancelled=None, exact_footprint: bool = False) -> _Route:
    """Plan a route from `start_xy` to `goal_xy`, retrying pinned seeds and narrower corridors;
    with `snap` the goal moves to a free stance, else a blocked goal is GOAL_BLOCKED."""
    def check_cancelled():
        if cancelled is not None and cancelled():
            raise DriveFailure("TIMEOUT", "drive deadline expired during route planning")

    check_cancelled()
    geometry = {"exact_footprint": True} if exact_footprint else {}
    checker, bounds = occupancy_grid(art, radius_m, **geometry)
    requested_start = (float(start_xy[0]), float(start_xy[1]))
    requested_goal = (float(goal_xy[0]), float(goal_xy[1]))
    faced = None if facing_xy is None else (float(facing_xy[0]), float(facing_xy[1]))
    radius = float(radius_m)
    goal_radius = float(radius if goal_radius_m is None else goal_radius_m)
    fixed_endpoints = not snap and start_radius_m == radius and goal_radius == radius
    goal_checker = occupancy_grid(art, goal_radius, **geometry)[0]
    transit_checker, transit_bounds = occupancy_grid(art, TRANSIT_STANCE_M, **geometry)
    start_checker = (checker if start_radius_m is None or start_radius_m == radius
                     else occupancy_grid(art, float(start_radius_m), **geometry)[0])
    start = ((float(start_xy[0]), float(start_xy[1]))
             if start_checker.is_safe(*requested_start) else None)
    walk_out = None
    if start is None:
        # a start inside an inflation walks out to the closest free stance
        snapped = _nearest_free(transit_checker, requested_start, TRANSIT_STANCE_M)
        if snapped is not None:
            start = (float(snapped[0]), float(snapped[1]))
            walk_out = (float(requested_start[0]), float(requested_start[1]))
    if snap:
        # with a facing point, the stance goes on the target's padded boundary, face-centred
        goal = None
        requested_faced = faced  # the reach test keeps the named point
        if faced is not None:
            cap = math.dist(faced, requested_goal)
            face = _face_centre(art, faced, requested_goal) if face_centre else None
            if face is not None and cap > 1e-6:
                centre, normal = face
                aim = (centre[0] + normal[0] * cap, centre[1] + normal[1] * cap)
                goal = _on_padded_boundary(goal_checker, centre, aim, cap)
                if goal is not None:
                    faced = (float(centre[0]), float(centre[1]))  # square to the face
            if goal is None:
                goal = _on_padded_boundary(goal_checker, faced, requested_goal, cap)
        if goal is None:
            goal = _nearest_free_toward(goal_checker, requested_goal, requested_start)
        # a stance out of the arm's reach is replaced from the target's reachable band
        if requested_faced is not None and (
                goal is None or math.dist(goal, requested_faced) > STANCE_REACH_M):
            near = _reachable_stance(goal_checker, requested_faced, requested_goal,
                                     STANCE_REACH_M)
            held = None if goal is None else math.dist(goal, requested_faced)
            if near is not None and (held is None
                                     or math.dist(near, requested_faced) < held):
                goal, faced = near, requested_faced
    else:
        goal = requested_goal if goal_checker.is_safe(*requested_goal) else None
    if start is None or goal is None:
        which = "start" if start is None else "goal"
        if which == "start":
            reason = (
                f"initial pose is inside the {goal_radius:.2f} m endpoint footprint of mapped "
                "geometry; route planning requires a checked translation out of it first"
            )
        else:
            reason = (
                f"goal blocked: no free stance within 2.00 m at radius {radius:.2f} m"
            )
        return _Route(
            None, which.upper() + "_BLOCKED", requested_start, requested_goal, start, goal,
            (), reason, "", checker, goal_checker, transit_checker, faced,
        )
    tried: list[int] = []
    if clearance_margin_m and isinstance(checker, GridCollisionChecker):
        path = plan_grid_astar(start, goal, checker, cancelled=cancelled,
                               clearance_margin_m=clearance_margin_m)
        if cancelled is not None and cancelled():
            raise DriveFailure("TIMEOUT", "drive deadline expired during route planning")
        if path is not None:
            return _Route(path, "", requested_start, requested_goal, start, goal, (), "", "",
                          checker, goal_checker, transit_checker, faced, clearance_margin_m)
    # interior corridor widths, widest first; start and goal keep their radii
    adds = (sorted({round((INTERMEDIATE_RADIUS_M - radius) * k / CORRIDOR_RUNGS, 4)
                    for k in range(1, CORRIDOR_RUNGS + 1)}, reverse=True)
            if preferred_clearance else [])
    rungs: list[tuple[float, object]] = []
    for add in adds:
        check_cancelled()
        wide_checker, wide_bounds = occupancy_grid(art, radius + add, **geometry)
        rungs.append((radius + add, wide_checker))
        for seed in PLAN_SEEDS:
            tried.append(int(seed))
            check_cancelled()
            path = plan_path(start, goal, wide_checker, seed=int(seed), bounds=wide_bounds,
                             cancelled=cancelled)
            check_cancelled()
            if path is None:
                continue
            # sample and centre the interior; endpoints are the validated start and goal
            interior = _sparse(path, (wide_checker, checker))
            interior = _with_samples(interior, current().drive.waypoint_spacing_m)
            interior = _centred(interior, rungs, [(radius, checker)], _padding_field(art))[1:-1]
            path_start = (float(start[0]), float(start[1]))
            path_goal = (float(goal[0]), float(goal[1]))
            candidate = [*([walk_out] if walk_out else []), path_start, *interior, path_goal]
            # fixed endpoints: every leg must fit the requested radius
            if fixed_endpoints and any(
                    not _line_free(checker, a, b) for a, b in itertools.pairwise(candidate)):
                # fall back to the planner's own interior
                candidate = [*([walk_out] if walk_out else []), path_start, *path[1:-1], path_goal]
                if any(not _line_free(checker, a, b) for a, b in itertools.pairwise(candidate)):
                    continue
            return _Route(
                candidate, "",
                requested_start, requested_goal, start, goal, tuple(tried), "", "", checker,
                goal_checker, transit_checker, faced,
            )
    tiers = [(checker, bounds, "")]
    if TRANSIT_STANCE_M < radius:
        tiers.append((transit_checker, transit_bounds, "TIGHT"))  # body plus drift
    # the body alone comes last
    for body_r in sorted({round(r, 4) for r in (radius - 0.05, GOAL_RADIUS_M)}, reverse=True):
        if body_r < min(radius, TRANSIT_STANCE_M) - 1e-6 and body_r >= GOAL_RADIUS_M - 1e-6:
            body_checker, body_bounds = occupancy_grid(art, body_r, **geometry)
            tiers.append((body_checker, body_bounds, f"BODY_{body_r:.2f}"))
    for tier_checker, tier_bounds, tier in tiers:
        check_cancelled()
        for seed in PLAN_SEEDS:
            tried.append(int(seed))
            check_cancelled()
            path = plan_path(start, goal, tier_checker, seed=int(seed), bounds=tier_bounds,
                             cancelled=cancelled)
            check_cancelled()
            if path is None:
                continue
            # endpoints are the validated start and goal
            candidate = [*([walk_out] if walk_out else []), (float(start[0]), float(start[1])),
                         *_sparse(path, (tier_checker,))[1:-1],
                         (float(goal[0]), float(goal[1]))]
            if fixed_endpoints and any(
                    not _line_free(checker, a, b) for a, b in itertools.pairwise(candidate)):
                candidate = [*([walk_out] if walk_out else []),
                             (float(start[0]), float(start[1])), *path[1:-1],
                             (float(goal[0]), float(goal[1]))]
                if any(not _line_free(checker, a, b) for a, b in itertools.pairwise(candidate)):
                    continue
            return _Route(
                candidate, "", requested_start,
                requested_goal, start, goal, tuple(tried), "", tier, checker, goal_checker,
                transit_checker, faced,
            )
    check_cancelled()
    diagnostic = ""
    relaxed = max(0.15, radius - 0.10)
    if relaxed < radius - 1e-6:
        loose, loose_bounds = occupancy_grid(art, relaxed, **geometry)
        # diagnostic only: a snapped start separates clearance from disconnection
        d_start = _nearest_free(loose, requested_start, relaxed)
        d_goal = _nearest_free(loose, requested_goal, relaxed)
        d_path = None
        if d_start is not None and d_goal is not None:
            d_path = plan_path(d_start, d_goal, loose, seed=int(PLAN_SEEDS[0]),
                               bounds=loose_bounds, cancelled=cancelled)
            check_cancelled()
        diagnostic = (
            f"clearance-limited: a diagnostic-only route exists at radius {relaxed:.2f} m, "
            f"but the requested {radius:.2f} m footprint remains enforced"
            if d_path is not None else
            f"no connected route was found even at diagnostic radius {relaxed:.2f} m"
        )
    return _Route(
        None, "NO_PATH", requested_start, requested_goal, start, goal, tuple(tried),
        f"no path after {len(tried)} seeded attempts", diagnostic, checker, goal_checker,
        transit_checker, faced,
    )


POSE_UNSAFE_RETRY_S = 1.0   # standing time before re-observing the map and planning again


def _replan_until_found(ctx, goal_xy, radius_m: float) -> _Route:
    """Re-observe the map and plan from the current pose until a route connects, escaping an
    unsafe stance; TIMEOUT at the deadline or after NO_PROGRESS_MAX_S without moving."""
    no_route = 0
    moved_at, moved_pose = float(ctx.now()), None
    stage(ctx, "planning the route again", "")
    while True:
        if ctx.deadline.expired():
            x, y, _ = pose(ctx)
            raise DriveFailure(
                "TIMEOUT",
                f"pose ({x:.3f}, {y:.3f}) found no connecting route before the deadline")
        x, y, yaw = pose(ctx)
        if moved_pose is None or math.hypot(x - moved_pose[0], y - moved_pose[1]) > 0.05 \
                or abs(wrap_angle(yaw - moved_pose[2])) > 0.05:
            moved_at, moved_pose = float(ctx.now()), (x, y, yaw)
        elif float(ctx.now()) - moved_at > NO_PROGRESS_MAX_S:
            raise DriveFailure(
                "TIMEOUT",
                f"pose ({x:.3f}, {y:.3f}) found no connecting route and no escape moved the "
                f"base for {NO_PROGRESS_MAX_S:.0f} s")
        art = ctx.observe("map")
        checker, _ = occupancy_grid(art, TRANSIT_STANCE_M)   # a transit stance: body + drift
        if not checker.contains(x, y):
            # off the map: face the goal, walk once toward it, look again
            bx0, bx1, by0, by1 = checker.bounds()
            ctx.log("drive.pose_off_map", {
                "pose": [round(x, 4), round(y, 4), round(yaw, 4)],
                "map_bounds": [round(v, 4) for v in (bx0, bx1, by0, by1)],
                "then": "one walk toward the goal"})
            final_turns(ctx, "drive.burst",
                        lambda px, py: math.atan2(goal_xy[1] - py, goal_xy[0] - px),
                        gate_rad=TURN_GATE_RAD,
                    why="no map data at this stance: face the goal, then one walk toward it")
            _burst(ctx, "offmap.walk", MIN_BURST_S, (WALK_SPEED, 0.0, 0.0),
                    why="no map data at this stance: one walk toward the goal, then look again")
            continue
        again = _plan_route(art, (x, y), goal_xy, radius_m, GOAL_RADIUS_M,
                            start_radius_m=GOAL_RADIUS_M)
        if again.path is not None and len(again.path) > 1:
            ctx.log("drive.route", {
                "code": "REPLANNED" + ("_TIGHT" if again.diagnostic == "TIGHT" else ""),
                "waypoints": len(again.path), "route_len_m": round(float(path_len(again.path)), 4),
                "route_world_xy": [[float(a), float(b)] for a, b in again.path],
                "from": [round(x, 4), round(y, 4)]})
            return again
        if not checker.is_safe(x, y):
            escape = _checked_escape(checker, x, y, yaw)
            if escape is None:
                escape = (BLIND_ESCAPE_M, math.pi)  # boxed in: one short reverse
                ctx.log("drive.no_checked_escape",
                        {"pose": [round(x, 4), round(y, 4), round(yaw, 4)],
                         "then": f"a blind {BLIND_ESCAPE_M} m reverse"})
            dist, angle = escape
            ctx.log("drive.escape", {
                "distance_m": round(dist, 4), "body_angle_rad": round(angle, 4),
                **held_move(ctx, "drive.burst", dist * math.cos(angle), 0.0,
                            why="this stance is inside the map's obstacles: escape along the "
                                "checked ray, then replan")})
            continue
        no_route += 1
        if no_route == 1 or no_route % 10 == 0:
            ctx.log("drive.no_route_yet", {"passes": no_route, "code": again.code,
                                           "reason": again.reason,
                                           "pose": [round(x, 4), round(y, 4)]})
        if no_route * POSE_UNSAFE_RETRY_S >= NO_ROUTE_PATIENCE_S:
            # no route for long enough: reverse and plan from elsewhere
            ctx.log("drive.escape", {"distance_m": BLIND_ESCAPE_M, "body_angle_rad": math.pi,
                                     "why": "no route from this stance",
                                     **held_move(ctx, "drive.burst", -BLIND_ESCAPE_M, 0.0,
                                                 why="no route from this stance for too long: "
                                                     "a blind reverse, then plan from elsewhere")})
            no_route = 0
            continue
        hold(ctx, POSE_UNSAFE_RETRY_S, 0.0, 0.0, 0.0)


class _Localized:
    """The localizer pose the drive steers by; under the quality floor the base stands and
    waits, then sets `resumed` so the caller re-picks its waypoint."""

    _owns_pose_consistency = True  # this wrapper owns the pose consistency check

    def __init__(self, ctx):
        self.ctx = ctx
        self.raw_observe = ctx.observe   # the world's own reads; `observe` below is the gated one
        self.resumed = False   # True after a wait: the caller re-picks its waypoint
        self.recovery_generation = 0
        self.route_revalidation_needed = False

    def observe(self, term: str):
        """`ctx.observe` with base_pose served by the gated `read`."""
        if term != "base_pose":
            return self.raw_observe(term)
        return np.asarray(self.read("observe"), dtype=float)

    def __call__(self, term: str):
        return self.observe(term)

    def serves(self, term: str) -> bool:
        """Keep sensor selection available after wrapping the pose read."""
        return hasattr(self.raw_observe, "serves") and self.raw_observe.serves(term)

    @property
    def execution_geometry_guarded(self) -> bool:
        return bool(getattr(self.raw_observe, "execution_geometry_guarded", False))

    def _raw_pose(self, *, accepted_only=False) -> tuple[float, float, float]:
        """The localizer's pose as the world reports it, a frozen or stale read waited out."""
        p = localizer.read(self.ctx, "base_pose", lambda: self.raw_observe("base_pose"),
                           functools.partial(hold, self.ctx, localizer.RETRY_S, 0.0, 0.0, 0.0),
                           accepted_only=accepted_only)
        return float(p[0]), float(p[1]), float(p[2])

    def read(self, where: str, *, accepted_only=False) -> tuple[float, float, float]:
        ctx = self.ctx
        self.resumed = False
        quality = float(self.raw_observe("loc_quality"))
        if quality >= LOC_QUALITY_FLOOR:
            waits = ctx.gate.waits
            live = self._raw_pose(accepted_only=accepted_only)
            if ctx.gate.waits != waits:     # the gate waited out a frozen or jumped read
                self.resumed = True
                self.recovery_generation += 1
                self.route_revalidation_needed = True
            return live
        before = self._raw_pose()
        stage(ctx, "waiting for the localizer",
              f"quality {quality:.2f}, needs {LOC_QUALITY_FLOOR}")
        ctx.log("drive.localization_degraded", {
            "where": where, "quality": round(quality, 3), "floor": LOC_QUALITY_FLOOR,
            "then": "standing until it returns"})
        waited = 0
        while quality < LOC_QUALITY_FLOOR:
            if ctx.deadline.expired():
                raise DriveFailure(
                    "TIMEOUT",
                    f"localizer quality {quality:.3f} under the floor for "
                    f"{waited * POSE_UNSAFE_RETRY_S:.1f} s at {where}")
            hold(ctx, POSE_UNSAFE_RETRY_S, 0.0, 0.0, 0.0)
            waited += 1
            quality = float(self.raw_observe("loc_quality"))
        live = self._raw_pose(accepted_only=accepted_only)
        ctx.log("drive.localization_recovered", {
            "where": where, "quality": round(quality, 3),
            "waited_s": round(waited * POSE_UNSAFE_RETRY_S, 2),
            "pose": [round(v, 4) for v in live],
            "jump_m": round(math.hypot(live[0] - before[0], live[1] - before[1]), 4)})
        self.resumed = True
        self.recovery_generation += 1
        self.route_revalidation_needed = True
        return live


ORIENT_LOOKAHEAD_M = 0.50  # a leg aims at the route this far ahead of the base


NEAR_AHEAD_M = (0.10, 0.55)  # base-frame forward box; an obstacle point in it stops the walk
NEAR_HALF_WIDTH_M = 0.30
NEAR_HEIGHT_M = (0.10, 1.60)     # map-frame z: above the floor, under anything the body clears
NEAR_STOP_M = 0.22               # anything this close ahead, mapped or not, stops the walk


def _near_field(ctx) -> dict | None:
    """Obstacle points inside the forward box, from `navigation_obstacles` when served, else
    the registered LiDAR obstacles."""
    if ctx.observe.serves("navigation_obstacles"):
        sample = ctx.observe("navigation_obstacles")
        return _obstacle_near_field(ctx, sample.points_xyz, sample.source,
                                    sample.height_above_ground_m)
    return _obstacle_near_field(ctx, ctx.observe("lidar_obstacles"), "registered_lidar")


CONTINUOUS_POLL_S = 0.1   # a continuous move re-reads the pose this often
CONTINUOUS_MAX_S = 20.0   # the longest one continuous move may last
LEG_ALIGN_RAD = math.radians(7.0)  # a turn ends inside this, well inside drive.leave_deg


def _bearing_to(ctx, target) -> tuple[float, float]:
    """(bearing to `target` from the base's nose, distance to it) — both from one pose read."""
    x, y, yaw = pose(ctx)
    return (wrap_angle(math.atan2(target[1] - y, target[0] - x) - yaw),
            math.hypot(target[0] - x, target[1] - y))


def _hold_until(ctx, vel, done, tag: str, why: str, limit_s: float,
                expect_s: float | None = None, check=None) -> int:
    """Hold one twist (or a callable re-read every poll), reading the pose every
    CONTINUOUS_POLL_S, until `done()` or a `check` returns False."""
    said = float(limit_s if expect_s is None else min(expect_s, limit_s))  # expected duration
    live = vel if callable(vel) else (lambda v=tuple(vel): v)
    if check is not None and check() is False:
        return 0  # hand the turn back to the caller
    first = live()
    announce(ctx, motion_words(first, said), why, tag, said, first)
    x0, y0, a0 = pose(ctx)
    wall_start = float(ctx.now())
    ticks, waited = 0, 0.0
    while waited < limit_s and not ctx.deadline.expired():
        if check is not None and check() is False:
            break
        if check is None:
            velocity = live()
        else:
            generation = (check.generation, getattr(check, "revision", 0))
            velocity = live()
            if check() is False:  # computing the twist may wait for a corrected pose
                break
            if (check.generation, getattr(check, "revision", 0)) != generation:
                continue  # discard a twist computed across a recovery or route rejoin
        ticks += hold(ctx, CONTINUOUS_POLL_S, *velocity)
        waited += CONTINUOUS_POLL_S
        if check is not None and check() is False:
            break
        if done():
            break
    x1, y1, a1 = pose(ctx)
    # burst telemetry for the continuous move
    ctx.log("drive.burst", {
        "tag": tag, "kind": "TURN" if math.hypot(first[0], first[1]) == 0.0 else "WALK",
        "commanded": round((abs(first[2]) if math.hypot(first[0], first[1]) == 0.0
                            else math.hypot(first[0], first[1])) * waited, 4),
        "seconds": round(waited, 3), "settle_s": 0.0, "ticks": ticks,
        "wall_elapsed_s": round(float(ctx.now()) - wall_start, 3),
        "moved_m": round(math.hypot(x1 - x0, y1 - y0), 4),
        "turned_rad": round(wrap_angle(a1 - a0), 4),
        "pose": [round(x1, 4), round(y1, 4), round(a1, 4)],
        "motion": motion_words(first, said), "why": why})
    return ticks


CONTINUOUS_LOOK_S = 0.3   # how often a driving leg looks for obstacles ahead
ARRIVAL_SETTLE_S = 3.0  # empty-handed stand at the end of a drive


def _settle_after_drive(ctx, held: dict) -> int:
    """Stand still at the end of the drive, only with both hands verifiably empty."""
    if ARRIVAL_SETTLE_S <= 0.0 or held.get("L") is not None or held.get("R") is not None:
        return 0
    t0, sent = float(ctx.now()), 0      # clocked: the loop is not 250 Hz
    while sent < round(ARRIVAL_SETTLE_S * TICK_HZ) and \
            (sent == 0 or float(ctx.now()) - t0 < ARRIVAL_SETTLE_S - 1e-9):
        step_base(ctx, LowerBody(vx=0.0, vy=0.0, wz=0.0, state="stand"))
        sent += 1
    ctx.log("drive.settled", {"seconds": ARRIVAL_SETTLE_S, "ticks": sent,
                              "why": "the base comes to rest before the next skill sights"})
    return sent


# close-in heading correction
DRIVE_YAW_MAX_RAD_S = 0.25
DRIVE_YAW_GAIN_HZ = 0.8
DRIVE_YAW_DEADBAND_RAD = math.radians(5.0)
LEG_LINEAR_CAP_MPS = 0.60  # cap on the linear command's magnitude


def _capped(vx: float, vy: float) -> tuple[float, float]:
    """(vx, vy) scaled together under LEG_LINEAR_CAP_MPS and LATERAL_SPEED, keeping direction."""
    mag = math.hypot(vx, vy)
    if mag <= 1e-9:
        return (0.0, 0.0)
    k = min(1.0, LEG_LINEAR_CAP_MPS / mag)
    if abs(vy) * k > LATERAL_SPEED:
        k = LATERAL_SPEED / abs(vy)
    return (vx * k, vy * k)


def _leg_twist(ctx, target, line0, on, tolerance_m: float, speed_target=None, *, cap_mps=None):
    """Track the route line, with task-supplied heading feedback when required."""
    bearing, gap = _bearing_to(ctx, target)
    if speed_target is not None:
        gap = _bearing_to(ctx, speed_target)[1]
    speed = _drive_speed(gap, tolerance_m)
    vx, vy = _capped(speed * math.cos(bearing),
                     speed * math.sin(bearing) + _cross_track_vy(ctx, line0, target, on))
    if cap_mps is not None:
        scale = min(1., cap_mps / max(math.hypot(vx, vy), 1e-9))
        vx, vy = vx * scale, vy * scale
    gain = getattr(ctx, "route_heading_gain_hz", 0.)
    return (vx, vy, _drive_yaw(bearing, gain_hz=gain) if gain else 0.)


def _drive_yaw(bearing: float, *, gain_hz: float = DRIVE_YAW_GAIN_HZ) -> float:
    """The close-in heading correction, zero inside DRIVE_YAW_DEADBAND_RAD."""
    if abs(bearing) < DRIVE_YAW_DEADBAND_RAD:
        return 0.0
    want = min(max(gain_hz * abs(bearing), WZ_MIN_ACCEPT), DRIVE_YAW_MAX_RAD_S)
    return math.copysign(want, bearing)


LEG_CROSS_TRACK_M = 0.20  # cross-track error at which a leg stops, turns and drives on


def _off_line(ctx, start, target) -> float:
    """Perpendicular distance from the straight line the leg set out along, metres."""
    x, y, _ = pose(ctx)
    dx, dy = target[0] - start[0], target[1] - start[1]
    span = math.hypot(dx, dy)
    if span < 1e-6:
        return 0.0
    return abs((x - start[0]) * dy - (y - start[1]) * dx) / span


# cross-track strafe back onto the line
STRAFE_DEADBAND_M = 0.06
STRAFE_GAIN_HZ = 1.0
STRAFE_MIN_MPS = 0.12
STRAFE_MAX_MPS = 0.20   # well under the cruise speed


def _cross_track_error(ctx, start, target) -> float:
    """Signed offset from the line `start`->`target` on the base's own lateral axis, metres,
    positive when it must move left to rejoin (vy's sign, gait.motion_words)."""
    x, y, yaw = pose(ctx)
    dx, dy = target[0] - start[0], target[1] - start[1]
    span = math.hypot(dx, dy)
    if span < 1e-6:
        return 0.0
    off = ((x - start[0]) * dy - (y - start[1]) * dx) / span   # + when right of the line
    return off * (dx * math.cos(yaw) + dy * math.sin(yaw)) / span  # onto the body's left


def _cross_track_vy(ctx, start, target, log: dict | None = None) -> float:
    """The lateral a driving slice carries to rejoin its line; zero inside the deadband."""
    error = _cross_track_error(ctx, start, target)
    if abs(error) < STRAFE_DEADBAND_M:
        if log is not None and log.pop("on", None):
            ctx.log("drive.strafe", {"state": "off", "error_m": round(error, 4)})
        return 0.0
    want = min(max(STRAFE_GAIN_HZ * abs(error), STRAFE_MIN_MPS), STRAFE_MAX_MPS)
    if log is not None and not log.get("on"):
        log["on"] = True
        ctx.log("drive.strafe", {"state": "on", "error_m": round(error, 4),
                                 "vy": round(math.copysign(want, error), 4)})
    return math.copysign(want, error)


TURN_BUDGET_RAD = math.radians(180.0)  # rotation after which a turn re-reads its bearing


def _stand(ctx) -> int:
    """Stand the base still for `TURN_WALK_GAP_S`; every turn is bracketed by one."""
    return hold(ctx, TURN_WALK_GAP_S, 0.0, 0.0, 0.0)


def _turn_budget(ctx, ceiling_rad: float = TURN_BUDGET_RAD):
    """A predicate that turns True once the base has rotated more than `ceiling_rad` in total,
    accumulated from the pose."""
    state = [pose(ctx)[2], 0.0]

    def spent() -> bool:
        now = pose(ctx)[2]
        state[1] += abs(wrap_angle(now - state[0]))
        state[0] = now
        return state[1] > float(ceiling_rad)

    return spent


def _turn_rate(bearing: float) -> float:
    """A turn-in-place rate proportional to the bearing, floored, and capped at the gait's rate."""
    drive = current().drive
    return math.copysign(min(max(drive.turn_gain_hz * abs(bearing), drive.turn_floor_rad_s),
                             turn_speed()), bearing)


def _drive_speed(gap: float, tolerance_m: float) -> float:
    """The gap past the tolerance times the gain, floored and capped at the cruise speed."""
    drive = current().drive
    return min(max(drive.gain_hz * max(gap - tolerance_m, 0.0), drive.floor_mps),
               drive.cruise_mps)


CLOSE_STEP_MAX = 3  # close-in step rounds inside drive.close_step_m
AIM_IN_M = 1.10   # the arrival heading is taken this far out, while the named point is ahead


def _close_twist(bearing: float, gap: float, tolerance_m: float,
                 off: float = 0.0, *, cap_mps=None) -> tuple[float, float, float]:
    """The body-frame twist that closes the last of the gap by translation, proportional to it,
    floored and capped, with `_drive_yaw(off)` as the only yaw."""
    want = max(gap - tolerance_m, 0.0)
    if want <= 0.0:
        return 0.0, 0.0, _drive_yaw(off)
    drive = current().drive
    speed = min(max(drive.close_gain_hz * want, drive.floor_mps), drive.close_mps)
    if cap_mps is not None:
        speed = min(speed, cap_mps)
    return speed * math.cos(bearing), speed * math.sin(bearing), _drive_yaw(off)


CLOSE_AWAY_CONFIRM_S = 0.5  # sustained moving away that refuses the approach


class _CloseMovingAway(DriveFailure):
    """The close-in slice stopped; the drive may spend another bounded attempt."""

    def __init__(self, detail: str, ticks: int):
        super().__init__("ARRIVAL_NOT_VERIFIED", detail)
        self.ticks = ticks


def _close_goal(ctx, goal_xy, tolerance_m: float, off, *, check=None, cap_mps=None) -> int:
    """Close by translation; stop on stagnation or moving away instead of chasing the goal."""
    best = _bearing_to(ctx, goal_xy)[1]
    progress_at = float(ctx.now())
    away_at = None
    moving_away = False
    reason = "close-in time limit"

    def done():
        nonlocal best, progress_at, away_at, reason, moving_away
        gap = _bearing_to(ctx, goal_xy)[1]
        now = float(ctx.now())
        if gap <= tolerance_m:
            reason = "settled outside arrival tolerance"
            return True
        if gap < best - STALL_PROGRESS_M:
            best, progress_at = gap, now
        if gap > best + STALL_PROGRESS_M:
            if away_at is None:
                away_at = now
            elif now - away_at >= CLOSE_AWAY_CONFIRM_S:
                moving_away = True
                reason = "moving away from the closest approach"
                return True
        else:
            away_at = None
        if now - progress_at >= STALL_WINDOW_S:
            reason = "no measurable progress toward the stance"
            return True
        return False

    ticks = _hold_until(
        ctx, lambda: _close_twist(*_bearing_to(ctx, goal_xy), tolerance_m, off(),
                                  **({"cap_mps": cap_mps} if cap_mps is not None else {})),
        done, "drive.close", "translate onto the stance with bounded progress",
        CONTINUOUS_MAX_S,
        expect_s=max(best - tolerance_m, 0.0) / max(current().drive.close_mps, 1e-6),
        **({"check": check} if check is not None else {}))
    ticks += _stand(ctx)
    gap = _bearing_to(ctx, goal_xy)[1]
    if gap > tolerance_m:
        ctx.log("drive.close_refused", {"gap_m": gap, "reason": reason,
                                       "best_gap_m": best})
        detail = f"{reason}; stopped {gap:.3f} m from stance"
        if moving_away:
            raise _CloseMovingAway(detail, ticks)
        raise DriveFailure("ARRIVAL_NOT_VERIFIED", detail)
    return ticks


def _obstacle_near_field(ctx, points, source, heights=None):
    """Count, imminence, centroid and nearest distance of obstacle points in the forward box,
    or None."""
    points = np.asarray(points, dtype=float)
    heights = points[:, 2] if heights is None else np.asarray(heights)
    base = ctx.observe("base_pose")
    x, y, yaw = float(base[0]), float(base[1]), float(base[2])
    dx, dy = points[:, 0] - x, points[:, 1] - y
    ahead = math.cos(yaw) * dx + math.sin(yaw) * dy
    side = -math.sin(yaw) * dx + math.cos(yaw) * dy
    keep = ((ahead > NEAR_AHEAD_M[0]) & (ahead < NEAR_AHEAD_M[1])
            & (np.abs(side) < NEAR_HALF_WIDTH_M)
            & (heights > NEAR_HEIGHT_M[0]) & (heights < NEAR_HEIGHT_M[1]))
    points, ahead = points[keep], ahead[keep]
    if not len(points):
        return None
    return {"count": len(points), "imminent": bool((ahead < NEAR_STOP_M).any()),
            "centroid_map": points.mean(axis=0).tolist(), "ahead_m": float(ahead.min()),
            "source": source}


def _eyes_ahead(ctx):
    """An obstacle sighted inside NEAR_STOP_M ahead while driving, else None."""
    if getattr(ctx.observe, "execution_geometry_guarded", False):
        return None  # the backend checks fresh geometry against the commanded footprint
    serves = getattr(ctx.observe, "serves", None)
    if serves is None or not (serves("lidar_obstacles") or serves("navigation_obstacles")):
        return None  # no obstacle observation: the route checks alone guard the walk
    seen = _near_field(ctx)
    return seen if seen is not None and seen["imminent"] else None


FACE_POINT_MIN_M = 0.15   # a facing point nearer the goal than this gives no yaw


def _resample(path, spacing: float) -> tuple:
    """The route's interior as points at most `spacing` apart, WAYPOINT_FINAL_SPACING_M over the
    last WAYPOINT_FINAL_M; start and goal dropped."""
    pts = [tuple(float(v) for v in q[:2]) for q in (path or ())]
    if len(pts) < 2:
        return ()
    segs = [(a, b, math.dist(a, b)) for a, b in itertools.pairwise(pts)]
    total = sum(d for _a, _b, d in segs)
    out: list[tuple[float, float]] = []
    done = 0.0
    for a, b, d in segs:
        if d <= 1e-9:
            continue
        t = 0.0
        while True:
            step = (WAYPOINT_FINAL_SPACING_M
                    if total - (done + t) <= WAYPOINT_FINAL_M else spacing)
            if t + step >= d - 1e-9:
                break
            t += step
            out.append((a[0] + t / d * (b[0] - a[0]), a[1] + t / d * (b[1] - a[1])))
        done += d
        out.append(b)
    return tuple(out[:-1])          # the goal is not a waypoint


def _run_in(waypoints, goal_xy, faced_xy, checker):
    """`waypoints` with a point `drive.run_in_m` out from the goal, away from what is faced, so
    the last leg arrives square; nearer waypoints are dropped."""
    run_in_m = current().drive.run_in_m
    if faced_xy is None or goal_xy is None or checker is None or run_in_m <= 0.0:
        return waypoints
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    dx, dy = gx - float(faced_xy[0]), gy - float(faced_xy[1])
    span = math.hypot(dx, dy)
    if span < FACE_POINT_MIN_M:
        return waypoints
    run = (gx + run_in_m * dx / span, gy + run_in_m * dy / span)
    if not checker.is_safe(*run):
        return waypoints
    kept = tuple(w for w in waypoints if math.dist(w, (gx, gy)) > run_in_m)
    return kept + (run,)


def _passed_waypoint(xy, start, target) -> bool:
    """Whether `xy` is past the target along its incoming segment and near it."""
    drive = current().drive
    dx, dy = target[0] - start[0], target[1] - start[1]
    span = math.hypot(dx, dy)
    passed_m = max(2.0 * drive.waypoint_tol_m, drive.waypoint_spacing_m)
    if span < 1e-6 or math.dist(xy, target) > passed_m:
        return False
    along = (xy[0] - target[0]) * dx + (xy[1] - target[1]) * dy
    across = abs((xy[0] - start[0]) * dy - (xy[1] - start[1]) * dx) / span
    return along >= 0 and across <= min(drive.waypoint_tol_m, LEG_CROSS_TRACK_M)


def _straight_waypoint(start, target, following) -> bool:
    """Resampled points on a straight segment do not define a corner to retain."""
    dx, dy = target[0] - start[0], target[1] - start[1]
    ex, ey = following[0] - target[0], following[1] - target[1]
    return (dx * ex + dy * ey > 0.
            and abs(dx * ey - dy * ex) <= 1e-6 * math.hypot(dx, dy) * math.hypot(ex, ey))


def _collision_failure(detail, checker, check, points, **info):
    """A NO_COLLISION_FREE_PATH failure that keeps the rejecting grid by reference."""
    failure = DriveFailure("NO_COLLISION_FREE_PATH", detail)
    failure.collision_witness = {"_checker": checker, "check": check,
                                 "points": points, **info}
    return failure


def _validate_preserved_route(checker, path, start, goal):
    """Validate the exact accepted endpoints and every segment, including resampling."""
    if not path or math.dist(path[0], start) > 1e-6 or math.dist(path[-1], goal) > 1e-6:
        raise DriveFailure("NO_COLLISION_FREE_PATH", "accepted route or exact endpoint is blocked")
    for name, point in (("accepted_start", start), ("accepted_goal", goal)):
        if not checker.is_safe(*point):
            raise _collision_failure("accepted route or exact endpoint is blocked",
                                     checker, name, (point, point))
    for a, b in itertools.pairwise(path):
        if not _line_free(checker, a, b):
            raise _collision_failure("accepted route or exact endpoint is blocked",
                                     checker, "accepted_segment", (a, b))


def _route_lookahead(xy, start, target):
    """Aim forward on the same segment; a new poll must not redefine the route's line."""
    dx, dy = target[0] - start[0], target[1] - start[1]
    span = math.hypot(dx, dy)
    if span < 1e-6:
        return target
    along = ((xy[0] - start[0]) * dx + (xy[1] - start[1]) * dy) / span
    ahead = min(span, max(0., along) + ORIENT_LOOKAHEAD_M)
    return (start[0] + ahead * dx / span, start[1] + ahead * dy / span)


def _local_route_splice(checker, xy, start, target, *, corner=False, cancelled, yaw=None,
                        clearance_margin_m=0., min_distance_m=0.):
    """One bounded exact-start connection to a forward point of the retained segment."""
    if cancelled() or not checker.is_safe(*xy):
        return None
    dx, dy = target[0] - start[0], target[1] - start[1]
    span = math.hypot(dx, dy)
    if span < 1e-6:
        return None
    along = ((xy[0] - start[0]) * dx + (xy[1] - start[1]) * dy) / span
    extent = ORIENT_LOOKAHEAD_M + LEG_CROSS_TRACK_M
    candidates = [target] if corner else [
        (start[0] + min(span, max(0., along + look)) * dx / span,
         start[1] + min(span, max(0., along + look)) * dy / span)
        for look in (ORIENT_LOOKAHEAD_M, .25, .10, 0.)]
    local = None
    if isinstance(checker, GridCollisionChecker):
        lo = [max(0, math.floor((xy[k] - extent - origin) / checker.cell))
              for k, origin in enumerate((checker.xmin, checker.ymin))]
        hi = [min(n, math.ceil((xy[k] + extent - origin) / checker.cell) + 1)
              for k, (origin, n) in enumerate(((checker.xmin, checker.nx),
                                                (checker.ymin, checker.ny)))]
        free = checker.grid[lo[0]:hi[0], lo[1]:hi[1]]
        local = GridCollisionChecker(checker.xmin + lo[0] * checker.cell,
                                     checker.ymin + lo[1] * checker.cell, checker.cell,
                                     *free.shape, free, checker.radius_baked)
    if local is None:
        clearance_margin_m = 0.
    # rank a few joins, direct connections before grid detours
    ranked, have_direct = [], False
    choices = ([(False, p) for p in candidates] if clearance_margin_m
               else [(True, p) for p in candidates] + [(False, p) for p in candidates])
    for direct, join in choices:
        if cancelled():
            break
        if not direct and have_direct and not clearance_margin_m:
            break
        if not (max(1e-6, min_distance_m) < math.dist(xy, join) <= extent
                and _line_free(checker, join, target)):
            continue
        if direct:
            path = [xy, join] if _line_free(checker, xy, join) else None
        else:
            path = (plan_grid_astar(xy, join, local, cancelled=cancelled,
                                   clearance_margin_m=clearance_margin_m) if local else None)
        if (path and tuple(path[0]) == tuple(xy) and tuple(path[-1]) == tuple(join)
                and path_len(path) <= 2 * extent
                and all(max(abs(p[k] - xy[k]) for k in (0, 1)) <= extent for p in path)
                and all(_line_free(checker, a, b) for a, b in itertools.pairwise(path))):
            if yaw is None:
                return path
            key = tuple(tuple(p) for p in path)
            if any(key == entry[1] for entry in ranked):
                continue
            have_direct = have_direct or direct
            headings = [yaw] + [math.atan2(b[1] - a[1], b[0] - a[0])
                                for a, b in zip(path, path[1:] + [target])
                                if math.dist(a, b) > 1e-6]
            turns = sum(abs(wrap_angle(b - a)) for a, b in itertools.pairwise(headings))
            dx, dy = path[1][0] - xy[0], path[1][1] - xy[1]
            lateral = abs(-math.sin(yaw) * dx + math.cos(yaw) * dy)
            score = turns + .25 * lateral + .5 * (path_len(path) + math.dist(path[-1], target))
            ranked.append((score, key, path))
            ranked.sort(key=lambda row: row[0])
            del ranked[3:]
    return ranked[0][2] if ranked else None  # a checked candidate survives search expiry


class _RouteMotionCheck:
    """One segment's bounded progress and geometry checks, shared across bursts."""

    def __init__(self, ctx, checker, target, remaining=(), radius_m=None,
                 checker_ref=None, route_id=None, route_start=None, clearance_margin_m=0.,
                 turn_translation_limit_m=TURN_DRIFT_CAP_M):
        self.ctx, self.checker, self.target = ctx, checker, target
        self.checker_ref = checker_ref if checker_ref is not None else [checker]
        self.route_id = route_id
        self.route_start = route_start
        self.clearance_margin_m = clearance_margin_m
        self.turn_translation_limit_m = turn_translation_limit_m
        self.rejoin_used, self.revision = False, 0
        self.remaining, self.radius_m = tuple(remaining), radius_m
        self.generation = getattr(ctx.observe, "recovery_generation", 0)
        x, y, yaw = pose(ctx)
        self.last_xy = (x, y)
        self.best_gap = math.dist(self.last_xy, target)
        self.best_bearing = abs(wrap_angle(math.atan2(target[1] - y, target[0] - x) - yaw))
        self.progress_at = float(ctx.now())
        self.turn_origin = None
        self.turn_episode_origin = None

    def turn(self, active):
        if active:
            if self.turn_episode_origin is None:
                self.turn_episode_origin = tuple(pose(self.ctx)[:2])
                self.ctx.log("drive.turn_translation", {
                    "phase": "start", "route_id": self.route_id,
                    "origin_xy": list(self.turn_episode_origin),
                    "limit_m": self.turn_translation_limit_m})
            self.turn_origin = self.turn_episode_origin
        else:
            self.turn_origin = self.turn_episode_origin = None

    def refuse(self, code, why, *, check=None, points=()):
        _stand(self.ctx)
        if check is not None:
            raise _collision_failure(why, self.checker, check, points, route_id=self.route_id)
        raise DriveFailure(code, why)

    def rejoin(self, *, corner=False):
        """Replace a failed child connector iteratively, without resetting the action budget."""
        if self.route_start is None or self.rejoin_used or self.turn_origin is not None:
            return False
        attempts = 3 if self.clearance_margin_m else 1
        for attempt in range(attempts):
            before = tuple(pose(self.ctx)[:2])
            try:
                return self._rejoin_once(corner=corner, min_distance_m=.25 if attempt else 0.)
            except DriveFailure as exc:
                witness = getattr(exc, "collision_witness", {})
                current = tuple(pose(self.ctx)[:2])
                if (attempt == attempts - 1 or witness.get("check") != "pose_to_target"
                        or math.dist(before, current) < .05
                        or self.ctx.deadline.expired()
                        or getattr(self.ctx.observe, "recovery_generation", 0) != self.generation
                        or getattr(self.ctx.observe, "route_revalidation_needed", False)):
                    raise
                # refresh the grid while stationary
                checker = occupancy_grid(self.ctx.observe("map"), self.radius_m,
                    **({"exact_footprint": True}
                       if getattr(self.checker, "exact_footprint", False) else {}))[0]
                if not checker.is_safe(*current):
                    raise
                self.checker = self.checker_ref[0] = checker
                self.last_xy = current
                self.rejoin_used = False
                self.ctx.log("drive.route_rejoin", {"phase": "retry", "attempt": attempt + 1,
                    "route_id": self.route_id, "pose_xy": list(current),
                    "reason": "child connector blocked; fresh connection to retained route"})
        return False

    def _rejoin_once(self, *, corner=False, min_distance_m=0.):
        """One checked local splice back to this segment, on the original deadline."""
        if self.route_start is None or self.rejoin_used or self.turn_origin is not None:
            return False
        ctx = self.ctx
        before = tuple(pose(ctx)[:2])
        _stand(ctx)
        xy = tuple(pose(ctx)[:2])
        retained = (self.route_start, self.target) + self.remaining
        if (getattr(ctx.observe, "recovery_generation", 0) != self.generation
                or getattr(ctx.observe, "route_revalidation_needed", False)
                or not _line_free(self.checker, before, xy)
                or not all(_line_free(self.checker, a, b)
                           for a, b in itertools.pairwise(retained))):
            return False
        self.rejoin_used = True
        search_end = time.monotonic() + CONTINUOUS_POLL_S
        path = _local_route_splice(
            self.checker, xy, self.route_start, self.target, corner=corner,
            cancelled=lambda: ctx.deadline.expired() or time.monotonic() >= search_end,
            yaw=pose(ctx)[2], clearance_margin_m=self.clearance_margin_m,
            min_distance_m=min_distance_m)
        if ctx.deadline.expired():
            self.refuse("TIMEOUT", "drive deadline expired during route splice search")
        if (getattr(ctx.observe, "recovery_generation", 0) != self.generation
                or getattr(ctx.observe, "route_revalidation_needed", False)):
            return False
        if path is None:
            return False
        self.revision += 1
        details = {"route_id": self.route_id, "radius_m": self.radius_m,
                   "preferred_clearance_margin_m": self.clearance_margin_m,
                   "route_world_xy": [list(p) for p in (*path, self.target, *self.remaining)],
                   "reason": "blocked corner shortcut" if corner else "blocked far connector"}
        ctx.log("drive.route_rejoin", {"phase": "start", **details})
        stage(ctx, "repairing route", "checked translation back to the retained route")
        onward = self.remaining[0] if corner and self.remaining else self.target
        try:
            self.turn(False)
            _walk_waypoints(ctx, path[1:], _held_snapshot(ctx), 0,
                            checker=self.checker, preserve_goal=True, goal_xy=onward,
                            radius_m=self.radius_m, checker_ref=self.checker_ref,
                            route_id=self.route_id, route_start=xy, allow_rejoin=False,
                            waypoint_tolerance_m=MIN_WALK_GAP_M,
                            clearance_margin_m=self.clearance_margin_m,
                            turn_translation_limit_m=self.turn_translation_limit_m)
            local = _RouteMotionCheck(ctx, self.checker_ref[0], self.target, self.remaining,
                                      self.radius_m, self.checker_ref, self.route_id,
                                      turn_translation_limit_m=self.turn_translation_limit_m)
            _stand(ctx)
            local()
            xy = tuple(pose(ctx)[:2])
            suffix = (xy, self.target) + self.remaining
            onward = self.remaining[0] if corner and self.remaining else self.target
            if (_off_line(ctx, self.route_start, self.target) > LEG_CROSS_TRACK_M
                    or not _line_free(local.checker, xy, onward)
                    or not all(_line_free(local.checker, a, b)
                               for a, b in itertools.pairwise(suffix))):
                local.refuse("NO_COLLISION_FREE_PATH", "rejoin did not restore the retained route")
        except Exception as exc:
            ctx.log("drive.route_rejoin", {"phase": "failed", "why": str(exc), **details})
            raise
        self.checker = local.checker
        self.generation = local.generation
        self.last_xy = xy  # the nested checker validated this translation's swept segments
        ctx.log("drive.route_rejoin", {**details, "phase": "complete", "pose_xy": list(xy),
                                      "route_world_xy": [list(p) for p in suffix]})
        return True

    def repair_reference(self, xy):
        """Replace the tracking line after standing still and validating its connector."""
        ctx = self.ctx
        _stand(ctx)
        settled = tuple(pose(ctx)[:2])
        if ctx.deadline.expired():
            self.refuse("TIMEOUT", "drive deadline expired during connector repair")
        if (getattr(ctx.observe, "recovery_generation", 0) != self.generation
                or getattr(ctx.observe, "route_revalidation_needed", False)):
            self.refuse("NO_PROGRESS",
                        "localization changed during connector repair; fresh route required")
        path = (settled, self.target) + self.remaining
        for name, a, b in [("swept_segment", xy, settled),
                           *(("connector_segment", a, b) for a, b in itertools.pairwise(path))]:
            if not _line_free(self.checker, a, b):
                self.refuse("NO_COLLISION_FREE_PATH", "checked tracking connector became blocked",
                            check=name, points=(a, b))
        previous = self.route_start
        self.route_start = self.last_xy = settled
        self.revision += 1
        ctx.log("drive.route_revalidated", {
            "reason": "tracking reference repaired", "route_id": self.route_id,
            "radius_m": self.radius_m, "previous_reference_xy": list(previous),
            "remaining_world_xy": [list(p) for p in path]})

    def __call__(self, *, turn_complete=False):
        ctx = self.ctx
        x, y, yaw = pose(ctx)
        if ctx.deadline.expired():
            self.refuse("TIMEOUT", "drive deadline expired before the next motion command")
        xy = (x, y)
        generation = getattr(ctx.observe, "recovery_generation", 0)
        if (generation != self.generation
                or getattr(ctx.observe, "route_revalidation_needed", False)):
            _stand(ctx)  # re-registered: stop the current twist
            x, y, yaw = pose(ctx)
            xy = (x, y)
            self.generation = getattr(ctx.observe, "recovery_generation", 0)
            if ctx.deadline.expired():
                self.refuse("TIMEOUT", "drive deadline expired during localization recovery")
            art = ctx.observe("map")
            if self.radius_m is not None:
                self.checker = occupancy_grid(art, self.radius_m,
                    **({"exact_footprint": True}
                       if getattr(self.checker, "exact_footprint", False) else {}))[0]
                self.checker_ref[0] = self.checker
            path = (xy, self.target) + self.remaining
            for a, b in itertools.pairwise(path):
                if not _line_free(self.checker, a, b):
                    self.refuse("NO_COLLISION_FREE_PATH",
                                "route invalid after localization recovery",
                                check="relocalized_segment", points=(a, b))
            self.last_xy = xy  # a corrected coordinate is not a measured swept segment
            if hasattr(ctx.observe, "route_revalidation_needed"):
                ctx.observe.route_revalidation_needed = False
            ctx.log("drive.route_revalidated", {
                "reason": "localization recovered", "route_id": self.route_id,
                "radius_m": self.radius_m, "map_generation": getattr(art, "scan_hash", None),
                "remaining_world_xy": [list(p) for p in path]})
            self.progress_at = float(ctx.now())
        if self.turn_origin is not None:
            translated = math.dist(xy, self.turn_origin)
            over = translated > self.turn_translation_limit_m
            if over or turn_complete:
                ctx.log("drive.turn_translation", {
                    "phase": "refused" if over else "settled", "route_id": self.route_id,
                    "origin_xy": list(self.turn_origin), "pose_xy": list(xy),
                    "translation_m": translated, "limit_m": self.turn_translation_limit_m})
            if over:
                self.refuse("NO_COLLISION_FREE_PATH", "turn translated beyond the checked bound: "
                            f"{translated:.3f} m > {self.turn_translation_limit_m:.3f} m")
        if not _line_free(self.checker, self.last_xy, xy):
            self.refuse("NO_COLLISION_FREE_PATH",
                        "measured swept segment crossed blocked route geometry",
                        check="swept_segment", points=(self.last_xy, xy))
        if turn_complete:
            self.turn_origin = None  # turn drift and sweep checked above
        connector_free = _line_free(self.checker, xy, self.target)
        off_reference = (self.route_start is not None
                         and _off_line(ctx, self.route_start, self.target) > LEG_CROSS_TRACK_M)
        if not connector_free or off_reference:
            if (self.turn_origin is not None and self.route_start is not None
                    and (not self.rejoin_used or connector_free)):
                # end the yaw first; the settled check may then rejoin
                ctx.log("drive.turn_handoff", {"route_id": self.route_id,
                        "radius_m": self.radius_m, "pose_xy": list(xy),
                        "target_xy": list(self.target), "reason": "route repair required"})
                return False
            if connector_free and off_reference and self.turn_origin is None:
                self.repair_reference(xy)
            elif not self.rejoin():
                self.refuse("NO_COLLISION_FREE_PATH",
                            "current pose cannot connect to the retained route",
                            check="pose_to_target", points=(xy, self.target))
            x, y, yaw = pose(ctx)
            xy = (x, y)
        self.last_xy = xy
        gap = math.dist(xy, self.target)
        bearing = abs(wrap_angle(math.atan2(self.target[1] - y, self.target[0] - x) - yaw))
        if gap < self.best_gap - STALL_PROGRESS_M or bearing < self.best_bearing - LEG_ALIGN_RAD:
            self.best_gap = min(gap, self.best_gap)
            self.best_bearing = min(bearing, self.best_bearing)
            self.progress_at = float(ctx.now())
        if float(ctx.now()) - self.progress_at >= STALL_WINDOW_S:
            self.refuse("NO_PROGRESS", "no measurable progress on the retained route")


def _walk_waypoints(ctx, waypoints, held: dict, ticks: int, *,
                    checker=None, preserve_goal=False, goal_xy=None,
                    radius_m=None, checker_ref=None, route_id=None,
                    route_start=None, allow_rejoin=True,
                    waypoint_tolerance_m=None, cap_mps=None,
                    clearance_margin_m=0.,
                    turn_translation_limit_m=TURN_DRIFT_CAP_M) -> tuple[int, dict]:
    """Walk the route's intermediate waypoints: turn in place outside the align gate, then drive
    with the lateral line correction until each is passed."""
    drive = current().drive
    leave_rad = math.radians(drive.leave_deg)
    if waypoint_tolerance_m is None:
        waypoint_tolerance_m = drive.waypoint_tol_m
    pass_tolerance_m = waypoint_tolerance_m
    if clearance_margin_m:
        waypoint_tolerance_m = min(waypoint_tolerance_m, MIN_WALK_GAP_M)
    origin = tuple(pose(ctx)[:2])
    checker_ref = checker_ref if checker_ref is not None else [checker]
    for i, wp in enumerate(waypoints, 1):
        start = (route_start if route_start is not None else origin) if i == 1 else waypoints[i - 2]
        leg_origin = (start if preserve_goal and route_start is not None else
                      tuple(pose(ctx)[:2]) if preserve_goal else None)
        following = waypoints[i] if i < len(waypoints) else goal_xy
        straight = (preserve_goal and route_start is not None and allow_rejoin
                    and following is not None and _straight_waypoint(start, wp, following))
        check = (_RouteMotionCheck(
            ctx, checker_ref[0], wp, tuple(waypoints[i:]) + ((goal_xy,) if goal_xy else ()),
            radius_m, checker_ref=checker_ref, route_id=route_id,
            route_start=start if route_start is not None else None,
            clearance_margin_m=clearance_margin_m,
            turn_translation_limit_m=turn_translation_limit_m) if preserve_goal else None)
        if check is not None and not allow_rejoin:
            check.rejoin_used = True

        def reached(wp=wp, start=start, following=following, check=check, straight=straight):
            x, y, yaw = pose(ctx)
            xy = (x, y)
            gap = math.dist(xy, wp)
            # Straight-line samples need a checked onward connector, not an exact visit.
            # Corners and local rejoin endpoints use the tighter tracking gates.
            reach_tolerance_m = waypoint_tolerance_m
            bearing = wrap_angle(math.atan2(wp[1] - y, wp[0] - x) - yaw)
            if (straight and gap <= pass_tolerance_m
                    and abs(bearing) > _align_gate(gap, waypoint_tolerance_m)):
                reach_tolerance_m = pass_tolerance_m
            if gap > reach_tolerance_m and not _passed_waypoint(xy, start, wp):
                return False
            current_checker = check.checker if check is not None else checker
            if (not straight and preserve_goal and route_start is not None and following is not None
                    and _off_line(ctx, wp, following) > LEG_CROSS_TRACK_M):
                return False
            if (current_checker is not None and following is not None
                    and not _line_free(current_checker, xy, following)):
                if preserve_goal and (math.dist(xy, wp) > MIN_WALK_GAP_M or not allow_rejoin):
                    return False  # retain the corner; stop_leg uses this same decision
                if check is not None and check.rejoin(corner=True):
                    return _line_free(check.checker, tuple(pose(ctx)[:2]), following)
                _stand(ctx)
                raise DriveFailure("NO_COLLISION_FREE_PATH",
                                   "waypoint shortcut is blocked; a fresh route is required")
            return True

        bearing, gap = _bearing_to(ctx, wp)
        if check is not None:
            check()
        if reached():
            continue
        ctx.log("drive.waypoint", {"index": i, "of": len(waypoints), "gap_m": round(gap, 4),
                                   "bearing_deg": round(math.degrees(bearing), 1),
                                   "xy": [round(float(wp[0]), 4), round(float(wp[1]), 4)]})
        while not ctx.deadline.expired():
            if check is not None:
                check()
            bearing, gap = _bearing_to(ctx, wp)
            if reached():
                ctx.log("drive.waypoint_passed", {
                    "index": i, "of": len(waypoints), "gap_m": round(gap, 4),
                    "bearing_deg": round(math.degrees(bearing), 1),
                    "why": "reached or progressed past it along the route"})
                break
            if abs(bearing) > _align_gate(gap, waypoint_tolerance_m):
                stage(ctx, f"turning onto waypoint {i}",
                      f"{math.degrees(abs(bearing)):.0f} deg off")
                ticks += _stand(ctx)
                if check is not None:
                    check.turn(True)
                over = _turn_budget(ctx)
                ticks += _hold_until(
                    ctx, lambda wp=wp: (0.0, 0.0, _turn_rate(_bearing_to(ctx, wp)[0])),
                    lambda wp=wp, over=over: (
                        (lambda b, g:  # noqa: PLC3002 -- one bearing read
                         abs(b) <= _align_gate(g, waypoint_tolerance_m))(
                            *_bearing_to(ctx, wp)) or over()),
                    "waypoint.turn", "turn in place until the waypoint is on the nose",
                    CONTINUOUS_MAX_S, expect_s=abs(bearing) / turn_speed(),
                    **({"check": check} if check is not None else {}))
                ticks += _stand(ctx)
                if check is not None:
                    check(turn_complete=True)
                held = _carry_check(ctx, held, f"waypoint {i} turn")
                continue
            stage(ctx, f"walking to waypoint {i} of {len(waypoints)}", f"{gap * 100:.0f} cm")
            if check is not None:
                check.turn(False)
            hit, looked = {}, [float(ctx.now())]
            line0 = (check.route_start if check is not None and check.route_start is not None else
                     leg_origin if preserve_goal else pose(ctx)[:2])

            def stop_leg(wp=wp, hit=hit, looked=looked, line0=line0, check=check) -> bool:
                b, g = _bearing_to(ctx, wp)
                if (reached() if preserve_goal else g <= drive.waypoint_tol_m):
                    return True
                if abs(b) > max(leave_rad, _align_gate(g, waypoint_tolerance_m)):
                    return True
                # without a route reference, leaving the line stops the leg
                if ((check is None or check.route_start is None)
                        and _off_line(ctx, line0, wp) > LEG_CROSS_TRACK_M):
                    if check is not None:
                        check.refuse("NO_COLLISION_FREE_PATH", "left the retained route corridor")
                    hit.update(off_line_m=round(_off_line(ctx, line0, wp), 3))
                    return True          # drifted off the line: stop, turn, drive on
                now = float(ctx.now())
                if now - looked[0] < CONTINUOUS_LOOK_S:
                    return False
                looked[0] = now
                ahead = _eyes_ahead(ctx)
                if ahead is not None:
                    hit.update(ahead)
                    return True
                return False

            ticks += _hold_until(
                ctx,
                lambda wp=wp,
                line0=line0, on={}, check=check: _leg_twist(
                    ctx, _route_lookahead(pose(ctx)[:2],
                                         check.route_start if check.route_start is not None
                                         else line0, wp)
                    if preserve_goal else wp,
                    check.route_start if check is not None and check.route_start is not None
                    else line0,
                    on, 0.0 if preserve_goal else drive.waypoint_tol_m,
                    wp if preserve_goal else None,
                    **({"cap_mps": drive.cruise_mps if not allow_rejoin else cap_mps}
                       if not allow_rejoin or cap_mps is not None else {})),
                stop_leg,
                "waypoint.go", "forward with the small yaw correction, to the next waypoint",
                CONTINUOUS_MAX_S,
                expect_s=(gap - drive.waypoint_tol_m) / max(drive.cruise_mps, 1e-6),
                **({"check": check} if check is not None else {}))
            if hit:
                ctx.log("drive.too_close", {"leg": f"waypoint {i}",
                                            **{k: v for k, v in hit.items() if k != "cells"}})
                if check is not None:
                    check.refuse("NO_COLLISION_FREE_PATH", "near-field obstacle stopped the route")
            held = _carry_check(ctx, held, f"waypoint {i}")
    return ticks, held


AIM_KEEP_CLICK_RAD = math.radians(15.0)  # clicked yaw kept within this of stance -> faced


def _continuous_drive(ctx, goal_xy, facing_xy, tolerance_m: float, held: dict,
                      route_facing: float = 0.0, asked_xy=None, waypoints=(),
                      route_checker=None, arrival_heading_required=True,
                      preserve_goal=False, radius_m=None, route_id=None, route_start=None,
                      clearance_margin_m=0., turn_translation_limit_m=TURN_DRIFT_CAP_M) -> dict:
    """Walk the waypoints, then close on the goal and take the arrival heading."""
    ticks = 0
    drive = current().drive
    align_rad, leave_rad = math.radians(drive.align_deg), math.radians(drive.leave_deg)

    def _aim_heading() -> tuple[float, str]:
        # goal -> facing, fixed once so the heading does not drift as the base moves
        if facing_xy is not None and asked_xy is not None:
            dx, dy = facing_xy[0] - asked_xy[0], facing_xy[1] - asked_xy[1]
            if math.hypot(dx, dy) > FACE_POINT_MIN_M:
                clicked = math.atan2(dy, dx)
                # the clicked yaw stands while it still points at the faced point
                from_stance = math.atan2(facing_xy[1] - goal_xy[1], facing_xy[0] - goal_xy[0])
                if abs(wrap_angle(clicked - from_stance)) <= AIM_KEEP_CLICK_RAD:
                    return clicked, "the clicked yaw (goal -> facing)"
                return route_facing, "the route's facing (the stance moved off the click)"
        return route_facing, "the route's own facing"  # no clicked yaw

    def _off() -> float:
        return wrap_angle(_aim_heading()[0] - pose(ctx)[2])

    checker_ref = [route_checker]
    cap_mps = drive.cruise_mps if preserve_goal and not arrival_heading_required else None
    ticks, held = _walk_waypoints(ctx, waypoints, held, ticks,
                                  checker=route_checker, preserve_goal=preserve_goal,
                                  goal_xy=goal_xy if preserve_goal else None, radius_m=radius_m,
                                  checker_ref=checker_ref, route_id=route_id,
                                  clearance_margin_m=clearance_margin_m,
                                  turn_translation_limit_m=turn_translation_limit_m,
                                  route_start=route_start if not arrival_heading_required else None,
                                  **({"cap_mps": cap_mps} if cap_mps is not None else {}))
    goal_line = ((waypoints[-1] if waypoints else route_start)
                 if preserve_goal and route_start is not None and not arrival_heading_required
                 else tuple(pose(ctx)[:2]) if preserve_goal else None)
    check = (_RouteMotionCheck(ctx, checker_ref[0], goal_xy, radius_m=radius_m,
                               checker_ref=checker_ref, route_id=route_id,
                               clearance_margin_m=clearance_margin_m,
                               turn_translation_limit_m=turn_translation_limit_m,
                               route_start=(waypoints[-1] if waypoints else route_start)
                               if route_start is not None and not arrival_heading_required
                               else None)
             if preserve_goal else None)
    rounds, drove, closes, aimed = 0, False, 0, False
    while not ctx.deadline.expired():
        rounds += 1
        bearing, gap = _bearing_to(ctx, goal_xy)
        if gap <= tolerance_m:
            break
        if check is not None:
            check()
        # a close overshoot is an arrival, not a U-turn
        if drove and abs(bearing) > math.pi / 2 and gap <= drive.overshoot_m:
            ctx.log("drive.overshot", {"gap_m": round(gap, 4),
                                       "bearing_deg": round(math.degrees(bearing), 1),
                                       "why": "past the goal and close: arrived, not turning back"})
            break
        # take the heading once, while the point is still ahead
        if arrival_heading_required and AIM_IN_M > 0.0 and not aimed and gap <= AIM_IN_M:
            aimed = True
            off_in, whose = _off(), _aim_heading()[1]
            ctx.log("drive.aim_in", {"gap_m": round(gap, 4), "from": whose,
                                     "off_deg": round(math.degrees(off_in), 1)})
            if abs(off_in) > align_rad:
                stage(ctx, "taking the heading before the last step",
                      f"{math.degrees(abs(off_in)):.0f} deg, {gap * 100:.0f} cm out")
                ticks += _stand(ctx)
                if check is not None:
                    check.turn(True)
                over = _turn_budget(ctx)
                ticks += _hold_until(
                    ctx, lambda: (0.0, 0.0, _turn_rate(_off())),
                    lambda over=over: abs(_off()) <= align_rad or over(),
                    "drive.aim_in",
                    "take the heading now: the last of the gap is stepped, not turned",
                    CONTINUOUS_MAX_S, expect_s=abs(off_in) / turn_speed(),
                    **({"check": check} if check is not None else {}))
                ticks += _stand(ctx)
                if check is not None:
                    check(turn_complete=True)
                held = _carry_check(ctx, held, "aim in")
            continue
        # close enough: step across the gap rather than turn onto it
        if (aimed or gap <= drive.close_step_m) and closes < CLOSE_STEP_MAX:
            closes += 1
            stage(ctx, "stepping onto the goal", f"{gap * 100:.0f} cm, no turn")
            if check is not None:
                check.turn(False)
            try:
                ticks += _close_goal(
                    ctx, goal_xy, tolerance_m, _off if arrival_heading_required else lambda: 0.,
                    **({"cap_mps": cap_mps} if cap_mps is not None else {}),
                    **({"check": check} if check is not None else {}))
            except _CloseMovingAway as exc:
                ticks += exc.ticks
                if closes >= CLOSE_STEP_MAX or ctx.deadline.expired():
                    raise
                held = _carry_check(ctx, held, f"close {closes} stopped")
                # the next pass computes a fresh twist to the same stance
                ctx.log("drive.close_retry", {"attempt": closes + 1,
                                              "of": CLOSE_STEP_MAX, "reason": str(exc)})
                drove = False  # this stopped slice did not verify an overshoot arrival
                continue
            drove = True
            held = _carry_check(ctx, held, f"close {closes}")
            continue
        if aimed:
            ctx.log("drive.close_spent", {"gap_m": round(gap, 4), "closes": closes,
                                          "why": "the heading is taken: stepping got this close"})
            break
        if abs(bearing) > LEG_ALIGN_RAD:
            stage(ctx, "turning to the goal", f"{math.degrees(abs(bearing)):.0f} deg off")
            ticks += _stand(ctx)
            if check is not None:
                check.turn(True)
            over = _turn_budget(ctx)
            ticks += _hold_until(
                ctx, lambda: (0.0, 0.0, _turn_rate(_bearing_to(ctx, goal_xy)[0])),
                lambda over=over: (
                    abs(_bearing_to(ctx, goal_xy)[0]) <= LEG_ALIGN_RAD or over()),
                "drive.turn", "turn until the nose is on the goal", CONTINUOUS_MAX_S,
                expect_s=abs(bearing) / turn_speed(),
                **({"check": check} if check is not None else {}))
            ticks += _stand(ctx)
            if check is not None:
                check(turn_complete=True)
            held = _carry_check(ctx, held, f"turn {rounds}")
            continue
        stage(ctx, "driving to the goal", f"{gap * 100:.0f} cm ahead")
        if check is not None:
            check.turn(False)
        seen: dict = {}
        looked = [float(ctx.now())]
        line0 = goal_line if preserve_goal else pose(ctx)[:2]

        def stop_driving(line0=line0, seen=seen, looked=looked, aimed=aimed) -> bool:
            bearing_now, gap_now = _bearing_to(ctx, goal_xy)
            if gap_now <= tolerance_m or abs(bearing_now) > leave_rad:
                return True
            if ((check is None or check.route_start is None)
                    and _off_line(ctx, line0, goal_xy) > LEG_CROSS_TRACK_M):
                if check is not None:
                    check.refuse("NO_COLLISION_FREE_PATH", "left the retained route corridor")
                seen.update(off_line_m=round(_off_line(ctx, line0, goal_xy), 3))
                return True              # drifted off the line: stop, turn, drive on
            # hand back at the aim band to take the heading
            if arrival_heading_required and not aimed and AIM_IN_M > 0.0 and gap_now <= AIM_IN_M:
                return True
            now = float(ctx.now())
            if now - looked[0] < CONTINUOUS_LOOK_S:
                return False
            looked[0] = now
            ahead = _eyes_ahead(ctx)
            if ahead is not None:
                seen.update(ahead)
                return True
            return False

        ticks += _hold_until(
            ctx, lambda line0=line0, on={}: _leg_twist(
                ctx, _route_lookahead(pose(ctx)[:2],
                                     check.route_start if check.route_start is not None
                                     else line0,
                                     goal_xy)
                if preserve_goal else goal_xy,
                check.route_start if check is not None and check.route_start is not None
                else line0,
                on,
                tolerance_m, goal_xy if preserve_goal else None,
                **({"cap_mps": cap_mps} if cap_mps is not None else {})),
            stop_driving,
            "drive.go", "drive straight at the goal", CONTINUOUS_MAX_S,
            expect_s=(gap - max(tolerance_m, AIM_IN_M
                               if arrival_heading_required and not aimed else 0.0))
            / max(drive.cruise_mps, 1e-6),
            **({"check": check} if check is not None else {}))
        drove = True
        held = _carry_check(ctx, held, f"drive {rounds}")
        if seen:
            ctx.log("drive.too_close", {k: v for k, v in seen.items() if k != "cells"})
            if check is not None:
                check.refuse("NO_COLLISION_FREE_PATH", "near-field obstacle stopped the route")
    # final heading turn when the aim-in did not take it
    if arrival_heading_required:
        aim0, whose = _aim_heading()
        off0 = _off()
        ctx.log("drive.final_heading", {
            "aim_rad": round(aim0, 4), "off_deg": round(math.degrees(off0), 1),
            "aimed_in": aimed, "from": whose})
        if abs(off0) > align_rad:
            stage(ctx, "arrived: facing the target", f"{math.degrees(abs(off0)):.0f} deg off")
            ticks += _stand(ctx)
            if check is not None:
                check.turn(True)
            over = _turn_budget(ctx)
            ticks += _hold_until(
                ctx, lambda: (0.0, 0.0, _turn_rate(_off())),
                lambda over=over: abs(_off()) <= align_rad or over(),
                "final.turn", "arrived: face the target from this stance",
                CONTINUOUS_MAX_S, expect_s=abs(off0) / turn_speed(),
                **({"check": check} if check is not None else {}))
            if check is not None:
                ticks += _stand(ctx)
                check(turn_complete=True)
    x, y, yaw = pose(ctx)
    gap = math.hypot(goal_xy[0] - x, goal_xy[1] - y)
    ctx.log("drive.continuous", {"rounds": rounds, "gap_m": round(gap, 4),
                                 "pose": [round(x, 4), round(y, 4), round(yaw, 4)]})
    return {"ticks": ticks, "gap_m": gap, "held": held}


def _burst(ctx, tag: str, seconds: float, vel, why: str = "") -> tuple[int, float, float]:
    """One open-loop WALK or TURN plus its settle, logged as `drive.burst` (gait.burst)."""
    return _gait_burst(ctx, "drive.burst", tag, seconds, vel, why=why)



def effective_planning_radius(requested_radius, held) -> float:
    """Resolve the same footprint for callers and preparation; UNKNOWN stays conservative."""
    radius = validate_radius_m(requested_radius)
    if any(v is UNKNOWN or v is not None for v in held.values()):
        radius = max(radius, CARRY_SWEEP_RADIUS_M)
    return radius


def prepare(ctx, args, *, cancelled=None) -> dict:
    """Plan the collision-checked route `execute` walks to `goal_xy_m` (map metres), facing
    `facing_xy_m` when given; the other args set footprint, tolerance and exploration flags."""
    ctx.progress("VALIDATE")
    goal_xy = require_xy(args.get("goal_xy_m"), "goal_xy_m")
    facing_xy = args.get("facing_xy_m")
    snap = bool(args.get("snap", True))  # whether a goal inside geometry may be relocated
    preserve_goal = args.get("preserve_goal", False)
    arrival_heading_required = args.get("arrival_heading_required", True)
    preferred_clearance = args.get("preferred_clearance", True)
    if any(type(flag) is not bool for flag in (
            preserve_goal, arrival_heading_required, preferred_clearance)):
        raise DriveFailure("CONTROLLER_REFUSED", "drive exploration flags must be booleans")
    if "execution_radius_m" in args and not preserve_goal:
        raise DriveFailure("CONTROLLER_REFUSED", "execution_radius_m requires preserve_goal")
    face_centre = not bool(args.get("facing_is_target", False))  # a target is not face-centred
    if facing_xy is not None:
        facing_xy = require_xy(facing_xy, "facing_xy_m")
    tolerance_m = finite(args.get("arrival_tolerance_m", current().drive.tolerance_m),
                         "arrival_tolerance_m")
    if tolerance_m <= 0.0:
        raise DriveFailure("CONTROLLER_REFUSED",
                           f"arrival_tolerance_m must be positive, got {tolerance_m}")
    turn_translation_limit_m = finite(args.get("turn_translation_limit_m", TURN_DRIFT_CAP_M),
                                      "turn_translation_limit_m")
    if turn_translation_limit_m <= 0.:
        raise DriveFailure("CONTROLLER_REFUSED", "turn_translation_limit_m must be positive")
    # Only a composition with measured sweep enforcement can admit a larger footprint.
    # A model argument cannot enlarge this capability; ordinary exploration is capped at .60 m.
    maximum_radius = (getattr(ctx, "max_footprint_radius_m", .60)
                      if preserve_goal and "execution_radius_m" in args else .60)
    held = _held_snapshot(ctx)
    requested_radius = args.get("radius_m", DEFAULT_RADIUS_M)
    try:
        if preserve_goal and "planning_radius_m" in args:
            requested_radius = validate_radius_m(requested_radius, maximum_m=maximum_radius)
            radius_m = validate_radius_m(args["planning_radius_m"], maximum_m=maximum_radius)
            if radius_m < requested_radius:
                raise ValueError("planning_radius_m must be >= requested radius_m")
        else:
            radius_m = effective_planning_radius(requested_radius, held)
            if "planning_radius_m" in args:
                radius_m = max(radius_m, validate_radius_m(args["planning_radius_m"]))
    except ValueError as exc:
        raise DriveFailure("CONTROLLER_REFUSED", str(exc)) from exc
    ctx.log("drive.planning_footprint", {"requested_radius_m": requested_radius,
        "selected_radius_m": args.get("planning_radius_m"), "radius_m": radius_m,
        "footprint_source": ("exploration" if preserve_goal and "planning_radius_m" in args
                             else "carry_aware"),
        "held": {hand: repr(value) for hand, value in held.items()}})
    execution_geometry = {}
    if "execution_radius_m" in args:
        execution_radius_m = finite(args["execution_radius_m"], "execution_radius_m")
        if not 0. < execution_radius_m <= radius_m:
            raise DriveFailure("CONTROLLER_REFUSED",
                               "execution_radius_m must be positive and <= planning radius_m")
        execution_geometry["execution_radius_m"] = execution_radius_m
    exact_footprint = {"exact_footprint": True} if execution_geometry else {}
    art = ctx.observe("map")
    quality = float(ctx.observe("loc_quality"))
    waited = 0.0
    while quality < LOC_QUALITY_FLOOR and waited < LOC_RECOVERY_WAIT_S:
        getattr(ctx, "wait", time.sleep)(0.25)   # no command: the localizer settles
        waited += 0.25
        quality = float(ctx.observe("loc_quality"))
    if quality < LOC_QUALITY_FLOOR:
        raise DriveFailure(
            "LOCALIZATION_DEGRADED",
            f"loc_quality {quality:.3f} < floor {LOC_QUALITY_FLOOR} before planning, "
            f"still after {waited:.1f} s",
        )
    x0, y0, yaw0 = pose(ctx)
    ctx.progress("PLAN")
    stage(ctx, "planning a route", f"to ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) m")
    requested = (float(goal_xy[0]), float(goal_xy[1]))
    backed = (None if preserve_goal else
              _usable_goal(art, goal_xy, facing_xy, radius_m, base_xy=(x0, y0)))
    if backed is not None:
        ctx.log("drive.goal_backed_off", {
            "requested": [round(float(v), 4) for v in goal_xy],
            "usable": [round(float(v), 4) for v in backed],
            "moved_m": round(math.dist(goal_xy, backed), 4),
        })
        goal_xy = backed
    start_xy = (x0, y0)
    ctx.log("drive.plan_start", {"pose": [round(x0, 4), round(y0, 4), round(yaw0, 4)],
                                 "start_xy": [round(v, 4) for v in start_xy],
                                 "behind_m": round(math.dist((x0, y0), start_xy), 3)})
    if preserve_goal:
        exact_checker = occupancy_grid(art, radius_m, **exact_footprint)[0]
        for name, point in (("exact_start", start_xy), ("accepted_goal", goal_xy)):
            if not exact_checker.is_safe(*point):
                raise _collision_failure("exact start or accepted goal is blocked",
                                         exact_checker, name, (point, point),
                                         map_generation=getattr(art, "scan_hash", None))
    endpoint_radius = radius_m if preserve_goal else GOAL_RADIUS_M
    def planning_cancelled():
        return (cancelled is not None and cancelled()) or ctx.deadline.expired()

    route = _plan_route(art, start_xy, goal_xy, radius_m, endpoint_radius,
                        start_radius_m=endpoint_radius, snap=False if preserve_goal else snap,
                        facing_xy=facing_xy,
                        face_centre=face_centre, preferred_clearance=preferred_clearance,
                        **exact_footprint,
                        cancelled=(planning_cancelled if cancelled is not None or
                                   (preserve_goal and not arrival_heading_required) else None),
                        **({"clearance_margin_m": EXPLORATION_CLEARANCE_MARGIN_M}
                           if preserve_goal and not arrival_heading_required else {}))
    if (preserve_goal or cancelled is not None) and planning_cancelled():
        raise DriveFailure("TIMEOUT", "drive deadline expired during route planning")
    if backed is not None and route.path is not None:
        # arrival is measured against the requested point
        route = _replace(route, requested_goal=(float(requested[0]), float(requested[1])))
    drive_back = None
    blocked_start = route.path is None and route.code == "START_BLOCKED"
    if preserve_goal and route.path is None:
        raise DriveFailure("NO_COLLISION_FREE_PATH", " ".join(
            p for p in (route.code, route.reason, route.diagnostic) if p))
    if blocked_start and not route.checker.contains(x0, y0):
        # off the sampled region is POSE_OFF_MAP, not a blocked start
        bx0, bx1, by0, by1 = route.checker.bounds()
        ctx.log("drive.pose_off_map", {
            "pose": [round(x0, 4), round(y0, 4), round(yaw0, 4)],
            "map_bounds": [round(v, 4) for v in (bx0, bx1, by0, by1)]})
        raise DriveFailure(
            "POSE_OFF_MAP",
            f"base at ({x0:.4f}, {y0:.4f}) is outside the occupancy map "
            f"x[{bx0:.4f}, {bx1:.4f}] y[{by0:.4f}, {by1:.4f}]")
    if blocked_start:
        # one backward walk clears the start; `execute` replans from where it lands
        landing = (x0 + START_BACKUP_M * math.cos(yaw0 + math.pi),
                   y0 + START_BACKUP_M * math.sin(yaw0 + math.pi))
        drive_back = (START_BACKUP_M, landing, math.pi)
        preview_path = [(x0, y0), landing]
        route_px = [
            [round(u), round(v)]
            for u, v in world_to_px(np.asarray(art.T_map_px, dtype=float),
                                    np.asarray(preview_path, dtype=float))
        ]
        ctx.log("drive.route", {
            "code": "BACKUP_THEN_REPLAN", "waypoints": 2,
            "route_len_m": round(START_BACKUP_M, 4), "start_snap_m": 0.0,
            "goal_snap_m": None, "seeds_tried": [], "radius_m": float(radius_m),
            "route_px": route_px,
            "route_world_xy": [[float(x), float(y)] for x, y in preview_path],
            "requested_goal_world_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "goal_world_xy": [float(landing[0]), float(landing[1])],
            "facing_world_xy": None,
        })
        return {"route": _without_grids(route), "route_px": route_px, "facing_xy": facing_xy,
                "tolerance_m": tolerance_m, "held": held, "drive_back": drive_back,
                "goal_xy": goal_xy, "radius_m": radius_m,
                **execution_geometry,
                "turn_translation_limit_m": turn_translation_limit_m}
    if route.path is None:
        raise DriveFailure(
            "NO_COLLISION_FREE_PATH",
            " ".join(p for p in (route.code, route.reason, route.diagnostic) if p),
        )
    if preserve_goal:
        # drop a short redundant first corner left by the exact endpoints
        path = list(route.path)
        while (not route.clearance_margin_m and len(path) > 2
               and math.dist(path[0], path[1]) < current().drive.waypoint_tol_m
               and _line_free(exact_checker, path[0], path[2])):
            del path[1]
        route = _replace(route, path=path)
    spacing = current().drive.waypoint_spacing_m
    walked = ((tuple(route.path[:1])
               + (_resample(route.path, spacing) if preserve_goal else
                  _run_in(_resample(route.path, spacing), route.goal,
                          route.faced or facing_xy, route.landing))
               + tuple(route.path[-1:])) if len(route.path) >= 2 else tuple(route.path))
    if preserve_goal:
        _validate_preserved_route(exact_checker, walked, start_xy, requested)
    route_id = (hashlib.sha256(np.asarray(walked, dtype="<f8").tobytes()
                               + np.asarray([radius_m], dtype="<f8").tobytes()).hexdigest()
                if preserve_goal else None)
    route_px = [
        [round(u), round(v)]
        for u, v in world_to_px(np.asarray(art.T_map_px, dtype=float),
                                np.asarray(walked, dtype=float))
    ]
    goal_px = world_to_px(np.asarray(art.T_map_px, dtype=float),
                          np.asarray([route.goal], dtype=float))[0]
    facing_px = (None if facing_xy is None else
                 world_to_px(np.asarray(art.T_map_px, dtype=float),
                             np.asarray([facing_xy], dtype=float))[0])
    # `walked`: the corners plus the resampled interior
    ctx.log("drive.route", {
        "code": "PATH_READY", "phase": "prepared", "waypoints": len(walked),
        "route_id": route_id, "map_generation": getattr(art, "scan_hash", None),
        "planner_corners": len(route.path),
        "route_len_m": round(float(path_len(route.path)), 4),
        "start_snap_m": round(math.hypot(route.start[0] - x0, route.start[1] - y0), 4),
        "goal_snap_m": round(math.hypot(route.goal[0] - goal_xy[0],
                                        route.goal[1] - goal_xy[1]), 4),
        "seeds_tried": list(route.seeds_tried), "radius_m": float(radius_m),
        **execution_geometry,
        "preferred_clearance_margin_m": route.clearance_margin_m,
        "turn_translation_limit_m": turn_translation_limit_m,
        "route_px": route_px,
        "route_world_xy": [[float(x), float(y)] for x, y in walked],
        "requested_goal_world_xy": [float(goal_xy[0]), float(goal_xy[1])],
        "goal_world_xy": [float(route.goal[0]), float(route.goal[1])],
        "facing_world_xy": (None if facing_xy is None else
                            [float(facing_xy[0]), float(facing_xy[1])]),
        "goal_px": [float(goal_px[0]), float(goal_px[1])],
        "facing_px": (None if facing_px is None else
                      [float(facing_px[0]), float(facing_px[1])]),
    })
    return {"route": _without_grids(route), "route_px": route_px, "facing_xy": facing_xy,
            "tolerance_m": tolerance_m, "held": held, "drive_back": None,
            "goal_xy": goal_xy, "radius_m": radius_m,
            "requested_radius_m": requested_radius,
            **({"planning_radius_m": radius_m}
               if preserve_goal and "planning_radius_m" in args else {}),
            **execution_geometry,
            "turn_translation_limit_m": turn_translation_limit_m,
            "preserve_goal": preserve_goal, "arrival_heading_required": arrival_heading_required,
            "route_rejoin": preserve_goal and not arrival_heading_required
            and bool(args.get("route_rejoin", False)),
            "walked_route": walked, "route_id": route_id}

def execute(ctx, prepared) -> dict:
    """Walk the prepared route, then measure arrival from base_pose: the arrival record."""
    preserve_goal = prepared.get("preserve_goal", False)
    route_px = prepared["route_px"]
    facing_xy = prepared["facing_xy"]
    tolerance_m = prepared["tolerance_m"]
    held = prepared["held"]
    if (preserve_goal and "requested_radius_m" in prepared
            and "planning_radius_m" not in prepared):
        widened = effective_planning_radius(prepared["requested_radius_m"],
                                            _held_snapshot(ctx))
        if widened > prepared["radius_m"] + 1e-9:
            raise DriveFailure("NO_COLLISION_FREE_PATH",
                "planning footprint widened after preparation; reprepare before moving")
    base_art = ctx.observe("map")
    route = prepared["route"]
    execution_radius_m = float(prepared.get("execution_radius_m", prepared["radius_m"]))
    exact_footprint = {"exact_footprint": True} if "execution_radius_m" in prepared else {}
    if preserve_goal:
        checker = occupancy_grid(base_art, execution_radius_m, **exact_footprint)[0]
        route = _replace(route, checker=checker, landing=checker, transit=checker)
        walked = tuple(prepared["walked_route"])
        _validate_preserved_route(checker, walked, route.requested_start, route.requested_goal)
        actual_start = tuple(pose(ctx)[:2])
        first_target = walked[1] if len(walked) > 1 else route.goal
        if not _line_free(checker, actual_start, first_target):
            raise _collision_failure("actual start cannot connect to accepted route",
                                     checker, "actual_start_connector",
                                     (actual_start, first_target),
                                     route_id=prepared.get("route_id"),
                                     map_generation=getattr(base_art, "scan_hash", None))
        ctx.log("drive.route", {"code": "PATH_READY", "phase": "execute",
                                "route_id": prepared.get("route_id"),
                                "map_generation": getattr(base_art, "scan_hash", None),
                                "route_world_xy": [list(p) for p in walked],
                                "actual_start_world_xy": list(actual_start),
                                "requested_goal_world_xy": list(route.requested_goal),
                                "goal_world_xy": list(route.goal), "route_px": route_px,
                                "radius_m": float(prepared["radius_m"]),
                                **({"execution_radius_m": execution_radius_m}
                                   if exact_footprint else {}),
                                "turn_translation_limit_m": prepared.get(
                                    "turn_translation_limit_m", TURN_DRIFT_CAP_M),
                                "preferred_clearance_margin_m": route.clearance_margin_m})
    else:
        route = _with_grids(route, base_art, float(prepared["radius_m"]))
    drive_back = prepared["drive_back"]

    ctx.progress("EXECUTE")
    if drive_back is not None:
        # blocked start: plan from wherever the base is
        route = _replan_until_found(ctx, prepared["goal_xy"],
                                    float(prepared["radius_m"]))
        art = ctx.observe("map")
        route_px = [
            [round(u), round(v)]
            for u, v in world_to_px(np.asarray(art.T_map_px, dtype=float),
                                    np.asarray(route.path, dtype=float))
        ]
        ctx.log("drive.route", {
            "code": "PATH_READY_AFTER_BACKUP", "waypoints": len(route.path),
            "route_len_m": round(float(path_len(route.path)), 4),
            "route_px": route_px,
            "route_world_xy": [[float(a), float(b)] for a, b in route.path],
            "requested_goal_world_xy": list(map(float, prepared["goal_xy"])),
            "goal_world_xy": list(map(float, route.goal)),
            "facing_world_xy": (None if facing_xy is None else list(map(float, facing_xy))),
        })
    ticks = 0
    carry_mode = any(v is UNKNOWN or v is not None for v in held.values())
    if carry_mode:
        # warm the gait up at zero twist before carrying
        t, _, _ = _burst(ctx, "carry.warmup", MIN_BURST_S, (0.0, 0.0, 0.0),
                    why="warm the gait up before carrying")
        ticks += t
        held = _carry_check(ctx, held, "carry warmup")
    # every base_pose read below passes the pose gate
    reck = _Localized(ctx)
    ctx = _replace(ctx, observe=reck)
    out = _continuous_drive(ctx, route.goal, facing_xy, tolerance_m, held,
                            route_facing=route.facing,
                            route_checker=route.checker,
                            arrival_heading_required=prepared.get(
                                "arrival_heading_required", True),
                            preserve_goal=preserve_goal,
                            clearance_margin_m=route.clearance_margin_m,
                            turn_translation_limit_m=prepared.get(
                                "turn_translation_limit_m", TURN_DRIFT_CAP_M),
                            radius_m=execution_radius_m,
                            route_id=prepared.get("route_id"),
                            route_start=(walked[0] if prepared.get("route_rejoin")
                                         else None),
                            asked_xy=route.requested_goal,
                            waypoints=walked[1:-1] if preserve_goal else _run_in(
                                _resample(route.path, current().drive.waypoint_spacing_m),
                                route.goal, route.faced or facing_xy,
                                route.landing))
    ticks += int(out["ticks"])
    held = _carry_check(ctx, out["held"], "final")
    ticks += _settle_after_drive(ctx, held)
    ctx.progress("VERIFY_ARRIVAL")
    asked = (route.requested_goal if route.requested_goal is not None else route.goal)
    if preserve_goal:
        try:
            x1, y1, yaw1 = reck.read("arrival", accepted_only=True)
        except StaleObservation as exc:
            raise DriveFailure("ARRIVAL_NOT_VERIFIED", str(exc)) from None
    else:
        x1, y1, yaw1 = pose(ctx)
    gap_m = math.dist((x1, y1), route.goal)
    landing_radius = execution_radius_m if preserve_goal else GOAL_RADIUS_M
    landing = occupancy_grid(ctx.observe("map"), landing_radius,
                             **(exact_footprint if preserve_goal else {}))[0]
    footprint_safe = bool(landing.is_safe(x1, y1))
    stage(ctx, "checking arrival",
          f"{gap_m * 100:.0f} cm from the stance, "
          f"{math.dist((x1, y1), asked) * 100:.0f} cm from the requested point")
    ctx.log("drive.arrival", {
        "pose": [round(x1, 4), round(y1, 4), round(yaw1, 4)],
        "gap_m": round(gap_m, 4),
        "snap_m": round(math.dist(route.goal, route.requested_goal), 4)
        if route.requested_goal is not None else 0.0,
        "requested_gap_m": round(math.dist((x1, y1), route.requested_goal), 4)
        if route.requested_goal is not None else None,
        "arrival_tolerance_m": tolerance_m,
        "footprint_safe": footprint_safe})
    ctx.progress("TERMINAL_RESULT")
    return {
        "arrived": gap_m <= tolerance_m and footprint_safe,
        "gap_m": round(gap_m, 4),
        "requested_gap_m": round(math.dist((x1, y1), route.requested_goal), 4)
        if route.requested_goal is not None else round(gap_m, 4),
        "legs": 0,
        "commanded_m": 0.0,
        "realized_m": 0.0,
        "route_px": route_px,
        "turns": 0,
        "continuous": True,
    }

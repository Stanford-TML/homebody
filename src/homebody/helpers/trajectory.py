"""Rate-bounded joint trajectories, the overhead reach path and the palm servo step."""
from collections import Counter
from itertools import pairwise

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import rotation_error

MINJERK_PEAK_RATE = 1.875
OUT_OF_REACH = "route outside arm reach"


def minjerk(value):
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def composite_arc(start, end, *, out, over, descend, side, count, samples):
    """COUNT torso-frame palm positions from START to END: out beside the body by OUT,
    OVER above the higher end, then straight down by DESCEND, min-jerk timed."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    if (start.shape != (3,) or end.shape != (3,) or not np.isfinite([start, end]).all()
            or side not in ("left", "right") or count < 2 or samples < 2
            or not np.isfinite([out, over, descend]).all() or min(out, over, descend) < 0):
        raise ValueError("Composite geometry requires finite endpoints and nonnegative clearances")
    delta = end[:2] - start[:2]
    distance = np.linalg.norm(delta)
    forward = np.r_[delta / distance, 0.] if distance > 1e-6 else np.array([1., 0., 0.])
    outboard = np.array([0., 1. if side == "left" else -1., 0.])
    up = np.array([0., 0., 1.])
    height = max(max(start[2], end[2]) + over, end[2] + descend, start[2])
    rise = height - start[2]
    chord = float(np.hypot(distance, out))
    tangent = (distance * forward - out * outboard) / chord if chord > 1e-12 else forward
    first_length = min(float(np.hypot(out, rise)), chord) / 3
    corner = start + out * outboard + rise * up
    above = np.r_[end[:2], height]
    second_length = np.linalg.norm(above - corner) / 3
    controls = ((start, start + first_length * outboard, corner - first_length * tangent, corner),
                (corner, corner + second_length * tangent, above + second_length * up, above))
    t = np.linspace(0., 1., samples)[:, None]
    pieces = [((1 - t)**3 * a + 3 * (1 - t)**2 * t * b +
               3 * (1 - t) * t**2 * c + t**3 * d) for a, b, c, d in controls]
    pieces.append((1 - t) * above + t * end)
    dense = np.concatenate((pieces[0], pieces[1][1:], pieces[2][1:]))
    points = arc_resample(dense, minjerk(np.linspace(0., 1., count)))
    points[0], points[-1] = start, end
    return points


def arc_resample(rows, fraction):
    rows = np.asarray(rows, float)
    steps = np.linalg.norm(np.diff(rows, axis=0), axis=1)
    keep = np.r_[True, steps > 0]
    rows = rows[keep]
    length = np.r_[0., np.cumsum(steps[steps > 0])]
    if length[-1] <= 1e-12:
        return np.repeat(rows[:1], len(fraction), axis=0)
    return np.column_stack([np.interp(fraction * length[-1], length, rows[:, j])
                            for j in range(rows.shape[1])])


def retime_path(rows, rate, dt):
    """ROWS resampled one per DT tick at joint rate RATE, repeated rows dropped."""
    rows = np.array(rows, dtype=float, copy=True)
    if (rows.ndim != 2 or len(rows) < 2 or not np.isfinite(rows).all() or
            not np.isfinite(rate) or not np.isfinite(dt) or rate <= 0 or dt <= 0):
        raise ValueError("A finite joint path and positive rate/tick are required")
    rows = rows[np.r_[True, np.any(np.diff(rows, axis=0) != 0, axis=1)]]
    result = [rows[:1]]
    for before, after in pairwise(rows):
        count = max(1, int(np.ceil(np.max(np.abs(after - before)) / (rate * dt))))
        result.append(before + np.linspace(0., 1., count + 1)[1:, None] * (after - before))
    result = np.vstack(result)
    return result if len(result) > 1 else np.vstack((result, result))


def smooth_path(rows, passes):
    rows = np.array(rows, dtype=float, copy=True)
    for _ in range(passes):
        rows[1:-1] = .25 * rows[:-2] + .5 * rows[1:-1] + .25 * rows[2:]
    return rows


def solve_reference(arm, start, end, positions, seed_bias, cancelled):
    """Joint rows solving POSITIONS from START to END with an eased orientation ramp, or
    None when a sample does not solve or CANCELLED."""
    current, _ = arm.forward(start)
    goal, _ = arm.forward(end)
    rotation = Rotation.from_matrix(current[:3, :3].T @ goal[:3, :3]).as_rotvec()
    blend = minjerk(np.linspace(0., 1., len(positions)))
    rows = [np.array(start, copy=True)]
    for index, position in enumerate(positions[1:-1], 1):
        if cancelled():
            return None
        target = np.eye(4)
        target[:3, 3] = position
        target[:3, :3] = current[:3, :3] @ Rotation.from_rotvec(blend[index] * rotation).as_matrix()
        line = start + blend[index] * (end - start)
        seed = (1 - seed_bias) * rows[-1] + seed_bias * line
        joints = arm.solve(target, seed)
        if joints is None:
            joints = arm.solve(target, seed, position_only=True)
        if joints is None:
            return None
        rows.append(joints)
    rows.append(np.array(end, copy=True))
    return np.array(rows)


def joint_rows(start, end, rate, dt):
    """A min-jerk ramp from START to END, one row per DT, whose peak joint rate is RATE."""
    start, end = np.asarray(start), np.asarray(end)
    duration = max(dt, MINJERK_PEAK_RATE * np.max(np.abs(end - start)) / rate)
    count = int(np.ceil(duration / dt))
    for index in range(1, count + 1):
        yield start + minjerk(index / count) * (end - start)


def servo_step(arm, measured, commanded, target, rate, settings):
    """One servo tick toward the torso-frame TARGET: the next joint command and the palm's
    position (m) and rotation (rad) errors."""
    dt = settings.tick
    palm, jacobian = arm.forward(measured)
    ep = target[:3, 3] - palm[:3, 3]
    er = rotation_error(target[:3, :3], palm[:3, :3])
    velocity = np.r_[ep * min(settings.position_gain, settings.max_linear_speed / max(np.linalg.norm(ep), 1e-9)),
                     er * min(settings.rotation_gain, settings.max_angular_speed / max(np.linalg.norm(er), 1e-9))]
    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + settings.damping_squared * np.eye(6), velocity)
    desired = np.clip(commanded + delta * dt,
                      measured - settings.max_joint_lag, measured + settings.max_joint_lag)
    desired = np.clip(desired, arm.lower + 0.02, arm.upper - 0.02)
    command = commanded + np.clip(desired - commanded, -rate * dt, rate * dt)
    return (command,
            float(np.linalg.norm(ep)), float(np.linalg.norm(er)))


def approach_path(arm, start, end, clearance, settings, cancelled, *, initial_lift_m=0., blockers=None):
    """Joint rows, endpoints included, of the first configured overhead arc from START to END
    that clears, refusals counted in BLOCKERS by reason. None when none clears or on cancel."""
    blockers = Counter() if blockers is None else blockers
    start, end = np.asarray(start, float), np.asarray(end, float)
    prefix = ()
    if initial_lift_m:
        raised_pose, _ = arm.forward(start)
        raised_pose = raised_pose.copy()
        raised_pose[2, 3] += initial_lift_m
        raised = arm.solve(raised_pose, start)
        if raised is None or not clearance.path(start, raised):
            return None
        prefix, start = (start.copy(),), raised
    if np.array_equal(start, end):
        return (*prefix, start.copy(), end.copy()) if not cancelled() and clearance.path(start, end) else None
    current, _ = arm.forward(start)
    goal, _ = arm.forward(end)
    for out in settings.spline_out:
        for over in settings.spline_over:
            if cancelled():
                return None
            positions = composite_arc(current[:3, 3], goal[:3, 3], out=out, over=over,
                descend=settings.spline_descend, side=arm.collision.side,
                count=settings.spline_waypoints, samples=settings.spline_samples)
            rows = solve_reference(arm, start, end, positions, settings.spline_seed_bias, cancelled)
            if rows is None:
                blockers[OUT_OF_REACH] += 1
                continue
            rows = smooth_path(rows, settings.spline_smooth_passes)
            clear = True
            for before, after in pairwise(rows):
                if cancelled():
                    return None
                reason = clearance.path_violation(before, after)
                if reason is not None:
                    blockers[reason] += 1
                    clear = False
                    break
            if clear:
                return (*prefix, *rows)
    return None

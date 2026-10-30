"""The exploration gait's command envelope and burst primitives: burst sizing, the closed-loop
final turn, dead-band scaling, and the occupancy raster baked into a collision checker.
Body-frame twists in m/s and rad/s, map-frame poses in metres."""
from __future__ import annotations

import functools
import math

import numpy as np

from homebody.helpers.geometry import wrap
from homebody.helpers.raster import px_to_world
from homebody.motion.exploration import localizer
from homebody.motion.exploration.nav import (
    ExactGridCollisionChecker,
    GridCollisionChecker,
    inflate,
    inflate_exact_footprint,
)
from homebody.primitives.navigation.actions.lower_body import (
    VX_DEADBAND,
    WZ_FLOOR,
    WZ_MIN_ACCEPT,
    LowerBody,
)
from homebody.primitives.navigation.contracts import DriveFailure, Refused
from homebody.real2sim.exploration.settings import current

TICK_HZ = 250.0  # one lower_body step per tick
WALK_SPEED = 0.70  # forward walking speed, m/s
LATERAL_SPEED = 0.40  # backward and lateral speed, m/s

# Settled burst travel (metres = slope * seconds + intercept) and turn ratio
WALK_SLOPE_M_PER_S = 0.4966
WALK_INTERCEPT_M = -0.0353
SIDE_SLOPE_M_PER_S = 0.40
SIDE_INTERCEPT_M = 0.0
TURN_FACTOR = 0.885


def walk_seconds(metres: float) -> float:
    """Seconds of a 0.70 m/s burst whose settled travel is `metres`."""
    return (float(metres) - WALK_INTERCEPT_M) / WALK_SLOPE_M_PER_S


def walk_metres(seconds: float) -> float:
    return WALK_SLOPE_M_PER_S * float(seconds) + WALK_INTERCEPT_M


def side_metres(seconds: float) -> float:
    """Settled travel of a 0.40 m/s sidestep of `seconds`."""
    return SIDE_SLOPE_M_PER_S * float(seconds) + SIDE_INTERCEPT_M


# Side and reverse burst windows (min, max seconds)
HELD_SIDE_S = (0.6, 1.0)
HELD_BACK_SPEED = 0.40
HELD_BACK_RATE_M_S = 0.18       # net reverse per second at HELD_BACK_SPEED
HELD_BACK_S = (1.0, 1.0)
MICRO_WALK_S = 0.1              # forward pulse for distances under a stride
MICRO_WALK_M = 0.056            # travel of that pulse


def side_seconds(metres: float, cap: float = HELD_SIDE_S[1], speed: float | None = None) -> float:
    """Seconds of a sidestep that travels `metres` at `speed`, within HELD_SIDE_S up to `cap`."""
    nominal = current().gait.side_mps
    speed = nominal if speed is None else speed
    base = (abs(metres) - SIDE_INTERCEPT_M) / SIDE_SLOPE_M_PER_S
    return float(min(cap, max(HELD_SIDE_S[0], base * nominal / float(speed))))


def back_seconds(metres: float, cap: float = HELD_BACK_S[1],
                 speed: float = HELD_BACK_SPEED) -> float:
    """Seconds of a reverse that travels `metres` at `speed`, within HELD_BACK_S up to `cap`."""
    rate = HELD_BACK_RATE_M_S * float(speed) / HELD_BACK_SPEED
    return float(min(cap, max(HELD_BACK_S[0], abs(metres) / rate)))


def forward_seconds(metres: float) -> float:
    """Seconds of a forward walk for `metres`, or the micro-pulse under a stride."""
    if metres < STRIDE_M:
        return MICRO_WALK_S
    return float(min(current().gait.max_walk_s, max(MICRO_WALK_S, walk_seconds(metres))))


def held_move(ctx, log_kind: str, dx: float, dy: float, why: str = "",
              back_cap_s: float = HELD_BACK_S[1], back_speed: float = HELD_BACK_SPEED,
              side_speed: float | None = None) -> dict:
    """Walk a torso-frame displacement (forward, left) as a forward or reverse burst, then a side
    burst; returns what was sent, expected and moved."""
    side_speed = current().gait.side_mps if side_speed is None else side_speed
    out = {"dx_m": round(float(dx), 4), "dy_m": round(float(dy), 4), "bursts": []}
    if dx > 0.0:
        seconds = forward_seconds(dx)
        expect = MICRO_WALK_M if seconds <= MICRO_WALK_S else walk_metres(seconds)
        _t, moved, _ = burst(ctx, log_kind, "restance", seconds, (WALK_SPEED, 0.0, 0.0), why=why)
        out["bursts"].append({"axis": "forward", "seconds": round(seconds, 3),
                              "expect_m": round(expect, 4), "moved_m": round(moved, 4)})
    elif dx < 0.0:
        seconds = back_seconds(dx, cap=back_cap_s, speed=back_speed)
        _t, moved, _ = burst(ctx, log_kind, "restance", seconds, (-float(back_speed), 0.0, 0.0),
                             why=why)
        out["bursts"].append({"axis": "back", "seconds": round(seconds, 3),
                              "expect_m": round(HELD_BACK_RATE_M_S * float(back_speed)
                                                / HELD_BACK_SPEED * seconds, 4),
                              "moved_m": round(moved, 4)})
    if abs(dy) > 1e-6:          # ignore rounding residue of a straight move
        seconds = side_seconds(dy, speed=side_speed)
        _t, moved, _ = burst(ctx, log_kind, "restance", seconds,
                             (0.0, math.copysign(side_speed, dy), 0.0), why=why)
        out["bursts"].append({"axis": "left" if dy > 0 else "right", "seconds": round(seconds, 3),
                              "expect_m": round(side_metres(seconds), 4),
                              "moved_m": round(moved, 4)})
    return out


def turn_speed() -> float:
    """The cruise turn rate: the gait's setting, at least WZ_FLOOR."""
    return max(current().gait.turn_rad_s, WZ_FLOOR)


MIN_BURST_S = 1.0  # shortest motion command
MIN_WALK_GAP_M = 0.15  # smallest gap a walk can close
SETTLE_S = 1.0  # zero-velocity dwell after a burst
FINAL_TURN_STEP_RAD = 0.6  # largest rotation one turn burst may realize


def turn_burst_cap_s(yaw_rate: float) -> float:
    """The longest turn burst at `yaw_rate`, at least the cruise-rate cap."""
    cruise = FINAL_TURN_STEP_RAD / (turn_speed() * TURN_FACTOR)
    realized = FINAL_TURN_STEP_RAD / (max(abs(float(yaw_rate)), 1e-9) * TURN_FACTOR)
    return max(cruise, realized)


# Final-turn servo
FINAL_TURN_SLICE_S = 0.15
FINAL_TURN_SLEW_RAD_S2 = 1.0
FINAL_TURN_MAX_SLICES = 60
FINAL_TURN_SETTLE_S = 0.8            # standing time after the last slice
LOC_QUALITY_FLOOR = 0.5  # localization quality below this refuses a drive
TURN_GATE_RAD = math.radians(10.0)  # heading error under which the drive does not turn first


def occupancy_grid(art, radius_m: float, *, exact_footprint=False,
                   obstacles=None) -> tuple[GridCollisionChecker, tuple]:
    """Bake a MapArtifact into a collision checker dilated by `radius_m`; returns (checker,
    (xmin, xmax, ymin, ymax)) of the cell centres in map metres."""
    if not math.isfinite(radius_m) or radius_m < 0:
        raise ValueError("footprint radius must be finite and nonnegative")
    occ = np.asarray(art.occupancy, dtype=bool)
    known = None
    if obstacles is not None:
        known = np.asarray(obstacles, dtype=bool)
        if known.shape != occ.shape:
            raise ValueError("obstacles shape must match occupancy")
        known = known & occ
    allowed = getattr(art, "center_allowed", None)
    if allowed is not None:
        allowed = np.asarray(allowed, dtype=bool)
        if allowed.shape != occ.shape:
            raise ValueError("map center_allowed shape must match occupancy")
    h, w = occ.shape
    t = np.asarray(art.T_map_px, dtype=float)
    p00 = px_to_world(t, np.array([[0.0, 0.0]]))[0]
    du = px_to_world(t, np.array([[1.0, 0.0]]))[0] - p00  # world step per +1 u
    dv = px_to_world(t, np.array([[0.0, 1.0]]))[0] - p00  # world step per +1 v
    if abs(du[0]) >= abs(dv[0]):  # u runs along world x
        blocked, sx, sy = occ.T, du[0], dv[1]  # -> [i along x, j along y]
        allowed = None if allowed is None else allowed.T
        known = None if known is None else known.T
    else:  # quarter-turned raster: v runs along world x
        blocked, sx, sy = occ, dv[0], du[1]
    if sx < 0.0:
        blocked = blocked[::-1, :]  # grid i increases with world x
        allowed = None if allowed is None else allowed[::-1, :]
        known = None if known is None else known[::-1, :]
    if sy < 0.0:
        blocked = blocked[:, ::-1]  # grid j increases with world y
        allowed = None if allowed is None else allowed[:, ::-1]
        known = None if known is None else known[:, ::-1]
    cell = 0.5 * (abs(float(sx)) + abs(float(sy)))
    corners = px_to_world(t, np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float))
    xmin, xmax = float(corners[:, 0].min()), float(corners[:, 0].max())
    ymin, ymax = float(corners[:, 1].min()), float(corners[:, 1].max())
    dilate = inflate_exact_footprint if exact_footprint else inflate
    free = ~dilate(np.ascontiguousarray(blocked), cell, float(radius_m))
    if allowed is not None:
        free &= allowed
    nx, ny = free.shape
    checker_type = ExactGridCollisionChecker if exact_footprint else GridCollisionChecker
    geometry = ({"occupied": np.array(blocked, dtype=bool, copy=True),
                 "obstacles": None if known is None else np.array(known, dtype=bool, copy=True),
                 "center_allowed": None if allowed is None else
                 np.array(allowed, dtype=bool, copy=True)} if exact_footprint else {})
    checker = checker_type(
        xmin=xmin, ymin=ymin, cell=cell, nx=int(nx), ny=int(ny),
        grid=np.ascontiguousarray(free, dtype=bool), radius_baked=float(radius_m),
        **geometry,
    )
    return checker, (xmin, xmax, ymin, ymax)


GOAL_RADIUS_M = 0.25  # endpoint radius


def pose(ctx) -> tuple[float, float, float]:
    """(x, y, yaw) in the map frame, waiting standing while the localizer is stale."""
    if getattr(ctx.observe, "_owns_pose_consistency", False):
        p = ctx.observe("base_pose")
    else:
        p = localizer.read(ctx, "base_pose", lambda: ctx.observe("base_pose"),
                           functools.partial(hold, ctx, localizer.RETRY_S, 0.0, 0.0, 0.0))
    return (float(p[0]), float(p[1]), float(p[2]))


def hold(ctx, seconds: float, vx: float, vy: float, wz: float, *, select=None,
         on_applied=None) -> int:
    """Stream a body-frame twist; an optional selector checks each unsent tick's twist."""
    ticks = max(1, round(seconds * TICK_HZ))
    # bounded by the clock as well as the tick count
    t0 = float(ctx.now())
    sent = 0
    previous = None
    feedback = False
    while sent < ticks and (sent == 0 or float(ctx.now()) - t0 < seconds - 1e-9):
        if select is None:
            began = float(ctx.now())
            applied = step_base(ctx, LowerBody(vx=vx, vy=vy, wz=wz, state="walk"))
            if applied is not None or feedback:
                if not feedback and sent:
                    ctx.gate.commanded(vx, vy, wz, max(sent / TICK_HZ, began - t0), began)
                actual = (vx, vy, wz) if applied is None else (applied.vx, applied.vy, applied.wz)
                now = float(ctx.now())
                ctx.gate.commanded(*actual, max(1. / TICK_HZ, now - began), now)
                feedback = True
        else:
            selecting = float(ctx.now())
            try:
                actual = select(vx, vy, wz)
            finally:
                elapsed = float(ctx.now()) - selecting
                if previous is not None and elapsed > 0.:
                    ctx.gate.commanded(*previous, elapsed, float(ctx.now()))
            began = float(ctx.now())
            applied = step_base(ctx, LowerBody(vx=actual[0], vy=actual[1], wz=actual[2], state="walk"))
            if applied is not None:
                actual = (applied.vx, applied.vy, applied.wz)
            now = float(ctx.now())
            ctx.gate.commanded(*actual, max(1. / TICK_HZ, now - began), now)
            previous = actual
        if on_applied is not None:
            on_applied(applied)
        sent += 1
    now = float(ctx.now())
    if select is None and not feedback:
        ctx.gate.commanded(vx, vy, wz, max(sent / TICK_HZ, now - t0), now)
    return sent


def step_base(ctx, command) -> LowerBody | None:
    """Send one lower_body command; a driver refusal raises DriveFailure CONTROLLER_REFUSED."""
    try:
        return ctx.step("lower_body", command)
    except Refused as refused:
        raise DriveFailure("CONTROLLER_REFUSED", f"base command refused: {refused}") from refused


def motion_words(vel, seconds: float) -> str:
    """The burst's commanded motion in words."""
    vx, vy, wz = (float(v) for v in vel)
    parts = []
    if wz:
        side = 'left' if wz > 0 else 'right'
        parts.append(f"turn {side} {math.degrees(abs(wz) * seconds):.0f} deg")
    if vx > 0:
        parts.append(f"walk forward {100.0 * vx * seconds:.0f} cm")
    elif vx < 0:
        parts.append(f"walk back {100.0 * abs(vx) * seconds:.0f} cm")
    if vy:
        parts.append(f"sidestep {'left' if vy > 0 else 'right'} {100.0 * abs(vy) * seconds:.0f} cm")
    return " + ".join(parts) if parts else f"stand still {seconds:.1f} s"


def announce(ctx, motion: str, why: str, tag: str, seconds: float, vel=None) -> None:
    """Log `base.action`: the motion, its reason, the twist and its duration."""
    vx, vy, wz = (0.0, 0.0, 0.0) if vel is None else (float(v) for v in vel)
    ctx.log("base.action", {"motion": motion, "why": why, "tag": tag,
                            "seconds": round(float(seconds), 3),
                            "vx": round(vx, 3), "vy": round(vy, 3), "wz": round(wz, 3)})


def burst(ctx, log_kind: str, tag: str, seconds: float, vel, *,
          settle_s: float = SETTLE_S, why: str = "",
          announced: bool = False, on_applied=None) -> tuple[int, float, float]:
    """One open-loop walk or turn plus its settle, logged under `log_kind`; a turn over the burst
    cap is refused. Returns (ticks, moved metres, turned radians)."""
    def refuse_overlong_turn(duration_s: float, yaw_rate: float) -> None:
        if abs(yaw_rate) > 0.0 and duration_s > turn_burst_cap_s(yaw_rate) + 1e-12:
            raise DriveFailure(
                "CONTROLLER_REFUSED",
                f"turn burst {duration_s:.3f} s at {abs(yaw_rate):.2f} rad/s exceeds the shared "
                f"{turn_burst_cap_s(yaw_rate):.3f} s cap; split it into bounded, re-sighted turns",
            )

    vx, vy, wz, seconds = (float(vel[0]), float(vel[1]), float(vel[2]), float(seconds))
    refuse_overlong_turn(seconds, wz)
    motion = motion_words((vx, vy, wz), seconds)
    if not announced:
        announce(ctx, motion, why, tag, seconds, (vx, vy, wz))
    x0, y0, a0 = pose(ctx)
    ticks = hold(ctx, seconds, vx, vy, wz, on_applied=on_applied)
    if settle_s > 0.0:
        ticks += hold(ctx, settle_s, 0.0, 0.0, 0.0)
    x1, y1, a1 = pose(ctx)
    # a walk with a small heading correction is still a walk
    turning = abs(wz) > 0.0 and math.hypot(vx, vy) == 0.0
    moved = math.hypot(x1 - x0, y1 - y0)
    turned = wrap(a1 - a0)
    ctx.log(log_kind, {
        "tag": tag, "kind": "TURN" if turning else "WALK",
        "commanded": round((wz if turning else math.hypot(vx, vy)) * seconds, 4),
        "seconds": round(float(seconds), 3), "settle_s": settle_s, "ticks": ticks,
        "moved_m": round(moved, 4), "turned_rad": round(turned, 4),
        "pose": [round(x1, 4), round(y1, 4), round(a1, 4)],
        "motion": motion, "why": why,
    })
    return ticks, moved, turned


def _turn_servo_velocity(previous: float, want: float, cruise: float) -> float:
    """One slew-limited step of an angular velocity toward what the error asks for."""
    want = float(max(-cruise, min(cruise, want)))
    step = FINAL_TURN_SLEW_RAD_S2 * FINAL_TURN_SLICE_S
    return float(max(previous - step, min(previous + step, want)))


def _above_wz_deadband(wz: float) -> float:
    """Raise a nonzero |wz| below WZ_MIN_ACCEPT to it; a near-zero wz becomes zero."""
    if abs(wz) < 1e-9:
        return 0.0
    if abs(wz) < WZ_MIN_ACCEPT:
        return math.copysign(WZ_MIN_ACCEPT, wz)
    return wz


STRIDE_SAMPLES = 4   # yaw readings averaged over one stride


class StrideYaw:
    """The heading over a stride: the circular mean of the last STRIDE_SAMPLES yaw readings."""

    def __init__(self) -> None:
        self._yaws: list[float] = []

    def read(self, yaw: float) -> float:
        self._yaws = (self._yaws + [float(yaw)])[-STRIDE_SAMPLES:]
        return math.atan2(sum(math.sin(v) for v in self._yaws),
                          sum(math.cos(v) for v in self._yaws))


def final_turns(ctx, log_kind: str, aim, gate_rad: float = TURN_GATE_RAD,
                turn_safe=None, after_turn=None,
                max_translation_m: float | None = None, why: str = "",
                slice_s: float = FINAL_TURN_SLICE_S) -> tuple[int, int]:
    """The final turn: a closed-loop, slew-limited yaw servo toward `aim(x, y)` from the first
    pose, in slices of `slice_s`. Returns (ticks streamed, slices issued)."""
    ticks, turns = 0, 0
    wz = 0.0
    x0, y0, _ = pose(ctx)
    stride = StrideYaw()
    slice_at = float(ctx.now())
    step_s = float(slice_s)
    slice_s = step_s
    cruise = turn_speed()
    gain_hz = current().gait.final_turn_gain_hz
    target_yaw = None
    applied = None
    effective_wz = 0.0

    def applied_turn(command):
        nonlocal applied
        applied = command

    for k in range(FINAL_TURN_MAX_SLICES):
        if ctx.deadline.expired():
            break
        x, y, yaw = pose(ctx)
        now = float(ctx.now())
        if turns:                       # the measured slice length
            slice_s = min(max(step_s, now - slice_at), step_s + 0.2)
        slice_at = now
        # steering uses the lag-corrected stride mean, the gate the raw heading
        yaw_now = yaw
        yaw = wrap(stride.read(yaw) + effective_wz * (STRIDE_SAMPLES - 1) / 2.0 * slice_s)
        drift = math.hypot(x - x0, y - y0)
        if max_translation_m is not None and drift > max_translation_m:
            ctx.log(f"{log_kind}.final_turn_capped", {
                "pose": [round(x, 4), round(y, 4), round(yaw, 4)],
                "translated_m": round(drift, 4), "limit_m": max_translation_m,
                "slices": turns,
            })
            break
        if target_yaw is None:
            target_yaw = float(aim(x, y))
        turn = wrap(target_yaw - yaw)
        if abs(wrap(target_yaw - yaw_now)) <= gate_rad:
            if wz:                      # ramp down and stand
                ticks += hold(ctx, step_s, 0.0, 0.0, _above_wz_deadband(wz / 2.0))
                ticks += hold(ctx, FINAL_TURN_SETTLE_S, 0.0, 0.0, 0.0)
                xs, ys, yaws = pose(ctx)   # the settled pose
                ctx.log(f"{log_kind}.final_turn_settled", {
                    "pose": [round(xs, 4), round(ys, 4), round(yaws, 4)],
                    "left_rad": round(wrap(target_yaw - yaws), 4),
                    "drift_m": round(math.hypot(xs - x0, ys - y0), 4), "slices": turns})
            break
        if turn_safe is not None and not turn_safe(x, y):
            ctx.log(f"{log_kind}.final_turn_skipped", {
                "pose": [round(x, 4), round(y, 4), round(yaw, 4)],
                "remaining_rad": round(turn, 4),
                "reason": "turn translation could leave collision-free space",
            })
            break
        if turns == 0:
            # one announcement for the whole servo
            announce(ctx, f"turn {'left' if turn > 0 else 'right'} {math.degrees(abs(turn)):.0f} "
                     "deg (closed loop)", why, f"{log_kind}.final_turn",
                     abs(turn) / cruise, (0.0, 0.0, math.copysign(cruise, turn)))
        wanted = gain_hz * turn
        wanted = math.copysign(max(abs(wanted), cruise), turn)
        wz = _turn_servo_velocity(wz, wanted, cruise)
        applied = None
        t, _, _ = burst(ctx, log_kind, f"final.turn{k}", step_s,
                        (0.0, 0.0, _above_wz_deadband(wz)), settle_s=0.0, why=why,
                        announced=True, on_applied=applied_turn)
        # Predict stride lag from the command sent; preserve the requested slew/ramp-down.
        effective_wz = wz if applied is None else float(applied.wz)
        ticks += t
        turns += 1
        if after_turn is not None:
            after_turn(k)
    return ticks, turns


DEADBAND_MARGIN = 1.02  # scaled commands land this factor above the dead-band
STRIDE_M = 0.10  # one stride; shorter walks use the micro-pulse


def planar_above_deadband(vx: float, vy: float) -> tuple[float, float]:
    """Scale a planar command whose larger component is under VX_DEADBAND up along its direction,
    or to a full stop."""
    largest = max(abs(vx), abs(vy))
    if largest <= 1e-9:
        return 0.0, 0.0
    if largest < VX_DEADBAND * DEADBAND_MARGIN:
        scale = VX_DEADBAND * DEADBAND_MARGIN / largest
        return vx * scale, vy * scale
    return vx, vy


def finite(value, name: str) -> float:
    """`value` as a finite float, else DriveFailure CONTROLLER_REFUSED."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise DriveFailure("CONTROLLER_REFUSED", f"{name} must be a number: {exc}") from exc
    if not math.isfinite(out):
        raise DriveFailure("CONTROLLER_REFUSED", f"{name} must be finite, got {out}")
    return out


def require_xy(value, name: str) -> tuple[float, float]:
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise DriveFailure("CONTROLLER_REFUSED", f"{name} must be an (x, y) pair: {exc}") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise DriveFailure("CONTROLLER_REFUSED", f"{name} must be finite, got ({x}, {y})")
    return (x, y)

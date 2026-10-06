"""The record of what a hand holds, shared by pick, navigate and place: registering,
lifting, verifying, compacting and monitoring a carry from hand and visual evidence."""
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from homebody.helpers.collision import Clearance
from homebody.helpers.convex import convex_shape
from homebody.helpers.geometry import inverse, points_in, transform
from homebody.helpers.grasp import visible_shape
from homebody.helpers.trajectory import joint_rows
from homebody.motion.arm import follow_rows, live_check, servo_best_effort, servo_unchecked
from homebody.primitives.observations import FreshEvidence
from homebody.skills.contract import Result, SkillFailure

RECOVERY_FAILURES = ("CARRY_CLEARED", "HOLD_UNCERTAIN", "RETRY_REQUIRED")


class Evidence(StrEnum):
    """What one observation says about the carry."""
    HELD = "held"
    ENCLOSED = "enclosed"
    LOST = "lost"
    UNKNOWN = "unknown"


class CarryState(StrEnum):
    """The carry record's ownership between verifications (`record["state"]`)."""
    PENDING = "pending"
    VERIFIED = "verified"
    UNCERTAIN = "uncertain"


class Recovery(StrEnum):
    """What recover() concluded from repeated fresh evidence."""
    CLEARED = "cleared"
    VERIFIED = "verified"
    ENCLOSED = "enclosed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class RecoveryPlan:
    epoch: int
    side: str
    label: int
    resume_lift: bool = False


@dataclass(frozen=True)
class CompactPlan:
    epoch: int
    side: str
    frame_id: tuple
    palm: np.ndarray
    rows: tuple


def begin_pending(ctx, frame, side, label, lift_target, *, points=None):
    """Register a pending carry of LABEL on SIDE, before the hand closes."""
    points = frame.target_points(label) if points is None else points
    record = geometry(frame, ctx.arms[side], side, points)
    shape = visible_shape(points, ctx.settings.grasp.shape_quantiles)
    palm = frame.palm(ctx.arms[side], side)
    record.update(label=label, epoch=frame.epoch, state=CarryState.PENDING, lift_verified=False,
                  baseline_center=shape.center.copy(),
                  baseline_upper=float(np.max(points[:, 2])),
                  baseline_palm_height=float(palm[2, 3]),
                  lift_target=np.array(lift_target, copy=True))
    ctx.carried[side] = record


def lift_pending(ctx, epoch, side):
    return servo_unchecked(ctx, epoch, side, ctx.carried[side]["lift_target"])


def recover(ctx, epoch, side):
    """Verify or clear the carry from repeated fresh observations, without moving the hand.
    Returns (Recovery, last frame). STALE_EPOCH if the record is from an older scene."""
    record = ctx.carried[side]
    if record["epoch"] != epoch:
        raise SkillFailure("STALE_EPOCH", "Grasp evidence belongs to an older scene")
    record["state"] = CarryState.UNCERTAIN
    ctx.check(epoch)
    ctx.actions.base_velocity(epoch, 0.0, 0.0, 0.0)
    settings, arm = ctx.settings.grasp, ctx.arms[side]
    samples = FreshEvidence(epoch)
    seat_anchor = None
    for index in range(settings.recovery_samples):
        if index:
            ctx.advance(epoch, settings.recovery_tick)
        frame = ctx.observe(epoch)
        current = evidence(frame, arm, side, record, settings)
        seating = current == Evidence.HELD and not record["lift_verified"]
        if seating:
            local = points_in(inverse(frame.palm(arm, side)), frame.target_points(record["label"]))
            if (seat_anchor is not None and
                    np.quantile(cKDTree(seat_anchor).query(local)[0], settings.shape_quantiles[1]) >
                    settings.target_surface_tolerance):
                samples = FreshEvidence(epoch)
            seat_anchor = local
        else:
            seat_anchor = None
        count = samples.update(frame, current if current != Evidence.UNKNOWN else None)
        if count is None or count < settings.recovery_confirmations:
            continue
        if current == Evidence.LOST:
            ctx.carried.pop(side)
            return Recovery.CLEARED, frame
        if current == Evidence.HELD:
            if seating:
                recapture(record, frame, arm, side, local)
            record.update(state=CarryState.VERIFIED, lift_verified=True)
            return Recovery.VERIFIED, frame
        record["state"] = CarryState.PENDING
        return Recovery.ENCLOSED, frame
    return Recovery.UNCERTAIN, frame


def recovery_result(skill, status):
    if status == Recovery.CLEARED:
        return Result(skill, "CARRY_CLEARED", "Repeated fresh evidence confirms the old target is separated and the hand has no retention; submit a fresh action")
    if status == Recovery.VERIFIED:
        return Result(skill, "RETRY_REQUIRED", "Carry verified from fresh evidence; submit the requested action again on the current frame")
    return Result(skill, "HOLD_UNCERTAIN", "Carry remains uncertain; no hand opening or release occurred. Repeat pick on the original target to recover")


def compact_carry(ctx, epoch, side):
    """Pull a verified load in toward the body and verify it again, returning the frame
    after. HOLD_UNCERTAIN if the carry is not verified before or after."""
    frame = ctx.observe(epoch)
    record = ctx.carried[side]
    held = continuing(frame, ctx.arms[side], side, record, ctx.settings.grasp)
    record['state'] = CarryState.UNCERTAIN
    if not held:
        raise SkillFailure("HOLD_UNCERTAIN", "Compact carry requires a currently verified grasp")
    plan = prepare_compact(frame, side, ctx.arms, record, ctx.settings)
    if plan is None:
        record['state'] = CarryState.VERIFIED
        return frame
    ctx.emit({'type': 'carry_prepared', 'phase': 'compact', 'plan': plan})
    ctx.stage("Compacting carry")
    ctx.check(epoch)
    follow_rows(ctx, epoch, side, plan.rows, live_check(ctx, side, (record['label'],)))
    servo_best_effort(ctx, epoch, side, plan.palm, exclude_labels=(record['label'],))
    ctx.stage("Verifying compact carry")
    status, frame = recover(ctx, epoch, side)
    if status != Recovery.VERIFIED:
        raise SkillFailure("HOLD_UNCERTAIN", "Compact carry did not retain repeated current hold evidence")
    return frame


def continuing(state, arm, side, record, settings, look=None):
    """Whether a verified carry still reads as held: from the hand's sensors, else from a
    camera frame (STATE, or the one LOOK returns)."""
    return record["state"] == CarryState.VERIFIED and (
        state.hands[side].retained(settings) or
        evidence(state if look is None else look(), arm, side, record, settings) == Evidence.HELD)


def evidence(frame, arm, side, record, settings):
    """What one observation says about the carry."""
    hand = frame.hands[side]
    retained = hand.retained(settings)
    efforts = hand.finger_effort_fraction > settings.hold_effort_fraction
    opposed = bool(efforts[0] and np.any(efforts[1:]))
    closed_empty = (hand.closure >= 1. - settings.hold_closure_gap and
                    bool(np.all(hand.finger_closure_gap <= settings.hold_closure_gap)))
    if not retained and not opposed and closed_empty:
        return Evidence.LOST
    if not frame.visible(record["label"]):
        return Evidence.UNKNOWN
    points = frame.target_points(record["label"])
    palm = frame.palm(arm, side)
    local = points_in(inverse(palm), points)
    in_payload = points_in(inverse(record["palm_to_payload"]), local)
    outside = np.maximum(np.abs(in_payload) - record["payload_half"], 0.0)
    separated = np.min(np.linalg.norm(outside, axis=1)) > settings.recovery_separation
    if not retained and not opposed and separated:
        return Evidence.LOST
    shape = visible_shape(points, settings.shape_quantiles)
    if record.get("lift_verified", False):
        strayed = np.linalg.norm(outside, axis=1)
        near = np.quantile(strayed, settings.shape_quantiles[1]) <= settings.recovery_match
    else:
        residual = cKDTree(record["local_points"]).query(local)[0]
        near = np.quantile(residual, settings.shape_quantiles[1]) <= settings.target_shift
    if near and opposed:
        lifted = record.get("lift_verified", False) or (
            shape.upper[2] - record["baseline_upper"] >= settings.lift_evidence and
            palm[2, 3] - record["baseline_palm_height"] >= settings.lift_evidence)
        if lifted:
            return Evidence.HELD
        if retained:
            return Evidence.ENCLOSED
    return Evidence.UNKNOWN


def recapture(record, frame, arm, side, local):
    """Widen the record's capture to the grasp-time view plus LOCAL (palm-frame points),
    which later checks are anchored to."""
    union = np.vstack((record["local_points"], local))
    record.update(geometry(frame, arm, side, points_in(frame.palm(arm, side), union)))
    record["local_points"] = local


def geometry(frame, arm, side, points):
    """The capture of POINTS in the palm frame: a footprint-aligned box, its hull, the points."""
    palm = frame.palm(arm, side)
    local = points_in(inverse(palm), points)
    center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2
    centered = points - center
    rotation = _capture_yaw(centered)
    aligned = centered @ rotation
    lower, upper = np.min(aligned, axis=0), np.max(aligned, axis=0)
    pose = transform(center + rotation @ ((lower + upper) / 2), rotation)
    return {"palm_to_payload": inverse(palm) @ pose,
            "payload_half": (upper - lower) / 2, "local_points": local,
            "payload_hull": convex_shape(aligned - (lower + upper) / 2)}


def _capture_yaw(points):
    """Rotation aligning the minimum-area XY rectangle of POINTS, identity when the hull fails."""
    xy = np.unique(points[:, :2], axis=0)
    if len(xy) < 2:
        return np.eye(3)
    singular = np.linalg.svd(xy - xy.mean(axis=0), compute_uv=False)
    if len(xy) < 3 or singular[1] <= np.finfo(float).eps * max(xy.shape) * singular[0]:
        directions = xy - xy[0]
        edges = directions[[np.argmax(np.linalg.norm(directions, axis=1))]]
        vertices = xy
    else:
        try:
            vertices = xy[ConvexHull(xy).vertices]
        except QhullError:
            return np.eye(3)
        edges = np.roll(vertices, -1, axis=0) - vertices
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2))
    rotations = [np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for a in angles]
    areas = np.array([np.prod(np.ptp(vertices @ rotation, axis=0)) for rotation in rotations])
    rounding = 64 * np.finfo(float).eps * np.max(np.ptp(xy, axis=0)) ** 2
    index = np.flatnonzero(areas <= areas.min() + rounding)[0]
    rotation = np.eye(3)
    rotation[:2, :2] = rotations[index]
    return rotation


def prepare_compact(frame, side, arms, record, settings):
    """The nearest clear radial retreat of the palm, level first, else raised by the lift
    height. None when none is needed or none is clear."""
    arm = arms[side]
    seed = np.clip(frame.arm_joints(side), arm.lower, arm.upper)
    palm = arm.forward(seed)[0]
    radius = np.linalg.norm(palm[:2, 3])
    target_radius = settings.grasp.carry_palm_radius
    record.pop('compact_pending', None)
    if radius <= target_radius + settings.grasp.carry_compact_min_delta:
        return None
    world_height = (frame.world_torso @ palm)[2, 3]
    clearance = Clearance(frame, arms, side, settings.collision, (record['label'],), record, exempt_start=True)
    radii = np.arange(target_radius, radius - settings.grasp.carry_compact_min_delta + 1e-9,
                      settings.grasp.carry_compact_step)
    for rise in (0., settings.grasp.lift_height):
        alternatives = []
        for proposed_radius in radii:
            proposed = palm.copy()
            proposed[:2, 3] *= proposed_radius / radius
            world_target = frame.world_torso @ proposed
            world_target[2, 3] = world_height + rise
            target = arm.solve(inverse(frame.world_torso) @ world_target, seed)
            if (target is None or clearance.path_violation(seed, target) is not None
                    or not in_view(frame, payload_points(record, world_target))):
                continue
            rows = joint_rows(seed, target, settings.physics.arm_rate, settings.servo.tick)
            plan = CompactPlan(frame.epoch, side, frame.frame_id, world_target, tuple(rows))
            if proposed_radius == target_radius:
                return plan
            alternatives.append((np.max(np.abs(target - seed)), proposed_radius, plan))
        if alternatives:
            return min(alternatives, key=lambda row: (row[0], row[1]))[2]
    record['compact_pending'] = True
    return None


def payload_points(record, palm):
    """World points of the carried object's last observed extent with the palm at PALM."""
    hull = record.get('payload_hull')
    if hull is None:
        return points_in(palm, record['local_points'])
    return points_in(palm @ record['palm_to_payload'], hull['vertices'])


def in_view(frame, points):
    """Whether every point lies in front of the camera and projects inside its image."""
    camera = points_in(inverse(frame.world_camera), points)
    projected = camera @ frame.intrinsics.T
    uv = projected[:, :2] / projected[:, 2, None]
    height, width = frame.depth.shape
    return bool(np.all(camera[:, 2] > 0.) and np.isfinite(uv).all() and
                np.all(uv >= 0.) and np.all(uv < [width, height]))

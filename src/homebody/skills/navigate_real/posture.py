"""Checked empty-arm folds for a blocked departure; an unblocked start never folds.

The map check uses transit's full-height occupancy extrusion. Current depth also
vetoes observed collisions; it is not used to invent free space behind an occlusion.
"""
import time
from dataclasses import dataclass, replace

import numpy as np

from homebody.helpers.collision import (
    BEARING_LINKS,
    HAND_END_LINKS,
    Clearance,
    near_pairs,
    overlap,
)
from homebody.helpers.map_sweep import projected_free
from homebody.motion.arm import ARM_LINKS, REST_SETTLED_RAD, TORSO_BOX
from homebody.motion.carry import robot_boxes, sweep_radius
from homebody.primitives.observations import ARM, FreshEvidence, RobotState, frozen_array
from homebody.skills.contract import SkillFailure


@dataclass(frozen=True)
class FoldPlan:
    epoch: int
    binding: tuple
    targets: tuple
    radius_before: float


@dataclass(frozen=True)
class _PostureState(RobotState):
    """Hypothetical joints for geometric checks, never a captured sensor Frame."""
    epoch: int
    sequence: int
    time: float
    world_torso: np.ndarray
    base_pose: np.ndarray
    joints: np.ndarray
    hands: object


def _binding(evidence):
    return evidence.frame_id, evidence.map_hash, evidence.generation


def _empty(frame, planning):
    return (not planning.carried and set(planning.arms) == {"left", "right"} and
            set(frame.hands) == {"left", "right"} and
            all(np.isfinite(hand.closure) and
                0 <= hand.closure <= planning.settings.placement.open_closure_limit and
                not hand.retained(planning.settings.grasp) for hand in frame.hands.values()))


def _geometry_available(frame, evidence):
    """This posture operation requires current metric RGB-D and a registered map."""
    nav = evidence.navigation
    return (nav is not None and evidence.registered and evidence.mobile and
            evidence.epoch == frame.epoch and bool(evidence.frame_id and evidence.map_hash) and
            np.isfinite(nav.T_map_px).all() and
            np.allclose(nav.T_map_px[2], [0., 0., 1.]) and
            abs(np.linalg.det(nav.T_map_px)) > 1e-12 and
            np.isfinite(frame.intrinsics).all() and
            frame.intrinsics[0, 0] > 0 and frame.intrinsics[1, 1] > 0 and
            np.isfinite(frame.world_camera).all() and np.isfinite(frame.world_torso).all() and
            np.any(np.isfinite(frame.depth) & (frame.depth > 0) & (frame.labels >= 0)))


def _posed(frame, side, q):
    joints = frame.joints.copy()
    joints[ARM[side]] = q
    return _PostureState(frame.epoch, frame.sequence, frame.time, frame.world_torso,
                         frame.base_pose, frozen_array(joints), frame.hands)


def _radius(frame, planning):
    return sweep_radius(frame, planning.arms, {}, planning.settings.motion.robot_radius)


def _reach_bound(arm, margin):
    # Triangle inequality: every downstream joint-to-corner distance is below this
    # sum, even though it includes unused finger branches and upstream translations.
    return (sum(np.linalg.norm(origin[:3, 3]) for origin in arm.origins) +
            sum(np.linalg.norm(origin[:3, 3]) for origin in arm.collision.finger_origins.values()) +
            max(np.linalg.norm(center) + np.linalg.norm(half + margin)
                for center, half in arm.collision.bounds.values()))


def _body_clear(arm, joints, fingers, other, gap, *, boxes=None, hands=None):
    """Whether one sampled pose clears the torso and OTHER by GAP, and itself by 2 GAP outside the
    bearing and hand-end exclusions. GAP bounds each point's travel between samples."""
    boxes = arm.collision.boxes(arm, joints) if boxes is None else boxes
    hands = arm.collision.hand_boxes(arm, joints, fingers) if hands is None else hands
    names = [name for name in arm.collision.links if name in arm.collision.bounds]
    torso_parts = [box for name, box in zip(names, boxes) if any(part in name for part in ARM_LINKS)]
    if any(overlap(box, TORSO_BOX, gap) for box in [*torso_parts, *hands]):
        return False
    if any(overlap(box, fixed, gap) for box in [*boxes, *hands] for fixed in other):
        return False
    if any(overlap(box, boxes[j], 2 * gap) for i, box in enumerate(boxes)
           for j in range(max(0, i - BEARING_LINKS))):
        return False
    return not any(overlap(hand, proximal, 2 * gap) for hand in hands
                   for proximal in boxes[:-HAND_END_LINKS])


def _arm_sweep(frame, planning, evidence, side, target, check, *, state=None):
    """Checked boxes along measured-to-reference interpolation, padded per interval.

    This follows the bounded reference interpolation used by arm execution. It does
    not claim a bound on arbitrary physical overshoot beyond the collision margins.
    """
    state = frame if state is None else state
    arm, collision = planning.arms[side], planning.settings.collision
    seed = state.arm_joints(side)
    if (not np.isfinite(target).all() or np.any(target < arm.lower) or
            np.any(target > arm.upper)):
        return None
    reach = _reach_bound(arm, collision.column_margin)
    travel_bound = reach * np.sum(np.abs(target - seed))
    count = max(1, int(np.ceil(max(np.max(np.abs(target - seed)) / collision.joint_step,
                                   travel_bound / .002))))
    gap = travel_bound / count
    clearance = Clearance(frame, planning.arms, side,
                          replace(collision, cloud_margin=collision.cloud_margin + gap),
                          exempt_start=False, other_arm_boxes=())
    # _path_clear checks the complete other-arm sweep, including asynchronous
    # progress. Do not mix its hypothetical joint state with captured RGB-D.
    swept = []
    fractions = np.linspace(0., 1., count + 1) if travel_bound else (0.,)
    for fraction in fractions:
        check()
        q = seed + fraction * (target - seed)
        boxes = arm.collision.boxes(arm, q)
        hands = arm.collision.hand_boxes(arm, q, frame.hands[side].joints)
        if (clearance.violation(q) or not _body_clear(
                arm, q, frame.hands[side].joints, [], gap, boxes=boxes, hands=hands)):
            return None
        world_boxes = robot_boxes(_posed(state, side, q), {side: arm}, {})
        if not projected_free(evidence.navigation, world_boxes, collision.column_margin, gap).all():
            return None
        swept.extend((center, rotation, half + gap) for center, rotation, half in (*boxes, *hands))
    return swept


def _path_clear(frame, planning, evidence, side, target, check=lambda: None, held_targets=None,
                *, state=None):
    """Check both complete arm paths, including every asynchronous relative timing."""
    state = frame if state is None else state
    moving = _arm_sweep(frame, planning, evidence, side, target, check, state=state)
    if moving is None:
        return False
    other_side = "left" if side == "right" else "right"
    other_q = state.arm_joints(other_side)
    held = other_q if held_targets is None else held_targets.get(other_side, other_q)
    other = _arm_sweep(frame, planning, evidence, other_side, held, check, state=state)
    if other is None:
        return False
    # Both unions include their own interval bounds. Pair all poses, rather than
    # assuming the two controllers traverse equal interpolation fractions together.
    for box in moving:
        check()
        near = near_pairs([box], other)[0]
        if any(overlap(box, fixed) for fixed, close in zip(other, near) if close):
            return False
    return True


def prepare(frame, planning, evidence, *, admitted, check=lambda: None):
    """Checked natural rest-width pose with departure headroom, or None; no effects."""
    check()
    if (not _empty(frame, planning) or not _geometry_available(frame, evidence) or admitted(frame)):
        return None
    rest = {side: planning.settings.placement.rest_arm(side) for side in planning.arms}
    if any(np.max(np.abs(frame.arm_joints(side) - q)) > planning.settings.placement.rest_tolerance
           for side, q in rest.items()):
        return None

    geometry_end = time.monotonic() + planning.settings.motion.arm_timeout

    def guard():
        check()
        if planning.cancelled():
            raise SkillFailure("CANCELLED", "Cancelled while checking an empty-arm fold")
        if time.monotonic() >= geometry_end:
            raise SkillFailure("TIMEOUT", "Empty-arm geometry search exceeded its wall-clock budget")

    # Keep pitch/elbow at the commanded walking shape. Within that natural class,
    # prefer radius headroom so settling does not undo a barely admitted departure.
    candidates = []
    for roll in (1., .75, .5):
        guard()
        proposed = frame
        for side, home in rest.items():
            q = frame.arm_joints(side).copy()
            q[0], q[1], q[3] = home[0], home[1] * roll, home[3]
            proposed = _posed(proposed, side, q)
        guard()
        if any(np.any(proposed.arm_joints(side) < arm.lower) or
               np.any(proposed.arm_joints(side) > arm.upper) for side, arm in planning.arms.items()):
            continue
        if not admitted(proposed):
            continue
        candidates.append((_radius(proposed, planning), proposed))
    for _, proposed in sorted(candidates, key=lambda candidate: candidate[0]):
        guard()
        order = sorted(planning.arms, key=lambda side: _radius(
            _posed(frame, side, proposed.arm_joints(side)), planning))
        current, targets = frame, []
        for side in order:
            q = proposed.arm_joints(side)
            if not _path_clear(frame, planning, evidence, side, q, guard, state=current):
                break
            current = _posed(current, side, q)
            targets.append((side, tuple(q)))
        else:
            return FoldPlan(frame.epoch, _binding(evidence), tuple(targets), _radius(frame, planning))
    return None


def execute(ctx, plan, *, validate_evidence, check=lambda: None):
    """Checked arm steps while held, returning measured settled joints for fresh planning.

    VALIDATE_EVIDENCE returns current validated NavigationEvidence. CHECK enforces the
    caller's navigation deadline; it is also polled in geometric checks.
    """
    from .task import require_navigation_freshness

    started = None

    def before_reference(frame, evidence):
        check()
        ctx.check(plan.epoch)
        if not 0 <= ctx.observations.time - frame.time <= ctx.settings.motion.observation_age:
            raise SkillFailure("STALE_OBSERVATION", "Empty-arm geometry expired before its command")
        require_navigation_freshness(evidence, ctx.observations.time)
        if started is not None and ctx.observations.time - started >= ctx.settings.motion.arm_timeout:
            raise SkillFailure("TIMEOUT", "Empty-arm fold did not settle before its bounded deadline")

    def current():
        check()
        ctx.check(plan.epoch)
        evidence = validate_evidence()
        if evidence.epoch != plan.epoch or _binding(evidence) != plan.binding:
            raise SkillFailure("STALE_EPOCH", "Navigation binding changed during empty-arm fold")
        frame = ctx.observe(plan.epoch)
        if not _geometry_available(frame, evidence):
            raise SkillFailure("STALE_OBSERVATION", "Empty-arm fold lost metric depth or registered map evidence")
        if not _empty(frame, ctx.planning()):
            raise SkillFailure("HOLD_UNCERTAIN", "Empty-arm fold requires both hands to remain empty and open")
        return frame, evidence

    frame, evidence = current()
    # Recheck the complete sequence against current base, joints and obstacles before
    # the first arm command; preparation never authorizes a stale interpolation.
    preview = frame
    for side, target in plan.targets:
        target = np.asarray(target)
        if not _path_clear(frame, ctx.planning(), evidence, side, target, check, state=preview):
            raise SkillFailure("COLLISION", "Current observations refuse the empty-arm fold")
        preview = _posed(preview, side, target)
    ctx.actions.stop(plan.epoch)
    # Preview can consume wall time while balance and the last commanded references continue.
    # Choose the initial hold from fresh measurements after the base hold succeeds.
    frame, evidence = current()
    held_targets = {side: frame.arm_joints(side).copy() for side in ctx.arms}
    for side, target in held_targets.items():
        if not _path_clear(frame, ctx.planning(), evidence, side, target, check):
            raise SkillFailure("COLLISION", "Fresh geometry refuses the initial measured arm hold")
    for side, target in held_targets.items():
        before_reference(frame, evidence)
        ctx.actions.arm_target(plan.epoch, side, target)
    ctx.stage("Folding empty arms for departure")
    ctx.emit({"type": "navigation_posture_started", "radius_before_m": _radius(frame, ctx.planning())})
    started = ctx.observations.time
    for side, target in plan.targets:
        target = np.asarray(target)
        stable = FreshEvidence(plan.epoch)
        while True:
            frame, evidence = current()
            if ctx.observations.time - started >= ctx.settings.motion.arm_timeout:
                raise SkillFailure("TIMEOUT", "Empty-arm fold did not settle before its bounded deadline")
            error = target - frame.arm_joints(side)
            count = stable.update(frame, np.max(np.abs(error)) <= REST_SETTLED_RAD)
            if count is not None and count >= ctx.settings.servo.stable_frames:
                break
            step = np.clip(error, -ctx.settings.physics.arm_rate * ctx.settings.servo.tick,
                           ctx.settings.physics.arm_rate * ctx.settings.servo.tick)
            reference = frame.arm_joints(side) + step
            if not _path_clear(frame, ctx.planning(), evidence, side, reference, check, held_targets):
                raise SkillFailure("COLLISION", "Live map or depth refuses the next empty-arm fold step")
            before_reference(frame, evidence)
            ctx.actions.arm_target(plan.epoch, side, reference)
            held_targets[side] = reference.copy()
            ctx.advance(plan.epoch)
        # STOP selects the backend's cached arm reference. Complete it before
        # commanding the fresh measured hold, so it cannot overwrite that target.
        ctx.actions.stop(plan.epoch)
        frame, evidence = current()
        reference = frame.arm_joints(side).copy()
        if not _path_clear(frame, ctx.planning(), evidence, side, reference, check, held_targets):
            raise SkillFailure("COLLISION", "Live geometry refuses the measured arm hold")
        before_reference(frame, evidence)
        ctx.actions.arm_target(plan.epoch, side, reference)
        held_targets[side] = reference
    frame, _ = current()
    ctx.emit({"type": "navigation_posture_settled", "radius_before_m": plan.radius_before,
              "radius_after_m": _radius(frame, ctx.planning()), "arrived": False})
    return frame

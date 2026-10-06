"""The place skill: release a held object at an observed support or a captured torso XY and
verify the observed outcome."""
from dataclasses import dataclass, replace
from functools import partial
from itertools import product

import numpy as np

from homebody.helpers.collision import Clearance, under_box
from homebody.helpers.geometry import inverse, points_in
from homebody.helpers.grasp import payload_offset, visible_shape
from homebody.motion.arm import (
    close_or_open,
    move_joints,
    reset_arm,
    servo,
    servo_best_effort,
    servo_unchecked,
)
from homebody.motion.carry import (
    RECOVERY_FAILURES,
    CarryState,
    RecoveryPlan,
    compact_carry,
    continuing,
    recover,
    recovery_result,
)
from homebody.primitives.observations import FreshEvidence
from homebody.vlm.topdown import normalized_click

from . import navigate
from .contract import Result, SkillFailure, click_argument, side_argument

NEEDS = ("base_velocity", "arm_target", "grip", "advance", "stop")

FAILURES = ("NOT_HOLDING", "SURFACE_CHANGED", "ARM_NOT_REACHED",
            "RELEASE_NOT_VERIFIED", "OUTCOME_UNKNOWN", "PLACEMENT_NOT_VERIFIED") + RECOVERY_FAILURES
PROMPT = ('place: hand, the holding hand, and one destination. release_point_1000 [x, y] on image 4, or '
          'x_ahead_m and y_left_m in the torso frame, releases the object\'s centre there at the current '
          'hand height plus delta_z_m (within ±{placement.max_release_height:g} m), and it drops onto what '
          'lies below. point_normalized_1000 [x, y], a pixel on a support surface in image 1, sets the '
          'object down release_height_m above it ({placement.release_height:g} to '
          '{placement.max_release_height:g} m, default {placement.release_height:g}). Either way the object is '
          'centred on the point: on another object it is stacked on top, and near an edge it can tip off. '
          'The palm reaches about '
          '{placement.forward_reach:g} m ahead of the torso, less when raised: a farther point is lowered and '
          'pulled toward the body, and one beyond reach is released as close to it as the arm gets.')


@dataclass(frozen=True)
class Plan:
    """One checked release. `pre_joints` is None beyond reach, `point` is where the landing is
    verified (world), and `released` is the let-go object's (centre, footprint radius, top)."""
    epoch: int
    side: str
    label: int
    point: np.ndarray
    pre_joints: np.ndarray
    pre: np.ndarray
    palm: np.ndarray
    support: bool = True
    released: tuple | None = None
    short_m: float = 0.0  # how far short of the requested point a beyond-reach release was


def prepare(arguments, selected, current, planning):
    """Resolve the destination and plan a checked release, or a RecoveryPlan for an uncertain carry."""
    side = side_argument(arguments)
    if side not in planning.carried and len(planning.carried) == 1:
        side = next(iter(planning.carried))
    surface, point, height = destination(arguments, selected, planning.settings)
    carried = planning.carried.get(side)
    if carried is None:
        raise SkillFailure("NOT_HOLDING", "No carried object in this hand")
    arm, settings = planning.arms[side], planning.settings
    if not continuing(current, arm, side, carried, settings.grasp):
        return RecoveryPlan(current.epoch, side, carried["label"])
    if surface == carried["label"]:
        raise SkillFailure("INVALID_ARGUMENT",
                           "The support click selects the carried object; click the surface to place it on")

    return plan_release(side, surface, point, height, selected, current, planning)


def plan_release(side, surface, point, height, selected, current, planning):
    """The checked release for a destination: a reachable pose, its pre-pose and a clear arm path."""
    carried, arm, settings = planning.carried[side], planning.arms[side], planning.settings
    withdraw = current.palm(arm, side)
    if surface is not None:
        palm, release = support_release(current, surface, point, height, carried, withdraw, settings)
        search = ReleaseSearch.support_click(settings)
    else:
        palm = drop_palm(selected, point, height, carried, withdraw)
        search = ReleaseSearch.metric_drop(settings)

    seed = current.arm_joints(side)
    center_offset = payload_offset(carried, withdraw[:3, :3])[0]
    pre, palm, pre_q, grasp_q = release_pose(arm, current.world_torso, palm, center_offset, seed, search)
    if surface is None:
        release = palm[:3, 3] + palm[:3, :3] @ carried["palm_to_payload"][:3, 3]
    clearance = Clearance(current, planning.arms, side, settings.collision, (carried["label"],), carried,
                          exempt_start=True)
    blocked = None if grasp_q is None else (clearance.path_violation(seed, pre_q) or
                                            clearance.path_violation(pre_q, grasp_q))
    if blocked:
        raise SkillFailure("COLLISION", f"The arm path to the selected release pose is blocked: {blocked}")
    if grasp_q is None:  # beyond reach: let go where the arm gets
        pre = palm.copy()
        pre[2, 3] += search.approach_lift
    return Plan(current.epoch, side, carried["label"], release, pre_q, pre, palm, surface is not None)


def execute(plan, ctx):
    """Approach, lower, open the hand, retract, then verify the landing and return the arm."""
    if isinstance(plan, RecoveryPlan):
        ctx.stage("Checking carry")
        status, _ = recover(ctx, plan.epoch, plan.side)
        return replace(recovery_result("place", status), details={"hand": plan.side})
    epoch, side, held, settings = plan.epoch, plan.side, (plan.label,), ctx.settings

    ctx.stage("Approaching support" if plan.support else "Approaching release point")
    reach = servo if plan.pre_joints is not None else partial(servo_best_effort, position_only=True)
    if plan.pre_joints is not None:
        move_joints(ctx, epoch, side, plan.pre_joints, held)
    reach(ctx, epoch, side, plan.pre, exclude_labels=held)
    if plan.pre[2, 3] > plan.palm[2, 3]:
        ctx.stage("Lowering")
        reach(ctx, epoch, side, plan.palm, exclude_labels=held)
    center_in_palm = attached_center(plan, ctx)
    if plan.pre_joints is None:  # verify the landing under where the arm got
        released = points_in(ctx.observe(epoch).palm(ctx.arms[side], side), center_in_palm)
        plan = replace(plan, point=np.r_[released[:2], plan.point[2]],
                       short_m=float(np.linalg.norm(released[:2] - plan.point[:2])))

    ctx.carried[side]["state"] = CarryState.UNCERTAIN
    ctx.stage("Releasing")
    frame = close_or_open(ctx, epoch, side, 0.0, settings.grasp.open_seconds)
    ctx.stage("Checking release")
    if frame.hands[side].closure > settings.placement.open_closure_limit:
        raise SkillFailure("RELEASE_NOT_VERIFIED", f"The {side} hand's measured fingers did not open")

    ctx.stage("Retracting")
    current = ctx.observe(epoch)
    target = current.palm(ctx.arms[side], side)
    plan = replace(plan, released=released_extent(target, ctx.carried[side]))
    target[2, 3] += max(settings.placement.withdraw_height,
                        object_top(target, ctx.carried[side]) + HAND_HANG_M
                        + settings.placement.withdraw_clearance - target[2, 3])
    back = current.world_torso[:2, 3] - target[:2, 3]
    target[:2, 3] += settings.placement.withdraw_back * back / max(np.linalg.norm(back), 1e-9)
    servo_unchecked(ctx, epoch, side, target)

    return verify_settled(plan, ctx, center_in_palm)


def abort(plan, ctx):
    """Keep a held object tucked; once the hand has opened, return the emptied arm to rest."""
    if isinstance(plan, RecoveryPlan) or ctx.observations.epoch != plan.epoch:
        return
    if plan.side in ctx.carried and ctx.carried[plan.side]["state"] != CarryState.VERIFIED:
        hand = ctx.observe(plan.epoch).hands[plan.side]
        if hand.closure <= ctx.settings.placement.open_closure_limit and not hand.retained(ctx.settings.grasp):
            ctx.carried.pop(plan.side)  # a tuck of an empty hand would refuse
    if plan.side not in ctx.carried:
        return_arm(plan, ctx)
        return
    ctx.stage("Returning arm")
    compact_carry(ctx, plan.epoch, plan.side)


def destination(arguments, selected, settings):
    """(support label, clicked point, release height) for a support click, or (None, torso XY,
    delta z) for a top-down or metric release."""
    ego = "point_normalized_1000" in arguments
    topdown = "release_point_1000" in arguments
    metric = "x_ahead_m" in arguments or "y_left_m" in arguments
    if sum((ego, topdown, metric)) != 1:
        raise SkillFailure("INVALID_ARGUMENT", "Choose exactly one place destination form")
    if ego:
        if "delta_z_m" in arguments:
            raise SkillFailure("INVALID_ARGUMENT", "Support clicks use release_height_m, not delta_z_m")
        height = arguments.get("release_height_m", settings.placement.release_height)
        if (type(height) not in (int, float) or
                not settings.placement.release_height <= height <= settings.placement.max_release_height):
            raise SkillFailure("INVALID_ARGUMENT", "release_height_m is outside the configured release range")
        label, point = click_argument(arguments, selected)
        return label, point, height
    if "release_height_m" in arguments:
        raise SkillFailure("INVALID_ARGUMENT", "Torso XY release uses delta_z_m, not release_height_m")
    delta = arguments.get("delta_z_m", 0.)
    if (type(delta) not in (int, float) or
            not -settings.placement.max_release_height <= delta <= settings.placement.max_release_height):
        raise SkillFailure("INVALID_ARGUMENT", "delta_z_m is outside the configured release range")
    if topdown:
        xy = normalized_click(arguments["release_point_1000"], settings.topdown)
    else:
        values = [arguments.get(key) for key in ("x_ahead_m", "y_left_m")]
        if not all(navigate.finite_number(value) for value in values):
            raise SkillFailure("INVALID_ARGUMENT", "Torso release requires finite x_ahead_m and y_left_m")
        xy = np.array(values)
    return None, xy, delta


def support_release(current, surface, point, height, carried, palm, settings):
    """The palm that rests the payload's bottom HEIGHT (m) above the clicked support, and the
    release point on it (world)."""
    placement, collision = settings.placement, settings.collision
    points = current.target_points(surface)
    closest = points[np.argmin(np.linalg.norm(points - point, axis=1))]
    if np.linalg.norm(closest - point) > placement.support_match:
        raise SkillFailure("SURFACE_CHANGED", "The clicked support no longer has nearby visible depth")
    center_offset, bottom_offset = payload_offset(carried, palm[:3, :3])

    def resting(release):
        placed = palm.copy()
        placed[:3, 3] = release + [0., 0., bottom_offset + height] - center_offset
        return placed

    release = point.copy()
    payload = resting(release) @ carried["palm_to_payload"]
    box = (payload[:3, 3], payload[:3, :3], carried["payload_half"])
    footprint = under_box(points, box, collision.cloud_margin)
    release[2] = np.max(points[footprint, 2]) if np.any(footprint) else closest[2]
    return resting(release), release


def drop_palm(selected, point, height, carried, palm):
    """The palm that puts the object's centre at POINT in the captured torso frame, HEIGHT (m)
    above the palm's current height."""
    center_offset = payload_offset(carried, palm[:3, :3])[0]
    captured_height = points_in(inverse(selected.world_torso), palm[:3, 3])[2]
    placed = palm.copy()
    placed[:3, 3] = points_in(selected.world_torso, np.r_[point, captured_height + height]) - center_offset
    return placed


@dataclass(frozen=True)
class ReleaseSearch:
    """The poses tried for a release, in order: pulls toward the torso, drops, twists."""
    pulls: tuple
    drops: tuple
    twists: tuple
    margin: float
    approach_lift: float

    @classmethod
    def support_click(cls, settings):
        """The exact clicked pose, approached from above."""
        return cls((0.,), (0.,), (0.,), 0., settings.grasp.approach_height)

    @classmethod
    def metric_drop(cls, settings):
        """Lower, then pull in. The pose must also solve the reach margin farther out."""
        placement = settings.placement
        pulls = tuple(np.arange(0., placement.max_pull + 1e-9, placement.pull_step))
        return cls(pulls, placement.release_drops, placement.release_twists,
                   placement.reach_margin_steps * placement.pull_step, 0.)


def release_pose(arm, world_torso, palm, center_offset, seed, search):
    """The first solvable (pre, palm, pre_q, grasp_q) of the search, or (palm, palm, None, None).
    CENTER_OFFSET is palm to object centre."""
    toward = world_torso[:2, 3] - palm[:2, 3]
    toward /= max(np.linalg.norm(toward), 1e-9)
    for pull, drop, twist in product(search.pulls, search.drops, search.twists):
        c, s = np.cos(twist), np.sin(twist)
        turned = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        candidate = palm.copy()
        candidate[:3, :3] = turned @ palm[:3, :3]
        candidate[:3, 3] = palm[:3, 3] + center_offset - turned @ center_offset
        candidate[:2, 3] += pull * toward
        candidate[2, 3] -= drop
        pre, probe = candidate.copy(), candidate.copy()
        pre[2, 3] += search.approach_lift
        probe[:2, 3] -= search.margin * toward
        pre_q = arm.solve(inverse(world_torso) @ pre, seed)
        grasp_q = None if pre_q is None else arm.solve(inverse(world_torso) @ candidate, pre_q)
        if grasp_q is not None and arm.solve(inverse(world_torso) @ probe, grasp_q) is not None:
            return pre, candidate, pre_q, grasp_q
    return palm, palm, None, None




def released_extent(palm, record):
    """The released object's world centre, footprint radius including the open hand, and top."""
    payload = palm @ record["palm_to_payload"]
    extent = np.abs(payload[:3, :3]) @ record["payload_half"]
    return payload[:3, 3], float(np.linalg.norm(extent[:2]) + HAND_WIDTH_M), float(payload[2, 3] + extent[2])


def object_top(palm, record):
    """World height of the held object's top with the palm at PALM."""
    payload = palm @ record["palm_to_payload"]
    return float(payload[2, 3] + np.abs(payload[2, :3]) @ record["payload_half"])


# --- Verification: did the object land, and leave the hand ---
def attached_center(plan, ctx):
    """The held object's centre in the palm frame, from its last view when that matches the
    carry record, else from the record."""
    before = ctx.observe(plan.epoch)
    center_in_palm = ctx.carried[plan.side]["palm_to_payload"][:3, 3]
    if not before.visible(plan.label):
        return center_in_palm
    center = visible_shape(before.target_points(plan.label), ctx.settings.grasp.shape_quantiles).center
    palm = before.palm(ctx.arms[plan.side], plan.side)
    if np.linalg.norm(center - points_in(palm, center_in_palm)) <= ctx.settings.grasp.recovery_match:
        return points_in(inverse(palm), center)
    return center_in_palm


def verify_settled(plan, ctx, center_in_palm):
    """Watch the release for settle_count ticks. A verified landing returns the arm and reports."""
    ctx.stage("Verifying placement")
    settling = Settling(plan, ctx, center_in_palm)
    for _ in range(ctx.settings.placement.settle_count):
        ctx.advance(plan.epoch, ctx.settings.placement.tick)
        if settling.update(ctx.observe(plan.epoch)) and plan.side not in ctx.carried:
            failed = complete_return(plan, ctx)
            if failed is not None:
                return failed
            if plan.support:
                return placed(plan, "it settled on the support")
            return placed(plan, "it settled below the release point")
    if not settling.seen:
        return unseen_release(plan, ctx, settling.open_hand.count)
    raise SkillFailure("PLACEMENT_NOT_VERIFIED",
                       f"The {plan.side} hand's fingers opened; separation from the hand and visible settling "
                       "were not both verified " + ("at the support" if plan.support
                                                    else "near the release XY; landing is unverified"))


class Settling:
    """Fresh evidence that the released object has landed near the release, still and separated
    from the open hand, or that the hand is repeatedly open."""

    def __init__(self, plan, ctx, center_in_palm):
        self.plan, self.ctx, self.center_in_palm = plan, ctx, center_in_palm
        self.landed, self.separated, self.open_hand = (FreshEvidence(plan.epoch) for _ in range(3))
        self.previous, self.seen = None, False

    def update(self, frame):
        """Whether this frame completes the stable run. The carry clears on repeated separation."""
        plan, ctx, settings = self.plan, self.ctx, self.ctx.settings
        hand = frame.hands[plan.side]
        opened = hand.closure <= settings.placement.open_closure_limit and not hand.retained(settings.grasp)
        self.open_hand.update(frame, opened)
        if not frame.visible(plan.label):
            self.landed.update(frame, False)
            self.separated.update(frame, False)
            self.previous = None
            return False
        self.seen = True
        shape = visible_shape(frame.target_points(plan.label), settings.grasp.shape_quantiles)
        attached = points_in(frame.palm(ctx.arms[plan.side], plan.side), self.center_in_palm)
        separated = opened and np.linalg.norm(shape.center - attached) > min(
            settings.grasp.recovery_separation, settings.placement.withdraw_height * .5)
        count = self.separated.update(frame, separated)
        if count is not None and count >= settings.grasp.recovery_confirmations:
            ctx.carried.pop(plan.side, None)
        stable = self.landed.update(frame, separated and self.near(shape) and self.still(shape))
        if stable is None:
            return False
        self.previous = shape.center
        return stable >= settings.placement.stable_frames

    def near(self, shape):
        placement, point = self.ctx.settings.placement, self.plan.point
        on_support = abs(shape.lower[2] - point[2]) < placement.support_height_tolerance
        return (np.linalg.norm(shape.center[:2] - point[:2]) < placement.xy_tolerance and
                (not self.plan.support or on_support))

    def still(self, shape):
        return (self.previous is not None and
                np.linalg.norm(shape.center - self.previous) < self.ctx.settings.placement.still_distance)


def unseen_release(plan, ctx, open_count):
    """Verify a release whose object left the view from repeated open-hand evidence, else
    OUTCOME_UNKNOWN."""
    if open_count < ctx.settings.grasp.recovery_confirmations:
        raise SkillFailure("OUTCOME_UNKNOWN",
                           f"The target is not visible and fresh {plan.side}-hand evidence cannot verify release")
    failed = complete_return(plan, ctx)
    if failed is not None:  # the hand measured open repeatedly: the object is out either way
        ctx.carried.pop(plan.side, None)
        return failed
    final = ctx.observe(plan.epoch).hands[plan.side]
    if final.closure > ctx.settings.placement.open_closure_limit or final.retained(ctx.settings.grasp):
        raise SkillFailure("OUTCOME_UNKNOWN",
                           f"The target is not visible and the returned {plan.side} hand may still retain it")
    ctx.carried.pop(plan.side, None)
    return placed(plan, "the hand opened and is empty")


def placed(plan, message):
    """An OK result saying what was verified."""
    short = f", {plan.short_m:.2f} m short of the requested point, as far as the arm reached" if plan.short_m else ""
    return Result("place", "OK", f"Released from the {plan.side} hand; {message}{short}.",
                  {"hand": plan.side, "short_of_point_m": round(plan.short_m, 3)})


def complete_return(plan, ctx):
    """Return the arm. A blocked return is a Result, None on success."""
    try:
        return_arm(plan, ctx)
    except SkillFailure as error:
        return Result("place", error.code, f"Released; arm return refused: {error}", {"hand": plan.side})
    return None


HAND_HANG_M = 0.05  # the open hand hangs 4-5 cm below the palm
HAND_WIDTH_M = 0.10  # the open hand's reach sideways from the palm


def return_arm(plan, ctx):
    """The one arm reset (`reset_arm`), kept over the object just released while over it."""
    above = None
    if plan.released is not None:
        frame = ctx.observe(plan.epoch)
        center, radius, top = plan.released
        local = points_in(inverse(frame.world_torso), np.r_[center[:2], top])
        above = (local[:2], radius, local[2] + HAND_HANG_M + ctx.settings.placement.withdraw_clearance)
    if not reset_arm(ctx, plan.epoch, plan.side, above):
        raise SkillFailure("ARM_NOT_REACHED", "Measured arm did not return near its rest pose")

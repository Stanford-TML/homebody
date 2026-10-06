"""The pick skill: walk in when needed, plan a grasp from visible depth, approach, close, lift,
verify the hold and stow the carry."""
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from homebody.helpers.collision import HAND_END_LINKS, Clearance
from homebody.helpers.geometry import inverse, points_in, rotation_error, wrap
from homebody.helpers.grasp import Shape, candidates, visible_shape
from homebody.helpers.target import TargetTrack, arm_occlusion_evidence
from homebody.helpers.trajectory import approach_path
from homebody.motion.arm import close_or_open, move_path, reset_arm, servo
from homebody.motion.carry import (
    RECOVERY_FAILURES,
    RecoveryPlan,
    begin_pending,
    compact_carry,
    lift_pending,
    recover,
    recovery_result,
)
from homebody.primitives.observations import Frame, FreshEvidence

from . import navigate
from .contract import Result, SkillFailure, click_argument, side_argument

NEEDS = ("base_velocity", "arm_target", "grip", "advance", "stop")

FAILURES = ("HAND_OCCUPIED", "CALIBRATION_MISSING", "NO_GRASP", "UNREACHABLE",
            "TARGET_MOVED", "ARM_NOT_REACHED", "LIFT_NOT_VERIFIED") + RECOVERY_FAILURES + navigate.FAILURES
PROMPT = ('pick: point_normalized_1000 [x, y], a pixel in the middle of the object\'s body in image 1; '
          'hand "right" (default) or "left". Walks to a grasp stance when needed, grasps the object and '
          'lifts it into a carry pose in front of the body; OK means it is held. A held object keeps the '
          'robot farther from furniture. With both hands loaded the robot may be unable to turn or back '
          'out, and a held object can be dragged out of the hand.')
RETRIED_APPROACH_FAILURES = ("COLLISION", "TARGET_LOST")


@dataclass(frozen=True)
class Plan:
    epoch: int
    side: str
    label: int
    approach_joints: tuple
    pre: np.ndarray
    palm: np.ndarray
    center: np.ndarray
    points: np.ndarray


@dataclass(frozen=True)
class ApproachPlan:
    """A walk-in before the grasp."""
    epoch: int
    side: str
    label: int
    drive_plan: navigate.Plan


def prepare(arguments, selected, current, planning):
    """A recheck of a carried target, a walk-in to a grasp stance, or a grasp from here."""
    side = side_argument(arguments)
    label, _ = click_argument(arguments, selected)
    if side in planning.carried:
        return RecoveryPlan(current.epoch, side, label,
                            resume_lift=planning.carried[side]["label"] == label)
    other = "left" if side == "right" else "right"
    if other in planning.carried and planning.carried[other]["label"] == label:
        raise SkillFailure("INVALID_ARGUMENT", f"The {other} hand carries the selected object; "
                                               f"repeat pick with the {other} hand to recheck it")
    return approach_stance(side, label, current, planning) or plan_target(side, label, current, planning)


def execute(plan, ctx):
    if isinstance(plan, RecoveryPlan):
        return recheck(plan, ctx)
    if isinstance(plan, ApproachPlan):
        plan = walk_in(plan, ctx)
    return grasp(plan, ctx)


def grasp(plan, ctx):
    """The pick from a planned grasp: open, approach, close, lift, verify, compact the carry."""
    ctx.stage("Opening hand")
    close_or_open(ctx, plan.epoch, plan.side, 0.0, ctx.settings.grasp.open_seconds)
    plan, target, frame = approach_or_replan(plan, ctx)
    begin_pending(ctx, frame, plan.side, plan.label, lift_above(frame, ctx, plan.side),
                  points=target.track.last_supported_points)
    ctx.stage("Closing hand")
    frame = close_or_open(ctx, plan.epoch, plan.side, 1.0, ctx.settings.grasp.close_seconds)
    frame = squeeze(plan, ctx, frame)
    ctx.carried[plan.side]["lift_target"] = lift_above(frame, ctx, plan.side)
    ctx.stage("Lifting")
    lift_pending(ctx, plan.epoch, plan.side)
    ctx.stage("Verifying grasp")
    status, frame = recover(ctx, plan.epoch, plan.side)
    if status == "cleared":
        ctx.stage("Returning arm")
        result = recovery_result("pick", status)
        if reset_arm(ctx, plan.epoch, plan.side):
            return result
        return Result(result.skill, result.code, f"{result.message}; the arm did not return near its rest pose")
    if status != "verified":
        raise SkillFailure("LIFT_NOT_VERIFIED", "Repeated current evidence did not verify lift; grasp remains uncertain")
    frame = compact_carry(ctx, plan.epoch, plan.side)
    shape = visible_shape(frame.target_points(plan.label), ctx.settings.grasp.shape_quantiles)
    palm = frame.palm(ctx.arms[plan.side], plan.side)
    record = ctx.carried[plan.side]
    return Result("pick", "OK", "Repeated hand resistance and current visible lift verified",
                  {"side": plan.side,
                   "observed_top_rise": float(shape.upper[2] - record["baseline_upper"]),
                   "palm_rise": float(palm[2, 3] - record["baseline_palm_height"])})


def recheck(plan, ctx):
    """Re-verify a carried target, resuming its lift if enclosed."""
    ctx.stage("Checking carry")
    status, _ = recover(ctx, plan.epoch, plan.side)
    if status == "enclosed" and plan.resume_lift:
        ctx.stage("Lifting")
        lift_pending(ctx, plan.epoch, plan.side)
        ctx.stage("Verifying grasp")
        status, _ = recover(ctx, plan.epoch, plan.side)
    if status == "verified":
        if ctx.carried[plan.side]["label"] == plan.label:
            compact_carry(ctx, plan.epoch, plan.side)
            return Result("pick", "OK", "Current repeated hand and visual evidence verifies the original grasp")
        return Result("pick", "HAND_OCCUPIED", "The original carried object is still verified; the hand was not opened")
    return recovery_result("pick", status)


def abort(plan, ctx):
    """Reset the arm after a refused grasp. The base does not move."""
    if plan.side in ctx.carried or ctx.observations.epoch != plan.epoch:
        return
    ctx.stage("Returning arm")
    if not reset_arm(ctx, plan.epoch, plan.side):
        raise SkillFailure("ARM_NOT_REACHED", "Measured arm did not return near its rest pose")


def approach_stance(side, label, current, planning):
    """An ApproachPlan to a square stance at grasp range, or None when already there."""
    if not planning.jaw_samples.get(side):
        return None
    points = current.target_points(label)
    shape = visible_shape(points, planning.settings.grasp.shape_quantiles)
    centre = shape.center[:2]
    ahead, left = navigate.body_frame(current.base_pose, centre - current.base_pose[:2])
    bounds = planning.settings.grasp
    toward_hand = -left if side == "right" else left
    reach = grasp_distance(side, shape, current, planning)
    if ahead <= 0 or (reach - bounds.approach_short <= ahead <= reach + bounds.approach_long and
                      bounds.approach_across <= toward_hand <= bounds.approach_side):
        return None
    stance = {"goal_xy_m": current.base_pose[:2].tolist(), "facing_xy_m": centre.tolist(),
              "snap": True, "hand": side}
    walk = navigate.prepare(stance, current, current, planning, reach=reach)
    ahead = (centre - walk.goal) @ np.array([np.cos(walk.yaw), np.sin(walk.yaw)])
    if ahead > reach + bounds.approach_long:
        raise SkillFailure("UNREACHABLE", f"The nearest clear stance leaves the object {ahead:.2f} m "
                                          f"ahead; the arm's grasp range is {reach:.2f} m")
    return ApproachPlan(current.epoch, side, label, walk)


def grasp_distance(side, shape, current, planning):
    """How far ahead of the base (m) the object's centre sits when the pre-grasp palm is
    grasp_range from the shoulder. UNREACHABLE when no stance brings it that close."""
    settings, arm = planning.settings.grasp, planning.arms[side]
    top = points_in(inverse(current.world_torso), shape.upper)[2]
    rise = top + settings.approach_height - arm.shoulder[2]
    if abs(rise) >= settings.grasp_range:
        where = "above" if rise > 0 else "below"
        raise SkillFailure("UNREACHABLE", f"The object's pre-grasp point is {abs(rise):.2f} m {where} "
                                          f"the shoulder; no stance brings it within the arm's "
                                          f"grasp range of {settings.grasp_range:.2f} m")
    horizontal = np.sqrt(settings.grasp_range ** 2 - rise ** 2)
    jaw = np.mean([np.asarray(sample["palm_to_jaw"])[0, 3] for sample in planning.jaw_samples[side]])
    return float(arm.shoulder[0] + horizontal + jaw)


def walk_in(plan, ctx):
    """Walk to the grasp stance, then plan the grasp from the new view."""
    ctx.stage("Walking to the grasp stance")
    navigate.execute(plan.drive_plan, ctx)
    ctx.advance(plan.epoch)
    current = ctx.observe(plan.epoch)
    ctx.stage("Planning grasp from the new stance")
    result = plan_target(plan.side, plan.label, current, ctx.planning())
    ctx.emit({"type": "skill_prepared", "skill": "pick", "plan": result, "walked_in": True})
    return result


def plan_target(side, label, current, planning):
    """The best routable grasp of the captured target from visible geometry."""
    if side in planning.carried:
        raise SkillFailure("HAND_OCCUPIED", "The selected hand has unresolved grasp ownership")
    if current.hands[side].retained(planning.settings.grasp):
        raise SkillFailure("HAND_OCCUPIED", "The selected hand already has retention evidence")
    samples = planning.jaw_samples.get(side)
    if not samples:
        raise SkillFailure("CALIBRATION_MISSING", "Robot-only jaw calibration is required")
    points = current.target_points(label)
    search = GraspSearch(side, label, current, planning)
    fitted = _width_fitted_candidates(points, samples, planning.settings.grasp, planning.cancelled)
    best = search.best(list(search.reachable(fitted)))
    if planning.cancelled():
        raise SkillFailure("CANCELLED", "Operator cancelled grasp preparation")
    if best is None:
        raise SkillFailure("NO_GRASP", f"No grasp passed visible-depth geometry, reach and clearance gates: "
                           f"{search.counts}; blockers: {dict(search.blockers)}")
    grasp, route = best
    return Plan(current.epoch, side, label, route, grasp.pre, grasp.palm,
                grasp.shape.center, points)


@dataclass(frozen=True)
class GraspCandidate:
    """A proposed jaw pose that a calibrated opening fits. Lower tiers rank first."""
    tier: int
    width: float
    pre: np.ndarray
    palm: np.ndarray
    shape: Shape


@dataclass(frozen=True)
class ReachableGrasp:
    """A candidate the arm reaches with a clear descent."""
    candidate: GraspCandidate
    pre_joints: np.ndarray


def _width_fitted_candidates(points, samples, settings, cancelled):
    """Grasp candidates whose measured width a calibrated jaw opening fits within aperture_error."""
    apertures = [sample["aperture"] for sample in samples]
    limits = min(apertures) - settings.aperture_error, max(apertures) + settings.aperture_error
    for sample in samples:
        if cancelled():
            raise SkillFailure("CANCELLED", "Operator cancelled grasp preparation")
        palm_to_jaw = np.asarray(sample["palm_to_jaw"])
        for tier, pre, palm, shape in candidates(points, palm_to_jaw, limits, settings):
            width = np.ptp(points @ (palm @ palm_to_jaw)[:3, 0])
            if abs(sample["aperture"] - width) <= settings.aperture_error:
                yield GraspCandidate(tier, width, pre, palm, shape)


class GraspSearch:
    """One frame's grasp gates (reach, descent clearance, approach route), with counts per gate
    and the blockers met."""
    GATES = ("width", "pre_reachable", "grasp_reachable", "descent_clear", "approach_clear")

    def __init__(self, side, label, current, planning):
        self.arm, self.seed = planning.arms[side], current.arm_joints(side)
        self.settings, self.cancelled = planning.settings.grasp, planning.cancelled
        self.torso = inverse(current.world_torso)
        collision = planning.settings.collision
        opened = np.zeros_like(current.hands[side].joints)  # the hand opens before it moves
        self.descent = Clearance(current, planning.arms, side, collision, (label,), fingers=opened)
        self.approach = Clearance(current, planning.arms, side, collision, exempt_start=True, fingers=opened)
        self.counts = dict.fromkeys(self.GATES, 0)
        self.blockers = Counter()

    def reachable(self, candidates):
        for candidate in candidates:
            self.counts["width"] += 1
            pre_joints = self.arm.solve(self.torso @ candidate.pre, self.seed)
            if pre_joints is None:
                continue
            self.counts["pre_reachable"] += 1
            grasp_joints = self.arm.solve(self.torso @ candidate.palm, pre_joints)
            if grasp_joints is None:
                continue
            self.counts["grasp_reachable"] += 1
            obstruction = self.descent.path_violation(pre_joints, grasp_joints)
            if obstruction is not None:
                self.blockers[obstruction] += 1
                continue
            self.counts["descent_clear"] += 1
            yield ReachableGrasp(candidate, pre_joints)

    def best(self, reachable):
        """The first routable (candidate, approach route) by tier, width and joint travel, trying
        each start lift in turn. None when none routes."""
        ranked = sorted(reachable, key=lambda grasp: (grasp.candidate.tier, grasp.candidate.width,
                                                      np.linalg.norm(grasp.pre_joints - self.seed)))
        for lift in (0., *self.settings.approach_start_lifts_m):
            for grasp in ranked:
                approach = approach_path(self.arm, self.seed, grasp.pre_joints, self.approach, self.settings,
                                         self.cancelled, initial_lift_m=lift, blockers=self.blockers)
                if approach is not None:
                    self.counts["approach_clear"] += 1
                    return grasp.candidate, approach
        return None


def approach_or_replan(plan, ctx):
    """Approach, replanning a blocked or lost approach up to approach_attempts times."""
    target = PickTarget(plan, ctx)
    attempts = ctx.settings.grasp.approach_attempts
    for attempt in range(1, attempts + 1):
        try:
            ctx.stage("Approaching")
            frame = approach(plan, ctx, target)
            break
        except SkillFailure as error:
            if error.code not in RETRIED_APPROACH_FAILURES or attempt == attempts:
                raise
            plan, target = replan(plan, ctx, target, error, attempt + 1)
    return plan, target, frame


def approach(plan, ctx, target):
    """Route to the pre-grasp, servo onto it, then descend to the grasp palm."""
    if len(plan.approach_joints):
        ctx.stage("Following grasp path")
        move_path(ctx, plan.epoch, plan.side, plan.approach_joints, tracked_target=target)
    ctx.stage("Aligning above target")
    servo(ctx, plan.epoch, plan.side, plan.pre, tracked_target=target)
    target.begin_descent()
    ctx.stage("Descending to grasp")
    return servo(ctx, plan.epoch, plan.side, plan.palm, tracked_target=target, exclude_labels=(plan.label,))


def replan(plan, ctx, target, error, attempt):
    """Replan from a fresh view of the same target, parking the arm first when it hides it."""
    ctx.actions.stop(plan.epoch)
    ctx.stage("Replanning approach")
    stopped = ctx.observe(plan.epoch)
    ctx.advance(plan.epoch)
    frame = ctx.observe(plan.epoch)
    packets = FreshEvidence(plan.epoch)
    packets.update(stopped, True)
    if packets.update(frame, True) is None:
        raise SkillFailure("STALE_OBSERVATION", "Grasp replanning requires a new sensor packet") from error
    if not target.supported(frame):
        ctx.stage("Clearing target view")
        if not reset_arm(ctx, plan.epoch, plan.side):
            raise SkillFailure("ARM_NOT_REACHED", "The arm hiding the target did not return") from error
        ctx.advance(plan.epoch)
        frame = ctx.observe(plan.epoch)
        if not target.supported(frame):
            raise SkillFailure("TARGET_LOST", "A new grasp requires a supported target view") from error
    ctx.emit({"type": "grasp_replan", "attempt": attempt, "label": plan.label,
              "frame_id": frame.frame_id, "reason": str(error)})
    plan = plan_target(plan.side, plan.label, frame, ctx.planning())
    ctx.emit({"type": "skill_prepared", "skill": "pick", "plan": plan, "attempt": attempt})
    return plan, target.rebased(plan)


def squeeze(plan, ctx, frame):
    """Close on by the squeeze past where the fingers stopped."""
    closure = frame.hands[plan.side].closure + ctx.settings.grasp.squeeze
    if closure <= 1.0:
        return frame
    ctx.stage("Squeezing")
    return close_or_open(ctx, plan.epoch, plan.side, closure, ctx.settings.grasp.close_seconds)


def lift_above(frame, ctx, side):
    """This frame's palm raised by lift_height, in world."""
    lift = frame.palm(ctx.arms[side], side)
    lift[2, 3] += ctx.settings.grasp.lift_height
    return lift


class PickTarget:
    """Track the planned target's supported surfaces during the approach and bound a finish
    on a partial view. The servo calls `update` on every read."""

    def __init__(self, plan, ctx, original_center=None, limits=None):
        self.plan, self.ctx = plan, ctx
        self.limits = ctx.settings.blind_finish if limits is None else limits
        self.capture_center = plan.center.copy() if original_center is None else original_center.copy()
        grasp = ctx.settings.grasp
        self.track = TargetTrack(plan.points, grasp.shape_quantiles, grasp.target_surface_tolerance,
                                 grasp.target_shift, self.capture_center)
        self.mode = None
        self.descending = False
        self.last_supported_time = None
        self.window = None
        self.shift = np.zeros(3)

    def begin_descent(self):
        """Enter the descent, where a partial view is bounded rather than refused."""
        self.descending = True

    def rebased(self, plan):
        """A tracker for the same target freshly planned, keeping the capture anchor."""
        return PickTarget(plan, self.ctx, self.capture_center, self.limits)

    def supported(self, frame):
        """Whether this frame shows the captured surfaces within their capture range."""
        return self._match(self._points(frame)).mode == "supported"

    def update(self, state):
        """Track one read of the approach. The returned shift (m, world) moves the planned palm."""
        palm = state.palm(self.ctx.arms[self.plan.side], self.plan.side)
        if isinstance(state, Frame):
            self._see(state, palm)
        goal = self.plan.palm.copy()
        goal[:3, 3] += self.shift
        if self.descending and (self.mode != "supported" or self.window is not None):
            self._require_corridor(palm, goal)
            if self.window is None:
                self.window = PartialWindow.opened(state, palm, self.limits.partial_travel)
        if self.window is not None and not self.window.within_bounds(
                state, palm, self.limits, self.ctx.settings.motion):
            raise SkillFailure("TARGET_LOST", "Partial-view finish exceeded its measured motion or time bound")
        return self.shift

    def _points(self, frame):
        label = self.plan.label
        return frame.target_points(label) if frame.visible(label) else np.empty((0, 3))

    def _match(self, points):
        """The track's verdict on these points. A moved target raises TARGET_MOVED."""
        match = self.track.update(points)
        if match.mode == "moved":
            raise SkillFailure("TARGET_MOVED", "Supported target geometry moved beyond the capture range")
        return match

    def _see(self, frame, palm):
        """Match this frame's view of the target and update the shift."""
        points = self._points(frame)
        match = self._match(points)
        self.shift = self.track.last_supported_shape.center - self.plan.center
        if match.mode == "uncertain":
            goal = self.plan.palm.copy()
            goal[:3, 3] += self.shift
            self._admit_hidden(frame, points, palm, goal)
            mode = "occluded"
        else:
            self.last_supported_time = frame.time
            mode = match.mode
        if mode != self.mode:
            self.ctx.emit({"type": "target_tracking", "mode": mode, "frame_id": frame.frame_id,
                           "partial_travel": self.window.travel if self.window is not None else 0.0,
                           "match": match})
            self.mode = mode

    def _admit_hidden(self, frame, points, palm, goal):
        """Allow a blind finish only when the arm hides the target, the palm is close and the
        target was seen recently. Otherwise TARGET_LOST."""
        if not self._arm_hides(frame, points):
            raise SkillFailure("TARGET_LOST", "Current target is not explained by measured arm occlusion")
        if self.mode == "occluded":
            return
        remaining = np.linalg.norm(palm[:3, 3] - goal[:3, 3])
        recent = (self.mode == "partial" and
                  frame.time - self.last_supported_time <= self.limits.occluded_recency)
        if not (remaining <= self.limits.occluded_reach and recent):
            raise SkillFailure("TARGET_LOST", "Current visible surfaces do not support the captured target pose")
        if self.window is None:
            self.window = PartialWindow.opened(frame, palm, remaining + self.limits.travel_margin)

    def _arm_hides(self, frame, points):
        """Whether the measured hand explains the missing target rays and what remains visible
        agrees with the captured surfaces."""
        grasp, side = self.ctx.settings.grasp, self.plan.side
        arm, joints = self.ctx.arms[side], frame.arm_joints(side)
        boxes = (arm.collision.boxes(arm, joints)[-HAND_END_LINKS:] +
                 arm.collision.hand_boxes(arm, joints, frame.hands[side].joints))
        occlusion = arm_occlusion_evidence(self.track.last_supported_points, points, frame, boxes,
                                           grasp.target_surface_tolerance)
        blocked = (occlusion.measured_fraction >= grasp.occlusion_min_rays and
                   occlusion.explained_missing_fraction >= grasp.occlusion_missing_explained and
                   occlusion.exposed_missing_fraction <= grasp.occlusion_exposed_missing_max)
        if len(points):
            overlap = np.mean(cKDTree(self.track.last_supported_points).query(points)[0] <=
                              grasp.target_surface_tolerance)
            consistent = overlap >= grasp.occlusion_visible_match
        else:
            consistent = occlusion.measured_fraction >= grasp.occlusion_no_view_rays
        return blocked and consistent

    def _require_corridor(self, palm, goal):
        """TARGET_LOST unless the palm lies within the planned descent's corridor."""
        grasp, motion = self.ctx.settings.grasp, self.ctx.settings.motion
        offset = palm[:3, 3] - goal[:3, 3]
        if (np.linalg.norm(offset[:2]) > grasp.approach_height or
                not -motion.palm_tolerance <= offset[2] <= grasp.approach_height + grasp.target_shift or
                np.linalg.norm(rotation_error(goal[:3, :3], palm[:3, :3])) > self.limits.palm_turn):
            raise SkillFailure("TARGET_LOST", "Partial target view outside the final approach corridor")


@dataclass
class PartialWindow:
    """The budget of a finish begun with the target partly visible: base and head still, palm
    travel and time bounded."""
    start: float
    base: np.ndarray
    camera: np.ndarray
    last_palm: np.ndarray
    travel_limit: float
    travel: float = 0.0

    @classmethod
    def opened(cls, frame, palm, travel_limit):
        return cls(frame.time, frame.base_pose.copy(), frame.world_camera.copy(),
                   palm[:3, 3].copy(), travel_limit)

    def within_bounds(self, frame, palm, limits, motion):
        self.travel += np.linalg.norm(palm[:3, 3] - self.last_palm)
        self.last_palm = palm[:3, 3].copy()
        return (frame.time - self.start <= motion.arm_timeout and self.travel <= self.travel_limit and
                np.linalg.norm(frame.base_pose[:2] - self.base[:2]) <= motion.palm_tolerance and
                abs(wrap(frame.base_pose[2] - self.base[2])) <= limits.base_turn and
                np.linalg.norm(frame.world_camera[:3, 3] - self.camera[:3, 3]) <= limits.head_drift and
                np.linalg.norm(rotation_error(frame.world_camera[:3, :3], self.camera[:3, :3])) <=
                limits.head_turn)

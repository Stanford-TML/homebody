"""The place skill: release a held object at an observed support or a captured torso XY and
verify the observed outcome."""
from dataclasses import dataclass, replace
from functools import partial
from itertools import product

import numpy as np

from homebody.helpers.collision import Clearance, under_box
from homebody.helpers.geometry import inverse, points_in, unproject
from homebody.helpers.grasp import payload_offset, visible_shape
from homebody.helpers.raster import world_to_px
from homebody.helpers.volume import outside
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
    robot_boxes,
)
from homebody.primitives.actions import ActionRejected, PhysicsInvalid
from homebody.primitives.observations import FreshEvidence, ObservationError
from homebody.vlm.topdown import normalized_click

from .contract import Result, SkillFailure, click_argument, refusal_code, side_argument
from .navigation_goal import finite_number

NEEDS = ("base_velocity", "arm_target", "grip", "advance", "stop")

PROMPT_FAILURES = ("NOT_HOLDING", "SURFACE_CHANGED", "ARM_NOT_REACHED",
                   "RELEASE_NOT_VERIFIED", "OUTCOME_UNKNOWN", "PLACEMENT_NOT_VERIFIED") + RECOVERY_FAILURES
FAILURES = PROMPT_FAILURES + ("UNREACHABLE",)
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


REAL_PROMPT = PROMPT[:PROMPT.index('The palm reaches about ')].replace(
    'current hand height', 'current object-centre height') + (
    'delta_z_m defaults to zero: keep the carried object height unless current observations justify a change. '
    'Prefer a visible support click when placing down on a surface. '
    'The exact requested placement must be reachable and verified before opening the hand. '
    'An unreachable target is refused with the object retained; navigate to a better stance and retry. '
    'The robot never substitutes a closer release point.')


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
    return_rotation: np.ndarray | None = None  # measured carry wrist orientation in the torso
    opened_at: float | None = None


@dataclass(frozen=True)
class DockPlan:
    """A release beyond reach under a navigation that docks: lift the carry over the destination
    when needed, dock straight in until the exact release solves, then plan it from there.
    `torso` is the selection's world torso, which fixes a torso-frame destination."""
    epoch: int
    side: str
    label: int
    surface: int | None
    point: np.ndarray
    height: float
    torso: np.ndarray
    released: tuple | None = None


def prepare(arguments, selected, current, planning):
    """Resolve the destination and plan a checked release, or a RecoveryPlan for an uncertain carry."""
    side = side_argument(arguments)
    if side not in planning.carried and len(planning.carried) == 1:
        side = next(iter(planning.carried))
    if "point_normalized_1000" in arguments and planning.target_selector is not None:
        selected, current = planning.target_selector(selected, current, arguments["point_normalized_1000"])
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

    try:
        return plan_release(side, surface, point, height, selected.world_torso, current, planning)
    except SkillFailure as error:
        if error.code != "UNREACHABLE" or not hasattr(planning.navigation, "dock"):
            raise
        return DockPlan(current.epoch, side, carried["label"], surface, point, height, selected.world_torso)


def plan_release(side, surface, point, height, world_torso, current, planning):
    """The checked release for a destination: a reachable pose, its pre-pose and a clear arm path."""
    carried, arm, settings = planning.carried[side], planning.arms[side], planning.settings
    withdraw = current.palm(arm, side)
    palm, release, search = release_target(side, surface, point, height, world_torso, current, planning)
    seed = current.arm_joints(side)
    center_offset = payload_offset(carried, withdraw[:3, :3])[0]
    pre, palm, pre_q, grasp_q = release_pose(arm, current.world_torso, palm, center_offset, seed, search)
    if grasp_q is None and surface is None and planning.manipulation.target_or_refuse:
        offset = carried["palm_to_payload"][:3, 3]
        release = points_in(palm, offset)
        target = inverse(current.world_torso) @ palm
        target[:3, 3] = points_in(inverse(current.world_torso), release)
        oriented = arm.solve(target, seed, position_only=True, point_in_palm=offset)
        if oriented is not None:
            candidate = current.world_torso @ arm.forward(oriented)[0]
            center_offset = candidate[:3, :3] @ offset
            candidate[:3, 3] = release - center_offset
            pre, palm, pre_q, grasp_q = release_pose(
                arm, current.world_torso, candidate, center_offset, oriented,
                replace(search, twists=(0.,)))
    if grasp_q is None and planning.manipulation.target_or_refuse:
        raise SkillFailure("UNREACHABLE", "The intended placement is unreachable; reposition before placing")
    if surface is None and release is None:
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
    rotation = arm.forward(seed)[0][:3, :3].copy() if planning.manipulation.target_or_refuse else None
    return Plan(current.epoch, side, carried["label"], release, pre_q, pre, palm, surface is not None,
                return_rotation=rotation)


def release_target(side, surface, point, height, world_torso, current, planning):
    """The world palm that releases at the destination, the release point (None for a torso-XY
    release) and the release search."""
    carried, settings = planning.carried[side], planning.settings
    withdraw = current.palm(planning.arms[side], side)
    if surface is not None:
        palm, release = support_release(current, surface, point, height, carried, withdraw, settings)
        return palm, release, ReleaseSearch.support_click(settings)
    search = (ReleaseSearch((0.,), (0.,), settings.placement.release_twists, 0., 0.)
              if planning.manipulation.target_or_refuse else ReleaseSearch.metric_drop(settings))
    return drop_palm(world_torso, point, height, carried, withdraw,
                     object_height=planning.manipulation.target_or_refuse), None, search


def execute(plan, ctx):
    """Approach, lower, open the hand, retract, then verify the landing and return the arm."""
    if isinstance(plan, RecoveryPlan):
        ctx.stage("Checking carry")
        status, _ = recover(ctx, plan.epoch, plan.side)
        return replace(recovery_result("place", status), details={"hand": plan.side})
    docked = isinstance(plan, DockPlan)
    if docked:
        plan = dock_in(plan, ctx)
    epoch, side, held, settings = plan.epoch, plan.side, (plan.label,), ctx.settings

    if ctx.manipulation.target_or_refuse:
        if plan.pre_joints is None:
            raise SkillFailure("UNREACHABLE", "Placement requires a checked route to the intended target")
        record = ctx.carried.get(side)
        if (record is None or record["label"] != plan.label or
                not continuing(ctx.observe(epoch), ctx.arms[side], side, record, settings.grasp)):
            raise SkillFailure("HOLD_UNCERTAIN", "The prepared carry is no longer verified; no release occurred")
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

    if ctx.manipulation.target_or_refuse:
        current = ctx.observe(epoch)
        actual = points_in(current.palm(ctx.arms[side], side), center_in_palm)
        if np.linalg.norm(actual[:2] - plan.point[:2]) > settings.placement.xy_tolerance:
            raise SkillFailure("ARM_NOT_REACHED", "The payload did not reach the intended release XY; hand kept closed")
        if not continuing(current, ctx.arms[side], side, ctx.carried[side], settings.grasp):
            raise SkillFailure("HOLD_UNCERTAIN", "Carry verification was lost before release; hand kept closed")
        target = withdrawal_target(plan, current, ctx.arms[side], settings.placement)
        if ctx.arms[side].solve(inverse(current.world_torso) @ target, current.arm_joints(side)) is None:
            raise SkillFailure("UNREACHABLE", "Release pose leaves no reachable withdrawal; hand kept closed. "
                               "Choose a lower release or reposition before placing.",
                               {"hand": side, "released": False, "phase": "withdrawal_preflight"})
    ctx.carried[side]["state"] = CarryState.UNCERTAIN
    plan = replace(plan, opened_at=ctx.observations.time)
    ctx.stage("Releasing")
    frame = close_or_open(ctx, epoch, side, 0.0, settings.grasp.open_seconds)
    ctx.stage("Checking release")
    if frame.hands[side].closure > settings.placement.open_closure_limit:
        raise SkillFailure("RELEASE_NOT_VERIFIED", f"The {side} hand's measured fingers did not open")

    current = ctx.observe(epoch)
    plan = replace(plan, released=released_extent(current.palm(ctx.arms[side], side), ctx.carried[side]))
    if not ctx.manipulation.target_or_refuse:
        retract(plan, ctx, current)

    # A release reachable without docking can leave the base beside the support.
    withdraw = ctx.manipulation.target_or_refuse and callable(getattr(ctx.navigation, "back_out", None))
    result = verify_settled(plan, ctx, center_in_palm, withdraw=withdraw)
    if docked and result.ok and not withdraw:
        ctx.stage("Backing clear of the support")
        ctx.navigation.back_out(ctx, epoch)
    return result


def retract(plan, ctx, current=None):
    """Withdraw the palm; strict placement first verifies release and uses live clearance."""
    ctx.stage("Retracting")
    current = ctx.observe(plan.epoch) if current is None else current
    if ctx.manipulation.target_or_refuse:
        require_empty(plan, ctx, current)
    placement = ctx.settings.placement
    target = withdrawal_target(plan, current, ctx.arms[plan.side], placement)
    if ctx.manipulation.target_or_refuse:
        record = ctx.carried.pop(plan.side, None)
        try:
            servo(ctx, plan.epoch, plan.side, target)
        finally:
            if record is not None:
                ctx.carried[plan.side] = record
    else:
        palm = current.palm(ctx.arms[plan.side], plan.side)
        target[2, 3] = max(target[2, 3], object_top(palm, ctx.carried[plan.side]) + HAND_HANG_M
                            + placement.withdraw_clearance)
        servo_unchecked(ctx, plan.epoch, plan.side, target)


def withdrawal_target(plan, current, arm, placement):
    """The withdrawal pose, checked for reach before strict release."""
    target = current.palm(arm, plan.side)
    target[2, 3] += placement.withdraw_height
    back = current.world_torso[:2, 3] - target[:2, 3]
    target[:2, 3] += placement.withdraw_back * back / max(np.linalg.norm(back), 1e-9)
    if plan.return_rotation is not None:
        target[:3, :3] = current.world_torso[:3, :3] @ plan.return_rotation
    return target


def dock_in(plan, ctx):
    """Choose the first floor-admitted docking with release margin and idle-hand clearance,
    retaining the first floor-admitted fallback if none clears the idle hand. Lift, execute under
    the navigation command checks, then plan the exact release from the measured docked stance."""
    distance, turn = raise_over(plan, ctx, plan.torso)
    ctx.stage("Docking toward the release")
    _, refused = ctx.navigation.dock(ctx, plan.epoch, distance, turn)
    try:
        return plan_release(plan.side, plan.surface, plan.point, plan.height, plan.torso,
                            ctx.observe(plan.epoch), ctx.planning())
    except SkillFailure as error:
        if error.code == "UNREACHABLE" and refused is not None:
            raise SkillFailure("UNREACHABLE", "Docking stopped short of reach; the object is retained",
                               {"refused_command": refused}) from error
        raise


def _docking_candidate(plan, ctx, current, world_torso):
    """Choose from the finite heading order using CURRENT's measured stance and carry."""
    sign = 1. if plan.side == "right" else -1.
    reaching = None
    for turn in sign * np.radians(DOCK_TURNS_DEG):
        distance = docking_distance(plan, ctx, current, world_torso, 0., turn)
        if distance is None:
            continue
        lift = carry_lift(plan, ctx, current, world_torso, distance, turn)
        distance = docking_distance(plan, ctx, current, world_torso, lift, turn)
        if distance is None:
            continue
        if not ctx.navigation.dock_floor_clear(ctx, current, distance, turn):
            continue
        reaching = reaching or (distance, turn)
        if idle_clear(plan, ctx, current, distance, turn):
            reaching = (distance, turn)
            break
    if reaching is None:
        raise SkillFailure("UNREACHABLE", "No straight docking within dock_distance brings the exact release "
                           "within reach; the object is retained")
    return reaching


DOCK_TURNS_DEG = (0., 10., 20., 30.)  # turned toward the idle hand's side


def docked(current, ahead, turn):
    """The world motion of a turn in place by TURN followed by AHEAD metres along the new heading."""
    x, y, yaw = current.base_pose
    c, s = np.cos(turn), np.sin(turn)
    motion = np.eye(4)
    motion[:2, :2] = [[c, -s], [s, c]]
    motion[:2, 3] = (x, y) - motion[:2, :2] @ (x, y) + ahead * np.array([np.cos(yaw + turn), np.sin(yaw + turn)])
    return motion


def side_corners(current, arm, side):
    """World corners of one arm's link and finger boxes at CURRENT's measured joints."""
    joints = current.arm_joints(side)
    return np.vstack([points_in(current.world_torso, (_CORNERS * half) @ rotation.T + center)
                      for center, rotation, half in (*arm.collision.boxes(arm, joints),
                                                     *arm.collision.hand_boxes(arm, joints,
                                                                               current.hands[side].joints))])


def over_occupied(navigation, points, margin):
    """Whether world points lie within MARGIN of occupied cells of NAVIGATION."""
    over = np.zeros(len(points), bool)
    for offset in ((0., 0.), (margin, 0.), (-margin, 0.), (0., margin), (0., -margin)):
        over |= occupied_at(navigation, points[:, :2] + offset)
    return over


def idle_clear(plan, ctx, current, ahead, turn):
    """Whether the idle arm stays off occupied cells, by the column margin, through the docking and
    the stopping distance past it."""
    idle = "left" if plan.side == "right" else "right"
    settings = ctx.settings
    beyond = ahead + settings.motion.stopping_distance + settings.collision.column_margin
    corners = points_in(docked(current, beyond, turn), side_corners(current, ctx.arms[idle], idle))
    return not over_occupied(current.navigation, corners, settings.collision.column_margin).any()


def carry_lift(plan, ctx, current, world_torso, ahead, turn=0.):
    """How far to lift the carry so that the payload and the holding arm's parts that lie over
    occupied cells once docked, or past it by the stopping distance, clear the returns under the
    payload's way to the release by carry_clearance. UNREACHABLE when nothing there is observed."""
    side, settings = plan.side, ctx.settings
    record, arm = ctx.carried[side], ctx.arms[side]
    palm = current.palm(arm, side)
    target, _, _ = release_target(side, plan.surface, plan.point, plan.height, world_torso, current, ctx.planning())
    payload, landing = palm @ record["palm_to_payload"], target @ record["palm_to_payload"]
    extent = np.abs(payload[:3, :3]) @ record["payload_half"]
    points = points_in(current.world_camera, unproject(current.depth, current.intrinsics,
                                                       (current.labels >= 0) & (current.labels != plan.label)))
    points = points[outside(points, robot_boxes(current, ctx.arms, ctx.carried), settings.collision.column_margin)]
    start, way = payload[:2, 3], landing[:2, 3] - payload[:2, 3]
    along = np.clip((points[:, :2] - start) @ way / max(way @ way, 1e-12), 0., 1.)
    under = np.linalg.norm(points[:, :2] - (start + along[:, None] * way), axis=1) <= (
        np.linalg.norm(extent[:2]) + settings.collision.column_margin)
    if not np.any(under):
        raise SkillFailure("UNREACHABLE", "Nothing under the way to the release is observed; the object is retained")
    corners = side_corners(current, arm, side)
    beyond = ahead + settings.motion.stopping_distance + settings.collision.column_margin
    over = np.zeros(len(corners), bool)
    for reach in (ahead, beyond):
        over |= over_occupied(current.navigation, points_in(docked(current, reach, turn), corners),
                              settings.collision.column_margin)
    lowest = min(payload[2, 3] - extent[2], *corners[over, 2])
    lift = float(points[under, 2].max() + settings.placement.carry_clearance - lowest)
    ctx.emit({"type": "carry_clearance", "lift_m": lift, "top_z": float(points[under, 2].max()),
              "lowest_z": float(lowest), "docking_m": float(ahead), "turn_rad": float(turn)})
    return max(lift, 0.)


def occupied_at(navigation, xy):
    """Whether map points lie on occupied cells of NAVIGATION; outside the raster counts as occupied."""
    pixels = np.rint(world_to_px(navigation.T_map_px, xy)).astype(int)
    height, width = navigation.occupied.shape
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    result = np.ones(len(xy), bool)
    result[inside] = navigation.occupied[pixels[inside, 1], pixels[inside, 0]]
    return result


def raise_over(plan, ctx, world_torso):
    """Return a docking verified from the measured carry after at most three bounded lifts.
    Re-select after lifting because arm motion can change the measured base stance."""
    epoch, side, settings = plan.epoch, plan.side, ctx.settings
    record, arm = ctx.carried[side], ctx.arms[side]
    raised = 0.
    for attempt in range(4):
        current = ctx.observe(epoch)
        ahead, turn = _docking_candidate(plan, ctx, current, world_torso)
        lift = carry_lift(plan, ctx, current, world_torso, ahead, turn)
        if lift <= .005:
            return ahead, turn
        if attempt == 3:
            raise SkillFailure("UNREACHABLE", "The measured carry still does not clear the destination after "
                               "three lifts; the object is retained")
        # A target inside the arm servo's tolerance can finish without clearing the support.
        lift += settings.motion.palm_tolerance
        if raised + lift > settings.placement.max_carry_raise:
            raise SkillFailure("UNREACHABLE", f"Clearing the destination needs a {raised + lift:.2f} m lift; "
                               "the object is retained")
        palm = current.palm(arm, side)
        palm[2, 3] += lift
        ctx.stage("Raising the carry over the destination")
        servo(ctx, epoch, side, palm, exclude_labels=(plan.label,))
        raised += lift
        if not continuing(ctx.observe(epoch), arm, side, record, settings.grasp):
            raise SkillFailure("HOLD_UNCERTAIN", "Carry verification was lost while lifting; no release occurred")


def docking_distance(plan, ctx, current, world_torso, lift=0., turn=0.):
    """How far straight ahead, after a turn by TURN, the exact release LIFT higher first solves with
    dock_reach_margin to spare, or None within dock_distance."""
    side, settings, planning = plan.side, ctx.settings, ctx.planning()
    arm, record = ctx.arms[side], ctx.carried[side]
    palm, _, search = release_target(side, plan.surface, plan.point, plan.height, world_torso, current, planning)
    palm[2, 3] += lift
    search = replace(search, margin=settings.placement.dock_reach_margin)
    center_offset = payload_offset(record, current.palm(arm, side)[:3, :3])[0]
    for distance in np.arange(0., settings.motion.dock_distance + 1e-9, .02):
        torso = docked(current, distance, turn) @ current.world_torso
        if release_pose(arm, torso, palm, center_offset, current.arm_joints(side), search)[3] is not None:
            return float(distance)
    return None


def abort(plan, ctx):
    """Keep a verified carry tucked; hold after uncertain strict release rather than reset at the support."""
    if isinstance(plan, RecoveryPlan) or ctx.observations.epoch != plan.epoch:
        return
    record = ctx.carried.get(plan.side)
    if ctx.manipulation.target_or_refuse and (record is None or record["state"] != CarryState.VERIFIED):
        ctx.stage("Holding after release failure")
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
        if not all(finite_number(value) for value in values):
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


def drop_palm(world_torso, point, height, carried, palm, *, object_height=False):
    """The palm that puts the object's centre at POINT in the captured torso frame, HEIGHT (m)
    above the current palm (SIM) or object-centre height (strict placement)."""
    center_offset = payload_offset(carried, palm[:3, :3])[0]
    reference = palm[:3, 3] + center_offset if object_height else palm[:3, 3]
    captured_height = points_in(inverse(world_torso), reference)[2]
    placed = palm.copy()
    placed[:3, 3] = points_in(world_torso, np.r_[point, captured_height + height]) - center_offset
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


def verify_settled(plan, ctx, center_in_palm, *, withdraw=False):
    """Watch the release for settle_count ticks. A verified landing returns the arm and reports."""
    ctx.stage("Verifying placement")
    settling = Settling(plan, ctx, center_in_palm)
    strict = ctx.manipulation.target_or_refuse
    retracted = not strict
    for _ in range(ctx.settings.placement.settle_count):
        ctx.advance(plan.epoch, ctx.settings.placement.tick)
        stable = settling.update(ctx.observe(plan.epoch))
        if not retracted:
            if settling.open_hand.count >= ctx.settings.grasp.recovery_confirmations:
                require_empty(plan, ctx)
                try:
                    retract(plan, ctx)
                except (SkillFailure, ObservationError, ActionRejected) as error:
                    # Repeated release proof must survive a later arm failure.
                    empty = False
                    try:
                        if not isinstance(error, PhysicsInvalid):
                            require_empty(plan, ctx)
                            empty = True
                    except (SkillFailure, ObservationError, ActionRejected):
                        pass
                    if empty:
                        ctx.carried.pop(plan.side, None)
                    message = ("The hand opened and remains empty; retraction failed. The released object needs no "
                               "recovery; if withdrawal is blocked, end the task with done and say why. "
                               if empty else "The hand opened, but its current state is unverified; retraction failed. ")
                    code = (error.code if isinstance(error, SkillFailure) else
                            refusal_code(error) if isinstance(error, ActionRejected) else "TARGET_LOST")
                    details = error.details if isinstance(error, SkillFailure) else {}
                    raise SkillFailure(code, message + str(error),
                                       {**details, "hand": plan.side, "opening_executed": True,
                                        "released": empty, "hand_empty": empty,
                                        "landing_verified": bool(stable), "phase": "retraction"}) from error
                retracted = True
                settling = Settling(plan, ctx, center_in_palm)
            continue
        separated = settling.separated.count >= ctx.settings.grasp.recovery_confirmations
        if stable and (separated if strict else plan.side not in ctx.carried):
            failed = complete_return(plan, ctx, withdraw=withdraw, landing_verified=True)
            if failed is not None:
                return failed
            if plan.support:
                return placed(plan, ctx, "it settled on the support", landing_verified=True)
            return placed(plan, ctx, "it settled below the release point", landing_verified=True)
    if not retracted:
        raise SkillFailure("OUTCOME_UNKNOWN", "Fresh empty-hand evidence did not verify release; holding position")
    if not settling.seen:
        return unseen_release(plan, ctx, settling.open_hand.count, withdraw=withdraw)
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
        if not ctx.manipulation.target_or_refuse and count is not None and count >= settings.grasp.recovery_confirmations:
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


def unseen_release(plan, ctx, open_count, *, withdraw=False):
    """Verify a release whose object left the view from repeated open-hand evidence, else
    OUTCOME_UNKNOWN."""
    if open_count < ctx.settings.grasp.recovery_confirmations:
        raise SkillFailure("OUTCOME_UNKNOWN",
                           f"The target is not visible and fresh {plan.side}-hand evidence cannot verify release")
    failed = complete_return(plan, ctx, withdraw=withdraw)
    if failed is not None:
        if not ctx.manipulation.target_or_refuse:
            ctx.carried.pop(plan.side, None)
        return failed
    final = ctx.observe(plan.epoch).hands[plan.side]
    if final.closure > ctx.settings.placement.open_closure_limit or final.retained(ctx.settings.grasp):
        raise SkillFailure("OUTCOME_UNKNOWN",
                           f"The target is not visible and the returned {plan.side} hand may still retain it")
    ctx.carried.pop(plan.side, None)
    message = "the hand opened and is empty"
    if ctx.manipulation.target_or_refuse:
        message += "; landing is unverified because the object is out of view"
    return placed(plan, ctx, message, landing_verified=False)


def placed(plan, ctx, message, *, landing_verified):
    """An OK result saying what was verified."""
    short = f", {plan.short_m:.2f} m short of the requested point, as far as the arm reached" if plan.short_m else ""
    details = {"hand": plan.side, "short_of_point_m": round(plan.short_m, 3)}
    if ctx.manipulation.target_or_refuse:
        details.update(released=True, landing_verified=landing_verified)
    return Result("place", "OK", f"Released from the {plan.side} hand; {message}{short}.", details)


def complete_return(plan, ctx, *, withdraw=False, landing_verified=False):
    """After release verification, optionally back clear before lowering the arm to rest."""
    empty = False
    strict = ctx.manipulation.target_or_refuse
    try:
        if strict:
            require_empty(plan, ctx)
            empty = True
            ctx.carried.pop(plan.side, None)
            history = getattr(ctx, "navigation_history", None)
            if landing_verified and history is not None and plan.opened_at is not None:
                history.forget(plan.label, plan.opened_at)
        if withdraw:
            ctx.stage("Backing clear of the support")
            _, refused = ctx.navigation.back_out(ctx, plan.epoch)
            if refused is not None:
                raise SkillFailure("ACTION_REJECTED", "Withdrawal was refused; the empty arm remains raised",
                                   {"withdrawal": refused})
        return_arm(plan, ctx)
    except (SkillFailure, ObservationError, ActionRejected) as error:
        if not strict and not isinstance(error, SkillFailure):
            raise
        code = (error.code if isinstance(error, SkillFailure) else
                refusal_code(error) if isinstance(error, ActionRejected) else "TARGET_LOST")
        details = error.details if isinstance(error, SkillFailure) else {}
        action = "withdrawal or arm return" if withdraw else "arm return"
        unverified = not empty if strict else code == "OUTCOME_UNKNOWN"
        outcome = "Release unverified" if unverified else "Released"
        return Result("place", code, f"{outcome}; {action} refused: {error}",
                      {"hand": plan.side, **details,
                       **({"opening_executed": True, "released": empty, "hand_empty": empty,
                           "landing_verified": landing_verified, "phase": "withdrawal_or_return"}
                          if strict else {})})
    return None


def require_empty(plan, ctx, current=None):
    """Recheck current hand state before retracting or giving up uncertain carry ownership."""
    current = ctx.observe(plan.epoch) if current is None else current
    hand = current.hands[plan.side]
    if hand.closure > ctx.settings.placement.open_closure_limit or hand.retained(ctx.settings.grasp):
        raise SkillFailure("OUTCOME_UNKNOWN", "Current hand evidence cannot verify an empty hand")


_CORNERS = np.array(list(product((-1., 1.), repeat=3)))
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

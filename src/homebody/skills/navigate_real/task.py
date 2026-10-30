"""The REAL navigate skill: the checked controller and Navigator that REAL2SIM also uses, bound to
task Robot primitives for REAL deployment and its MuJoCo qualification."""
from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import asdict, dataclass, replace

import numpy as np

from homebody.helpers import volume
from homebody.helpers.geometry import inverse, points_in, unproject, wrap
from homebody.helpers.navigation import body_frame, free_space, run_in_point
from homebody.helpers.raster import world_to_px
from homebody.motion.carry import (
    compact_carry,
    continuing,
    robot_boxes,
    robot_part_names,
    sweep_radius,
)
from homebody.motion.exploration import gait
from homebody.motion.exploration.localizer import FRESH_POLL_S, FRESH_WAIT_S, PoseGate
from homebody.primitives.actions import PhysicsInvalid
from homebody.primitives.navigation.actions.lower_body import WZ_MIN_ACCEPT, LowerBody, clamp_lower
from homebody.primitives.navigation.context import Ctx, Deadline
from homebody.primitives.navigation.contracts import (
    DriveFailure,
    Refused,
    StaleObservation,
    UndeclaredNeed,
)
from homebody.primitives.navigation.errors import ActionRefused, MapObstacleRefused, Stopped
from homebody.primitives.navigation.observations.gripper_state import HELD, UNKNOWN, GripperState
from homebody.primitives.navigation.observations.map import MapArtifact
from homebody.primitives.navigation.task import NavigationEvidence
from homebody.primitives.observations import NavigationMap, ObservationError
from homebody.real2sim.exploration.settings import current as exploration_settings

from ..contract import Result, SkillFailure
from ..navigation_goal import goal_arguments, stance
from . import controller
from .navigation import (
    NavigationCapabilities,
    NavigationPort,
    NavigationState,
    Navigator,
)

COMMAND_HORIZON_S = .20
DOCK_LEVER_M = .60  # the farthest a docked part sits from the base
DOCK_LOOK_S = .10
NEEDS = ("base_velocity", "arm_target", "advance", "stop")
FAILURES = ("NO_ROUTE", "STALL", "TIMEOUT", "ARRIVAL_NOT_VERIFIED", "CARRY_CLEARED", "HOLD_UNCERTAIN",
            "FOOTPRINT_UNAVAILABLE")
PROMPT = ('navigate: goal_map_1000 and facing_map_1000, [x, y] on image 3: goal, the floor where to '
          'stand; facing, the spot to work on next: the object to pick, or where the held object should land, '
          'inside a surface rather than at its edge. goal_xy_m and facing_xy_m take world [x, y] instead. '
          'With snap (default true) the robot stands square to the mapped edge on the goal side of the '
          'facing point, at least {motion.stance_reach:g} m from it, straight ahead or in front of hand '
          '("left" or "right") when given; with snap false it stands exactly at goal, turned toward facing. '
          'A held object is carried low in front of the body and sweeps across the surface the robot walks '
          'up to, and can push objects already on it.')


@dataclass(frozen=True)
class Plan:
    epoch: int
    goal: np.ndarray
    yaw: float | None
    waypoints: list
    requested_goal: np.ndarray
    radius_m: float
    execution_radius_m: float
    payload_signature: str
    frame_id: str
    map_hash: str
    generation: object
    departure_goal: object = None
    run_in: object = None


def payload_signature(carried, *, include_state=True):
    """Only changes to ownership, certainty or measured payload geometry invalidate a plan."""
    h = hashlib.sha256()
    for side, record in sorted(carried.items()):
        h.update(repr((side, record.get("label"), record.get("state") if include_state else None)).encode())
        for key in ("palm_to_payload", "payload_half"):
            h.update(np.asarray(record.get(key, ()), dtype=float).tobytes())
    return h.hexdigest()


def _column_binding(frame):
    """Acquisition provenance; unregistered frames cannot contribute historical support."""
    meta = frame.sensor_metadata or {}
    if (meta.get("registered_pose") is not True or "navigation_generation" not in meta
            or not meta.get("map_frame") or not meta.get("map_sha256")
            or not meta.get("calibration_hash")):
        return None
    calibration = hashlib.sha256()
    calibration.update(frame.intrinsics.tobytes())
    calibration.update(str(frame.depth.shape).encode())
    calibration.update(str(meta["calibration_hash"]).encode())
    calibration.update(str(meta.get("camera_identity", "")).encode())
    calibration.update(frame.navigation.T_map_px.tobytes())
    calibration.update(repr((frame.navigation.occupied.shape, frame.navigation.resolution)).encode())
    return (frame.epoch, meta["map_frame"], meta["map_sha256"],
            meta["navigation_generation"], calibration.hexdigest())


def observe_frame(ctx, frame):
    """Keep already accepted metric observations across pick and navigation, without a capture."""
    binding = _column_binding(frame)
    if binding is None:
        ctx.navigation_history = None
        return
    if ctx.navigation_history is None:
        seconds = ctx.settings.collision.column_seconds
        ctx.navigation_history = volume.ColumnHistory(
            seconds=seconds, period=DOCK_LOOK_S,
            max_frames=math.ceil(seconds / DOCK_LOOK_S) + 1)
    sampled = np.zeros(frame.depth.shape, bool)
    stride = max(4, int(np.ceil(np.sqrt(frame.depth.size / 20000))))
    while np.prod(np.ceil(np.asarray(frame.depth.shape) / stride)) > 20000:
        stride += 1
    sampled[::stride, ::stride] = True
    valid = sampled & (frame.labels >= 0) & np.isfinite(frame.depth) & (frame.depth > 0)
    points = points_in(frame.world_camera, unproject(frame.depth, frame.intrinsics, valid))
    labels = frame.labels[valid]
    keep = np.linalg.norm(points[:, :2] - frame.base_pose[:2], axis=1) <= 2 * DOCK_LEVER_M
    own = robot_boxes(frame, ctx.arms, ctx.carried)
    keep &= volume.outside(points, own, ctx.settings.collision.column_margin)
    acquired = min(frame.time, (frame.sensor_metadata or {}).get("depth_time", frame.time))
    held = tuple(record["label"] for record in ctx.carried.values()
                 if record.get("state") == "verified" and record.get("epoch") == frame.epoch
                 and record.get("label", 0) > 0)
    ctx.navigation_history.add(acquired, binding, points[keep], labels[keep], held_labels=held)


def footprint(frame, planning, look=None):
    """Validate measured geometry; loaded travel checks its base circle and parts separately."""
    limit = planning.manipulation.max_carried_objects
    if limit is not None and len(planning.carried) > limit:
        raise SkillFailure("HOLD_UNCERTAIN", "Too many unresolved carried objects for this navigation policy")
    if any(side not in planning.carried and hand.closure > planning.settings.placement.open_closure_limit
           for side, hand in frame.hands.items()):
        raise SkillFailure("HOLD_UNCERTAIN", "A closed hand has no verified payload record; its travel envelope is unknown")
    for side, record in planning.carried.items():
        if side not in planning.arms or record.get("palm_to_payload") is None or record.get("payload_half") is None:
            raise SkillFailure("FOOTPRINT_UNAVAILABLE", "The carried object's measured extent is missing")
        if not continuing(frame, planning.arms[side], side, record, planning.settings.grasp, look=look):
            raise SkillFailure("HOLD_UNCERTAIN", "Navigation requires current verified carry evidence")
        transform, half = np.asarray(record["palm_to_payload"]), np.asarray(record["payload_half"])
        if (transform.shape != (4, 4) or half.shape != (3,)
                or not np.isfinite(transform).all() or not np.isfinite(half).all() or np.any(half < 0)):
            raise SkillFailure("FOOTPRINT_UNAVAILABLE", "The retained payload extent is not valid measured geometry")
    radius = sweep_radius(frame, planning.arms, planning.carried, planning.settings.motion.robot_radius)
    if planning.carried:
        radius = max(.24, planning.settings.motion.robot_radius)
    execution = radius + planning.settings.collision.cloud_margin
    planning_radius = max(.30, execution + .03)
    return planning_radius, execution


def _failure(error):
    code = getattr(error, "code", "ACTION_REJECTED")
    result = {"NO_COLLISION_FREE_PATH": "NO_ROUTE", "NO_PROGRESS": "STALL",
              "ARRIVAL_NOT_VERIFIED": "ARRIVAL_NOT_VERIFIED", "TIMEOUT": "TIMEOUT"}.get(code, "ACTION_REJECTED")
    cause_details = getattr(getattr(error, "__cause__", None), "details", {})
    return SkillFailure(result, str(error), {**cause_details, "navigation_code": code,
                                           **getattr(error, "details", {})})


def prepare(arguments, selected, current, planning, reach=None):
    """Resolve selected-map coordinates and admit against current measured geometry, without motion."""
    if planning.navigation_evidence is None:
        raise SkillFailure("ACTION_REJECTED", "Checked navigation requires declared localization and obstacle evidence")
    if planning.settings.physics.control_hz != gait.TICK_HZ:
        raise SkillFailure("ACTION_REJECTED", f"Checked navigation streams one lower_body command per "
                           f"{1 / gait.TICK_HZ:g} s; physics.control_hz must be {gait.TICK_HZ:g}")
    if (not np.array_equal(selected.navigation.T_map_px, current.navigation.T_map_px)
            or selected.navigation.occupied.shape != current.navigation.occupied.shape):
        raise SkillFailure("STALE_OBSERVATION", "The selected navigation-map transform changed; select again")
    bridge = _Bridge(planning, current)
    sample = bridge.sample()
    radius, execution = footprint(current, planning)
    requested, facing, yaw, snap = goal_arguments(arguments, selected.navigation)
    settings = planning.settings.motion
    if facing is not None and np.linalg.norm(facing - requested) <= settings.facing_min_distance:
        facing = None
    goal, yaw = stance(arguments, sample.navigation, free_space(sample.navigation, radius),
        requested, facing, yaw, snap, sample.base_pose[:2], settings,
        settings.stance_reach if reach is None else reach)
    bridge.radius = execution
    bridge.admitted_radius = radius
    ctx = bridge.context()
    route_arguments = {"goal_xy_m": goal.tolist(), "snap": False,
        "preserve_goal": True, "arrival_heading_required": False, "route_rejoin": True,
        "preferred_clearance": False, "radius_m": radius, "planning_radius_m": radius,
        "execution_radius_m": execution, "arrival_tolerance_m": settings.arrival_distance}
    try:
        prepared = controller.prepare(ctx, route_arguments, cancelled=planning.cancelled)
        waypoints = prepared["route"].path or []
    except DriveFailure as error:
        checker, _ = gait.occupancy_grid(bridge.artifact(), radius, exact_footprint=True)
        if (error.code != "NO_COLLISION_FREE_PATH" or checker.is_safe(*sample.base_pose[:2])
                or not checker.is_safe(*goal)):
            raise _failure(error) from error
        waypoints = []  # execution owns checked departure recovery, never a blind backup
    approach = run_in_point(sample.navigation, free_space(sample.navigation, radius),
                            goal, yaw, sample.base_pose[:2], settings)
    if approach is not None:
        try:
            prepared = controller.prepare(ctx, {**route_arguments, "goal_xy_m": approach.tolist(),
                                                "arrival_tolerance_m": settings.waypoint_radius},
                                          cancelled=planning.cancelled)
        except DriveFailure as error:
            if error.code != "NO_COLLISION_FREE_PATH":
                raise _failure(error) from error
            approach = None
        else:
            waypoints = prepared["route"].path or []
    return Plan(current.epoch, goal, yaw, list(waypoints), requested, radius, execution,
                payload_signature(planning.carried), sample.frame_id, sample.map_hash, sample.generation,
                run_in=approach)


def _navigation_runtime(plan, ctx, deadline=None):
    bridge = _Bridge(ctx.planning(), ctx.observe(plan.epoch), ctx, plan)
    bridge.check_binding(bridge.sample())
    bridge.radius = plan.execution_radius_m
    live = bridge.context()
    if deadline is not None:
        live.deadline = deadline
    caps = NavigationCapabilities(plan.execution_radius_m, plan.radius_m, (plan.radius_m,),
                                  exploration_settings().drive.final_turn_drift_m, 3.)
    port = NavigationPort(caps, lambda: plan.epoch, bridge.guard, bridge.fresh, bridge.hold,
        lambda stage, text: ctx.stage(text), lambda radius: max(radius, plan.radius_m),
        lambda: (bridge.artifact(), bridge.sample().navigation.occupied, bridge.sample().generation),
        lambda: bridge.now() - bridge.sample().map_time, bridge.stationary,
        bridge.motion_refusal, ctx.cancelled, lambda: bridge.fresh(live, plan.epoch, require_map=True), True)
    if ctx.navigation_state is None:
        ctx.navigation_state = NavigationState()
    return bridge, live, Navigator(port, ctx.navigation_state)


def _compact_after_departure(plan, ctx, bridge, live):
    """One held, collision-checked carry retry, followed by same-goal fresh preparation."""
    bridge.hold(live)
    bridge.fresh(live, plan.epoch, require_map=True)
    sample = bridge.sample()
    checker = gait.occupancy_grid(bridge.artifact(), plan.radius_m, exact_footprint=True)[0]
    if not checker.is_safe(*sample.base_pose[:2]):
        raise SkillFailure("ACTION_REJECTED", "Carry compaction requires a retained clear departure footprint")
    deadline = live.deadline
    binding = _Bridge(ctx.planning(), bridge.measure(), ctx)
    binding._navigation_binding = (plan.frame_id, plan.map_hash, plan.generation)
    geometry = payload_signature(ctx.carried, include_state=False)
    def check_navigation():
        binding.sample(force=True)
        if payload_signature(ctx.carried, include_state=False) != geometry:
            raise SkillFailure("STALE_EPOCH", "Carry ownership or geometry changed during compaction")
    bounded = replace(ctx, cancelled=lambda: ctx.cancelled() or deadline.expired(),
                      authority=check_navigation)
    bounded._skill_name = ctx._skill_name
    try:
        bridge.log("carry.departure_verified", {"pose": sample.base_pose.tolist(),
                   "planning_radius_m": plan.radius_m, "remaining_s": deadline.remaining()})
        for side in tuple(ctx.carried):
            if ctx.carried[side].get("compact_pending"):
                bounded.check(plan.epoch)
                compact_carry(bounded, plan.epoch, side)
        bounded.check(plan.epoch)
        current = bounded.observe(plan.epoch)
        fresh = prepare({"goal_xy_m": plan.goal.tolist(), "yaw": plan.yaw, "snap": False},
                        current, current, bounded.planning())
        bounded.check(plan.epoch)
        if not np.array_equal(fresh.goal, plan.goal):
            raise SkillFailure("ACTION_REJECTED", "Carry refresh changed the accepted destination")
        fresh = replace(fresh, requested_goal=plan.requested_goal)
        ctx.emit({"type": "navigation", "kind": "carry.reprepared", "details": {
            "goal_xy": fresh.goal.tolist(), "planning_radius_m": fresh.radius_m,
            "execution_radius_m": fresh.execution_radius_m, "remaining_s": deadline.remaining(),
            "payload_signature": fresh.payload_signature}})
        return fresh
    except SkillFailure as error:
        if deadline.expired() and not ctx.cancelled():
            raise SkillFailure("TIMEOUT", "Navigation deadline expired during carry compaction or preparation") from error
        raise


def _fold_empty_departure(plan, ctx, bridge, live):
    """Restore an empty robot's departure by changing its checked arm geometry, not clearance."""
    if (not ctx.manipulation.target_or_refuse or ctx.carried or
            set(ctx.arms) != {"left", "right"}):
        return plan, None
    sample = bridge.sample(force=True)
    checker = gait.occupancy_grid(bridge.artifact(), plan.radius_m, exact_footprint=True)[0]
    # A clear destination does not restore departure from the blocked measured pose.
    if checker.is_safe(*sample.base_pose[:2]):
        return plan, None
    from . import posture

    current, planning = ctx.observe(plan.epoch), ctx.planning()
    artifact = bridge.artifact()

    def admitted(candidate):
        radius, _ = footprint(candidate, planning)
        return gait.occupancy_grid(artifact, radius, exact_footprint=True)[0].is_safe(
            *sample.base_pose[:2])

    deadline = live.deadline

    def check():
        ctx.check(plan.epoch)
        if deadline.expired():
            raise SkillFailure("TIMEOUT", "Navigation deadline expired during checked arm folding")

    fold = posture.prepare(current, planning, sample, admitted=admitted, check=check)
    if fold is None:
        return plan, None
    binding = _Bridge(planning, current, ctx)
    binding._navigation_binding = (plan.frame_id, plan.map_hash, plan.generation)

    def validate_evidence():
        if deadline.expired():
            raise SkillFailure("TIMEOUT", "Navigation deadline expired during checked arm folding")
        if ctx.carried:
            raise SkillFailure("STALE_EPOCH", "Carry ownership changed during empty-arm folding")
        return binding.sample(force=True)

    bounded = replace(ctx, authority=validate_evidence)
    bounded._skill_name = ctx._skill_name
    ctx.emit({"type": "navigation", "kind": "posture.prepared", "details": {
        "goal_xy": plan.goal.tolist(), "planning_radius_m": plan.radius_m,
        "remaining_s": deadline.remaining()}, "plan": fold})
    settled = posture.execute(bounded, fold, validate_evidence=validate_evidence, check=check)
    check()
    radius, execution = footprint(settled, bounded.planning())
    details = {"before_planning_radius_m": plan.radius_m, "planning_radius_m": radius,
               "execution_radius_m": execution, "pose": settled.base_pose.tolist(),
               "arrived": False, "remaining_s": deadline.remaining()}
    ctx.emit({"type": "navigation", "kind": "posture.settled", "details": details})
    try:
        fresh = prepare({"goal_xy_m": plan.goal.tolist(), "yaw": plan.yaw, "snap": False},
                        settled, settled, bounded.planning())
    except SkillFailure as error:
        if deadline.expired() and not ctx.cancelled():
            raise SkillFailure("TIMEOUT", "Navigation deadline expired while refreshing the folded departure",
                               {"departure_posture": details, "arrived": False}) from error
        raise SkillFailure(error.code, str(error), {**error.details,
            "departure_posture": details, "arrived": False}) from error
    check()
    if not np.array_equal(fresh.goal, plan.goal):
        raise SkillFailure("ACTION_REJECTED", "Empty-arm refresh changed the accepted destination")
    return replace(fresh, requested_goal=plan.requested_goal), details


def execute(plan, ctx):
    """Translate through the shared controller, then independently check the facing turn."""
    ctx.check(plan.epoch)
    if payload_signature(ctx.carried) != plan.payload_signature:
        raise SkillFailure("STALE_EPOCH", "Carry ownership or geometry changed; prepare navigation again")
    try:
        bridge, live, navigator = _navigation_runtime(plan, ctx)
    except ObservationError as error:
        raise SkillFailure("STALE_OBSERVATION", str(error)) from error
    bridge.log("navigation.settings", {group: asdict(getattr(exploration_settings(), group))
                                       for group in ("drive", "gait")})
    deadline = live.deadline
    physics_invalid = False
    try:
        refreshed, departure_posture = _fold_empty_departure(plan, ctx, bridge, live)
        if refreshed is not plan:
            plan = refreshed
            bridge, live, navigator = _navigation_runtime(plan, ctx, deadline)
        if any(record.get("compact_pending") for record in ctx.carried.values()):
            sample = bridge.sample()
            checker = gait.occupancy_grid(bridge.artifact(), plan.radius_m, exact_footprint=True)[0]
            if not checker.is_safe(*sample.base_pose[:2]):
                navigator.drive(live, plan.goal.tolist(), None, plan.radius_m,
                    requested_radius_m=plan.radius_m, arrival_tolerance_m=ctx.settings.motion.arrival_distance,
                    stop_after_departure=True)
            plan = _compact_after_departure(plan, ctx, bridge, live)
            bridge, live, navigator = _navigation_runtime(plan, ctx, deadline)
        route_goal = plan.goal if plan.run_in is None else plan.run_in
        # Run-in hands off to the checked final stance; it is not the final arrival.
        route_tolerance = (ctx.settings.motion.arrival_distance if plan.run_in is None
                           else ctx.settings.motion.waypoint_radius)
        navigator.drive(live, route_goal.tolist(), None, plan.radius_m,
                        requested_radius_m=plan.radius_m,
                        arrival_tolerance_m=route_tolerance)
        if plan.run_in is not None:
            bridge.hold(live)
        correction = None
        if plan.yaw is not None:
            bridge.fresh(live, plan.epoch, require_map=True)
            ctx.stage("Facing the target")
            if ctx.carried or plan.run_in is not None:
                # Hold the checked stance through the strict servo: a loaded yaw turn can
                # translate substantially, and a prepared run-in aims before its final close.
                initial = bridge.sample(force=True)
                # A prepared approach owns a checked final leg, not a larger drift allowance.
                if (plan.run_in is not None or math.dist(initial.base_pose[:2], plan.goal)
                        <= exploration_settings().drive.final_turn_drift_m):
                    _, correction = _correct_settled_arrival(plan, ctx, bridge, live, navigator, initial)
            else:
                gait.final_turns(live, "drive.facing", lambda x, y: plan.yaw,
                                 gate_rad=ctx.settings.motion.arrival_angle)
        if correction is None or "skipped" in correction:
            bridge.hold(live)
        final = bridge.sample(force=True)
        distance = math.dist(final.base_pose[:2], plan.goal)
        angle = 0. if plan.yaw is None else abs(float(wrap(final.base_pose[2] - plan.yaw)))
        if (correction is None and distance <= exploration_settings().drive.final_turn_drift_m
                and (distance > ctx.settings.motion.arrival_distance
                     or angle > ctx.settings.motion.arrival_angle)):
            final, correction = _correct_settled_arrival(plan, ctx, bridge, live, navigator, final)
            distance = math.dist(final.base_pose[:2], plan.goal)
            angle = 0. if plan.yaw is None else abs(float(wrap(final.base_pose[2] - plan.yaw)))
        if distance > ctx.settings.motion.arrival_distance or angle > ctx.settings.motion.arrival_angle:
            raise SkillFailure("ARRIVAL_NOT_VERIFIED", "Current settled pose did not verify requested position and facing",
                               {"pose": final.base_pose.tolist(), "goal_xy": plan.goal.tolist(), "yaw": plan.yaw,
                                "distance_m": distance, "angle_rad": angle,
                                "arrival_distance_m": ctx.settings.motion.arrival_distance,
                                "arrival_angle_rad": ctx.settings.motion.arrival_angle,
                                "verification_phase": "settled_pose",
                                **({"correction": correction} if correction is not None else {})})
        if ctx.carried:
            measured = bridge.measure(force=True)
            execution = bridge.measured_execution_radius()
            checker = gait.occupancy_grid(bridge.artifact(final), execution, exact_footprint=True,
                                         obstacles=final.navigation.occupied)[0]
            checks = {name: checker.check_motion(pose[:2], pose[:2], allow_egress=False)
                      for name, pose in (("current", final.base_pose), ("measured", measured.base_pose))}
            full = gait.occupancy_grid(bridge.artifact(final), plan.radius_m, exact_footprint=True)[0]
            checks["goal"] = full.check_motion(plan.goal, plan.goal, allow_egress=False)
            if not all(check["safe"] for check in checks.values()):
                raise SkillFailure("ARRIVAL_NOT_VERIFIED", "Current settled base footprint is blocked",
                                   {"footprint": checks, "verification_phase": "settled_footprint"})
            # Zero commands must stay admissible for stopping: admit the current parts
            # under the full-history rules, including initial-volume exemptions.
            bridge._motion_check(0., 0., 0.)
            bridge.guard()
            if live.deadline.expired():
                raise SkillFailure("TIMEOUT", "Original navigation deadline expired during arrival verification")
            require_navigation_freshness(final, bridge.now())
            if (not 0 <= bridge.now() - measured.time <= ctx.settings.motion.observation_age
                    or not bridge._dock_checked_at <= bridge.now() <= bridge._dock_ray_expiry):
                raise SkillFailure("STALE_OBSERVATION", "Measured component evidence expired during arrival verification")
            bridge.log("drive.arrival_verified", {"pose_at": final.pose_time, "measured_at": measured.time,
                "verified_at": bridge.now(), "distance_m": distance, "angle_rad": angle,
                "stationary_components_admitted": True})
        else:
            measured = bridge.measure(force=True)
            radius, execution = footprint(measured, bridge.planning)
            artifact = bridge.artifact(final)
            checker = gait.occupancy_grid(artifact, radius, exact_footprint=True)[0]
            current = checker.check_motion(final.base_pose[:2], final.base_pose[:2], allow_egress=False)
            body = checker.check_motion(measured.base_pose[:2], measured.base_pose[:2], allow_egress=False)
            blocked = not current["safe"] or not body["safe"]
            if blocked:
                motion_checker = gait.occupancy_grid(artifact, execution, exact_footprint=True)[0]
                checks = {"current": current, "measured": body,
                          "goal": checker.check_motion(plan.goal, plan.goal, allow_egress=False),
                          "execution": motion_checker.check_motion(
                              final.base_pose[:2], final.base_pose[:2], allow_egress=False)}
            bridge.guard()
            if live.deadline.expired():
                raise SkillFailure("TIMEOUT", "Original navigation deadline expired during arrival verification")
            require_navigation_freshness(final, bridge.now())
            if not 0 <= bridge.now() - measured.time <= ctx.settings.motion.observation_age:
                raise SkillFailure("STALE_OBSERVATION", "Measured geometry expired during arrival verification")
            if blocked:
                raise SkillFailure("ARRIVAL_NOT_VERIFIED", "Current settled footprint does not retain planning clearance",
                    {"pose": final.base_pose.tolist(), "goal_xy": plan.goal.tolist(),
                     "measured_pose": measured.base_pose.tolist(), "measured_at": measured.time,
                     "pose_at": final.pose_time,
                     "distance_m": distance, "angle_rad": angle, "planning_radius_m": radius,
                     "execution_radius_m": execution, "footprint": checks, "carried_hands": [],
                     "verification_phase": "settled_footprint"})
        return Result("navigate", "OK", "Checked translation and settled facing verified",
                      {"goal_xy": plan.goal.tolist(), "pose": final.base_pose.tolist(),
                       "planning_radius_m": plan.radius_m, "execution_radius_m": plan.execution_radius_m,
                       "navigation": navigator.state._last_drive_result,
                       "departure_posture": departure_posture,
                       **({"correction": correction} if correction is not None else {})})
    except PhysicsInvalid:
        physics_invalid = True
        raise
    except (DriveFailure, ActionRefused) as error:
        raise _failure(error) from error
    except (Stopped, StaleObservation, Refused) as error:
        raise SkillFailure("ACTION_REJECTED", str(error), getattr(error, "details", {})) from error
    except ObservationError as error:
        raise SkillFailure("STALE_OBSERVATION", str(error)) from error
    finally:
        if not physics_invalid and ctx.observations.epoch == plan.epoch:
            ctx.actions.base_velocity(plan.epoch, 0., 0., 0.)


def _correct_settled_arrival(plan, ctx, bridge, live, navigator, initial):
    """One strict same-target correction; never an egress or a new recovery attempt."""
    motion, grids = ctx.settings.motion, {}
    details = {"before_pose": initial.base_pose.tolist(),
               "before_distance_m": math.dist(initial.base_pose[:2], plan.goal),
               "goal_xy": plan.goal.tolist(), "remaining_s": live.deadline.remaining()}
    previous = initial.base_pose[:2].copy()

    def update_grids(sample, execution):
        nav = sample.navigation
        if ctx.carried:
            key = (sample.frame_id, sample.map_hash, sample.generation, nav.resolution)
            if (grids.get("key") != key or
                    not np.array_equal(grids.get("occupied"), nav.occupied) or
                    not np.array_equal(grids.get("transform"), nav.T_map_px)):
                grids.update(key=key, occupied=nav.occupied, transform=nav.T_map_px,
                    full=gait.occupancy_grid(bridge.artifact(), plan.radius_m, exact_footprint=True)[0])
            full = grids["full"]
            # This private checker serves exact sweeps only. No stale-radius A* graph
            # is exposed; the current measured radius participates in every query.
            grids["motion"] = replace(full, grid=None, radius_baked=execution, obstacles=full.occupied)
        else:
            key = (sample.map_hash, sample.generation, execution)
            if grids.get("key") != key or not np.array_equal(grids.get("occupied"), nav.occupied):
                art = bridge.artifact()
                grids.update(key=key, occupied=nav.occupied,
                    full=gait.occupancy_grid(art, plan.radius_m, exact_footprint=True)[0],
                    motion=gait.occupancy_grid(art, execution, exact_footprint=True,
                                               obstacles=nav.occupied)[0])
        return grids["motion"]

    def checked_sample(*, after_tick=False):
        bridge.guard(plan.epoch)
        if not after_tick and live.deadline.expired():
            raise SkillFailure("TIMEOUT", "Original navigation deadline expired during stance correction")
        sample = bridge.sample(force=True)
        # A loaded stance checks each available hand sample before another command;
        # ordinary route control uses the bridge's measurement cadence.
        if ctx.carried:
            bridge.measure(force=True)
        # Same measured geometry owner; no prospective refusal can hide a measured failure.
        execution = bridge.measured_execution_radius()
        update_grids(sample, execution)
        return sample, execution

    def refuse(message, check):
        bridge.log("drive.arrival_correction_refused", {"reason": message, **check})
        raise ActionRefused(message, code="ACTION_REJECTED", details=check)

    def check_current(sample, *, optional=False, correcting=False):
        xy = sample.base_pose[:2]
        # Keep planning clearance at the accepted goal; executed motion uses the
        # measured execution footprint, as route following and landing do.
        goal = grids["full"].check_motion(plan.goal, plan.goal, allow_egress=False)
        start = grids["motion"].check_motion(xy, xy, allow_egress=False)
        segment = (grids["motion"].check_motion(xy, plan.goal, allow_egress=False)
                   if correcting else None)
        if not goal["safe"] or not start["safe"] or (segment is not None and not segment["safe"]):
            check = {"goal": goal, "start": start, "segment": segment, "arrived": False}
            if optional:
                return check
            refuse("The strict current footprint or direct correction segment is blocked", check)
        return None

    sample, _ = checked_sample()
    at_goal = (math.dist(sample.base_pose[:2], plan.goal) <= motion.arrival_distance
               and (plan.yaw is None or abs(float(wrap(sample.base_pose[2] - plan.yaw))) <= motion.arrival_angle))
    unavailable = check_current(sample, optional=True, correcting=not at_goal)
    if unavailable is not None:
        details.update(skipped={"reason": "strict_correction_unavailable", **unavailable},
                       current_pose=sample.base_pose.tolist())
        bridge.log("drive.arrival_correction_skipped", details)
        return initial, details
    previous = sample.base_pose[:2].copy()

    def check_measured(sample, execution, *, correcting=False):
        nonlocal previous
        xy = sample.base_pose[:2]
        sweep = grids["motion"].check_motion(previous, xy, allow_egress=False)
        if not sweep["safe"]:
            check = {"previous_xy": previous.tolist(), "xy": xy.tolist(),
                     "yaw": float(sample.base_pose[2]), "remaining_m": math.dist(xy, plan.goal),
                     "radius_m": plan.radius_m, "execution_radius_m": execution,
                     "failed_stage": "sweep", "allow_egress": False, "sweep": sweep}
            refuse("Measured stance correction crossed its strict footprint",
                   {"recovery_check": check, "arrived": False})
        previous = xy.copy()
        check_current(sample, correcting=correcting)

    def checked_step(channel, command):
        sample, execution = checked_sample()
        check_measured(sample, execution, correcting=bool(command.vx or command.vy or command.wz))
        if live.deadline.expired():
            raise SkillFailure("TIMEOUT", "Original navigation deadline expired during correction admission")
        applied = live.step(channel, command)
        sample, execution = checked_sample(after_tick=True)
        check_measured(sample, execution)
        if live.deadline.expired():
            raise SkillFailure("TIMEOUT", "Original navigation deadline expired during stance correction")
        return applied

    checked = replace(live, step=checked_step)
    bridge.log("drive.arrival_correction", details)
    ctx.stage("Correcting the settled stance")
    previous_checker = bridge._stance_checker
    try:
        if ctx.carried:
            bridge._stance_checker = update_grids
        _stance_to(bridge, checked, plan.goal, plan.yaw, motion, arrival=True)
    finally:
        bridge._stance_checker = previous_checker
    final, execution = checked_sample()
    check_measured(final, execution)
    details.update(after_pose=final.base_pose.tolist(),
                   after_distance_m=math.dist(final.base_pose[:2], plan.goal),
                   after_remaining_s=live.deadline.remaining())
    return final, details


def check_held_footprint(plan, ctx, frame):
    """Admit a held frame under the plan's map binding and complete measured circle; no route."""
    ctx.check(plan.epoch)
    if frame.epoch != plan.epoch:
        raise SkillFailure("STALE_EPOCH", "Held grasp frame belongs to a different epoch")
    bridge = _Bridge(ctx.planning(), frame, ctx, plan)
    sample = bridge.sample(force=True)
    radius, execution = footprint(frame, ctx.planning())
    checker = gait.occupancy_grid(bridge.artifact(), radius, exact_footprint=True)[0]
    if not checker.is_safe(*sample.base_pose[:2]) or not checker.is_safe(*frame.base_pose[:2]):
        raise SkillFailure("NO_ROUTE", "The held measured footprint is not clear for grasp re-admission")
    ctx.check(plan.epoch)
    require_navigation_freshness(sample, ctx.observations.time)
    if not 0 <= ctx.observations.time - frame.time <= ctx.settings.motion.observation_age:
        raise SkillFailure("STALE_OBSERVATION", "Held grasp geometry expired during admission")
    return {"frame_id": sample.frame_id, "map_hash": sample.map_hash, "generation": sample.generation,
            "pose": sample.base_pose.tolist(), "planning_radius_m": radius,
            "execution_radius_m": execution}


def refine(plan, ctx):
    """Close onto the planned stance more tightly than arrival verification requires, for a
    manipulation whose reach needs it: turn onto the stance heading holding position, walk
    straight onto the stance holding the heading, and settle. Every command passes the same
    footprint, freshness and authority checks; a refused command ends the close without
    further motion."""
    ctx.check(plan.epoch)
    if payload_signature(ctx.carried) != plan.payload_signature:
        raise SkillFailure("STALE_EPOCH", "Carry ownership or geometry changed; prepare navigation again")
    try:
        bridge, live, _ = _navigation_runtime(plan, ctx)
    except ObservationError as error:
        raise SkillFailure("STALE_OBSERVATION", str(error)) from error
    motion = ctx.settings.motion
    physics_invalid, refused = False, None
    try:
        try:
            _stance_to(bridge, live, plan.goal, plan.yaw, motion)
        except DriveFailure as error:
            if not isinstance(error.__cause__, MapObstacleRefused):
                raise
            refused = error.__cause__.details
            gait.hold(live, motion.settle_seconds, 0., 0., 0.)
        pose = bridge.sample(force=True).base_pose
        details = {"pose": pose.tolist(), "goal_xy": plan.goal.tolist(), "yaw": plan.yaw,
                   "distance_m": math.dist(pose[:2], plan.goal),
                   "angle_rad": None if plan.yaw is None else abs(float(wrap(plan.yaw - pose[2]))),
                   "refused_command": refused}
        bridge.log("stance.refined", details)
        return Result("navigate", "OK", "Closed onto the stance" if refused is None else
                      "The measured footprint stopped the close", details)
    except PhysicsInvalid:
        physics_invalid = True
        raise
    except (DriveFailure, ActionRefused) as error:
        raise _failure(error) from error
    except (Stopped, StaleObservation, Refused) as error:
        raise SkillFailure("ACTION_REJECTED", str(error)) from error
    except ObservationError as error:
        raise SkillFailure("STALE_OBSERVATION", str(error)) from error
    finally:
        if not physics_invalid and ctx.observations.epoch == plan.epoch:
            ctx.actions.base_velocity(plan.epoch, 0., 0., 0.)


def dock_floor_clear(ctx, current, ahead, turn=0.):
    """Admit the base path to a candidate docking endpoint without motion or a new task frame.

    Use CURRENT's pose for the candidate, with fresh filtered navigation evidence and docking's
    configured circle. A turn needs a clear start; straight motion retains checked egress.
    This does not certify stopping travel beyond the endpoint or arm, finger and payload sweeps.
    """
    ctx.check(current.epoch)
    if not 0 <= ctx.observations.time - current.time <= ctx.settings.motion.observation_age:
        raise SkillFailure("STALE_OBSERVATION", "The docking candidate frame is stale")
    planning = ctx.planning()
    if planning.navigation_evidence is None:
        raise SkillFailure("ACTION_REJECTED", "Checked navigation requires declared localization and obstacle evidence")
    bridge = _Bridge(planning, current, ctx)
    bridge.docking = True
    sample = bridge.sample()
    if (not np.array_equal(current.navigation.T_map_px, sample.navigation.T_map_px)
            or current.navigation.occupied.shape != sample.navigation.occupied.shape):
        raise SkillFailure("STALE_OBSERVATION", "The docking candidate map frame changed")
    checker = bridge._dock_floor_checker(sample)
    x, y, yaw = current.base_pose
    end = (x + ahead * math.cos(yaw + turn), y + ahead * math.sin(yaw + turn))
    return ((not turn or checker.is_safe(x, y))
            and checker.check_motion((x, y), end, allow_egress=True)["safe"])


def dock(ctx, epoch, ahead, turn=0.):
    """Turn in place by TURN, then walk straight AHEAD metres (negative backs up) along the new
    heading, under the docking checks: the floor footprint against every occupied cell, and arm,
    finger and payload boxes over occupied cells only where observed columns clear them. A refused
    command stops the docking where it is. Returns the settled pose and the refusal's details, or
    None."""
    def walk(bridge, live):
        pose = bridge.sample(force=True).base_pose
        yaw = pose[2] + turn
        if turn:
            bridge.docking = True
            try:
                _turn_to(bridge, live, yaw, ctx.settings.motion)
            except DriveFailure as error:
                if not isinstance(error.__cause__, MapObstacleRefused):
                    raise
                return error.__cause__.details
            finally:
                bridge.docking = False
        target = pose[:2] + ahead * np.array([math.cos(yaw), math.sin(yaw)])
        return _dock_walk(bridge, live, target, yaw, ctx.settings.motion)
    return _docking(ctx, epoch, walk)


def back_out(ctx, epoch):
    """Back straight out under the docking checks until the measured footprint's planning radius
    admits the pose. No motion when it already does. Returns the settled pose and the refusal's
    details, or None."""
    def walk(bridge, live):
        radius, _ = footprint(ctx.observe(epoch), ctx.planning())
        return _back_out(ctx, bridge, live, radius)
    pose, refused = _docking(ctx, epoch, walk)
    if refused is None:
        current = ctx.measure(epoch)
        bridge = _Bridge(ctx.planning(), current, ctx)
        pose = bridge.sample(force=True).base_pose
        radius, _ = footprint(current, ctx.planning())
        checker = gait.occupancy_grid(bridge.artifact(), radius, exact_footprint=True)[0]
        if not checker.is_safe(*pose[:2]):
            refused = {"reason": "Settled withdrawal footprint remains blocked",
                       "pose": pose.tolist(), "planning_radius_m": radius}
        bridge.log("dock.departure_checked", {"pose": pose.tolist(), "planning_radius_m": radius,
                                              "safe": refused is None})
    return pose, refused


def _docking(ctx, epoch, walk):
    """Run WALK(bridge, live) on a bridge bound to the current navigation, then stand and measure."""
    ctx.check(epoch)
    physics_invalid = False
    try:
        bridge = _Bridge(ctx.planning(), ctx.observe(epoch), ctx)
        live = bridge.context()
        refused = walk(bridge, live)
        gait.hold(live, ctx.settings.motion.settle_seconds, 0., 0., 0.)
        pose = bridge.sample(force=True).base_pose
        bridge.log("dock.settled", {"pose": pose.tolist(), "refused_command": refused})
        return pose, refused
    except PhysicsInvalid:
        physics_invalid = True
        raise
    except (DriveFailure, ActionRefused) as error:
        raise _failure(error) from error
    except (Stopped, StaleObservation, Refused) as error:
        raise SkillFailure("ACTION_REJECTED", str(error)) from error
    except ObservationError as error:
        raise SkillFailure("STALE_OBSERVATION", str(error)) from error
    finally:
        if not physics_invalid and ctx.observations.epoch == epoch:
            ctx.actions.base_velocity(epoch, 0., 0., 0.)


def _dock_walk(bridge, live, target, yaw, motion):
    """Walk straight to TARGET, holding YAW when supplied, under the docking checks. A refused command
    stops the walk where it is; returns its details, or None."""
    bridge.docking = True
    try:
        _walk_to(bridge, live, target, yaw, motion)
    except MapObstacleRefused as error:
        return error.details
    except DriveFailure as error:
        if not isinstance(error.__cause__, MapObstacleRefused):
            raise
        return error.__cause__.details
    finally:
        bridge.docking = False
    return None


def _back_out(ctx, bridge, live, radius):
    """Back a start that footprint RADIUS does not admit toward a fixed world target under the
    docking checks: to the first pose that admits it with the controller's turning room, else the
    farthest that admits it, within dock_distance. No motion when it admits the start or no pose
    on the way does; a refused command stops where it is. Returns the refusal's details, or None."""
    sample = bridge.sample(force=True)
    checker = gait.occupancy_grid(bridge.artifact(), radius, exact_footprint=True)[0]
    start, yaw = sample.base_pose[:2], sample.base_pose[2]
    if checker.is_safe(*start):
        return None
    room = exploration_settings().drive.final_turn_drift_m
    turning = gait.occupancy_grid(bridge.artifact(), radius + room, exact_footprint=True)[0]
    back = -np.array([math.cos(yaw), math.sin(yaw)])
    distances = np.arange(.02, ctx.settings.motion.dock_distance + 1e-9, .02)
    admitted = [d for d in distances if checker.is_safe(*(start + d * back))]
    roomy = [d for d in admitted if turning.is_safe(*(start + d * back))]
    distance = roomy[0] if roomy else admitted[-1] if admitted else None
    bridge.log("dock.back_out", {"pose": sample.base_pose.tolist(), "radius_m": radius, "turning_room_m": room,
                                 "admitted_after_m": admitted[0] if admitted else None, "distance_m": distance})
    if distance is None:
        return None
    ctx.stage("Backing clear under the docking checks")
    refused = _dock_walk(bridge, live, start + distance * back, None, ctx.settings.motion)
    bridge.hold(live)
    return refused


def _stance_to(bridge, live, target, yaw, motion, *, arrival=False):
    """Aim and close, then verify settled ordinary arrival with bounded reclosure.

    Ordinary arrival stops at its final bounds; refinement keeps its strict bounds.
    Reclosure stays on the checked command path, within this stance and the action's deadline.
    """
    target = np.asarray(target, float)
    phase = "aim" if yaw is not None else "close"
    best, progressed = math.inf, bridge.now()
    close_distance = min(motion.stop_distance, motion.arrival_distance / 2)
    stop_distance = motion.arrival_distance if arrival else close_distance
    stop_angle = motion.arrival_angle if arrival else motion.arrival_angle / 2
    arrived = False

    def reached(pose):
        return (np.linalg.norm(target - pose[:2]) <= stop_distance
                and (yaw is None or abs(float(wrap(yaw - pose[2]))) <= stop_angle))

    def select(vx, vy, wz):
        nonlocal arrived
        pose = bridge.sample().base_pose
        arrived |= reached(pose)
        if arrived:
            return 0., 0., 0.
        if (bridge.docking or bridge.planning.carried) and not wz and (vx or vy):
            return _dock_command(bridge, live.deadline, vx, vy, wz)
        return vx, vy, wz

    rounds, missed = 0, None
    while True:
        close_stalled = False
        while not live.deadline.expired():
            pose = bridge.sample(force=True).base_pose
            offset = target - pose[:2]
            distance = float(np.linalg.norm(offset))
            angle = 0. if yaw is None else float(wrap(yaw - pose[2]))
            facing = abs(angle) <= motion.arrival_angle / 2
            if phase == "aim" and facing:
                phase, best = "close", math.inf
            if reached(pose):
                break
            remaining = abs(angle) * motion.robot_radius if phase == "aim" else distance
            if remaining < best - motion.progress_distance:
                best, progressed = remaining, bridge.now()
            if bridge.now() - progressed > motion.stall_timeout:
                if phase == "aim":
                    raise SkillFailure("STALL", "No sustained progress toward the stance heading")
                close_stalled = True
                break
            if phase == "aim":
                velocity = np.zeros(2)
            elif distance <= close_distance:
                velocity = np.clip(motion.turn_hold_gain * body_frame(pose, offset),
                                   -motion.final_speed, motion.final_speed)
            else:
                speed = float(np.clip(motion.close_gain * distance, motion.min_close_speed, motion.final_speed))
                velocity = body_frame(pose, offset) * speed / distance
            if facing:
                velocity = gait.planar_above_deadband(*velocity)
                turn = gait._above_wz_deadband(motion.heading_gain * angle)
            else:
                rate = max(motion.min_turn_speed, WZ_MIN_ACCEPT,
                           min(motion.turn_speed, motion.heading_gain * abs(angle)))
                turn = math.copysign(rate, angle)
            gait.hold(live, motion.tick, float(velocity[0]), float(velocity[1]), turn, select=select)
            if arrived:
                break
        if live.deadline.expired():
            raise SkillFailure("TIMEOUT", "Original navigation deadline expired while closing onto the stance")
        bridge.log("stance.settling", {"arrival": arrival, "pose": bridge.sample(force=True).base_pose.tolist(),
                   "started_at": bridge.now(), "stop_distance_m": stop_distance, "stop_angle_rad": stop_angle})
        gait.hold(live, motion.settle_seconds, 0., 0., 0.)

        if not arrival:
            return
        if live.deadline.expired():
            raise SkillFailure("TIMEOUT", "Original navigation deadline expired while settling the stance")
        pose = bridge.sample(force=True).base_pose
        distance = float(np.linalg.norm(target - pose[:2]))
        angle = 0. if yaw is None else abs(float(wrap(yaw - pose[2])))
        if reached(pose):
            return
        stalled = missed is not None and missed - distance < motion.close_min_gain
        if rounds >= motion.close_rounds or stalled or close_stalled:
            raise SkillFailure("ARRIVAL_NOT_VERIFIED", "Bounded stance reclosure did not verify settled arrival",
                {"pose": pose.tolist(), "goal_xy": target.tolist(), "distance_m": distance,
                 "angle_rad": angle, "close_rounds": rounds, "stalled": stalled or close_stalled,
                 "verification_phase": "settled_pose"})
        rounds, missed = rounds + 1, distance
        phase = "aim" if angle > motion.arrival_angle / 2 else "close"
        best, progressed, arrived = math.inf, bridge.now(), False


def _turn_to(bridge, live, yaw, motion):
    """Turn in place onto YAW within half the arrival angle, holding the position where the turn
    began, for at most stall_timeout."""
    anchor, began = bridge.sample(force=True).base_pose[:2].copy(), bridge.now()
    while not live.deadline.expired():
        pose = bridge.sample(force=True).base_pose
        angle = float(wrap(yaw - pose[2]))
        if abs(angle) <= motion.arrival_angle / 2 or bridge.now() - began >= motion.stall_timeout:
            return
        rate = max(motion.min_turn_speed, WZ_MIN_ACCEPT, min(motion.turn_speed, motion.heading_gain * abs(angle)))
        hold = np.clip(motion.turn_hold_gain * body_frame(pose, anchor - pose[:2]),
                       -motion.final_speed, motion.final_speed)
        gait.hold(live, motion.tick, float(hold[0]), float(hold[1]), math.copysign(rate, angle))
    raise SkillFailure("TIMEOUT", "Navigation deadline expired while turning onto the stance heading")


def _walk_to(bridge, live, target, yaw, motion):
    """Walk straight to TARGET, the heading YAW held when given, until within the stop distance
    or for at most stall_timeout."""
    began = bridge.now()
    while not live.deadline.expired():
        pose = bridge.sample(force=True).base_pose
        offset = np.asarray(target, float) - pose[:2]
        distance = float(np.linalg.norm(offset))
        if (distance <= min(motion.stop_distance, motion.arrival_distance / 2)
                or bridge.now() - began >= motion.stall_timeout):
            return
        speed = float(np.clip(motion.close_gain * distance, motion.min_close_speed, motion.final_speed))
        vx, vy = gait.planar_above_deadband(*(body_frame(pose, offset) * speed / distance))
        error = 0. if yaw is None else float(wrap(yaw - pose[2]))
        turn = (math.copysign(max(WZ_MIN_ACCEPT, motion.heading_gain * abs(error)), error)
                if abs(error) > motion.arrival_angle / 2 else 0.)
        if bridge.docking and not turn:
            gait.hold(live, motion.tick, float(vx), float(vy), turn,
                      select=lambda x, y, w: _dock_command(bridge, live.deadline, x, y, w))
        else:
            gait.hold(live, motion.tick, float(vx), float(vy), turn)
    raise SkillFailure("TIMEOUT", "Navigation deadline expired while walking onto the stance")


def _dock_command(bridge, deadline, vx, vy, wz):
    """Choose a shorter straight sweep before this tick, never retry a sent command."""
    try:
        bridge._motion_check(vx, vy, wz)
    except MapObstacleRefused:
        if wz or not (vx or vy):
            raise
        scale = gait.VX_DEADBAND * gait.DEADBAND_MARGIN / max(abs(vx), abs(vy))
        if scale >= 1.:
            raise
        nominal = [vx, vy, wz]
        vx, vy = vx * scale, vy * scale
        bridge._motion_check(vx, vy, wz)
        bridge.log("dock.slowed", {"nominal_command": nominal, "command": [vx, vy, wz]})
    if deadline.expired():
        raise SkillFailure("TIMEOUT", "Original docking deadline expired while selecting the next command")
    return vx, vy, wz


def require_navigation_freshness(sample, now):
    """Refuse localization, obstacle or map stamps outside their limits (s) of NOW, or from another
    clock. NOW includes the time spent checking geometry."""
    for label, stamp, limit in (("localization", sample.pose_time, .3),
                                ("obstacles", sample.obstacle_time, .5),
                                ("map", sample.map_time, 3.)):
        if not 0 <= now - stamp <= limit:
            raise SkillFailure("STALE_OBSERVATION", f"Current {label} evidence is stale or from a different clock")


class _Bridge:
    """The controller's declared reads and bounded command surface, backed by Robot."""
    execution_geometry_guarded = True

    def __init__(self, planning, frame, context=None, plan=None):
        self.planning, self.frame, self.outer, self.plan = planning, frame, context, plan
        self.radius = .30
        self.admitted_radius = .60 if plan is None else plan.radius_m
        self._sample = None
        self._sample_read = -float("inf")
        self._checker, self._checker_key = None, None
        self._stance_checker = None
        self._measure = frame
        self._measure_time = -float("inf")
        self._gate = PoseGate()
        self._navigation_binding = None
        self._footprint_key = None
        self._footprint_value = None
        self._physics_fault = None
        self._turn_anchor = None
        self.docking = False
        self._last_look = -float("inf")
        self._column_frames = deque(maxlen=math.ceil(planning.settings.motion.observation_age / DOCK_LOOK_S) + 1)
        self._columns, self._columns_key = None, None
        self._dock_checker, self._dock_checker_key, self._dock_key = None, None, None
        self._dock_checked_at = -math.inf
        self._dock_ray_expiry = -math.inf

    def now(self):
        return self.frame.time if self.outer is None else self.outer.observations.time

    def guard(self, epoch=None, **kwargs):
        if self._physics_fault is not None:
            raise self._physics_fault
        if self.planning.cancelled():
            raise SkillFailure("CANCELLED", "Operator cancelled navigation")
        if self.outer is not None:
            self.outer.check(self.frame.epoch)
            if self.plan is not None and payload_signature(self.outer.carried) != self.plan.payload_signature:
                raise SkillFailure("STALE_EPOCH", "Carry state changed during navigation")

    def sample(self, force=False):
        self.guard()
        if force or self._sample is None or self.now() - self._sample_read >= .04:
            try:
                sample = self.planning.navigation_evidence()
            except ObservationError as error:
                raise SkillFailure("STALE_OBSERVATION", str(error)) from error
            if not isinstance(sample, NavigationEvidence):
                raise SkillFailure("ACTION_REJECTED", "Invalid declared navigation evidence")
            if sample.epoch != self.frame.epoch or not sample.registered:
                raise SkillFailure("STALE_OBSERVATION", "Navigation requires registered localization in the current epoch")
            now = sample.time if self.outer is None else self.now()
            require_navigation_freshness(sample, now)
            if sample.localization_quality < gait.LOC_QUALITY_FLOOR or not sample.mobile:
                raise SkillFailure("ACTION_REJECTED", "Localization quality or motion readiness refuses navigation")
            self.check_binding(sample)
            sample = self.without_own_payload(sample)
            self._sample, self._sample_read = sample, now
        return self._sample

    def without_own_payload(self, sample):
        """Exclude a verified carry's own returns from live obstacles, keeping every static cell
        and every cell another return marks.

        Exclusion needs the robot state at the obstacle acquisition time; without it nothing is
        excluded. Returns inside the retained payload box at that state are excluded. With
        identities, returns of the carried label within grasp.recovery_match of the box (the
        distance carry verification accepts for the held object's surface) are excluded too, but
        only inside the measured complete envelope at acquisition. The independently checked
        arm and payload sweeps cover this extent even when the floor circle is smaller."""
        state = sample.obstacle_state
        if not self.planning.carried or sample.static_navigation is None or state is None:
            return sample
        points = sample.obstacles
        keep = np.ones(len(points), dtype=bool)
        settings = self.planning.settings
        admitted = None
        for side, record in self.planning.carried.items():
            if (record.get("state") != "verified" or side not in self.planning.arms
                    or record.get("palm_to_payload") is None or record.get("payload_half") is None):
                continue
            arm = self.planning.arms[side]
            local = points_in(inverse(state.palm(arm, side) @ record["palm_to_payload"]), points)
            own = np.all(np.abs(local) <= record["payload_half"], axis=1)
            if sample.obstacle_labels is not None and continuing(state, arm, side, record, settings.grasp):
                if admitted is None:
                    admitted = sweep_radius(state, self.planning.arms, self.planning.carried,
                                            settings.motion.robot_radius) + settings.collision.cloud_margin
                outside = np.linalg.norm(np.maximum(np.abs(local) - record["payload_half"], 0.), axis=1)
                inside = np.linalg.norm(points[:, :2] - state.base_pose[:2], axis=1) <= admitted
                own |= ((sample.obstacle_labels == record["label"]) & inside &
                        (outside <= settings.grasp.recovery_match))
            keep &= ~own
        navigation, static = sample.navigation, sample.static_navigation
        if (keep.all() or navigation.occupied.shape != static.occupied.shape
                or not np.array_equal(navigation.T_map_px, static.T_map_px)):
            return sample

        def cells(selected):
            mask = np.zeros(static.occupied.shape, dtype=bool)
            pixels = np.rint(world_to_px(navigation.T_map_px, selected[:, :2])).astype(int)
            valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < mask.shape[1]) &
                     (pixels[:, 1] >= 0) & (pixels[:, 1] < mask.shape[0]))
            mask[pixels[valid, 1], pixels[valid, 0]] = True
            return mask

        occupied = (navigation.occupied & ~(cells(points[~keep]) & ~cells(points[keep]))) | static.occupied
        return replace(sample, navigation=NavigationMap(occupied, navigation.T_map_px, navigation.resolution),
                       obstacles=points[keep],
                       obstacle_labels=None if sample.obstacle_labels is None else sample.obstacle_labels[keep])

    def check_binding(self, sample):
        identity = (sample.frame_id, sample.map_hash, sample.generation)
        if self._navigation_binding is None:
            self._navigation_binding = identity
        if self.plan is not None:
            expected = (self.plan.frame_id, self.plan.map_hash, self.plan.generation)
            if identity != expected:
                raise SkillFailure("STALE_EPOCH", "Localization or map binding changed; prepare a fresh route")
        elif identity != self._navigation_binding:
            raise SkillFailure("STALE_EPOCH", "Localization or map binding changed during preparation")

    def artifact(self, sample=None):
        sample = self.sample() if sample is None else sample
        nav = sample.navigation
        raster = np.repeat(np.where(nav.occupied, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
        return MapArtifact(raster, nav.T_map_px, nav.occupied, sample.source, sample.map_hash,
                           np.ones_like(nav.occupied, bool))

    def measure(self, *, force=False):
        if self.outer is not None and (force or self.now() - self._measure_time >= .04):
            try:
                self._measure = self.outer.measure(self.frame.epoch)
            except ObservationError as error:
                raise SkillFailure("STALE_OBSERVATION", str(error)) from error
            self._measure_time = self.now()
        return self._measure

    def serves(self, name):
        return name in controller.NEEDS.obs

    def __call__(self, name):
        if not self.serves(name):
            raise UndeclaredNeed(f"Checked navigation did not declare {name}")
        sample = self.sample()
        if name == "base_pose":
            return sample.base_pose
        if name == "loc_quality":
            return sample.localization_quality
        if name == "map":
            return self.artifact()
        if name == "lidar_obstacles":
            # The controller expects height above floor in its third column.
            points = sample.obstacles.copy()
            points[:, 2] = sample.heights_above_floor()
            return points
        measured = self.measure()
        if name == "qpos":
            return measured.joints
        if name == "gripper_state":
            limit = self.planning.settings.placement.open_closure_limit
            held = [HELD if side in self.planning.carried else
                    (UNKNOWN if measured.hands[side].closure > limit else None)
                    for side in ("left", "right")]
            return GripperState(measured.hands["left"].closure, measured.hands["right"].closure, *held)
        raise UndeclaredNeed(name)

    def fresh(self, ctx=None, epoch=None, **kwargs):
        """Hand held navigation a current map AND the measurements used by preparation."""
        started = self.now()
        while True:
            self.sample(force=True)
            self.guard()
            if ctx is not None and ctx.deadline.expired():
                raise SkillFailure("TIMEOUT", "Navigation deadline reached during held refresh")
            try:
                self.measure(force=True)
            except SkillFailure as error:
                details = error.details
                # Only an aged, same-epoch packet is eligible for a held refresh.
                # Clock/frame changes and failures without packet evidence always refuse.
                if (ctx is None or error.code != "STALE_OBSERVATION"
                        or details.get("packet_epoch") != self.frame.epoch
                        or not math.isfinite(details.get("age_s", math.nan))
                        or details["age_s"] <= details["max_age_s"]):
                    raise
                waited = self.now() - started
                remaining = min(FRESH_WAIT_S - waited, ctx.deadline.remaining())
                tick = 1. / self.planning.settings.physics.control_hz
                if remaining < tick:
                    raise SkillFailure(error.code, str(error),
                                       {**details, "waited_s": waited}) from error
                self.log("navigation.measurement_wait", {**details, "waited_s": waited})
                self.wait(min(FRESH_POLL_S, remaining))
                continue
            self.guard()
            require_navigation_freshness(self._sample, self.now())
            if ctx is not None and ctx.deadline.expired():
                raise SkillFailure("TIMEOUT", "Navigation deadline reached during held refresh")
            if self.now() > started:
                self.log("navigation.measurement_refreshed", {
                    "waited_s": self.now() - started, "captured_at": self._measure.time,
                    "age_s": self.now() - self._measure.time})
            return

    def stationary(self, generation=None):
        self.guard()
        current = self.sample(force=True).generation
        if generation is not None and generation != current:
            raise SkillFailure("STALE_EPOCH", "Localization generation changed while settling")
        return current

    def hold(self, ctx):
        if self.outer is not None:
            gait.hold(ctx, .20, 0., 0., 0.)

    def measured_execution_radius(self):
        """Revalidate retention and return the current floor circle for motion admission."""
        frame = self.measure()
        footprint_key = (frame.frame_id, payload_signature(self.planning.carried))
        if self._footprint_key != footprint_key:
            self._footprint_value = footprint(frame, self.planning,
                look=None if self.outer is None else lambda: self.outer.observe(self.frame.epoch))
            self._footprint_key = footprint_key
        _radius, execution = self._footprint_value
        if self.plan is not None and execution > self.plan.radius_m + 1e-6:
            raise SkillFailure("STALE_EPOCH", "Measured arm or payload envelope grew; prepare navigation again")
        return max(self.radius, execution)

    def _motion_check(self, vx, vy, wz):
        if self.planning.carried:
            self.measured_execution_radius()
        if self.docking or self.planning.carried:
            return self._dock_check(vx, vy, wz)
        sample = self.sample()
        checked_radius = self.measured_execution_radius()
        execution = self._footprint_value[1]
        if self._stance_checker is not None:
            checker = self._stance_checker(sample, checked_radius)
        else:
            key = (sample.map_hash, sample.generation, self._sample_read, checked_radius)
            if self._checker_key != key:
                self._checker = gait.occupancy_grid(self.artifact(), checked_radius, exact_footprint=True,
                                                    obstacles=sample.navigation.occupied)[0]
                self._checker_key = key
            checker = self._checker
        x, y, yaw = sample.base_pose
        delta = COMMAND_HORIZON_S * np.array([vx * math.cos(yaw) - vy * math.sin(yaw),
                               vx * math.sin(yaw) + vy * math.cos(yaw)])
        verdict = checker.check_motion((x, y), (x + delta[0], y + delta[1]), allow_egress=True)
        if not verdict["safe"] or (wz and not checker.is_safe(x, y)):
            details = {"sweep": verdict, "command": [vx, vy, wz],
                       "execution_radius_m": execution}
            self.log("command.refused", details)
            raise MapObstacleRefused("Current measured footprint refuses this bounded motion", details=details)

    def _dock_floor_checker(self, sample):
        """The same exact base-circle admission for candidate paths and checked commands."""
        settings = self.planning.settings
        motion, collision = settings.motion, settings.collision
        # The transit minimum is not the close-docking footprint. Docking uses
        # the configured base circle and the same measured component sweeps below.
        base_radius = (max(.24, motion.robot_radius)
                       if self.planning.carried and not self.docking else motion.robot_radius)
        radius = base_radius + collision.cloud_margin
        if self._stance_checker is not None:
            return self._stance_checker(sample, radius)
        nav = sample.navigation
        # A new sensor read does not change inflation when its final floor is identical.
        # Immutable bytes also detect changed geometry under the same acquisition ID.
        checker_key = (sample.epoch, sample.frame_id, sample.map_hash, sample.generation, radius,
                       nav.resolution, nav.occupied.shape, nav.occupied.dtype.str, nav.occupied.tobytes(),
                       nav.T_map_px.shape, nav.T_map_px.dtype.str, nav.T_map_px.tobytes())
        if self._dock_checker_key != checker_key:
            self._dock_checker = gait.occupancy_grid(self.artifact(sample), radius, exact_footprint=True,
                                                     obstacles=nav.occupied)[0]
            self._dock_checker_key = checker_key
        return self._dock_checker

    def _dock_check(self, vx, vy, wz):
        """Docking checks parts instead of one circle: the floor footprint against every occupied
        cell, and arm, finger and payload boxes over occupied cells against the observed columns,
        over the command horizon and the stopping distance."""
        sample, state = self.sample(), self.measure()
        history = self.outer.navigation_history
        key = (self._sample_read, state.frame_id, vx, vy, wz, self.docking,
               id(history), None if history is None else history.revision)
        now = self.now()
        if key == self._dock_key and self._dock_checked_at <= now <= self._dock_ray_expiry:
            return
        settings = self.planning.settings
        motion, collision = settings.motion, settings.collision
        checker = self._dock_floor_checker(sample)
        planar = np.array([vx, vy])
        speed = float(np.linalg.norm(planar))
        travel = planar * COMMAND_HORIZON_S + (planar / speed * motion.stopping_distance if speed else 0.)
        turn = wz * COMMAND_HORIZON_S
        x, y, yaw = state.base_pose
        c, s = math.cos(yaw), math.sin(yaw)
        end = (x + c * travel[0] - s * travel[1], y + s * travel[0] + c * travel[1])
        verdict = checker.check_motion((x, y), end, allow_egress=True)
        if not verdict["safe"] or (wz and not checker.is_safe(x, y)):
            self._refuse("The floor footprint refuses this docking motion",
                         {"sweep": verdict, "command": [vx, vy, wz], "floor_radius_m": checker.radius_baked})
        boxes = robot_boxes(state, self.planning.arms, self.planning.carried)
        lever = max((np.linalg.norm(np.asarray(center)[:2] - state.base_pose[:2]) +
                     np.linalg.norm(np.asarray(half) + collision.column_margin)
                     for center, rotation, half in boxes), default=DOCK_LEVER_M)
        count = int(np.ceil((np.linalg.norm(travel) + abs(turn) * lever) / collision.column_spacing)) + 1
        steps = [(f * travel[0], f * travel[1], f * turn) for f in np.linspace(0., 1., max(2, count))]
        columns = self.columns(sample, state)
        flagged, parts = volume.swept_uncleared(
            boxes, state.base_pose, steps,
            self._occupied(sample.navigation), columns, margin=collision.column_margin,
            spacing=collision.column_spacing)
        ray_expiry = math.inf
        if len(flagged):
            occupied, _ = columns.evidence(flagged)
            cleared, ray_expiry = self._depth_cleared(sample, flagged)
            keep = occupied | ~cleared
            flagged, parts = flagged[keep], parts[keep]
        if len(flagged):
            nearest = int(np.argmin(np.linalg.norm(flagged[:, :2] - (x, y), axis=1)))
            witness = flagged[nearest]
            seen_at, beneath = columns.evidence(witness[None])
            self._refuse("Observed columns do not clear an arm, finger or payload sweep over occupied cells",
                         {"evidence": "occupied" if occupied[keep].any() else "unobserved",
                          "observed_at_height": bool(seen_at[0]), "observed_beneath": bool(beneath[0]),
                          "command": [vx, vy, wz], "uncleared_points": len(flagged), "witness": witness.tolist(),
                          "witness_height_m": float(sample.heights_above_floor(witness[None])[0]),
                          "witness_part": robot_part_names(self.planning.arms, self.planning.carried)[parts[nearest]]})
        history = self.outer.navigation_history
        self._dock_key = (*key[:6], id(history), None if history is None else history.revision)
        self._dock_checked_at = now
        self._dock_ray_expiry = ray_expiry

    def _depth_cleared(self, sample, points):
        """Cleared points and expiry, using captured rays within the same freshness limit."""
        clear = np.zeros(len(points), bool)
        expiry = math.inf
        if not self._column_frames:
            return clear, expiry
        latest, history = self._column_frames[-1], self.outer.navigation_history
        binding = _column_binding(latest)
        expected = (sample.epoch, sample.frame_id, sample.map_hash, sample.generation)
        now, age = self.now(), self.planning.settings.motion.observation_age

        def eligible(frame):
            meta = frame.sensor_metadata or {}
            return (meta.get("depth_rays_from_world_camera") is True
                    and _column_binding(frame) == binding
                    and all(0 <= now - stamp <= age for stamp in (frame.time, meta.get("depth_time", frame.time))))

        if (binding is None or not eligible(latest)
                or binding[:4] != expected
                or (history is not None and binding != history.binding)):
            return clear, expiry
        for frame in reversed(self._column_frames):
            if eligible(frame):
                pending = np.flatnonzero(~clear)
                clear[pending] = volume.depth_clears(points[pending], frame.world_camera, frame.intrinsics,
                                                    frame.depth, frame.labels >= 0,
                                                    self.planning.settings.collision.column_margin)
                if clear[pending].any():
                    expiry = min(expiry, frame.time + age,
                                 (frame.sensor_metadata or {}).get("depth_time", frame.time) + age)
                if clear.all():
                    break
        return clear, expiry

    def columns(self, sample, state):
        """Current hits plus bounded, registered support history; live freshness is independent."""
        collision = self.planning.settings.collision
        now = self.now()
        if now < self._last_look or now - self._last_look >= DOCK_LOOK_S:
            frame = self.outer.observe(self.frame.epoch)
            if self._column_frames:
                previous = self._column_frames[-1]
                meta, previous_meta = frame.sensor_metadata or {}, previous.sensor_metadata or {}
                if (now < self._last_look or frame.time < previous.time
                        or meta.get("depth_time", frame.time) < previous_meta.get("depth_time", previous.time)
                        or _column_binding(frame) != _column_binding(previous)
                        or meta.get("depth_rays_from_world_camera") is not True):
                    self._column_frames.clear()
            self._column_frames.append(frame)
            while self._column_frames and now - self._column_frames[0].time > self.planning.settings.motion.observation_age:
                self._column_frames.popleft()
            self._last_look = now
        history = self.outer.navigation_history
        excluded = tuple(sorted(record["label"] for record in self.planning.carried.values()
                                if record.get("state") == "verified" and record.get("label", 0) > 0))
        if history is not None:
            binding = (sample.epoch, sample.frame_id, sample.map_hash, sample.generation)
            if history.binding[:4] != binding:
                self.outer.navigation_history = None
                raise SkillFailure("STALE_OBSERVATION", "Support history and navigation frames changed")
            # Expiry and clock rollback must invalidate reuse before materializing hits.
            history._clock(now, history.binding, query=True)
        live = sample.obstacles[volume.outside(
            sample.obstacles, robot_boxes(state, self.planning.arms, self.planning.carried),
            collision.column_margin)]
        key = (history, None if history is None else history.revision,
               excluded, collision.column_spacing, len(live))
        if (self._columns_key == key and self._columns is not None
                and np.array_equal(self._columns.hits[:len(live)], live)):
            return self._columns
        points = (np.empty((0, 3)) if history is None else
                  history.points(now, history.binding, excluded_labels=excluded))
        self._columns = volume.Columns(np.vstack([live, points]), collision.column_spacing)
        self._columns_key = key
        return self._columns

    @staticmethod
    def _occupied(navigation):
        """Whether map points lie on occupied cells; outside the raster counts as occupied."""
        def occupied(xy):
            pixels = np.rint(world_to_px(navigation.T_map_px, xy)).astype(int)
            height, width = navigation.occupied.shape
            inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
            result = np.ones(len(xy), bool)
            result[inside] = navigation.occupied[pixels[inside, 1], pixels[inside, 0]]
            return result
        return occupied

    def _refuse(self, message, details):
        self.log("command.refused", details)
        raise MapObstacleRefused(message, details=details)

    def motion_refusal(self, vx, vy, wz=0.):
        """Check a recovery probe without writing, advancing, or changing its turn anchor."""
        try:
            self._checked_command(LowerBody(vx=vx, vy=vy, wz=wz), update_anchor=False)
        except MapObstacleRefused as error:
            return error
        return None

    def _checked_command(self, command, *, update_anchor):
        self.guard()
        cmd = clamp_lower(command, self)
        loaded = bool(self.planning.carried)
        motion = self.planning.settings.motion
        speed = motion.loaded_drive_speed if loaded else motion.drive_speed
        turn = motion.loaded_turn_speed if loaded else motion.turn_speed
        vx, vy, wz = (float(value or 0) for value in (cmd.vx, cmd.vy, cmd.wz))
        if vx == 0. and vy == 0. and wz != 0.:
            # SIM Transit.send's position hold, before the checked combined sweep.
            pose = self.sample().base_pose
            anchor = self._turn_anchor if self._turn_anchor is not None else pose[:2].copy()
            if update_anchor:
                self._turn_anchor = anchor
            if not (loaded or self.docking) or np.linalg.norm(anchor - pose[:2]) > motion.stop_distance:
                vx, vy = np.clip(motion.turn_hold_gain * body_frame(pose, anchor - pose[:2]),
                                 -motion.final_speed, motion.final_speed)
        elif update_anchor and (vx != 0. or vy != 0.):
            # A zero hold pauses the turn; only translation starts a new anchor.
            self._turn_anchor = None
        scale = min(1., speed / max(math.hypot(vx, vy), 1e-9))
        vx, vy, wz = vx * scale, vy * scale, float(np.clip(wz, -turn, turn))
        if vx or vy or wz:
            self._motion_check(vx, vy, wz)
            self.guard()
            require_navigation_freshness(self._sample, self.now())
            if not 0 <= self.now() - self._measure.time <= motion.observation_age:
                raise SkillFailure("STALE_OBSERVATION", "Measured geometry expired during collision checks")
            if ((self.docking or loaded)
                    and not self._dock_checked_at <= self.now() <= self._dock_ray_expiry):
                raise SkillFailure("STALE_OBSERVATION", "Depth evidence expired during collision checks")
        return vx, vy, wz

    def step(self, channel, command):
        if self.outer is None:
            raise RuntimeError("Navigation preparation cannot command motion")
        if channel != "lower_body" or not isinstance(command, LowerBody):
            raise UndeclaredNeed("Checked navigation exposes only lower_body")
        vx, vy, wz = self._checked_command(command, update_anchor=True)
        try:
            self.outer.actions.base_velocity(self.frame.epoch, vx, vy, wz)
            self.log("command", {"vx": vx, "vy": vy, "wz": wz,
                     "requested": {"vx": command.vx, "vy": command.vy, "wz": command.wz}})
            self.outer.advance(self.frame.epoch, 1. / self.planning.settings.physics.control_hz)
        except PhysicsInvalid as error:
            self._physics_fault = error
            raise
        return LowerBody(vx=vx, vy=vy, wz=wz)

    def wait(self, seconds):
        if self.outer is None:
            raise SkillFailure("STALE_OBSERVATION", "Preparation cannot wait for missing localization")
        self.outer.actions.base_velocity(self.frame.epoch, 0., 0., 0.)
        tick = 1. / self.planning.settings.physics.control_hz
        self.outer.advance(self.frame.epoch, max(tick, round(seconds / tick) * tick))

    def log(self, kind, payload=None):
        """Emit a navigation event; private diagnostics such as the controller's occupancy grid
        (keys starting with _) stay out of the recording."""
        if self.outer is not None:
            details = {key: value for key, value in (payload or {}).items() if not key.startswith("_")}
            self.outer.emit({"type": "navigation", "kind": kind, "details": details})

    def context(self):
        return Ctx(self, self.step, self.now, Deadline(self.now, self.planning.settings.motion.drive_timeout),
                   lambda text: None if self.outer is None else self.outer.stage(text),
                   self.log, self._gate, self.wait, max_footprint_radius_m=max(.60, self.admitted_radius),
                   route_heading_gain_hz=self.planning.settings.motion.heading_gain
                       if self.planning.carried else 0.)

"""Bounded checked navigation and recovery over declared primitive capabilities."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import cKDTree

from homebody.helpers.raster import px_to_world
from homebody.motion.exploration import gait, localizer
from homebody.motion.exploration.nav import _grid_segment_free, plan_grid_astar
from homebody.primitives.navigation.contracts import (
    DriveFailure,
    HardwareUnavailable,
    Refused,
    StaleObservation,
)
from homebody.primitives.navigation.errors import (
    ActionRefused,
    MapObstacleRefused,
    MotionBlocked,
    ObservationUnavailable,
    Stopped,
    TaskScopeRefused,
)
from homebody.real2sim.exploration.settings import current as current_settings
from homebody.skills.contract import SkillFailure

from . import controller as drive

RECOVERY_MEMORY_DISTANCE_M = 0.25
RECOVERY_MEMORY_HEADING_RAD = 0.25
RECOVERY_MEMORY_LIMIT = 16
RECOVERY_OBSERVATION_RETRIES = 2


@dataclass(frozen=True)
class NavigationCapabilities:
    """Observation and command guarantees selected by the composition root."""

    execution_radius_m: float
    minimum_radius_m: float
    padded_radii_m: tuple[float, ...]
    turn_translation_limit_m: float
    map_max_age_s: float


@dataclass
class NavigationState:
    _last_drive_route: list = field(default_factory=list)
    _last_drive_result: dict | None = None
    _escape_attempts: list = field(default_factory=list)
    _escape_episode_origin: tuple | None = None
    _escape_history_epoch: object = None


@dataclass(frozen=True)
class NavigationPort:
    """Only clocks, current observations, authority, and bounded command support."""

    capabilities: NavigationCapabilities
    epoch: Callable
    guard: Callable
    fresh: Callable
    hold: Callable
    stage: Callable
    planning_radius: Callable
    recovery_map: Callable
    map_age: Callable
    stationary_guard: Callable
    motion_refusal: Callable | None
    stopped: Callable
    require_ready: Callable
    can_refresh_observations: bool


def _retreat_grids(art, obstacles, radius_m, execution_radius_m, grids):
    """Cache the full and motion collision grids once per frozen map."""
    if "full" not in grids:
        grids["full"], _ = gait.occupancy_grid(art, radius_m, exact_footprint=True)
        grids["motion"], _ = gait.occupancy_grid(
            art, execution_radius_m, exact_footprint=True, obstacles=obstacles)


def _retreat_corridor(art, obstacles, xy, yaw, remaining_m, radius_m, previous_xy=None,
                     *, grids=None, execution_radius_m=None):
    """Check actual swept motion and a planning-clear endpoint with stopping room."""
    execution_radius_m = radius_m if execution_radius_m is None else execution_radius_m
    grids = {} if grids is None else grids
    _retreat_grids(art, obstacles, radius_m, execution_radius_m, grids)
    full, motion = grids["full"], grids["motion"]
    direction = np.array([math.cos(yaw), math.sin(yaw)])
    endpoint = np.asarray(xy) - remaining_m * direction
    stopping = endpoint - .15 * direction
    result = {"margin_escape": not full.is_safe(*xy), "radius_m": radius_m,
              "planning_radius_m": radius_m,
              "minimum_radius_m": execution_radius_m, "execution_radius_m": execution_radius_m,
              "error": ""}
    segments = [("recovery corridor", xy, stopping)]
    if previous_xy is not None:
        segments.insert(0, ("swept recovery motion", previous_xy, xy))
    for label, start, end in segments:
        verdict = motion.check_motion(start, end, allow_egress=True)
        result["sweep" if label.startswith("swept") else "corridor"] = verdict
        if not verdict["safe"]:
            result["error"] = f"{label}: {verdict['reason']}"
            return result
    stopping_verdict = full.check_motion(endpoint, stopping)
    result["stopping"] = stopping_verdict
    if not stopping_verdict["safe"]:
        result["error"] = "recovery endpoint and stopping room do not restore the planning margin"
    return result


class Navigator:
    """The one checked navigation algorithm; no sensor owner or engine identity."""

    def __init__(self, port: NavigationPort, state: NavigationState | None = None):
        self.port = port
        self.capabilities = port.capabilities
        self.state = state if state is not None else NavigationState()

    def recover(self, ctx, radius_m, goal_xy=None, *, requested_radius_m=None):
        try:
            return self._checked_escape(ctx, radius_m, goal_xy,
                                        requested_radius_m=requested_radius_m)
        except BaseException:
            # stand still before re-raising
            self.port.hold(ctx)
            raise

    def drive(self, ctx, goal_xy, facing_xy, radius_m: float,
              *, requested_radius_m=None, arrival_tolerance_m=None,
              stop_after_departure=False) -> dict | None:
        """Preserve one destination/deadline across replans and checked staging moves."""
        self.port.require_ready()
        epoch = self.port.epoch()
        staging = facing_xy is None
        args = {"goal_xy_m": list(goal_xy), "radius_m": float(radius_m), "snap": False}
        if arrival_tolerance_m is not None:
            args["arrival_tolerance_m"] = arrival_tolerance_m
        request = {} if requested_radius_m is None else {"requested_radius_m": requested_radius_m}
        if request:
            args.update(radius_m=requested_radius_m, planning_radius_m=radius_m)
        args.update(preserve_goal=True, arrival_heading_required=facing_xy is not None,
                    preferred_clearance=False)
        if facing_xy is None:
            args["turn_translation_limit_m"] = self.capabilities.turn_translation_limit_m
            args["execution_radius_m"] = self.capabilities.execution_radius_m
            args["route_rejoin"] = True
        if facing_xy is not None:
            args["facing_xy_m"] = list(facing_xy)
        ctx.log("drive.request", dict(args))
        self.state._last_drive_route = []
        self.state._last_drive_result = None

        def no_step(channel, payload):
            raise RuntimeError("prepare may not command")  # the localizer coasts on a RuntimeError

        attempts, repairs, failures = 0, [], []
        retreat_eligible, replanned, retreated = False, False, False
        connector_replanned = False
        repair_tiers = []
        active_radius_m = None
        planning_radius_m = radius_m
        started = ctx.now()

        def failure_details(reason, cause_details=None):
            return {"goal_xy": list(goal_xy), "route_xy": self.state._last_drive_route,
                       "original_reason": failures[0] if failures else reason,
                       "failures": failures, "attempts": attempts, "repairs": repairs,
                       "requested_radius_m": args["radius_m"], "active_radius_m": active_radius_m,
                       "elapsed_s": ctx.now() - started, "remaining_s": ctx.deadline.remaining(),
                       **(cause_details or {})}

        def unavailable(code, reason, cause_details=None):
            details = failure_details(reason, cause_details)
            ctx.log("drive.action_refused", {"code": code, "reason": reason, **details})
            raise ActionRefused(reason, code=code, details=details)

        def fresh(**kwargs):
            try:
                self.port.fresh(ctx, epoch, **kwargs)
            except SkillFailure as error:
                details = failure_details(str(error), error.details)
                ctx.log("drive.handoff_refused", {"code": error.code, "reason": str(error), **details})
                raise SkillFailure(error.code, str(error), details) from error

        while True:
            planning = True
            try:
                fresh(require_map=True)
                if ctx.deadline.expired():
                    self.port.hold(ctx)
                    unavailable("TIMEOUT", "action deadline reached while held")
                planning_radius_m = self.port.planning_radius(args["radius_m"])
                if request:
                    planning_radius_m = max(radius_m, planning_radius_m)
                args["planning_radius_m"] = planning_radius_m
                self.port.stage("repairing" if replanned else "planning",
                                   f"planning to accepted destination at {planning_radius_m:.2f} m")
                prepared = drive.prepare(replace(ctx, step=no_step), args,
                                         cancelled=lambda: self.port.guard(epoch))
                self.port.guard(epoch)
                if prepared.get("drive_back") is not None:
                    raise DriveFailure("NO_COLLISION_FREE_PATH",
                                       "no checked route from this start; backup fallback refused")
                tier = getattr(prepared.get("route"), "diagnostic", "")
                if tier:
                    raise DriveFailure("NO_COLLISION_FREE_PATH",
                        f"drive would only fit at the {tier} tier, under {args['radius_m']:.2f} m")
                if math.dist(prepared["route"].goal, goal_xy) > 1e-6:
                    raise Stopped("driver relocated the accepted goal; holding position")
                path = prepared.get("walked_route", prepared["route"].path
                                    if hasattr(prepared["route"], "path") else [])
                self.state._last_drive_route = [[float(x), float(y)] for x, y in (path or [])]
                if ctx.deadline.expired():
                    self.port.hold(ctx)
                    unavailable("TIMEOUT", "action deadline reached during planning")
                attempts += 1
                # the admitted execution radius is the one checked retreat uses
                active_radius_m = float(prepared["radius_m"])
                repair_tiers.clear()  # only planning failures may try another tier
                planning = False
                self.port.stage("driving", "following the validated route")
                arrival = drive.execute(ctx, prepared)
                self.port.guard(epoch)
                ctx.log("drive.result", {"attempt": attempts - 1, "result": arrival})
                if not arrival.get("arrived", False):
                    raise DriveFailure("ARRIVAL_NOT_VERIFIED",
                                       "driver did not verify settled arrival")
                if staging:
                    fresh(require_map=True)
                    try:
                        settled = localizer.read(ctx, "base_pose", lambda: ctx.observe("base_pose"),
                            lambda: gait.hold(ctx, localizer.RETRY_S, 0., 0., 0.),
                            accepted_only=True)
                    except StaleObservation as exc:
                        raise DriveFailure("ARRIVAL_NOT_VERIFIED", str(exc)) from None
                    checker, _ = gait.occupancy_grid(ctx.observe("map"),
                        args.get("execution_radius_m", active_radius_m), exact_footprint=True)
                    if (math.dist(settled[:2], goal_xy) > args.get("arrival_tolerance_m", current_settings().drive.tolerance_m)
                            or not checker.is_safe(*settled[:2])):
                        raise DriveFailure("ARRIVAL_NOT_VERIFIED",
                                           "arrival changed during freshness check")
                    self.port.guard(epoch)
                    ctx.log("drive.arrival_confirmed", {"pose": list(settled),
                            "goal_xy": list(goal_xy), "gap_m": math.dist(settled[:2], goal_xy)})
                self.state._last_drive_result = {
                    "goal_xy": list(goal_xy), "route_xy": self.state._last_drive_route,
                    "attempts": attempts, "repairs": repairs, "failures": failures,
                    "requested_radius_m": args["radius_m"], "active_radius_m": active_radius_m,
                    "elapsed_s": ctx.now() - started, "arrival": arrival}
                return
            except (DriveFailure, Refused, HardwareUnavailable) as failure:
                self.port.hold(ctx)
                self.port.guard(epoch, allow_pause=True)
                code = getattr(failure, "code", type(failure).__name__)
                detail = getattr(failure, "detail", str(failure))
                cause = failure.__cause__ if isinstance(failure, DriveFailure) else failure
                cause_details = getattr(cause, "details", {})
                if isinstance(cause, TaskScopeRefused):
                    unavailable(cause.code, str(cause), cause_details)
                ordinary = ((cause is None and code in {
                                "NO_COLLISION_FREE_PATH", "ARRIVAL_NOT_VERIFIED", "NO_PROGRESS",
                                "TIMEOUT"})
                            or isinstance(cause, (MotionBlocked, ObservationUnavailable)))
                row = {"attempt": attempts, "code": code, "detail": detail,
                       "phase": "plan" if planning else "execute", "radius_m": args["radius_m"],
                       "requested_radius_m": args["radius_m"], "active_radius_m": active_radius_m,
                       "elapsed_s": ctx.now() - started}
                if cause_details:
                    row["cause_details"] = cause_details
                failures.append(row)
                ctx.log("drive.failure", row)
                witness = getattr(failure, "collision_witness", None)
                if witness is not None:
                    ctx.log("drive.collision_refused", {**row, **witness})
                if not ordinary:
                    raise Stopped(f"drive {code}: {detail}") from failure
                if (isinstance(cause, ObservationUnavailable)
                        and not self.port.can_refresh_observations):
                    raise Stopped("cannot verify fresh observations after the refusal") from failure
                fresh(require_map=True)
                if ctx.deadline.expired():
                    unavailable("TIMEOUT", f"action deadline reached after {code}: {detail}", cause_details)
                geometric = ((cause is None and code == "NO_COLLISION_FREE_PATH")
                             or isinstance(cause, MapObstacleRefused))
                if planning and geometric and repair_tiers:
                    args["radius_m"] = repair_tiers.pop(0)
                    repairs[-1]["radius_m"] = args["radius_m"]
                    ctx.log("drive.clearance_retry", {
                        "goal_xy_m": list(goal_xy), "from_radius_m": row["radius_m"],
                        "radius_m": args["radius_m"], "remaining_s": ctx.deadline.remaining()})
                    continue
                # plan and execution failures share checked recovery
                retreat_eligible |= geometric and (staging or not planning)
                repair_connector = (facing_xy is None and replanned and not connector_replanned
                                    and not planning and geometric and witness is not None
                                    and witness.get("check") == "pose_to_target")
                check_retreat = (replanned and retreat_eligible and geometric and not repair_connector
                                 and not retreated)
                if not replanned:
                    recovery_reason = "fresh same-goal replan first"
                elif repair_connector:
                    recovery_reason = "fresh flat connector replan before retreat"
                elif not retreat_eligible or not geometric:
                    recovery_reason = "no eligible geometric failure"
                elif retreated:
                    recovery_reason = "checked recovery already used for this destination"
                else:
                    recovery_reason = "check reverse corridor"
                ctx.log("drive.recovery_decision", {
                    "code": code, "phase": row["phase"], "check_retreat": check_retreat,
                    "requested_radius_m": args["radius_m"], "attempted_radius_m": args["radius_m"],
                    "active_radius_m": active_radius_m,
                    "reason": recovery_reason, "elapsed_s": row["elapsed_s"]})
                if not replanned or repair_connector:
                    replanned = True
                    connector_replanned |= repair_connector
                    kind = "connector_replan" if repair_connector else "same_goal_replan"
                    if geometric and not planning and facing_xy is None:

                        # narrower tiers, used only after a planning refusal
                        repair_tiers = [r for r in self.capabilities.padded_radii_m
                                        if self.capabilities.minimum_radius_m <= r < args["radius_m"]]
                    repairs.append({"kind": kind, "t": ctx.now() - started,
                                    "radius_m": args["radius_m"]})
                    self.port.stage("repairing", f"fresh same-goal plan after {code}")
                    ctx.log("drive.retry", {"goal_xy_m": list(goal_xy), "kind": kind,
                                            "from_radius_m": row["radius_m"],
                                            "radius_m": args["radius_m"],
                                            "remaining_s": ctx.deadline.remaining()})
                    continue
                if check_retreat:
                    repairs.append({"kind": "checked_retreat", "t": ctx.now() - started,
                                    "radius_m": active_radius_m or planning_radius_m})
                    self.port.stage("repairing", "checking a staging departure")
                    try:
                        recovery_radius = active_radius_m or planning_radius_m
                        if request:
                            recovery_radius = max(recovery_radius,
                                self.port.planning_radius(requested_radius_m))
                        recovery_start = gait.pose(ctx)
                        try:
                            retreated = True
                            retreat_result = self._checked_retreat(ctx, recovery_radius)
                            if staging and retreat_result:
                                continuation = self._recovery_departure(
                                    ctx, recovery_radius, goal_xy, **request)
                                if continuation is None:
                                    raise ActionRefused("reverse ended without a usable departure",
                                                        code="RETREAT_UNAVAILABLE")
                                retreat_result = retreat_result | continuation
                        except ActionRefused as refused:
                            if (not staging or refused.code.startswith("TASK_")
                                    or refused.details.get("recovery_check", {}).get("failed_stage")
                                    == "sweep"):
                                raise
                            failures.append({"code": refused.code, "detail": str(refused)})
                            self.port.guard(epoch)
                            retreat_result = None
                            if math.dist(gait.pose(ctx)[:2], recovery_start[:2]) > 1e-6:
                                retreat_result = self._recovery_departure(
                                    ctx, recovery_radius, goal_xy, **request)
                            if retreat_result is None:
                                retreat_result = self._checked_escape(
                                    ctx, recovery_radius, goal_xy, **request)
                        if retreat_result:
                            repairs[-1].update(retreat_result)
                    except ActionRefused as refused:
                        failures.append({"code": refused.code, "detail": str(refused)})
                        if not refused.code.startswith("TASK_"):
                            fresh()
                        unavailable(refused.code, f"checked retreat unavailable: {refused}",
                                    refused.details)
                    except (HardwareUnavailable, Refused, DriveFailure) as refused:
                        self.port.hold(ctx)
                        raise Stopped(f"checked retreat interrupted: {refused}") from refused
                    self.port.guard(epoch)
                    if stop_after_departure and retreat_result:
                        result = {"outcome": "departure_restored", "arrived": False,
                                  "goal_xy": list(goal_xy), "departure": retreat_result,
                                  "attempts": attempts, "repairs": repairs, "failures": failures,
                                  "elapsed_s": ctx.now() - started}
                        self.state._last_drive_result = result
                        ctx.log("drive.departure_restored", result)
                        return result
                    if (retreat_result and retreat_result.get("outcome")
                            == "repositioned_goal_unavailable"):
                        unavailable("RECOVERY_GOAL_UNAVAILABLE",
                                    "repositioned to a usable departure; accepted goal "
                                    "remains unavailable; select another destination")
                    if staging:
                        replanned = connector_replanned = False
                    ctx.log("drive.retry", {"goal_xy_m": list(goal_xy),
                                            "kind": "after_checked_retreat",
                                            "remaining_s": ctx.deadline.remaining()})
                    continue
                unavailable(code, f"action unavailable after local repair: {detail}", cause_details)
            except ActionRefused:
                raise
            except (Stopped, SkillFailure) as exc:
                ctx.log("drive.stopped", {"reason": str(exc), "failures": failures,
                                          "repairs": repairs})
                self.port.hold(ctx)
                raise
            except Exception as exc:
                ctx.log("drive.unexpected", {"error": f"{type(exc).__name__}: {exc}",
                                             "failures": failures, "repairs": repairs})
                self.port.hold(ctx)
                raise


    def _settled_recovery_pose(self, ctx):
        """A measured, accepted base pose, never a prediction."""
        try:
            if getattr(ctx.observe, "_owns_pose_consistency", False):
                pose = ctx.observe("base_pose")
            else:
                pose = localizer.read(ctx, "base_pose", lambda: ctx.observe("base_pose"),
                    lambda: gait.hold(ctx, localizer.RETRY_S, 0., 0., 0.), accepted_only=True)
            return tuple(float(v) for v in pose)
        except StaleObservation as exc:
            raise ActionRefused(str(exc), code="OBSERVATIONS_UNAVAILABLE") from exc


    def _recovery_departure(self, ctx, radius_m, goal_xy, *, requested_radius_m=None):
        """With the base stopped, verify a departure across a confirmation stand; None when no
        usable departure remains."""
        epoch = self.port.epoch()
        generation = self.port.stationary_guard()
        self.port.fresh(ctx, epoch, require_map=True)
        held_pose = self._settled_recovery_pose(ctx)
        art, _, version = self.port.recovery_map()
        if requested_radius_m is not None:
            radius_m = max(radius_m, self.port.planning_radius(requested_radius_m))
        checker, _ = gait.occupancy_grid(art, radius_m, exact_footprint=True)
        held_start = {"pose": list(held_pose), "map_version": version,
                      "planning_clear": checker.is_safe(*held_pose[:2])}
        # stand again: the start must stay clear across the stand
        self.port.hold(ctx)
        self.port.fresh(ctx, epoch, require_map=True)
        x, y, yaw = self._settled_recovery_pose(ctx)
        art, _, version = self.port.recovery_map()
        checker, _ = gait.occupancy_grid(art, radius_m, exact_footprint=True)
        geometry = {"execution_radius_m": self.capabilities.execution_radius_m}
        result = {"pose": [x, y, yaw], "map_version": version,
                  "goal_xy": goal_xy, "outcome": "no_usable_departure",
                  "held_start": held_start,
                  "planning_radius_m": radius_m,
                  "execution_radius_m": geometry["execution_radius_m"]}

        def admitted(heading, turning=False):
            if ctx.deadline.expired():
                raise ActionRefused("recovery continuation deadline reached", code="TIMEOUT")
            guard = self.port.motion_refusal
            if not callable(guard):
                return True
            vx, vy = ((0., 0.) if turning else
                      gait.planar_above_deadband(.15 * math.cos(heading-yaw),
                                                .15 * math.sin(heading-yaw)))
            bearing = math.atan2(math.sin(heading-yaw), math.cos(heading-yaw))
            refused = guard(vx, vy, drive._turn_rate(bearing) if turning else 0.)
            if isinstance(refused, TaskScopeRefused):
                raise ActionRefused(str(refused), code=refused.code,
                                    details=refused.details) from refused
            if isinstance(refused, ObservationUnavailable):
                raise ActionRefused(str(refused), code="OBSERVATIONS_UNAVAILABLE")
            if ctx.deadline.expired():
                raise ActionRefused("recovery continuation deadline reached", code="TIMEOUT")
            return refused is None

        prepared = None
        if checker.is_safe(x, y) and goal_xy is not None:
            def no_step(_channel, command):
                if any(getattr(command, key, 0.) for key in ("vx", "vy", "wz")):
                    raise RuntimeError("recovery continuation preparation may not move")
                raise ActionRefused("continuation pose needs a fresh held observation",
                                    code="OBSERVATIONS_UNAVAILABLE")

            try:
                prepared = drive.prepare(replace(ctx, step=no_step), {
                    "goal_xy_m": list(goal_xy), "radius_m": (radius_m if requested_radius_m is None
                        else requested_radius_m), "planning_radius_m": radius_m, "snap": False,
                    "preserve_goal": True, "arrival_heading_required": False,
                    "preferred_clearance": False, "route_rejoin": True,
                    "turn_translation_limit_m": self.capabilities.turn_translation_limit_m, **geometry},
                    cancelled=lambda: self.port.guard(epoch))
            except StaleObservation as exc:
                raise ActionRefused(str(exc), code="OBSERVATIONS_UNAVAILABLE") from exc
            except DriveFailure as exc:
                if exc.code == "TIMEOUT":
                    raise ActionRefused(str(exc), code="TIMEOUT") from exc
                if exc.code != "NO_COLLISION_FREE_PATH":
                    raise
                result["reason"] = str(exc)
        if prepared is not None:
            radius_m = float(prepared["radius_m"])
        # re-read the pose and map after preparation and admit the actual start
        self.port.fresh(ctx, epoch, require_map=True)
        x, y, yaw = self._settled_recovery_pose(ctx)
        if requested_radius_m is not None:
            radius_m = max(radius_m, self.port.planning_radius(requested_radius_m))
            if prepared is not None and radius_m > prepared["radius_m"] + 1e-9:
                prepared = None  # a wider footprint needs a new route
        result["planning_radius_m"] = radius_m
        art, _, version = self.port.recovery_map()
        checker, _ = gait.occupancy_grid(art, radius_m, exact_footprint=True)
        result.update(pose=[x, y, yaw], map_version=version,
                      departure_verdict="no_checked_departure")
        if not checker.is_safe(x, y):
            result["departure_verdict"] = "starting_footprint_blocked"
        elif prepared is not None:
            path = prepared["walked_route"]
            start = (x, y)
            target = next((p for p in path[1:] if math.dist(start, p) > 1e-6), start)
            heading = math.atan2(target[1]-y, target[0]-x)
            bearing = abs(math.atan2(math.sin(heading-yaw), math.cos(heading-yaw)))
            tolerance = (drive.MIN_WALK_GAP_M if prepared["route"].clearance_margin_m
                         else current_settings().drive.waypoint_tol_m)
            align = (drive._align_gate(math.dist(start, target), tolerance)
                     if len(path) > 2 else drive.LEG_ALIGN_RAD)
            close_step = len(path) <= 2 and math.dist(start, target) <= current_settings().drive.close_step_m
            at_goal = math.dist(start, goal_xy) <= prepared["tolerance_m"]
            if not at_goal and not _grid_segment_free(checker, start, target):
                result["departure_verdict"] = "first_connector_blocked"
            elif at_goal or admitted(heading, not close_step and bearing > align):
                result.update(outcome="route_restored", departure_verdict="admitted",
                              route_xy=[list(start), *[list(p) for p in path[1:]]])
            else:
                result["departure_verdict"] = "motion_admission_refused"
        else:
            for offset in (0., math.pi, math.pi/2, -math.pi/2,
                           math.pi/4, -math.pi/4, 3*math.pi/4, -3*math.pi/4):
                heading = yaw + offset
                end = (x + .35 * math.cos(heading), y + .35 * math.sin(heading))
                if _grid_segment_free(checker, (x, y), end) and admitted(heading):
                    result["outcome"] = ("departure_restored" if goal_xy is None
                                         else "repositioned_goal_unavailable")
                    result["departure_verdict"] = "admitted"
                    break
        if result["departure_verdict"] == "admitted" and not held_start["planning_clear"]:
            result.update(outcome="no_usable_departure",
                          departure_verdict="planning_clearance_not_retained_while_held")
            result.pop("route_xy", None)
        self.port.guard(epoch)
        self.port.stationary_guard(generation)
        if ctx.deadline.expired():
            raise ActionRefused("recovery continuation deadline reached", code="TIMEOUT")
        if result["outcome"] != "no_usable_departure":
            for record in self.state._escape_attempts:
                if (record["suppressed"]
                        and (record.get("check") or {}).get("failed_stage") == "sweep"):
                    record["suppressed"] = False
                    ctx.log("recovery.memory", {"reason": "verified_held_departure",
                            "pose": result["pose"], "check": record["check"]})
        ctx.log("recovery.continuation", result)
        return result if result["outcome"] != "no_usable_departure" else None


    def _checked_escape(self, ctx, radius_m, goal_xy=None, *, requested_radius_m=None):
        """Stage along a checked direction, then let the driver reconnect and turn."""
        request = {} if requested_radius_m is None else {"requested_radius_m": requested_radius_m}
        epoch = self.port.epoch()
        continuation_pending = False
        observation_failures = 0
        while True:
            self.port.guard(epoch)
            if ctx.deadline.expired():
                raise ActionRefused("recovery action deadline reached", code="TIMEOUT")
            try:
                if continuation_pending:
                    result = self._recovery_departure(ctx, radius_m, goal_xy, **request)
                    if result is not None:
                        result = {"kind": "checked_staging", **result}
                    continuation_pending = False
                else:
                    result = self._checked_escape_from_pose(ctx, radius_m, goal_xy, **request)
            except ActionRefused as exc:
                if exc.code != "OBSERVATIONS_UNAVAILABLE":
                    raise
                self.port.hold(ctx)
                observation_failures += 1
                if observation_failures > RECOVERY_OBSERVATION_RETRIES:
                    raise ActionRefused("recovery observations repeatedly unavailable; holding",
                                        code="OBSERVATIONS_UNAVAILABLE",
                                        details={"failures": observation_failures}) from exc
                self.port.fresh(ctx, epoch, require_map=True)
                continuation_pending = True
                continue
            self.port.guard(epoch)
            if result is not None:
                return result


    def _checked_escape_from_pose(self, ctx, radius_m, goal_xy, *, requested_radius_m=None):
        """Search one fresh pose; None requests rebuilding after a refused move shifted it."""
        request = {} if requested_radius_m is None else {"requested_radius_m": requested_radius_m}
        if request:
            radius_m = max(radius_m, self.port.planning_radius(requested_radius_m))
        epoch = self.port.epoch()
        self.port.fresh(ctx, epoch, require_map=True)
        x, y, yaw = gait.pose(ctx)
        art, obstacles, version = self.port.recovery_map()
        candidates, grid_cache = [], {}
        history = self._escape_history(ctx, epoch, (x, y), art, obstacles, version,
                                       grids=grid_cache)
        for record in history:
            if (record["suppressed"]
                    and (record.get("check") or {}).get("failed_stage") == "sweep"):
                raise ActionRefused(
                    "previous measured recovery sweep remains blocked; holding",
                    code="RETREAT_UNAVAILABLE", details={"recovery_check": record["check"]})

        execution_radius = self.capabilities.execution_radius_m
        grids = grid_cache.setdefault((radius_m, execution_radius), {})
        initial = _retreat_corridor(art, obstacles, (x, y), yaw, 0., radius_m,
                                   grids=grids, execution_radius_m=execution_radius)
        v, u = np.nonzero(art.occupancy)
        blocked_xy = px_to_world(art.T_map_px, np.column_stack((u, v)))
        tree = cKDTree(blocked_xy) if len(v) else None
        for offset in (math.pi, 0., math.pi / 2, -math.pi / 2,
                       math.pi / 4, -math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4):
            heading = yaw + offset
            failures = [r for r in history if r["suppressed"] and abs(math.atan2(
                math.sin(heading-r["heading"]), math.cos(heading-r["heading"])))
                < RECOVERY_MEMORY_HEADING_RAD]
            if failures:
                continue
            for distance in (1., .7, .4, .2):
                self.port.guard(epoch)
                if ctx.deadline.expired():
                    raise ActionRefused("recovery action deadline reached", code="TIMEOUT")
                check = _retreat_corridor(art, obstacles, (x, y), heading-math.pi,
                                          distance, radius_m, grids=grids,
                                          execution_radius_m=execution_radius)
                if check["error"]:
                    continue
                end = (x + distance * math.cos(heading), y + distance * math.sin(heading))
                if any(math.dist(end, r["origin"]) < RECOVERY_MEMORY_DISTANCE_M
                       for r in history):
                    continue  # no return to a recent origin
                clearance = tree.query(end)[0] if tree is not None else 2.
                gain = 0. if goal_xy is None else math.dist((x, y), goal_xy)-math.dist(end, goal_xy)
                score = min(float(clearance), 1.2) + .2 * gain - .1 * abs(math.sin(offset))
                candidates.append((score, distance, heading, end))
        candidates.sort(reverse=True)
        if (goal_xy is not None and "full" in grids
                and not grids["full"].is_safe(*goal_xy) and grids["full"].is_safe(x, y)):
            raise ActionRefused("accepted goal is blocked; current departure remains usable",
                                code="RECOVERY_UNAVAILABLE")
        # prefer candidates A* connects to the goal
        if goal_xy is not None and candidates and grids["full"].is_safe(*goal_xy):
            def cancelled():
                self.port.guard(epoch)
                return ctx.deadline.expired()

            connected = []
            for candidate in candidates[:8]:
                if cancelled():
                    raise ActionRefused("recovery search deadline reached", code="TIMEOUT")
                if plan_grid_astar(candidate[3], tuple(goal_xy), grids["full"],
                                   cancelled=cancelled) is not None:
                    connected.append(candidate)
            candidates = connected or candidates
        ctx.log("recovery.candidates", {"pose": [x, y, yaw], "map_version": version,
                "radius_m": radius_m, "goal_xy": goal_xy,
                "execution_radius_m": execution_radius,
                "candidates": [{"score": c[0], "distance_m": c[1], "heading": c[2],
                                "endpoint": c[3]} for c in candidates]})
        attempted = set()
        for _, distance, heading, end in candidates:
            direction = round(heading, 6)
            if direction in attempted:
                continue
            attempted.add(direction)
            self.port.fresh(ctx, epoch, require_map=True)
            record = {"epoch": epoch, "origin": (x, y), "heading": heading,
                      "suppressed": False}
            history.append(record)
            del history[:-RECOVERY_MEMORY_LIMIT]
            mode = "padding escape" if initial["margin_escape"] else "staging move"
            self.port.stage("repairing", f"checked {mode} up to {distance:.1f} m")
            try:
                result = self._checked_retreat(ctx, radius_m,
                    travel_heading=heading, target_distance=distance)
            except ActionRefused as exc:
                ctx.log("recovery.candidate_refused", {"heading": heading, "reason": str(exc),
                        "code": exc.code, "details": exc.details})
                if (exc.code.startswith("TASK_")
                        or exc.details.get("recovery_check", {}).get("failed_stage") == "sweep"):
                    raise
                if exc.code == "OBSERVATIONS_UNAVAILABLE":
                    raise  # The outer loop owns the bounded held observation retry.
                # the measured pose after the refusal
                current = ctx.observe("base_pose")
                self.port.guard(epoch)
                if getattr(self.state, "_escape_episode_origin", None) is None:
                    self.state._escape_episode_origin = tuple(current[:2])
                record.update(suppressed=True, failure_code=exc.code,
                              settled_pose=list(current),
                              check=exc.details.get("recovery_check"))
                ctx.log("recovery.memory", {"reason": "maneuver_refused",
                        "episode_origin": self.state._escape_episode_origin,
                        "origin": record["origin"], "heading": heading,
                        "pose": list(current), "code": exc.code,
                        "check": record["check"]})
                if math.dist(current[:2], (x, y)) > 1e-6:
                    continuation = self._recovery_departure(ctx, radius_m, goal_xy, **request)
                    if continuation is not None:
                        return {"kind": "checked_staging", "heading": heading,
                                "maneuver_reason": str(exc), **continuation}
                if math.dist(current[:2], (x, y)) > localizer.SWAY_M:
                    history.append({"epoch": epoch, "origin": tuple(current[:2]),
                                    "heading": heading, "suppressed": False})
                    del history[:-RECOVERY_MEMORY_LIMIT]
                    ctx.log("recovery.rebuild", {"from_pose": [x, y, yaw],
                            "pose": list(current), "reason": str(exc),
                            "remaining_s": ctx.deadline.remaining()})
                    return None
                continue
            if result:
                continuation = self._recovery_departure(ctx, radius_m, goal_xy, **request)
                if continuation is not None:
                    return result | {"kind": "checked_staging", "heading": heading,
                                     "maneuver_outcome": result["outcome"], **continuation}
                current = gait.pose(ctx)
                if math.dist(current[:2], (x, y)) > localizer.SWAY_M:
                    history.append({"epoch": epoch, "origin": tuple(current[:2]),
                                    "heading": heading, "suppressed": False})
                    del history[:-RECOVERY_MEMORY_LIMIT]
                    return None
        raise ActionRefused("no further checked maneuver restores a usable departure",
                            code="RECOVERY_UNAVAILABLE")


    def record_measured_sweep_failure(self, ctx, epoch, check, code):
        """Record a measured sweep failure once per check in the epoch's bounded escape history."""
        if self.state._escape_history_epoch != epoch:
            self.state._escape_attempts = []
            self.state._escape_history_epoch = epoch
        if not any(r["suppressed"] and r.get("check") == check
                   for r in self.state._escape_attempts):
            while len(self.state._escape_attempts) >= RECOVERY_MEMORY_LIMIT:
                removable = next((i for i, record in enumerate(self.state._escape_attempts)
                    if not (record["suppressed"]
                            and (record.get("check") or {}).get("failed_stage") == "sweep")), None)
                if removable is None:
                    ctx.log("recovery.memory", {"reason": "witness_capacity_invariant",
                                               "check": check})
                    raise RuntimeError("Measured recovery witness capacity was not reserved")
                del self.state._escape_attempts[removable]
            self.state._escape_attempts.append({"epoch": epoch,
                "origin": tuple(check["previous_xy"]), "heading": check["yaw"] + math.pi,
                "suppressed": True, "failure_code": code, "check": check})
            self.state._escape_episode_origin = tuple(check["previous_xy"])
            ctx.log("recovery.memory", {"reason": "measured_sweep_failed", "check": check})

    def _escape_history(self, ctx, epoch, xy, art, obstacles, version, *, grids=None):
        """Retire refused attempts after material progress or once their refused check is clear."""
        history = getattr(self.state, "_escape_attempts", [])[-RECOVERY_MEMORY_LIMIT:]
        if getattr(self.state, "_escape_history_epoch", None) != epoch:
            history = []
            self.state._escape_episode_origin = None
            self.state._escape_history_epoch = epoch
        anchor = getattr(self.state, "_escape_episode_origin", None)
        moved = anchor is not None and math.dist(xy, anchor) >= RECOVERY_MEMORY_DISTANCE_M
        grids = {} if grids is None else grids
        for record in history:
            if not record["suppressed"]:
                continue
            self.port.guard(epoch)
            if ctx.deadline.expired():
                raise ActionRefused("recovery action deadline reached", code="TIMEOUT")
            check = record.get("check")
            measured_failure = (check or {}).get("failed_stage") == "sweep"
            reason = "material_displacement" if moved and not measured_failure else None
            if reason is None and check is not None:
                cached = grids.setdefault((check["radius_m"], check["execution_radius_m"]), {})
                if check.get("allow_egress") is False and measured_failure:
                    _retreat_grids(art, obstacles, check["radius_m"], check["execution_radius_m"], cached)
                    current = {"sweep": cached["motion"].check_motion(
                        check["previous_xy"], check["xy"], allow_egress=False)}
                else:
                    current = _retreat_corridor(art, obstacles, check["xy"], check["yaw"],
                        check["remaining_m"], check["radius_m"], check["previous_xy"],
                        grids=cached, execution_radius_m=check["execution_radius_m"])
                if current.get(check["failed_stage"], {}).get("safe", False):
                    reason = "refused_geometry_now_clear"
            if reason is not None:
                record["suppressed"] = False
            else:
                reason = ("refused_geometry_still_blocked" if check is not None
                          else "no_material_displacement")
            ctx.log("recovery.memory", {"reason": reason, "episode_origin": anchor,
                    "origin": record["origin"], "heading": record["heading"],
                    "pose": list(xy), "map_version": version,
                    "suppressed": record["suppressed"]})
        if moved:
            self.state._escape_episode_origin = None
        self.state._escape_attempts = history
        return history


    def _checked_retreat(self, ctx, radius_m: float, *, travel_heading=None,
                         target_distance=1.0) -> dict:
        """One checked translation along `travel_heading`, straight back when it is omitted."""

        epoch = self.port.epoch()
        requested_distance, speed, past = target_distance, 0.15, 0.15
        execution_radius = self.capabilities.execution_radius_m
        distance, time_limit = requested_distance, 10.0
        started = ctx.now()
        first_check = True
        settled = False
        partial = False
        padding_escape = False
        grids, grid_version = {}, None
        try:
            x0, y0, yaw0 = gait.pose(ctx)
            heading = yaw0 + math.pi if travel_heading is None else travel_heading
            if any(r["suppressed"] and (r.get("check") or {}).get("failed_stage") == "sweep"
                   for r in self.state._escape_attempts):
                art, obstacles, version = self.port.recovery_map()
                history = self._escape_history(ctx, epoch, (x0, y0), art, obstacles, version)
                for record in history:
                    if (record["suppressed"]
                            and (record.get("check") or {}).get("failed_stage") == "sweep"):
                        raise ActionRefused(
                            "previous measured recovery sweep remains blocked; holding",
                            code="RETREAT_UNAVAILABLE", details={"recovery_check": record["check"]})
            hx, hy = math.cos(heading), math.sin(heading)
            previous_xy = (x0, y0)
            observer = getattr(ctx, "observe", None)
            generation = getattr(observer, "recovery_generation", 0)
            waits = ctx.gate.waits

            def checked_pose():
                self.port.guard(epoch)
                x, y, yaw = gait.pose(ctx)
                if (ctx.gate.waits != waits
                        or getattr(observer, "recovery_generation", 0) != generation):
                    raise Stopped("checked retreat localization changed; holding for a fresh plan")
                if self.port.stopped():
                    raise Stopped("checked retreat interrupted")
                if ctx.deadline.expired():
                    raise ActionRefused("checked retreat action deadline reached", code="TIMEOUT")
                yaw_error = (yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
                if abs(yaw_error) > math.radians(10):
                    raise ActionRefused("checked retreat exceeded its heading bound",
                                        code="RETREAT_UNAVAILABLE")
                dx, dy = x - x0, y - y0
                along = hx * dx + hy * dy
                if along < -localizer.SWAY_M or along > distance + past:
                    raise ActionRefused("checked retreat exceeded its progress or time bound",
                                        code="RETREAT_UNAVAILABLE")
                return x, y, yaw, dx, dy, along

            ctx.log("recovery.request", {"pose": [x0, y0, yaw0], "distance_m": requested_distance,
                                         "speed_mps": speed, "radius_m": radius_m,
                                         "planning_radius_m": radius_m,
                                         "motion": "recovery translation",
                                         "execution_radius_m": execution_radius})
            while True:
                self.port.guard(epoch)
                art, obstacles, map_version = self.port.recovery_map()
                blocked = art.occupancy
                if map_version is None or map_version != grid_version:
                    grids, grid_version = {}, map_version
                # Full-map construction must precede the execution pose acquisition.
                _retreat_grids(art, obstacles, radius_m, execution_radius, grids)
                x, y, yaw, dx, dy, along = checked_pose()
                if ctx.now() - started > time_limit and not settled:
                    if along <= 0:
                        raise ActionRefused("checked recovery made no progress in time",
                                            code="RETREAT_UNAVAILABLE")
                    # out of time: stop and validate the partial distance
                    partial = along < distance
                    gait.hold(ctx, 0.2, 0.0, 0.0, 0.0)
                    settled = True
                    continue
                if settled and (along <= 0 if partial else along < distance):
                    raise ActionRefused(
                        "checked retreat did not retain its progress after stopping",
                        code="RETREAT_UNAVAILABLE")
                if first_check:
                    # the longest clear distance, chosen once
                    candidates = []
                    for tenths in range(round(10 * requested_distance), 1, -1):
                        distance = tenths / 10.
                        remaining = max(0., distance - along)
                        check = _retreat_corridor(art, obstacles, (x, y), heading-math.pi,
                                                  remaining, radius_m, previous_xy, grids=grids,
                                                  execution_radius_m=execution_radius)
                        candidates.append({"distance_m": distance, "error": check["error"]})
                        if not check["error"]:
                            break
                    time_limit = min(10., max(3., 1.5 * distance / speed))
                    ctx.log("recovery.target", {
                        "requested_distance_m": requested_distance,
                        "validated_distance_m": distance if not check["error"] else None,
                        "time_limit_s": time_limit, "candidates": candidates})
                    padding_escape = check["margin_escape"]
                else:
                    remaining = 0. if partial else max(0., distance - along)
                    check = _retreat_corridor(art, obstacles, (x, y), heading-math.pi,
                                              remaining, radius_m, previous_xy, grids=grids,
                                              execution_radius_m=execution_radius)
                    if check["error"] and travel_heading is not None and along >= .05:
                        stopped = _retreat_corridor(art, obstacles, (x, y), heading-math.pi,
                                                   0., radius_m, previous_xy, grids=grids,
                                                   execution_radius_m=execution_radius)
                        if not stopped["error"]:
                            check, partial, remaining = stopped, True, 0.
                self.port.guard(epoch)
                if first_check:
                    map_age = self.port.map_age()
                    map_limit = self.capabilities.map_max_age_s
                    if map_age is None or not 0 <= map_age <= map_limit:
                        raise ActionRefused("checked retreat requires a fresh map",
                                            code="OBSERVATIONS_UNAVAILABLE")
                if ctx.deadline.expired():
                    raise ActionRefused("checked retreat action deadline reached", code="TIMEOUT")
                if (ctx.gate.waits != waits
                        or getattr(observer, "recovery_generation", 0) != generation):
                    raise Stopped("checked retreat localization changed; holding for a fresh plan")
                if first_check or check["error"]:
                    # the map masks as a diagnostic witness
                    witness = {"schema": 1, "frame": "exploration_start", "frame_epoch": epoch,
                               "map_version": map_version, "shape": list(blocked.shape),
                               "T_map_px": art.T_map_px.tolist(),
                               "previous_xy": list(previous_xy), "remaining_m": remaining,
                               "occupancy_packbits_hex": np.packbits(blocked).tobytes().hex(),
                               "center_allowed_packbits_hex": np.packbits(
                                   art.center_allowed).tobytes().hex(),
                               "obstacles_packbits_hex": np.packbits(obstacles).tobytes().hex()}
                    ctx.log("recovery.corridor", {**check, "pose": [x, y, yaw],
                                                 "travel_heading": heading,
                                                 "witness": witness})
                if not check["error"]:
                    # Selection/logging may take time. Reobserve, then check the actual sweep
                    # with the already-built geometry.
                    x, y, yaw, dx, dy, along = checked_pose()
                    remaining = 0. if partial else max(0., distance - along)
                    check = _retreat_corridor(art, obstacles, (x, y), heading-math.pi,
                        remaining, radius_m, previous_xy, grids=grids,
                        execution_radius_m=execution_radius)
                    self.port.guard(epoch)
                    if ctx.deadline.expired():
                        raise ActionRefused("checked retreat action deadline reached", code="TIMEOUT")
                    if settled and (along <= 0 if partial else along < distance):
                        raise ActionRefused(
                            "checked retreat did not retain its progress after stopping",
                            code="RETREAT_UNAVAILABLE")
                if check["error"]:
                    failed_stage = next(key for key in ("sweep", "corridor", "stopping")
                                        if key in check and not check[key]["safe"])
                    raise ActionRefused(check["error"], code="RETREAT_UNAVAILABLE", details={
                        "recovery_check": {"xy": [x, y], "yaw": heading-math.pi,
                            "previous_xy": list(previous_xy), "remaining_m": remaining,
                            "radius_m": radius_m, "execution_radius_m": execution_radius,
                            "failed_stage": failed_stage}})
                if ctx.now() - started > time_limit and not settled:
                    continue
                first_check = False
                previous_xy = (x, y)
                if along >= distance or partial:
                    if not settled:
                        gait.hold(ctx, 0.2, 0.0, 0.0, 0.0)
                        settled = True
                        continue
                    result = {"outcome": "partial" if partial else "complete",
                              "pose": [x, y, yaw], "distance_m": along,
                              "displacement_m": math.hypot(dx, dy),
                              "cross_track_m": -hy * dx + hx * dy,
                              "endpoint_error_m": math.hypot(dx-distance*hx, dy-distance*hy),
                              "padding_escape": padding_escape,
                              "execution_radius_m": execution_radius,
                              "requested_distance_m": requested_distance,
                              "validated_distance_m": distance,
                              "elapsed_s": ctx.now() - started}
                    ctx.log("recovery.partial" if partial else "recovery.complete", result)
                    return result
                vx, vy = speed * math.cos(heading-yaw), speed * math.sin(heading-yaw)
                vx, vy = (0. if abs(v) < 1e-9 else v for v in (vx, vy))
                if travel_heading is not None or max(abs(vx), abs(vy)) < speed:
                    vx, vy = gait.planar_above_deadband(vx, vy)
                gait.hold(ctx, 0.1, vx, vy, 0.0)
        except (Stopped, Refused, DriveFailure, HardwareUnavailable) as exc:
            check = exc.details.get("recovery_check") if isinstance(exc, ActionRefused) else None
            if check is not None and check["failed_stage"] == "sweep":
                self.record_measured_sweep_failure(ctx, epoch, check, exc.code)
            ctx.log("recovery.refused", {"error": str(exc)})
            if isinstance(exc, ActionRefused):
                raise
            cause = exc.__cause__ if isinstance(exc, DriveFailure) else exc
            if isinstance(cause, TaskScopeRefused):
                raise ActionRefused(str(exc), code=cause.code, details=cause.details) from exc
            if isinstance(cause, ObservationUnavailable):
                raise ActionRefused(str(exc), code="OBSERVATIONS_UNAVAILABLE") from exc
            if isinstance(cause, MotionBlocked):
                raise ActionRefused(f"checked retreat refused: {exc}",
                                    code="RETREAT_UNAVAILABLE", details=cause.details) from exc
            raise Stopped(f"checked retreat refused: {exc}") from exc
        finally:
            try:
                if not settled:
                    gait.hold(ctx, 0.2, 0.0, 0.0, 0.0)
            except (Refused, DriveFailure, HardwareUnavailable) as exc:
                ctx.log("recovery.hold_failure", {"error": str(exc)})
                raise Stopped(f"checked retreat could not hold: {exc}") from exc

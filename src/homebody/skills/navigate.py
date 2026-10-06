"""The navigate skill: route on the static map to a stance and verify arrival from localized poses."""
import math
import sys
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from homebody.helpers.geometry import wrap
from homebody.helpers.navigation import (
    cell_free,
    clearance,
    free_space,
    inside,
    line_free,
    pixel,
    route,
    square_stance,
)
from homebody.motion.carry import compact_carry, continuing, recover

from .contract import Result, SkillFailure

NEEDS = ("base_velocity", "arm_target", "advance", "stop")

FAILURES = ("NO_ROUTE", "STALL", "TIMEOUT", "ARRIVAL_NOT_VERIFIED", "CARRY_CLEARED", "HOLD_UNCERTAIN")
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
    requested_goal: np.ndarray | None = None
    departure_goal: np.ndarray | None = None
    run_in: np.ndarray | None = None


def prepare(arguments, selected, current, planning, reach=None):
    """Resolve the goal and facing, choose the free stance, plan the waypoints. REACH (m)
    replaces the configured stance reach."""
    settings = planning.settings.motion
    reach = settings.stance_reach if reach is None else reach
    requested, facing, yaw, snap = goal_arguments(arguments, selected.navigation)
    if facing is not None and np.linalg.norm(facing - requested) <= settings.facing_min_distance:
        facing = None
    navigation, pose = current.navigation, current.base_pose
    free = free_space(navigation, settings.robot_radius)
    goal, yaw = stance(arguments, navigation, free, requested, facing, yaw, snap, pose[:2], settings, reach)
    departure, path, run_in = route_waypoints(navigation, free, pose, goal, yaw, settings)
    return Plan(current.epoch, goal, yaw, path, requested_goal=requested,
                departure_goal=departure, run_in=run_in)


def execute(plan, ctx):
    """Drive the plan through Transit's phases one tick at a time."""
    settings = ctx.settings.motion
    transit = Transit(plan, ctx)
    while transit.frame.time - transit.start < settings.drive_timeout:
        transit.check_carry()
        transit.locate()
        result = getattr(transit, transit.phase)()
        if result is not None:
            return result
        transit.account_progress()
        ctx.advance(plan.epoch, settings.tick)
        transit.frame = ctx.measure(plan.epoch)
    raise SkillFailure("TIMEOUT", "Navigation deadline exceeded without verified arrival")


def goal_arguments(arguments, navigation):
    """The requested goal, the facing point (or None), the yaw (or None) and whether to snap.
    The bare x/y form defaults snap off."""
    bare = "x" in arguments or "y" in arguments
    if bare:
        if any(key in arguments for key in ("goal_map_1000", "goal_xy_m")):
            raise SkillFailure("INVALID_ARGUMENT", "Do not mix x/y with a goal click or goal_xy_m")
        if not all(finite_number(arguments.get(key)) for key in ("x", "y")):
            raise SkillFailure("INVALID_ARGUMENT", "navigate requires finite x and y in map metres")
        goal = np.array([arguments["x"], arguments["y"]], dtype=float)
    else:
        goal = map_point(arguments, "goal", navigation)
    facing = (map_point(arguments, "facing", navigation)
              if any(key in arguments for key in ("facing_map_1000", "facing_xy_m")) else None)
    yaw = arguments.get("yaw")
    if yaw is not None and not finite_number(yaw):
        raise SkillFailure("INVALID_ARGUMENT", "yaw must be finite radians")
    if yaw is not None and facing is not None:
        raise SkillFailure("INVALID_ARGUMENT", "Supply facing or yaw, not both")
    snap = arguments.get("snap", not bare)
    if type(snap) is not bool:
        raise SkillFailure("INVALID_ARGUMENT", "snap must be boolean")
    return goal, facing, yaw, snap


def map_point(arguments, name, navigation):
    """One map click or metric point as map metres, never both."""
    image_key, metric_key = name + "_map_1000", name + "_xy_m"
    keys = [key for key in (image_key, metric_key) if key in arguments]
    if len(keys) != 1:
        raise SkillFailure("INVALID_ARGUMENT", f"Supply exactly one {image_key} or {metric_key}")
    key = keys[0]
    value = arguments[key]
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            not all(finite_number(item) for item in value)):
        raise SkillFailure("INVALID_ARGUMENT", f"{key} requires two finite numbers")
    if key == metric_key:
        return np.asarray(value, dtype=float)
    if not all(0 <= item <= 1000 for item in value):
        raise SkillFailure("INVALID_ARGUMENT", f"{key} must lie in [0,1000]")
    height, width = navigation.occupied.shape
    point = navigation.T_map_px @ np.r_[np.asarray(value) * [width, height] / 1000., 1.]
    return np.round(point[:2] / point[2], 3)


def finite_number(value):
    return type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max


def stance(arguments, navigation, free, requested, facing, yaw, snap, start, settings, reach):
    """The goal and yaw to stand at, the yaw None when nothing fixes a heading."""
    if snap and facing is not None:
        hand = arguments.get("hand")
        if hand not in (None, "left", "right"):
            raise SkillFailure("INVALID_ARGUMENT", 'hand must be "left" or "right"')
        shift = {None: 0.0, "right": settings.right_hand_offset, "left": -settings.left_hand_offset}[hand]
        square = square_stance(navigation, facing, requested, reach, settings.stance_clearance, shift, free)
        if square is None:
            raise SkillFailure("NO_ROUTE", "No square stance clear of the faced edge on the clicked side")
        goal, yaw = square
    else:
        goal = requested
        if snap:
            goal = snapped_goal(navigation, free, requested, start,
                                settings.goal_snap_max_distance, settings.goal_snap_slack)
            if goal is None:
                raise SkillFailure("NO_ROUTE", "No free stance within the bounded goal snap distance")
        aim = requested if facing is None and not np.array_equal(goal, requested) else facing
        if yaw is None and aim is not None and np.linalg.norm(aim - goal) > 1e-6:
            yaw = math.atan2(aim[1] - goal[1], aim[0] - goal[0])
    return np.asarray(goal, float), yaw


def route_waypoints(navigation, free, pose, goal, yaw, settings):
    """The departure point (or None), the waypoints from there to GOAL, and the run-in point they
    pass (or None)."""
    departure = departure_point(navigation, free, pose, goal, settings)
    origin = pose[:2] if departure is None else departure
    head_on = run_in_point(navigation, free, goal, yaw, origin, settings)
    path, run_in = routed(navigation, origin, goal, head_on, settings)
    if path is None and departure is None:
        nearby = snapped_goal(navigation, free, pose[:2], pose[:2],
                              settings.start_snap_distance, settings.goal_snap_slack)
        if nearby is not None:
            path, run_in = routed(navigation, nearby, goal, head_on, settings)
    if path is None:
        raise SkillFailure("NO_ROUTE", "No collision-free route to requested base stance")
    return departure, path, run_in


class Transit:
    """One navigation, one tick at a time, through the phases depart, route, aim, close and settle."""

    def __init__(self, plan, ctx):
        self.plan, self.ctx, self.settings = plan, ctx, ctx.settings.motion
        settings = self.settings
        if ctx.carried:
            self.drive_speed = min(settings.drive_speed, settings.loaded_drive_speed)
            self.turn_speed = min(settings.turn_speed, settings.loaded_turn_speed)
        else:
            self.drive_speed, self.turn_speed = settings.drive_speed, settings.turn_speed
        self.frame = ctx.observe(plan.epoch)
        self.start = self.frame.time
        self.phase = "depart" if plan.departure_goal is not None else "route"
        self.waypoints, self.index = plan.waypoints, 0
        self.origin = self.frame.base_pose[:2].copy()
        self.navigation = self.frame.navigation
        self.free = free_space(self.navigation, settings.robot_radius)
        self.heading, self.anchor = 0.0, None
        self.settle_since, self.rounds, self.missed = None, 0, math.inf
        self.measure, self.best, self.progressed = None, math.inf, self.frame.time

    def depart(self):
        """Back straight out to the departure point, tuck a pending carry, then route from here."""
        plan, ctx, settings = self.plan, self.ctx, self.settings
        ctx.stage("Backing clear before turning")
        behind = body_frame(self.pose, plan.departure_goal - self.pose[:2])
        if behind[0] <= -settings.arrival_distance:
            self.send(-settings.final_speed, float(behind[1]), 0.0)
            return None
        for side in tuple(ctx.carried):
            if ctx.carried[side].get("compact_pending"):
                ctx.actions.base_velocity(plan.epoch, 0., 0., 0.)
                self.frame = compact_carry(ctx, plan.epoch, side)
                self.locate()
        target = plan.goal if plan.run_in is None else plan.run_in
        leg = planned_route(self.navigation, self.pose[:2], target, settings)
        if leg is not None:
            self.waypoints = through(leg, target, plan.goal)
        self.index, self.origin = 0, self.pose[:2].copy()
        return self.enter("route")

    def route(self):
        """Follow the waypoints, handing over to aim near the stance on the final leg."""
        plan, settings, pose = self.plan, self.settings, self.pose
        self.ctx.stage("Following route")
        if self.on_final_leg() and self.distance <= settings.aim_in_distance and checked_segment(
                self.navigation, self.free, pose[:2], plan.goal):
            return self.enter("aim")
        self.pass_waypoints()
        carrot, onto_leg = self.carrot()
        delta = carrot - pose[:2]
        self.heading = wrap(math.atan2(delta[1], delta[0]) - pose[2])
        if abs(self.heading) > settings.turn_before_translate:
            self.send(0.0, 0.0, math.copysign(self.turn_speed, self.heading))
        else:
            sideways = settings.route_lateral_gain * body_frame(pose, onto_leg)[1]
            self.send(self.drive_speed, float(np.clip(sideways, -settings.final_speed, settings.final_speed)),
                      settings.heading_gain * self.heading)
        return None

    def aim(self):
        """Turn in place onto the stance heading; within half the arrival angle, close."""
        settings = self.settings
        self.ctx.stage("Turning to face the stance heading")
        if abs(self.angle) <= settings.arrival_angle / 2:
            return self.enter("close")
        rate = max(settings.min_turn_speed, min(self.turn_speed, settings.heading_gain * abs(self.angle)))
        self.send(0.0, 0.0, math.copysign(rate, self.angle))
        return None

    def close(self):
        """Translate onto the stance with the heading held; stop within stop_distance."""
        settings = self.settings
        self.ctx.stage("Closing in on the stance")
        if self.distance <= min(settings.stop_distance, settings.arrival_distance / 2):
            self.settle_since = self.frame.time
            return self.enter("settle")
        body = body_frame(self.pose, self.plan.goal - self.pose[:2])
        speed = float(np.clip(settings.close_gain * self.distance, settings.min_close_speed, settings.final_speed))
        self.send(*(body * speed / self.distance), settings.heading_gain * self.angle)
        return None

    def settle(self):
        """Stand still for settle_seconds, then verify arrival. A miss re-aims or re-closes, at most
        close_rounds times."""
        plan, settings, distance, angle = self.plan, self.settings, self.distance, self.angle
        self.ctx.stage("Verifying arrival")
        self.send(0.0, 0.0, 0.0)
        if self.frame.time - self.settle_since < settings.settle_seconds:
            return None
        tolerance = settings.arrival_distance + (settings.close_slack if self.rounds else 0.)
        if distance <= tolerance and abs(angle) <= settings.arrival_angle:
            return Result("navigate", "OK", "Arrival verified after the gait settled",
                          {"distance": distance, "angle": abs(angle),
                           "base_pose": self.pose.tolist(),
                           "requested_goal": (plan.requested_goal if plan.requested_goal is not None
                                              else plan.goal).tolist(),
                           "resolved_goal": plan.goal.tolist(),
                           "resolved_yaw": plan.yaw})
        stalled = self.rounds and self.missed - distance < settings.close_min_gain
        self.rounds, self.missed = self.rounds + 1, distance
        if self.rounds > settings.close_rounds or stalled:
            raise SkillFailure("ARRIVAL_NOT_VERIFIED",
                               f"Settled {distance:.3f} m and {abs(angle):.3f} rad from the stance")
        self.phase = "aim" if abs(angle) > settings.arrival_angle / 2 else "close"
        return None

    def check_carry(self):
        """Recover a carry that stops reading as held. Cleared or uncertain ends the skill."""
        plan, ctx = self.plan, self.ctx
        for side in tuple(ctx.carried):
            if continuing(self.frame, ctx.arms[side], side, ctx.carried[side], ctx.settings.grasp,
                          look=lambda: ctx.observe(plan.epoch)):
                continue
            ctx.stage("Checking carry")
            status, self.frame = recover(ctx, plan.epoch, side)
            if status != "verified":
                code = "CARRY_CLEARED" if status == "cleared" else "HOLD_UNCERTAIN"
                raise SkillFailure(code, f"Navigation stopped to recheck carry: {status}; submit a fresh action")
            self.best = math.inf

    def locate(self):
        """The pose, and the distance and heading error to the stance, from the frame."""
        self.pose = self.frame.base_pose
        self.distance = float(np.linalg.norm(self.plan.goal - self.pose[:2]))
        self.angle = 0.0 if self.plan.yaw is None else wrap(self.plan.yaw - self.pose[2])

    def enter(self, phase):
        """Hand over to PHASE for this tick."""
        self.phase = phase
        return getattr(self, phase)()

    def send(self, forward, lateral, turn):
        """Command the base for this tick. A turn in place holds the position where it began."""
        settings, pose = self.settings, self.pose
        if forward == 0.0 and lateral == 0.0 and turn != 0.0:
            self.anchor = pose[:2].copy() if self.anchor is None else self.anchor
            forward, lateral = np.clip(settings.turn_hold_gain * body_frame(pose, self.anchor - pose[:2]),
                                       -settings.final_speed, settings.final_speed)
        else:
            self.anchor = None
        self.ctx.check(self.plan.epoch)
        self.ctx.actions.base_velocity(self.plan.epoch, float(np.clip(forward, -self.drive_speed, self.drive_speed)),
                                       float(np.clip(lateral, -self.drive_speed, self.drive_speed)),
                                       float(np.clip(turn, -self.turn_speed, self.turn_speed)))

    def carrot(self):
        """The point the route steers at, route_lookahead along the current leg, and the step from
        the base onto the leg."""
        end, pose = self.waypoints[self.index], self.pose[:2]
        start = self.waypoints[self.index - 1] if self.index else self.origin
        leg = end - start
        length = float(np.linalg.norm(leg))
        if length < 1e-6:
            return end, np.zeros(2)
        along = (pose - start) @ leg / length
        onto_leg = start + leg * (np.clip(along, 0.0, length) / length) - pose
        ahead = np.clip(along + self.settings.route_lookahead, 0.0, length)
        return start + leg * (ahead / length), onto_leg

    def on_final_leg(self):
        """Whether the stance may be aimed at directly."""
        plan = self.plan
        return (plan.run_in is None or self.index >= len(self.waypoints) - 1 or
                np.linalg.norm(self.pose[:2] - plan.run_in) < self.settings.waypoint_radius)

    def pass_waypoints(self):
        """Advance past reached or overshot corners, never past the run-in point."""
        waypoints, settings, pose = self.waypoints, self.settings, self.pose[:2]
        corners = len(waypoints) - (2 if self.plan.run_in is not None else 1)
        while self.index < len(waypoints) - 1:
            here, ahead = waypoints[self.index], waypoints[self.index + 1]
            reached = np.linalg.norm(pose - here) < settings.waypoint_radius
            passed = (self.index < corners and (pose - here) @ (ahead - here) > 0 and
                      checked_segment(self.navigation, self.free, pose, ahead))
            if not (reached or passed):
                return
            self.index += 1

    def remaining(self):
        """The stall measure's key (phase, turning) and the remaining distance in metres."""
        plan, settings, phase, pose = self.plan, self.settings, self.phase, self.pose[:2]
        turning = phase == "aim" or (phase == "route" and abs(self.heading) > settings.turn_before_translate)
        if turning:
            left = abs(self.angle if phase == "aim" else self.heading) * settings.robot_radius
        elif phase == "depart":
            left = np.linalg.norm(plan.departure_goal - pose)
        elif phase == "route":
            left = sum(np.linalg.norm(b - a) for a, b in pairwise([pose, *self.waypoints[self.index:]]))
        else:
            left = self.distance
        return (phase, turning), float(left)

    def account_progress(self):
        """The stall rule: the remaining measure must shrink by progress_distance within stall_timeout."""
        settings = self.settings
        if self.phase == "settle":
            self.measure = None
            return
        measure, remaining = self.remaining()
        if measure != self.measure:
            self.measure, self.best = measure, math.inf
        if remaining < self.best - settings.progress_distance:
            self.best, self.progressed = remaining, self.frame.time
        if self.frame.time - self.progressed <= settings.stall_timeout:
            return
        if self.phase != "close":
            raise SkillFailure("STALL", "No sustained progress toward the stance")
        self.phase, self.settle_since = "settle", self.frame.time
        self.send(0.0, 0.0, 0.0)


def snapped_goal(navigation, free, goal, start, max_distance, slack):
    """GOAL when free, else the free cell nearest START among those within SLACK of the nearest
    to GOAL. None when none lies within MAX_DISTANCE."""
    if cell_free(free, pixel(navigation, goal)):
        return goal
    cells = np.argwhere(free)
    points = np.column_stack((cells[:, ::-1], np.ones(len(cells)))) @ navigation.T_map_px.T
    points = points[:, :2] / points[:, 2, None]
    distance = np.linalg.norm(points - goal, axis=1)
    nearby = distance <= max_distance
    if not nearby.any():
        return None
    nearby &= distance <= distance[nearby].min() + slack
    choices = points[nearby]
    return choices[np.argmin(np.linalg.norm(choices - start, axis=1))]


def departure_point(navigation, free, pose, goal, settings):
    """Where the base backs straight out to before turning. None when it may turn where it
    stands or cannot back out."""
    start, cell = pose[:2], pixel(navigation, pose[:2])
    metric = clearance(navigation)
    turning_room = max(settings.stance_clearance + settings.arrival_distance,
                       settings.robot_radius + settings.route_clearance_margin)
    cramped = not inside(metric, cell) or metric[cell] < turning_room
    leaving = cramped and np.linalg.norm(goal - start) > settings.aim_in_distance
    if not leaving and cell_free(free, cell):
        return None
    heading = np.array([math.cos(pose[2]), math.sin(pose[2])])
    departure = start - settings.departure_distance * heading
    return departure if reverse_clear(navigation, free, start, departure, settings.robot_radius) else None


def run_in_point(navigation, free, goal, yaw, origin, settings):
    """The point run_in_distance behind the stance. None without a heading, from an origin
    already nearer, or when the straight leg to the stance is not free."""
    if yaw is None or np.linalg.norm(goal - origin) <= settings.run_in_distance:
        return None
    run_in = goal - settings.run_in_distance * np.array([math.cos(yaw), math.sin(yaw)])
    return run_in if checked_segment(navigation, free, run_in, goal) else None


def routed(navigation, origin, goal, run_in, settings):
    """The waypoints from ORIGIN to GOAL and the run-in point they pass (None when routed
    straight). None, None when nothing routes."""
    for target in (goal,) if run_in is None else (run_in, goal):
        path = planned_route(navigation, origin, target, settings)
        if path is not None:
            return through(path, target, goal), None if target is goal else run_in
    return None, None


def planned_route(navigation, origin, target, settings):
    """A route at the robot radius with route_clearance_margin where it can be kept."""
    return route(navigation, origin, target, settings.robot_radius, settings.route_clearance_margin)


def through(path, target, goal):
    """PATH ending exactly at TARGET, then on to GOAL when the target is a run-in point."""
    path[-1] = target
    return path if target is goal else [*path, goal]


def checked_segment(navigation, free, start, end):
    """Whether the straight segment START-END lies in free cells of the inflated map."""
    cells = (pixel(navigation, start), pixel(navigation, end))
    return all(cell_free(free, cell) for cell in cells) and line_free(*cells, free)


def reverse_clear(navigation, free, start, end, radius):
    """Whether the base can back straight from START to END. The check begins at the first free
    cell behind START, which must lie within RADIUS."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    distance = float(np.linalg.norm(end - start))
    direction = (end - start) / max(distance, 1e-9)
    for back in np.arange(0., min(radius, distance) + 1e-9, navigation.resolution):
        point = start + back * direction
        if cell_free(free, pixel(navigation, point)):
            return checked_segment(navigation, free, point, end)
    return False


def body_frame(pose, delta):
    """The map offset DELTA in the frame of the base at POSE: (ahead, left)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])

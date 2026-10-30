"""Navigation arguments and stance selection for task skills."""
import math
import sys

import numpy as np

from homebody.helpers.navigation import snapped_goal, square_stance

from .contract import SkillFailure


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

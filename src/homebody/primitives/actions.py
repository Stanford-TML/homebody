"""Command refusals and the argument checks a backend applies before it acts."""
import math

import numpy as np


class ActionRejected(RuntimeError):
    """A command refused: stale epoch, unavailable control or out-of-range argument."""


class PhysicsInvalid(ActionRejected):
    """Physics cannot continue in this epoch until the scene is reset."""


class ActionLimits:
    """Argument checks shared by backends. Each check returns the validated value as floats."""
    def __init__(self, settings, calibration):
        self.settings = settings
        self.arm_limits = {side: np.asarray(bounds)
                           for side, bounds in calibration["arm_limits"].items()}

    @staticmethod
    def epoch(requested, current):
        if type(requested) is not int or requested != current:
            raise ActionRejected("Command belongs to an expired execution generation")

    @staticmethod
    def hand(side):
        if side not in ("left", "right"):
            raise ActionRejected("Hand must be left or right")

    @staticmethod
    def numbers(values, length):
        data = np.asarray(values, dtype=object)
        numeric = (int, float, np.integer, np.floating)
        maximum = float(np.finfo(float).max)
        if data.shape != (length,) or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, numeric)
            or not -maximum <= value <= maximum for value in data
        ):
            raise ActionRejected(f"Command requires {length} finite numbers")
        return data.astype(float)

    def velocity(self, forward, lateral, yaw_rate):
        values = self.numbers([forward, lateral, yaw_rate], 3)
        limits = np.array([self.settings.motion.drive_speed, self.settings.motion.drive_speed,
                           self.settings.motion.turn_speed])
        if np.any(np.abs(values) > limits + 1e-9):
            raise ActionRejected("Base velocity exceeds configured limits")
        return values

    def arm(self, side, joints):
        self.hand(side)
        values = self.numbers(joints, 7)
        lower, upper = self.arm_limits[side]
        if np.any(values < lower) or np.any(values > upper):
            raise ActionRejected("Arm target exceeds joint limits")
        return values

    def closure(self, side, value):
        self.hand(side)
        result = float(self.numbers([value], 1)[0])
        if not 0 <= result <= 1 + self.settings.grasp.squeeze:
            raise ActionRejected("Gripper closure must lie between open and the squeeze past closed")
        return result

    def duration(self, seconds):
        result = float(self.numbers([seconds], 1)[0])
        if not 0 < result <= 1:
            raise ActionRejected("Advance duration must be between zero and one second")
        steps = result * self.settings.physics.control_hz
        if steps < 1 or not math.isclose(steps, round(steps), rel_tol=0., abs_tol=1e-7):
            raise ActionRejected("Advance duration must be whole controller periods")
        return result

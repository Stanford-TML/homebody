"""The robot interface skills use: an epoch, a clock, frames, measurements and five commands."""
from typing import Protocol, runtime_checkable

import numpy as np

from .observations import Frame, Measurement


@runtime_checkable
class Robot(Protocol):
    """Every command names its epoch. A refused command raises ActionRejected, and a robot
    that cannot continue without a reset raises PhysicsInvalid."""

    @property
    def epoch(self) -> int:
        """The command generation. A reset starts a new one and refuses older commands."""

    @property
    def time(self) -> float:
        """Seconds on the clock that stamps each Frame."""

    def snapshot(self) -> Frame:
        """The newest synchronized frame."""

    def measure(self) -> Measurement:
        """The newest joints, hands and poses, without rendering the camera."""

    def base_velocity(self, epoch: int, forward: float, lateral: float, yaw_rate: float) -> None:
        """Body-frame m/s and rad/s, dropped unless renewed within `physics.command_timeout`."""

    def arm_target(self, epoch: int, side: str, joints: np.ndarray) -> None:
        """One arm's seven joint positions in radians; the robot limits their rate and range."""

    def grip(self, epoch: int, side: str, closure: float) -> None:
        """One hand's closure: 0 open, 1 the calibrated close, up to `grasp.squeeze` beyond it."""

    def advance(self, epoch: int, seconds: float) -> None:
        """Block while SECONDS pass under the current commands: whole control periods, at most 1 s."""

    def stop(self, epoch: int) -> None:
        """Zero the base and hold both arms. The hands keep their closure."""

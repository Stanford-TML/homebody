"""The real Unitree G1 backend, a placeholder: `Robot` plus `reset` and `close`, every
member raising NotReleased."""
import numpy as np

from homebody.primitives.observations import Frame, Measurement
from homebody.primitives.release import NotReleased

WHAT = "The real-G1 backend"


class RealG1Backend:
    """The hardware Robot, driven by skills exactly as the simulator is."""

    def __init__(self, root, settings):
        raise NotReleased(WHAT)

    @property
    def epoch(self) -> int:
        """The command generation. An operator stop starts a new one."""
        raise NotReleased(WHAT)

    @property
    def time(self) -> float:
        """The robot's clock, which stamps each frame at capture."""
        raise NotReleased(WHAT)

    def snapshot(self) -> Frame:
        """The newest frame, all fields captured together."""
        raise NotReleased(WHAT)

    def measure(self) -> Measurement:
        """Joints, hands and poses between camera frames."""
        raise NotReleased(WHAT)

    def base_velocity(self, epoch: int, forward: float, lateral: float, yaw_rate: float) -> None:
        """The locomotion command, dropped by a deadman unless renewed."""
        raise NotReleased(WHAT)

    def arm_target(self, epoch: int, side: str, joints: np.ndarray) -> None:
        """One arm's joint reference in radians, rate- and range-limited."""
        raise NotReleased(WHAT)

    def grip(self, epoch: int, side: str, closure: float) -> None:
        """One Dex3 hand's closure."""
        raise NotReleased(WHAT)

    def advance(self, epoch: int, seconds: float) -> None:
        """Wait SECONDS of real time under the current commands."""
        raise NotReleased(WHAT)

    def stop(self, epoch: int) -> None:
        """Zero the base and hold both arms at their applied reference."""
        raise NotReleased(WHAT)

    def reset(self) -> None:
        """Start a new epoch."""
        raise NotReleased(WHAT)

    def close(self) -> None:
        """Release the robot's command authority."""
        raise NotReleased(WHAT)

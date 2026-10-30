"""Per-action clock and observation/actuation context."""
from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homebody.motion.exploration.localizer import PoseGate
    from homebody.primitives.navigation.actions.lower_body import LowerBody

class Deadline:
    """Robot-time budget. Reads the backend's clock, never the wall — a step's duration
    measured on the wall is a different number on a busier node for the same episode."""

    def __init__(self, now: Callable[[], float], seconds: float):
        self._now = now
        self._end = now() + seconds
        self._paused_at: float | None = None

    def remaining(self) -> float:
        return self._end - (self._now() if self._paused_at is None else self._paused_at)

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    @contextmanager
    def paused(self):
        """Stop the clock while nothing on the robot is being spent: the planner thinking,
        the user deciding. Sim time does not advance there anyway; wall time does."""
        if self._paused_at is not None:
            yield
            return
        self._paused_at = self._now()
        try:
            yield
        finally:
            self._end += self._now() - self._paused_at
            self._paused_at = None

@dataclass
class Ctx:
    """What a skill sees: no backend name and no config, so a skill cannot discover where it runs."""

    observe: Callable[[str], Any]
    # A transformed lower-body command may be returned after its successful step.
    step: Callable[[str, Any], LowerBody | None]
    now: Callable[[], float]
    deadline: Deadline
    progress: Callable[[str], None]
    log: Callable[..., None]
    gate: PoseGate
    wait: Callable[[float], None] = time.sleep
    max_footprint_radius_m: float = .60
    route_heading_gain_hz: float = 0.

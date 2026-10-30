"""Explicit manipulation limits and target-verification policy."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ManipulationPolicy:
    max_carried_objects: int | None = None
    target_or_refuse: bool = False

    def __post_init__(self):
        if self.max_carried_objects is not None and self.max_carried_objects < 1:
            raise ValueError("max_carried_objects must be positive")


REAL_MANIPULATION = ManipulationPolicy(max_carried_objects=1, target_or_refuse=True)

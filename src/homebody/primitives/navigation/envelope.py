"""The footprint-radius check shared by navigation and the backend."""
import math
from typing import Any


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")  # noqa: TRY004 -- radius errors are ValueError
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_radius_m(radius_m: float | None, *, maximum_m=.60) -> float:
    """A footprint within the composition's measured bound; exploration defaults to .60 m."""
    radius = _finite_float(radius_m, "radius_m")
    maximum = _finite_float(maximum_m, "maximum_m")
    if maximum < .20 or not 0.20 <= radius <= maximum:
        raise ValueError(f"radius_m must be within the configured 0.20..{maximum:.2f} m range")
    return radius

"""Score recorded physics against placement targets. Never import this module into skills."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path

SCHEMA = 1
SETTLE_SECONDS = 2.0
MAX_SAMPLE_GAP = 0.2
LINEAR_SPEED = 0.03
ANGULAR_SPEED = 0.15
CONTACT_PENETRATION = 0.015
MOTION_INTERVAL = 0.05
MAX_INTERIOR_PLANES = 32
DISTURBED_HORIZONTAL = 0.05  # m
DISTURBED_TURN = 0.35  # rad
MODEL_COMPLETED = "model_done"
SCRIPTED_COMPLETED = "skills_completed"
COMPLETED = {MODEL_COMPLETED, SCRIPTED_COMPLETED}


def vector(value: object, size: int) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"Expected finite vector of length {size}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not -sys.float_info.max <= v <= sys.float_info.max
           for v in value):
        raise ValueError("Nonfinite or nonnumeric vector")
    return value


def strings(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise ValueError("Expected contact/warning name list")
    return value


def targets(spec: dict) -> dict:
    if not isinstance(spec, dict) or spec.get("schema_version") != SCHEMA or spec.get("frame") != "ws_map":
        raise ValueError("Unsupported target schema or frame")
    objects = spec.get("objects")
    if not isinstance(objects, dict) or not objects:
        raise ValueError("At least one explicit target is required")
    for name, goal in objects.items():
        if not isinstance(name, str) or not name or not isinstance(goal, dict):
            raise ValueError("Invalid target object")
        bounds = goal["bounds"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("Target bounds must be [min, max]")
        low, high = (vector(b, 3) for b in bounds)
        if any(a >= b for a, b in zip(low, high)):
            raise ValueError("Target bounds have no volume")
        if "support_contacts" in goal:
            strings(goal["support_contacts"])
        if "interior_halfspaces" in goal:
            planes = goal["interior_halfspaces"]
            if not isinstance(planes, list) or not 1 <= len(planes) <= MAX_INTERIOR_PLANES:
                raise ValueError("Interior requires one to 32 plane equations")
            for plane in planes:
                normal = vector(plane, 4)[:3]
                size = math.hypot(*normal)
                if not math.isfinite(size) or size == 0:
                    raise ValueError("Interior plane requires a finite nonzero normal")
    return objects


def stacked_supports(objects: dict) -> dict:
    """Static supports each object rests on through other movable objects it touches."""
    touching = {name: set(strings(state.get("object_contacts", []))) for name, state in objects.items()}
    resolved = {}
    for name in objects:
        seen, queue, supports = {name}, [name], set()
        while queue:
            current = queue.pop()
            supports.update(strings(objects[current]["support_contacts"]))
            for other in touching[current]:
                if other in objects and other not in seen:
                    seen.add(other)
                    queue.append(other)
        resolved[name] = supports
    return resolved


def pose_speeds(state: dict, earlier: dict, seconds: float) -> tuple[float, float]:
    """Linear (m/s) and angular (rad/s) speed from two recorded poses SECONDS apart."""
    distance = math.dist(vector(state["position"], 3), vector(earlier["position"], 3))
    dot = abs(sum(a * b for a, b in zip(vector(state["quaternion_wxyz"], 4), vector(earlier["quaternion_wxyz"], 4))))
    return distance / seconds, 2 * math.acos(min(1.0, dot)) / seconds


def object_reason(state: dict, goal: dict, stacked: set | None = None,
                  speeds: tuple[float, float] | None = None, *, require_still: bool = False) -> str | None:
    """Why STATE is not placed in GOAL, or None."""
    position = vector(state["position"], 3)
    quat = vector(state["quaternion_wxyz"], 4)
    if abs(sum(v * v for v in quat) - 1) > 0.01:
        raise ValueError("Object quaternion is not normalized")
    corners = state["corners"]
    if not isinstance(corners, list) or len(corners) < 8:
        raise ValueError("At least eight transformed object extent points are required")
    corners = [vector(c, 3) for c in corners]
    linear = math.dist(vector(state["linear_velocity"], 3), [0, 0, 0])
    angular = math.dist(vector(state["angular_velocity"], 3), [0, 0, 0])
    if speeds is not None:
        linear, angular = speeds
    hands = strings(state["hand_contacts"])
    robot = strings(state["robot_contacts"])
    supports = set(strings(state["support_contacts"])) | (stacked or set())
    low, high = goal["bounds"]
    if "interior_halfspaces" in goal:
        floor = low[2] - CONTACT_PENETRATION
        if any(any(v < a or v > b for v, a, b in zip(c[:2], low[:2], high[:2])) or not floor <= c[2] <= high[2]
               for c in corners):
            return "outside_target"
    else:
        if (any(c[0] < low[0] or c[0] > high[0] or c[1] < low[1] or c[1] > high[1]
                or c[2] > high[2] for c in corners)
                or not low[2] <= position[2] <= high[2]):
            return "outside_target"
    for plane in goal.get("interior_halfspaces", []):
        scale = math.sqrt(sum(n * n for n in plane[:3]))
        for corner in corners:
            value = sum(n * v for n, v in zip(plane[:3], corner)) + plane[3]
            if not math.isfinite(value) or value > CONTACT_PENETRATION * scale:
                return "outside_target"
    if require_still:
        if linear > LINEAR_SPEED:
            return "moving"
        if angular > ANGULAR_SPEED:
            return "rotating"
    if hands:
        return "hand_contact"
    if robot:
        return "robot_contact"
    if not supports:
        return "unsupported"
    if "support_contacts" in goal and not set(supports).intersection(goal["support_contacts"]):
        return "wrong_support"
    return None


def disturbed(state: dict, start: dict, stacked: set) -> bool:
    """Whether an untargeted object moved, turned, fell to the floor, or is held or unsupported."""
    moved = math.dist(vector(state["position"], 3)[:2], vector(start["position"], 3)[:2])
    alignment = abs(sum(a * b for a, b in zip(vector(state["quaternion_wxyz"], 4), vector(start["quaternion_wxyz"], 4))))
    turned = 2 * math.acos(min(1.0, alignment)) > DISTURBED_TURN
    supports = set(strings(state["support_contacts"])) | stacked
    floored = "floor" in supports and "floor" not in strings(start["support_contacts"])
    held = strings(state["hand_contacts"]) or strings(state["robot_contacts"])
    return moved > DISTURBED_HORIZONTAL or turned or floored or bool(held) or not supports


def score_attempt(samples: Iterable[dict], target: dict, *, require_settled: bool = True) -> dict:
    """Score the placement over SAMPLES: still for SETTLE_SECONDS by default, else the last
    sample alone."""
    goals = targets(target)
    start = previous = epoch = None
    history = []
    count = 0
    invalid: set[str] = set()
    reasons = {name: "no_samples" for name in goals}
    starts, others = None, {}
    for sample in samples:
        count += 1
        try:
            if sample["schema_version"] != SCHEMA:
                raise ValueError("Unsupported sample schema")
            now = vector([sample["time"]], 1)[0]
            if now < 0:
                raise ValueError("Negative simulation time")
            sample_epoch = sample["epoch"]
            if type(sample_epoch) is not int or sample_epoch < 0:
                raise ValueError("Invalid epoch")
            if previous is not None and now <= previous:
                invalid.add("nonmonotonic_time")
            if epoch is not None and sample_epoch != epoch:
                invalid.add("reset_during_attempt")
            epoch = sample_epoch
            if sample["physics_valid"] is not True:
                invalid.add("invalid_physics")
            if sample["robot_upright"] is not True:
                invalid.add("robot_fall")
            if strings(sample["warnings"]):
                invalid.add("physics_warning")
            stacked = stacked_supports(sample["objects"])
            starts = starts or sample["objects"]
            if starts.keys() != sample["objects"].keys():
                raise ValueError("The recorded objects changed during the attempt")
            others = {name: "disturbed" if disturbed(state, starts[name], stacked[name]) else None
                      for name, state in sample["objects"].items() if name not in goals}
            earlier = next((row for row in reversed(history) if now - row["time"] >= MOTION_INTERVAL - 1e-9), None)
            speeds = {name: None if earlier is None else
                      pose_speeds(sample["objects"][name], earlier["objects"][name], now - earlier["time"])
                      for name in goals}
            reasons = {name: object_reason(sample["objects"][name], goal, stacked[name], speeds[name],
                                           require_still=require_settled)
                       for name, goal in goals.items()}
            history.append(sample)
            gap = previous is not None and now - previous > MAX_SAMPLE_GAP + 1e-9
            if gap:
                invalid.add("sample_gap")
            still = not any(others.values()) and all(
                object_reason(sample["objects"][name], goal, stacked[name], speeds[name],
                              require_still=True) is None for name, goal in goals.items())
            if not still or gap:
                start = None
            if still and start is None:
                start = now
            previous = now
        except (KeyError, TypeError, ValueError, AttributeError):
            invalid.add("invalid_sample")
            start = None
    duration = max(0.0, previous - start) if previous is not None and start is not None else 0.0
    criteria = {"max_sample_gap": MAX_SAMPLE_GAP, "disturbed_horizontal_max": DISTURBED_HORIZONTAL,
                "disturbed_turn_max": DISTURBED_TURN}
    if require_settled:
        criteria.update(settle_seconds=SETTLE_SECONDS, linear_speed_max=LINEAR_SPEED,
                        angular_speed_max=ANGULAR_SPEED)
    return {"schema_version": SCHEMA, "success": bool(count and not invalid and
            not any(reasons.values()) and not any(others.values()) and (not require_settled or duration >= SETTLE_SECONDS - 1e-9)),
            "scoring_policy": "settled_placement" if require_settled else "terminal_placement",
            "sample_count": count,
            "settled_seconds": duration, "invalid_reasons": sorted(invalid),
            "objects": {name: {"terminal_reason": reason} for name, reason in reasons.items()},
            "untargeted": {name: {"terminal_reason": reason} for name, reason in others.items()},
            "criteria": criteria}


def score_directory(path: Path, *, require_settled: bool = True) -> dict:
    """Score one attempt directory. An unreadable attempt is reported as unscored, and only
    a COMPLETED task can succeed."""
    result = {"attempt": path.name, "success": False}
    try:
        target = json.loads((path / "target.json").read_text())
        with (path / "physics.jsonl").open() as stream:
            result.update(score_attempt((json.loads(line) for line in stream if line.strip()), target,
                                        require_settled=require_settled))
        result["physical_success"] = result["success"]
        outcome = json.loads((path / "outcome.json").read_text())
        if not isinstance(outcome, dict) or not isinstance(outcome.get("status"), str) or not outcome["status"]:
            raise ValueError("Missing or invalid terminal lifecycle status")
        if vector([outcome["finished_unix"]], 1)[0] < 0:
            raise ValueError("Invalid terminal lifecycle time")
        result["execution_status"] = outcome["status"]
        result["success"] = result["physical_success"] and outcome["status"] in COMPLETED
        result["status"] = "scored"
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(success=False, status="unscored", error_type=type(error).__name__)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempts", type=Path, help="Parent directory containing ALL attempt directories")
    parser.add_argument("--out", type=Path, required=True, help="New report path (never overwrite evidence)")
    parser.add_argument("--terminal-only", action="store_true",
                        help="Score the last sample alone, without the two-second stillness window")
    args = parser.parse_args()
    attempts = [score_directory(path, require_settled=not args.terminal_only)
                for path in sorted(args.attempts.iterdir()) if path.is_dir()]
    report = {"schema_version": SCHEMA, "attempt_count": len(attempts),
              "success_count": sum(a["success"] for a in attempts), "attempts": attempts}
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "attempts"}))
    return 0 if attempts and all(a["success"] for a in attempts) else 1


if __name__ == "__main__":
    raise SystemExit(main())

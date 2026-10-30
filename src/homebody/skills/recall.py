"""Recall historical photographs near a map viewpoint, with valid empty results."""
from dataclasses import dataclass

import numpy as np

from homebody.memory import MemoryError

from .contract import Result, SkillFailure
from .remember import required

NEEDS = ()
PASSIVE = True
FAILURES = ("MEMORY_UNAVAILABLE", "MEMORY_INVALID", "MEMORY_IO")
PROMPT = ('recall: pose [x, y, yaw] in map metres and radians (defaults to the current pose), '
          'k photographs from 1 to 8 (default 1), op "retrieve" (default) or "remove". '
          'Spatial photo memory ranks stored viewpoints; it does not search description text. '
          'An absent result is valid. Returned photographs are historical: inspect current '
          'evidence before assuming an object is still present or reachable. Never move just to recall.')


@dataclass(frozen=True)
class Plan:
    epoch: int
    pose: tuple
    k: int
    op: str


def prepare(arguments, selected, current, planning):
    memory = required(planning)
    try:
        memory.binding.check_navigation(current.navigation)
    except MemoryError as error:
        raise SkillFailure("MEMORY_INVALID", str(error)) from error
    value = arguments.get("pose", current.base_pose)
    try:
        pose = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise SkillFailure("INVALID_ARGUMENT", "recall pose must be finite [x, y, yaw]") from error
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise SkillFailure("INVALID_ARGUMENT", "recall pose must be finite [x, y, yaw]")
    k, op = arguments.get("k", 1), arguments.get("op", "retrieve")
    if type(k) is not int or not 1 <= k <= 8 or op not in ("retrieve", "remove"):
        raise SkillFailure("INVALID_ARGUMENT", "recall requires k in 1..8 and op retrieve or remove")
    return Plan(current.epoch, tuple(pose.tolist()), k, op)


def execute(plan, ctx):
    ctx.check(plan.epoch)
    ctx.stage("Recalling historical views" if plan.op == "retrieve" else "Removing remembered views")
    try:
        found = required(ctx).recall(plan.pose, k=plan.k, op=plan.op)
    except MemoryError as error:
        raise SkillFailure("MEMORY_INVALID", str(error)) from error
    except OSError as error:
        raise SkillFailure("MEMORY_IO", f"Memory access failed: {error}") from error
    message = found["reason"] if found["absent"] else (
        f"{'Recalled' if plan.op == 'retrieve' else 'Removed'} {len(found['hits'])} historical views")
    return Result("recall", "OK", message, found)

"""Remember the current photograph and map-localized pose without commanding motion."""
from dataclasses import dataclass

from homebody.memory import MemoryError

from .contract import Result, SkillFailure

NEEDS = ()
PASSIVE = True
FAILURES = ("MEMORY_UNAVAILABLE", "MEMORY_INVALID", "MEMORY_IO")
PROMPT = ('remember: content, a short description of the current view. Saves this photograph, '
          'localized pose and available calibrated depth in the named persistent memory without '
          'moving. Store distinct useful views; this observation does not certify future presence.')


@dataclass(frozen=True)
class Plan:
    epoch: int
    content: str


def required(owner):
    memory = getattr(owner, "memory", None)
    if memory is None:
        raise SkillFailure("MEMORY_UNAVAILABLE", "No map-bound named memory is configured")
    return memory


def prepare(arguments, selected, current, planning):
    required(planning)
    content = arguments.get("content", "")
    if not isinstance(content, str):
        raise SkillFailure("INVALID_ARGUMENT", "remember content must be text")
    return Plan(current.epoch, content)


def execute(plan, ctx):
    ctx.stage("Remembering current view")
    frame = ctx.observe(plan.epoch)
    try:
        hit = required(ctx).remember(frame, plan.content)
    except MemoryError as error:
        raise SkillFailure("MEMORY_INVALID", str(error)) from error
    except OSError as error:
        raise SkillFailure("MEMORY_IO", f"Memory write failed; existing records retained: {error}") from error
    return Result("remember", "OK", "Saved the current localized photograph", hit)

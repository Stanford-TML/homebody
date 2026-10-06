"""The skill protocol (NEEDS, FAILURES, PROMPT, prepare, execute, optional abort) and the
prepare/execute lifecycle every registered skill runs."""
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from homebody.helpers.kinematics import Arm
from homebody.primitives.actions import ActionRejected, PhysicsInvalid
from homebody.primitives.observations import ObservationError
from homebody.primitives.robot import Robot
from homebody.session.settings import Settings

DECLARATIONS = ("NEEDS", "FAILURES", "PROMPT", "prepare", "execute")


class SkillFailure(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Result:
    """A skill's one terminal outcome, under its registered name."""
    skill: str
    code: str
    message: str
    details: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.code == "OK"


@dataclass(frozen=True)
class Planning:
    """What preparation reads besides frames."""
    arms: Mapping[str, Arm]
    jaw_samples: Mapping[str, list]
    settings: Settings
    carried: Mapping[str, dict]
    cancelled: Callable[[], bool] = lambda: False


@dataclass
class Context:
    """Execution's only handle on the robot."""
    observations: Robot
    actions: Robot
    arms: Mapping[str, Arm]
    settings: Settings = field(default_factory=Settings)
    jaw_samples: Mapping[str, list] = field(default_factory=dict)
    cancelled: Callable[[], bool] = lambda: False
    emit: Callable[[dict], None] = lambda event: None
    carried: dict = field(default_factory=dict)
    _skill_name: str | None = field(default=None, init=False, repr=False)
    _stage_name: str | None = field(default=None, init=False, repr=False)

    def planning(self):
        return Planning(self.arms, self.jaw_samples, self.settings, self.carried, self.cancelled)

    def stage(self, name):
        """Emit a stage event, once per distinct name."""
        if name != self._stage_name:
            self._stage_name = name
            self.emit({"type": "skill_stage", "skill": self._skill_name, "stage": name})

    def check(self, epoch):
        """End the skill as CANCELLED or STALE_EPOCH unless it may go on."""
        if self.cancelled():
            raise SkillFailure("CANCELLED", "Operator cancelled the skill")
        if epoch != self.observations.epoch:
            raise SkillFailure("STALE_EPOCH", "Scene changed; prepare a new action")

    def observe(self, epoch):
        """A current frame of EPOCH, or STALE_OBSERVATION."""
        frame = self._current(epoch, self.observations.snapshot)
        self.emit({"type": "observation", "frame": frame})
        return frame

    def measure(self, epoch):
        """The current measurements of EPOCH without the camera, or STALE_OBSERVATION."""
        return self._current(epoch, self.observations.measure)

    def _current(self, epoch, read):
        self.check(epoch)
        packet = read()
        self.check(epoch)
        if packet.epoch != epoch or not 0 <= self.observations.time - packet.time <= (
            self.settings.motion.observation_age
        ):
            raise SkillFailure("STALE_OBSERVATION", "Current sensor evidence is stale")
        return packet

    def advance(self, epoch, seconds=None):
        """Let SECONDS pass, one servo tick by default."""
        self.check(epoch)
        self.actions.advance(epoch, self.settings.servo.tick if seconds is None else seconds)
        self.check(epoch)


COMMON_FAILURES = ("CANCELLED", "STALE_EPOCH", "STALE_OBSERVATION", "INVALID_ARGUMENT",
                   "TARGET_LOST", "ACTION_REJECTED", "PHYSICS_INVALID", "COLLISION")


def declared(name, skill):
    """SKILL once its declarations fit the protocol under NAME, else TypeError."""
    missing = [declaration for declaration in DECLARATIONS if not hasattr(skill, declaration)]
    if missing:
        raise TypeError(f"Skill {name!r} does not declare {', '.join(missing)}")
    if not isinstance(skill.NEEDS, tuple) or not all(
            callable(getattr(Robot, command, None)) for command in skill.NEEDS):
        raise TypeError(f"Skill {name!r}: NEEDS must be a tuple of Robot commands: {skill.NEEDS!r}")
    if not isinstance(skill.FAILURES, tuple):
        raise TypeError(f"Skill {name!r}: FAILURES must be a tuple of result codes")
    if not skill.PROMPT.startswith(f"{name}: "):
        raise TypeError(f"Skill {name!r}: PROMPT must start with {name + ': '!r}")
    return skill


def refusal_code(error):
    """The declared code of a backend refusal."""
    return "PHYSICS_INVALID" if isinstance(error, PhysicsInvalid) else "ACTION_REJECTED"


class Runner:
    def __init__(self, context):
        self.context = context

    def run(self, name, arguments, selected_frame):
        """A declared result. A programming error propagates after the terminal event."""
        ctx = self.context
        ctx._skill_name, ctx._stage_name = name, None
        ctx.emit({"type": "skill_started", "skill": name, "frame_id": selected_frame.frame_id})
        attempt = Attempt(ctx, name, selected_frame)
        try:
            result = attempt.run(arguments)
        except BaseException as error:
            attempt.finish(Result(name, "ERROR", type(error).__name__, {"traceback": traceback.format_exc()}))
            raise
        return attempt.finish(result)


class Attempt:
    """One skill invocation and how far it got."""

    def __init__(self, ctx, name, selected_frame):
        from .registry import REGISTRY
        self.ctx, self.name, self.selected, self.epoch = ctx, name, selected_frame, selected_frame.epoch
        self.skill = REGISTRY.get(name)
        self.declared = COMMON_FAILURES + (() if self.skill is None else self.skill.FAILURES)
        self.plan, self.executing, self.aborting = None, False, False

    def run(self, arguments):
        """Prepare and execute, turning a typed failure into its declared result code."""
        try:
            return self.attempt(arguments)
        except SkillFailure as error:
            if error.code not in self.declared:
                raise RuntimeError(f"Skill raised an undeclared failure code: {error.code}") from error
            self.aborting = self.executing and error.code != "CANCELLED"
            return Result(self.name, error.code, str(error))
        except ObservationError as error:
            self.aborting = self.executing
            return Result(self.name, "TARGET_LOST" if self.executing else "INVALID_ARGUMENT", str(error))
        except (PhysicsInvalid, ActionRejected) as error:
            self.aborting = self.executing and not isinstance(error, PhysicsInvalid)
            return Result(self.name, refusal_code(error), str(error))

    def attempt(self, arguments):
        ctx = self.ctx
        ctx.stage("Preparing")
        if self.skill is None:
            raise SkillFailure("INVALID_ARGUMENT", f"Unknown skill {self.name!r}")
        current = ctx.observe(self.epoch)
        for command in self.skill.NEEDS:
            if not callable(getattr(ctx.actions, command, None)):
                raise SkillFailure("ACTION_REJECTED", f"Missing action capability: {command}")
        self.plan = self.skill.prepare(arguments, self.selected, current, ctx.planning())
        ctx.emit({"type": "skill_prepared", "skill": self.name, "plan": self.plan})
        ctx.check(self.epoch)
        self.executing = True
        result = self.skill.execute(self.plan, ctx)
        if result.skill != self.name or result.code not in {"OK", *self.declared}:
            raise RuntimeError("Skill returned an undeclared terminal result")
        return result

    def finish(self, result):
        """Abort if needed, hold or stop the base, and always emit the terminal event."""
        ctx = self.ctx
        try:
            if self.aborting and hasattr(self.skill, "abort"):
                result = aborted(self.skill, self.plan, ctx, result)
            if self.executing and ctx.observations.epoch == self.epoch:
                result = stopped(ctx, self.epoch, result)
        finally:
            ctx.emit({"type": "skill_finished", "result": result})
        return result


def aborted(skill, plan, ctx, result):
    """Run the skill's abort, keeping its own code and appending any abort problem to the message."""
    try:
        skill.abort(plan, ctx)
    except (SkillFailure, ObservationError, PhysicsInvalid, ActionRejected) as error:
        return Result(result.skill, result.code, f"{result.message}; abort: {error}", result.details)
    return result


def stopped(ctx, epoch, result):
    """Hold the base after success or stop everything after a failure. A rejected stop fails
    only a successful result."""
    try:
        if result.ok:
            ctx.actions.base_velocity(epoch, 0.0, 0.0, 0.0)
        else:
            ctx.actions.stop(epoch)
    except (PhysicsInvalid, ActionRejected) as error:
        return Result(result.skill, refusal_code(error) if result.ok else result.code,
                      f"{result.message}; stop rejected: {error}", result.details)
    return result


def side_argument(arguments):
    side = arguments.get("hand", "right")
    if side not in ("left", "right"):
        raise SkillFailure("INVALID_ARGUMENT", "hand must be left or right")
    return side


def click_argument(arguments, frame):
    point = arguments.get("point_normalized_1000")
    if (not isinstance(point, (list, tuple)) or len(point) != 2
            or any(type(value) not in (int, float) or not 0 <= value <= 1000 for value in point)):
        raise SkillFailure("INVALID_ARGUMENT", "point_normalized_1000 requires two numbers in [0, 1000]")
    u, v = frame.normalized_pixel(point)
    return frame.select(u, v), frame.point(u, v)

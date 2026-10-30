"""The skill protocol (NEEDS, FAILURES, PROMPT, prepare, execute, optional abort) and the
prepare/execute lifecycle every registered skill runs."""
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from homebody.primitives.actions import ActionRejected, PhysicsInvalid
from homebody.primitives.observations import ObservationError
from homebody.primitives.robot import Robot
from homebody.session.settings import Settings
from homebody.skills.manipulation_policy import ManipulationPolicy

if TYPE_CHECKING:  # annotations only: the arm kinematics load trimesh
    from homebody.helpers.kinematics import Arm

DECLARATIONS = ("NEEDS", "FAILURES", "PROMPT", "prepare", "execute")


class SkillFailure(RuntimeError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.details = {} if details is None else details


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
    arms: Mapping[str, "Arm"]
    jaw_samples: Mapping[str, list]
    settings: Settings
    carried: Mapping[str, dict]
    cancelled: Callable[[], bool] = lambda: False
    memory: object | None = None
    navigation: object | None = None
    manipulation: ManipulationPolicy = field(default_factory=ManipulationPolicy)
    target_selector: Callable | None = None
    navigation_evidence: Callable | None = None


@dataclass
class Context:
    """Execution's only handle on the robot."""
    observations: Robot
    actions: Robot
    arms: Mapping[str, "Arm"]
    settings: Settings = field(default_factory=Settings)
    jaw_samples: Mapping[str, list] = field(default_factory=dict)
    cancelled: Callable[[], bool] = lambda: False
    emit: Callable[[dict], None] = lambda event: None
    carried: dict = field(default_factory=dict)
    memory: object | None = None
    navigation: object | None = None
    manipulation: ManipulationPolicy = field(default_factory=ManipulationPolicy)
    target_selector: Callable | None = None
    navigation_evidence: Callable | None = None
    paused: Callable[[], bool] | None = None
    authority: Callable[[], None] | None = None
    navigation_state: object | None = None
    navigation_history: object | None = None
    _skill_name: str | None = field(default=None, init=False, repr=False)
    _stage_name: str | None = field(default=None, init=False, repr=False)

    def begin(self, name, skill):
        """Start SKILL's attempt under NAME: its stage events follow."""
        self._skill_name, self._stage_name = name, None

    def settle(self, epoch, result):
        """Hold or stop the base after an executed skill."""
        return stopped(self, epoch, result)

    def planning(self):
        return Planning(self.arms, self.jaw_samples, self.settings, self.carried, self.cancelled,
                        memory=self.memory, navigation=self.navigation,
                        manipulation=self.manipulation, target_selector=self.target_selector,
                        navigation_evidence=self.navigation_evidence)

    def stage(self, name):
        """Emit a stage event, once per distinct name."""
        if name != self._stage_name:
            self._stage_name = name
            self.emit({"type": "skill_stage", "skill": self._skill_name, "stage": name})

    def check(self, epoch):
        """End the skill as CANCELLED or STALE_EPOCH unless it may go on."""
        if self.cancelled():
            raise SkillFailure("CANCELLED", "Operator cancelled the skill")
        if self.paused is not None and self.paused():
            raise SkillFailure("PAUSED", "Operator paused; resume requires fresh preparation")
        if epoch != self.observations.epoch:
            raise SkillFailure("STALE_EPOCH", "Scene changed; prepare a new action")
        if self.authority is not None:
            self.authority()

    def observe(self, epoch):
        """A current frame of EPOCH, or STALE_OBSERVATION."""
        frame = self._current(epoch, self.observations.snapshot)
        observe_frame = getattr(self.navigation, "observe_frame", None)
        if callable(observe_frame):
            observe_frame(self, frame)
        self.emit({"type": "observation", "frame": frame})
        return frame

    def measure(self, epoch):
        """The current measurements of EPOCH without the camera, or STALE_OBSERVATION."""
        return self._current(epoch, self.observations.measure)

    def _current(self, epoch, read):
        self.check(epoch)
        packet = read()
        self.check(epoch)
        now, limit = self.observations.time, self.settings.motion.observation_age
        age = now - packet.time
        if packet.epoch != epoch or not 0 <= age <= limit:
            raise SkillFailure("STALE_OBSERVATION", "Current sensor evidence is stale",
                               {"observed_at": now, "captured_at": packet.time,
                                "age_s": age, "max_age_s": limit,
                                "packet_epoch": packet.epoch, "expected_epoch": epoch})
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
    surface = getattr(skill, "SURFACE", Robot)
    for declaration in ("NEEDS", "OPTIONAL"):
        commands = getattr(skill, declaration, ())
        if not isinstance(commands, tuple) or not all(
                callable(getattr(surface, command, None)) for command in commands):
            raise TypeError(f"Skill {name!r}: {declaration} must be a tuple of {surface.__name__} "
                            f"commands: {commands!r}")
    if not isinstance(skill.FAILURES, tuple):
        raise TypeError(f"Skill {name!r}: FAILURES must be a tuple of result codes")
    if getattr(skill, "PASSIVE", False) and (skill.NEEDS or getattr(skill, "OPTIONAL", ())):
        raise TypeError(f"Skill {name!r}: passive skills cannot declare actuation")
    if not skill.PROMPT.startswith(f"{name}: "):
        raise TypeError(f"Skill {name!r}: PROMPT must start with {name + ': '!r}")
    return skill


def refusal_code(error):
    """The declared code of a backend refusal."""
    return "PHYSICS_INVALID" if isinstance(error, PhysicsInvalid) else "ACTION_REJECTED"


class Runner:
    """Runs the skills of REGISTRY, the SIM skills unless another registry is given, on one context."""

    def __init__(self, context, registry=None):
        if registry is None:
            from .registry import REGISTRY
            registry = REGISTRY
        self.context, self.registry = context, registry

    def run(self, name, arguments, selected_frame):
        """A declared result. A programming error propagates after the terminal event."""
        attempt = self.prepare(name, arguments, selected_frame)
        return attempt.result if attempt.finished else self._phase(attempt, attempt.execute)

    def prepare(self, name, arguments, selected_frame):
        """An attempt holding its plan for `execute`; a refused preparation has already finished it."""
        attempt = Attempt(self.context, name, selected_frame, self.registry.get(name))
        self.context.begin(name, attempt.skill)
        self.context.emit({"type": "skill_started", "skill": name, "frame_id": selected_frame.frame_id})
        self._phase(attempt, lambda: attempt.prepare(arguments))
        return attempt

    def execute(self, attempt):
        """Execute a prepared attempt once."""
        if attempt.finished:
            raise RuntimeError("Skill attempt already finished")
        self.context.begin(attempt.name, attempt.skill)
        return self._phase(attempt, attempt.execute)

    def cancel(self, attempt, message="Prepared action discarded"):
        """Finish a prepared attempt without executing it."""
        if attempt.finished:
            return attempt.result
        return attempt.finish(Result(attempt.name, "CANCELLED", message))

    def _phase(self, attempt, operation):
        try:
            result = attempt.guarded(operation)
        except BaseException as error:
            attempt.finish(Result(attempt.name, "ERROR", type(error).__name__,
                                  {"traceback": traceback.format_exc()}))
            raise
        return None if result is None else attempt.finish(result)


class Attempt:
    """One skill invocation and how far it got."""

    def __init__(self, ctx, name, selected_frame, skill):
        self.ctx, self.name, self.selected, self.epoch = ctx, name, selected_frame, selected_frame.epoch
        self.skill = skill
        self.declared = COMMON_FAILURES + (() if self.skill is None else self.skill.FAILURES)
        if getattr(ctx, "paused", None) is not None:
            self.declared += ("PAUSED",)
        self.plan, self.executing, self.aborting = None, False, False
        self.finished, self.result = False, None

    def guarded(self, operation):
        """OPERATION's value, or its typed failure as the declared result code."""
        try:
            return operation()
        except SkillFailure as error:
            if error.code not in self.declared:
                raise RuntimeError(f"Skill raised an undeclared failure code: {error.code}") from error
            self.aborting = self.executing and error.code != "CANCELLED"
            return Result(self.name, error.code, str(error), error.details)
        except ObservationError as error:
            self.aborting = self.executing
            return Result(self.name, "TARGET_LOST" if self.executing else "INVALID_ARGUMENT", str(error))
        except (PhysicsInvalid, ActionRejected) as error:
            self.aborting = self.executing and not isinstance(error, PhysicsInvalid)
            return Result(self.name, refusal_code(error), str(error))

    def prepare(self, arguments):
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

    def execute(self):
        ctx = self.ctx
        ctx.check(self.epoch)
        self.executing = True
        result = self.skill.execute(self.plan, ctx)
        if result.skill != self.name or result.code not in {"OK", *self.declared}:
            raise RuntimeError("Skill returned an undeclared terminal result")
        return result

    def finish(self, result):
        """Abort if needed, then always try to hold or stop the base, and emit one terminal
        event. An unexpected abort or settle fault ends the attempt as ERROR, keeping the attempt's
        code in details["original_code"], and is raised after the terminal event."""
        ctx, faults = self.ctx, []
        try:
            if self.aborting and hasattr(self.skill, "abort"):
                try:
                    result = aborted(self.skill, self.plan, ctx, result)
                except Exception as error:  # noqa: BLE001 -- the base is settled before re-raising
                    faults.append(error)
        finally:
            try:
                if (self.executing and not getattr(self.skill, "PASSIVE", False)
                        and ctx.observations.epoch == self.epoch):
                    result = ctx.settle(self.epoch, result)
            except Exception as error:  # noqa: BLE001 -- one terminal event before re-raising
                faults.append(error)
            finally:
                if faults:
                    result = Result(result.skill, "ERROR", f"{result.message}; cleanup: " + "; ".join(
                        f"{type(error).__name__}: {error}" for error in faults),
                        {**result.details, "original_code": result.code,
                         "traceback": "".join(traceback.format_exception(faults[0]))})
                self.finished, self.result = True, result
                ctx.emit({"type": "skill_finished", "result": result})
        if faults:
            raise faults[0]
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

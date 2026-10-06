"""Typed operating settings. Load once; pass explicitly; record with every run."""
import math
import sys
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .config import (
    Agent,
    BlindFinish,
    Camera,
    Collision,
    Grasp,
    Navigate,
    Physics,
    Placement,
    Recording,
    Servo,
    SharedMotion,
    Topdown,
)


@dataclass(frozen=True)
class Motion(Navigate, SharedMotion):
    """Preserve the flat motion schema while its defaults have distinct owners."""


SIGNED = frozenset({"topdown.ahead", "topdown.left", "topdown.height", "placement.rest_arm_left",
                    "placement.rest_arm_right", "placement.release_twists", "grasp.approach_across"})
NONNEGATIVE = frozenset({"physics.noslip_iterations", "motion.route_clearance_margin",
                         "motion.left_hand_offset", "motion.right_hand_offset", "motion.close_slack",
                         "motion.close_min_gain", "grasp.spline_out", "placement.release_drops",
                         "agent.done_settle_seconds"})


def in_range(label, number):
    """True when NUMBER is finite and of LABEL's sign: any when SIGNED, zero or more when NONNEGATIVE, else positive."""
    if type(number) not in (int, float) or not number <= sys.float_info.max:
        return False
    return number > -sys.float_info.max if label in SIGNED else number >= 0 if label in NONNEGATIVE else number > 0


def check_value(label, value, default, variable_length):
    """Raise ValueError unless VALUE has DEFAULT's kind: nonempty text, or numbers in LABEL's range (a tuple of the default's length unless VARIABLE_LENGTH, integers where the default is one)."""
    if isinstance(default, str):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be nonempty text")
        return
    if isinstance(default, tuple) and (not isinstance(value, tuple) or not value
                                       or not variable_length and len(value) != len(default)):
        raise ValueError(f"{label} requires a tuple of numbers of the declared size")
    if not all(in_range(label, number) for number in (value if isinstance(default, tuple) else (value,))):
        sign = "" if label in SIGNED else "nonnegative " if label in NONNEGATIVE else "positive "
        raise ValueError(f"{label} must contain {sign}{'integers' if type(default) is int else 'finite numbers'}")
    if type(default) is int and type(value) is not int:
        raise ValueError(f"{label} must be an integer")


@dataclass(frozen=True)
class Settings:
    physics: Physics = field(default_factory=Physics)
    camera: Camera = field(default_factory=Camera)
    motion: Motion = field(default_factory=Motion)
    agent: Agent = field(default_factory=Agent)
    servo: Servo = field(default_factory=Servo)
    grasp: Grasp = field(default_factory=Grasp)
    collision: Collision = field(default_factory=Collision)
    placement: Placement = field(default_factory=Placement)
    topdown: Topdown = field(default_factory=Topdown)
    recording: Recording = field(default_factory=Recording)
    blind_finish: BlindFinish = field(default_factory=BlindFinish)

    def __post_init__(self):
        for group in fields(self):
            section = getattr(self, group.name)
            if type(section) is not group.type:
                raise ValueError(f"{group.name} must be a {group.type.__name__} settings group")
            defaults = group.type()
            for item in fields(section):
                check_value(f"{group.name}.{item.name}", getattr(section, item.name),
                            getattr(defaults, item.name), item.type == tuple[float, ...])
        p = self.physics
        if p.hand_contact[0] < 2 * p.timestep:
            raise ValueError("Hand contact time constant must be at least twice the physics timestep")
        if p.solver not in ("PGS", "Newton", "CG"):
            raise ValueError("Unknown MuJoCo solver")
        if p.integrator not in ("Euler", "RK4", "implicit", "implicitfast"):
            raise ValueError("Unknown MuJoCo integrator")
        if p.cone not in ("pyramidal", "elliptic"):
            raise ValueError("Unknown MuJoCo friction cone")
        ticks = 1 / (p.control_hz * p.timestep)
        if p.control_hz % p.policy_hz or ticks < 1 or not math.isclose(ticks, round(ticks)):
            raise ValueError("Physics, arm and policy clocks must divide exactly")
        if self.camera.horizontal_fov >= 179 or self.camera.near >= self.camera.far:
            raise ValueError("Camera requires field of view below 179 and near < far")
        if not 0 < self.grasp.shape_quantiles[0] < self.grasp.shape_quantiles[1] < 1:
            raise ValueError("Shape quantiles must be ordered between zero and one")
        if not p.min_base_height < p.max_base_height or p.max_tilt_degrees >= 90:
            raise ValueError("Base height limits must be ordered and tilt limit below 90 degrees")
        if self.placement.release_height > self.placement.max_release_height:
            raise ValueError("Release height exceeds its limit")
        if self.motion.final_speed > self.motion.drive_speed:
            raise ValueError("Final approach speed exceeds the drive speed limit")
        if not self.motion.stop_distance < self.motion.arrival_distance:
            raise ValueError("Navigation must stop inside its arrival distance")
        for value in (p.hand_torque_fraction, self.grasp.depth_fraction,
                      self.grasp.hold_effort_fraction, self.grasp.hold_closure_gap,
                      self.grasp.occlusion_min_rays, self.grasp.occlusion_missing_explained,
                      self.grasp.occlusion_exposed_missing_max,
                      self.grasp.occlusion_visible_match, self.grasp.occlusion_no_view_rays,
                      self.placement.open_closure_limit):
            if value > 1:
                raise ValueError("Force, depth and closure fractions must not exceed one")
        if not 2 <= self.grasp.recovery_confirmations <= self.grasp.recovery_samples:
            raise ValueError("Carry recovery requires at least two confirmations within the sample budget")
        if self.grasp.approach_attempts > 2:
            raise ValueError("Grasp approach permits at most two attempts")
        if self.grasp.occlusion_no_view_rays <= self.grasp.occlusion_min_rays:
            raise ValueError("Fully hidden target requires stronger measured arm occlusion")
        if min(self.grasp.spline_samples, self.grasp.spline_waypoints) < 2:
            raise ValueError("Spline sampling requires at least two points")
        if self.grasp.spline_seed_bias > 1:
            raise ValueError("Spline seed bias must not exceed one")
        if any(bounds[0] >= bounds[1] for bounds in (self.topdown.ahead, self.topdown.left,
                                                    self.topdown.height)):
            raise ValueError("Top-down bounds must be ordered")
        if any(round((bounds[1] - bounds[0]) * self.topdown.pixels_per_metre) < 2
               for bounds in (self.topdown.ahead, self.topdown.left)):
            raise ValueError("Top-down image dimensions must have at least two pixels")
        if max(self.motion.tick, self.servo.tick, self.placement.tick, self.grasp.recovery_tick) > 1:
            raise ValueError("Skill ticks must fit the one-second action limit")
        for period in (self.motion.tick, self.servo.tick, self.placement.tick,
                       self.grasp.recovery_tick):
            steps = period * p.control_hz
            if steps < 1 or not math.isclose(steps, round(steps), rel_tol=0., abs_tol=1e-9):
                raise ValueError("Skill ticks must be whole controller periods")

    @classmethod
    def load(cls, path: Path | None = None, scene: Path | None = None,
             override: Path | None = None) -> "Settings":
        """Base settings, then the scene package's `settings.toml`, then OVERRIDE (FileNotFoundError when missing). Tables merge key by key, anything else replaces."""
        raw = tomllib.loads(path.read_text()) if path else {}
        for overlay in (scene / "settings.toml" if scene else None, override):
            if overlay and overlay.is_file():
                for name, section in tomllib.loads(overlay.read_text()).items():
                    tables = isinstance(section, dict) and isinstance(raw.get(name, {}), dict)
                    raw[name] = {**raw.get(name, {}), **section} if tables else section
            elif overlay is override and override is not None:
                raise FileNotFoundError(f"Settings override not found: {override}")
        types = {group.name: group.type for group in fields(cls)}
        unknown = raw.keys() - types.keys()
        if unknown:
            raise ValueError(f"Unknown settings sections: {sorted(unknown)}")
        values = {}
        for name, kind in types.items():
            section = raw.get(name, {})
            if not isinstance(section, dict):
                raise TypeError(f"{name} must be a settings table")
            unknown = section.keys() - {item.name for item in fields(kind)}
            if unknown:
                raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
            defaults = asdict(kind())
            values[name] = kind(**{
                key: tuple(value) if isinstance(defaults[key], tuple) and isinstance(value, list)
                else value for key, value in section.items()
            })
        return cls(**values)

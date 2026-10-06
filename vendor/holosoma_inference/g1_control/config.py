# Vendored G1 controller kernel, derived from Holosoma and adapted by the authors; see THIRD_PARTY_NOTICES.md.
"""Typed configuration for the package-owned Unitree G1 controller kernel.

The numerical profiles in this module are derived from two independently
versioned sources:

* Amazon FAR Holosoma ``4ed2cebf9780f3efb59621657916006afadb87dc``
  for the native FastSAC policy, state layout, gains, and lifecycle.
* The authors' AMO/Holosoma adapter (see the repository's
  ``THIRD_PARTY_NOTICES.md``).  Its network assets are byte-identical to OpenTeleVision/AMO
  ``34caaf943660e6f9420e35f64e86dd56fb51dd0e``.

All behavior-bearing numbers live here.  Runtime modules consume immutable
profile objects instead of defining their own module-level constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class PolicyFamily(str, Enum):
    """Supported lower-body policy families."""

    fast_sac = "fast_sac"
    amo = "amo"


class HistoryOrder(str, Enum):
    """Placement of the current proprioceptive sample in AMO history."""

    append_current_before_flatten = "append_current_before_flatten"
    flatten_before_append_current = "flatten_before_append_current"


class HeadingControl(str, Enum):
    """Meaning of AMO's yaw command."""

    integrated_rate = "integrated_rate"


class GravityBehavior(str, Enum):
    """Failure policy for optional rigid-body gravity compensation."""

    best_effort = "best_effort"
    required = "required"
    disabled = "disabled"


@dataclass(frozen=True)
class RobotProfile:
    """G1 joint identity, default pose, gains, and motor mapping."""

    joint_names: tuple[str, ...]
    default_positions: tuple[float, ...]
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    joint_to_motor: tuple[int, ...]
    motor_to_joint: tuple[int, ...]
    joint_offsets_rad: tuple[float, ...]
    #: COMMANDED arm pose when nothing else owns the arms (the Y-init ramp
    #: target and the walking hold), 14 values for joints 15..28. None means
    #: "same as default_positions" — the historical behaviour. This is
    #: deliberately separate from default_positions because the POLICY
    #: OBSERVATION normalises measured joints against default_positions
    #: (amo.py: observed_positions - observed_default); changing the observed
    #: default would shift the network inputs, while changing only the
    #: commanded idle pose stays in-distribution — AMO conditions on the
    #: MEASURED arm pose and is trained to balance under arbitrary arm demos.
    idle_arm_positions: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        joint_count = len(self.joint_names)
        fields = (
            self.default_positions,
            self.stiffness,
            self.damping,
            self.joint_to_motor,
            self.motor_to_joint,
            self.joint_offsets_rad,
        )
        if joint_count != 29 or any(len(values) != joint_count for values in fields):
            raise ValueError("the G1 control profile must describe exactly 29 joints")
        if self.idle_arm_positions is not None and len(self.idle_arm_positions) != 14:
            raise ValueError(
                "idle_arm_positions must hold exactly the 14 arm joints (15..28)"
            )

    @property
    def idle_positions(self) -> tuple[float, ...]:
        """Full 29-joint pose to ramp to and hold when the arms are unowned."""
        if self.idle_arm_positions is None:
            return self.default_positions
        return tuple(self.default_positions[:15]) + tuple(self.idle_arm_positions)
        expected = list(range(joint_count))
        if sorted(self.joint_to_motor) != expected or sorted(self.motor_to_joint) != expected:
            raise ValueError("G1 joint/motor mappings must be permutations of 0..28")


@dataclass(frozen=True)
class LoopProfile:
    """Lifecycle and timing behavior shared by the controller loop."""

    control_hz: float
    initialization_ticks: int
    desired_base_height_m: float
    repeated_state_stop_ticks: int | None
    isolate_inference_errors: bool
    # The adapter's machine-seam envelope. The kernel
    # is tick-paced, so staleness is counted in ticks of control_hz rather than
    # wall time; the host hard-fails unless the profile runs at its declared Hz.
    velocity_staleness_s: float | None
    velocity_command_limit: float | None
    torso_height_range_m: tuple[float, float]
    torso_orientation_limit_rad: float

    def __post_init__(self) -> None:
        if self.control_hz <= 0 or self.initialization_ticks <= 0:
            raise ValueError("controller timing values must be positive")
        if self.velocity_staleness_s is not None and self.velocity_staleness_s <= 0:
            raise ValueError("velocity_staleness_s must be positive or None")
        if self.velocity_command_limit is not None and self.velocity_command_limit <= 0:
            raise ValueError("velocity_command_limit must be positive or None")
        low, high = self.torso_height_range_m
        if not (0.0 < low <= high):
            raise ValueError("torso_height_range_m must be an increasing positive range")
        if self.torso_orientation_limit_rad < 0.0:
            raise ValueError("torso_orientation_limit_rad must be non-negative")

    @property
    def velocity_staleness_ticks(self) -> int | None:
        """The adapter's 1.0 s dead-man window expressed in control ticks."""

        if self.velocity_staleness_s is None:
            return None
        return max(1, round(self.control_hz * self.velocity_staleness_s))


@dataclass(frozen=True)
class FastSacPolicyProfile:
    """Native Holosoma FastSAC inference contract."""

    model_artifact_id: str
    input_name: str
    output_name: str
    observation_terms: tuple[str, ...]
    observation_dimensions: tuple[tuple[str, int], ...]
    observation_scales: tuple[tuple[str, float], ...]
    action_scale: float
    raw_action_clip: float
    gait_period_s: float


@dataclass(frozen=True)
class AmoPolicyProfile:
    """The AMO observation, action, and safety contract."""

    policy_artifact_id: str
    adapter_artifact_id: str
    statistics_artifact_id: str
    gravity_urdf_artifact_id: str
    observed_joint_indices: tuple[int, ...]
    zero_velocity_indices: tuple[int, ...]
    lower_body_joint_count: int
    proprio_dimension: int
    demo_dimension: int
    privileged_dimension: int
    history_length: int
    extra_history_length: int
    action_scale: float
    raw_action_clip: float
    angular_velocity_scale: float
    joint_velocity_scale: float
    control_dt_s: float
    gait_frequency_hz: float
    in_place_velocity_threshold: float
    torso_height_m: float
    torch_threads: int
    jump_limits_rad: tuple[float, ...]
    history_order: HistoryOrder
    heading_control: HeadingControl
    gravity_behavior: GravityBehavior

    @property
    def observation_dimension(self) -> int:
        """Total width consumed by the AMO policy network."""

        return (
            self.proprio_dimension
            + self.demo_dimension
            + self.privileged_dimension
            + self.history_length * self.proprio_dimension
        )


@dataclass(frozen=True)
class G1ControllerProfile:
    """Complete fixed profile for one lower-body controller."""

    profile_id: str
    family: PolicyFamily
    robot: RobotProfile
    loop: LoopProfile
    policy: FastSacPolicyProfile | AmoPolicyProfile

    def __post_init__(self) -> None:
        if self.family is PolicyFamily.fast_sac and not isinstance(self.policy, FastSacPolicyProfile):
            raise ValueError("FastSAC controller profile requires FastSacPolicyProfile")
        if self.family is PolicyFamily.amo and not isinstance(self.policy, AmoPolicyProfile):
            raise ValueError("AMO controller profile requires AmoPolicyProfile")


def _joint_names() -> tuple[str, ...]:
    return (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )


def _robot_profile(
    default_positions: tuple[float, ...],
    stiffness: tuple[float, ...],
    damping: tuple[float, ...],
    idle_arm_positions: tuple[float, ...] | None = None,
) -> RobotProfile:
    identity = tuple(range(29))
    return RobotProfile(
        joint_names=_joint_names(),
        default_positions=default_positions,
        stiffness=stiffness,
        damping=damping,
        joint_to_motor=identity,
        motor_to_joint=identity,
        joint_offsets_rad=(0.0,) * 29,
        idle_arm_positions=idle_arm_positions,
    )


def fast_sac_profile() -> G1ControllerProfile:
    """Return native Holosoma's fixed G1 FastSAC deployment profile."""

    default_positions = (
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    )
    stiffness = (
        40.179238471,
        99.098427777,
        40.179238471,
        99.098427777,
        28.501246196,
        28.501246196,
        40.179238471,
        99.098427777,
        40.179238471,
        99.098427777,
        28.501246196,
        28.501246196,
        40.179238471,
        28.501246196,
        28.501246196,
        14.250623098,
        14.250623098,
        14.250623098,
        14.250623098,
        14.250623098,
        16.778327481,
        16.778327481,
        14.250623098,
        14.250623098,
        14.250623098,
        14.250623098,
        14.250623098,
        16.778327481,
        16.778327481,
    )
    damping = (
        2.557889765,
        6.308801854,
        2.557889765,
        6.308801854,
        1.814445687,
        1.814445687,
        2.557889765,
        6.308801854,
        2.557889765,
        6.308801854,
        1.814445687,
        1.814445687,
        2.557889765,
        1.814445687,
        1.814445687,
        0.907222843,
        0.907222843,
        0.907222843,
        0.907222843,
        0.907222843,
        1.068141502,
        1.068141502,
        0.907222843,
        0.907222843,
        0.907222843,
        0.907222843,
        0.907222843,
        1.068141502,
        1.068141502,
    )
    policy = FastSacPolicyProfile(
        model_artifact_id="g1-holosoma-fastsac",
        input_name="actor_obs",
        output_name="action",
        # Holosoma sorts configured terms before flattening them.
        observation_terms=(
            "actions",
            "base_ang_vel",
            "command_ang_vel",
            "command_lin_vel",
            "cos_phase",
            "dof_pos",
            "dof_vel",
            "projected_gravity",
            "sin_phase",
        ),
        observation_dimensions=(
            ("actions", 29),
            ("base_ang_vel", 3),
            ("command_ang_vel", 1),
            ("command_lin_vel", 2),
            ("cos_phase", 2),
            ("dof_pos", 29),
            ("dof_vel", 29),
            ("projected_gravity", 3),
            ("sin_phase", 2),
        ),
        observation_scales=(
            ("actions", 1.0),
            ("base_ang_vel", 0.25),
            ("command_ang_vel", 1.0),
            ("command_lin_vel", 1.0),
            ("cos_phase", 1.0),
            ("dof_pos", 1.0),
            ("dof_vel", 0.05),
            ("projected_gravity", 1.0),
            ("sin_phase", 1.0),
        ),
        action_scale=0.25,
        raw_action_clip=100.0,
        gait_period_s=1.0,
    )
    return G1ControllerProfile(
        profile_id="g1-holosoma-fastsac-v1",
        family=PolicyFamily.fast_sac,
        robot=_robot_profile(default_positions, stiffness, damping),
        loop=LoopProfile(
            control_hz=50.0,
            initialization_ticks=500,
            desired_base_height_m=0.75,
            repeated_state_stop_ticks=None,
            isolate_inference_errors=False,
            velocity_staleness_s=1.0,
            velocity_command_limit=1.0,
            torso_height_range_m=(0.75, 0.75),
            torso_orientation_limit_rad=0.0,
        ),
        policy=policy,
    )


def amo_profile() -> G1ControllerProfile:
    """Return the AMO-through-Holosoma G1 profile.

    This intentionally selects append-before-flatten history and integrated
    yaw-rate semantics.  Those choices reproduce the authors' adapter, not the
    slightly different upstream Psi0 observation helper.
    """

    default_positions = (
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        0.2,
        0.3,
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        -0.2,
        0.3,
        0.0,
        0.0,
        0.0,
    )
    stiffness = (
        150.0,
        150.0,
        150.0,
        300.0,
        80.0,
        20.0,
        150.0,
        150.0,
        150.0,
        300.0,
        80.0,
        20.0,
        400.0,
        400.0,
        400.0,
        50.0,
        50.0,
        50.0,
        50.0,
        30.0,
        30.0,
        30.0,
        50.0,
        50.0,
        50.0,
        50.0,
        30.0,
        30.0,
        30.0,
    )
    damping = (
        2.0,
        2.0,
        2.0,
        4.0,
        2.0,
        1.0,
        2.0,
        2.0,
        2.0,
        4.0,
        2.0,
        1.0,
        15.0,
        15.0,
        15.0,
        7.5,
        7.5,
        7.5,
        7.5,
        6.0,
        6.0,
        6.0,
        7.5,
        7.5,
        7.5,
        7.5,
        6.0,
        6.0,
        6.0,
    )
    jump_limits = (
        (math.pi / 2,) * 15
        + (math.pi / 3,) * 5
        + (math.pi,) * 2
        + (math.pi / 3,) * 5
        + (math.pi,) * 2
    )
    policy = AmoPolicyProfile(
        policy_artifact_id="g1-amo-policy",
        adapter_artifact_id="g1-amo-adapter",
        statistics_artifact_id="g1-amo-normalization",
        gravity_urdf_artifact_id="g1-dynamics-urdf",
        observed_joint_indices=tuple(range(19)) + tuple(range(22, 26)),
        zero_velocity_indices=(4, 5, 10, 11, 13, 14),
        lower_body_joint_count=15,
        proprio_dimension=93,
        demo_dimension=17,
        privileged_dimension=3,
        history_length=10,
        extra_history_length=25,
        action_scale=0.25,
        raw_action_clip=40.0,
        angular_velocity_scale=0.25,
        joint_velocity_scale=0.05,
        control_dt_s=0.02,
        gait_frequency_hz=1.3,
        in_place_velocity_threshold=0.1,
        torso_height_m=0.75,
        torch_threads=1,
        jump_limits_rad=jump_limits,
        history_order=HistoryOrder.append_current_before_flatten,
        heading_control=HeadingControl.integrated_rate,
        gravity_behavior=GravityBehavior.best_effort,
    )
    return G1ControllerProfile(
        profile_id="g1-amo-through-holosoma-v1",
        family=PolicyFamily.amo,
        # The COMMANDED idle/init arm pose is the FastSAC/Holosoma walk pose
        # (shoulders abducted 11.5 deg, elbows 34 deg): AMO's own arm default
        # carries zero shoulder roll, so the swinging arms brush the body. default_positions is
        # untouched — it normalises the policy observation (see RobotProfile).
        robot=_robot_profile(
            default_positions, stiffness, damping,
            idle_arm_positions=(
                0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
                0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
            ),
        ),
        loop=LoopProfile(
            control_hz=50.0,
            initialization_ticks=500,
            desired_base_height_m=0.75,
            repeated_state_stop_ticks=10,
            isolate_inference_errors=True,
            velocity_staleness_s=1.0,
            velocity_command_limit=1.0,
            torso_height_range_m=(0.75, 0.75),
            torso_orientation_limit_rad=0.0,
        ),
        policy=policy,
    )


_profile_factories: tuple[Callable[[], G1ControllerProfile], ...] = (
    fast_sac_profile,
    amo_profile,
)


def controller_profiles() -> tuple[G1ControllerProfile, ...]:
    """Construct the registered lower-body profiles in stable catalog order."""

    profiles = tuple(factory() for factory in _profile_factories)
    if len({profile.profile_id for profile in profiles}) != len(profiles):
        raise RuntimeError("G1 controller profile ids must be unique")
    return profiles


def controller_profile_ids() -> tuple[str, ...]:
    return tuple(profile.profile_id for profile in controller_profiles())


def default_controller_profile_id() -> str:
    return controller_profile_ids()[0]


def profile_by_id(profile_id: str) -> G1ControllerProfile:
    """Resolve one registered lower-body profile."""

    profiles = {profile.profile_id: profile for profile in controller_profiles()}
    try:
        return profiles[profile_id]
    except KeyError as exc:
        available = ", ".join(profiles)
        raise KeyError(
            f"unknown G1 controller profile {profile_id!r}; available: {available}"
        ) from exc

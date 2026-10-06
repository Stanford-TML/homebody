"""The simulation's settings. Each group is a frozen dataclass of defaults.
`configs/simulation.toml` overrides them by group and key, and a scene package's
`settings.toml` overlays that."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ENV = "src_kitchen"  # the environment a run uses unless --env names another
DEFAULT_SCENE = Path("assets/real2sim") / DEFAULT_ENV


@dataclass(frozen=True)
class Physics:
    """MuJoCo stepping and the simulated hand (`hand_kp` in N m/rad)."""
    timestep: float = 0.001
    control_hz: int = 250
    policy_hz: int = 50
    solver: str = "Newton"
    integrator: str = "Euler"
    # pyramidal cone at impratio 1 lets resting objects creep
    cone: str = "elliptic"
    impratio: float = 10.0
    noslip_iterations: int = 2
    hand_kp: float = 2.0
    hand_kd: float = 0.10
    hand_torque_fraction: float = 0.80
    hand_friction: tuple[float, float, float] = (4.0, 0.20, 0.020)
    hand_contact: tuple[float, float] = (0.004, 1.0)
    arm_rate: float = 0.525
    arm_authority_seconds: float = 2.0
    repeated_state_stop_ticks: int = 25
    hand_rate: float = 2.0
    command_timeout: float = 0.20
    min_base_height: float = 0.45
    max_base_height: float = 1.05
    max_tilt_degrees: float = 45.0

@dataclass(frozen=True)
class Camera:
    width: int = 640
    height: int = 480
    horizontal_fov: float = 69.0
    near: float = 0.02
    far: float = 15.0
    observer_width: int = 960
    observer_height: int = 540

@dataclass(frozen=True)
class Recording:
    """Sampling periods of the recorded streams, in simulation seconds. Each video plays at its sampling rate."""
    frame_period: float = 0.5
    room_period: float = 1.0
    shoulder_period: float = 0.25
    physics_period: float = 0.1
    publish_period: float = 0.5

@dataclass(frozen=True)
class Servo:
    tick: float = 0.04
    perception_period: float = 0.20
    position_gain: float = 2.0
    rotation_gain: float = 2.0
    max_linear_speed: float = 0.08
    max_angular_speed: float = 0.4
    damping_squared: float = 0.002
    max_joint_lag: float = 0.35
    rotation_tolerance: float = 0.12
    stable_frames: int = 3

@dataclass(frozen=True)
class Collision:
    cloud_margin: float = 0.008
    voxel_size: float = 0.01
    min_hit_voxels: int = 1
    joint_step: float = 0.04

@dataclass(frozen=True)
class Topdown:
    ahead: tuple[float, float] = (0.0, 1.05)
    left: tuple[float, float] = (-0.70, 0.70)
    height: tuple[float, float] = (-0.90, 0.60)
    pixels_per_metre: float = 760.0
    grid_metres: float = 0.10
    label_metres: float = 0.20
    chest_size: tuple[float, float] = (0.22, 0.26)

# name -> (provider, model id, CLI effort). "-api" names use a key, bare names a CLI login
MODELS = {
    "astra": ("codex", "gpt-6-astra", "high"),
    "astra-api": ("openai-api", "gpt-6-astra", None),
    "sol": ("codex", "gpt-5.6-sol", "high"),
    "sol-api": ("openai-api", "gpt-5.6-sol", None),
    "luna": ("codex", "gpt-5.6-luna", "high"),
    "luna-api": ("openai-api", "gpt-5.6-luna", None),
    "terra": ("codex", "gpt-5.6-terra", "high"),
    "terra-api": ("openai-api", "gpt-5.6-terra", None),
    "opus": ("claude-code", "claude-opus-5-5", "high"),
    "opus-api": ("claude-api", "claude-opus-5-5", None),
    "fable": ("claude-code", "claude-fable-5-1", "high"),
    "fable-api": ("claude-api", "claude-fable-5-1", None),
    "sonnet": ("claude-code", "claude-sonnet-5-5", "high"),
    "sonnet-api": ("claude-api", "claude-sonnet-5-5", None),
    "haiku": ("claude-code", "claude-haiku-4-5-20251001", "high"),
    "haiku-api": ("claude-api", "claude-haiku-4-5-20251001", None),
}


@dataclass(frozen=True)
class Agent:
    """The model, by one name from MODELS, and its limits. The robot stands `done_settle_seconds` after `done` before the final physics sample."""
    model: str = "astra"
    # "model" takes the model's own effort, max means Codex's xhigh
    reasoning_effort: str = "model"
    timeout: float = 180.0
    # retries of a busy service, after busy_wait, twice that, three times that seconds
    busy_retries: int = 3
    busy_wait: float = 20.0
    max_steps: int = 80
    done_settle_seconds: float = 3.0

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"agent.model must be one of {', '.join(MODELS)}")
        if self.reasoning_effort not in ("model", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("agent.reasoning_effort must be model, low, medium, high, xhigh or max")

    @property
    def provider(self) -> str:
        return MODELS[self.model][0]

    @property
    def model_id(self) -> str:
        return MODELS[self.model][1]

    @property
    def effort(self) -> str:
        """The CLI effort level: the model's own unless reasoning_effort names a level."""
        return MODELS[self.model][2] if self.reasoning_effort == "model" else self.reasoning_effort


@dataclass(frozen=True)
class SharedMotion:
    arm_timeout: float = 15.0
    palm_tolerance: float = 0.015
    observation_age: float = 0.25


@dataclass(frozen=True)
class Navigate:
    """Drive speeds, stances and arrival. An arrival is verified within `arrival_distance`, or `arrival_distance + close_slack` once re-closed."""
    drive_speed: float = 0.35
    turn_speed: float = 0.45
    final_speed: float = 0.20
    loaded_drive_speed: float = 0.25
    loaded_turn_speed: float = 0.30
    min_close_speed: float = 0.15
    min_turn_speed: float = 0.15
    close_gain: float = 1.0
    heading_gain: float = 1.5
    turn_hold_gain: float = 1.0
    robot_radius: float = 0.28
    route_clearance_margin: float = 0.0
    stance_reach: float = 0.46
    stance_clearance: float = 0.15
    right_hand_offset: float = 0.14
    left_hand_offset: float = 0.14
    departure_distance: float = 0.30
    run_in_distance: float = 0.60
    aim_in_distance: float = 0.80
    goal_snap_max_distance: float = 2.0
    start_snap_distance: float = 0.15
    goal_snap_slack: float = 0.10
    facing_min_distance: float = 0.10
    waypoint_radius: float = 0.23
    route_lookahead: float = 0.50
    route_lateral_gain: float = 1.0
    turn_before_translate: float = 0.35
    stop_distance: float = 0.03
    arrival_distance: float = 0.10
    arrival_angle: float = 0.10
    settle_seconds: float = 1.0
    close_rounds: int = 3
    close_slack: float = 0.20
    close_min_gain: float = 0.02
    drive_timeout: float = 90.0
    stall_timeout: float = 6.0
    progress_distance: float = 0.03
    tick: float = 0.10


@dataclass(frozen=True)
class Grasp:
    """Grasp planning, approach and retention limits: the walk-in stands with the pre-grasp palm `grasp_range` from the shoulder, and a pick closes at least `squeeze` past where the fingers stopped."""
    shape_quantiles: tuple[float, float] = (0.02, 0.98)
    depth_max: float = 0.055
    depth_fraction: float = 0.5
    insertion_depth_scales: tuple[float, ...] = (1.0, 0.65, 0.3)
    approach_height: float = 0.10
    lift_height: float = 0.18
    lift_evidence: float = 0.05
    carry_palm_radius: float = 0.30
    carry_compact_min_delta: float = 0.03
    carry_compact_step: float = 0.01
    yaw_samples: int = 12
    finger_spread: float = 0.052
    pad_band: float = 0.015
    pad_band_points: int = 12
    approach_attempts: int = 2
    aperture_error: float = 0.008
    open_seconds: float = 1.0
    close_seconds: float = 1.5
    squeeze: float = 0.12
    target_shift: float = 0.08
    target_surface_tolerance: float = 0.015
    occlusion_min_rays: float = 0.4
    occlusion_missing_explained: float = 0.5
    occlusion_exposed_missing_max: float = 0.1
    occlusion_visible_match: float = 0.5
    occlusion_no_view_rays: float = 0.8
    hold_effort_fraction: float = 0.04
    hold_closure_gap: float = 0.04
    recovery_samples: int = 5
    recovery_confirmations: int = 3
    recovery_tick: float = 0.10
    recovery_separation: float = 0.08
    recovery_match: float = 0.05
    spline_out: tuple[float, ...] = (0.0, 0.10, 0.15)
    spline_over: tuple[float, ...] = (0.06, 0.10, 0.14)
    spline_descend: float = 0.08
    spline_samples: int = 4001
    spline_waypoints: int = 32
    spline_smooth_passes: int = 4
    spline_seed_bias: float = 0.75
    approach_start_lifts_m: tuple[float, ...] = (0.10, 0.20, 0.30)
    grasp_range: float = 0.36
    approach_short: float = 0.08
    approach_long: float = 0.03
    approach_across: float = 0.04
    approach_side: float = 0.24

    def __post_init__(self):
        if (min(self.grasp_range, self.approach_short, self.approach_long) <= 0
                or self.approach_across >= self.approach_side
                or min(self.approach_start_lifts_m, default=0) < 0):
            raise ValueError("Grasp range and approach band must be positive and ordered")


@dataclass(frozen=True)
class BlindFinish:
    """Limits on finishing a grasp whose target is partly visible or hidden by the arm: only the planned descent may continue, with the base and head still and the palm's travel budgeted."""
    partial_travel: float = 0.20
    occluded_reach: float = 0.30
    occluded_recency: float = 0.08
    travel_margin: float = 0.08
    palm_turn: float = 0.24
    base_turn: float = 0.10
    head_drift: float = 0.08
    head_turn: float = 0.24

    def __post_init__(self):
        if min(vars(self).values()) <= 0:
            raise ValueError("Blind finish limits must be positive")


@dataclass(frozen=True)
class Placement:
    release_height: float = 0.015
    withdraw_height: float = 0.03
    withdraw_back: float = 0.05
    withdraw_clearance: float = 0.03
    rest_tolerance: float = 0.35
    pull_step: float = 0.02
    max_pull: float = 0.20
    reach_margin_steps: int = 2
    release_drops: tuple[float, ...] = (0., 0.05, 0.10)
    release_twists: tuple[float, ...] = (0., 0.3, -0.3, 0.6, -0.6)
    # the walking policy's idle arm pose
    rest_arm_left: tuple[float, float, float, float, float, float, float] = (0.2, 0.2, 0., 0.6, 0., 0., 0.)
    rest_arm_right: tuple[float, float, float, float, float, float, float] = (0.2, -0.2, 0., 0.6, 0., 0., 0.)
    max_release_height: float = 0.30
    forward_reach: float = 0.42
    support_match: float = 0.04
    open_closure_limit: float = 0.15
    settle_count: int = 15
    tick: float = 0.10
    xy_tolerance: float = 0.12
    support_height_tolerance: float = 0.06
    still_distance: float = 0.01
    stable_frames: int = 3

    def rest_arm(self, side):
        return np.asarray(self.rest_arm_left if side == "left" else self.rest_arm_right, dtype=float)

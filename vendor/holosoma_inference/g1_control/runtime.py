# Vendored G1 controller kernel, derived from Holosoma and adapted by the authors; see THIRD_PARTY_NOTICES.md.
"""Holosoma-compatible lifecycle around a single package-owned G1 writer.

The transition order and 500-tick initialization ramp are derived from Amazon
FAR Holosoma revision ``4ed2cebf9780f3efb59621657916006afadb87dc``.
Hardware transport and model inference remain injected seams so failures stay
inside the component that owns them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import G1ControllerProfile
from .contracts import (
    ControllerCommand,
    G1State,
    LowCommand,
    LowerBodyPolicy,
    RobotPort,
    TorsoCommand,
    VelocityCommand,
)


class ControllerKillRequested(RuntimeError):
    """The operator requested process-level shutdown."""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class ControllerStatus:
    """Dependency-free snapshot of controller lifecycle state."""

    active: bool
    initializing: bool
    walk_enabled: bool
    initialization_tick: int
    frozen_state_ticks: int
    stopped_by_watchdog: bool
    velocity_stale: bool
    command_ticks: int
    last_error: str | None
    gravity_degraded: bool
    gravity_error: str | None


@dataclass(frozen=True)
class TickResult:
    """State and command used by one completed controller tick."""

    state: G1State
    command: LowCommand
    status: ControllerStatus


class G1ControllerKernel:
    """Single-tick G1 control kernel with no hidden process or input ownership."""

    def __init__(
        self,
        profile: G1ControllerProfile,
        port: RobotPort,
        policy: LowerBodyPolicy,
    ) -> None:
        if policy.profile != profile:
            raise ValueError("policy and controller profiles must match exactly")
        self.profile = profile
        self.port = port
        self.policy = policy
        self._active = False
        self._initializing = False
        self._walk_enabled = False
        self._initialization_tick = 0
        self._frozen_state_ticks = 0
        self._stopped_by_watchdog = False
        self._command_ticks = 0
        self._last_error: str | None = None
        self._last_state_vector: tuple[float, ...] | None = None
        self._velocity = VelocityCommand()
        self._velocity_age_ticks = 0
        self._velocity_stale = False
        self._torso = self._clamped_torso(TorsoCommand(height_m=profile.loop.desired_base_height_m))
        self._arm_targets: tuple[float, ...] | None = None

    @property
    def status(self) -> ControllerStatus:
        gravity_degraded = bool(getattr(self.policy, "gravity_degraded", False))
        gravity_error = getattr(self.policy, "gravity_error", None)
        return ControllerStatus(
            active=self._active,
            initializing=self._initializing,
            walk_enabled=self._walk_enabled,
            initialization_tick=self._initialization_tick,
            frozen_state_ticks=self._frozen_state_ticks,
            stopped_by_watchdog=self._stopped_by_watchdog,
            velocity_stale=self._velocity_stale,
            command_ticks=self._command_ticks,
            last_error=self._last_error,
            gravity_degraded=gravity_degraded,
            gravity_error=gravity_error,
        )

    @property
    def velocity(self) -> VelocityCommand:
        return self._velocity

    def set_velocity(self, command: VelocityCommand) -> None:
        """Apply the adapter's machine-seam clamp and Holosoma's stand gate."""

        limit = self.profile.loop.velocity_command_limit
        if limit is not None:
            command = VelocityCommand(
                linear_xy=(
                    _clamp(command.linear_xy[0], -limit, limit),
                    _clamp(command.linear_xy[1], -limit, limit),
                ),
                angular_z=_clamp(command.angular_z, -limit, limit),
            )
        self._velocity = command if self._walk_enabled else VelocityCommand()
        self._velocity_age_ticks = 0
        self._velocity_stale = False

    def _clamped_torso(self, command: TorsoCommand) -> TorsoCommand:
        """Reduce the torso seam to the adapter's envelope (height-only, rpy 0)."""

        low, high = self.profile.loop.torso_height_range_m
        orientation_limit = self.profile.loop.torso_orientation_limit_rad
        return TorsoCommand(
            height_m=_clamp(command.height_m, low, high),
            roll_rad=_clamp(command.roll_rad, -orientation_limit, orientation_limit),
            pitch_rad=_clamp(command.pitch_rad, -orientation_limit, orientation_limit),
            yaw_rad=_clamp(command.yaw_rad, -orientation_limit, orientation_limit),
        )

    def set_torso(self, command: TorsoCommand) -> None:
        self._torso = self._clamped_torso(command)

    def set_arm_targets(self, targets: tuple[float, ...] | None) -> None:
        if targets is None:
            self._arm_targets = None
            return
        values = tuple(float(value) for value in targets)
        if len(values) != 14 or not all(math.isfinite(value) for value in values):
            raise ValueError("arm target must contain 14 finite values")
        self._arm_targets = values

    def handle_command(self, command: ControllerCommand) -> None:
        """Apply one discrete Holosoma-compatible lifecycle command."""

        if command is ControllerCommand.start:
            self._active = True
            self._initializing = False
            self._stopped_by_watchdog = False
            self.policy.on_start()
        elif command is ControllerCommand.stop:
            self._stop()
        elif command is ControllerCommand.initialize:
            # Holosoma retains the active flag while the higher-priority init
            # branch owns q_target; START later exits the ramp.
            self._initializing = True
            self._initialization_tick = 0
        elif command is ControllerCommand.toggle_walk:
            self._walk_enabled = not self._walk_enabled
            if not self._walk_enabled:
                self._velocity = VelocityCommand()
        elif command is ControllerCommand.zero_velocity:
            self._velocity = VelocityCommand()
        elif command is ControllerCommand.stiffness_up:
            self.port.stiffness_level += 0.1
        elif command is ControllerCommand.stiffness_down:
            self.port.stiffness_level -= 0.1
        elif command is ControllerCommand.stiffness_up_fine:
            self.port.stiffness_level += 0.01
        elif command is ControllerCommand.stiffness_down_fine:
            self.port.stiffness_level -= 0.01
        elif command is ControllerCommand.stiffness_reset:
            self.port.stiffness_level = 1.0
        elif command is ControllerCommand.kill:
            raise ControllerKillRequested("operator requested G1 controller shutdown")
        else:
            raise ValueError(f"unsupported controller command: {command!r}")

    def _stop(self) -> None:
        self._active = False
        self._initializing = False
        self.policy.on_stop()

    def _initialization_target(self, state: G1State) -> tuple[float, ...]:
        fraction = self._initialization_tick / self.profile.loop.initialization_ticks
        # idle_positions, not default_positions: the ramp aims at the pose the
        # robot should STAND in (arms included), which a profile may pin
        # separately from the observation-normalising defaults.
        target = tuple(
            measured + (default - measured) * fraction
            for measured, default in zip(
                state.joint_positions,
                self.profile.robot.idle_positions,
                strict=True,
            )
        )
        self._initialization_tick = min(
            self._initialization_tick + 1,
            self.profile.loop.initialization_ticks,
        )
        return target

    def _command_for_state(self, state: G1State) -> LowCommand:
        torques = (0.0,) * 29
        if self._initializing:
            targets = self._initialization_target(state)
        elif not self._active:
            targets = state.joint_positions
        else:
            try:
                result = self.policy.infer(
                    state,
                    self._velocity,
                    self._torso,
                    self._arm_targets,
                )
                targets = result.joint_targets
                torques = result.feedforward_torques
                self._last_error = None
            except Exception as exc:
                if not self.profile.loop.isolate_inference_errors:
                    raise
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._stop()
                targets = state.joint_positions
                torques = (0.0,) * 29

        commanded_positions = tuple(
            target + offset
            for target, offset in zip(
                targets,
                self.profile.robot.joint_offsets_rad,
                strict=True,
            )
        )
        return LowCommand(
            positions=commanded_positions,
            velocities=(0.0,) * 29,
            feedforward_torques=torques,
            stiffness=self.profile.robot.stiffness,
            damping=self.profile.robot.damping,
        )

    def _observe_link_freshness(self, state: G1State) -> None:
        limit = self.profile.loop.repeated_state_stop_ticks
        if limit is None:
            return
        current = state.holosoma_vector()
        if self._last_state_vector is not None and current == self._last_state_vector:
            self._frozen_state_ticks += 1
            if self._frozen_state_ticks >= limit:
                if self._active:
                    self._stop()
                    self._stopped_by_watchdog = True
                self._frozen_state_ticks = 0
        else:
            self._frozen_state_ticks = 0
        self._last_state_vector = current

    def _expire_stale_velocity(self) -> None:
        """Dead-man seam: zero velocity 1.0 s after the last command.

        One-shot per command like the adapter's timer (its ``_last_vel_time > 0``
        guard), so an idle kernel does not re-fire after an explicit zero.
        """

        limit = self.profile.loop.velocity_staleness_ticks
        if limit is None or self._velocity_stale:
            return
        if self._velocity_age_ticks >= limit:
            self._velocity = VelocityCommand()
            self._velocity_stale = True
        else:
            self._velocity_age_ticks += 1

    def tick(self) -> TickResult:
        """Read, infer or hold, publish, then update the link watchdog."""

        # The adapter polls (and expires) the velocity command before use each cycle.
        self._expire_stale_velocity()
        # Holosoma advances FastSAC phase before reading lowstate every cycle,
        # including hold and initialization cycles.  AMO's hook is a no-op.
        self.policy.before_tick(self._velocity)
        state = self.port.read_state()
        command = self._command_for_state(state)
        self.port.write_command(command)
        self._command_ticks += 1
        # The AMO watchdog stops after the current command is sent.
        self._observe_link_freshness(state)
        return TickResult(state=state, command=command, status=self.status)


__all__ = [
    "ControllerKillRequested",
    "ControllerStatus",
    "G1ControllerKernel",
    "TickResult",
]

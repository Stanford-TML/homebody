# Vendored G1 controller kernel, derived from Holosoma and adapted by the authors; see THIRD_PARTY_NOTICES.md.
"""AMO inference semantics behind the package-owned G1 contract.

The networks and core observation math originate from OpenTeleVision/AMO and
Psi0 under Apache-2.0.  The lifecycle adaptation follows the AMO/Holosoma
adapter of the authors; see the repository's
``THIRD_PARTY_NOTICES.md``.  In particular,
this implementation preserves that adapter's append-before-flatten history and
integrated yaw-rate behavior; it does not silently substitute upstream Psi0's
different history ordering.

NumPy and Torch are deferred until policy construction.  The factory imports
them in that order before constructing Pinocchio gravity dynamics, a C++ runtime
load order that is known to be compatible.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import zipfile
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from .config import (
    AmoPolicyProfile,
    G1ControllerProfile,
    GravityBehavior,
    HistoryOrder,
    PolicyFamily,
)
from .contracts import (
    G1State,
    GravityCompensator,
    PolicyResult,
    TorsoCommand,
    VelocityCommand,
)


def _prepare_amo_runtime():
    """Load AMO's numerical runtimes in the reviewed NumPy-then-Torch order."""

    import numpy as np
    import torch

    return np, torch


def _private_cache_owner(cache_directory: Path) -> int:
    cache_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory_status = cache_directory.lstat()
    owner = getattr(os, "getuid", lambda: directory_status.st_uid)()
    permissions = stat.S_IMODE(directory_status.st_mode)
    if (
        stat.S_ISLNK(directory_status.st_mode)
        or not stat.S_ISDIR(directory_status.st_mode)
        or directory_status.st_uid != owner
        or permissions & 0o077
        or (permissions & 0o700) != 0o700
    ):
        raise RuntimeError("AMO checkpoint cache must be an owner-only directory")
    return owner


def _assert_private_stream(stream: BinaryIO, owner: int) -> None:
    status = os.fstat(stream.fileno())
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_uid != owner
        or status.st_nlink not in (0, 1)
        or stat.S_IMODE(status.st_mode) & 0o077
    ):
        raise RuntimeError("AMO checkpoint staging file is not owner-private")


def _copy_verified_source(source: Path, destination: BinaryIO, source_sha256: str) -> None:
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ValueError("AMO artifact SHA-256 must be canonical lowercase hexadecimal")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    with os.fdopen(descriptor, "rb") as source_stream:
        source_status = os.fstat(source_stream.fileno())
        if not stat.S_ISREG(source_status.st_mode):
            raise RuntimeError("AMO source artifact must be a regular file")
        digest = hashlib.sha256()
        for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
            destination.write(chunk)
            digest.update(chunk)

    if digest.hexdigest() != source_sha256:
        raise RuntimeError("AMO source artifact no longer matches its reviewed SHA-256")
    destination.flush()
    os.fsync(destination.fileno())
    destination.seek(0)


def _rewrite_cuda_device(source: BinaryIO, destination: BinaryIO) -> int:
    cuda_literal = b'torch.device("cuda:0")'
    cpu_literal = b'torch.device("cpu")'
    rewrites = 0
    source.seek(0)
    with zipfile.ZipFile(source, "r") as input_archive:
        with zipfile.ZipFile(destination, "w") as output_archive:
            for entry in input_archive.infolist():
                data = input_archive.read(entry.filename)
                if entry.filename.endswith(".py") and cuda_literal in data:
                    data = data.replace(cuda_literal, cpu_literal)
                    rewrites += 1
                output_archive.writestr(entry, data)

    destination.flush()
    os.fsync(destination.fileno())
    destination.seek(0)
    with zipfile.ZipFile(destination, "r") as completed_archive:
        for entry in completed_archive.infolist():
            if (
                entry.filename.endswith(".py")
                and cuda_literal in completed_archive.read(entry.filename)
            ):
                raise RuntimeError("derived AMO checkpoint still contains a CUDA device literal")
    destination.seek(0)
    return rewrites


@contextmanager
def verified_checkpoint_stream(
    source: Path,
    cache_directory: Path,
    source_sha256: str,
    *,
    rewrite_cuda: bool = False,
) -> Iterator[BinaryIO]:
    """Yield one owner-private stream containing exactly the reviewed bytes.

    Verification hashes the same private file descriptor later consumed by
    Torch.  The package artifact path is never reopened for model loading, so
    replacing it after verification cannot change the executable bytes.
    """

    owner = _private_cache_owner(cache_directory)
    with tempfile.TemporaryFile(mode="w+b", dir=cache_directory) as verified:
        _assert_private_stream(verified, owner)
        _copy_verified_source(Path(source), verified, source_sha256)
        if not rewrite_cuda:
            yield verified
            return

        with tempfile.TemporaryFile(mode="w+b", dir=cache_directory) as derived:
            _assert_private_stream(derived, owner)
            rewrites = _rewrite_cuda_device(verified, derived)
            if rewrites:
                yield derived
            else:
                verified.seek(0)
                yield verified


class AmoPolicy:
    """Stateful 15-joint AMO network with full-width G1 targets."""

    def __init__(
        self,
        profile: G1ControllerProfile,
        policy_path: str | Path,
        adapter_path: str | Path,
        statistics_path: str | Path,
        *,
        cache_directory: str | Path,
        policy_sha256: str,
        adapter_sha256: str,
        statistics_sha256: str,
        gravity: GravityCompensator | None = None,
        gravity_initialization_error: str | None = None,
    ) -> None:
        if profile.family is not PolicyFamily.amo or not isinstance(profile.policy, AmoPolicyProfile):
            raise ValueError("AmoPolicy requires an AMO controller profile")

        np, torch = _prepare_amo_runtime()

        self._profile = profile
        self._policy_profile = profile.policy
        self._np = np
        self._torch = torch
        self._gravity = gravity
        gravity_expected = self._policy_profile.gravity_behavior is not GravityBehavior.disabled
        if gravity_expected and gravity is None and gravity_initialization_error is None:
            gravity_initialization_error = "gravity compensator is unavailable"
        self._gravity_degraded = gravity_expected and gravity_initialization_error is not None
        self._gravity_error = gravity_initialization_error if self._gravity_degraded else None

        if self._policy_profile.torch_threads:
            torch.set_num_threads(self._policy_profile.torch_threads)
            try:
                torch.set_num_interop_threads(self._policy_profile.torch_threads)
            except RuntimeError:
                # Torch permits setting the inter-op pool only before it starts.
                pass

        device = "cuda" if torch.cuda.is_available() else "cpu"
        staging_directory = Path(cache_directory)
        with verified_checkpoint_stream(
            Path(policy_path),
            staging_directory,
            policy_sha256,
            rewrite_cuda=device == "cpu",
        ) as policy_stream:
            self._network = torch.jit.load(policy_stream, map_location=device)
        self._network.eval()
        with verified_checkpoint_stream(
            Path(adapter_path),
            staging_directory,
            adapter_sha256,
        ) as adapter_stream:
            self._adapter = torch.jit.load(adapter_stream, map_location=device)
        self._adapter.eval()
        with verified_checkpoint_stream(
            Path(statistics_path),
            staging_directory,
            statistics_sha256,
        ) as statistics_stream:
            statistics = torch.load(
                statistics_stream,
                weights_only=False,
                map_location=device,
            )
        self._device = device
        self._input_mean = torch.tensor(statistics["input_mean"], device=device, dtype=torch.float32)
        self._input_std = torch.tensor(statistics["input_std"], device=device, dtype=torch.float32)
        self._output_mean = torch.tensor(statistics["output_mean"], device=device, dtype=torch.float32)
        self._output_std = torch.tensor(statistics["output_std"], device=device, dtype=torch.float32)

        config = self._policy_profile
        self._default = np.asarray(profile.robot.default_positions, dtype=np.float64)
        self._observed = np.asarray(config.observed_joint_indices, dtype=np.int64)
        self._last_action = np.zeros(29, dtype=np.float64)
        self._gait_cycle = np.array([0.25, 0.25], dtype=np.float64)
        self._target_yaw = 0.0
        self._yaw_offset = 0.0
        self._in_place = True
        self._yaw_latched = False
        self._arm_latch = None
        self._previous_target = None

        self._demo_template = np.zeros(config.demo_dimension, dtype=np.float64)
        arm_demo_indices = np.r_[15:19, 22:26]
        self._demo_template[:8] = self._default[arm_demo_indices]
        self._demo_template[14:17] = config.torso_height_m
        self._proprio_history = deque(
            [np.zeros(config.proprio_dimension) for _ in range(config.history_length)],
            maxlen=config.history_length,
        )
        self._extra_history = deque(
            [np.zeros(config.proprio_dimension) for _ in range(config.extra_history_length)],
            maxlen=config.extra_history_length,
        )

    @property
    def profile(self) -> G1ControllerProfile:
        return self._profile

    @property
    def gravity_degraded(self) -> bool:
        return self._gravity_degraded

    @property
    def gravity_error(self) -> str | None:
        return self._gravity_error

    def before_tick(self, velocity: VelocityCommand) -> None:
        del velocity

    def on_start(self) -> None:
        # The adapter resets transition latches but retains network
        # history, gait state, and last_action across stop/start cycles.
        self._yaw_latched = False
        self._arm_latch = None
        self._previous_target = None

    def on_stop(self) -> None:
        self._previous_target = None

    def _quaternion_to_euler(self, quaternion):
        np = self._np
        scalar, x, y, z = quaternion
        roll = np.arctan2(2.0 * (scalar * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sine_pitch = 2.0 * (scalar * y - z * x)
        pitch = np.copysign(np.pi / 2.0, sine_pitch) if abs(sine_pitch) >= 1.0 else np.arcsin(sine_pitch)
        yaw = np.arctan2(2.0 * (scalar * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.array([roll, pitch, yaw])

    def _observe(
        self,
        state: G1State,
        velocity: VelocityCommand,
        torso: TorsoCommand,
    ):
        np = self._np
        torch = self._torch
        config = self._policy_profile
        positions = np.asarray(state.joint_positions, dtype=np.float64)
        velocities = np.asarray(state.joint_velocities, dtype=np.float64)
        angular_velocity = np.asarray(state.base_angular_velocity, dtype=np.float64)
        euler = self._quaternion_to_euler(state.quaternion_wxyz)

        self._target_yaw += velocity.angular_z * config.control_dt_s
        heading_error = euler[2] - self._yaw_offset - self._target_yaw
        heading_error = np.remainder(heading_error + np.pi, 2.0 * np.pi) - np.pi
        if self._in_place:
            heading_error = 0.0
            # Resync, not just mask. The integrated target drifts from the
            # true heading through every HOLD zero-burst, clamp, and slip
            # while a drive steers; an open-loop WALK cannot hide that drift.
            # A stale target would unmask a large heading error the instant
            # stepping began, and the policy would pursue the old heading
            # instead of walking straight. Gluing the target to the measured
            # heading while standing makes every motion start from zero
            # error; in place the masked error is 0.0 either way.
            self._target_yaw = float(euler[2]) - self._yaw_offset

        observed_positions = positions[self._observed]
        observed_velocities = velocities[self._observed].copy()
        observed_velocities[list(config.zero_velocity_indices)] = 0.0
        observed_default = self._default[self._observed]
        observed_last_action = self._last_action[self._observed]
        gait = np.sin(self._gait_cycle * 2.0 * np.pi)

        adapter_input = np.concatenate([np.zeros(4), observed_positions[15:]])
        adapter_input[:4] = (
            torso.height_m,
            torso.yaw_rad,
            torso.pitch_rad,
            torso.roll_rad,
        )
        with torch.no_grad():
            adapter_tensor = torch.tensor(
                adapter_input,
                device=self._device,
                dtype=torch.float32,
            ).unsqueeze(0)
            normalized = (adapter_tensor - self._input_mean) / (self._input_std + 1e-8)
            adapted = self._adapter(normalized.view(1, -1))
            adapted = adapted * self._output_std + self._output_mean

        proprio = np.concatenate(
            [
                angular_velocity * config.angular_velocity_scale,
                euler[:2],
                (np.sin(heading_error), np.cos(heading_error)),
                observed_positions - observed_default,
                observed_velocities * config.joint_velocity_scale,
                observed_last_action,
                gait,
                adapted.cpu().numpy().squeeze(),
            ]
        )
        if proprio.shape != (config.proprio_dimension,):
            raise RuntimeError(f"AMO proprioception has shape {proprio.shape}")

        demo = self._demo_template.copy()
        demo[:8] = observed_positions[15:]
        demo[8] = velocity.linear_xy[0]
        demo[9] = velocity.linear_xy[1]
        threshold = config.in_place_velocity_threshold
        self._in_place = (
            abs(velocity.linear_xy[0]) < threshold
            and abs(velocity.linear_xy[1]) < threshold
            and abs(velocity.angular_z) < threshold
        )
        demo[11:14] = (torso.yaw_rad, torso.pitch_rad, torso.roll_rad)
        demo[14:17] = torso.height_m

        if config.history_order is HistoryOrder.append_current_before_flatten:
            self._proprio_history.append(proprio)
            history = np.asarray(self._proprio_history).flatten()
        else:
            history = np.asarray(self._proprio_history).flatten()
            self._proprio_history.append(proprio)
        self._extra_history.append(proprio)
        observation = np.concatenate(
            [proprio, demo, np.zeros(config.privileged_dimension), history]
        )
        if observation.shape != (config.observation_dimension,):
            raise RuntimeError(f"AMO observation has shape {observation.shape}")

        self._gait_cycle = np.remainder(
            self._gait_cycle + config.control_dt_s * config.gait_frequency_hz,
            1.0,
        )
        close_to_stance = np.abs(self._gait_cycle - 0.25) < 0.05
        if self._in_place and np.any(close_to_stance):
            self._gait_cycle[:] = (0.25, 0.25)
        if not self._in_place and np.all(close_to_stance):
            self._gait_cycle[:] = (0.25, 0.75)
        return observation

    def _arm_target(
        self,
        arm_targets: tuple[float, ...] | None,
    ):
        np = self._np
        if arm_targets is not None:
            target = np.asarray(arm_targets, dtype=np.float64)
            if target.shape != (14,) or not np.all(np.isfinite(target)):
                raise ValueError("AMO arm target must contain 14 finite values")
            self._arm_latch = target.copy()
        if self._arm_latch is None:
            # The commanded idle pose, NOT self._default[15:]: the profile may
            # pin a walk-safe arm pose (shoulders abducted, elbows bent)
            # separately from the observation-normalising defaults.
            self._arm_latch = np.asarray(
                self._profile.robot.idle_positions[15:], dtype=np.float64
            )
        return self._arm_latch

    def _act(self, observation, measured_positions, arm_target):
        np = self._np
        torch = self._torch
        config = self._policy_profile
        with torch.no_grad():
            observation_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self._device)
            extra_tensor = torch.tensor(
                np.asarray(self._extra_history).flatten().copy(),
                dtype=torch.float,
            ).view(1, -1).to(self._device)
            raw = self._network(observation_tensor, extra_tensor).cpu().numpy().squeeze()
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        if raw.shape != (config.lower_body_joint_count,):
            raise RuntimeError(f"AMO action has shape {raw.shape}, expected (15,)")
        raw = np.clip(raw, -config.raw_action_clip, config.raw_action_clip)
        targets = self._default.copy()
        targets[: config.lower_body_joint_count] += raw * config.action_scale
        targets[config.lower_body_joint_count :] = arm_target
        self._last_action = np.concatenate(
            [
                raw.copy(),
                (measured_positions[config.lower_body_joint_count :] - self._default[config.lower_body_joint_count :])
                / config.action_scale,
            ]
        )
        return targets, raw

    def _feedforward(self, targets) -> tuple[float, ...]:
        behavior = self._policy_profile.gravity_behavior
        if behavior is GravityBehavior.disabled or self._gravity is None:
            if behavior is GravityBehavior.required and self._gravity is None:
                raise RuntimeError("AMO profile requires gravity compensation")
            return (0.0,) * 29
        try:
            torques = self._gravity.torques(tuple(targets.tolist()))
            return torques
        except Exception as exc:
            if not self._gravity_degraded:
                self._gravity_error = f"{type(exc).__name__}: {exc}"
            self._gravity_degraded = True
            if behavior is GravityBehavior.required:
                raise
            return (0.0,) * 29

    def infer(
        self,
        state: G1State,
        velocity: VelocityCommand,
        torso: TorsoCommand,
        arm_targets: tuple[float, ...] | None,
    ) -> PolicyResult:
        np = self._np
        positions = np.asarray(state.joint_positions, dtype=np.float64)
        if not self._yaw_latched:
            self._yaw_offset = float(self._quaternion_to_euler(state.quaternion_wxyz)[2])
            self._target_yaw = 0.0
            self._yaw_latched = True

        observation = self._observe(state, velocity, torso)
        arm_target = self._arm_target(arm_targets)
        targets, raw = self._act(observation, positions, arm_target)

        if self._previous_target is not None:
            jump = np.abs(targets - self._previous_target)
            limits = np.asarray(self._policy_profile.jump_limits_rad, dtype=np.float64)
            if np.any(jump > limits):
                targets = self._previous_target.copy()
        self._previous_target = targets.copy()
        return PolicyResult(
            joint_targets=tuple(targets.tolist()),
            feedforward_torques=self._feedforward(targets),
            raw_action=tuple(raw.tolist()),
        )

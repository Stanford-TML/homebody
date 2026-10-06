"""The G1 controller: AMO balance at 50 Hz, arm gates and Dex3 closure at 250 Hz."""
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from holosoma_inference.g1_control.amo import AmoPolicy
from holosoma_inference.g1_control.config import GravityBehavior, amo_profile
from holosoma_inference.g1_control.contracts import (
    ControllerCommand,
    G1State,
    VelocityCommand,
)
from holosoma_inference.g1_control.runtime import G1ControllerKernel

from homebody.primitives.observations import ARMS

SIDES = {"left": slice(0, 7), "right": slice(7, 14)}


class MotorPort:
    def __init__(self, model, data, names):
        self.data = data
        joints = [model.joint(name).id for name in names]
        self.q = model.jnt_qposadr[joints]
        self.v = model.jnt_dofadr[joints]
        self.command = None
        self.stiffness_level = 1.0
        self.damping_level = 1.0

    def read_state(self):
        d = self.data
        return G1State(base_position=tuple(d.qpos[:3]), quaternion_wxyz=tuple(d.qpos[3:7]),
                       joint_positions=tuple(d.qpos[self.q]),
                       base_linear_velocity=tuple(d.qvel[:3]),
                       base_angular_velocity=tuple(d.qvel[3:6]),
                       joint_velocities=tuple(d.qvel[self.v]))

    def write_command(self, command):
        self.command = command

    def torques(self):
        c = self.command
        if c is None:
            return np.zeros(29)
        return (np.asarray(c.feedforward_torques)
                + self.stiffness_level * np.asarray(c.stiffness)
                * (np.asarray(c.positions) - self.data.qpos[self.q])
                + self.damping_level * np.asarray(c.damping)
                * (np.asarray(c.velocities) - self.data.qvel[self.v]))


class ArmLimit:
    """One arm's gate: authority ramp, then rate, joint-range and effort clamps in that order."""
    def __init__(self, calibration, side, rate, authority_seconds, initial):
        self.q = np.array(initial, dtype=float)
        self.rate, self.authority_seconds = rate, authority_seconds
        self.lower, self.upper = np.array(calibration["arm_limits"][side])
        self.kp, self.kd = np.array(calibration["gains"]["track"])
        self.effort = np.array(calibration["arm_effort"])
        self.margin = calibration["limit_margin"]
        self.weight = 0.0

    def step(self, target, measured, dt):
        self.weight = min(1.0, self.weight + dt / self.authority_seconds)
        self.q = np.clip(target, self.q - self.rate * dt, self.q + self.rate * dt)
        self.q = np.clip(self.q, self.lower + self.margin, self.upper - self.margin)
        kp = self.kp * self.weight
        lead = self.effort / (self.kp * max(self.weight, 0.001))
        return np.clip(self.q, measured - lead, measured + lead), kp, self.kd


def configure_finger_servo(model, actuator, joint, settings):
    """Set one finger's position servo and its force cap, which is also the effort ceiling."""
    model.actuator_gainprm[actuator, 0] = settings.hand_kp
    model.actuator_biasprm[actuator, 1:3] = -settings.hand_kp, -settings.hand_kd
    limit = settings.hand_torque_fraction * np.max(np.abs(model.jnt_actfrcrange[joint]))
    model.actuator_forcerange[actuator] = -limit, limit
    model.jnt_actfrcrange[joint] = -limit, limit


def configure_hand_contacts(model, settings):
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if "_hand_" in name and name.endswith("_col"):
            model.geom_friction[gid] = settings.hand_friction
            model.geom_condim[gid] = 6
            model.geom_priority[gid] = 2
            model.geom_solref[gid] = settings.hand_contact


class Controller:
    def __init__(self, model, data, root: Path, settings, cache: Path):
        self.model, self.data, self.settings = model, data, settings
        calibration = json.loads((root / "assets/robot/calibration.json").read_text())
        self.calibration = calibration
        base = amo_profile()
        self.profile = replace(base, policy=replace(base.policy,
            gravity_behavior=GravityBehavior.disabled), loop=replace(base.loop,
            repeated_state_stop_ticks=settings.repeated_state_stop_ticks, isolate_inference_errors=False))
        if not np.isclose(1. / settings.policy_hz, self.profile.policy.control_dt_s):
            raise ValueError("The supplied AMO weights require their trained policy frequency")
        self.port = MotorPort(model, data, self.profile.robot.joint_names)
        manifest = json.loads((root / "assets/models/manifest.json").read_text())
        hashes = {a["file"]: a["sha256"] for a in manifest["files"]}
        models = root / "assets/models"
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.policy = AmoPolicy(self.profile, models / "amo_jit.pt", models / "adapter_jit.pt",
            models / "adapter_norm_stats.pt", cache_directory=cache,
            policy_sha256=hashes["amo_jit.pt"], adapter_sha256=hashes["adapter_jit.pt"],
            statistics_sha256=hashes["adapter_norm_stats.pt"])
        self.kernel = G1ControllerKernel(self.profile, self.port, self.policy)
        self.body_actuators = np.array([self.actuator(j) for j in self.profile.robot.joint_names])
        self.arm_target = data.qpos[self.port.q[ARMS]].copy()
        self.gates = {side: ArmLimit(calibration, side, settings.arm_rate, settings.arm_authority_seconds,
                                     self.arm_target[arm]) for side, arm in SIDES.items()}
        self.hand_q, self.hand_act, self.hand_target, self.closed, self.hand_pace = {}, {}, {}, {}, {}
        for side in ("left", "right"):
            names = calibration["hand_joints"][side]
            self.hand_q[side] = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
            self.hand_act[side] = np.array([self.actuator(n) for n in names])
            self.hand_target[side] = np.zeros(7)
            ray = np.array(calibration["hand_closed"][side]) * calibration["hand_close_fraction"]
            ray[:3] *= calibration["thumb_close_fraction"]
            self.closed[side] = ray
            closed = np.abs(np.array(calibration["hand_closed"][side]))
            self.hand_pace[side] = np.maximum(closed / closed.max(), 0.001)
            for name, actuator in zip(names, self.hand_act[side]):
                configure_finger_servo(model, actuator, model.joint(name).id, settings)
        configure_hand_contacts(model, settings)
        self.tick_count = 0
        self.kernel.handle_command(ControllerCommand.start)
        self.kernel.set_arm_targets(tuple(self.arm_target))

    def actuator(self, joint):
        matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == self.model.joint(joint).id)
        if len(matches) != 1:
            raise ValueError(f"Expected one actuator for {joint}")
        return int(matches[0])

    def hold_arms(self):
        """Hold each arm at its gate's current reference, not the measured joints, which sag under load."""
        for side, arm in SIDES.items():
            self.arm_target[arm] = self.gates[side].q

    def velocity(self, vx=0.0, vy=0.0, wz=0.0):
        moving = bool(vx or vy or wz)
        if moving != self.kernel.status.walk_enabled:
            self.kernel.handle_command(ControllerCommand.toggle_walk)
        self.kernel.set_velocity(VelocityCommand(linear_xy=(vx, vy), angular_z=wz))

    def tick(self):
        """Step one control period of physics."""
        dt = 1.0 / self.settings.control_hz
        targets = self.arm_target.copy()
        kp, kd = np.zeros_like(targets), np.zeros_like(targets)
        q, v = self.port.q[ARMS], self.port.v[ARMS]
        for side, arm in SIDES.items():
            targets[arm], kp[arm], kd[arm] = self.gates[side].step(
                targets[arm], self.data.qpos[q[arm]], dt)
            delta = self.settings.hand_rate * dt * self.hand_pace[side]
            acts = self.hand_act[side]
            current = self.data.ctrl[acts]
            self.data.ctrl[acts] = current + np.clip(self.hand_target[side] - current, -delta, delta)
        self.kernel.set_arm_targets(tuple(targets))
        if self.tick_count % (self.settings.control_hz // self.settings.policy_hz) == 0:
            self.kernel.tick()
        self.tick_count += 1
        for _ in range(round(dt / self.settings.timestep)):
            torque = self.port.torques()
            torque[ARMS] = kp * (targets - self.data.qpos[q]) - kd * self.data.qvel[v]
            self.data.ctrl[self.body_actuators] = np.clip(torque,
                self.model.actuator_ctrlrange[self.body_actuators, 0],
                self.model.actuator_ctrlrange[self.body_actuators, 1])
            mujoco.mj_step(self.model, self.data)

"""The body host: the one process that owns the G1's legs, a 50 Hz loop with the kernel, the
AMO policy on CPU, both arm gates and the remote. Clients send intent over UDP JSON
(127.0.0.1:29400) and read state back; a lost client stands the robot, a stop damps it.

    python -m homebody.backends.g1.body_host serve | start | status | sit | stop | deploy | halt
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import json
import math
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from pathlib import Path

import numpy as np

from homebody.backends.g1 import profile as g1
from homebody.backends.g1.assets import load_policy, validate_assets
from homebody.backends.g1.unitree import UnitreeG1Port
from homebody.backends.g1.wireless import UnitreeWirelessInput
from homebody.backends.g1.writer import SoleWriter, lease_path
from homebody.primitives.navigation.actions.lower_body import V_PLANAR_MAX, WZ_MAX
from homebody.primitives.navigation.contracts import HardwareUnavailable
from homebody.real2sim.exploration.settings import current, from_json, install

from .assets import REPO
from .kernel import _real_controller_kernel_type, body_stop_reason

DEFAULT_BIND = "127.0.0.1:29400"  # clear of the DDS ports (7400 + 250·domain)
PIDFILE = Path("/tmp/homebody-body-host.pid")
HOST_LEASE = Path("/tmp/homebody-g1-body-host.lock")
LOGFILE = Path("/tmp/homebody-body-host.log")
UNIT = "homebody-body-host"
CONTROL_HZ = 50.0  # host loop; AMO keeps its own 20 ms policy period
AMO_PERIOD_S = 0.02
LOOP_RESYNC_S = 0.05
TWIST_DEADMAN_S = 0.2
HEARTBEAT_LOST_S = 0.5
LISTENER_RETENTION_S = 5.0  # a silent client still receives state this long
STATE_RECOVERY_S = 3.0
SIT_CONFIRM_S = 3.0
START_MAX_TILT_DEG = 25.0
AUTO_ARM_TICKS = round(0.1 * CONTROL_HZ)  # 100 ms of changing states before the host arms itself
HALT_DAMP_S = 1.0
LINK_STALE_TICKS = round(0.1 * CONTROL_HZ)  # 100 ms of changing lowstate
STATE_STALE_S = 0.5  # the client's view: a host silent this long is gone
READ_FAILS_STOP = round(0.2 * CONTROL_HZ)  # 200 ms of failed lowstate reads
MAX_DATAGRAMS_PER_TICK = 64
STIFFNESS_LEVELS = (0.5, 1.2)
RT_PRIORITY = 60
CPU_THREADS = 2
INTENT_KEYS = ("arms", "torso", "walk", "vel", "cmd")
ENV_PREFIXES = ("PYTHON", "CYCLONE", "LD_LIBRARY", "UNITREE", "ROS_", "MUJOCO")
COMPUTE_ENV = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY")
EVENTS_KEPT = 32


def parse_addr(text: str) -> tuple[str, int]:
    host, _, port = str(text).rpartition(":")
    return host or "127.0.0.1", int(port)


def _bind_control_socket(bind: str) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(parse_addr(bind))  # no SO_REUSEADDR: reserve the sole control endpoint
        sock.setblocking(False)
    except OSError as exc:
        sock.close()
        raise SystemExit(
            f"body host: cannot reserve control address {bind}: {exc}. "
            "Refusing to open another motor interface. An existing host may still be balancing; "
            "secure the robot and stop that host through its own checkout before retrying. "
            "Do not change the address or remove locks to bypass the conflict."
        ) from exc
    return sock


def lease_held(path: Path | str) -> bool:
    """Whether another open file holds `path`'s flock."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def host_alive() -> bool:
    return lease_held(HOST_LEASE)


def running_pid() -> int | None:
    try:
        pid = int(PIDFILE.read_text().strip())
        if b"body_host" not in Path(f"/proc/{pid}/cmdline").read_bytes():
            return None  # a stale file naming a reused pid
        return pid
    except (OSError, ValueError):
        return None


def tilt_deg(quat_wxyz) -> float:
    """The pelvis z axis's angle from vertical."""
    _w, x, y, _z = (float(v) for v in quat_wxyz)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def _claim_time(claim: str) -> float:
    try:
        return float(str(claim).split(":", 1)[0])
    except ValueError:
        return 0.0


# -- the kernel: the real subclass, with a stop that damps every joint ---------------------------
def host_kernel_type():
    from holosoma_inference.g1_control.contracts import LowCommand, VelocityCommand
    from holosoma_inference.g1_control.runtime import ControllerCommand

    base = _real_controller_kernel_type()

    class HostKernel(base):
        """The real kernel, damping every joint after any stop: zero stiffness, the profile's
        damping, no feedforward, the measured pose as target; only A undamps it."""

        damped = False

        def handle_command(self, command) -> None:
            if command in (ControllerCommand.initialize, ControllerCommand.start):
                self.damped = False
            super().handle_command(command)

        def _stop(self) -> None:
            self._walk_enabled = False  # A redeploys standing, never into a stale walk
            self._velocity = VelocityCommand()
            self.damped = True
            super()._stop()

        def _observe_link_freshness(self, state) -> None:
            limit = self.profile.loop.repeated_state_stop_ticks
            frozen_ramp = (self._initializing and not self._active and limit is not None
                           and state.holosoma_vector() == self._last_state_vector
                           and self._frozen_state_ticks + 1 >= limit)
            super()._observe_link_freshness(state)
            if frozen_ramp:  # the vendor counter covers active AMO; apply its same limit to Y
                self._stop()
                self._stopped_by_watchdog = True

        def limp_command(self, state):
            """Every stopped tick sends zero stiffness and the profile damping."""
            return self.damped_command(state)

        @property
        def limp(self) -> bool:
            """Damped and neither balancing nor ramping: every command is damped_command."""
            return self.damped and not (self._active or self._initializing)

        def damped_command(self, state):
            offsets = self.profile.robot.joint_offsets_rad
            return LowCommand(
                positions=tuple(q + o for q, o in zip(state.joint_positions, offsets)),
                velocities=(0.0,) * 29,
                feedforward_torques=(0.0,) * 29,
                stiffness=(0.0,) * 29,
                damping=tuple(float(d) for d in self.profile.robot.damping),
            )

        def _command_for_state(self, state):
            command = super()._command_for_state(state)
            return self.limp_command(state) if self.limp else command

        def republish_arms(self, command) -> None:
            if self.limp:  # the host writes a fresh damped command instead (BodyHost.tick)
                return
            super().republish_arms(command)

    return HostKernel


def _limit_cpu_threads() -> None:
    """Limit torch to the body settings' thread count."""
    with contextlib.suppress(ImportError):
        import torch

        torch.set_num_threads(current().body.threads)




# -- the host ----------------------------------------------------------------------------------
class BodyHost:
    """The loop. `port` is a RobotPort (UnitreeG1Port on the robot, a MuJoCo port in simulation);
    `plant` (simulation only) advances physics by one tick after the command is written."""

    def __init__(
        self,
        port,
        *,
        bind: str = DEFAULT_BIND,
        sock: socket.socket | None = None,
        wireless=None,
        plant=None,
        console_deploy: bool = False,
        clock=time.monotonic,
    ):
        self.port, self.plant, self.clock = port, plant, clock
        self.policy = load_policy(g1.PROFILE)
        _limit_cpu_threads()
        self.kernel = host_kernel_type()(g1.PROFILE, port, self.policy)
        self.gates = {side: g1.ArmGate(side=side) for side in ("left", "right")}
        state = port.read_state()
        self.arm_target14 = np.asarray(state.joint_positions[15:29], dtype=float)
        self.settle_to_idle = False   # set by Y's ramp, cleared by the first arms a skill sends
        for side, gate in self.gates.items():
            gate.prime(np.asarray(state.joint_positions[g1.ARM_SLICE[side]], dtype=float))
            gate.arm(1.0, ramp_s=2.0)
        self.kernel.set_arm_targets(tuple(float(v) for v in self.arm_target14))
        self.wireless = wireless
        self.console_deploy = bool(console_deploy)  # simulation tests only: Y/A from the client
        self.sock = _bind_control_socket(bind) if sock is None else sock
        now = clock()
        self.clients: dict[tuple[str, int], float] = {}  # every listener, by when last heard
        self.instance = uuid.uuid4().hex
        self.driver: str | None = None  # the claim whose intent is obeyed
        self.driver_heard = self.last_twist = now - 10.0
        self.driver_lost = self.twist_latched = True
        self.ignored_claims: set[str] = set()
        self.armed = False
        self.last_event = ""
        self.halt_at: float | None = None  # when `halt` arrived: the host ends HALT_DAMP_S later
        self.fresh_ticks = 0
        self.stale_ticks = 0  # consecutive ticks whose lowstate did not change
        self.deploy_needs_release = False  # a link fault requires fresh neutral, then Y/A
        self.last_vector: tuple | None = None
        self.sit_requested_at = -1e9
        self.counts = {"tick_errors": 0, "read_errors": 0, "bad_messages": 0}
        self.read_fails = 0
        self.events: deque = deque(maxlen=EVENTS_KEPT)
        self.not_following = None
        self.loop_hz = 0.0
        self.ticks = 0
        self.last_state = state
        self.last_command = None
        self.stop = threading.Event()
        self.was_running = False
        self.scheduling = "not set"
        self.simulated = False  # a MuJoCo simulation, never mistaken for the robot's host

    def event(self, text: str) -> None:
        self.events.append([round(time.time(), 3), text])
        self.last_event = text  # sent in every status message
        with contextlib.suppress(OSError, ValueError):
            print(f"[body-host] {time.strftime('%H:%M:%S')} {text}", flush=True)

    def _count(self, key: str, text: str) -> None:
        self.counts[key] += 1
        n = self.counts[key]
        if n in (1, 10, 100) or n % 1000 == 0:
            self.event(f"{text} ({key} {n})")

    # -- inbound ----------------------------------------------------------------------------
    def drain(self) -> None:
        for _ in range(MAX_DATAGRAMS_PER_TICK):
            try:
                data, addr = self.sock.recvfrom(65536)
            except OSError:  # BlockingIOError: nothing waiting
                return
            try:
                msg = json.loads(data)
                if not isinstance(msg, dict):
                    raise ValueError("a message is a JSON object")  # noqa: TRY004 -- handled as a bad message below
                self.apply(msg, addr)
            except (ValueError, TypeError, KeyError) as exc:
                self._count("bad_messages", f"refused from {addr[0]}:{addr[1]}: {exc}")

    def apply(self, msg: dict, addr: tuple[str, int]) -> None:
        now = self.clock()
        if now - self.clients.get(addr, -1e9) > HEARTBEAT_LOST_S:
            self.event(f"client connected from {addr[0]}:{addr[1]}")
        self.clients[addr] = now
        claim = str(msg.get("claim") or f"{addr[0]}:{addr[1]}")
        if claim == self.driver:
            self.driver_heard, self.driver_lost = now, False
        intent = {k: msg[k] for k in INTENT_KEYS if msg.get(k) is not None}
        if not intent:
            return
        if claim != self.driver and not self._take_over(claim, now):
            if claim not in self.ignored_claims:
                self.ignored_claims.add(claim)
                self.event(f"ignoring {addr[0]}:{addr[1]}: an older client than the driver")
            return
        self._apply_intent(intent, now)

    def _take_over(self, claim: str, now: float) -> bool:
        current = self.driver
        if current is not None and not self.driver_lost:
            if _claim_time(claim) < _claim_time(current):
                return False
            self.stand(f"a newer client took over from {current}")
        self.driver, self.driver_heard, self.driver_lost = claim, now, False
        self.event(f"driver: {claim}")
        self.stand_tall("a new driver")   # a driver that died lowered leaves no crouch behind
        return True

    def _apply_intent(self, intent: dict, now: float) -> None:
        from holosoma_inference.g1_control.contracts import TorsoCommand, VelocityCommand
        from holosoma_inference.g1_control.runtime import ControllerCommand

        if not self.kernel.status.active:
            intent.pop("walk", None)  # stopping or stopped: A redeploys standing, never walking
            intent.pop("vel", None)
        if "arms" in intent:
            arms = np.asarray(intent["arms"], dtype=float)
            if arms.shape != (14,) or not np.isfinite(arms).all():
                raise ValueError(f"arms must be 14 finite numbers, got shape {arms.shape}")
            self.arm_target14 = arms
            self.settle_to_idle = False
        if "torso" in intent:
            pitch, roll, height = (float(v) for v in intent["torso"])
            if not all(math.isfinite(v) for v in (pitch, roll, height)):
                raise ValueError("torso must be finite")
            self.kernel.set_torso(TorsoCommand(height_m=height, roll_rad=roll, pitch_rad=pitch))
        if "walk" in intent and self.kernel.status.walk_enabled != bool(intent["walk"]):
            self.kernel.handle_command(ControllerCommand.toggle_walk)
        if "vel" in intent:
            vx, vy, wz = (float(v) for v in intent["vel"])
            if not all(math.isfinite(v) for v in (vx, vy, wz)):
                raise ValueError("vel must be finite")
            planar = math.hypot(vx, vy)
            if planar > V_PLANAR_MAX:
                vx, vy = vx * V_PLANAR_MAX / planar, vy * V_PLANAR_MAX / planar
            wz = max(-WZ_MAX, min(WZ_MAX, wz))
            self.kernel.set_velocity(VelocityCommand(linear_xy=(vx, vy), angular_z=wz))
            self.last_twist, self.twist_latched = now, False
        cmd = intent.get("cmd")
        if cmd is None:
            return
        if cmd == "arm":
            self._arm("on the client's ARM")
        elif cmd == "stand":
            self.stand("console stand")
        elif cmd == "halt":
            self.halt("the launcher's Ctrl+\\")
        elif cmd == "sit":
            if now - self.sit_requested_at <= SIT_CONFIRM_S:
                self.sit_requested_at = -1e9
                self.request_stop("SIT confirmed")
            else:
                self.sit_requested_at = now
                self.event(f"sit requested: send sit again within {SIT_CONFIRM_S:.0f} s")
        elif cmd in ("initialize", "start") and self.console_deploy:
            self._arm("on the simulation client's deploy")
            self.deploy(cmd, "client, simulation")
        else:
            raise ValueError(f"unknown cmd {cmd!r}")

    def _arm(self, how: str) -> None:
        """Make rt/lowcmd live and damped: every joint limp until Y then A."""
        if self.armed:
            return
        self.armed = self.kernel.damped = True
        self.event(f"ARMED {how}: rt/lowcmd live, every joint LIMP until Y then A")

    @property
    def link_live(self) -> bool:
        return self.read_fails == 0 and self.stale_ticks < LINK_STALE_TICKS

    def stand_tall(self, why: str) -> None:
        """Return the torso to the profile's standing height; `stand` leaves the torso alone."""
        from holosoma_inference.g1_control.contracts import TorsoCommand

        standing = float(self.kernel.profile.loop.desired_base_height_m)
        was = getattr(getattr(self.kernel, "_torso", None), "height_m", standing)
        self.kernel.set_torso(TorsoCommand(height_m=standing))
        if abs(float(was) - standing) > 1e-6:
            self.event(f"TORSO ({why}): {float(was):.3f} -> {standing:.3f} m")

    def stand(self, why: str) -> None:
        from holosoma_inference.g1_control.contracts import VelocityCommand
        from holosoma_inference.g1_control.runtime import ControllerCommand

        moving = self.kernel.status.walk_enabled or not self.twist_latched
        self.kernel.set_velocity(VelocityCommand())
        if self.kernel.status.walk_enabled:
            self.kernel.handle_command(ControllerCommand.toggle_walk)
        self.twist_latched = True
        if moving:
            self.event(f"STAND ({why}): walk off, zero velocity, policy still balancing")

    def deploy(self, name: str, source: str) -> None:
        """Y (initialize) or A (start): only armed, only on a live state link, never after a
        halt; A only on an upright body."""
        from holosoma_inference.g1_control.runtime import ControllerCommand

        key = "A" if name == "start" else "Y"
        if not self.armed:  # a policy stepped before ARM would jump onto the wire at ARM
            self.event(f"{key} ignored ({source}): not armed; the host arms itself after "
                       f"{AUTO_ARM_TICKS} live states")
            return
        if self.halt_at is not None:
            self.event(f"{key} REFUSED ({source}): HALTED — the host is ending")
            return
        if not self.link_live:
            self.event(f"{key} REFUSED ({source}): no live robot state ({self.read_fails} failed "
                       f"reads, unchanged for {self.stale_ticks} ticks)")
            return
        if self.deploy_needs_release:
            self.event(f"{key} REFUSED ({source}): release remote buttons on a fresh link, "
                       "then press Y or A again")
            return
        tilt = tilt_deg(self.last_state.quaternion_wxyz)
        if name == "start" and tilt > START_MAX_TILT_DEG:
            self.event(f"A REFUSED ({source}): pelvis tilted {tilt:.0f} deg — stand the robot up")
            return
        self.stand_tall(f"{key} deploys standing")
        self.kernel.handle_command(ControllerCommand(name))
        self.event(f"{name} ({source})")

    def request_stop(self, why: str) -> None:
        """Stop the policy at once; every joint is damped from this tick (HostKernel)."""
        from holosoma_inference.g1_control.runtime import ControllerCommand

        was_active = self.kernel.status.active or self.kernel.status.initializing
        self.kernel.handle_command(ControllerCommand.stop)
        if was_active or not self.kernel.damped:
            self.event(f"{why}: policy STOPPED — every joint damped (limp)")

    def halt(self, why: str) -> None:
        if self.halt_at is None:
            self.halt_at = self.clock()
            self.request_stop(f"HALT ({why})")
            self.event(
                f"HALT: the host ends in {HALT_DAMP_S:.0f} s; nothing will command the robot"
            )

    def _track_link(self, state) -> None:
        vector = (state.joint_positions, state.quaternion_wxyz)
        self.stale_ticks = 0 if vector != self.last_vector else self.stale_ticks + 1
        self.fresh_ticks = self.fresh_ticks + 1 if vector != self.last_vector else 0
        self.last_vector = vector

    def deadmen(self) -> None:
        from holosoma_inference.g1_control.contracts import VelocityCommand

        now = self.clock()
        if not self.twist_latched and now - self.last_twist > TWIST_DEADMAN_S:
            self.kernel.set_velocity(VelocityCommand())
            self.twist_latched = True
        if (self.driver is not None and not self.driver_lost
                and now - self.driver_heard > HEARTBEAT_LOST_S):
            self.driver_lost = True
            self.stand(f"driver silent {now - self.driver_heard:.2f} s")
            if self.armed and not self.kernel.ramping and not self.kernel.limp:
                # A lost driver's arms settle to the idle pose at IDLE_SETTLE_RAD_S.
                self.settle_to_idle = True
                self.event("driver gone: arms settling to the idle pose")
        if (
            self.halt_at is not None
            and now - self.halt_at >= HALT_DAMP_S
            and not self.stop.is_set()
        ):
            self.event("HALT: body host ENDED")
            self.stop.set()

    def poll_remote(self) -> None:
        if self.wireless is None:
            return
        from holosoma_inference.g1_control.runtime import ControllerCommand

        try:
            sample = self.wireless.poll(self.port)
        except Exception as exc:  # noqa: BLE001 -- the remote is a convenience; the legs are not
            self._count("tick_errors", f"remote read failed: {type(exc).__name__}: {exc}")
            return
        if (self.deploy_needs_release and self.link_live
                and self.fresh_ticks >= AUTO_ARM_TICKS and sample.neutral):
            self.deploy_needs_release = False
        for name in sample.commands:
            if name == "kill":
                self.request_stop(f"KILL (L1+R1, keys {sample.key!r})")
            elif name == "stop":
                self.request_stop(f"B ({sample.key})")
            elif name in ("start", "initialize"):
                self.deploy(name, str(sample.key))
            elif name == "toggle_walk" and not self.kernel.status.active:
                self.event(f"walk ignored ({sample.key}): no policy running; A deploys standing")
            else:
                self.kernel.handle_command(ControllerCommand(name))
                if name.startswith("stiffness"):
                    lo, hi = STIFFNESS_LEVELS
                    self.port.stiffness_level = min(max(self.port.stiffness_level, lo), hi)
                self.event(f"{name} ({sample.key})")

    # -- one tick -------------------------------------------------------------------------------
    def tick(self, period: float, amo_due: bool) -> None:

        self.drain()
        self.deadmen()
        try:
            state = self.port.read_state()
        except Exception as exc:  # noqa: BLE001 -- a dropped state link: the last command keeps flowing
            self.read_fails += 1
            self.fresh_ticks = 0
            self._count("read_errors", f"state read failed: {type(exc).__name__}: {exc}")
            status = self.kernel.status
            if self.read_fails >= READ_FAILS_STOP:
                self.deploy_needs_release = True
                if status.active or status.initializing:
                    self.request_stop(f"state link lost for {READ_FAILS_STOP} reads")
            self._resend_last(self.last_state)
            if self.plant is not None:
                self.plant.step(self, period)
            return
        if self.read_fails:
            self.event(f"state link back after {self.read_fails} failed reads")
            self.read_fails = 0
        self.last_state = state
        self._track_link(state)
        measured14 = np.asarray(state.joint_positions[15:29], dtype=float)
        limp = self.kernel.limp
        if self.kernel.ramping or limp or not self.armed:
            # The ramp or a limp body owns the arms: the target follows the measured pose.
            self.arm_target14 = measured14.copy()
            self.settle_to_idle = bool(self.kernel.ramping)
            for side, gate in self.gates.items():
                sl = g1.ARM_SLICE[side]
                gate.prime(measured14[sl.start - 15 : sl.stop - 15])
        if not self.armed:  # nothing on the wire before ARM; the gates start from the pose then
            if self.plant is not None:
                self.plant.step(self, period)
            self.ticks += 1
            self.publish(state)
            # Arm after AUTO_ARM_TICKS changing states, so a frozen SDK buffer never arms.
            if self.fresh_ticks >= AUTO_ARM_TICKS:
                self._arm("at start")
            return
        try:
            if self.kernel.ramping:
                target = measured14
                self.kernel.set_arm_gains(None, None)
            else:
                if self.settle_to_idle:
                    idle = np.asarray(g1.PROFILE.robot.idle_arm_positions, dtype=float)
                    step = g1.IDLE_SETTLE_RAD_S * period
                    self.arm_target14 = self.arm_target14 + np.clip(
                        idle - self.arm_target14, -step, step)
                    self.settle_to_idle = bool(np.abs(idle - self.arm_target14).max() > 1e-6)
                target = self.arm_target14.copy()
                kp14, kd14 = np.zeros(14), np.zeros(14)
                for side, gate in self.gates.items():
                    sl = g1.ARM_SLICE[side]
                    measured = np.asarray(state.joint_positions[sl], dtype=float)
                    shaped = gate(target[sl.start - 15 : sl.stop - 15], measured, period)
                    target[sl.start - 15 : sl.stop - 15] = shaped["q_out"]
                    kp14[sl.start - 15 : sl.stop - 15] = shaped["kp"]
                    kd14[sl.start - 15 : sl.stop - 15] = shaped["kd"]
                    if shaped["not_following"]:
                        self.not_following = [
                            side,
                            float(shaped["cmd_rate"]),
                            float(shaped["meas_rate"]),
                        ]
                self.kernel.set_arm_gains(kp14, kd14)
            self.kernel.set_arm_targets(tuple(float(v) for v in target))
        except Exception as exc:  # noqa: BLE001 -- an arm fault holds the arms where they are; the legs tick on
            self._count("tick_errors", f"arm gate fault, arms held: {type(exc).__name__}: {exc}")
            self.arm_target14 = measured14.copy()
            self.kernel.set_arm_targets(tuple(float(v) for v in measured14))
        self.poll_remote()
        if amo_due:
            self.last_command = self.kernel.tick().command
        elif self.kernel.limp:  # a stop between AMO ticks is damped at once
            self.last_command = self.kernel.limp_command(state)
            self.port.write_command(self.last_command)
        else:
            self.kernel.republish_arms(self.last_command)
        running = bool(self.kernel.status.active or self.kernel.status.initializing)
        if self.was_running and not running:
            reason = body_stop_reason(self.kernel.status) or "on request"
            if self.kernel.status.stopped_by_watchdog:
                self.deploy_needs_release = True
                self.fresh_ticks = 0
            self.event(f"policy stopped: {reason} — every joint damped (limp)")
            # The vendor watchdog stops after publishing its last policy command.
            self._resend_last(state)
        self.was_running = running
        if self.plant is not None:
            self.plant.step(self, period)
        self.ticks += 1
        self.publish(state)

    def _resend_last(self, state) -> None:
        """Resend the last command on a tick without one; a limp body gets a fresh damped
        command."""
        if not self.armed:
            return
        if self.kernel.limp:
            self.last_command = self.kernel.limp_command(state)
        if self.last_command is not None:
            with contextlib.suppress(Exception):
                self.port.write_command(self.last_command)

    def publish(self, state) -> None:
        now = self.clock()
        for addr in [a for a, heard in self.clients.items() if now - heard > LISTENER_RETENTION_S]:
            del self.clients[addr]
        if not self.clients:
            return
        status = self.kernel.status
        msg = {
            "tick": self.ticks,
            "instance": self.instance,
            "t": round(time.time(), 4),
            "q": [round(float(v), 5) for v in state.joint_positions],
            "dq": [round(float(v), 4) for v in state.joint_velocities],
            "quat": [round(float(v), 6) for v in state.quaternion_wxyz],
            "gyro": [round(float(v), 5) for v in state.base_angular_velocity],
            "base": [round(float(v), 4) for v in state.base_position],
            "vlin": [round(float(v), 4) for v in state.base_linear_velocity],
            "armed": self.armed,
            "driver": self.driver,
            "active": status.active,
            "initializing": status.initializing,
            "walk": status.walk_enabled,
            "init_tick": status.initialization_tick,
            "watchdog": status.stopped_by_watchdog,
            "error": status.last_error,
            "event": self.last_event,
            "ramping": bool(self.kernel.ramping),
            "init_ticks": int(self.kernel.profile.loop.initialization_ticks),
            "simulated": self.simulated,
            "damped": bool(self.kernel.damped),
            "halted": self.halt_at is not None,
            "link_live": self.link_live,
            "stiffness": round(float(self.port.stiffness_level), 3),
            "loop_hz": round(self.loop_hz, 1),
            "counts": self.counts,
            "events": list(self.events)[-8:],
            "not_following": self.not_following,
            "remote": self.wireless is not None,
            "scheduling": self.scheduling,
        }
        self.not_following = None
        data = json.dumps(msg, separators=(",", ":")).encode()
        for addr in self.clients:
            with contextlib.suppress(OSError):
                self.sock.sendto(data, addr)

    def run(self) -> None:
        period = 1.0 / CONTROL_HZ
        next_tick = next_amo = self.clock()
        rate_at, rate_ticks = next_tick, 0
        while not self.stop.is_set():
            now = self.clock()
            amo_due = now >= next_amo
            if amo_due:
                next_amo += AMO_PERIOD_S
                if now - next_amo > LOOP_RESYNC_S:
                    next_amo = now + AMO_PERIOD_S
            try:
                self.tick(period, amo_due)
            except Exception as exc:  # noqa: BLE001 -- any tick fault stops the policy
                self._count("tick_errors", f"tick failed: {type(exc).__name__}: {exc}")
                self.deploy_needs_release = True
                self.fresh_ticks = 0
                try:
                    self.request_stop("unexpected control tick failure")
                except Exception as stop_exc:  # noqa: BLE001 -- still resend the hold
                    self._count("tick_errors", f"policy stop failed: {type(stop_exc).__name__}: "
                                f"{stop_exc}")
                finally:
                    self._resend_last(self.last_state)
            rate_ticks += 1
            if now - rate_at >= 1.0:
                self.loop_hz, rate_at, rate_ticks = rate_ticks / (now - rate_at), now, 0
            next_tick += period
            if self.clock() - next_tick > LOOP_RESYNC_S:
                next_tick = self.clock()
            delay = next_tick - self.clock()
            if delay > 0:
                time.sleep(delay)


def default_cores() -> str:
    n = os.cpu_count() or 1
    return f"{n - 2},{n - 1}" if n >= 8 else ""


def _realtime(cores: str) -> str:
    """Pin the cores and set FIFO priority before the SDK and torch start threads; returns
    what was set."""
    got = []
    if cores:
        try:
            os.sched_setaffinity(0, {int(c) for c in cores.split(",") if c.strip()})
            got.append(f"cores {cores}")
        except (OSError, ValueError) as exc:
            got.append(f"cores {cores} REFUSED ({exc})")
    if os.sched_getscheduler(0) != os.SCHED_FIFO:
        with contextlib.suppress(OSError):
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(RT_PRIORITY))
    if os.sched_getscheduler(0) == os.SCHED_FIFO:
        got.append(f"SCHED_FIFO {os.sched_getparam(0).sched_priority}")
    else:
        got.append("not real-time (default scheduler: start it with sudo)")
    return ", ".join(got)


def serve(args) -> int:
    if args.log:  # opened by this account: root may not create its file in sticky /tmp
        fd = os.open(args.log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    iface = args.iface
    if not iface:
        raise SystemExit("body host: robot.iface (or --iface) names the robot-LAN interface")
    with contextlib.ExitStack() as resources:
        # Reserve the endpoint for the whole startup.
        sock = resources.enter_context(_bind_control_socket(args.bind))
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # AMO on CPU: no shared GPU, no driver stall
        os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        scheduling = _realtime(args.cores)
        validate_assets()
        if lease_held(lease_path()):
            raise SystemExit(
                f"body host: an in-process run holds {lease_path()} and owns rt/lowcmd — two writers "
                "on the same motors is refused; end that run first"
            )
        lease = SoleWriter(HOST_LEASE)  # one host per machine
        resources.callback(lease.release)
        port = UnitreeG1Port(g1.PROFILE, iface)
        host = BodyHost(port, bind=args.bind, sock=sock, wireless=UnitreeWirelessInput())
        host.scheduling, host.simulated = scheduling, bool(args.simulated)
        gc.collect()
        gc.freeze()
        PIDFILE.write_text(str(os.getpid()))
        # SIGTERM, SIGHUP and SIGINT are logged and ignored; SIGUSR1 or SIGKILL ends the host.
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(
                sig, lambda s, _f: host.event(f"ignored {signal.Signals(s).name}: the legs stay up")
            )
        signal.signal(
            signal.SIGUSR1,
            lambda s, _f: (
                host.stop.set()
                if not (host.kernel.status.active or host.kernel.status.initializing)
                else host.event("stop refused: policy active or ramping (sit first)")
            ),
        )
        host.event(f"serving on {args.bind} · iface {iface} · {scheduling} · AMO on CPU")
        code = 0
        try:
            host.run()
        except BaseException:  # noqa: BLE001 -- printed before the hard exit below, which would swallow it
            traceback.print_exc()
            code = 1
        finally:
            with contextlib.suppress(OSError):
                PIDFILE.unlink()
    host.event("host ended")
    # skip interpreter teardown: the SDK's DDS threads can block it
    os._exit(code)


def _stale_host(simulated: bool) -> str:
    """Why a live host must be replaced, or "": a simulation host when the robot is launching,
    or a stopped host on a dead robot link."""
    client = BodyClient(DEFAULT_BIND)
    answering = client.wait_fresh(2.0)
    s = client.state or {}
    client.closed.set()
    if not answering:
        return "not answering"
    if s.get("simulated") and not simulated:
        return "a MuJoCo simulation run, not the robot"
    if simulated and not s.get("simulated"):
        return "REAL"
    if s.get("active"):
        return ""  # a balancing robot is never taken from its host
    if s.get("watchdog") or s.get("link_live") is False:
        return "stopped on a dead robot link"
    return ""


def _unit_pid() -> int | None:
    for c in (["systemctl"], ["systemctl", "--user"]):
        out = subprocess.run([*c, "show", "-p", "MainPID", "--value", UNIT], capture_output=True,
                             check=False)
        if out.stdout.strip() not in (b"", b"0"):
            return int(out.stdout)
    return None


def _end_stale_host() -> bool:
    pid = running_pid() or _unit_pid()
    if pid:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGUSR1)
    for i in range(50):
        time.sleep(0.2)
        if pid is None or not Path(f"/proc/{pid}").exists():
            break
        if i == 25:  # then SIGKILL a host that has not ended
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
    for _ in range(25):  # wait for systemd to release the unit name
        states = [
            subprocess.run([*c, "is-active", UNIT], capture_output=True, text=True,
                           check=False).stdout
            for c in (["systemctl"], ["systemctl", "--user"])
        ]
        if not any(x.strip() in ("active", "activating", "deactivating") for x in states):
            return not host_alive()
        time.sleep(0.2)
    return False


def halt(args) -> int:
    """The kill (a launcher's Ctrl+\\): the host stops the policy, damps every joint and ends;
    one still alive HALT_DAMP_S + 2 s later is SIGKILLed."""
    pid = running_pid() or _unit_pid()
    client = BodyClient(args.bind)
    for _ in range(3):
        client.send(cmd="halt")
        time.sleep(0.05)
    client.closed.set()

    def gone() -> bool:
        return not host_alive() and not (pid and Path(f"/proc/{pid}").exists())

    give_up = time.monotonic() + HALT_DAMP_S + 2.0
    try:
        while time.monotonic() < give_up and not gone():
            time.sleep(0.1)
    except KeyboardInterrupt:  # the halt is sent; Ctrl-C only ends the wait
        print("halt sent; not waiting for the host to end (Ctrl-C)")
        return 0
    if not gone() and pid:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
        time.sleep(1.0)
    if gone():
        print("lower body controller ENDED: policy stopped, joints damped, no commands after")
        return 0
    print(f"body host {pid} is still alive: kill -KILL {pid}", file=sys.stderr)
    return 1


def start(args) -> int:
    """Start a detached host: a transient system unit (FIFO, OOM-exempt) with passwordless sudo,
    a user unit otherwise."""
    if host_alive():
        why = _stale_host(bool(args.simulated))
        if why == "REAL":
            print("a real body host is running: not starting a simulation over it", file=sys.stderr)
            return 1
        if not why:
            print(f"body host already running (pid {running_pid()})")
            return 0
        print(f"body host (pid {running_pid()}) is {why}: replacing it", file=sys.stderr)
        if not _end_stale_host():
            print("the stale body host would not exit: `sudo systemctl kill -s KILL homebody-body-host`")
            return 1
    try:
        with _bind_control_socket(args.bind):
            pass  # fail before spawning; serve reserves it again
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if LOGFILE.exists() and LOGFILE.stat().st_size > 50_000_000:
        LOGFILE.replace(LOGFILE.with_suffix(".log.1"))
    cores = args.cores or default_cores()
    cmd = [sys.executable, "-m", "homebody.backends.g1.body_host", "serve", "--bind", args.bind]
    cmd += ["--log", str(LOGFILE)] + (["--cores", cores] if cores else [])
    cmd += (["--iface", args.iface] if args.iface else []) + (["--simulated"] if args.simulated else [])
    cmd = cmd[:3] + (["--settings-json", args.settings_json] if args.settings_json else []) + cmd[3:]
    common = ["--unit", UNIT, "--collect", f"--working-directory={REPO}"]
    common += [f"--setenv=HOMEBODY_ROOT={REPO}"]
    common += ["-p", "StandardOutput=journal", "-p", "StandardError=journal"]
    common += [f"--setenv={k}={v}" for k, v in os.environ.items()
               if k.startswith(ENV_PREFIXES) or k in COMPUTE_ENV]
    base_env = ("PATH", "HOME", "USER")
    common += [f"--setenv={k}={os.environ[k]}" for k in base_env if os.environ.get(k)]
    use_sudo = args.use_sudo and (
        subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False).returncode == 0
    )
    if use_sudo:
        unit = ["sudo", "-n", "systemd-run", *common]
        unit += [f"--uid={os.getuid()}", f"--gid={os.getgid()}"]
        # the RT limit lets serve set FIFO itself
        unit += ["-p", f"LimitRTPRIO={RT_PRIORITY}", "-p", "OOMScoreAdjust=-900"]
        unit += ["-p", "ManagedOOMPreference=omit"]
        mode = "system unit: SCHED_FIFO, OOM-exempt, survives logout"
    elif args.no_systemd:
        unit, mode = [], "setsid child (unsupervised: dies with its session scope)"
    else:
        unit = ["systemd-run", "--user", *common]
        mode = (
            "user unit, not real-time: systemd-oomd may kill it under memory pressure, and "
            "logging out of every session stops it; give this user passwordless sudo for "
            "`systemd-run` to get the system unit"
        )
    if unit:
        subprocess.run([*unit, *cmd], check=True, stdout=subprocess.DEVNULL)
    else:
        child_env = dict(os.environ, HOMEBODY_ROOT=str(REPO))
        child_env.pop("HOMEBODY_WORKSPACE", None)
        with open(LOGFILE, "ab") as log:
            subprocess.Popen(
                cmd,
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                env=child_env,
            )
    systemctl = ["systemctl"] if use_sudo else ["systemctl", "--user"]
    try:
        for i in range(300):
            time.sleep(0.2)
            if host_alive() and running_pid():
                print(f"body host started (pid {running_pid()}) as a {mode}; log {LOGFILE}")
                return 0
            state = subprocess.run(
                [*systemctl, "is-active", UNIT], capture_output=True, text=True, check=False
            )
            if unit and i > 10 and state.stdout.strip() not in ("active", "activating"):
                break
    except KeyboardInterrupt:  # the unit keeps starting on its own; the next launch adopts it
        print("interrupted while the body host starts; it keeps starting — launch again")
        return 1
    tail = LOGFILE.read_text(errors="replace")[-1500:] if LOGFILE.exists() else ""
    print(
        f"body host did not come up: {LOGFILE} (and journalctl -u {UNIT}):\n{tail}", file=sys.stderr
    )
    return 1


# -- the client (inside the exploration process) ------------------------------------------------
class BodyClient:
    """The client side of the wire: sends intent, keeps the latest body state, counts ticks. Its
    claim carries its creation time: the host obeys the newest client and ignores older ones."""

    def __init__(self, addr: str = DEFAULT_BIND):
        host, port = parse_addr(addr)
        self.addr = (socket.gethostbyname(host), port)
        self.claim = f"{time.time():.6f}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1" if self.addr[0].startswith("127.") else "0.0.0.0", 0))
        self.sock.settimeout(0.2)
        self.state: dict | None = None
        self.state_at = 0.0
        self.tick = threading.Condition()
        self.ticks = 0
        self.closed = threading.Event()
        self._lock = threading.Lock()
        self._arms: list[float] | None = None
        self.interruptions = 0
        self._interrupted = False
        self._heartbeat_at = time.monotonic()
        self.last_interruption = ""
        self._seen_events: set = set()
        self.on_event = None
        self.on_state = None
        self.callback_error = ""
        self.callback_dropped = 0
        self._event_queue = queue.Queue(maxsize=64)
        self._threads = [threading.Thread(target=target, name=name, daemon=True)
                         for target, name in ((self._receive, "homebody-body-client-rx"),
                                              (self._heartbeat, "homebody-body-client-hb"),
                                              (self._events, "homebody-body-client-events"))]
        for worker in self._threads:
            worker.start()

    def send(self, **fields) -> None:
        fields["claim"], fields["hb"] = self.claim, time.time()
        with contextlib.suppress(OSError):
            self.sock.sendto(json.dumps(fields, separators=(",", ":")).encode(), self.addr)

    def set_arms(self, arm14) -> None:
        """Keep a validated arm reference alive through this client's existing heartbeat."""
        values = np.asarray(arm14, dtype=float)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("Arm heartbeat requires fourteen finite joint angles")
        self.require_fresh()
        with self._lock:
            self._arms = values.tolist()
            self.send(arms=self._arms)

    def _heartbeat(self) -> None:
        while not self.closed.is_set():
            self.check_link()
            with self._lock:
                self._heartbeat_at = time.monotonic()
                arms = self._arms
                self.send(arms=arms)
            self.closed.wait(0.02)

    def interrupt(self, reason: str) -> None:
        """Invalidate an in-flight motion and discard its cached arm target."""
        with self._lock:
            if self._interrupted:
                return
            self._interrupted = True
            self.interruptions += 1
            self.last_interruption = reason
            self._arms = None
        self.send(cmd="stand")

    def check_link(self) -> None:
        now = time.monotonic()
        if now - self._heartbeat_at > HEARTBEAT_LOST_S:
            self.interrupt(f"client heartbeat paused for {now - self._heartbeat_at:.3f} s")
        elif self.state is not None and now - self.state_at > STATE_STALE_S:
            self.interrupt(f"body state is {now - self.state_at:.3f} s old")

    def require_fresh(self) -> None:
        self.check_link()
        if not self.fresh():
            raise HardwareUnavailable("body state is stale; motion stopped while waiting for "
                                      "fresh state; replan the interrupted skill")

    def _receive(self) -> None:
        while not self.closed.is_set():
            try:
                data, sender = self.sock.recvfrom(65536)
                if sender != self.addr:
                    continue
                msg = json.loads(data)
                if not isinstance(msg, dict) or not isinstance(msg.get("tick"), int):
                    continue
                # Queued UDP packets are not new sensor readings after a process stall.
                stamp = float(msg["t"])
                age = time.time() - stamp
                if not math.isfinite(stamp) or age > STATE_STALE_S or age < -STATE_STALE_S:
                    continue
                if self.state is not None and stamp <= float(self.state["t"]):
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
            self.check_link()
            if self.state is not None and msg.get("instance") != self.state.get("instance"):
                self.interrupt("body host restarted")
            with self.tick:
                self.state, self.state_at = msg, time.monotonic() - max(age, 0.0)
                self.ticks += 1
                self.tick.notify_all()
            with self._lock:
                self._interrupted = False
            events = msg.get("events", [])
            for event in events if isinstance(events, list) else []:
                if (not isinstance(event, (list, tuple)) or len(event) != 2
                        or not isinstance(event[0], (int, float, str))
                        or not isinstance(event[1], str)):
                    continue
                stamp, text = event
                if (stamp, text) not in self._seen_events:
                    self._seen_events.add((stamp, text))
                    try:
                        self._event_queue.put_nowait(text)
                    except queue.Full:
                        self.callback_dropped += 1
            if self.on_state is not None:
                try:
                    self.on_state(msg)
                except Exception as exc:  # noqa: BLE001 -- callbacks never block receipt
                    self.callback_error = f"state callback: {type(exc).__name__}: {exc}"

    def _events(self) -> None:
        """Run the event callbacks off the receive thread."""
        while not self.closed.is_set():
            try:
                text = self._event_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.on_event is not None:
                try:
                    self.on_event(text)
                except Exception as exc:  # noqa: BLE001 -- callbacks never block receipt
                    self.callback_error = f"event callback: {type(exc).__name__}: {exc}"

    def fresh(self) -> bool:
        return (not self.closed.is_set() and self.state is not None
                and time.monotonic() - self.state_at <= STATE_STALE_S)

    def wait_fresh(self, timeout_s: float) -> bool:
        give_up = time.monotonic() + timeout_s
        while time.monotonic() < give_up:
            if self.fresh():
                return True
            time.sleep(0.05)
        return self.fresh()

    def close(self, *, hold: bool = True) -> None:
        """Detach and join owned readers; motion clients send stand, display listeners do not."""
        with self._lock:
            if not self.closed.is_set():
                self._arms = None
                if hold:
                    self.send(cmd="stand")
                self.closed.set()
                self.sock.close()
        workers = [worker for worker in self._threads if worker is not threading.current_thread()]
        for worker in workers:
            worker.join(timeout=1.0)
        if any(worker.is_alive() for worker in workers):
            raise RuntimeError("body client worker did not stop")


class HostPort:
    """The read side of a RobotPort, served from the host's published state."""

    def __init__(self, client: BodyClient):
        self.client = client
        self._cached: tuple[tuple[str | None, int], object] | None = None

    def read_state(self):
        from holosoma_inference.g1_control.contracts import G1State

        self.client.require_fresh()
        msg = self.client.state
        sample = (msg.get("instance"), msg["tick"])
        if self._cached is not None and self._cached[0] == sample:
            return self._cached[1]
        state = G1State(
            base_position=msg["base"],
            quaternion_wxyz=msg["quat"],
            joint_positions=msg["q"],
            base_linear_velocity=msg.get("vlin") or (0.0, 0.0, 0.0),
            base_angular_velocity=msg["gyro"],
            joint_velocities=msg["dq"],
        )
        self._cached = (sample, state)
        return state

    def write_command(self, command) -> None:
        raise RuntimeError("the body host owns rt/lowcmd: callers send intent over the client")

    def close(self) -> None:
        self.client.close()


class RemoteStatus:
    def __init__(self, msg: dict):
        msg = msg or {}
        self.active = bool(msg.get("active"))
        self.initializing = bool(msg.get("initializing"))
        self.walk_enabled = bool(msg.get("walk"))
        self.initialization_tick = int(msg.get("init_tick") or 0)
        self.stopped_by_watchdog = bool(msg.get("watchdog"))
        self.last_error = msg.get("error")
        self.last_event = str(msg.get("event") or "")


class RemoteKernel:
    """The kernel's surface RealBackend uses, forwarded to the host."""

    def __init__(self, client: BodyClient):
        self.client = client
        self.profile = g1.PROFILE

    @property
    def status(self) -> RemoteStatus:
        return RemoteStatus(self.client.state)

    @property
    def ramping(self) -> bool:
        return bool((self.client.state or {}).get("ramping"))

    def handle_command(self, command) -> None:
        name = getattr(command, "value", str(command))
        if name == "toggle_walk":
            self.client.require_fresh()
            self.client.send(walk=not self.status.walk_enabled)
        elif name == "zero_velocity":
            self.client.send(vel=[0.0, 0.0, 0.0])
        elif name == "stop":
            self.client.send(cmd="stand")  # a console stop is a stand; only the remote/sit stops
        else:
            raise ValueError(f"the body host does not take {name!r} from the client")

    def set_velocity(self, velocity) -> None:
        vx, vy = (float(v) for v in velocity.linear_xy)
        if vx or vy or velocity.angular_z:
            self.client.require_fresh()
        self.client.send(vel=[vx, vy, float(velocity.angular_z)])

    def set_torso(self, torso) -> None:
        self.client.require_fresh()
        self.client.send(
            torso=[float(torso.pitch_rad), float(torso.roll_rad), float(torso.height_m)]
        )


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m homebody.backends.g1.body_host")
    ap.add_argument("--settings-json", default="",
                    help="the run's meta/settings.json; the defaults in config.py otherwise")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "start"):
        p = sub.add_parser(name)
        p.add_argument("--bind", default=None)
        p.add_argument("--iface", default=None)
        p.add_argument("--cores", default=None)
        p.add_argument("--no-systemd", action="store_true")
        p.set_defaults(simulated=False)
        p.add_argument("--log", default="")
    for name in ("status", "sit", "stop", "deploy", "halt"):
        p = sub.add_parser(name)
        p.add_argument("--bind", default=None)
    args = ap.parse_args()
    if args.settings_json:
        install(from_json(Path(args.settings_json).read_text()))
    settings = current()
    args.bind = args.bind or settings.body.address or DEFAULT_BIND
    if args.cmd in ("serve", "start"):
        args.iface = args.iface if args.iface is not None else settings.robot.iface.strip()
        args.cores = args.cores if args.cores is not None else settings.body.cores
        args.use_sudo = settings.body.use_sudo
    if args.cmd == "serve":
        return serve(args)
    if args.cmd == "start":
        return start(args)
    if args.cmd == "halt":
        return halt(args)
    pid = running_pid()
    if args.cmd == "stop":
        if not pid:
            print("body host not running")
            return 0
        os.kill(pid, signal.SIGUSR1)
        print(f"asked pid {pid} to stop (refused while the policy is active: sit first)")
        return 0
    client = BodyClient(args.bind)
    if not client.wait_fresh(2.0):
        print(f"body host not answering on {args.bind} (lease held: {host_alive()}, pid {pid})")
        return 1
    if args.cmd == "sit":
        client.send(cmd="sit")
        time.sleep(0.2)
        client.send(cmd="sit")
        time.sleep(0.3)
    elif args.cmd == "deploy":  # the simulation's A (a real host takes A only from the remote)
        client.send(cmd="start")
        time.sleep(0.5)
    s = client.state or {}
    keys = ("armed", "driver", "active", "initializing", "walk", "loop_hz", "scheduling")
    keys += ("remote", "stiffness", "counts", "error", "watchdog")
    print(json.dumps({k: s.get(k) for k in keys}), flush=True)
    for _stamp, text in s.get("events", []):
        print("  ", text)
    client.closed.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

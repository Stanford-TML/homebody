"""Actual host lifecycle with inert transport and kernel, without hardware or services."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from holosoma_inference.g1_control.contracts import G1State, VelocityCommand

from homebody.backends.g1 import body_host


class Clock:
    value = 100.
    def __call__(self):
        return self.value


class Port:
    stiffness_level = 1.
    def __init__(self, clock):
        self.clock, self.sequence, self.moving, self.failing = clock, 0, True, False
        self.writes = []
    def read_state(self):
        if self.failing:
            raise RuntimeError('injected lowstate failure')
        if self.moving:
            self.sequence += 1
        q = (.0001 * self.sequence,) + (0.,) * 28
        return G1State((0., 0., 0.), (1., 0., 0., 0.), q, (0.,) * 3, (0.,) * 3, (0.,) * 29)
    def write_command(self, command):
        self.writes.append({'time': self.clock(), 'command': command})


class Socket:
    def __init__(self):
        self.inbox, self.outbox = [], []
        self.callback = None
    def recvfrom(self, _size):
        if not self.inbox:
            raise BlockingIOError
        return json.dumps(self.inbox.pop(0)).encode(), ('127.0.0.1', 32100)
    def sendto(self, data, addr):
        row = json.loads(data)
        self.outbox.append(row)
        if self.callback:
            self.callback(row)
    def queue(self, **message):
        self.inbox.append(message)


class Kernel:
    """Inert collaborator: records host commands; does not emulate control/physics."""
    def __init__(self, profile, port, _policy):
        self.profile, self.port = profile, port
        self.status = SimpleNamespace(active=False, initializing=False, walk_enabled=False,
                                      stopped_by_watchdog=False, initialization_tick=0, last_error=None)
        self.damped = False
        self.ramping = False
        self.velocity = VelocityCommand()
        self.commands = []
    @property
    def limp(self):
        return self.damped and not self.status.active and not self.status.initializing
    def handle_command(self, command):
        name = command.value
        self.commands.append({'command': name, 'time': self.port.clock()})
        if name == 'stop':
            self.status.active = self.status.initializing = self.status.walk_enabled = False
            self.damped = True
        elif name == 'toggle_walk':
            self.status.walk_enabled = not self.status.walk_enabled
        elif name == 'start':
            self.status.active, self.damped = True, False
    def set_arm_targets(self, values):
        self.arms = values
    def set_arm_gains(self, *_values):
        pass
    def set_velocity(self, velocity):
        self.velocity = velocity
    def set_torso(self, torso):
        self._torso = torso
    def limp_command(self, _state):
        return 'damped'
    def tick(self):
        command = 'damped' if self.limp else 'active'
        self.port.write_command(command)
        return SimpleNamespace(command=command)
    def republish_arms(self, command):
        if command is not None:
            self.port.write_command(command)


def make_host():
    clock = Clock()
    port, sock = Port(clock), Socket()
    with patch.object(body_host, 'load_policy', return_value=object()), \
         patch.object(body_host, '_limit_cpu_threads'), \
         patch.object(body_host, 'host_kernel_type', return_value=Kernel):
        host = body_host.BodyHost(port, sock=sock, clock=clock)
    return host, clock, port, sock


def step(host, clock, hz, *, due=True):
    clock.value += 1 / hz
    host.tick(1 / hz, due)


def running(host):
    host.armed = True
    host.kernel.status.active = True
    host.kernel.damped = False
    host.last_command = 'active'


def auto_arm(hz):
    host, clock, port, _ = make_host()
    limit = body_host.AUTO_ARM_TICKS
    port.moving = False
    for _ in range(limit + 2):
        step(host, clock, hz)
    assert not host.armed, 'Frozen initial lowstate must not auto-arm'
    began = clock()
    port.moving = True
    for _ in range(limit - 1):
        step(host, clock, hz)
        assert not host.armed
    step(host, clock, hz)
    assert host.armed and host.kernel.damped
    assert not port.writes, 'Auto-arm tick must not publish active motor command'
    return clock() - began


def link_refusal(hz):
    host, clock, port, _ = make_host()
    host.armed = True
    step(host, clock, hz)
    port.moving = False
    began = clock()
    for _ in range(body_host.LINK_STALE_TICKS - 1):
        step(host, clock, hz)
        assert host.link_live
    step(host, clock, hz)
    assert not host.link_live
    host.deploy('start', 'offline lifecycle probe')
    assert not host.kernel.status.active
    assert not any(row['command'] == 'start' for row in host.kernel.commands)
    assert 'no live robot state' in host.last_event
    return clock() - began


def read_failure(hz):
    host, clock, port, _ = make_host()
    running(host)
    port.failing = True
    began = clock()
    for _ in range(body_host.READ_FAILS_STOP - 1):
        step(host, clock, hz)
        assert host.kernel.status.active
        assert port.writes[-1]['command'] == 'active'
    step(host, clock, hz)
    assert not host.kernel.status.active and host.deploy_needs_release
    assert port.writes[-1]['command'] == 'damped'
    stops = [row for row in host.kernel.commands if row['command'] == 'stop']
    assert len(stops) == 1
    assert stops[0]['time'] == port.writes[-1]['time']
    return clock() - began


def intent(hz, command):
    host, clock, port, sock = make_host()
    running(host)
    host.kernel.status.walk_enabled = True
    host.kernel.velocity = VelocityCommand((.1, 0.), .1)
    host.driver = '100:probe'
    host.driver_heard = clock()
    host.driver_lost = False
    began = clock()
    sock.queue(claim=host.driver, cmd=command)
    step(host, clock, hz)
    if command == 'stand':
        assert host.kernel.status.active and not host.kernel.status.walk_enabled
        assert host.kernel.velocity == VelocityCommand()
        assert sock.outbox[-1]['walk'] is False
        assert port.writes[-1]['command'] == 'active'
    else:
        assert not host.kernel.status.active and host.kernel.limp
        assert port.writes[-1]['command'] == 'damped'
    assert not sock.inbox
    return clock() - began


@pytest.mark.parametrize("probe,expected", [(auto_arm, .1), (link_refusal, .1), (read_failure, .2)])
def test_host_fault_windows_have_fixed_duration(probe, expected):
    elapsed = probe(50)
    assert elapsed == pytest.approx(expected)


@pytest.mark.parametrize("hz", [250, 50])
@pytest.mark.parametrize("command", ["stand", "halt"])
def test_host_applies_stop_intent_within_one_cycle(hz, command):
    elapsed = intent(hz, command)
    assert elapsed == pytest.approx(1 / hz)

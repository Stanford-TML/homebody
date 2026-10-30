"""Client timing with synthetic arrival times and no motor writes."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from homebody.backends.g1 import real


class ArrivalClock:
    def __init__(self, packets, duration, *, delays=()):
        self.now = 0.
        self.packets = list(packets)
        self.duration = duration
        self.delays = dict(delays)
        self.waits = 0
        self.client = None
        self.stop = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def wait_for(self, predicate, timeout):
        self.waits += 1
        assert self.waits < 2000, 'Client loop busy-spun'
        if predicate():
            return True
        arrival = self.packets[0] if self.packets else float('inf')
        self.now = min(arrival, self.now + timeout, self.duration)
        self.now += self.delays.get(self.waits, 0.)
        # A delayed receiver can coalesce several host publications into its latest state.
        while self.packets and self.packets[0] <= self.now + 1e-10:
            self.client.state_at = self.packets.pop(0)
            self.client.ticks += 1
            self.client.state = {'loop_hz': 50., 'tick': self.client.ticks}
        if self.now >= self.duration - 1e-10:
            self.stop.set()
        return predicate()


def run_client(monkeypatch, packets, duration, *, delays=(), hands=True, fail_hand=False,
               lock_delay=0.):
    clock = ArrivalClock(packets, duration, delays=delays)
    backend = real.RealBackend.__new__(real.RealBackend)
    client = SimpleNamespace(ticks=0, tick=clock, state_at=0., state={'loop_hz': 50.},
                             interrupt=Mock(), send=Mock())
    client.fresh = lambda: clock.now - client.state_at <= real.STATE_STALE_S
    clock.client = client
    backend._host = client
    backend._control_stop = clock.stop
    backend._diagnostics_dir = None
    backend._control_lock = threading.RLock()
    backend._control_tick = threading.Condition(backend._control_lock)
    if lock_delay:
        lock = backend._control_lock

        class DelayedLock:
            delayed = False

            def __enter__(self):
                lock.acquire()
                if not self.delayed:
                    clock.now += lock_delay
                    self.delayed = True

            def __exit__(self, *_args):
                lock.release()

        backend._control_lock = DelayedLock()
    backend._control_ticks = 0
    backend._control_error = None
    backend._epoch = 1
    backend._lower_deadman = Mock()
    backend._note_body_flip = Mock()
    backend._kernel = SimpleNamespace(status=object())
    backend._lowpass_tilt = lambda _quat: (0., 0.)
    backend._term_stamp = {}
    backend.trace_t, backend.trace_q = [], []
    backend.now = lambda: clock.now
    hand_calls, body_reads = [], []

    def hand_tick(period):
        hand_calls.append((clock.now, period))
        if fail_hand:
            raise RuntimeError('injected hand fault')

    def read_state():
        body_reads.append((clock.now, client.ticks))
        return SimpleNamespace(quaternion_wxyz=(1., 0., 0., 0.),
                               joint_positions=np.full(29, client.ticks))

    backend._dex3 = SimpleNamespace(tick=hand_tick) if hands else None
    backend._port = SimpleNamespace(read_state=read_state)
    watch = Mock()
    monkeypatch.setattr(real.stall_watch, 'StallWatch', lambda _path: watch)
    monkeypatch.setattr(real, 'time', SimpleNamespace(monotonic=lambda: clock.now,
                                                     time=lambda: clock.now))
    backend._host_loop()
    return backend, client, clock, hand_calls, body_reads


def test_stale_body_stops_hand_updates_and_preserves_recovery_deadline(monkeypatch):
    backend, client, _, hands, _ = run_client(monkeypatch, [.02], 3.2)
    assert isinstance(backend._control_error, real.RealHardwareRequired)
    assert backend._epoch == 2
    assert max(when for when, _ in hands) <= .52 + 1e-10
    assert 3.02 <= backend.now() <= 3.024 + 1e-10
    assert client.interrupt.call_count > 0
    client.send.assert_called_once_with(cmd='stand')


@pytest.mark.parametrize('hands', [True, False])
def test_explicit_stop_prevents_further_hand_and_body_writes(monkeypatch, hands):
    backend, _, clock, calls, reads = run_client(monkeypatch, [.02, .04], .02, hands=hands)
    assert backend._control_error is None and clock.stop.is_set()
    assert all(when < .02 for when, _ in calls)
    assert reads == []

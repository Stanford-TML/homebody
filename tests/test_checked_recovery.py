"""Recovery must emit real checked commands despite slow preparation, and stop on new risk."""
import math
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from homebody.backends.g1.real import RealBackend
from homebody.motion.exploration import gait
from homebody.motion.exploration.localizer import PoseGate
from homebody.primitives.navigation.context import Ctx, Deadline
from homebody.primitives.navigation.errors import ActionRefused, Stopped
from homebody.real2sim.exploration import explore_backend
from homebody.skills.navigate_real import navigation


@pytest.fixture
def moving_rig(monkeypatch):
    state = SimpleNamespace(t=100., pose=np.array([.24, 0., 0.]), epoch=1, stopped=False,
                            commands=[], events=[], integrate=True, sample_delay=0.,
                            on_log=lambda *_: None, on_step=lambda *_: None)
    clock = SimpleNamespace(time=lambda: state.t, monotonic=lambda: state.t)
    monkeypatch.setattr(explore_backend, 'time', clock)
    occupied = np.zeros((101, 101), bool)
    occupied[:, 50] = True
    art = SimpleNamespace(occupancy=occupied, center_allowed=np.ones_like(occupied),
                          T_map_px=np.array([[.05, 0., -2.5], [0., .05, -2.5], [0., 0., 1.]]))
    checker, _ = gait.occupancy_grid(art, .25, exact_footprint=True, obstacles=occupied)
    live = SimpleNamespace(scope_ready=True, task_region_revision=0)
    backend = object.__new__(explore_backend.ExploreRealBackend)
    source = SimpleNamespace(
        motion_epoch=lambda: state.epoch, frame_reset=False, observe_only=False,
        require_actuation=lambda: None, stop_flag=SimpleNamespace(is_set=lambda: state.stopped),
        ring_t=state.t, note_twist=lambda *_: None,
        localization=SimpleNamespace(lock=threading.Lock(), _last_points=np.array([[0., 0., .5]])))
    backend._explore_source = source
    backend.explorer = SimpleNamespace(live=live)
    backend._execution_cache = ((id(live), 1, 0), checker, None)
    backend.observation_refusal = lambda **_: None

    def guard(epoch):
        if state.stopped or state.epoch != epoch:
            raise Stopped('test authority ended')

    def observe(term):
        if term == 'gripper_state':
            return SimpleNamespace(held_l=None, held_r=None)
        if term == 'map':
            return art
        if term == 'loc_quality':
            return 1.
        assert term == 'base_pose'
        backend._execution_base_pose_sample = (
            state.epoch, state.t, state.t - state.sample_delay, tuple(state.pose))
        return state.pose.copy()

    observe._owns_pose_consistency = True
    observe.recovery_generation = 0

    def transport(_backend, command):
        state.commands.append(command)
        vx, vy, wz = command.vx or 0., command.vy or 0., command.wz or 0.
        if state.integrate:
            yaw = state.pose[2]
            state.pose += .004 * np.array([
                math.cos(yaw)*vx-math.sin(yaw)*vy,
                math.sin(yaw)*vx+math.cos(yaw)*vy, wz])
        state.t += .004
        source.ring_t = state.t
        state.on_step(command)

    monkeypatch.setattr(RealBackend, 'apply_lower_body', transport)

    def step(channel, command):
        assert channel == 'lower_body'
        if any((command.vx, command.vy, command.wz)):
            guard(1)
        backend.apply_lower_body(command)

    def log(kind, payload):
        state.events.append((kind, payload))
        state.on_log(kind, payload)

    ctx = Ctx(observe, step, lambda: state.t, Deadline(lambda: state.t, 30.),
              lambda _: None, log, PoseGate())
    port = navigation.NavigationPort(
        navigation.NavigationCapabilities(.25, .30, (.33,), .5, 2.),
        epoch=lambda: state.epoch, guard=guard, fresh=lambda *_a, **_k: guard(1),
        hold=lambda c: gait.hold(c, .2, 0., 0., 0.), stage=lambda *_: None,
        planning_radius=lambda r: r, recovery_map=lambda: (art, occupied, 1),
        map_age=lambda: .01, stationary_guard=lambda *_: 1,
        motion_refusal=backend.motion_refusal, stopped=lambda: state.stopped,
        require_ready=lambda: None, can_refresh_observations=True)
    return state, art, ctx, navigation.Navigator(port), backend


def moving(state):
    return [c for c in state.commands if any((c.vx, c.vy, c.wz))]


@pytest.mark.parametrize('yaw', [0., math.pi/2, -math.pi/2, math.pi])
def test_escape_moves_out_of_overlap_with_execution_checks(moving_rig, yaw):
    state, art, ctx, nav, _ = moving_rig
    state.pose[2] = yaw
    checker, _ = gait.occupancy_grid(art, .30, exact_footprint=True)
    assert not checker.is_safe(*state.pose[:2])
    result = nav._checked_retreat(ctx, .30, travel_heading=0., target_distance=.4)
    assert result['outcome'] == 'complete' and result['padding_escape']
    assert moving(state) and state.pose[0] >= .64
    assert checker.is_safe(*state.pose[:2])
    assert all(c.wz == 0. for c in moving(state))
    assert not any((state.commands[-1].vx, state.commands[-1].vy, state.commands[-1].wz))


@pytest.mark.parametrize('change', ['pose', 'yaw', 'epoch', 'generation', 'stop', 'lidar'])
def test_rechecks_after_preparation_and_never_moves_on_new_risk(moving_rig, change):
    state, _, ctx, nav, backend = moving_rig

    def changed(kind, _):
        if kind != 'recovery.corridor':
            return
        if change == 'pose':
            state.pose[0] = .10  # Worse penetration since the preceding sample.
        elif change == 'yaw':
            state.pose[2] = math.radians(11)
        elif change == 'epoch':
            state.epoch = 2
        elif change == 'generation':
            ctx.observe.recovery_generation += 1
        elif change == 'stop':
            state.stopped = True
        else:
            backend._explore_source.localization._last_points = np.array([[.50, 0., .5]])
            backend._explore_source.ring_t += .001

    state.on_log = changed
    with pytest.raises((ActionRefused, Stopped)):
        nav._checked_retreat(ctx, .30, travel_heading=0., target_distance=.4)
    assert not moving(state)
    assert not any(k in {'recovery.complete', 'recovery.partial'} for k, _ in state.events)


def test_stale_sensor_sample_is_not_repaired_by_reacquisition(moving_rig):
    state, _, ctx, nav, _ = moving_rig
    state.sample_delay = .6
    with pytest.raises(ActionRefused) as caught:
        nav._checked_retreat(ctx, .30, travel_heading=0., target_distance=.4)
    assert caught.value.code == 'OBSERVATIONS_UNAVAILABLE'
    assert not moving(state)


@pytest.mark.parametrize('hazard', ['new_obstacle', 'unknown', 'boundary'])
def test_egress_does_not_bypass_other_obstructions(moving_rig, hazard):
    state, art, ctx, nav, _ = moving_rig
    if hazard == 'new_obstacle':
        art.occupancy[:, 59] = True
    elif hazard == 'unknown':
        obstacles = art.occupancy.copy()
        art.occupancy[:, 50] = False
        art.occupancy[:, 55] = True
        nav.port = navigation.replace(nav.port, recovery_map=lambda: (art, obstacles, 2))
    else:
        art.center_allowed[:, 57:] = False
    with pytest.raises(ActionRefused):
        nav._checked_retreat(ctx, .30, travel_heading=0., target_distance=.4)
    assert not moving(state)


def test_no_motion_is_not_reported_as_recovery_success(moving_rig):
    state, _, ctx, nav, _ = moving_rig
    state.integrate = False
    with pytest.raises(ActionRefused, match='no progress'):
        nav._checked_retreat(ctx, .30, travel_heading=0., target_distance=.4)
    assert moving(state)
    assert not any(k in {'recovery.complete', 'recovery.partial'} for k, _ in state.events)
    assert state.t < 105.


def test_registered_departure_recovery_moves_and_preserves_refused_destination(moving_rig):
    from homebody.real2sim.exploration.skills import REGISTRY
    from homebody.real2sim.exploration.sources import RobotSource
    from homebody.session.live_context import LiveContext
    from homebody.skills.contract import Runner

    state, _, ctx, nav, backend = moving_rig
    source = object.__new__(RobotSource)
    source.backend, source.turn_drift_m = backend, .5
    source.stop_flag = backend._explore_source.stop_flag
    source.motion_epoch, source._motion_guard = nav.port.epoch, nav.port.guard
    source._fresh_for_motion, source._hold_motion = nav.port.fresh, nav.port.hold
    source._motion_stage, source.planning_radius_m = nav.port.stage, nav.port.planning_radius
    source._recovery_map, source.stationary_guard = nav.port.recovery_map, nav.port.stationary_guard
    backend.explorer.map_age_s = nav.port.map_age
    source._resumable_motion = lambda _seconds, _operation, run: run(ctx)
    events = []
    world = SimpleNamespace(source=source, live=backend.explorer.live, task_region_generation=1,
        check=lambda: None, radius=.30, remaining=lambda: 30., refresh_planning_radius=dict,
        emit=lambda kind, **data: events.append({'type': kind, **data}))
    context = LiveContext(world, emit=events.append)
    runner = Runner(context, REGISTRY)
    record = {'n': 4, 'proposed_goal': [0., 0.], 'rejection': 'goal blocked'}
    result = runner.run('departure_recovery', {'record': record}, context.frame())

    assert result.ok and result.details['recovered']
    assert moving(state) and state.pose[0] > .5
    assert record == {'n': 4, 'proposed_goal': [0., 0.], 'rejection': 'goal blocked'}
    assert sum(e['type'] == 'skill_finished' for e in events) == 1
    recovery = next(e['result'] for e in events if e['type'] == 'recovery_result')
    assert recovery['outcome'] == 'repositioned_goal_unavailable'
    assert not any(k == 'drive.arrival_confirmed' for k, _ in state.events)


def nonzero_commands(state):
    return [c for c in state.commands if any((c.vx, c.vy, c.wz))]


@pytest.mark.parametrize('entry', ['recover', 'drive'])
def test_measured_sweep_failure_cannot_start_another_burst(request, entry):
    state, _art, ctx, navigator, _backend = request.getfixturevalue('moving_rig')
    original_guard = navigator.port.guard
    navigator.port = replace(navigator.port, guard=lambda epoch, **_: original_guard(epoch))
    if entry == 'drive':
        state.pose[2] = math.pi  # The driver's initial reverse points out of the wall.
    observed = {'nonzero': 0, 'failed_sweep': None, 'injected': False}

    def log(kind, payload):
        if (kind == 'recovery.corridor' and payload.get('sweep', {}).get('safe') is False
                and observed['injected']):
            observed['failed_sweep'] = payload

    def after_step(command):
        if not any((command.vx, command.vy, command.wz)):
            return
        if observed['failed_sweep'] is not None:
            pytest.fail('A new nonzero command followed an already-failed measured sweep')
        observed['nonzero'] += 1
        if observed['nonzero'] == 25:
            # Prospective outward commands passed the motion gate; the observed base
            # moves inward during the first burst.
            state.pose[0] -= .020
            observed['injected'] = True

    state.on_log, state.on_step = log, after_step
    with pytest.raises(ActionRefused) as caught:
        if entry == 'recover':
            navigator.recover(ctx, .30)
        else:
            navigator.drive(ctx, (.9, 0.), None, .30)
    assert observed['failed_sweep'] is not None
    assert observed['nonzero'] == 25
    # The outer action must not replace the measured failure with a generic
    # no-candidate result; later decisions need the causal witness.
    assert caught.value.details['recovery_check']['failed_stage'] == 'sweep'
    assert caught.value.details['recovery_check']['previous_xy'] != (
        caught.value.details['recovery_check']['xy'])
    assert not any(kind in {'recovery.complete', 'drive.arrival_confirmed'}
                   for kind, _ in state.events)


def test_actual_failure_persists_in_new_navigator_with_shared_state(request):
    state, _, ctx, nav, _ = request.getfixturevalue('moving_rig')
    moving_ticks = 0

    def perturb_after_prediction(command):
        nonlocal moving_ticks
        if any((command.vx, command.vy, command.wz)):
            moving_ticks += 1
            if moving_ticks == 25:
                state.pose[0] -= .020

    state.on_step = perturb_after_prediction
    with pytest.raises(ActionRefused) as first:
        nav._checked_retreat(ctx, .30, travel_heading=0., target_distance=.4)
    witness = first.value.details['recovery_check']
    assert witness['failed_stage'] == 'sweep'
    assert len(nav.state._escape_attempts) == 1
    state.on_step = lambda _: None
    moving_before = len(nonzero_commands(state))
    next_nav = navigation.Navigator(nav.port, nav.state)
    with pytest.raises(ActionRefused) as later:
        next_nav._checked_retreat(ctx, .30, travel_heading=0., target_distance=.4)
    assert later.value.details['recovery_check'] == witness
    assert len(nonzero_commands(state)) == moving_before == 25
    assert len(next_nav.state._escape_attempts) == 1

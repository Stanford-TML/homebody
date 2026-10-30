"""REAL display geometry is derived from frozen evidence, without backend access."""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from homebody.primitives.observations import Frame, HandState, NavigationMap
from homebody.ui import real_scene


class Handle(SimpleNamespace):
    def remove(self):
        self.removed = True


class Scene:
    def __init__(self):
        self.handles, self.calls = {}, []

    def __getattr__(self, method):
        def call(*args, **kwargs):
            self.calls.append((method, args, kwargs))
            if not method.startswith("add_"):
                return None
            handle = Handle(args=args, removed=False, visible=kwargs.pop("visible", True), **kwargs)
            self.handles[args[0]] = handle
            return handle
        return call


class Server:
    def __init__(self):
        self.scene = Scene()
        self.gui = SimpleNamespace(add_markdown=lambda text: Handle(content=text))

    def atomic(self):
        return nullcontext()

    def on_client_connect(self, callback):
        self.connected = callback

    def get_clients(self):
        return {}


def observation(sequence=1, *, epoch=1, position=(0., 0., 0.)):
    torso = np.eye(4)
    torso[:3, 3] = (*position[:2], 1.1)
    hands = {side: HandState(np.arange(7) * .01, .1, np.zeros(3), np.zeros(3))
             for side in ("left", "right")}
    return Frame(epoch, sequence, sequence * .2, np.full((4, 6, 3), 170, np.uint8),
                 np.ones((4, 6)), np.array([[2., 0, 2.], [0, 3., 1.], [0, 0, 1.]]),
                 torso, torso, np.asarray(position), np.arange(29) * .001, hands,
                 np.zeros((4, 6), np.int32),
                 NavigationMap(np.array([[True, False, False], [False, False, True]]),
                               np.array([[.2, 0., -1.], [0., -.2, 2.], [0., 0., 1.]]), .2,
                               source_hash="b" * 64, frame_id="measured-room"))


class Robot:
    def __init__(self, *args):
        self.frames, self.hidden = [], True

    def show(self, frame):
        self.frames.append(frame)
        self.hidden = False
        return True

    def hide(self):
        self.hidden = True


def view(monkeypatch):
    monkeypatch.setattr(real_scene, "MeasuredRobot", Robot)
    clock = [1.]
    server = Server()
    scene = real_scene.RealSceneView(server, Path("."), navigation=observation().navigation,
                                      floor_height_m=-1.3, clock=lambda: clock[0])
    return scene, server, clock


def test_frame_updates_measured_robot_depth_and_unsmoothed_travel_without_backend(monkeypatch):
    scene, server, clock = view(monkeypatch)
    first = observation()
    scene.show(first)
    assert scene._robot.frames == [first]
    assert scene._cloud is not None and not scene._robot.hidden
    old_map = scene._map
    clock[0] += .3
    second = observation(2, position=(.13, .07, .4))
    scene.show(second)
    assert scene._map is old_map
    np.testing.assert_array_equal(scene._trail, [[0, 0], [.13, .07]])
    np.testing.assert_allclose(server.scene.handles["/real/traveled"].points[0, :, :2], scene._trail)
    np.testing.assert_allclose(server.scene.handles["/real/traveled"].points[0, :, 2], -1.275)
    assert scene.home[1][2] == -1.3 and scene.home[0][2] > -1.3
    assert "No reconstruction mesh" in scene.note.content


def test_dynamic_obstacles_replace_map_but_mismatched_source_or_frame_hides_scene(monkeypatch):
    scene, _, clock = view(monkeypatch)
    frame = observation()
    scene.show(frame)
    old_map = scene._map
    occupied = frame.navigation.occupied.copy()
    occupied[1, 1] = True
    clock[0] += .3
    scene.show(replace(observation(2), navigation=replace(frame.navigation, occupied=occupied)))
    assert old_map.removed and scene._map is not old_map
    scene.show_route([[0, 0], [1, 1]])
    for field, value in (("source_hash", "a" * 64), ("frame_id", "another-room")):
        scene.show(replace(observation(3), navigation=replace(frame.navigation, **{field: value})))
        assert scene._robot.hidden and not scene._map.visible and not scene._cloud.visible
        assert not scene._route_handles and not scene._trail
        assert "does not match" in scene.note.content

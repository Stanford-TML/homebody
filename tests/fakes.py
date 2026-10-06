"""Shared test fakes: a planning arm with no collision geometry, and a console over a mocked
web server. Kept for tests of the skill framework, the console and the cockpit tool."""
from unittest.mock import MagicMock

import numpy as np
from scipy.spatial.transform import Rotation

from homebody.helpers.geometry import transform
from homebody.session.settings import Topdown
from homebody.skills.contract import SkillFailure
from homebody.ui.console import Console, Controls


class EmptyCollisionGeometry:
    def boxes(self, arm, joints):
        return []

    def hand_boxes(self, arm, joints, fingers):
        return []

    def torso(self):
        return np.array([20, 20, 20]), np.eye(3), np.ones(3)


class PlanningArm:
    lower = np.full(7, -10.)
    upper = np.full(7, 10.)
    shoulder = np.array([0., 0., 1.])
    collision = EmptyCollisionGeometry()
    collision.side = "right"

    def __init__(self):
        self.failure = None
        self.failure_raised = False
        self.solves = []

    def forward(self, joints):
        if self.failure:
            self.failure_raised = True
            raise SkillFailure(self.failure, "Injected approach observation failure")
        return (transform(np.array([0., 0., 1.]) + joints[:3],
                          Rotation.from_rotvec(np.array(joints[3:6])).as_matrix()),
                np.c_[np.eye(6), np.zeros(6)])

    def solve(self, target, seed, **kwargs):
        self.solves.append((target.copy(), seed.copy()))
        return np.r_[target[:3, 3] - [0, 0, 1], Rotation.from_matrix(target[:3, :3]).as_rotvec(), 0.]


class SceneRecorder:
    """Stands in for the 3D view and records what the console asked it to draw."""
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))

    def last(self, name):
        return next(args for called, args, _ in reversed(self.calls) if called == name)


def mocked_console(monkeypatch, settings=None):
    """A real Console over a mocked web server, with a recording 3D view."""
    import socket

    import viser
    server = MagicMock()
    for widget in ("add_image", "add_html", "add_markdown", "add_button", "add_text"):
        getattr(server.gui, widget).side_effect = lambda *args, kind=widget, **kwargs: MagicMock(kind=kind)
    monkeypatch.setattr(socket, "socket", MagicMock())
    monkeypatch.setattr(viser, "ViserServer", MagicMock(return_value=server))
    console = Console(Controls(), topdown_settings=settings or Topdown(pixels_per_metre=100.))
    console.scene_view = SceneRecorder()
    return console

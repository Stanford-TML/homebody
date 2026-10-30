"""Remember and recall run through the production Runner with no actuator command."""

import numpy as np
from test_real_memory import binding, frame

from homebody.memory import NamedMemory
from homebody.skills import recall, remember
from homebody.skills.contract import Context, Runner, declared


class PassiveRobot:
    def __init__(self):
        self.epoch, self.time = 2, 1.
        self.pose, self.value, self.sequence = (0., 0., 0.), 42, 1
        self.commands = []

    def snapshot(self):
        observation = frame(self.pose, value=self.value, sequence=self.sequence)
        observation.epoch = self.epoch
        observation.frame_id = (self.epoch, self.sequence)
        observation.time = self.time
        return observation

    def base_velocity(self, *args):
        self.commands.append(("velocity", args))

    def stop(self, *args):
        self.commands.append(("stop", args))


def compose(tmp_path, *, memory=True):
    robot, events = PassiveRobot(), []
    store = NamedMemory.open(tmp_path, "views", binding()) if memory else None
    ctx = Context(robot, robot, {}, memory=store, emit=events.append)
    registry = {"remember": declared("remember", remember), "recall": declared("recall", recall)}
    return robot, store, Runner(ctx, registry), events


def run(runner, robot, name, arguments):
    return runner.run(name, arguments, robot.snapshot())


def test_registered_memory_skills_capture_and_recall_without_commanding_even_a_hold(tmp_path):
    robot, store, runner, events = compose(tmp_path)
    saved = run(runner, robot, "remember", {"content": "work surface"})
    result = run(runner, robot, "recall", {})
    assert saved.ok and result.ok and not result.details["absent"]
    hit = result.details["hits"][0]
    assert hit["record_id"] == saved.details["record_id"]
    np.testing.assert_array_equal(store.load_image(hit), robot.snapshot().rgb)
    assert len([event for event in events if event["type"] == "skill_finished"]) == 2
    assert robot.commands == []

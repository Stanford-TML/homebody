"""Compose a persistent task session from robot capabilities and a bound map."""
import json
import math
from pathlib import Path

from homebody.memory import NamedMemory
from homebody.skills.manipulation_policy import REAL_MANIPULATION
from homebody.skills.navigate_real import task as navigate_real
from homebody.skills.registry import task_registry

from .session import Session


def compose(backend, provider, settings, recorder, controls, *, root, memory_root,
            memory_name, binding, navigation, ui=None, arms=None, jaw_samples=None,
            **capabilities):
    root = Path(root)
    if arms is None:
        from homebody.helpers.kinematics import Arm
        arms = {side: Arm(root / "assets/robot/g1_29dof_with_hand.urdf", side)
                for side in ("left", "right")}
    if jaw_samples is None:
        jaw_samples = json.loads((root / "assets/robot/calibration.json").read_text())["jaw_samples"]
    diagonal = math.hypot(*navigation.occupied.shape) * navigation.resolution
    memory = NamedMemory.open(memory_root, memory_name, binding, map_diagonal_m=diagonal)
    if ui is not None:
        ui.arms = arms
    return Session(backend, provider, settings, recorder, controls, arms=arms,
                   jaw_samples=jaw_samples, ui=ui, memory=memory,
                   registry=task_registry(memory=True),
                   navigation=navigate_real, manipulation=REAL_MANIPULATION,
                   evaluator=None, persistent=True, **capabilities)

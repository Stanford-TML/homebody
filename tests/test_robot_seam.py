"""Skills reach a robot only through primitives.Robot; the simulator is one implementation."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from homebody.backends.mujoco import MujocoBackend
from homebody.primitives.robot import Robot

ROOT = Path(__file__).resolve().parents[1]


def test_every_registered_skill_loads_without_the_simulator():
    code = ("import sys, homebody.skills.registry\n"
            "loaded = [name for name in sys.modules if name.split('.')[0] == 'mujoco'"
            " or name.startswith('homebody.backends')]\n"
            "assert not loaded, loaded\n")
    source = os.pathsep.join(str(ROOT / part) for part in ("src", "vendor"))
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            env=dict(os.environ, PYTHONPATH=source),
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr


def test_the_simulator_implements_the_robot_protocol():
    """Checked without a scene: an instance that skipped __init__, given the epoch and the
    clock its properties read (Python 3.11 evaluates protocol properties in isinstance)."""
    backend = MujocoBackend.__new__(MujocoBackend)
    backend.epoch, backend.data = 0, SimpleNamespace(time=0.0)
    assert isinstance(backend, Robot)

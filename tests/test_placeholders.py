"""The REAL backend satisfies the public interface and requires explicit deployment."""

import pytest

from homebody.backends.real import RealG1Backend
from homebody.primitives.robot import Robot


def test_real_backend_declares_robot_contract_without_implicit_deployment():
    for name in [*(n for n in dir(Robot) if not n.startswith("_")), "reset", "close"]:
        assert hasattr(RealG1Backend, name), name
    # A bare constructor must never discover hardware or infer calibration.
    with pytest.raises(TypeError, match="config"):
        RealG1Backend(None, None)

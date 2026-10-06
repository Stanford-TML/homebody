"""The real-G1 backend and the real2sim pipeline are declared, not implemented: each says so."""
import inspect

import pytest

from homebody import real2sim
from homebody.backends.real import RealG1Backend
from homebody.primitives.release import NotReleased
from homebody.primitives.robot import Robot


def test_the_real_backend_declares_every_member_a_backend_needs_and_raises_on_each():
    for name in [*(n for n in dir(Robot) if not n.startswith("_")), "reset", "close"]:
        assert hasattr(RealG1Backend, name), name
    with pytest.raises(NotReleased, match="real-G1 backend is not part of this release"):
        RealG1Backend(None, None)
    backend = object.__new__(RealG1Backend)
    for name, member in inspect.getmembers(RealG1Backend):
        if isinstance(member, property):
            with pytest.raises(NotReleased):
                getattr(backend, name)
        elif inspect.isfunction(member) and not name.startswith("_"):
            arguments = [None] * (len(inspect.signature(member).parameters) - 1)
            with pytest.raises(NotReleased):
                member(backend, *arguments)


@pytest.mark.parametrize("stage", [real2sim.explore, real2sim.capture, real2sim.reconstruct, real2sim.package, real2sim.validate])
def test_each_real2sim_stage_raises_not_released(stage):
    with pytest.raises(NotReleased, match="real2sim pipeline is not part of this release"):
        stage(*[None] * len(inspect.signature(stage).parameters))

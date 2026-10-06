"""Real2sim, from a room to a scene package: not released. Each stage raises NotReleased."""
from homebody.primitives.release import NotReleased

WHAT = "The real2sim pipeline"


def explore(robot):
    """Not released."""
    raise NotReleased(WHAT)


def capture(recording):
    """Not released."""
    raise NotReleased(WHAT)


def reconstruct(captured):
    """Not released."""
    raise NotReleased(WHAT)


def package(reconstruction, scene):
    """Not released."""
    raise NotReleased(WHAT)


def validate(scene):
    """Not released."""
    raise NotReleased(WHAT)

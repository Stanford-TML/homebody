"""The registered skills by name. Naming a module `skills/<name>.py` in SKILLS registers it;
MODULES maps `navigate` to `skills/navigate_sim.py`."""
from collections.abc import Mapping
from importlib import import_module

from .contract import declared

SKILLS = ("navigate", "pick", "place")
MODULES = {"navigate": "navigate_sim"}


class Prompts(Mapping):
    """Each registered skill's PROMPT, read on access."""

    def __getitem__(self, name):
        return REGISTRY[name].PROMPT

    def __iter__(self):
        return iter(REGISTRY)

    def __len__(self):
        return len(REGISTRY)


REGISTRY = {name: declared(name, import_module(f".{MODULES.get(name, name)}", __package__))
            for name in SKILLS}
PROMPTS = Prompts()


def task_registry(*, memory=False):
    """Registered task skills with checked navigation and optional photo memory."""
    # loaded with the task policy, not with the SIM registry
    from .navigate_real import task as navigate_real

    registry = dict(REGISTRY, navigate=declared("navigate", navigate_real))
    if memory:
        for name in ("remember", "recall"):
            registry[name] = declared(name, import_module(f".{name}", __package__))
    return registry

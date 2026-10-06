"""The registered skills by name. Naming a module `skills/<name>.py` in SKILLS registers it."""
from collections.abc import Mapping
from importlib import import_module

from .contract import declared

SKILLS = ("navigate", "pick", "place")


class Prompts(Mapping):
    """Each registered skill's PROMPT, read on access."""

    def __getitem__(self, name):
        return REGISTRY[name].PROMPT

    def __iter__(self):
        return iter(REGISTRY)

    def __len__(self):
        return len(REGISTRY)


REGISTRY = {name: declared(name, import_module(f".{name}", __package__)) for name in SKILLS}
PROMPTS = Prompts()

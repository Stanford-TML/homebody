"""The decision providers by name. A provider class names itself (LABEL), what it needs (EXECUTABLE or KEY) and, for a CLI, its effort."""
from homebody.session.settings import Agent

from .api import ClaudeApi, OpenAiApi
from .claude import ClaudeCode
from .codex import Codex

PROVIDERS = {"codex": Codex, "claude-code": ClaudeCode, "openai-api": OpenAiApi, "claude-api": ClaudeApi}


def make(settings: Agent):
    return PROVIDERS[settings.provider](settings)


def requirement(provider: str) -> tuple[str | None, str | None]:
    """The executable a CLI provider runs, or the environment variable an API provider reads."""
    return getattr(PROVIDERS[provider], "EXECUTABLE", None), getattr(PROVIDERS[provider], "KEY", None)


def effort(settings: Agent) -> str | None:
    """The reasoning effort a CLI is given, None for the APIs."""
    level = getattr(PROVIDERS[settings.provider], "effort", None)
    return level(settings) if level else None


def model_label(settings: Agent) -> str:
    """The model as the console names it: its tool or API, its model id and effort."""
    level = effort(settings)
    return f"{PROVIDERS[settings.provider].LABEL} ({settings.model_id}{f', {level} effort' if level else ''})"

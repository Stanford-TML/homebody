"""One Claude Code decision, using the operator's login. Its only tool is Read on this turn's images."""
import json
import shutil
from collections.abc import Callable
from pathlib import Path

from homebody.session.settings import Agent

from .provider import Decision, ProviderError, attachments, run_cli, schema


class ClaudeCode:
    LABEL, EXECUTABLE = "Claude", "claude"

    def __init__(self, settings: Agent, executable: str = EXECUTABLE):
        self.settings = settings
        self.executable = executable

    @staticmethod
    def effort(settings: Agent) -> str:
        return settings.effort

    def command(self, names: tuple[str, ...]) -> list[str]:
        executable = shutil.which(self.executable)
        if executable is None:
            raise ProviderError(f"Claude Code executable unavailable: {self.executable}")
        return [executable, "--print", "--output-format", "json", "--restricted", "--safe-mode",
                "--strict-mcp-config", "--json-schema", json.dumps(schema(names)),
                "--tools", "Read", "--allowedTools", "Read(./observation-*)", "--permission-mode", "dontAsk",
                "--no-session-persistence", "--effort", self.effort(self.settings), "--model", self.settings.model_id]

    def generate(self, prompt: str, *, images: tuple[Path, ...], names: tuple[str, ...],
                 attempt: Path, tick: Callable[[], None],
                 cancelled: Callable[[], bool]) -> Decision:
        """Run one decision, calling tick() while the CLI runs."""
        request = prompt + "".join(f"\nAttachment {i}: read ./{name}"
                                   for i, name in enumerate(attachments(images), 1))
        stdout = run_cli("Claude Code", lambda work, attached: self.command(names), request,
                         images=images, attempt=attempt, model=self.settings.model_id,
                         timeout=self.settings.timeout, tick=tick, cancelled=cancelled)
        return structured_output(stdout, names)


def structured_output(stdout: str, names: tuple[str, ...]) -> Decision:
    """Claude Code wraps the schema-checked reply in its JSON result envelope."""
    try:
        result = json.loads(stdout).get("structured_output")
    except ValueError as exc:
        raise ProviderError(f"Claude Code transport failed: {exc}") from exc
    if result is None:
        raise ProviderError("Claude Code returned no structured output")
    return Decision.parse(json.dumps(result), names)

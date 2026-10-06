"""One tool-free Codex CLI decision, using the operator's external login."""
import json
import shutil
from collections.abc import Callable
from pathlib import Path

from homebody.session.settings import Agent

from .provider import Decision, ProviderError, run_cli, schema

DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "code_mode", "code_mode_host", "view_image",
    "apps", "plugins", "remote_plugin", "multi_agent", "multi_agent_v2", "hooks",
    "browser_use", "browser_use_external", "computer_use", "image_generation",
    "skill_search", "memories", "goals", "shell_snapshot", "sleep_tool",
    "in_app_browser", "in_app_local_automation", "browser_use_full_cdp_access",
)


class Codex:
    LABEL, EXECUTABLE = "Codex", "codex"

    def __init__(self, settings: Agent, executable: str = EXECUTABLE):
        self.settings = settings
        self.executable = executable

    @staticmethod
    def effort(settings: Agent) -> str:
        return {"max": "xhigh"}.get(settings.effort, settings.effort)

    def command(self, work: Path, images: tuple[Path, ...]) -> list[str]:
        executable = shutil.which(self.executable)
        if executable is None:
            raise ProviderError(f"Codex executable unavailable: {self.executable}")
        command = [executable, "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                   "--skip-git-repo-check", "--sandbox", "read-only", "--cd", str(work),
                   "--output-schema", str(work / "schema.json"), "--output-last-message",
                   str(work / "decision.json"), "--json", "--color", "never", "--model",
                   self.settings.model_id, "-c", 'web_search="disabled"',
                   "-c", f'model_reasoning_effort="{self.effort(self.settings)}"', "-c", 'approval_policy="never"']
        for feature in DISABLED_FEATURES:
            command += ["--disable", feature]
        for image in images:
            command += ["--image", str(image)]
        return command + ["-"]

    def generate(self, prompt: str, *, images: tuple[Path, ...], names: tuple[str, ...],
                 attempt: Path, tick: Callable[[], None],
                 cancelled: Callable[[], bool]) -> Decision:
        """Run one decision, calling tick() while the CLI runs."""
        reply = run_cli("Codex", self.command, prompt, images=images, attempt=attempt,
                        files={"schema.json": json.dumps(schema(names), indent=2)},
                        reply="decision.json", stdout="stdout.jsonl", model=self.settings.model_id,
                        timeout=self.settings.timeout, tick=tick, cancelled=cancelled)
        return Decision.parse(reply, names)

"""The decision boundary carries observations and structured choices, never actuators."""
import base64
import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


class ProviderError(RuntimeError):
    pass


class Cancelled(ProviderError):
    pass


class ProviderBusy(ProviderError):
    """The service said it is busy, rate-limited or briefly failing: the same request may succeed later."""


BUSY = ("at capacity", "rate limit", "rate_limit", "overloaded", "too many requests", "internal server error",
        "bad gateway", "service unavailable")
BUSY_STATUS = (429, 500, 502, 503, 529)  # rate limits, server errors, overload


def failure(message):
    """ProviderBusy when MESSAGE says the service is busy, else ProviderError."""
    return (ProviderBusy if any(word in message.lower() for word in BUSY) else ProviderError)(message)


def reported_errors(path):
    """The error messages a CLI wrote to its JSON output, newest last."""
    if not path.is_file():
        return ""
    errors = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and (event.get("type") == "error" or "error" in str(event.get("type", ""))):
            errors.append(str(event.get("message") or event.get("error") or ""))
        elif isinstance(event, dict) and event.get("is_error"):
            errors.append(str(event.get("result") or ""))
    return "; ".join(dict.fromkeys(error for error in errors if error and not expected_notice(error)))


def expected_notice(error: str) -> bool:
    """True for Codex's routine notice that a disabled feature's tool is unavailable."""
    return "is unavailable because" in error and "is disabled" in error


@dataclass(frozen=True)
class Decision:
    text: str
    skill: str
    arguments: dict[str, Any]

    @classmethod
    def parse(cls, raw: str, names: tuple[str, ...]) -> "Decision":
        try:
            value = json.loads(raw)
            if set(value) != {"text", "skill", "arguments_json"}:
                raise ValueError("Expected text, skill, arguments_json")
            if not isinstance(value["text"], str) or value["skill"] not in names:
                raise ValueError("Invalid text or skill")
            arguments = json.loads(value["arguments_json"], parse_constant=_invalid_constant)
            if not isinstance(arguments, dict):
                raise TypeError("Arguments must be a JSON object")
            _finite_numbers(arguments)
            return cls(value["text"], value["skill"], arguments)
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise ProviderError(f"Invalid structured decision: {exc}") from exc


def _finite_numbers(arguments):
    pending = [arguments]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif type(value) in (int, float) and not -sys.float_info.max <= value <= sys.float_info.max:
            raise ValueError("Argument number is outside the finite floating-point range")


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def schema(names: tuple[str, ...]) -> dict:
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "skill": {"type": "string", "enum": list(names)},
            "arguments_json": {"type": "string"},
        },
        "required": ["text", "skill", "arguments_json"],
        "additionalProperties": False,
    }


def attachments(images: tuple[Path, ...]) -> dict[str, Path]:
    """The name each image gets in a decision CLI's work directory, in order."""
    return {f"observation-{i}{image.suffix}": image for i, image in enumerate(images)}


def run_cli(name: str, command: Callable[[Path, tuple[Path, ...]], list[str]], prompt: str, *,
            images: tuple[Path, ...], attempt: Path, model: str, timeout: float,
            tick: Callable[[], None], cancelled: Callable[[], bool],
            files: Mapping[str, str] = MappingProxyType({}), reply: str | None = None,
            stdout: str = "stdout.json") -> str:
    """Run one decision CLI with `prompt` on stdin in a fresh directory holding the images and `files`, and return its stdout, or its `reply` file. `attempt` keeps the inputs, prompt, command, outputs and status.json. Only OSError becomes ProviderError."""
    with recorded_attempt(attempt, prompt, model) as status:
        try:
            with tempfile.TemporaryDirectory(prefix="homebody-decision-") as directory:
                work = Path(directory)
                inputs = attachments(images)
                for filename, image in inputs.items():
                    shutil.copyfile(image, work / filename)
                for filename, text in files.items():
                    (work / filename).write_text(text)
                shutil.copytree(work, attempt, dirs_exist_ok=True)
                argv = command(work, tuple(work / filename for filename in inputs))
                (attempt / "command.json").write_text(json.dumps(argv, indent=2))
                with (attempt / "prompt.txt").open() as stdin, (attempt / stdout).open("w") as out, \
                        (attempt / "stderr.txt").open("w") as err:
                    process = subprocess.Popen(argv, cwd=work, stdin=stdin, stdout=out, stderr=err,
                                               env=cli_environment(), start_new_session=True)
                try:
                    wait(lambda: process.poll() is None, name, timeout, tick, cancelled)
                finally:
                    if process.poll() is None:
                        kill_group(process)
                    status["returncode"] = process.returncode
                if process.returncode:
                    detail = reported_errors(attempt / stdout) or (attempt / "stderr.txt").read_text()[-2000:]
                    raise failure(f"{name} exited {process.returncode}: {detail}")
                if reply is not None:
                    shutil.copyfile(work / reply, attempt / reply)
            return (attempt / (reply or stdout)).read_text()
        except OSError as exc:
            raise ProviderError(f"{name} transport failed: {exc}") from exc


def cli_environment() -> dict[str, str]:
    """The environment without API keys."""
    return {name: value for name, value in os.environ.items() if not name.endswith("_API_KEY")}


def kill_group(process):
    """End the CLI and the native process it may wrap: SIGTERM to its group, then SIGKILL."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=3)
            break
        except subprocess.TimeoutExpired:
            continue
    process.wait()


@contextmanager
def recorded_attempt(attempt: Path, prompt: str, model: str | None):
    """Write prompt.txt and, however the call ends, status.json ("ok", "cancelled", "error" or "interrupted") under ATTEMPT. Yields the status dict for further fields."""
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "prompt.txt").write_text(prompt)
    start = time.monotonic()
    status = {"model": model, "started_unix": time.time(), "status": "interrupted"}
    try:
        yield status
        status["status"] = "ok"
    except Cancelled as exc:
        status.update(status="cancelled", error=str(exc))
        raise
    except ProviderError as exc:
        status.update(status="error", error=str(exc))
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - start
        (attempt / "status.json").write_text(json.dumps(status, indent=2))


def wait(running: Callable[[], bool], name: str, timeout: float, tick: Callable[[], None],
         cancelled: Callable[[], bool]):
    """Call TICK while RUNNING(), raising Cancelled when CANCELLED() and ProviderError after TIMEOUT seconds."""
    start = time.monotonic()
    while running():
        if cancelled():
            raise Cancelled("Decision cancelled by operator")
        if time.monotonic() - start > timeout:
            raise ProviderError(f"{name} timed out after {timeout}s")
        tick()


def image_data(image: Path) -> str:
    return base64.b64encode(image.read_bytes()).decode("ascii")


CREDENTIAL_HEADERS = ("authorization", "x-api-key")


def scrubbed(text: str, headers: Mapping[str, str]) -> str:
    """TEXT with every credential header value replaced, with or without its scheme."""
    for name, value in headers.items():
        if name.lower() in CREDENTIAL_HEADERS:
            for secret in {value, value.removeprefix("Bearer ")}:
                text = text.replace(secret, "[redacted]")
    return text


def error_body(error: urllib.error.HTTPError) -> str:
    """The start of an HTTP error reply's body; a body cut off in transit says so."""
    try:
        return error.read().decode(errors="replace")[:2000]
    except (OSError, http.client.HTTPException) as failure:
        return f"body unreadable ({type(failure).__name__})"


def run_request(name: str, url: str, headers: Mapping[str, str], body: dict, *, prompt: str,
                attempt: Path, timeout: float, tick: Callable[[], None],
                cancelled: Callable[[], bool], redacted: dict | None = None) -> dict:
    """POST one JSON request to a decision API on a daemon thread, calling tick() meanwhile, and return its JSON object reply. `attempt` keeps prompt.txt, request.json (`redacted` in place of a body carrying image bytes), reply.json and status.json. Headers are never written and credentials are scrubbed from errors. HTTP and network failures raise ProviderError, ProviderBusy when the service says it is busy."""
    with recorded_attempt(attempt, prompt, body.get("model")):
        (attempt / "request.json").write_text(json.dumps(redacted or body, indent=2))
        request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                         headers=dict(headers))
        outcome: dict = {}

        def post():
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    outcome["reply"] = response.read().decode(errors="replace")
            except urllib.error.HTTPError as error:
                outcome["error"] = f"{name} HTTP {error.code}: {error_body(error)}"
                outcome["busy"] = error.code in BUSY_STATUS
            except (OSError, http.client.HTTPException) as error:
                outcome["error"] = f"{name} transport failed: {error}"

        def tick_and_join():
            tick()
            worker.join(0.02)

        worker = threading.Thread(target=post, name=f"{name} request", daemon=True)
        worker.start()
        wait(worker.is_alive, name, timeout, tick_and_join, cancelled)
        if "error" in outcome:
            message = scrubbed(outcome["error"], headers)
            raise ProviderBusy(message) if outcome.get("busy") else failure(message)
        if "reply" not in outcome:
            raise RuntimeError(f"{name} request thread died; its traceback went to stderr")
        (attempt / "reply.json").write_text(outcome["reply"])
        try:
            reply = json.loads(outcome["reply"])
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{name} returned no JSON: {exc}") from exc
        if not isinstance(reply, dict):
            raise ProviderError(f"{name} returned no JSON object: {outcome['reply'][:500]}")
        return reply

"""One structured decision per turn from the Anthropic or OpenAI API. The key comes from the environment."""
import json
import os
from collections.abc import Callable
from pathlib import Path

from homebody.session.settings import Agent

from .provider import Decision, ProviderError, image_data, run_request, schema

MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def key(variable: str) -> str:
    """The stripped key from the environment, or ProviderError when unset or not visible ASCII."""
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ProviderError(f"{variable} is not set")
    if not all("!" <= character <= "~" for character in value):
        raise ProviderError(f"{variable} holds characters no API key has; set it again")
    return value


class ClaudeApi:
    """Anthropic Messages API; the decision is the forced input of one tool."""
    URL = "https://api.anthropic.com/v1/messages"
    LABEL, KEY = "Claude", "ANTHROPIC_API_KEY"

    def __init__(self, settings: Agent):
        self.settings = settings

    def request(self, prompt: str, images: tuple[Path, ...], names: tuple[str, ...]) -> dict:
        content = [{"type": "image", "source": {"type": "base64", "media_type": MEDIA[image.suffix.lower()],
                                                "data": image_data(image)}} for image in images]
        content.append({"type": "text", "text": prompt})
        return {"model": self.settings.model_id, "max_tokens": 1024,
                "messages": [{"role": "user", "content": content}],
                "tools": [{"name": "decide", "description": "The one structured decision for this turn.",
                           "input_schema": schema(names)}],
                "tool_choice": {"type": "tool", "name": "decide"}}

    def generate(self, prompt: str, *, images: tuple[Path, ...], names: tuple[str, ...],
                 attempt: Path, tick: Callable[[], None], cancelled: Callable[[], bool]) -> Decision:
        headers = {"x-api-key": key(self.KEY), "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        body = self.request(prompt, images, names)
        reply = run_request("Claude API", self.URL, headers, body, prompt=prompt, attempt=attempt,
                            timeout=self.settings.timeout, tick=tick, cancelled=cancelled,
                            redacted=redact(body, images))
        try:
            used = next((block for block in reply["content"] if block["type"] == "tool_use"), None)
            if used is None:
                raise ProviderError(f"Claude API returned no decision: {json.dumps(reply)[:500]}")
            return Decision.parse(json.dumps(used["input"]), names)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(f"Claude API reply had an unexpected shape: {json.dumps(reply)[:500]}") from exc


class OpenAiApi:
    """OpenAI Chat Completions API with a strict JSON schema response."""
    URL = "https://api.openai.com/v1/chat/completions"
    LABEL, KEY = "OpenAI", "OPENAI_API_KEY"

    def __init__(self, settings: Agent):
        self.settings = settings

    def request(self, prompt: str, images: tuple[Path, ...], names: tuple[str, ...]) -> dict:
        content = [{"type": "image_url", "image_url": {"url": f"data:{MEDIA[image.suffix.lower()]};base64,{image_data(image)}"}}
                   for image in images]
        content.append({"type": "text", "text": prompt})
        return {"model": self.settings.model_id, "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "decision", "strict": True, "schema": schema(names)}}}

    def generate(self, prompt: str, *, images: tuple[Path, ...], names: tuple[str, ...],
                 attempt: Path, tick: Callable[[], None], cancelled: Callable[[], bool]) -> Decision:
        headers = {"Authorization": f"Bearer {key(self.KEY)}", "content-type": "application/json"}
        body = self.request(prompt, images, names)
        reply = run_request("OpenAI API", self.URL, headers, body, prompt=prompt, attempt=attempt,
                            timeout=self.settings.timeout, tick=tick, cancelled=cancelled,
                            redacted=redact(body, images))
        try:
            content = reply["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"OpenAI API returned no decision: {json.dumps(reply)[:500]}") from exc
        return Decision.parse(content, names)


def redact(body: dict, images: tuple[Path, ...]) -> dict:
    """The request as recorded: image bytes replaced by the image file names, in order."""
    names = iter(image.name for image in images)
    record = json.loads(json.dumps(body))
    for message in record["messages"]:
        for part in message["content"]:
            if part.get("type") == "image":
                part["source"] = {"file": next(names)}
            elif part.get("type") == "image_url":
                part["image_url"] = {"file": next(names)}
    return record

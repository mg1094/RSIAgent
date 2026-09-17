"""Native Anthropic Messages API client.

The system prompt is a top-level field in the Messages API rather than a
message, so it is split out here.  Requests are same-shape as
:class:`~rsiagent.llm.openai_compat.OpenAIChatClient` from the caller's view.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

from ..errors import InfrastructureFailure
from .base import BaseLLMClient, LLMResponse, Message, Usage

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"


class AnthropicChatClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        name: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__()
        self.model = model
        self.name = name or model
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        self.timeout = timeout
        self.max_retries = max_retries
        if not self.api_key:
            raise InfrastructureFailure(
                "llm", "no API key: set ANTHROPIC_API_KEY (or pass api_key=)"
            )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> LLMResponse:
        system_parts = [m.content for m in messages if m.role == "system"]
        turns = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        if not turns:
            turns = [{"role": "user", "content": "(no instruction)"}]

        body: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "max_tokens": max_tokens or 8192,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if stop:
            body["stop_sequences"] = list(stop)

        url = f"{self.base_url}/messages"
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return self._record(self._parse(payload))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:600]
                if exc.code != 429 and 400 <= exc.code < 500:
                    raise InfrastructureFailure(
                        "llm",
                        f"{self.model} HTTP {exc.code} from {self.base_url}: {detail}",
                    ) from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < self.max_retries - 1:
                time.sleep(min(2.0**attempt, 8.0))

        raise InfrastructureFailure(
            "llm", f"{self.model} failed after {self.max_retries} attempts: {last_error}"
        )

    def _parse(self, payload: dict[str, Any]) -> LLMResponse:
        blocks = payload.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage_blob = payload.get("usage") or {}
        return LLMResponse(
            text=text,
            model=payload.get("model", self.model),
            usage=Usage(
                prompt_tokens=int(usage_blob.get("input_tokens", 0) or 0),
                completion_tokens=int(usage_blob.get("output_tokens", 0) or 0),
            ),
            finish_reason=payload.get("stop_reason"),
            raw=payload,
        )

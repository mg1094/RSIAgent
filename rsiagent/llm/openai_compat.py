"""OpenAI-compatible chat completions over stdlib HTTP.

Works against any endpoint that speaks ``POST /chat/completions`` — OpenAI,
OpenRouter, vLLM, Ollama, DeepSeek, MiniMax, GLM, Kimi, and friends.  Keeping
this on ``urllib`` means the framework has no mandatory third-party dependency.

Note on endpoints: several vendors serve the same model on region-specific
hosts and *keys do not transfer between them*.  A 401 from one host is often a
wrong-host error rather than a bad key, so the error message below prints the
base URL that was actually used.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from ..errors import InfrastructureFailure
from .base import BaseLLMClient, LLMResponse, Message, Usage

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIChatClient(BaseLLMClient):
    """Chat-completions client with bounded retries on transport failures."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        name: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        default_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.name = name or model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_headers = dict(default_headers or {})
        self.extra_body = dict(extra_body or {})
        if not self.api_key:
            raise InfrastructureFailure(
                "llm",
                "no API key: set OPENAI_API_KEY (or pass api_key=). "
                "Use ScriptedClient for offline runs.",
            )

    # -- request plumbing -------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "rsiagent/0.1",
        }
        headers.update(self.default_headers)
        return headers

    def _payload(
        self,
        messages: Sequence[Message],
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stop: Sequence[str] | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if stop:
            body["stop"] = list(stop)
        body.update(self.extra_body)
        return body

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(self._payload(messages, temperature, top_p, max_tokens, stop)).encode(
            "utf-8"
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return self._record(self._parse(payload))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:600]
                # 4xx other than 429 will not fix themselves.
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
        try:
            choice = payload["choices"][0]
        except (KeyError, IndexError) as exc:
            raise InfrastructureFailure("llm", f"malformed response: {payload}") from exc

        message = choice.get("message") or {}
        text = message.get("content") or ""
        # Some providers put reasoning in a side channel and leave content empty.
        if not text and message.get("reasoning_content"):
            text = message["reasoning_content"]

        usage_blob = payload.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_blob.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage_blob.get("completion_tokens", 0) or 0),
        )
        return LLMResponse(
            text=text,
            model=payload.get("model", self.model),
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            raw=payload,
        )

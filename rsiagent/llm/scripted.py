"""Deterministic offline clients.

These exist so the full three-phase lifecycle can be exercised with no API key,
no network, and no GPU: the same orchestration code that drives GLM-5.3 in the
paper drives a handler function here.  They are *not* language models, and a
scripted run reproduces the protocol, never the reported benchmark scores.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence

from .base import BaseLLMClient, LLMResponse, Message, Usage

Handler = Callable[[Sequence[Message]], str]


class ScriptedClient(BaseLLMClient):
    """Dispatches each call to a caller-supplied handler.

    ``handler`` receives the full message history and returns the assistant
    text.  When ``replay`` is given instead, responses are popped in order and
    the final response repeats once the queue is exhausted.
    """

    def __init__(
        self,
        handler: Handler | None = None,
        *,
        replay: Iterable[str] | None = None,
        name: str = "scripted",
    ) -> None:
        super().__init__()
        if handler is None and replay is None:
            raise ValueError("ScriptedClient needs either a handler or a replay list")
        self.name = name
        self._handler = handler
        self._queue: deque[str] = deque(replay or ())
        self._last: str = ""
        self.transcript: list[tuple[list[Message], str]] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> LLMResponse:
        history = list(messages)
        if self._handler is not None:
            text = self._handler(history)
        elif self._queue:
            text = self._queue.popleft()
            self._last = text
        else:
            text = self._last

        self.transcript.append((history, text))
        return self._record(
            LLMResponse(
                text=text,
                model=self.name,
                usage=Usage(
                    prompt_tokens=sum(len(m.content) // 4 for m in history),
                    completion_tokens=len(text) // 4,
                ),
                finish_reason="stop",
            )
        )


class EchoClient(BaseLLMClient):
    """Returns a fixed string.  Useful for prompt-shape tests."""

    def __init__(self, text: str, name: str = "echo") -> None:
        super().__init__()
        self.name = name
        self.text = text
        self.seen: list[list[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> LLMResponse:
        self.seen.append(list(messages))
        return self._record(LLMResponse(text=self.text, model=self.name))


class RoleRoutedClient(BaseLLMClient):
    """Routes to a per-role client using a marker embedded in the system message.

    A thin convenience for wiring one scripted backend across all three roles
    without constructing three separate clients by hand.
    """

    MARKER = "[[role:"

    def __init__(self, routes: dict[str, BaseLLMClient], default: BaseLLMClient) -> None:
        super().__init__()
        self.name = "role-routed"
        self.routes = routes
        self.default = default

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> LLMResponse:
        target = self.default
        for message in messages:
            if message.role == "system" and self.MARKER in message.content:
                tag = message.content.split(self.MARKER, 1)[1].split("]", 1)[0].strip()
                target = self.routes.get(tag, self.default)
                break
        response = target.complete(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        return self._record(response)

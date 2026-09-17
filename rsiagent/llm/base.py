"""Model transport interfaces.

RSIAgent treats the model as a text completion with a message history.  Every
role — actor, verifier, curriculum, observer — is bound to one
:class:`LLMClient`, and role separation is a property of *contexts*
(:mod:`rsiagent.agents.context`), not of the client.  The paper makes this
explicit: the verifier and curriculum agents share a model but never share a
context.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # convenience in f-strings
        return self.text


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface every backend must provide."""

    name: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> LLMResponse:
        """Return one completion for ``messages``."""


class BaseLLMClient:
    """Shared bookkeeping: cumulative usage per client instance."""

    name: str = "base"

    def __init__(self) -> None:
        self.total_usage = Usage()
        self.calls = 0

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> LLMResponse:  # pragma: no cover - abstract
        raise NotImplementedError

    def _record(self, response: LLMResponse) -> LLMResponse:
        self.calls += 1
        self.total_usage = self.total_usage + response.usage
        return response

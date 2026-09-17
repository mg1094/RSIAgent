"""Role-scoped conversation contexts.

"Separate contexts give each agent the information needed for its role without
sharing private reasoning across roles" (Appendix A.1).  A context is the unit
of both memory (the message history) and isolation, and its *lifetime* is
specified by the paper:

* **Actor** — persists through one branch's work, then through that same
  branch's distillation, reconciliation, and diagnosis.  This is what makes
  "the actor agent that produced an experience also decides what to retain from
  it" possible: the trajectory is still in context when learning begins.
* **Verifier** — "persists across candidate revisions within a target attempt,
  but is not shared across distinct projects or target attempts".
* **Curriculum** — "persists within an exploration lineage".

Nothing here is shared between two contexts, so an accidental read of another
role's reasoning is a programming error rather than a prompt-design risk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..llm.base import LLMClient, Message, Usage


@dataclass
class Turn:
    """One recorded exchange, kept for traces and tests."""

    index: int
    response: str
    prompt_chars: int = 0


class AgentContext:
    """A message history bound to one client and one role."""

    def __init__(
        self,
        client: LLMClient,
        system_prompt: str,
        *,
        role: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.role = role
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self._messages: list[Message] = [Message(role="system", content=system_prompt)]
        self.turns: list[Turn] = []
        self.usage = Usage()

    # -- history ----------------------------------------------------------
    @property
    def messages(self) -> Sequence[Message]:
        return tuple(self._messages)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def append(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))  # type: ignore[arg-type]

    def extend(self, messages: Sequence[Message]) -> None:
        self._messages.extend(messages)

    def note(self, content: str) -> None:
        """Add an out-of-band observation the model should see as context."""
        self.append("user", content)

    def reset_history(self) -> None:
        """Drop the conversation but keep the role binding.

        Used when an attempt is replayed in a reset environment: the paper
        resets interaction history between independent attempts while the
        harness stays the same.
        """
        self._messages = [Message(role="system", content=self.system_prompt)]
        self.turns.clear()

    # -- model calls ------------------------------------------------------
    def ask(self, content: str | None = None) -> str:
        """Optionally append a user turn, then get one assistant reply."""
        if content is not None:
            self.append("user", content)

        response = self.client.complete(
            self._messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        self._messages.append(Message(role="assistant", content=response.text))
        self.usage = self.usage + response.usage
        self.turns.append(
            Turn(
                index=len(self.turns),
                response=response.text,
                prompt_chars=sum(len(m.content) for m in self._messages),
            )
        )
        return response.text

    def transcript(self, *, limit: int = 60) -> str:
        """A compact rendering for journals and failure reports."""
        lines: list[str] = []
        for message in self._messages[-limit:]:
            body = message.content.strip()
            if len(body) > 2000:
                body = body[:1000] + "\n... [truncated] ...\n" + body[-1000:]
            lines.append(f"[{message.role}] {body}")
        return "\n\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"AgentContext(role={self.role!r}, turns={len(self.turns)}, "
            f"messages={len(self._messages)})"
        )


@dataclass
class ContextFactory:
    """Builds contexts for one role from a single client binding."""

    client: LLMClient
    system_prompt: str
    role: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    spawned: list[AgentContext] = field(default_factory=list)

    def __call__(self) -> AgentContext:
        context = AgentContext(
            self.client,
            self.system_prompt,
            role=self.role,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        self.spawned.append(context)
        return context

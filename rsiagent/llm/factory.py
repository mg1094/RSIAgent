"""Build the per-role clients a run needs."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import RoleModelConfig, RunConfig
from ..errors import ConfigurationError
from .base import LLMClient
from .openai_compat import OpenAIChatClient


@dataclass
class RoleClients:
    """One client per role.

    ``observer`` is optional and only consulted when the actor issues a ``look``
    action.  The paper's reference configuration binds visual observations to
    the same model as the verifier agent.
    """

    actor: LLMClient
    verifier: LLMClient
    curriculum: LLMClient
    observer: LLMClient | None = None


def _build_one(role: RoleModelConfig, provider: str) -> LLMClient:
    if provider in ("openai", "openai_compat", "openrouter", "auto"):
        return OpenAIChatClient(
            role.model,
            default_headers=role.extra.get("headers"),
            extra_body=role.extra.get("body", {}),
        )
    if provider == "anthropic":
        from .anthropic_client import AnthropicChatClient

        return AnthropicChatClient(role.model)
    raise ConfigurationError(f"unknown provider: {provider!r}")


def build_clients(config: RunConfig, *, provider: str = "auto") -> RoleClients:
    """Instantiate clients for every role in ``config``."""
    return RoleClients(
        actor=_build_one(config.actor, provider),
        verifier=_build_one(config.verifier, provider),
        curriculum=_build_one(config.curriculum, provider),
        observer=_build_one(config.observer, provider) if config.observer else None,
    )

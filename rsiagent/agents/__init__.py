"""The three agent roles and the contexts that isolate them."""

from .actor import ActorAgent, ActorRun, UserChannel
from .context import AgentContext, ContextFactory
from .curriculum import CurriculumAgent, Handoff, OutcomeSummary
from .verifier import VerdictResult, VerifierAgent

__all__ = [
    "ActorAgent",
    "ActorRun",
    "AgentContext",
    "ContextFactory",
    "CurriculumAgent",
    "Handoff",
    "OutcomeSummary",
    "UserChannel",
    "VerdictResult",
    "VerifierAgent",
]

"""Exception hierarchy for RSIAgent.

The distinction that matters throughout this package is between a *grounded*
outcome (the environment said PASS/FAIL) and an *ungrounded* one (a crash, a
timeout, a malformed model response).  Algorithm A1 of the paper is explicit
that ungrounded outcomes "suspend advancement; they are not converted into task
verdicts", so every failure path raises one of the exceptions below rather than
returning a verdict.
"""

from __future__ import annotations


class RSIAgentError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(RSIAgentError):
    """The run configuration is internally inconsistent or incomplete."""


class EnvironmentError_(RSIAgentError):
    """The environment could not be driven as requested.

    Named with a trailing underscore so it never shadows the builtin.
    """


class ProgramTimeout(EnvironmentError_):
    """A submitted program exceeded its per-call execution limit."""


class CheckpointError(EnvironmentError_):
    """A snapshot could not be taken or restored."""


class UnverifiedOutcome(RSIAgentError):
    """The verifier could not resolve a material claim.

    This is the ``UNVERIFIED`` verdict escaping as an exception.  The reference
    implementation treats it as blocking: it is neither a success nor a failure
    and must not become a learning example.
    """

    def __init__(self, reason: str, findings: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.findings = findings


class InfrastructureFailure(RSIAgentError):
    """Something outside the agent protocol broke (transport, host, grader).

    Unscored by definition.  The protocol records it and suspends the lineage
    rather than recording a zero.
    """

    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(f"[{stage}] {reason}")
        self.stage = stage
        self.reason = reason


class ModelResponseError(RSIAgentError):
    """A model response could not be parsed into the required structure."""


class MemoryIntegrityError(RSIAgentError):
    """Frozen memory changed, or a write was attempted where it is forbidden."""

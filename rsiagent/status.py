"""Lifecycle vocabulary.

The paper is careful about one distinction in particular: a lineage that stops
because it converged must not be confused with one that stopped because the
budget ran out, the verifier could not resolve a claim, or the host broke.
"`STALLED` and budget exits must remain distinguishable from successful
convergence", and "unresolved verification or infrastructure errors suspend
advancement; they are not converted into task verdicts."
"""

from __future__ import annotations

from enum import Enum


class Verdict(str, Enum):
    """A grounded judgement about a candidate."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNVERIFIED = "UNVERIFIED"

    @property
    def grounded(self) -> bool:
        """Whether this counts as evidence for learning."""
        return self is not Verdict.UNVERIFIED


class Decision(str, Enum):
    """A curriculum agent's selection outcome."""

    PROJECTS = "PROJECTS"
    PROJECT = "PROJECT"
    READY_FOR_TARGET = "READY_FOR_TARGET"
    SATURATED = "SATURATED"
    STALLED = "STALLED"

    @property
    def procures_work(self) -> bool:
        return self in (Decision.PROJECTS, Decision.PROJECT)


class TerminalStatus(str, Enum):
    """How a lineage ended."""

    COMPLETED = "completed"
    """DRS finished under its declared stopping policy (curriculum_review or
    verifier_pass).  A completed lifecycle need not have a successful target
    verdict."""

    SATURATED = "saturated"
    """BRS ended because the curriculum agent judged further exploration to have
    insufficient expected value."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    """A recorded budget boundary was reached.  Not convergence."""

    STALLED = "stalled"
    """The curriculum agent could author no productive work."""

    UNVERIFIED = "unverified"
    """A material claim remained unresolved.  Blocks advancement."""

    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    """Host, transport, or grader failure.  Unscored — not an official zero."""

    ABORTED = "aborted"
    """Caller-requested stop."""

    @property
    def is_success(self) -> bool:
        return self is TerminalStatus.COMPLETED

    @property
    def is_scored(self) -> bool:
        """Whether a benchmark score may be recorded for this lineage."""
        return self not in (
            TerminalStatus.INFRASTRUCTURE_FAILURE,
            TerminalStatus.UNVERIFIED,
        )


class Phase(str, Enum):
    """Runtime phases, mapped to the paper's three stages."""

    BRS = "phase1_brs"
    DRS = "phase2_drs"
    EVAL = "phase3_eval"

    @property
    def paper_stage(self) -> str:
        return {
            Phase.BRS: "Broad Recursive Self-exploration",
            Phase.DRS: "Deep Recursive Self-exploration",
            Phase.EVAL: "Test-time memory reuse",
        }[self]

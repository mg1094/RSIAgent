"""The actor–verifier harness.

This loop is shared by all three phases — "RSI and test-time execution use the
same agent framework".  What changes between phases is only whether the
curriculum agent is consulted and whether memory accepts writes.

The ordering inside one revision is what makes verification trustworthy::

    actor works  ->  checkpoint  ->  verifier probes  ->  restore checkpoint

The verifier is told it may "observe its files, processes, localhost services,
network, GUI, and IPC and may run arbitrary tests" precisely because the
restore guarantees "none of your in-VM effects enter the scored state".  Doing
the restore in the harness rather than asking the verifier to be careful is the
difference between a property and a hope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..agents.actor import ActorAgent, ActorRun
from ..agents.context import AgentContext
from ..agents.verifier import VerdictResult, VerifierAgent
from ..env.base import Environment
from ..errors import InfrastructureFailure, UnverifiedOutcome
from ..status import Verdict


@dataclass
class AttemptResult:
    """One actor–verifier attempt at a task or project."""

    verdict: Verdict
    instruction: str
    revisions: int = 0
    actor_runs: list[ActorRun] = field(default_factory=list)
    verdicts: list[VerdictResult] = field(default_factory=list)
    final_findings: str = ""
    terminal: str = "verdict"
    """How the attempt stopped.

    ``verdict``               a grounded judgement settled it
    ``revisions_exhausted``   the actor used its revisions and the last grounded
                              verdict stands — a FAIL, but a *final* one
    ``unverified``            the verifier could not resolve a material claim
    ``iteration_limit``       the actor ran out of turns before declaring done
    """

    @property
    def actor_run(self) -> ActorRun:
        return self.actor_runs[-1]

    @property
    def passed(self) -> bool:
        return self.verdict is Verdict.PASS

    @property
    def grounded(self) -> bool:
        return self.verdict.grounded

    @property
    def programs_run(self) -> int:
        return sum(r.programs_run for r in self.actor_runs)

    @property
    def iterations(self) -> int:
        return sum(r.iterations for r in self.actor_runs)

    def render(self) -> str:
        return f"{self.verdict.value} after {self.revisions + 1} revision(s)"

    def delivery_note(self) -> str:
        """What the learning step should be told about how this ended."""
        if self.terminal == "iteration_limit":
            return (
                "The attempt hit its iteration limit before the actor declared "
                "completion. The verdict below judges whatever state it left behind."
            )
        if self.terminal == "unverified":
            return (
                "Verification could not resolve a material claim. This outcome is "
                "NOT a success and NOT a failure; it must not be learned from."
            )
        return ""


class TaskHarness:
    """Runs the action–verification loop until the verifier is satisfied."""

    def __init__(
        self,
        actor: ActorAgent,
        verifier: VerifierAgent,
        *,
        max_revisions: int = 3,
        max_unverified_retries: int = 1,
    ) -> None:
        self.actor = actor
        self.verifier = verifier
        self.max_revisions = max_revisions
        self.max_unverified_retries = max_unverified_retries

    def attempt(
        self,
        environment: Environment,
        *,
        actor_context: AgentContext,
        verifier_context: AgentContext,
        instruction: str,
        memory_text: str | None,
        practice: bool = False,
        project_id: str = "",
        task_id: str = "",
        fixtures: dict[str, str] | None = None,
        iteration_limit: int = 500,
        watchdog_s: float = 36_000.0,
        program_timeout_s: float | None = None,
        environment_note: str = "",
    ) -> AttemptResult:
        """Attempt ``instruction``, revising against verifier findings."""
        result = AttemptResult(verdict=Verdict.UNVERIFIED, instruction=instruction)
        unverified_retries = 0
        feedback = ""

        for revision in range(self.max_revisions + 1):
            result.revisions = revision

            actor_run = self.actor.run(
                actor_context,
                environment,
                instruction=instruction,
                memory_text=memory_text,
                practice=practice,
                project_id=project_id,
                task_id=task_id,
                fixtures=fixtures,
                iteration_limit=iteration_limit,
                watchdog_s=watchdog_s,
                program_timeout_s=program_timeout_s,
                environment_note=environment_note if revision == 0 else feedback,
            )
            result.actor_runs.append(actor_run)

            verdict_result = self._verify(
                environment,
                verifier_context,
                instruction=instruction,
                practice=practice,
                allow_unverified=not practice,
            )
            result.verdicts.append(verdict_result)
            result.verdict = verdict_result.verdict
            result.final_findings = verdict_result.findings

            if verdict_result.verdict is Verdict.PASS:
                result.terminal = "verdict"
                return result

            if verdict_result.verdict is Verdict.FAIL:
                if revision == self.max_revisions:
                    # Out of revisions.  The verdict is a grounded, final FAIL —
                    # not an exhausted iteration budget, and the distinction
                    # matters to anything reading the journal afterwards.
                    result.terminal = "revisions_exhausted"
                    return result
                feedback = self._revision_feedback(verdict_result)
                actor_context.append("user", feedback)
                continue

            # UNVERIFIED: the actor may supply evidence for independent
            # reinspection, but the outcome never becomes a learning example.
            if unverified_retries >= self.max_unverified_retries:
                result.terminal = "unverified"
                return result
            unverified_retries += 1
            feedback = self._evidence_request(verdict_result)
            actor_context.append("user", feedback)

        result.terminal = "iteration_limit"
        return result

    # -- internals --------------------------------------------------------
    def _verify(
        self,
        environment: Environment,
        verifier_context: AgentContext,
        *,
        instruction: str,
        practice: bool,
        allow_unverified: bool,
    ) -> VerdictResult:
        """Verify inside a checkpoint, then restore unconditionally."""
        checkpoint = environment.snapshot(label="verify")
        try:
            return self.verifier.verify(
                verifier_context,
                environment,
                instruction=instruction,
                practice=practice,
                allow_unverified=allow_unverified,
            )
        except Exception as exc:
            if isinstance(exc, InfrastructureFailure):
                raise
            raise InfrastructureFailure(
                "verification", f"verifier raised {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            # Always roll back: a crashed verifier must not leave its probes in
            # the state that gets graded.
            environment.restore(checkpoint)
            environment.discard(checkpoint)

    @staticmethod
    def _revision_feedback(verdict: VerdictResult) -> str:
        return (
            "## Verifier findings on your last candidate — verdict FAIL\n\n"
            f"{verdict_result_findings(verdict)}\n\n"
            "These findings are binding: the verifier inspected the candidate "
            "directly rather than reading your description of it. Repair the "
            "candidate, or produce evidence that the finding is mistaken. Do not "
            "restate your previous answer."
        )

    @staticmethod
    def _evidence_request(verdict: VerdictResult) -> str:
        return (
            "## Verifier could not resolve a material claim — verdict UNVERIFIED\n\n"
            f"{verdict_result_findings(verdict)}\n\n"
            "This is not a failure, and no score has been recorded. Supply "
            "evidence the verifier can independently re-inspect: produce the "
            "artifact, print the value, or demonstrate the requirement holds. "
            "Narrating that it works is not evidence."
        )


def verdict_result_findings(verdict: VerdictResult) -> str:
    return verdict.findings.strip() or "(verifier returned no findings)"


__all__ = [
    "AttemptResult",
    "TaskHarness",
    "UnverifiedOutcome",
    "verdict_result_findings",
]

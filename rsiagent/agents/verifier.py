"""The verifier agent: independent, evidence-grounded judgement.

Three properties make this role load-bearing in the paper, and each is enforced
here rather than requested in prose:

1. **It cannot see the actor.**  The verifier receives an
   :class:`~rsiagent.env.base.Environment` and an instruction.  No transcript,
   no memory, no program log is ever passed in.  "The verifier agent is
   isolated from the actor agent's private reasoning and memory, which helps
   reduce correlated errors during evaluation."
2. **It may probe but not repair.**  Probes run inside a checkpoint that the
   harness restores afterwards, so the scored candidate is untouched.
3. **Unresolved is a real answer.**  The target interface returns
   ``UNVERIFIED``, and the protocol treats it as blocking rather than as a
   soft PASS.  Practice projects get PASS/FAIL only, matching Appendix A.1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..env.base import Environment
from ..errors import InfrastructureFailure, ModelResponseError
from ..parsing import parse_actor_action, parse_verdict
from ..prompts import verifier as verifier_prompts
from ..status import Verdict
from .context import AgentContext


@dataclass
class VerdictResult:
    """A grounded judgement plus the evidence behind it."""

    verdict: Verdict
    findings: str = ""
    probes_run: int = 0
    context: AgentContext | None = None
    duration_s: float = 0.0
    exhausted: bool = False
    """True when the verifier ran out of budget before concluding."""

    @property
    def status(self) -> str:
        return self.verdict.value

    @property
    def grounded(self) -> bool:
        return self.verdict.grounded

    def render(self) -> str:
        return f"VERDICT: {self.verdict.value}\nFINDINGS: {self.findings.strip()}"


class VerifierAgent:
    """Inspects candidates inside a checkpoint the harness controls."""

    def __init__(self, *, max_probes: int = 40, program_timeout_s: float | None = None) -> None:
        self.max_probes = max_probes
        self.program_timeout_s = program_timeout_s

    def verify(
        self,
        context: AgentContext,
        environment: Environment,
        *,
        instruction: str,
        practice: bool = False,
        iteration_limit: int = 60,
        allow_unverified: bool = True,
    ) -> VerdictResult:
        """Judge the candidate currently in ``environment``.

        The caller is responsible for snapshotting before this call and
        restoring after it.  That ordering is the whole reason a verifier can be
        allowed to run arbitrary probes.
        """
        request = verifier_prompts.verification_request(instruction, practice=practice)
        context.append("user", request)

        allowed = ("PASS", "FAIL", "UNVERIFIED") if allow_unverified else ("PASS", "FAIL")
        started = time.monotonic()
        result = VerdictResult(verdict=Verdict.UNVERIFIED, context=context)

        for _ in range(iteration_limit):
            reply = context.ask()

            # Any verdict ends the inspection, including one this interface does
            # not offer.  A practice verifier that answers UNVERIFIED has not
            # resolved the project; it must not be mistaken for a malformed
            # reply, and it must not become a PASS or a FAIL by default.
            if _has_verdict(reply, ALL_VERDICTS):
                parsed = parse_verdict(reply, allowed=ALL_VERDICTS)
                if parsed.status not in allowed:
                    result.verdict = Verdict.UNVERIFIED
                    result.findings = (
                        f"The verifier returned {parsed.status}, which this interface "
                        f"does not offer (allowed: {', '.join(allowed)}). Recorded as "
                        f"unresolved.\n\n{parsed.findings}"
                    )
                    result.duration_s = time.monotonic() - started
                    return result
                result.verdict = Verdict(parsed.status)
                result.findings = parsed.findings
                result.duration_s = time.monotonic() - started
                return result

            # No verdict yet: expect a probe program.
            try:
                action = parse_actor_action(reply)
            except ModelResponseError as exc:
                raise ModelResponseError(
                    f"verifier reply is neither a probe program nor a VERDICT: {reply[:200]!r}"
                ) from exc

            if action.kind != "program":
                raise ModelResponseError(
                    f"verifier may only run probes or conclude; got ACTION: {action.kind}"
                )
            if result.probes_run >= self.max_probes:
                raise InfrastructureFailure(
                    "verification",
                    f"verifier exceeded {self.max_probes} probes without concluding",
                )

            probe = environment.execute(
                action.program,
                kind=action.program_kind,  # type: ignore[arg-type]
                timeout=self.program_timeout_s,
            )
            result.probes_run += 1
            context.append("user", probe.render())

        # Budget exhausted without a verdict.  Do NOT invent one: the paper is
        # explicit that an unresolved outcome blocks rather than decides.
        result.exhausted = True
        result.findings = (
            "Verifier exhausted its inspection budget without reaching a verdict. "
            "Recorded as UNVERIFIED; this is not a judgement about the candidate."
        )
        result.verdict = Verdict.UNVERIFIED
        result.duration_s = time.monotonic() - started
        return result


ALL_VERDICTS: tuple[str, ...] = ("PASS", "FAIL", "UNVERIFIED")


def _has_verdict(text: str, allowed: tuple[str, ...]) -> bool:
    import re

    pattern = re.compile(rf"VERDICT\s*:\s*({'|'.join(allowed)})\b", re.IGNORECASE)
    return bool(pattern.search(text or ""))

"""Experience-owned learning.

'The actor agent that produced an experience also decides what to retain from
it.'  What that means mechanically is that consolidation runs *in the context
that did the work* — so these tests assert both the effects on memory and the
information the learner is given.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rsiagent.agents.context import ContextFactory
from rsiagent.errors import UnverifiedOutcome
from rsiagent.learning.consolidation import consolidate
from rsiagent.llm.scripted import ScriptedClient
from rsiagent.memory.bank import MemoryBank
from rsiagent.prompts import learning as learning_prompts
from rsiagent.status import Verdict


def make_context(replies: list[str]):
    calls: list[str] = []

    def handler(messages):
        calls.append(messages[-1].content)
        index = min(len(calls) - 1, len(replies) - 1)
        return replies[index]

    context = ContextFactory(ScriptedClient(handler), "system", role="actor")()
    return context, calls


def test_unverified_outcomes_are_never_consolidated(tmp_path: Path):
    bank = MemoryBank(tmp_path / "memory")
    context, _ = make_context(["ACTION: done"])

    with pytest.raises(UnverifiedOutcome):
        consolidate(
            context,
            bank.session(),
            verdict=Verdict.UNVERIFIED,
            findings="could not resolve",
        )
    assert bank.is_empty


def test_distillation_then_reconciliation_both_apply(tmp_path: Path):
    """Two learning steps, in order, both editing the same working copy."""
    bank = MemoryBank(tmp_path / "memory")
    session = bank.session()
    (session.path / "old.md").write_text("stale advice")

    context, calls = make_context(
        [
            # Step 1: distillation adds a procedure and removes stale advice.
            "```memory:write procs/bom.md\nOpen with utf-8-sig.\n```\n\n"
            "```memory:delete old.md\n```\n\nACTION: done",
            # Step 2: reconciliation narrows an over-broad claim.
            "```memory:write procs/bom.md\nOpen the northern export with utf-8-sig.\n```\n"
            "ACTION: done",
            "The remaining uncertainty is the date convention.\n\nACTION: done",
        ]
    )

    result = consolidate(context, session, verdict=Verdict.PASS, findings="it worked")

    assert result.committed
    assert bank.files()["procs/bom.md"] == b"Open the northern export with utf-8-sig.\n"
    assert "old.md" not in bank.files(), "the delete in step 1 must survive"
    assert "date convention" in result.diagnosis
    assert result.stats_after.file_count == 1

    # The reconciliation step is handed the *draft* state, not the original.
    assert "procs/bom.md" in calls[1]
    assert "## Learning step 1 of 2" in calls[0]
    assert "## Learning step 2 of 2" in calls[1]


def test_failed_outcomes_can_still_teach(tmp_path: Path):
    """'A FAIL may still contain valuable evidence, but must not be recorded as
    a verified success.'"""
    bank = MemoryBank(tmp_path / "memory")
    session = bank.session()
    context, calls = make_context(
        [
            "```memory:write failures/export.md\nA default export drops the override.\n```\n"
            "ACTION: done",
            "ACTION: done",
            "Diagnosis.\n\nACTION: done",
        ]
    )

    result = consolidate(
        context, session, verdict=Verdict.FAIL, findings="the export settings differed"
    )

    assert result.committed
    assert result.memory_changed
    assert bank.files()["failures/export.md"].startswith(b"A default export")
    # The lesson is recorded as a failure, and the prompt said so.
    assert "**Verdict:** FAIL" in calls[0]


def test_diagnosis_is_separate_from_memory(tmp_path: Path):
    """The diagnosis is a handoff, not a memory write."""
    bank = MemoryBank(tmp_path / "memory")
    context, calls = make_context(
        ["ACTION: done", "ACTION: done", "Only conclusions, not reasoning.\n\nACTION: done"]
    )

    result = consolidate(context, bank.session(), verdict=Verdict.PASS, findings="ok")

    assert result.diagnosis == "Only conclusions, not reasoning."
    assert bank.is_empty, "a diagnosis must not itself become memory"
    assert "Learning diagnosis" in calls[2]
    assert "Do not include private chain-of-thought" in calls[2]


def test_no_commit_leaves_the_bank_untouched(tmp_path: Path):
    """The draft is promotable but need not be promoted."""
    bank = MemoryBank(tmp_path / "memory")
    session = bank.session()
    context, _ = make_context(
        ["```memory:write draft.md\nbody\n```\nACTION: done", "ACTION: done", "d\nACTION: done"]
    )

    result = consolidate(context, session, verdict=Verdict.PASS, findings="ok", commit=False)

    assert not result.committed
    assert result.memory_changed, "the session itself did change"
    assert bank.is_empty, "but canonical memory did not"


def test_learner_may_leave_memory_unchanged(tmp_path: Path):
    """'... you may add, revise, reorganize, delete, or leave memory unchanged.'"""
    bank = MemoryBank(tmp_path / "memory")
    context, _ = make_context(
        [
            "Nothing here generalises.\n\nACTION: done",
            "Confirmed.\n\nACTION: done",
            "Nothing to report.\n\nACTION: done",
        ]
    )

    result = consolidate(context, bank.session(), verdict=Verdict.PASS, findings="ok")

    assert not result.memory_changed
    assert bank.is_empty


def test_diagnosis_can_be_skipped(tmp_path: Path):
    bank = MemoryBank(tmp_path / "memory")
    context, calls = make_context(["ACTION: done", "ACTION: done"])

    result = consolidate(
        context,
        bank.session(),
        verdict=Verdict.PASS,
        findings="ok",
        enable_diagnosis=False,
    )

    assert result.diagnosis == ""
    assert len(calls) == 2, "no third call when the diagnosis is disabled"


def test_prompt_carries_the_verifier_findings(tmp_path: Path):
    bank = MemoryBank(tmp_path / "memory")
    context, calls = make_context(["ACTION: done", "ACTION: done", "d\nACTION: done"])

    consolidate(
        context,
        bank.session(),
        verdict=Verdict.FAIL,
        findings="the delimiter was wrong on two of three exports",
    )

    assert "the delimiter was wrong on two of three exports" in calls[0]


def test_learning_prompts_are_the_papers_prompts():
    """Spot-check that the quoted instructions survived."""
    text = learning_prompts.distillation("FAIL", "f", "m")
    assert "Make causal claims only when your trajectory and evidence support them" in text
    assert "there is no required schema, length, number of files, or number of" in text

    recon = learning_prompts.reconciliation("m")
    assert "Do not merely append this episode" in recon

    assert "Do not include private chain-of-thought" in learning_prompts.DIAGNOSIS

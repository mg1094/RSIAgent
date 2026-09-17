"""Phase 1 — the wave barrier.

The rules under test are the ones that make parallel exploration safe to
consolidate afterwards:

* branches share one immutable pre-wave snapshot and cannot see each other;
* verdicts are grounded before any commit;
* commits are serial, in the order the curriculum agent authored;
* each commit sees the preceding ones;
* the budget is checked *between* waves, so a wave is never truncated;
* an unresolved branch blocks the wave.
"""

from __future__ import annotations

from pathlib import Path

from protocolkit import (
    SATURATED,
    RecordingActor,
    RecordingVerifier,
    always_pass,
    brs_events,
    build_runner,
    events,
    project,
    wave,
    write_program,
)

from rsiagent.status import Decision, TerminalStatus


def test_wave_commits_in_authored_order_and_cumulatively(tmp_path: Path):
    """'Each update operates on the latest canonical memory, including preceding
    updates from the same wave.'"""
    actor = RecordingActor(
        program_for=lambda instruction: write_program("42"),
        # Each branch records one fact, so we can watch them stack up.
        memory_edit_for=lambda memory, prompt: (
            {"notes/bom.md": "ENV_FACT: csv_bom"}
            if "csv_bom" not in memory
            else {"notes/encoding.md": "ENV_FACT: encoding_detected"}
        ),
    )
    verifier = RecordingVerifier(always_pass)
    runner, journal, memory = build_runner(
        tmp_path,
        actor=actor,
        verifier=verifier,
        curriculum=[wave(("p1", "do the first thing"), ("p2", "do the second thing")), SATURATED],
    )

    result = runner.run()

    assert result.phase1.status is TerminalStatus.SATURATED
    commits = brs_events(journal, "memory_commit")
    assert [c["project"] for c in commits] == ["p1", "p2"], "commits must follow the authored order"

    # The second branch's learning saw the first branch's fact.
    assert len(actor.learning_memory_seen) >= 4  # 2 branches x (distill + reconcile)
    second_distillation = actor.learning_memory_seen[2]
    assert "csv_bom" in second_distillation, (
        "a later branch must see earlier commits from its own wave"
    )
    assert set(memory.files()) == {"notes/bom.md", "notes/encoding.md"}


def test_branches_share_one_pre_wave_snapshot(tmp_path: Path):
    """'Every project in a wave begins with the same immutable pre-wave memory
    snapshot.'  Neither branch can see the other's work while working."""
    actor = RecordingActor(
        program_for=lambda instruction: write_program("42"),
        memory_edit_for=lambda memory, prompt: {"notes/one.md": "ENV_FACT: first"},
    )
    verifier = RecordingVerifier(always_pass)
    runner, journal, memory = build_runner(
        tmp_path,
        actor=actor,
        verifier=verifier,
        curriculum=[wave(("p1", "first"), ("p2", "second")), SATURATED],
    )

    runner.run()

    # Both branches opened on the same snapshot, so neither recorded any prior
    # fact at work time.  The commits then merged them.
    opening_commits = brs_events(journal, "memory_commit")
    assert len(opening_commits) == 2
    assert memory.stats().file_count == 1


def test_budget_is_checked_between_waves_not_within_them(tmp_path: Path):
    """'The budget is checked between completed waves, without interrupting an
    ongoing wave' — so the realized count can exceed the nominal budget."""
    actor = RecordingActor(program_for=lambda instruction: write_program("42"))
    verifier = RecordingVerifier(always_pass)
    runner, journal, memory = build_runner(
        tmp_path,
        actor=actor,
        verifier=verifier,
        curriculum=[
            wave(("a1", "a"), ("a2", "a")),
            wave(("b1", "b"), ("b2", "b")),
            SATURATED,
        ],
        stage_overrides={"brs_project_budget": 3, "brs_wave_width": 2, "brs_concurrency": 1},
    )

    result = runner.run()

    # Wave 0 runs (2 projects, under budget).  Wave 1 starts because the budget
    # check happens only after wave 0 completed — and wave 1 is not truncated
    # even though it takes the count to 4, over the nominal 3.
    assert result.phase1.total_projects == 4
    assert result.phase1.status is TerminalStatus.BUDGET_EXHAUSTED
    assert len(result.phase1.waves) == 2


def test_saturation_stops_exploration(tmp_path: Path):
    actor = RecordingActor(program_for=lambda instruction: write_program("42"))
    verifier = RecordingVerifier(always_pass)
    runner, _, _ = build_runner(
        tmp_path,
        actor=actor,
        verifier=verifier,
        curriculum=[wave(("p1", "only one")), SATURATED],
    )

    result = runner.run()

    assert result.phase1.status is TerminalStatus.SATURATED
    assert len(result.phase1.waves) == 1


def test_unresolved_branch_blocks_the_wave(tmp_path: Path):
    """'An incomplete or quarantined branch blocks the wave.  Its unpublished
    memory is not merged.'"""

    def judge(out: str) -> tuple[str, str]:
        return "UNVERIFIED", "the artifact could not be resolved."

    actor = RecordingActor(
        program_for=lambda instruction: write_program("42"),
        memory_edit_for=lambda memory, prompt: {"notes/should_not_exist.md": "ENV_FACT: x"},
    )
    verifier = RecordingVerifier(judge)
    runner, journal, memory = build_runner(
        tmp_path,
        actor=actor,
        verifier=verifier,
        curriculum=[wave(("p1", "first"), ("p2", "second")), SATURATED],
    )

    result = runner.run()

    assert result.phase1.status is TerminalStatus.UNVERIFIED
    assert result.phase1.waves[0].blocked
    assert events(journal, "wave_blocked")
    assert not brs_events(journal, "memory_commit"), "a blocked wave commits nothing"
    assert memory.is_empty


def test_drop_branch_policy_commits_the_grounded_branches(tmp_path: Path):
    """The alternative to halting: keep what was grounded and record the loss."""
    state = {"n": 0}

    def judge(out: str) -> tuple[str, str]:
        # The first project to be verified resolves; the second does not.
        state["n"] += 1
        if state["n"] <= 2:
            return "UNVERIFIED", "cannot resolve this one."
        return "PASS", "fine."

    actor = RecordingActor(
        program_for=lambda instruction: write_program("42"),
        memory_edit_for=lambda memory, prompt: {"notes/kept.md": "ENV_FACT: kept"},
    )
    verifier = RecordingVerifier(judge)
    runner, journal, memory = build_runner(
        tmp_path,
        actor=actor,
        verifier=verifier,
        curriculum=[wave(("p1", "first"), ("p2", "second")), SATURATED],
        stage_overrides={"brs_on_blocked": "drop_branch", "brs_concurrency": 1},
    )

    result = runner.run()

    assert not result.phase1.waves[0].blocked
    dropped = events(journal, "branch_dropped")
    assert [d["project"] for d in dropped] == ["p1"]
    assert [c["project"] for c in brs_events(journal, "memory_commit")] == ["p2"]


def test_curriculum_decisions_are_recorded(tmp_path: Path):
    actor = RecordingActor(program_for=lambda instruction: write_program("42"))
    verifier = RecordingVerifier(always_pass)
    runner, journal, _ = build_runner(
        tmp_path,
        actor=actor,
        verifier=verifier,
        curriculum=[wave(("p1", "one")), SATURATED],
    )

    runner.run()

    decisions = [h["decision"] for h in events(journal, "curriculum_handoff")]
    assert decisions == [Decision.PROJECTS.value, Decision.SATURATED.value]


def test_second_wave_sees_first_wave_outcomes(tmp_path: Path):
    """'The next wave is selected only after these commits finish, allowing the
    curriculum agent to respond to both verified outcomes and the knowledge
    actually retained.'"""
    actor = RecordingActor(
        program_for=lambda instruction: write_program("42"),
        memory_edit_for=lambda memory, prompt: {"notes/fact.md": "ENV_FACT: learned"},
    )
    verifier = RecordingVerifier(always_pass)
    runner, journal, _ = build_runner(
        tmp_path,
        actor=actor,
        verifier=verifier,
        curriculum=[wave(("p1", "one")), wave(("p2", "two")), SATURATED],
        stage_overrides={"brs_concurrency": 1},
    )
    runner.run()

    curriculum_prompts = runner.clients.curriculum.transcript
    second_wave_prompt = curriculum_prompts[1][0][-1].content
    assert "ENV_FACT: learned" in second_wave_prompt, (
        "the second wave must be authored against committed memory"
    )
    assert "PASS" in second_wave_prompt, "and against the first wave's outcomes"


def test_drs_only_configuration_skips_phase1(tmp_path: Path):
    actor = RecordingActor(program_for=lambda instruction: write_program("42"))
    verifier = RecordingVerifier(always_pass)
    runner, journal, _ = build_runner(
        tmp_path,
        actor=actor,
        verifier=verifier,
        curriculum=[project("p1", "practice"), SATURATED],
        stage_overrides={"brs_concurrency": 1},
    )
    runner.config = runner.config.replace(stages=frozenset({"drs", "eval"}))

    result = runner.run()

    assert result.phase1 is None
    assert not events(journal, "curriculum_handoff") or all(
        h["decision"] != Decision.PROJECTS.value for h in events(journal, "curriculum_handoff")
    )

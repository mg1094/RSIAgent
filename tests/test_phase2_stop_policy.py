"""Phase 2 — the two stopping policies.

These are the rules most easily implemented wrong, because the intuitive design
("stop when the target passes") is the *non-default* one:

* under ``curriculum_review``, a target PASS is not terminal — the curriculum
  agent still reviews it, and if it selects practice, the target must be
  re-attempted afterwards;
* ``verifier_pass`` is the variant that does stop on a PASS;
* STALLED buys one final target attempt and is never recorded as success;
* an unresolved target outcome grounds no learning at all.
"""

from __future__ import annotations

from pathlib import Path

from rsiagent.status import TerminalStatus, Verdict

from protocolkit import (
    READY,
    STALLED,
    RecordingActor,
    RecordingVerifier,
    always_pass,
    build_runner,
    events,
    project,
    write_program,
)

DRS_ONLY = frozenset({"drs", "eval"})


def sequenced(*values: str):
    """A program_for that writes each value in turn, then repeats the last."""
    state = {"i": 0}

    def program_for(instruction: str) -> str:
        value = values[min(state["i"], len(values) - 1)]
        state["i"] += 1
        return write_program(value)

    return program_for


def failing(out: str) -> tuple[str, str]:
    return "FAIL", "out/answer.txt is not 42."


def unresolved(out: str) -> tuple[str, str]:
    return "UNVERIFIED", "the artifact could not be resolved."


def test_verifier_pass_stops_on_a_target_pass(tmp_path: Path):
    """'The explicit verifier_pass variant instead ends DRS after a grounded
    target PASS and memory consolidation, without a curriculum review.'"""
    actor = RecordingActor(program_for=sequenced("42"))
    runner, journal, _ = build_runner(
        tmp_path,
        actor=actor,
        verifier=RecordingVerifier(always_pass),
        curriculum=[READY],  # would be ready if asked; it must not be asked
        stage_overrides={"stop_policy": "verifier_pass"},
    )
    runner.config = runner.config.replace(stages=DRS_ONLY)

    result = runner.run()

    assert result.phase2.status is TerminalStatus.COMPLETED
    assert result.phase2.final_verdict is Verdict.PASS
    assert len(result.phase2.cycles) == 1
    assert not events(journal, "curriculum_review"), (
        "verifier_pass must not consult the curriculum agent"
    )


def test_curriculum_review_completes_after_a_pass_with_no_practice(tmp_path: Path):
    """'If the preceding target passed and no new practice was selected, finish
    DRS.'"""
    actor = RecordingActor(program_for=sequenced("42"))
    runner, journal, _ = build_runner(
        tmp_path,
        actor=actor,
        verifier=RecordingVerifier(always_pass),
        curriculum=[READY],
        stage_overrides={"stop_policy": "curriculum_review"},
    )
    runner.config = runner.config.replace(stages=DRS_ONLY)

    result = runner.run()

    assert result.phase2.status is TerminalStatus.COMPLETED
    assert len(result.phase2.cycles) == 1
    assert len(events(journal, "curriculum_review")) == 1
    assert not result.phase2.practice_projects


def test_practice_after_a_pass_forces_another_target_attempt(tmp_path: Path):
    """'Any additional practice requires another target attempt.'"""
    actor = RecordingActor(program_for=sequenced("42"))
    runner, journal, _ = build_runner(
        tmp_path,
        actor=actor,
        verifier=RecordingVerifier(always_pass),
        # Review 1 (after target PASS) asks for practice; review 2 (after the
        # practice) returns control.
        curriculum=[project("p1", "extra practice"), READY],
    )
    runner.config = runner.config.replace(stages=DRS_ONLY)

    result = runner.run()

    assert result.phase2.practice_projects == ["p1"]
    assert len(result.phase2.cycles) == 2, (
        "a target PASS followed by practice must be re-attempted"
    )
    assert result.phase2.status is TerminalStatus.COMPLETED


def test_failed_target_with_ready_review_is_reattempted(tmp_path: Path):
    """Step 6: 'Otherwise, return to Step 3.'"""
    actor = RecordingActor(program_for=sequenced("41", "42"))
    runner, journal, _ = build_runner(
        tmp_path,
        actor=actor,
        verifier=RecordingVerifier(always_pass),
        curriculum=[READY, READY],
        stage_overrides={"max_target_revisions": 0},
    )
    runner.config = runner.config.replace(stages=DRS_ONLY)

    result = runner.run()

    assert len(result.phase2.cycles) == 2
    assert result.phase2.cycles[0].verdict is Verdict.FAIL
    assert result.phase2.cycles[1].verdict is Verdict.PASS
    assert result.phase2.status is TerminalStatus.COMPLETED


def test_stalled_buys_one_final_attempt_and_is_not_success(tmp_path: Path):
    """'A curriculum STALLED decision instead allows one final target attempt
    and learning update before recording the actual terminal verdict.'"""
    actor = RecordingActor(program_for=sequenced("41", "42"))
    runner, journal, _ = build_runner(
        tmp_path,
        actor=actor,
        verifier=RecordingVerifier(always_pass),
        curriculum=[STALLED],
        stage_overrides={"max_target_revisions": 0},
    )
    runner.config = runner.config.replace(stages=DRS_ONLY)

    result = runner.run()

    # The final attempt actually passed, but the lineage still ended STALLED:
    # a stop is not convergence.
    assert result.phase2.final_verdict is Verdict.PASS
    assert result.phase2.status is TerminalStatus.STALLED
    assert result.status is TerminalStatus.STALLED, (
        "STALLED must remain distinguishable from successful convergence"
    )


def test_unverified_target_grounds_no_learning(tmp_path: Path):
    """'An unresolved outcome does not become a successful or failed learning
    example.'"""
    actor = RecordingActor(
        program_for=sequenced("42"),
        memory_edit_for=lambda memory, prompt: {"notes/should_not_exist.md": "ENV_FACT: x"},
    )
    runner, journal, memory = build_runner(
        tmp_path,
        actor=actor,
        verifier=RecordingVerifier(unresolved),
        curriculum=[READY],
        stage_overrides={"max_target_revisions": 0, "max_practice_revisions": 0},
    )
    runner.config = runner.config.replace(stages=DRS_ONLY)

    result = runner.run()

    assert result.phase2.status is TerminalStatus.UNVERIFIED
    assert result.status is TerminalStatus.UNVERIFIED
    assert not [
        c for c in events(journal, "memory_commit") if c.get("unit") == "target"
    ], "an unresolved outcome must not be consolidated"
    assert memory.is_empty


def test_practice_cap_is_enforced(tmp_path: Path):
    actor = RecordingActor(program_for=sequenced("42"))
    runner, journal, _ = build_runner(
        tmp_path,
        actor=actor,
        verifier=RecordingVerifier(always_pass),
        # The curriculum agent never stops asking for practice.
        curriculum=[project("p1", "one"), project("p2", "two"), project("p3", "three")],
        stage_overrides={"drs_max_practice_projects": 2},
    )
    runner.config = runner.config.replace(stages=DRS_ONLY)

    result = runner.run()

    assert result.phase2.status is TerminalStatus.BUDGET_EXHAUSTED
    assert len(result.phase2.practice_projects) == 2


def test_curriculum_review_after_practice_sees_the_practice_outcome(tmp_path: Path):
    """Step 5: '... actor memory update, then curriculum review.'  The review
    that follows a practice project looks at *that* project's verdict."""
    actor = RecordingActor(program_for=sequenced("41", "42"))
    runner, journal, _ = build_runner(
        tmp_path,
        actor=actor,
        verifier=RecordingVerifier(always_pass),
        curriculum=[project("p1", "practice thing"), READY],
        stage_overrides={"max_target_revisions": 0},
    )
    runner.config = runner.config.replace(stages=DRS_ONLY)

    result = runner.run()

    reviews = events(journal, "curriculum_review")
    assert len(reviews) >= 2
    assert reviews[1]["decision"] == "READY_FOR_TARGET"

    second_prompt = runner.clients.curriculum.transcript[1][0][-1].content
    assert "practice:p1" in second_prompt, "the review must name the practice unit"
    assert "**Verdict:** PASS" in second_prompt, (
        "the review after practice must see the practice's verdict, not the "
        "older target attempt's"
    )


def test_target_attempts_run_in_a_reset_environment(tmp_path: Path):
    """'... once practice returns control, the target is attempted again in a
    reset environment using the updated memory.'"""
    actor = RecordingActor(program_for=sequenced("41", "42"))
    runner, journal, _ = build_runner(
        tmp_path,
        actor=actor,
        verifier=RecordingVerifier(always_pass),
        curriculum=[project("p1", "leave a mess behind"), READY],
        stage_overrides={"max_target_revisions": 0},
    )
    runner.config = runner.config.replace(stages=DRS_ONLY)

    runner.run()

    starts = events(journal, "target_attempt_start")
    assert len(starts) == 2
    assert starts[0]["cycle"] == 0 and starts[1]["cycle"] == 1

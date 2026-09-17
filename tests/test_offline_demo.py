"""End-to-end regression on the offline demo.

This is the strongest assertion in the suite: it runs the whole lifecycle four
times, on a real environment with real subprocesses, and checks that memory
actually changes behaviour in the direction the paper describes.

If this test passes, the framework runs.  If it fails, something in the
protocol broke — not just a helper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from offline_demo.demo_env import TASK, build_evaluator  # noqa: E402
from offline_demo.offline_llm import build_clients  # noqa: E402

from rsiagent.config import ExecutionLimits, ExplorationConfig, RoleModelConfig, RunConfig  # noqa: E402
from rsiagent.env.pool import LocalEnvironmentPool  # noqa: E402
from rsiagent.memory.bank import MemoryBank  # noqa: E402
from rsiagent.rsi.protocol import RSIRunner, run_baseline  # noqa: E402
from rsiagent.runtime.journal import Journal  # noqa: E402
from rsiagent.status import TerminalStatus  # noqa: E402


def config_for(run_dir: Path, stages: frozenset[str]) -> RunConfig:
    scripted = RoleModelConfig(model="scripted", max_tokens=2048)
    return RunConfig(
        actor=scripted,
        verifier=scripted,
        curriculum=scripted,
        workspace=run_dir / "workspace",
        run_dir=run_dir,
        stages=stages,
        exploration=ExplorationConfig(
            brs_project_budget=8,
            brs_concurrency=2,
            brs_wave_width=2,
            stop_policy="curriculum_review",
            max_practice_revisions=1,
            max_target_revisions=1,
        ),
        limits=ExecutionLimits(
            target_iterations=6,
            practice_iterations=6,
            program_timeout_s=30.0,
            allow_stall_role_switch=False,
        ),
    )


class Arm:
    """A completed condition plus the artifacts it wrote."""

    def __init__(self, result, config: RunConfig) -> None:
        self.result = result
        self.config = config

    def __getattr__(self, name):
        # Delegate to the result so tests can read `.partial`, `.phase2`, ...
        return getattr(self.result, name)

    @property
    def memory(self) -> MemoryBank:
        return MemoryBank(self.config.memory_path)

    def facts(self) -> set[str]:
        """The ``ENV_FACT`` names this run's memory ended up carrying."""
        return {
            line.split(":", 1)[1].strip()
            for line in self.memory.render().splitlines()
            if line.startswith("ENV_FACT:")
        }


def run_arm(tmp_path: Path, name: str, stages: frozenset[str]) -> Arm:
    run_dir = tmp_path / name
    config = config_for(run_dir, stages)
    journal = Journal(run_dir / "journal.jsonl")
    runner = RSIRunner(
        config,
        TASK,
        build_clients(),
        env_factory=LocalEnvironmentPool(run_dir / "envs"),
        evaluator=build_evaluator(),
        journal=journal,
        memory=MemoryBank(config.memory_path),
    )
    result = runner.run()
    journal.close()
    return Arm(result, config)


@pytest.fixture(scope="module")
def arms(tmp_path_factory) -> dict:
    """Run all four stage conditions once and share the results."""
    tmp = tmp_path_factory.mktemp("ablation")
    return {
        "baseline": run_arm(tmp, "baseline", frozenset({"eval"})),
        "brs_only": run_arm(tmp, "brs_only", frozenset({"brs", "eval"})),
        "drs_only": run_arm(tmp, "drs_only", frozenset({"drs", "eval"})),
        "full": run_arm(tmp, "full", frozenset({"brs", "drs", "eval"})),
    }


def test_full_lifecycle_solves_the_task(arms):
    full = arms["full"]
    assert full.status is TerminalStatus.COMPLETED
    assert full.partial == pytest.approx(1.0)
    assert full.binary
    assert full.phase2.final_verdict.value == "PASS"


def test_stage_ordering_matches_the_papers_claim(arms):
    """Every exploration stage beats none, and both stages beat either alone."""
    baseline = arms["baseline"].partial
    full = arms["full"].partial

    assert baseline < arms["brs_only"].partial < full
    assert baseline < arms["drs_only"].partial < full


def test_baseline_writes_a_well_formed_but_empty_rollup(arms):
    """The w/o RSI condition still produces an artifact — it just parses nothing."""
    baseline = arms["baseline"]
    assert 0.0 < baseline.partial < 0.5
    assert baseline.score.components["report_parses"] == 1.0
    assert baseline.score.components["source_files"] == 0.0


def test_broad_exploration_learns_the_structural_conventions(arms):
    """BRS acquires the quirks visible from reading the directory."""
    facts = arms["brs_only"].facts()
    assert {"csv_bom", "csv_semicolon"} <= facts
    assert "amount_decimal_comma" not in facts, (
        "the money convention is not discoverable without attempting the rollup"
    )


def test_deep_exploration_adds_what_broad_exploration_cannot(arms):
    """DRS contributes exactly the conventions that only bite on a real attempt."""
    assert {"csv_bom", "csv_semicolon", "amount_decimal_comma", "date_dmy"} <= arms["full"].facts()


def test_target_passes_only_after_every_convention_is_known(arms):
    """The target cannot pass while any convention is unlearned — which is why
    the full run needs three target cycles and two practice projects."""
    full = arms["full"]
    verdicts = [cycle.verdict.value for cycle in full.phase2.cycles]
    assert verdicts == ["FAIL", "FAIL", "PASS"]
    assert len(full.phase2.practice_projects) == 2


def test_frozen_memory_is_verified_unchanged(arms):
    full = arms["full"]
    assert full.phase3.memory_intact
    assert not full.phase3.integrity_error
    assert full.frozen is not None
    full.memory.assert_frozen(full.frozen)  # raises if it drifted


def test_journal_records_memory_growth(arms):
    """Figure A1's series: file counts and byte totals per checkpoint."""
    from rsiagent.runtime.journal import Journal

    journal = Journal(arms["full"].config.journal_path)
    growth = journal.memory_growth()
    journal.close()

    assert [g["kind"] for g in growth] == [
        "brs_project",
        "brs_project",
        "drs_checkpoint",
        "drs_practice",
        "drs_checkpoint",
        "drs_practice",
        "drs_checkpoint",
        "freeze",
    ]
    # Memory only ever grows in this domain, and a FAIL still contributes.
    assert growth[0]["verdict"] == "FAIL"
    assert growth[-1]["file_count"] >= growth[0]["file_count"]


def test_baseline_arm_uses_the_same_harness(tmp_path: Path):
    """The w/o RSI condition is not a different agent — just no memory."""
    run_dir = tmp_path / "baseline_direct"
    config = config_for(run_dir, frozenset({"eval"})).replace(
        enable_memory=False, memory_writeback=False
    )
    journal = Journal(run_dir / "journal.jsonl")
    result = run_baseline(
        config,
        TASK,
        build_clients(),
        env_factory=LocalEnvironmentPool(run_dir / "envs"),
        evaluator=build_evaluator(),
        journal=journal,
    )
    journal.close()

    assert result.status is TerminalStatus.COMPLETED
    assert 0.0 < result.partial < 0.5
    assert result.verdict.value == "FAIL"

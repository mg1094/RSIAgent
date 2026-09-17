"""Run the RSIAgent lifecycle offline, end to end, with no API key.

    python examples/offline_demo/run.py --arm both

This executes the real protocol: real subprocesses, real artifacts, real
verifier probes, real memory commits, and a real evaluator reading the artifact
the actor produced.  Only the four model roles are stand-ins (see
``offline_llm.py``).  The score it prints demonstrates the *protocol*, not the
capability of any model.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil

# Allow running this file directly, without installing the package.
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))  # repo root, for `rsiagent`
sys.path.insert(0, str(_HERE.parent))  # examples/, for `offline_demo`

from offline_demo.demo_env import TASK, build_evaluator
from offline_demo.offline_llm import build_clients
from rsiagent.config import ExecutionLimits, ExplorationConfig, RoleModelConfig, RunConfig
from rsiagent.env.pool import LocalEnvironmentPool
from rsiagent.memory.bank import MemoryBank
from rsiagent.rsi.protocol import RSIRunner, run_baseline
from rsiagent.runtime.journal import Journal

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_config(run_dir: Path, *, stages: frozenset[str]) -> RunConfig:
    """The paper's shape, scaled to a laptop.

    Budgets are small so the demo finishes in seconds.  Everything structural —
    the wave barrier, the practice cap, the stopping policy, the stage-ablation
    switch — is the same setting the manuscript uses.
    """
    scripted = RoleModelConfig(model="scripted", max_tokens=4096)
    return RunConfig(
        actor=dataclasses.replace(scripted, model="scripted-actor"),
        verifier=dataclasses.replace(scripted, model="scripted-verifier"),
        curriculum=dataclasses.replace(scripted, model="scripted-curriculum"),
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


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def run_baseline_arm(config: RunConfig):
    """The ``w/o RSI`` condition: same harness, no exploration, no memory."""
    run_dir = config.run_dir
    shutil.rmtree(run_dir, ignore_errors=True)
    env_factory = LocalEnvironmentPool(run_dir / "envs")
    journal = Journal(run_dir / "journal.jsonl")

    result = run_baseline(
        config,
        TASK,
        build_clients(),
        env_factory=env_factory,
        evaluator=build_evaluator(),
        journal=journal,
    )
    journal.close()
    print(result.summary())
    print(f"\n  journal: {run_dir / 'journal.jsonl'}")
    return result


PHASE_LABELS = {
    "phase1_brs": "Phase 1 — Broad Recursive Self-exploration",
    "phase2_drs": "Phase 2 — Deep Recursive Self-exploration",
    "phase3_eval": "Phase 3 — frozen-memory evaluation",
}

ARMS = {
    "baseline": (frozenset({"eval"}), "RSIAgent (w/o RSI)"),
    "brs-only": (frozenset({"brs", "eval"}), "RSIAgent (w/o DRS)"),
    "drs-only": (frozenset({"drs", "eval"}), "RSIAgent (w/o BRS)"),
    "full": (frozenset({"brs", "drs", "eval"}), "RSIAgent (Full RSI)"),
}


def run_rsi_arm(config: RunConfig, *, trace: bool = True):
    """One RSI condition.  ``config.stages`` selects which stages run."""
    run_dir = config.run_dir
    shutil.rmtree(run_dir, ignore_errors=True)
    env_factory = LocalEnvironmentPool(run_dir / "envs")

    seen_phase = {"value": None}

    def narrate(event) -> None:
        if not trace:
            return
        if event.phase and event.phase != seen_phase["value"]:
            seen_phase["value"] = event.phase
            label = PHASE_LABELS.get(event.phase, event.phase)
            print(f"\n--- {label} " + "-" * max(0, 52 - len(label)))
        _print_event(event.event, event.data)

    journal = Journal(run_dir / "journal.jsonl", on_event=narrate)

    runner = RSIRunner(
        config,
        TASK,
        build_clients(),
        env_factory=env_factory,
        evaluator=build_evaluator(),
        journal=journal,
        memory=MemoryBank(config.memory_path),
    )

    result = runner.run()
    journal.close()

    banner("RESULT")
    print(result.summary())

    print("\n--- memory growth " + "-" * 55)
    for record in journal.memory_growth():
        print(
            f"  {record['phase']:<12} {record['kind']:<14} {record['label']:<22} "
            f"{record['verdict'] or '-':<6} {record['file_count']:>2} files "
            f"{record['total_bytes']:>6} bytes"
        )

    print(f"\n--- final memory {memory_stats_line(result)}" + "-" * 40)
    for path, blob in sorted(MemoryBank(config.memory_path).files().items()):
        first_line = blob.decode("utf-8", "replace").splitlines()[0]
        print(f"  {path:<28} {first_line}")

    print(f"\n  journal: {journal.path}")
    print(f"  frozen:  {result.frozen.path if result.frozen else '-'}")
    return result


def memory_stats_line(result) -> str:
    if not result.memory:
        return "\n"
    return (
        f" ({result.memory.file_count} files, {result.memory.total_bytes} bytes, "
        f"hash {result.memory.tree_hash[:12]})\n"
    )


def _print_event(name: str, data: dict) -> None:
    if name == "curriculum_handoff":
        print(
            f"  [curriculum] wave {data['wave']}: {data['decision']} {data.get('projects') or ''}"
        )
    elif name == "project_start":
        print(f"    [project] {data['project']}")
    elif name == "project_verdict":
        print(
            f"    [verdict] {data['project']}: {data['verdict']} "
            f"({data.get('programs', 0)} programs)"
        )
    elif name == "wave_committed":
        print(f"  [wave {data['wave']}] committed {data['committed']}")
    elif name == "target_verdict":
        print(f"  [target cycle {data['cycle']}] {data['verdict']}")
    elif name == "curriculum_review":
        print(f"  [curriculum] review -> {data['decision']}")
    elif name == "practice_start":
        print(f"    [practice] {data['project']}")
    elif name == "practice_verdict":
        print(f"    [verdict] {data['project']}: {data['verdict']}")
    elif name == "memory_frozen":
        print(f"  [frozen] {data['file_count']} files, hash {data['tree_hash'][:12]}")
    elif name == "official_score":
        print(
            f"  [official score] partial={data['partial']:.4f} "
            f"binary={data['binary']} (agent verdict {data['agent_verdict']})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=("baseline", "rsi", "full", "brs-only", "drs-only", "ablation", "all"),
        default="all",
        help="'all' runs the full stage ablation; 'rsi'/'full' run Full RSI alone",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "runs" / "offline_demo",
        help="where journals, memory, and environments are written",
    )
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    scores: dict[str, float] = {}

    if args.arm in ("baseline", "ablation", "all"):
        banner("ARM: w/o RSI — same harness, exploration and memory disabled")
        baseline = run_baseline_arm(build_config(out / "baseline", stages=ARMS["baseline"][0]))
        scores[ARMS["baseline"][1]] = baseline.partial

    if args.arm in ("brs-only", "ablation", "all"):
        banner("ARM: w/o DRS — evaluate the memory Broad exploration acquired, as-is")
        result = run_rsi_arm(
            build_config(out / "brs_only", stages=ARMS["brs-only"][0]), trace=False
        )
        print(result.summary())
        scores[ARMS["brs-only"][1]] = result.partial

    if args.arm in ("drs-only", "ablation", "all"):
        banner("ARM: w/o BRS — Deep exploration starting from empty memory")
        result = run_rsi_arm(
            build_config(out / "drs_only", stages=ARMS["drs-only"][0]), trace=False
        )
        print(result.summary())
        scores[ARMS["drs-only"][1]] = result.partial

    if args.arm in ("rsi", "full", "ablation", "all"):
        banner(
            "ARM: Full RSI — Broad exploration, then Deep refinement, then "
            "frozen-memory evaluation\n"
            "(the trace below is the whole lifecycle)"
        )
        result = run_rsi_arm(build_config(out / "full", stages=ARMS["full"][0]))
        scores[ARMS["full"][1]] = result.partial

    if len(scores) > 1:
        banner("STAGE COMPARISON")
        print(f"  {'condition':<26} {'partial':>9}  {'out of 7':>9}")
        print(f"  {'-' * 26} {'-' * 9}  {'-' * 9}")
        for condition, partial in scores.items():
            print(f"  {condition:<26} {partial:>9.4f}  {partial * 7:>6.2f}/7")
        print(
            "\n  What this shows: every exploration stage beats no exploration, and the\n"
            "  two-stage lifecycle beats either stage alone.\n"
            "\n  What it does not show: the paper's exact ordering. Its w/o-BRS condition\n"
            "  falls below baseline on two of four tasks; here it does not, because the\n"
            "  stand-in curriculum happens to queue the two hardest facts. The demo is\n"
            "  evidence that the protocol runs and that memory changes behaviour — not\n"
            "  evidence about any model's capability."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

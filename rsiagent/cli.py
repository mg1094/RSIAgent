"""Command line interface.

    rsiagent demo                 # run the offline demo, no API key needed
    rsiagent run --task t.py:task # run a lineage against a task
    rsiagent inspect runs/latest  # summarise a finished run
    rsiagent memory runs/latest   # print the frozen memory

``run`` loads a task from a Python file so a benchmark adapter can live outside
this package: point ``--task`` at a module path and an attribute name.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

from .config import RunConfig, load_config
from .env.pool import LocalEnvironmentPool
from .errors import RSIAgentError
from .llm.factory import RoleClients, build_clients
from .llm.scripted import ScriptedClient
from .memory.bank import MemoryBank
from .rsi.protocol import RSIRunner, run_baseline
from .runtime.journal import Journal, read_journal
from .tasks.spec import Evaluator, TaskQuery, UnscoredEvaluator

STAGE_SETS = {
    "baseline": frozenset({"eval"}),
    "brs-only": frozenset({"brs", "eval"}),
    "drs-only": frozenset({"drs", "eval"}),
    "rsi": frozenset({"brs", "drs", "eval"}),
    "full": frozenset({"brs", "drs", "eval"}),
}


# --------------------------------------------------------------------------
def _load_object(spec: str):
    """Load ``path/to/file.py:attribute``."""
    if ":" not in spec:
        raise RSIAgentError(f"--task must be 'path/to/file.py:attribute', got {spec!r}")
    path_text, attribute = spec.rsplit(":", 1)
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise RSIAgentError(f"task file not found: {path}")

    module_spec = importlib.util.spec_from_file_location(f"_rsiagent_task_{path.stem}", path)
    if module_spec is None or module_spec.loader is None:  # pragma: no cover
        raise RSIAgentError(f"could not load {path}")
    module = importlib.util.module_from_spec(module_spec)

    # A task file often imports a sibling helper; make its directory importable.
    sys.path.insert(0, str(path.parent))
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise RSIAgentError(f"{path} has no attribute {attribute!r}") from exc


def _resolve_task(spec: str) -> TaskQuery:
    task = _load_object(spec)
    if not isinstance(task, TaskQuery):
        raise RSIAgentError(f"--task must resolve to a TaskQuery, got {type(task).__name__}")
    return task


def _resolve_evaluator(spec: str | None) -> Evaluator:
    if not spec:
        return UnscoredEvaluator("no --evaluator supplied")
    evaluator = _load_object(spec)
    if not hasattr(evaluator, "score"):
        raise RSIAgentError("--evaluator must resolve to an object with a score() method")
    return evaluator


def _resolve_clients(spec: str | None, config: RunConfig, provider: str) -> RoleClients:
    """Build the role clients, optionally from a user-supplied policy module.

    ``--policies path/to/policies.py:attribute`` accepts either a ready
    :class:`RoleClients` or a plain mapping::

        policies = {
            "actor": lambda messages: "```python\\n...\\n```",
            "verifier": lambda messages: "VERDICT: PASS\\nFINDINGS: ...",
            "curriculum": lambda messages: '{"decision": "SATURATED", ...}',
        }

    The mapping form is the one to reach for when experimenting: it lets the
    whole lifecycle run offline, with no API key, against a real environment.
    """
    if spec is None:
        return build_clients(config, provider=provider)

    supplied = _load_object(spec)
    if isinstance(supplied, RoleClients):
        return supplied
    if isinstance(supplied, dict):
        missing = {"actor", "verifier", "curriculum"} - set(supplied)
        if missing:
            raise RSIAgentError(f"--policies mapping is missing: {sorted(missing)}")
        return RoleClients(
            actor=ScriptedClient(supplied["actor"], name="policies:actor"),
            verifier=ScriptedClient(supplied["verifier"], name="policies:verifier"),
            curriculum=ScriptedClient(supplied["curriculum"], name="policies:curriculum"),
            observer=(
                ScriptedClient(supplied["observer"], name="policies:observer")
                if supplied.get("observer")
                else None
            ),
        )
    raise RSIAgentError(
        "--policies must resolve to a RoleClients or a mapping with "
        "actor/verifier/curriculum entries"
    )


# --------------------------------------------------------------------------
def cmd_demo(args: argparse.Namespace) -> int:
    """Run the bundled offline demo.

    The demo lives under ``examples/``; it is imported rather than duplicated so
    there is exactly one copy of the walkthrough.
    """
    import subprocess

    script = Path(__file__).resolve().parents[1] / "examples" / "offline_demo" / "run.py"
    if not script.exists():
        print("offline demo not found; run from a source checkout", file=sys.stderr)
        return 2
    command = [sys.executable, str(script), "--arm", args.arm, "--out", str(args.out)]
    return subprocess.call(command)


def cmd_run(args: argparse.Namespace) -> int:
    config: RunConfig = load_config(args.config) if args.config else _default_config(args)
    task = _resolve_task(args.task)
    evaluator = _resolve_evaluator(args.evaluator)

    config = config.replace(stages=STAGE_SETS[args.arm])
    if args.arm == "baseline":
        config = dataclasses.replace(config, enable_memory=False, memory_writeback=False)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    clients = _resolve_clients(args.policies, config, args.provider)
    env_factory = LocalEnvironmentPool(
        config.run_dir / "envs", program_timeout_s=config.limits.program_timeout_s
    )
    journal = Journal(config.run_dir / "journal.jsonl", on_event=_narrator(args.quiet))

    if args.arm == "baseline":
        result = run_baseline(
            config, task, clients, env_factory=env_factory, evaluator=evaluator, journal=journal
        )
        journal.close()
        print(result.summary())
        return 0

    runner = RSIRunner(
        config,
        task,
        clients,
        env_factory=env_factory,
        evaluator=evaluator,
        journal=journal,
        memory=MemoryBank(config.memory_path),
    )
    result = runner.run()
    journal.close()

    print()
    print(result.summary())
    if args.json:
        print(
            json.dumps(
                {
                    "task": task.id,
                    "status": result.status.value,
                    "partial": result.partial,
                    "binary": result.binary,
                    "memory": result.memory.as_dict() if result.memory else None,
                },
                indent=2,
            )
        )
    return 0 if result.status.is_scored else 1


def _default_config(args: argparse.Namespace) -> RunConfig:
    from .config import ExecutionLimits, ExplorationConfig, RoleModelConfig

    run_dir = Path(args.out).expanduser().resolve()
    return RunConfig(
        actor=RoleModelConfig(model=args.actor_model),
        verifier=RoleModelConfig(model=args.verifier_model),
        curriculum=RoleModelConfig(model=args.curriculum_model),
        workspace=run_dir / "workspace",
        run_dir=run_dir,
        exploration=ExplorationConfig(),
        limits=ExecutionLimits(program_timeout_s=args.program_timeout),
    )


def _narrator(quiet: bool):
    if quiet:
        return None

    def narrate(event) -> None:
        interesting = {
            "curriculum_handoff",
            "project_verdict",
            "wave_committed",
            "wave_blocked",
            "target_verdict",
            "curriculum_review",
            "practice_verdict",
            "memory_frozen",
            "official_score",
        }
        if event.event not in interesting:
            return
        detail = event.data
        if event.event == "curriculum_handoff":
            print(
                f"[{event.phase}] curriculum -> {detail['decision']} {detail.get('projects', '')}"
            )
        elif event.event in ("project_verdict", "practice_verdict"):
            print(f"  {detail.get('project')}: {detail.get('verdict')}")
        elif event.event == "target_verdict":
            print(f"[{event.phase}] target cycle {detail['cycle']}: {detail['verdict']}")
        elif event.event == "curriculum_review":
            print(f"  curriculum -> {detail['decision']}")
        elif event.event == "official_score":
            print(f"[eval] partial={detail['partial']:.4f} binary={detail['binary']}")
        else:
            print(f"[{event.phase}] {event.event}")

    return narrate


# --------------------------------------------------------------------------
def cmd_inspect(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    journal_path = run_dir / "journal.jsonl"
    if not journal_path.exists():
        print(f"no journal at {journal_path}", file=sys.stderr)
        return 2

    events = read_journal(journal_path)
    print(f"journal: {journal_path}")
    print(f"events:  {len(events)}\n")

    counts: dict[str, int] = {}
    for event in events:
        counts[event.event] = counts.get(event.event, 0) + 1
    for name, count in sorted(counts.items()):
        print(f"  {name:<26} {count:>4}")

    starts = [e for e in events if e.event == "run_start"]
    completes = [e for e in events if e.event == "run_complete"]
    if starts:
        print(f"\ntask: {starts[0].data.get('task')}")
    if completes:
        blob = completes[-1].data
        print(f"status: {blob.get('status')}")
        if blob.get("score"):
            print(f"score:  partial={blob['score']['partial']} binary={blob['score']['binary']}")
        memory = blob.get("memory") or {}
        if memory:
            print(f"memory: {memory.get('file_count')} files, {memory.get('total_bytes')} bytes")

    checkpoints = [e for e in events if e.event == "memory_checkpoint"]
    if checkpoints:
        print("\nmemory growth:")
        for entry in checkpoints:
            print(
                f"  {entry.phase:<12} {entry.data['kind']:<14} {entry.data['label']:<24} "
                f"{entry.data.get('verdict') or '-':<6} "
                f"{entry.data['file_count']:>3} files {entry.data['total_bytes']:>7} bytes"
            )
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    frozen = run_dir / "frozen_memory"
    root = frozen if (frozen.exists() and not args.live) else run_dir / "memory"
    if not root.exists():
        print(f"no memory at {root}", file=sys.stderr)
        return 2

    bank = MemoryBank(root, writeback=False)
    stats = bank.stats()
    print(f"{root}  ({stats.file_count} files, {stats.total_bytes} bytes)")
    print(f"tree hash: {stats.tree_hash}\n")
    for path, blob in sorted(bank.files().items()):
        print(f"--- {path} " + "-" * max(0, 60 - len(path)))
        print(blob.decode("utf-8", "replace").rstrip())
        print()
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except RSIAgentError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    print("configuration is valid")
    print(json.dumps(config.to_dict(), indent=2))
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsiagent",
        description="Autonomous exploration for recursive self-improvement.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="run the offline demo (no API key needed)")
    demo.add_argument(
        "--arm", default="all", choices=("baseline", "rsi", "brs-only", "drs-only", "all")
    )
    demo.add_argument("--out", type=Path, default=Path("runs/offline_demo"))
    demo.set_defaults(func=cmd_demo)

    run = sub.add_parser("run", help="run an RSI lineage against a task")
    run.add_argument("--task", required=True, help="path/to/task.py:attribute")
    run.add_argument("--evaluator", default=None, help="path/to/evaluator.py:attribute")
    run.add_argument("--config", default=None, help="JSON or YAML run configuration")
    run.add_argument("--arm", default="rsi", choices=tuple(STAGE_SETS))
    run.add_argument("--out", default="runs/latest", help="run directory (without --config)")
    run.add_argument("--provider", default="auto", choices=("auto", "openai", "anthropic"))
    run.add_argument(
        "--policies",
        default=None,
        help="path/to/policies.py:attribute supplying role clients; runs offline",
    )
    run.add_argument("--actor-model", default="glm-5.3")
    run.add_argument("--verifier-model", default="kimi-k3")
    run.add_argument("--curriculum-model", default="kimi-k3")
    run.add_argument("--program-timeout", type=float, default=600.0)
    run.add_argument("--quiet", action="store_true", help="suppress the live trace")
    run.add_argument("--json", action="store_true", help="also print a JSON result")
    run.set_defaults(func=cmd_run)

    inspect = sub.add_parser("inspect", help="summarise a finished run")
    inspect.add_argument("run_dir")
    inspect.set_defaults(func=cmd_inspect)

    memory = sub.add_parser("memory", help="print a run's memory")
    memory.add_argument("run_dir")
    memory.add_argument(
        "--live", action="store_true", help="show canonical memory, not the frozen copy"
    )
    memory.set_defaults(func=cmd_memory)

    validate = sub.add_parser("validate", help="check a run configuration")
    validate.add_argument("config")
    validate.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(f"rsiagent {__version__}")
        return 0

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except RSIAgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

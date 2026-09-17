"""Helpers for driving the whole lifecycle with scripted policies.

These build a real :class:`~rsiagent.rsi.protocol.RSIRunner` — real
environments, real subprocesses, real memory — and only substitute the model
transport.  Protocol tests written against this exercise the same code paths a
benchmark run takes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from rsiagent.config import ExecutionLimits, ExplorationConfig, RoleModelConfig, RunConfig
from rsiagent.env.pool import LocalEnvironmentPool
from rsiagent.llm import Message, RoleClients, ScriptedClient
from rsiagent.memory.bank import MemoryBank
from rsiagent.rsi.protocol import RSIRunner
from rsiagent.runtime.journal import Journal, read_journal
from rsiagent.tasks.spec import TaskQuery, TaskScore

# --------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------
TASK = TaskQuery(
    id="answer",
    instruction="Write the number 42 into `out/answer.txt`.",
    fixtures={"data/input.txt": "nothing useful here\n"},
)


def answer_score(environment) -> TaskScore:
    if not environment.exists("out/answer.txt"):
        return TaskScore(partial=0.0, components={"answer": 0.0}, notes="no artifact")
    body = environment.read_text("out/answer.txt").strip()
    return TaskScore(
        partial=1.0 if body == "42" else 0.0,
        components={"answer": 1.0 if body == "42" else 0.0},
        notes=f"out/answer.txt = {body!r}",
    )


class FixedEvaluator:
    name = "fixed"

    def score(self, environment) -> TaskScore:
        return answer_score(environment)


# --------------------------------------------------------------------------
# Curriculum scripts
# --------------------------------------------------------------------------
def wave(*projects: tuple[str, str]) -> str:
    return json.dumps(
        {
            "decision": "PROJECTS",
            "rationale": "test wave",
            "projects": [{"id": pid, "instruction": instruction} for pid, instruction in projects],
        }
    )


def project(pid: str, instruction: str) -> str:
    return json.dumps(
        {
            "decision": "PROJECT",
            "rationale": "test project",
            "project": {"id": pid, "instruction": instruction},
        }
    )


SATURATED = json.dumps({"decision": "SATURATED", "rationale": "nothing left"})
STALLED = json.dumps({"decision": "STALLED", "rationale": "cannot author anything"})
READY = json.dumps({"decision": "READY_FOR_TARGET", "rationale": "ready"})


# --------------------------------------------------------------------------
# Policy builders
# --------------------------------------------------------------------------
def write_program(value: str, path: str = "out/answer.txt") -> str:
    directory = path.rsplit("/", 1)[0] if "/" in path else "."
    return (
        "import os\n"
        f"os.makedirs({directory!r}, exist_ok=True)\n"
        f"open({path!r}, 'w').write({value!r})\n"
    )


def fence(program: str) -> str:
    return f"```python\n{program}\n```"


class RecordingActor:
    """An actor policy that writes a program per instruction.

    Also handles the three learning prompts, because in a real run the actor
    client serves both work and consolidation — the whole point of
    experience-owned learning.
    """

    def __init__(
        self,
        program_for: Callable[[str], str | None],
        *,
        memory_edit_for: Callable[[str, str], dict[str, str]] | None = None,
        revision_program_for: Callable[[str, str], str | None] | None = None,
    ) -> None:
        self.program_for = program_for
        self.memory_edit_for = memory_edit_for
        self.revision_program_for = revision_program_for
        self.work_instructions: list[str] = []
        self.learning_memory_seen: list[str] = []
        self.distillations: list[str] = []

    def __call__(self, messages: Sequence[Message]) -> str:
        last = messages[-1].content

        if "Learning step 1 of 2" in last:
            memory = _section(last, "## Current memory")
            self.learning_memory_seen.append(memory)
            self.distillations.append(last)
            edits = self.memory_edit_for(memory, last) if self.memory_edit_for else {}
            blocks = [f"```memory:write {path}\n{body}\n```" for path, body in edits.items()]
            blocks.append("ACTION: done\nSUMMARY: consolidated")
            return "\n\n".join(blocks)

        if "Learning step 2 of 2" in last:
            self.learning_memory_seen.append(_section(last, "## Memory as it now stands"))
            return "ACTION: done\nSUMMARY: reconciled"

        if "Learning diagnosis" in last:
            return "Format conventions were the repeated obstacle.\n\nACTION: done"

        if last.lstrip().startswith("$ ("):
            return "ACTION: done\nSUMMARY: program finished"

        if "## Verifier findings" in last or "UNVERIFIED" in last:
            if self.revision_program_for is not None:
                program = self.revision_program_for(last, self.work_instructions[-1])
                if program:
                    return fence(program)
            return "ACTION: done\nSUMMARY: no repair available"

        # A fresh work turn: the charter carries the instruction.
        instruction = _instruction_from(last)
        self.work_instructions.append(instruction)
        program = self.program_for(instruction)
        if program is None:
            return "ACTION: done\nSUMMARY: nothing to do"
        return fence(program)


class RecordingVerifier:
    """A verifier that probes the artifact and judges by a rule."""

    def __init__(self, judge: Callable[[str], tuple[str, str]]) -> None:
        self.judge = judge
        self.requests: list[str] = []

    def __call__(self, messages: Sequence[Message]) -> str:
        last = messages[-1].content
        if not last.lstrip().startswith("$ ("):
            self.requests.append(last)
            return fence("print(open('out/answer.txt').read())")
        status, findings = self.judge(last)
        return f"VERDICT: {status}\nFINDINGS: {findings}"


def _section(text: str, heading: str) -> str:
    if heading not in text:
        return ""
    body = text.split(heading, 1)[1]
    for stop in ("## End of memory", "## How to", "## Memory as it stands"):
        if stop in body:
            body = body.split(stop, 1)[0]
    return body


def _instruction_from(charter: str) -> str:
    for marker in ("**Instruction:**", "**Instruction (authoritative):**"):
        if marker in charter:
            return charter.split(marker, 1)[1].split("\n", 1)[0].strip()
    return charter.strip().splitlines()[0] if charter.strip() else ""


# --------------------------------------------------------------------------
# Runner construction
# --------------------------------------------------------------------------
def build_config(run_dir: Path, **exploration_overrides) -> RunConfig:
    scripted = RoleModelConfig(model="scripted", max_tokens=2048)
    return RunConfig(
        actor=scripted,
        verifier=scripted,
        curriculum=scripted,
        workspace=run_dir / "workspace",
        run_dir=run_dir,
        exploration=ExplorationConfig(**exploration_overrides),
        limits=ExecutionLimits(
            target_iterations=3,
            practice_iterations=3,
            program_timeout_s=20.0,
            allow_stall_role_switch=False,
        ),
    )


def build_runner(
    tmp_path: Path,
    *,
    actor: RecordingActor,
    verifier: RecordingVerifier,
    curriculum: Iterable[str],
    task: TaskQuery = TASK,
    stage_overrides: dict | None = None,
):
    """Assemble a runner plus the journal and memory it will use."""
    config = build_config(tmp_path, **(stage_overrides or {}))
    config.run_dir.mkdir(parents=True, exist_ok=True)

    # Replay in order, repeating the final entry once the script runs out — so a
    # script must end with a terminal decision or the lineage will loop.
    clients = RoleClients(
        actor=ScriptedClient(actor, name="actor"),
        verifier=ScriptedClient(verifier, name="verifier"),
        curriculum=ScriptedClient(replay=list(curriculum), name="curriculum"),
    )
    journal = Journal(config.run_dir / "journal.jsonl")
    memory = MemoryBank(config.memory_path)
    runner = RSIRunner(
        config,
        task,
        clients,
        env_factory=LocalEnvironmentPool(config.run_dir / "envs"),
        evaluator=FixedEvaluator(),
        journal=journal,
        memory=memory,
    )
    return runner, journal, memory


def events(journal: Journal, name: str) -> list[dict]:
    return [e.data for e in read_events(journal) if e.event == name]


def brs_events(journal: Journal, name: str) -> list[dict]:
    """Events from Phase 1 only.

    ``memory_commit`` is emitted by both phases with different payloads — BRS
    commits carry ``project``, DRS commits carry ``unit`` — so tests that care
    about the wave barrier must filter.
    """
    return [e.data for e in read_events(journal) if e.event == name and e.phase == "phase1_brs"]


def read_events(journal: Journal):
    return read_journal(journal.path)


def always_pass(out: str) -> tuple[str, str]:
    body = out.split("\n", 1)[-1].strip()
    return ("PASS", "contains 42.") if body == "42" else ("FAIL", f"contains {body!r}.")

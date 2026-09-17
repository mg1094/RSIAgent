"""Shared fixtures.

Every test here runs offline: no API key, no network, no VM.  The environments
are real directories and the programs really execute, so the protocol paths
being tested are the same ones a benchmark run would take.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rsiagent.agents.actor import ActorAgent  # noqa: E402
from rsiagent.agents.context import ContextFactory  # noqa: E402
from rsiagent.agents.verifier import VerifierAgent  # noqa: E402
from rsiagent.env.local import LocalWorkspaceEnvironment  # noqa: E402
from rsiagent.llm.scripted import ScriptedClient  # noqa: E402
from rsiagent.prompts import actor as actor_prompts  # noqa: E402
from rsiagent.prompts import verifier as verifier_prompts  # noqa: E402
from rsiagent.runtime.harness import TaskHarness  # noqa: E402
from rsiagent.tasks.spec import ScriptedEvaluator, TaskQuery, provision  # noqa: E402


@pytest.fixture
def workspace(tmp_path: Path) -> LocalWorkspaceEnvironment:
    return LocalWorkspaceEnvironment(tmp_path / "workspace", name="test")


@pytest.fixture
def actor() -> ActorAgent:
    return ActorAgent()


@pytest.fixture
def verifier() -> VerifierAgent:
    return VerifierAgent()


@pytest.fixture
def actor_factory():
    def build(handler, system_prompt: str = actor_prompts.ACTOR_SYSTEM) -> ContextFactory:
        return ContextFactory(ScriptedClient(handler), system_prompt, role="actor")

    return build


@pytest.fixture
def verifier_factory():
    def build(handler, system_prompt: str = verifier_prompts.PRACTICE_VERIFIER_SYSTEM):
        return ContextFactory(ScriptedClient(handler), system_prompt, role="verifier")

    return build


@pytest.fixture
def harness(actor: ActorAgent, verifier: VerifierAgent) -> TaskHarness:
    return TaskHarness(actor, verifier, max_revisions=1)


@pytest.fixture
def simple_task() -> TaskQuery:
    """A one-file task: write ``out/answer.txt`` containing ``42``."""
    return TaskQuery(
        id="write-42",
        instruction="Write the number 42 into `out/answer.txt`.",
        fixtures={"data/input.txt": "the answer is not here\n"},
    )


@pytest.fixture
def answer_evaluator() -> ScriptedEvaluator:
    def has_answer(environment) -> bool:
        return environment.exists("out/answer.txt") and "42" in environment.read_text(
            "out/answer.txt"
        )

    return ScriptedEvaluator({"answer": has_answer})


def provisioned(environment: LocalWorkspaceEnvironment, task: TaskQuery):
    provision(environment, task.fixtures, reset=True)
    return environment

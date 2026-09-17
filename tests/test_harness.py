"""The actor-verifier harness.

The load-bearing property here is that verification cannot contaminate the
candidate: "the harness restores that checkpoint before the candidate is graded,
so none of your in-VM effects enter the scored state."
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from rsiagent.agents.actor import ActorAgent
from rsiagent.agents.context import ContextFactory
from rsiagent.env.local import LocalWorkspaceEnvironment
from rsiagent.llm.scripted import ScriptedClient
from rsiagent.prompts import actor as actor_prompts
from rsiagent.prompts import verifier as verifier_prompts
from rsiagent.tasks.spec import TaskQuery, provision


def _fence(program: str) -> str:
    return f"```python\n{program}\n```"


def writer(value: str) -> str:
    """A program that writes ``value`` to out/answer.txt, creating out/ first.

    ``out/`` does not exist in the fixture, and a silently failing program is
    indistinguishable from a wrong one — which is exactly the confusion the
    verifier tests must not inherit.
    """
    return (
        "import os\n"
        "os.makedirs('out', exist_ok=True)\n"
        f"open('out/answer.txt','w').write({value!r})\n"
    )


def reader() -> str:
    return "print(open('out/answer.txt').read())\n"


def actor_context(handler: Callable[[Sequence], str]):
    return ContextFactory(ScriptedClient(handler), actor_prompts.ACTOR_SYSTEM, role="actor")()


def verifier_context(handler: Callable[[Sequence], str]):
    return ContextFactory(
        ScriptedClient(handler), verifier_prompts.PRACTICE_VERIFIER_SYSTEM, role="verifier"
    )()


def acting(*programs: str, summary: str = "finished") -> Callable[[Sequence], str]:
    """A handler that submits each program in turn, then declares done.

    Successive calls continue through ``programs``, so a handler shared across
    revisions submits a different program on the second attempt.
    """
    state = {"n": 0}

    def handler(messages):
        last = messages[-1].content
        if last.lstrip().startswith("$ (") or last.startswith("[observation]"):
            return f"ACTION: done\nSUMMARY: {summary}"
        program = programs[min(state["n"], len(programs) - 1)]
        state["n"] += 1
        return _fence(program)

    return handler


def probing(judge: Callable[[str], tuple[str, str]], probe: str) -> Callable[[Sequence], str]:
    """A handler that runs one probe, then judges from the probe's output."""

    def handler(messages):
        last = messages[-1].content
        if not last.lstrip().startswith("$ ("):
            return _fence(probe)
        status, findings = judge(last)
        return f"VERDICT: {status}\nFINDINGS: {findings}"

    return handler


def _attempt(harness, workspace, context_pair, task, **kwargs):
    actor_ctx, verifier_ctx = context_pair
    return harness.attempt(
        workspace,
        actor_context=actor_ctx,
        verifier_context=verifier_ctx,
        instruction=task.instruction,
        memory_text=None,
        iteration_limit=3,
        **kwargs,
    )


# --------------------------------------------------------------------------
def test_verifier_probes_are_rolled_back(
    harness, workspace: LocalWorkspaceEnvironment, simple_task: TaskQuery
):
    """A verifier that mutates the environment must leave no trace."""
    provision(workspace, simple_task.fixtures, reset=True)

    probe = reader() + "open('out/answer.txt','w').write('PROBE WAS HERE')"

    def judge(out: str) -> tuple[str, str]:
        return "PASS", "out/answer.txt contains 42."

    result = _attempt(
        harness,
        workspace,
        (
            actor_context(acting(writer("42"))),
            verifier_context(probing(judge, probe)),
        ),
        simple_task,
    )

    assert result.passed
    assert workspace.read_text("out/answer.txt") == "42", (
        "the verifier's write must have been rolled back"
    )


def test_verifier_sees_the_candidate_but_not_the_actor(
    harness, workspace: LocalWorkspaceEnvironment, simple_task: TaskQuery
):
    """The verifier's context must never contain the actor's transcript."""
    provision(workspace, simple_task.fixtures, reset=True)

    actor_secret = "SECRET-ACTOR-REASONING"
    seen: list[str] = []

    def verifier_handler(messages):
        seen.append("\n".join(m.content for m in messages))
        return probing(lambda out: ("PASS", "ok"), reader())(messages)

    def actor_handler(messages):
        # The actor narrates something private before acting.
        if len(messages) == 2:
            return f"My private plan: {actor_secret}\n" + _fence(writer("42"))
        return "ACTION: done\nSUMMARY: done"

    _attempt(
        harness,
        workspace,
        (actor_context(actor_handler), verifier_context(verifier_handler)),
        simple_task,
    )

    assert seen, "verifier ran"
    assert all(actor_secret not in blob for blob in seen), (
        "actor-private reasoning leaked into the verifier context"
    )


def test_fail_findings_reach_the_actor_and_it_revises(
    harness, workspace: LocalWorkspaceEnvironment, simple_task: TaskQuery
):
    """The harness loops 'until the verifier agent confirms that all task
    requirements have been satisfied'."""
    provision(workspace, simple_task.fixtures, reset=True)
    seen_feedback: list[str] = []

    actor_handler = acting(writer("41"), writer("42"))

    def recording_actor(messages):
        feedback = [m.content for m in messages if "Verifier findings" in m.content]
        if feedback:
            seen_feedback.append(feedback[-1])
        return actor_handler(messages)

    def judge(out: str) -> tuple[str, str]:
        body = out.split("\n", 1)[-1].strip()
        if body == "42":
            return "PASS", "out/answer.txt contains 42."
        return "FAIL", f"out/answer.txt contains {body or 'nothing'}, not 42."

    result = _attempt(
        harness,
        workspace,
        (
            actor_context(recording_actor),
            verifier_context(probing(judge, reader())),
        ),
        simple_task,
    )

    assert result.passed
    assert result.revisions == 1, "one revision after the FAIL"
    assert seen_feedback, "the actor must see the verifier's findings"
    assert "41, not 42" in seen_feedback[-1]
    assert workspace.read_text("out/answer.txt") == "42"


def test_unverified_is_not_turned_into_a_verdict(
    harness, workspace: LocalWorkspaceEnvironment, simple_task: TaskQuery
):
    """'An unresolved outcome does not become a successful or failed learning
    example' — and it does not become a PASS either."""
    provision(workspace, simple_task.fixtures, reset=True)

    result = _attempt(
        harness,
        workspace,
        (
            actor_context(acting(writer("42"))),
            verifier_context(
                probing(
                    lambda out: ("UNVERIFIED", "cannot confirm the value was written."),
                    reader(),
                )
            ),
        ),
        simple_task,
    )

    assert result.verdict.value == "UNVERIFIED"
    assert not result.grounded
    assert not result.passed
    assert result.terminal == "unverified"


def test_practice_verification_offers_no_unverified_verdict(
    harness, workspace: LocalWorkspaceEnvironment, simple_task: TaskQuery
):
    """Appendix A.1: practice gets 'one grounded PASS or FAIL per project'."""
    provision(workspace, simple_task.fixtures, reset=True)
    offered: list[bool] = []

    def verifier_handler(messages):
        request = "\n".join(m.content for m in messages if m.role == "user")
        offered.append("UNVERIFIED" in request)
        return probing(lambda out: ("PASS", "fine."), "print('ok')")(messages)

    result = _attempt(
        harness,
        workspace,
        (
            actor_context(acting(writer("42"))),
            verifier_context(verifier_handler),
        ),
        simple_task,
        practice=True,
        project_id="p1",
    )

    assert result.passed
    assert offered and not any(offered), "practice must not offer UNVERIFIED"


def test_actor_receives_memory_and_can_look(
    workspace: LocalWorkspaceEnvironment, simple_task: TaskQuery
):
    """`look` and the memory block are both part of the actor's context."""
    provision(workspace, simple_task.fixtures, reset=True)
    seen: list[str] = []

    def handler(messages):
        seen.append(messages[-1].content)
        if len(seen) == 1:
            return "ACTION: look\nHINT: the file tree"
        return "ACTION: done\nSUMMARY: looked around"

    ActorAgent().run(
        ContextFactory(ScriptedClient(handler), actor_prompts.ACTOR_SYSTEM, role="actor")(),
        workspace,
        instruction=simple_task.instruction,
        memory_text="### env/notes.md\nUse utf-8-sig.",
    )

    assert "Use utf-8-sig." in seen[0], "memory must reach the actor"
    assert "[observation]" in seen[1], "the look result must come back"
    assert "data/input.txt" in seen[1], "the observation should show the fixtures"


def test_unavailable_user_channel_is_reported_honestly(
    workspace: LocalWorkspaceEnvironment, simple_task: TaskQuery
):
    """An `ask` with no user configured must not fabricate an answer."""
    provision(workspace, simple_task.fixtures, reset=True)
    seen: list[str] = []

    def handler(messages):
        seen.append(messages[-1].content)
        if len(seen) == 1:
            return "ACTION: ask\nQUESTION: which region should I use?"
        return "ACTION: done\nSUMMARY: no user available"

    ActorAgent(user_channel=None).run(
        ContextFactory(ScriptedClient(handler), actor_prompts.ACTOR_SYSTEM, role="actor")(),
        workspace,
        instruction=simple_task.instruction,
        memory_text=None,
    )

    assert "user channel unavailable" in seen[1]


def test_work_phase_isolation(workspace: LocalWorkspaceEnvironment):
    """The verifier never sees the actor's private scratch."""
    workspace.execute("open('visible.txt','w').write('candidate')", kind="python")
    workspace.write_private("reasoning/notes.txt", "private chain of thought")

    assert set(workspace.list_files()) == {"visible.txt"}
    assert not workspace.exists("reasoning/notes.txt")


def test_visual_observation_seam(workspace: LocalWorkspaceEnvironment):
    """A GUI environment returns an image; `describe_look` turns it into text.

    The bundled local environment is text-only, so this exercises the seam
    directly: an environment that reports an image, and a describer that the
    actor is expected to route it through.
    """
    from rsiagent.env.base import Environment, ExecResult, Observation

    class GuiEnvironment(Environment):
        name = "stub-gui"

        def reset(self) -> None: ...
        def execute(self, program, *, kind="python", timeout=None) -> ExecResult:
            return ExecResult(program=program, kind=kind)

        def observe(self, hint: str = "") -> Observation:
            return Observation(kind="image", image_path=Path("/tmp/screen.png"))

        def list_files(self, subdir: str = "") -> list[str]:
            return []

        def read_bytes(self, relpath: str) -> bytes:
            return b""

        def exists(self, relpath: str) -> bool:
            return False

        def write_fixture(self, relpath, content) -> None: ...
        def snapshot(self, label: str) -> str:
            return "cp"

        def restore(self, checkpoint_id: str) -> None: ...
        def discard(self, checkpoint_id: str) -> None: ...

    seen: list[str] = []

    def handler(messages):
        seen.append(messages[-1].content)
        if len(seen) == 1:
            return "ACTION: look\nHINT: the canvas"
        return "ACTION: done\nSUMMARY: saw it"

    described: list[str] = []

    def describe(image_path: str) -> str:
        described.append(image_path)
        return "A dark canvas with a single white 42 in the centre."

    ActorAgent(describe_look=describe).run(
        ContextFactory(ScriptedClient(handler), actor_prompts.ACTOR_SYSTEM, role="actor")(),
        GuiEnvironment(),
        instruction="Read the number on screen.",
        memory_text=None,
    )

    assert described == ["/tmp/screen.png"], "the describer must receive the image path"
    assert "A dark canvas with a single white 42" in seen[1]

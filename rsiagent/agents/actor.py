"""The actor agent: code as policy.

"The actor agent uses executable programs to interact with software, inspect
artifacts, and revise actions from execution feedback" (website, "Inside the
agent harness").  The action loop here is deliberately thin — submit a program,
observe its combined output, revise — because everything interesting in
RSIAgent happens *around* the actor, in the curriculum and the memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from ..env.base import Environment, ExecResult
from ..errors import ModelResponseError
from ..prompts import actor as actor_prompts
from ..parsing import ActorAction, parse_actor_action, parse_memory_edits
from .context import AgentContext

UserChannel = Callable[[str], str]


@dataclass
class ActorRun:
    """What one actor attempt produced."""

    context: AgentContext
    iterations: int = 0
    programs_run: int = 0
    memory_edits: int = 0
    terminal: str = "limit"  # "done" | "limit" | "watchdog"
    summary: str = ""
    last_exec: ExecResult | None = None
    transcript: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def declared_done(self) -> bool:
        return self.terminal == "done"


class ActorAgent:
    """Drives one task to completion inside an environment."""

    def __init__(
        self,
        *,
        observer: object | None = None,
        user_channel: UserChannel | None = None,
        describe_look: Callable[[str], str] | None = None,
    ) -> None:
        self.observer = observer
        self.user_channel = user_channel
        self.describe_look = describe_look

    # -- prompt assembly --------------------------------------------------
    @staticmethod
    def memory_section(memory_text: str | None) -> str:
        """Render the memory block, or the empty-memory notice.

        ``None`` means "this configuration has no memory at all" — the paper's
        ``w/o RSI`` baseline — which is different from an empty memory bank.
        """
        if memory_text is None:
            return actor_prompts.EMPTY_MEMORY_SECTION
        return actor_prompts.memory_block(memory_text or "(memory is empty)")

    def build_opening(
        self,
        charter: str,
        *,
        memory_text: str | None,
        environment_note: str = "",
    ) -> str:
        parts = [charter]
        if environment_note:
            parts.append(environment_note)
        parts.append(self.memory_section(memory_text))
        return "\n\n".join(parts)

    # -- the loop ---------------------------------------------------------
    def run(
        self,
        context: AgentContext,
        environment: Environment,
        *,
        instruction: str,
        memory_text: str | None,
        practice: bool = False,
        project_id: str = "",
        task_id: str = "",
        fixtures: dict[str, str] | None = None,
        iteration_limit: int = 500,
        watchdog_s: float = 36_000.0,
        program_timeout_s: float | None = None,
        environment_note: str = "",
        memory_session: object | None = None,
    ) -> ActorRun:
        """Attempt the instruction, leaving the candidate in ``environment``.

        The environment is *not* reset here.  Resetting between independent
        attempts is the runner's job, and doing it inside the actor would make
        revision-after-feedback impossible.

        ``memory_session`` lets the actor keep notes while it works.  Those
        edits live in a throwaway working copy: "Work-phase memory edits are
        discarded before learning, and the actor agent receives the current
        canonical memory at the update boundary" (Appendix A.2).  The session
        passed here is discarded by the caller; learning opens a new one.
        """
        if practice:
            charter = actor_prompts.practice_charter(project_id, instruction, fixtures)
        else:
            charter = actor_prompts.target_charter(task_id or "target", instruction)

        opening = self.build_opening(
            charter, memory_text=memory_text, environment_note=environment_note
        )
        context.append("user", opening)

        run = ActorRun(context=context)
        started = time.monotonic()

        for iteration in range(iteration_limit):
            if time.monotonic() - started > watchdog_s:
                run.terminal = "watchdog"
                run.transcript.append("[watchdog] iteration budget exhausted")
                break

            run.iterations = iteration + 1
            reply = context.ask()

            # Work-phase scratch notes.  Applied to the throwaway session, so
            # they never reach canonical memory on their own.
            if memory_session is not None:
                for edit in parse_memory_edits(reply):
                    if edit.op == "write":
                        memory_session.write(edit.path, edit.content)  # type: ignore[attr-defined]
                    else:
                        memory_session.delete(edit.path)  # type: ignore[attr-defined]
                    run.memory_edits += 1

            action = parse_actor_action(reply)

            if action.kind == "program":
                result = environment.execute(
                    action.program,
                    kind=action.program_kind,  # type: ignore[arg-type]
                    timeout=program_timeout_s,
                )
                run.programs_run += 1
                run.last_exec = result
                context.append("user", result.render())
                run.transcript.append(f"[program:{action.program_kind}] {result.render(400)}")

            elif action.kind == "look":
                run.transcript.append(f"[look] {action.hint}")
                context.append("user", self._observation(environment, action))

            elif action.kind == "ask":
                run.transcript.append(f"[ask] {action.question}")
                context.append("user", self._answer(action))

            elif action.kind == "done":
                run.terminal = "done"
                run.summary = action.summary
                run.transcript.append(f"[done] {action.summary[:400]}")
                break
            else:  # pragma: no cover - parse_actor_action raises first
                raise ModelResponseError(f"unhandled actor action: {action.kind}")

        run.duration_s = time.monotonic() - started
        return run

    # -- action handlers --------------------------------------------------
    def _observation(self, environment: Environment, action: ActorAction) -> str:
        observation = environment.observe(action.hint)
        if observation.kind == "image" and observation.image_path is not None:
            # A vision-capable deployment would attach the image here.  The text
            # fallback keeps the loop honest about what it can actually see.
            described = (
                self.describe_look(str(observation.image_path)) if self.describe_look else ""
            )
            return (
                f"[observation] image at {observation.image_path}\n"
                f"{described or '(no visual describer configured)'}"
            )
        return f"[observation]\n{observation.text}"

    def _answer(self, action: ActorAction) -> str:
        if self.user_channel is None:
            return (
                "[user channel unavailable] No user is reachable in this run. "
                "Proceed using only what you can establish from the environment."
            )
        return f"[user] {self.user_channel(action.question)}"

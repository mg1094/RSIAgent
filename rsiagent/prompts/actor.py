"""Actor agent prompts (paper Prompt P1 and the practice charters).

The quoted passages are reproduced from Appendix B.1 of the paper; the
operational scaffolding around them (the action grammar) is this
implementation's, since the paper prints "selected passages ... not complete
system messages".
"""

from __future__ import annotations

ACTOR_SYSTEM = """\
You solve ONE task on a real Ubuntu machine by WRITING PROGRAMS. You cannot see \
or click the screen: your only way to act is to submit one complete program at a \
time (python3 or bash) — programs may drive running applications where a task \
needs it. It runs on the machine; its combined stdout+stderr and exit code come \
back to you. Work like an engineer at a REPL: investigate first, then commit a \
solution, then verify it.

Code is your control channel, not a reinterpretation of the task: every action \
you take is a submitted program, and those programs may inspect and operate the \
machine or automate a required application. Choose the implementation yourself. \
Whatever method you choose, the candidate must satisfy the task's literal \
requirements, including any named-application, native-editable-state, behavior, \
rendered-output, workflow, or provenance requirement.

## How to reply

Reply with EXACTLY ONE of the following, and nothing else.

A program to run:

    ```python
    <your program>
    ```

    ```bash
    <your program>
    ```

Or, to inspect something without running a program:

    ACTION: look
    HINT: <what you want to see>

Or, when a user channel exists and you need information you cannot obtain:

    ACTION: ask
    QUESTION: <your question>

Or, when you believe the task is complete:

    ACTION: done
    SUMMARY: <what you produced and why it satisfies every requirement>

## Rules

- One program per turn. Do not batch several unrelated programs into one reply.
- Investigate before you commit. Read the inputs, check the tools that exist,
  and confirm your assumptions against the actual environment rather than
  against what you expect a machine like this to do.
- When a program fails, read its output and repair the program. Do not repeat a
  failing program unchanged.
- Before declaring done, check the task's actual requirements — not just that
  some output exists. Prefer a read-only check that demonstrates each named
  requirement holds.
"""

PRACTICE_CHARTER = """\
## Practice project

This is a practice project authored by the curriculum agent to build reusable \
capability. It is not the target task, and no score depends on it.

**Project id:** {project_id}
**Instruction:** {instruction}

{fixtures_note}

Work until you judge the project's instruction satisfied, then reply with \
`ACTION: done`. What matters is what you learn that transfers, not whether this \
particular project looks impressive.
"""

TARGET_CHARTER = """\
## Task

**Task id:** {task_id}
**Instruction:** {instruction}

Work until the instruction is satisfied under its literal requirements, then \
reply with `ACTION: done`.
"""

MEMORY_SECTION = """\
## Your durable memory

The following files were written by you (or by an earlier instance of you) while \
working in this environment. They persist across tasks. Use them: they are the \
whole point of the exploration that produced them. They can also be wrong or \
out of date — if this environment contradicts them, trust the environment and \
fix the memory when you next consolidate.

{memory}

## End of memory
"""

EMPTY_MEMORY_SECTION = """\
## Your durable memory

Memory is currently empty. You have no prior experience in this environment.
"""


def memory_block(rendered: str) -> str:
    return MEMORY_SECTION.format(memory=rendered)


def practice_charter(
    project_id: str,
    instruction: str,
    fixtures: dict[str, str] | None = None,
) -> str:
    if fixtures:
        listing = "\n".join(f"  - {name}" for name in sorted(fixtures))
        note = f"**Inputs provided for you in the workspace:**\n{listing}"
    else:
        note = "**Inputs:** none beyond what is already in the workspace."
    return PRACTICE_CHARTER.format(
        project_id=project_id, instruction=instruction, fixtures_note=note
    )


def target_charter(task_id: str, instruction: str) -> str:
    return TARGET_CHARTER.format(task_id=task_id, instruction=instruction)

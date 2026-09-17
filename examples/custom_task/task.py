"""A minimal task definition — the two objects ``rsiagent run`` needs.

A task is a :class:`TaskQuery`: an id, an authoritative instruction, and the
input fixtures that define a fresh environment.  An evaluator is anything with
a ``score(environment) -> TaskScore`` method.

The split matters.  The instruction is the *only* thing the agents see.  The
evaluator is sealed: it runs after the actor–verifier loop and its output goes
to the journal, never to a prompt.  So this file can be as explicit as it likes
about what "correct" means without leaking the answer into the run.
"""

from __future__ import annotations

from rsiagent.tasks.spec import ScriptedEvaluator, TaskQuery

INSTRUCTION = """\
Write the number 42 into `out/answer.txt`.

The file must exist, must contain exactly the two characters `42`, and must not
contain anything else.
"""

task = TaskQuery(
    id="write-42",
    instruction=INSTRUCTION,
    fixtures={
        "data/input.txt": "this file is a decoy and contains no answer\n",
        "README.md": "# Stub project\n\nProduce `out/answer.txt`.\n",
    },
)


def _answer_written(environment) -> float:
    """1.0 when out/answer.txt contains exactly `42`."""
    if not environment.exists("out/answer.txt"):
        return 0.0
    return 1.0 if environment.read_text("out/answer.txt").strip() == "42" else 0.0


def _no_trailing_junk(environment) -> float:
    """A second, equally weighted check.

    Two checks rather than one so the score is graded rather than binary: a
    partial-credit metric is what the paper reports, and a single all-or-nothing
    check would hide the difference between "nothing written" and "written but
    sloppy".
    """
    if not environment.exists("out/answer.txt"):
        return 0.0
    return 1.0 if environment.read_text("out/answer.txt") == "42" else 0.0


evaluator = ScriptedEvaluator(
    {"answer_written": _answer_written, "exact_content": _no_trailing_junk},
    notes="write-42: two equally weighted checks",
)

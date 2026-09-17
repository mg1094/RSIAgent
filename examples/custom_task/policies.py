"""Offline stand-ins for the three model roles.

Pass this to ``rsiagent run --policies policies.py:policies`` and the entire
lifecycle runs with no API key, no network, and no model.  It exists so you can
watch the *protocol* work — what the curriculum agent is asked, what the
verifier is handed, when memory is committed — before pointing it at a real
model.

Each entry is a plain callable taking the role's message history and returning
the role's reply text.  To use real models instead, drop ``--policies`` and set
``OPENAI_API_KEY``; the prompts, parsing, and protocol are identical.

The single idea worth watching here: :func:`actor` writes ``41`` or ``42``
depending on whether a *memory file* is present in its prompt.  Nothing else
differs between the first target attempt and the last.  That is the whole claim
of the framework, made small enough to read.
"""

from __future__ import annotations

import json

MEMORY_FACT = "The correct value is 42, not 41."


def _fence(program: str) -> str:
    return f"```python\n{program}\n```"


def _memory_text(messages) -> str:
    """Everything the actor was ever shown in a user turn."""
    return "\n".join(m.content for m in messages if m.role == "user")


# --------------------------------------------------------------------------
# Actor
# --------------------------------------------------------------------------
def actor(messages) -> str:
    """Acts by writing programs, then distills what it learned.

    The same callable serves the work phase and the three learning prompts,
    because in a real run the actor client does both — that is what
    "experience-owned learning" means: the agent that did the work is the one
    that decides what to keep from it.
    """
    last = messages[-1].content

    # --- learning step 1: distillation ---------------------------------
    if "Learning step 1 of 2" in last:
        if MEMORY_FACT in last:
            return "Nothing here was not already recorded.\n\nACTION: done"
        return (
            f"```memory:write notes/answer.md\n{MEMORY_FACT}\n```\n\n"
            "ACTION: done\nSUMMARY: recorded the correct value"
        )

    # --- learning step 2: reconciliation, and the curriculum handoff ----
    if "Learning step 2 of 2" in last or "Learning diagnosis" in last:
        return "ACTION: done"

    # --- work phase -----------------------------------------------------
    # An exec result, or verifier feedback, means the actor is done acting.
    if last.lstrip().startswith("$ (") or "## Verifier findings" in last:
        return "ACTION: done\nSUMMARY: the program ran; the artifact is on disk"

    # The only thing that differs between attempts.
    value = "42" if MEMORY_FACT in _memory_text(messages) else "41"
    return _fence(
        "import os\n"
        "os.makedirs('out', exist_ok=True)\n"
        f"open('out/answer.txt', 'w').write('{value}')\n"
        "print('wrote', open('out/answer.txt').read())\n"
    )


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------
def verifier(messages) -> str:
    """Probes the real artifact, then judges from what it actually saw.

    Note what this function is *not* given: no memory, no actor transcript. The
    verifier's information boundary is structural — the harness never passes
    them — so it cannot be talked out of a finding by the actor's own account
    of its work.
    """
    last = messages[-1].content

    if not last.lstrip().startswith("$ ("):
        # First turn: run a probe.  The harness restores a checkpoint after
        # this, so a probe is free to look at anything.
        return _fence("print(open('out/answer.txt').read())\n")

    body = last.split("\n", 1)[-1].strip()
    if body == "42":
        return "VERDICT: PASS\nFINDINGS: out/answer.txt contains exactly '42'."
    return (
        "VERDICT: FAIL\n"
        f"FINDINGS: out/answer.txt contains {body!r}, but the task requires '42'. "
        "The value written is wrong, not merely formatted differently."
    )


# --------------------------------------------------------------------------
# Curriculum
# --------------------------------------------------------------------------
def curriculum(messages) -> str:
    """Chooses the next experience.

    This one is deliberately the least interesting policy possible — it never
    asks for practice — so that the *only* route to a passing score is the
    target attempt's own learning.  Swap in a policy that returns
    ``{"decision": "PROJECT", ...}`` to watch Phase 1 and Phase 2 practice run.
    """
    prompt = messages[-1].content

    if "Phase 1 (Broad Recursive Self-exploration)" in prompt:
        return json.dumps(
            {
                "decision": "SATURATED",
                "rationale": (
                    "The task is a single scalar write with no prerequisite "
                    "capabilities worth practising in isolation; the informative "
                    "experience is the target attempt itself."
                ),
            }
        )

    return json.dumps(
        {
            "decision": "READY_FOR_TARGET",
            "rationale": (
                "No practice has greater expected value than returning to the "
                "target lifecycle. This is a readiness decision, not a verdict "
                "on correctness."
            ),
        }
    )


policies = {"actor": actor, "verifier": verifier, "curriculum": curriculum}

"""Memory consolidation prompts (paper Prompts P5, P6, P7).

These three are delivered to the *actor* context that performed the work.  That
is the paper's "Experience-Owned Learning": the agent that produced an
experience also decides what to retain from it.  The curriculum agent never
approves memory wording, and the verifier never touches memory at all.
"""

from __future__ import annotations

DISTILLATION = """\
## Learning step 1 of 2 — distillation

The task is over. Its verdict is final and the environment will be reset; do not \
continue changing the terminal candidate for credit.

You now decide what, if anything, this experience is worth retaining.

Make causal claims only when your trajectory and evidence support them, and \
preserve uncertainty where they do not. A FAIL may still contain valuable \
evidence, but must not be recorded as a verified success. A PASS does not prove \
your interpretation was right — check whether the evidence actually supports the \
general rule you are tempted to write down.

You may investigate remaining questions if the available project state makes that \
useful, and you may add, revise, reorganize, delete, or leave memory unchanged. \
You own the content, representation, retrieval strategy, scope, and stopping \
decision; there is no required schema, length, number of files, or number of \
turns.

Declare done when the memory you choose to carry forward is ready.

## How to make memory edits

Emit one or more of these blocks:

    ```memory:write path/to/file.md
    <full new content of that file>
    ```

    ```memory:delete path/to/file.md
    ```

Then finish with `ACTION: done`.

Write for the agent who will read this next, in a different task, with no memory \
of today. Prefer an actionable procedure or a specific constraint over a summary \
of what happened.

## Terminal outcome

**Verdict:** {verdict}

**Verifier findings:**
{findings}

## Current memory

{memory}
"""

RECONCILIATION = """\
## Learning step 2 of 2 — reconciliation

Your first-pass memory update is a draft and has not been promoted.

Look for conflicting assertions, unsupported causal explanations, stale \
environment assumptions, and conclusions broader than the observed evidence. \
Investigate when that is useful; otherwise narrow or qualify claims, preserve \
uncertainty, reorganize them, or remove them. Do not merely append this episode \
while leaving contradicted older advice stated as fact.

There is no required claim table, schema, report, length, number of files, or \
number of turns.

## How to make memory edits

Emit `memory:write` and `memory:delete` blocks exactly as in the previous step, \
then finish with `ACTION: done`. Emit no blocks if the draft already survives \
your review.

## Memory as it now stands (including your draft update)

{memory}
"""

DIAGNOSIS = """\
## Learning diagnosis for curriculum review

Explain the hypotheses that now seem most useful for choosing the next experience: \
attempted approaches, observed limitations, plausible causal reasons, what appears \
reliable, what remains uncertain, and which distinctions or stress cases could \
discriminate among competing explanations.

Do not include private chain-of-thought or a turn-by-turn transcript: provide only \
the concise conclusions and hypotheses you choose to communicate. There is no \
required schema, length, or organization.

This handoff is read by the curriculum agent, which cannot see your memory, your \
transcript, or the verifier's report. Write it to be useful on its own.

Reply with the diagnosis as plain text, then `ACTION: done`.
"""


def distillation(verdict: str, findings: str, memory: str) -> str:
    return DISTILLATION.format(verdict=verdict, findings=findings.strip(), memory=memory)


def reconciliation(memory: str) -> str:
    return RECONCILIATION.format(memory=memory)

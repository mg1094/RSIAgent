# A custom task, end to end

The smallest complete example: one task, one evaluator, three offline policies.
It runs in about a second and needs no API key.

```bash
# No memory, one attempt. Fails — the actor guesses 41.
rsiagent run --task task.py:task --evaluator task.py:evaluator \
             --policies policies.py:policies --arm baseline --out runs/base

# The full lifecycle. Succeeds, because the failed attempt became memory.
rsiagent run --task task.py:task --evaluator task.py:evaluator \
             --policies policies.py:policies --arm rsi --out runs/rsi

rsiagent inspect runs/rsi
rsiagent memory  runs/rsi
```

```
baseline   partial=0.0000 binary=False          2 checks, 0 passed
rsi        partial=1.0000 binary=True           2 checks, 2 passed
```

## What to look at

`policies.py` is the teaching material. Read `actor` first:

```python
value = "42" if MEMORY_FACT in _memory_text(messages) else "41"
```

That is the entire difference between the failing attempt and the passing one.
Nothing about the model changed, nothing was fine-tuned — a file the actor wrote
during the first attempt's *learning* step came back in the second attempt's
prompt. `rsiagent memory runs/rsi` prints that file:

```
--- notes/answer.md ---
The correct value is 42, not 41.
```

Then read the journal to see the loop that produced it:

```bash
rsiagent inspect runs/rsi
```

```
target cycle 0: FAIL       <- actor wrote 41, verifier caught it
  curriculum -> READY_FOR_TARGET
target cycle 1: PASS       <- memory existed this time
```

## Swapping in real models

Delete `--policies` and supply credentials:

```bash
export OPENAI_API_KEY=sk-...
rsiagent run --task task.py:task --evaluator task.py:evaluator \
             --arm rsi --actor-model gpt-4o --verifier-model gpt-4o --curriculum-model gpt-4o
```

Everything else — the prompts under `rsiagent/prompts/`, the parsers, the wave
barrier, the memory protocol — is unchanged. Only the text generation moves
from a Python function to a model.

## Writing your own

Two objects are all `rsiagent run` needs.

**A `TaskQuery`** — an id, an authoritative instruction, and input fixtures.
The instruction is the only thing the agents ever see, so it must state every
binding requirement: named applications, editable state, rendered appearance,
workflow, provenance. Anything you leave out, the verifier cannot check.

**An `Evaluator`** — anything with `score(environment) -> TaskScore`. It is
sealed: it runs after the actor–verifier loop and its result goes to the
journal, never to a prompt. It is also free to be partial-credit, which is what
makes the paper's `Partial` metric informative:

```python
from rsiagent.tasks.spec import ScriptedEvaluator, TaskScore

evaluator = ScriptedEvaluator(
    {
        "file_exists": lambda env: env.exists("out/report.json"),
        "has_all_regions": lambda env: ...,
        "totals_correct": lambda env: ...,
    }
)
```

Each check returns a bool or a float in `[0, 1]`; the score is their mean.

> **The evaluator is the task.** Grading leniency does not stay contained — an
> accepted mistake becomes a memory update, and that update shapes every later
> attempt. It is the one component here worth over-investing in.

# Architecture

RSIAgent separates four things that are easy to entangle: **acting**,
**judging**, **deciding what to do next**, and **measuring the result**. Each
gets its own role, its own context, and — where the paper requires it — a
structural guarantee rather than a prompt-level request.

The organising claim of the paper is that an agent's exploration history is
worth more than its transcript:

> An agent can repair a mistake during one attempt and still have to discover
> the same lesson again in the next.

So the interesting machinery is not the action loop. It is the machinery that
decides *what an experience is worth keeping*, *who is allowed to keep it*, and
*when the accumulated memory stops changing*.

---

## Layering

Imports flow strictly downward. There are no cycles; the two leaf modules
(`errors`, `parsing`, `status`) exist precisely to keep it that way.

```
L0  errors · parsing · status · config          pure, no internal deps
L1  llm · env · memory · prompts · tasks        transport & state
L2  agents                                      the three roles + contexts
L3  learning                                    memory consolidation
L4  runtime                                     harness & journal
L5  rsi                                         the Algorithm A1 lifecycle
L6  cli
```

## Module map

| Path | Responsibility |
| --- | --- |
| `rsiagent/config.py` | `RunConfig` and the paper's Table A3 defaults; the `stages` ablation switch |
| `rsiagent/status.py` | `Verdict`, `Decision`, `TerminalStatus`, `Phase` |
| `rsiagent/parsing.py` | Model output → protocol state. The boundary where silent corruption would enter |
| `rsiagent/llm/` | `LLMClient` protocol; OpenAI-compatible and Anthropic backends; deterministic offline clients |
| `rsiagent/env/` | `Environment` protocol, the local implementation, and per-branch factories |
| `rsiagent/memory/` | `MemoryBank`, snapshots, sessions, freezing, tree hashing |
| `rsiagent/prompts/` | Every role's prompt, with the paper's quoted passages marked |
| `rsiagent/tasks/` | `TaskQuery`, `Project`, `TaskScore`, and the sealed `Evaluator` protocol |
| `rsiagent/agents/` | `AgentContext` and the actor, verifier, and curriculum agents |
| `rsiagent/learning/` | Distillation → reconciliation → diagnosis |
| `rsiagent/runtime/` | `TaskHarness` (the action–verification loop) and the JSONL journal |
| `rsiagent/rsi/` | `phase1_brs`, `phase2_drs`, `phase3_eval`, and `protocol` |

---

## The boundaries, and how each is enforced

The paper specifies information boundaries in prose. Prose is not a mechanism,
so each boundary here is a property of the types involved.

### The verifier cannot see the actor

> The verifier agent is isolated from the actor agent's private reasoning and
> memory, which helps reduce correlated errors during evaluation.

`VerifierAgent.verify` takes an environment and an instruction. There is no
parameter through which a transcript, a memory file, or a program log could
arrive. The actor's scratch lives outside the environment root entirely
(`<root>.private/`), so it is not merely unreferenced — it is unreadable from
anything the verifier is handed.

### Verification cannot contaminate the candidate

> When this inspection ends, the harness restores that checkpoint before the
> candidate is graded, so none of your in-VM effects enter the scored state.

`TaskHarness._verify` snapshots, delegates, and restores in a `finally` block.
A verifier that crashes mid-probe still rolls back. This is what makes it safe
to tell the verifier it may run arbitrary tests — and it is a *harness*
guarantee, not a request that the verifier be careful.

### The curriculum agent cannot author memory

> The curriculum agent may inspect a disposable copy to guide exploration, but
> its local edits are not synchronized back.

`MemoryBank.readonly_view()` returns a `MemorySnapshot`, which has no `commit`
and no `write`. The curriculum agent is passed renders of that snapshot, never a
session. There is no code path by which it could edit canonical memory, because
it is never given an object that can.

### Only completed learning is promoted

> Work-phase memory edits are discarded before learning, and the actor agent
> receives the current canonical memory at the update boundary.

A branch works inside a `MemorySession` opened on the pre-wave snapshot. At
consolidation time that session is discarded, and learning opens a *new* one on
live canonical memory. Whatever the actor scribbled while working cannot reach
the bank, and the learner cannot be reasoning about a stale copy.

### Branches are isolated; commits are ordered and cumulative

> Every project in a wave begins with the same immutable pre-wave memory
> snapshot and cannot observe sibling work or outcomes. […] Each update operates
> on the latest canonical memory, including preceding updates from the same wave.

Isolation is physical: `LocalEnvironmentPool` hands each branch its own
directory tree. Ordering is explicit: `BroadExplorer.run_wave` collects every
verdict before opening a single session, then commits in the authored order,
each on a fresh read of the bank.

### Frozen memory is provably unchanged

> The runner records the frozen memory's file-tree hash, copies it into the
> evaluation environment, and disables curriculum decisions and host memory
> writeback.

`MemoryBank.freeze_to` records a SHA-256 over the sorted `(path, content)` set.
Phase 3 re-checks it afterwards and reports an integrity violation as an
infrastructure failure — never as a score.

---

## The three phases

### Phase 1 — Broad Recursive Self-exploration

`rsiagent/rsi/phase1_brs.py`

A loop over **waves**. The curriculum agent authors a wave of independent
projects; each runs and is verified in an isolated environment, concurrently up
to `brs_concurrency`. When *all* verdicts are grounded, the branches consolidate
one at a time, in authored order. The next wave is authored only after those
commits are durable.

Two details that are easy to get wrong and are implemented explicitly:

- **The budget is checked between waves.** A wave already underway is never
  truncated, so the realized project count can exceed the nominal budget.
- **An unresolved branch blocks the wave.** By default nothing from that wave is
  merged (`brs_on_blocked="halt"`); `"drop_branch"` continues with the grounded
  branches and records the dropped ones.

### Phase 2 — Deep Recursive Self-exploration

`rsiagent/rsi/phase2_drs.py`

The target is attempted in a reset environment, then both the outcome *and* the
actor's learning diagnosis go to the curriculum agent, which either requests
practice or returns readiness. The loop's shape follows Algorithm A1 steps 3–6,
including the parts that surprise people:

- **A target PASS is not terminal** under the default `curriculum_review`
  policy. If the curriculum agent selects practice afterwards, the target must
  be re-attempted — tracked with `practice_since_target`, because a pass earned
  before new practice is a pass under memory that no longer exists.
- **STALLED is not convergence.** It buys one final target attempt and is
  recorded as `STALLED`, never as `COMPLETED`.
- **The review after a practice project sees the practice's outcome**, not the
  older target attempt's.

### Phase 3 — Test-time memory reuse

`rsiagent/rsi/phase3_eval.py`

Freeze, reset, run the *same* harness with learning switched off, then score.
The evaluator runs after the actor–verifier loop and its output goes only to the
journal — `TaskScore` never reaches a prompt, which is what "does not supply
scores or hidden checks to the learning agents" means in practice.

---

## Memory

Memory is "a collection of actor-authored files, with no required schema, file
count, or length." Nothing in `memory/` imposes structure on its contents; the
rules are all about *ownership and timing*:

| Object | Can do | Lifetime |
| --- | --- | --- |
| `MemoryBank` | Promote sessions, snapshot, freeze | The lineage |
| `MemorySnapshot` | Be read and materialized | Until discarded |
| `MemorySession` | Write, delete, commit | One consolidation |
| frozen `MemoryBank` | Be read | Phase 3 |

The `ENV_FACT:` convention you will see in the offline demo is the *demo's*
convention, invented by its stand-in policies — a small, legible stand-in for an
agent reading back its own written procedure notes. The framework itself has no
opinion about how memory is written down.

---

## The journal

One JSONL line per event, written as things happen. It is a host artifact:
nothing in it is ever fed back into a prompt, which is what keeps scores and
audit records outside the learning loop.

`Journal.memory_growth()` returns the memory-checkpoint series — file count,
byte total, tree hash, and the verdict that produced it — which is the data
behind the paper's memory-growth figures. Because it is read back from the file
rather than kept in memory, a finished run can be analysed in a fresh process.

---

## Extension seams

| To change | Implement |
| --- | --- |
| The environment (container, VM, remote) | `rsiagent.env.base.Environment` |
| The model provider | `rsiagent.llm.base.LLMClient`, or `build_clients(provider=...)` |
| The official evaluator | `rsiagent.tasks.spec.Evaluator` |
| The benchmark integration | A `TaskQuery` + `Evaluator` pair, driven by `RSIRunner` |

The `Environment` protocol is deliberately four verbs — reset, execute,
snapshot, restore — plus inspection. Anything that can offer those can host the
whole lifecycle.

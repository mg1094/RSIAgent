# Contributing

Thanks for looking. This is a learning-oriented reimplementation of a research
framework, so the most valuable contributions are the ones that make the
protocol easier to understand or harder to get wrong.

## Getting set up

```bash
git clone <your fork>
cd RSIAgent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                        # ~90 tests, ~15s, no API key needed
python examples/offline_demo/run.py
```

Nothing in the test suite touches the network or needs credentials. If a test
of yours does, it belongs behind a marker and off by default.

## What this project cares about

**The paper is the specification.** Every module docstring quotes the passage it
implements, and [`docs/PAPER_MAPPING.md`](docs/PAPER_MAPPING.md) is a
claim-by-claim traceability table. If you change behaviour, update the mapping
and say which sentence moved.

**Boundaries are structural, not stylistic.** The verifier cannot see the actor
because no parameter carries a transcript; the curriculum agent cannot author
memory because it is handed a snapshot with no `commit` method. A change that
makes one of these "a convention" instead of a property is a regression even if
every test still passes. `tests/` exists mostly to make those properties hard to
break by accident.

**Tests run the real thing.** Real subprocesses, real files, real memory commits.
Mocks are for the model transport only — that is what `ScriptedClient` is. A test
that mocks the environment is testing the mock.

## Good first contributions

- **A new `Environment`.** A Docker, `firecracker`, or SSH-backed implementation
  of `rsiagent.env.base.Environment` would be the single most useful addition:
  it is the difference between "runs on my laptop" and "runs safely".
- **A worked task.** Another `examples/` entry with a real evaluator and a
  documented failure mode.
- **Protocol tests.** If you have found a way to get the wave barrier, the DRS
  stopping rules, or memory ownership to do the wrong thing, a failing test is a
  complete contribution on its own.
- **Documentation.** If a paragraph took you three reads, it needs a fourth
  draft.

## Before you open a PR

```bash
pytest
ruff check . && ruff format --check .
```

Both run in CI on 3.10, 3.11, and 3.12.

Commit messages: a short imperative subject, then a body explaining *why* —
particularly for protocol changes, where the reason is usually a sentence from
the paper.

## What is out of scope

The benchmark integrations from the reference implementation (OSWorld 2.0 and
Agents' Last Exam, with their QEMU images, guest transport, and sealed graders)
are deliberately not reproduced here. A PR that adds an adapter is welcome; a PR
that vendors benchmark assets is not — they carry their own licenses and are
large.

Reimplementing the original project's source files is also out of scope. This
repository is written from the paper, and keeping it that way is what makes it
useful as a second reading of the method.

## Reporting bugs

Use the issue templates. For anything touching isolation, sandboxing, or the
code-execution model, read [`SECURITY.md`](SECURITY.md) first — the host-
execution behaviour is documented and expected, but a boundary failing is not.

## License

Contributions are accepted under Apache-2.0, the license this repository ships
with. See [`NOTICE`](NOTICE) for attribution to the original paper.

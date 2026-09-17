# Security

## The one thing to read before running anything

**This framework executes code that a language model wrote.**

The actor's action space *is* executable programs. When you point RSIAgent at a
real model, that model's output is run as Python or Bash. That is not a bug or a
misconfiguration — it is the framework's design ("code as policy"), and the
paper's reference implementation handles it by keeping the actor inside a QEMU
virtual machine with a restorable checkpoint.

The environment bundled here, `LocalWorkspaceEnvironment`
(`rsiagent/env/local.py`), does **not** do that. It runs subprocesses directly on
the host, with the privileges of the process that started it. It is a
convenience implementation, chosen so the protocol can run on a laptop. It is
not a security boundary and should not be treated as one.

Practical consequences:

| Risk | What it means here |
| --- | --- |
| Arbitrary code execution | A model-authored program can read, write, or delete anything your user account can. |
| Credential exposure | The actor's environment is scrubbed (`HOME` is redirected, a minimal `PATH` is set), but files readable by your account are still readable by the actor's programs. |
| Persistence | Nothing is sandboxed. A program may leave state outside the workspace. |
| Network | Programs may make network requests. |

**Run it only against environments you control, in a disposable container or
VM, with no valuable credentials mounted.** If you need the isolation the paper
describes, implement `rsiagent.env.base.Environment` against your own sandbox —
the protocol needs only four verbs (`reset`, `execute`, `snapshot`, `restore`)
plus inspection, and everything above that layer is unchanged.

For the bundled offline demo and the documented examples, all policy roles are
deterministic Python functions in this repository, not models. Those run
programs too, but the programs are readable in the source and produce only the
files the examples describe.

## Prompt injection

An agent exploring an unfamiliar environment reads files it did not write. Some
of that content may contain text shaped like instructions — a README saying
"ignore your previous instructions", a CSV whose header row reads like a
command.

The framework's own boundaries limit the blast radius rather than the exposure:

- The **verifier never sees the actor's memory or transcript**, so injected text
  in a workspace file cannot be laundered into a verdict through the actor's
  account of its work.
- The **curriculum agent cannot author memory**, so injected text cannot reach
  canonical memory through the exploration path.
- **Memory is written only by the actor**, from grounded verifier findings.

None of that stops a model from being misled about the *task* within one
attempt, nor from writing a hostile memory file when it consolidates. Treat the
memory bank as untrusted content and read it before reusing it
(`rsiagent memory runs/latest`).

## Known limitations

These are documented in the README and are design scope, not vulnerabilities:

- `LocalWorkspaceEnvironment` provides no isolation (above).
- The verifier's checkpoint restore protects the *candidate* from the verifier's
  probes. It does not roll back anything a program did to shared external state
  — a remote service, a shared database, a network resource.
- `MemoryBank.promote` replaces the canonical directory. A crash between the
  rename and the cleanup leaves `<name>.previous` behind; it is removed on the
  next successful promote and contains no live state.

## Reporting

This is a learning-oriented reimplementation, not a production system, and it
has no security team. If you find a vulnerability, please open a GitHub issue
unless public disclosure would put users at risk — in that case, open a minimal
issue asking for a private contact channel, with no details in the body.

Please include the version or commit, what you ran, and what happened. Reports
about the code-execution model above ("it ran my program") are expected
behaviour and are documented; reports about the *boundaries* failing — a
verifier reading actor memory, a frozen memory bank accepting a write, a
memory path escaping its root — are exactly what this project wants to hear
about, and there are tests for each in `tests/`.

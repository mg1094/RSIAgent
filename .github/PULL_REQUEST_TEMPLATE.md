## What this changes

<!-- One or two sentences. If it is a protocol change, name the sentence in the
paper it follows or departs from. -->

## Why

<!-- The reason, not the diff. For protocol changes this is usually a passage
from the paper. -->

## Checklist

- [ ] `pytest` passes
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] If behaviour changed, the relevant module docstring and
      `docs/PAPER_MAPPING.md` are updated
- [ ] If a boundary moved, there is a test that fails when it is removed
- [ ] Tests do not require an API key or the network

## Notes for the reviewer

<!-- Anything you were unsure about, or deliberately left out. -->

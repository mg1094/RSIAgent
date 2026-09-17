"""Parsers are the boundary between model output and protocol state.

A mis-parse here does not crash loudly — it silently writes the wrong memory or
records the wrong verdict three phases later.  These tests pin the behaviours
that keep that from happening.
"""

from __future__ import annotations

import pytest

from rsiagent.errors import ModelResponseError
from rsiagent.parsing import (
    parse_actor_action,
    parse_handoff,
    parse_memory_edits,
    parse_verdict,
)


class TestActorAction:
    def test_python_fence_becomes_program(self):
        action = parse_actor_action("Here you go:\n```python\nprint(1)\n```")
        assert action.kind == "program"
        assert action.program_kind == "python"
        assert action.program == "print(1)\n"

    def test_bash_fence_becomes_program(self):
        action = parse_actor_action("```bash\nls -la\n```")
        assert action.kind == "program"
        assert action.program_kind == "bash"

    def test_program_wins_over_a_done_line(self):
        """A model that narrates 'done' but still emits a program has work to do."""
        action = parse_actor_action("ACTION: done\n```python\nprint(1)\n```")
        assert action.kind == "program"

    def test_look_and_ask_and_done(self):
        assert parse_actor_action("ACTION: look\nHINT: the file tree").hint == "the file tree"
        assert parse_actor_action("ACTION: ask\nQUESTION: which region?").question == (
            "which region?"
        )
        assert parse_actor_action("ACTION: done\nSUMMARY: finished").summary == "finished"

    def test_no_action_is_an_error(self):
        with pytest.raises(ModelResponseError):
            parse_actor_action("I think we should probably do something.")


class TestVerdict:
    def test_simple_verdict(self):
        parsed = parse_verdict("VERDICT: PASS\nFINDINGS: all good")
        assert parsed.status == "PASS"
        assert parsed.findings == "all good"

    def test_last_verdict_wins(self):
        """A verifier that reasons aloud may restate the task before concluding."""
        text = (
            "The candidate might FAIL if the file were missing.\n"
            "VERDICT: PASS\nFINDINGS: the file is present and correct."
        )
        assert parse_verdict(text).status == "PASS"

    def test_disallowed_verdict_is_not_matched(self):
        """Practice projects take two verdicts; UNVERIFIED must not slip through."""
        with pytest.raises(ModelResponseError):
            parse_verdict("VERDICT: UNVERIFIED\nFINDINGS: unclear", allowed=("PASS", "FAIL"))

    def test_missing_verdict_is_an_error(self):
        with pytest.raises(ModelResponseError):
            parse_verdict("I looked at it and it seemed fine.")


class TestHandoff:
    def test_projects_wave(self):
        handoff = parse_handoff(
            """```json
            {"decision": "PROJECTS",
             "rationale": "coverage",
             "projects": [{"id": "a", "instruction": "do a"},
                          {"id": "b", "instruction": "do b", "fixtures": {"x.txt": "hi"}}]}
            ```"""
        )
        assert handoff.decision == "PROJECTS"
        assert [p.id for p in handoff.projects] == ["a", "b"]
        assert handoff.projects[1].fixtures == {"x.txt": "hi"}

    def test_prose_around_json_is_tolerated(self):
        handoff = parse_handoff(
            'Sure! Here is my decision:\n{"decision": "SATURATED", "rationale": "done"}\n'
        )
        assert handoff.decision == "SATURATED"

    def test_terminal_decision_may_not_carry_projects(self):
        with pytest.raises(ModelResponseError):
            parse_handoff(
                '{"decision": "SATURATED", "projects": [{"id": "a", "instruction": "x"}]}'
            )

    def test_procuring_decision_requires_projects(self):
        with pytest.raises(ModelResponseError):
            parse_handoff('{"decision": "PROJECTS", "projects": []}')

    def test_duplicate_project_ids_rejected(self):
        with pytest.raises(ModelResponseError):
            parse_handoff(
                '{"decision": "PROJECTS", "projects": ['
                '{"id": "a", "instruction": "1"}, {"id": "a", "instruction": "2"}]}'
            )

    def test_single_project_shorthand(self):
        handoff = parse_handoff(
            '{"decision": "PROJECT", "project": {"id": "p", "instruction": "do it"}}'
        )
        assert [p.id for p in handoff.projects] == ["p"]


class TestMemoryEdits:
    def test_write_and_delete(self):
        edits = parse_memory_edits(
            "```memory:write notes/a.md\nENV_FACT: x\n```\n\n```memory:delete notes/b.md\n```"
        )
        assert [(e.op, e.path) for e in edits] == [
            ("write", "notes/a.md"),
            ("delete", "notes/b.md"),
        ]
        assert edits[0].content == "ENV_FACT: x\n"

    def test_colon_form_is_accepted(self):
        edits = parse_memory_edits("```memory:write:notes/a.md\nbody\n```")
        assert edits[0].path == "notes/a.md"

    @pytest.mark.parametrize("path", ["/etc/passwd", "~/secrets", "../../escape.md", ".."])
    def test_paths_may_not_escape_the_bank(self, path):
        with pytest.raises(ModelResponseError):
            parse_memory_edits(f"```memory:write {path}\nbody\n```")

    def test_missing_path_is_an_error(self):
        with pytest.raises(ModelResponseError):
            parse_memory_edits("```memory:write\nbody\n```")

    def test_non_memory_fences_are_ignored(self):
        assert parse_memory_edits("```python\nprint(1)\n```") == []

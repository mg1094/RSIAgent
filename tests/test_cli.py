"""CLI surface.

The ``run`` path is the one a user actually touches, so it gets exercised end
to end here rather than trusted.  Every test drives the real command line
against the real example in ``examples/custom_task``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsiagent.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "custom_task"
TASK = f"{EXAMPLE / 'task.py'}:task"
EVALUATOR = f"{EXAMPLE / 'task.py'}:evaluator"
POLICIES = f"{EXAMPLE / 'policies.py'}:policies"


def run(args: list[str]) -> int:
    return main(args)


# --------------------------------------------------------------------------
def test_task_and_evaluator_exist():
    """Guards the documented ``file.py:attribute`` convention."""
    assert EXAMPLE.exists(), "examples/custom_task is part of the documented entry path"
    assert "task = TaskQuery(" in (EXAMPLE / "task.py").read_text()
    assert "evaluator = ScriptedEvaluator(" in (EXAMPLE / "task.py").read_text()
    assert "policies = {" in (EXAMPLE / "policies.py").read_text()


def test_baseline_arm_scores_zero(tmp_path: Path, capsys):
    """Memory disabled: the actor guesses wrong and stays wrong."""
    code = run(
        [
            "run",
            "--task",
            TASK,
            "--evaluator",
            EVALUATOR,
            "--policies",
            POLICIES,
            "--arm",
            "baseline",
            "--out",
            str(tmp_path / "base"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "partial=0.0000" in out


def test_full_arm_learns_and_passes(tmp_path: Path, capsys):
    """The whole point: a failed attempt becomes memory, and the next one passes."""
    code = run(
        [
            "run",
            "--task",
            TASK,
            "--evaluator",
            EVALUATOR,
            "--policies",
            POLICIES,
            "--arm",
            "rsi",
            "--out",
            str(tmp_path / "rsi"),
            "--quiet",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "partial=1.0000" in out
    assert "binary=True" in out


def test_json_output_is_machine_readable(tmp_path: Path, capsys):
    run(
        [
            "run",
            "--task",
            TASK,
            "--evaluator",
            EVALUATOR,
            "--policies",
            POLICIES,
            "--arm",
            "rsi",
            "--out",
            str(tmp_path / "rsi"),
            "--quiet",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") :])
    assert payload["task"] == "write-42"
    assert payload["status"] == "completed"
    assert payload["partial"] == pytest.approx(1.0)
    assert payload["memory"]["file_count"] == 1


def test_inspect_and_memory_read_a_finished_run(tmp_path: Path, capsys):
    out_dir = tmp_path / "rsi"
    run(
        [
            "run",
            "--task",
            TASK,
            "--evaluator",
            EVALUATOR,
            "--policies",
            POLICIES,
            "--arm",
            "rsi",
            "--out",
            str(out_dir),
            "--quiet",
        ]
    )
    capsys.readouterr()

    assert run(["inspect", str(out_dir)]) == 0
    report = capsys.readouterr().out
    assert "memory growth:" in report
    assert "target-cycle" in report

    assert run(["memory", str(out_dir)]) == 0
    memory = capsys.readouterr().out
    assert "notes/answer.md" in memory
    assert "The correct value is 42, not 41." in memory


def test_missing_task_file_is_a_clean_error(tmp_path: Path, capsys):
    code = run(
        [
            "run",
            "--task",
            f"{tmp_path / 'nope.py'}:task",
            "--policies",
            POLICIES,
            "--arm",
            "rsi",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_bad_attribute_is_a_clean_error(tmp_path: Path, capsys):
    code = run(
        [
            "run",
            "--task",
            f"{EXAMPLE / 'task.py'}:nope",
            "--policies",
            POLICIES,
            "--arm",
            "rsi",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    assert "no attribute" in capsys.readouterr().err


def test_incomplete_policies_mapping_is_rejected(tmp_path: Path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text("policies = {'actor': lambda m: ''}\n")
    code = run(
        [
            "run",
            "--task",
            TASK,
            "--evaluator",
            EVALUATOR,
            "--policies",
            f"{bad}:policies",
            "--arm",
            "rsi",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    assert "missing" in capsys.readouterr().err


def test_no_api_key_gives_an_actionable_message(tmp_path: Path, capsys, monkeypatch):
    """Without --policies and without credentials, say what to do about it."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = run(
        [
            "run",
            "--task",
            TASK,
            "--evaluator",
            EVALUATOR,
            "--arm",
            "rsi",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "no API key" in err
    assert "--policies" in err or "ScriptedClient" in err


def test_validate_accepts_and_rejects(tmp_path: Path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"run_dir": "runs/x", "stages": ["brs", "drs", "eval"]}))
    assert run(["validate", str(good)]) == 0
    assert "valid" in capsys.readouterr().out

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"run_dir": "runs/x", "stages": ["brs"]}))
    assert run(["validate", str(bad)]) == 1
    assert "eval" in capsys.readouterr().err


def test_no_command_prints_help(capsys):
    assert run([]) == 0
    assert "usage:" in capsys.readouterr().out

"""Parsers for the structured replies each role must produce.

Model output is adversarial input.  Everything here is written to fail loudly
(:class:`~rsiagent.errors.ModelResponseError`) rather than to guess, because a
mis-parsed curriculum handoff or verdict is exactly the kind of silent
corruption that turns into a wrong memory update three phases later.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .errors import ModelResponseError

_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_ACTION_RE = re.compile(r"^\s*ACTION:\s*(\w+)", re.MULTILINE)
_FIELD_RE = re.compile(r"^\s*([A-Z_]+):\s*(.*)$", re.MULTILINE)


# --------------------------------------------------------------------------
# Actor actions
# --------------------------------------------------------------------------
@dataclass
class ActorAction:
    """One parsed actor reply."""

    kind: str  # "program" | "look" | "ask" | "done"
    program: str = ""
    program_kind: str = "python"  # "python" | "bash"
    hint: str = ""
    question: str = ""
    summary: str = ""
    raw: str = ""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.kind == "program":
            first = self.program.strip().splitlines()[0] if self.program.strip() else ""
            return f"ActorAction(program[{self.program_kind}]: {first[:60]!r})"
        return f"ActorAction({self.kind})"


def parse_actor_action(text: str) -> ActorAction:
    """Parse one actor reply into an action.

    A fenced program takes precedence over an ``ACTION:`` line: models often
    narrate "ACTION: done" and then still emit the program that proves it, and
    running the program is the more informative choice.
    """
    fences = _FENCE_RE.findall(text or "")
    for info, body in fences:
        tag = info.strip().lower()
        if tag in ("python", "python3", "py"):
            return ActorAction(kind="program", program=body, program_kind="python", raw=text)
        if tag in ("bash", "sh", "shell"):
            return ActorAction(kind="program", program=body, program_kind="bash", raw=text)

    match = _ACTION_RE.search(text or "")
    if not match:
        raise ModelResponseError(
            "actor reply contains neither a ```python/```bash program nor an ACTION: line; "
            f"got: {text[:200]!r}"
        )

    verb = match.group(1).lower()
    fields = {k.lower(): v.strip() for k, v in _FIELD_RE.findall(text)}

    if verb == "look":
        return ActorAction(kind="look", hint=fields.get("hint", ""), raw=text)
    if verb == "ask":
        return ActorAction(kind="ask", question=fields.get("question", ""), raw=text)
    if verb in ("done", "finish", "submit"):
        return ActorAction(kind="done", summary=fields.get("summary", ""), raw=text)

    raise ModelResponseError(f"unknown actor ACTION: {verb!r}")


# --------------------------------------------------------------------------
# Verifier verdicts
# --------------------------------------------------------------------------
@dataclass
class ParsedVerdict:
    status: str  # PASS | FAIL | UNVERIFIED
    findings: str = ""
    raw: str = ""


def parse_verdict(
    text: str, *, allowed: tuple[str, ...] = ("PASS", "FAIL", "UNVERIFIED")
) -> ParsedVerdict:
    """Parse a verifier reply.

    The *last* verdict-shaped line wins, because a verifier that reasons out
    loud may restate the task before concluding.
    """
    statuses = "|".join(allowed)
    pattern = re.compile(rf"VERDICT\s*:\s*({statuses})\b", re.IGNORECASE)
    matches = list(pattern.finditer(text or ""))
    if not matches:
        raise ModelResponseError(
            f"verifier reply has no VERDICT among {allowed}; got: {(text or '')[:200]!r}"
        )

    final = matches[-1]
    status = final.group(1).upper()

    findings_match = re.search(r"FINDINGS\s*:\s*(.*)", text[final.end() :], re.DOTALL)
    findings = findings_match.group(1).strip() if findings_match else ""
    if not findings:
        # Fall back to whatever preceded the verdict.
        findings = text[: final.start()].strip()
    return ParsedVerdict(status=status, findings=findings, raw=text)


# --------------------------------------------------------------------------
# Curriculum handoffs
# --------------------------------------------------------------------------
@dataclass
class ParsedProject:
    id: str
    instruction: str
    fixtures: dict[str, str] = field(default_factory=dict)


@dataclass
class ParsedHandoff:
    decision: str  # PROJECTS | PROJECT | SATURATED | STALLED | READY_FOR_TARGET
    rationale: str = ""
    projects: list[ParsedProject] = field(default_factory=list)
    raw: str = ""


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a reply, tolerating prose around it."""
    for info, body in _FENCE_RE.findall(text or ""):
        if "json" in info.strip().lower() or body.lstrip().startswith("{"):
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                pass
    stripped = (text or "").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelResponseError(f"curriculum reply is not valid JSON: {exc}") from exc
    raise ModelResponseError(f"curriculum reply contains no JSON object: {stripped[:200]!r}")


def _coerce_project(blob: Any) -> ParsedProject:
    if not isinstance(blob, dict):
        raise ModelResponseError(f"project entry must be an object, got {type(blob).__name__}")
    instruction = str(blob.get("instruction") or "").strip()
    if not instruction:
        raise ModelResponseError("project entry is missing a non-empty 'instruction'")
    raw_fixtures = blob.get("fixtures") or {}
    if not isinstance(raw_fixtures, dict):
        raise ModelResponseError("'fixtures' must be an object mapping path -> content")
    fixtures = {
        str(k): (v if isinstance(v, str) else json.dumps(v)) for k, v in raw_fixtures.items()
    }
    return ParsedProject(
        id=str(blob.get("id") or f"p{abs(hash(instruction)) % 100_000:05d}"),
        instruction=instruction,
        fixtures=fixtures,
    )


def parse_handoff(text: str) -> ParsedHandoff:
    blob = _extract_json(text)
    decision = str(blob.get("decision") or "").strip().upper()
    if not decision:
        raise ModelResponseError("curriculum handoff has no 'decision'")

    projects_raw = blob.get("projects")
    if projects_raw is None and blob.get("project"):
        projects_raw = [blob["project"]]
    projects = [_coerce_project(p) for p in (projects_raw or [])]

    handoff = ParsedHandoff(
        decision=decision,
        rationale=str(blob.get("rationale") or "").strip(),
        projects=projects,
        raw=text,
    )
    _validate_handoff(handoff)
    return handoff


def _validate_handoff(handoff: ParsedHandoff) -> None:
    """Reject handoffs whose decision and payload disagree."""
    procuring = {"PROJECTS", "PROJECT"}
    terminal = {"SATURATED", "STALLED", "READY_FOR_TARGET"}

    if handoff.decision in procuring and not handoff.projects:
        raise ModelResponseError(f"decision {handoff.decision} requires at least one project")
    if handoff.decision in terminal and handoff.projects:
        raise ModelResponseError(
            f"decision {handoff.decision} must not carry projects (got {len(handoff.projects)})"
        )
    if handoff.decision not in procuring | terminal:
        raise ModelResponseError(f"unknown curriculum decision: {handoff.decision!r}")

    seen: set[str] = set()
    for project in handoff.projects:
        if project.id in seen:
            raise ModelResponseError(f"duplicate project id in one handoff: {project.id}")
        seen.add(project.id)


# --------------------------------------------------------------------------
# Memory edit blocks
# --------------------------------------------------------------------------
@dataclass
class MemoryEdit:
    op: str  # "write" | "delete"
    path: str
    content: str = ""


def parse_memory_edits(text: str) -> list[MemoryEdit]:
    """Extract ``memory:write`` / ``memory:delete`` blocks from a reply.

    Blocks are applied in the order they appear, so a later write to a path
    overrides an earlier one.

    Both ``memory:write path/to.md`` and ``memory:write:path/to.md`` are
    accepted — models drift between the two, and the difference carries no
    meaning worth failing a lineage over.
    """
    edits: list[MemoryEdit] = []
    for info, body in _FENCE_RE.findall(text or ""):
        spec = info.strip()
        if not spec.lower().startswith("memory:"):
            continue
        head, _, rest = spec.partition(":")
        del head
        pieces = re.split(r"[:\s]+", rest.strip(), maxsplit=1)
        op = pieces[0].strip().lower()
        path = pieces[1].strip() if len(pieces) > 1 else ""
        if not path:
            raise ModelResponseError(f"memory block without a path: {spec!r}")
        _reject_unsafe_path(path)
        if op == "write":
            edits.append(MemoryEdit(op="write", path=path, content=body))
        elif op == "delete":
            edits.append(MemoryEdit(op="delete", path=path))
        else:
            raise ModelResponseError(f"unknown memory operation: {op!r} in {spec!r}")
    return edits


def _reject_unsafe_path(path: str) -> None:
    """Memory paths must stay inside the bank."""
    normalised = path.replace("\\", "/").strip()
    if normalised.startswith("/") or normalised.startswith("~"):
        raise ModelResponseError(f"memory path must be relative: {path!r}")
    parts = [p for p in normalised.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ModelResponseError(f"memory path must not escape the bank: {path!r}")
    if not parts:
        raise ModelResponseError("memory path is empty")


def parse_diagnosis(text: str) -> str:
    """Strip the trailing ``ACTION: done`` from a diagnosis reply."""
    cleaned = _ACTION_RE.sub("", text or "")
    return cleaned.strip()

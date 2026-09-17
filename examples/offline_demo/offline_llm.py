"""Deterministic stand-ins for the four model roles.

These are **not language models**.  They are small reactive policies that stand
in for GLM-5.3 and Kimi-K3 so the lifecycle can run offline.  What they do
reproduce faithfully is the *information flow* the paper depends on:

* the **actor** writes real programs, runs them, and reads real exec output;
* the **verifier** probes the real artifact tree and grounds its verdict in what
  it actually observes;
* the **curriculum** chooses the next experience from the previous *grounded
  outcome*, not from a script;
* the **learner** writes memory only from verifier findings, and only for
  facts it does not already know.

The one genuinely scripted part is the *vocabulary*: a real model would author
project instructions and programs from scratch, while these policies select from
templates keyed by a small set of environment facts.  That is the substitution
to keep in mind when reading a demo transcript — the score it produces is a
demonstration of the protocol, not evidence about a model.

The memory protocol used here is deliberately plain: a memory file lists
``ENV_FACT: <name>`` lines, and the actor reads those names back out of the
memory block in its prompt.  That is a stand-in for an agent reading its own
written procedure notes, and it keeps the causal chain — evidence, finding,
memory, behaviour change — inspectable end to end.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from rsiagent.llm import Message, RoleClients, ScriptedClient

from .demo_env import EXPECTED, FIXTURES

# --------------------------------------------------------------------------
# Environment facts
# --------------------------------------------------------------------------
FACT_BOM = "csv_bom"
FACT_DELIMITER = "csv_semicolon"
FACT_DECIMAL = "amount_decimal_comma"
FACT_DATE = "date_dmy"

FACT_NOTES = {
    FACT_BOM: (
        "csv_bom",
        "`data/sales_north.csv` begins with a UTF-8 BOM. Opening it with "
        "encoding='utf-8' leaves the first header as '\\ufeffregion', so "
        "row['region'] raises KeyError. Open with encoding='utf-8-sig'.",
    ),
    FACT_DELIMITER: (
        "csv_semicolon",
        "Delimiters are not uniform across `data/`. `sales_north.csv` is "
        "comma-separated; `sales_south.csv` and `sales_east.csv` are "
        "semicolon-separated. Sniff the delimiter from the header line "
        "instead of assuming ','.",
    ),
    FACT_DECIMAL: (
        "amount_decimal_comma",
        "`sales_east.csv` writes money in European convention: '.' is the "
        "thousands separator and ',' is the decimal separator ('1.234,56'). "
        "The other exports use '.' as the decimal separator. Decide per value — "
        "if it contains a comma, strip '.' and replace ',' with '.'; otherwise "
        "parse it unchanged. Applying the European transform to the whole run "
        "would inflate the dot-decimal exports by 100x.",
    ),
    FACT_DATE: (
        "date_dmy",
        "Every date column is DD/MM/YYYY. There is no ISO 8601 column "
        "anywhere in `data/`. Convert with parts = s.split('/') and build "
        "'YYYY-MM' as parts[2] + '-' + parts[1].",
    ),
}

_FACT_PATTERNS = {
    FACT_BOM: re.compile(r"BOM|\\ufeff|utf-8-sig", re.IGNORECASE),
    FACT_DELIMITER: re.compile(r"semicolon|delimiter|';'|\";\"", re.IGNORECASE),
    FACT_DECIMAL: re.compile(r"decimal comma|thousands separator|1\.234,56", re.IGNORECASE),
    FACT_DATE: re.compile(r"DD/MM/YYYY|date convention|not ISO", re.IGNORECASE),
}

ALL_FACTS = (FACT_BOM, FACT_DELIMITER, FACT_DECIMAL, FACT_DATE)


def facts_in(text: str) -> set[str]:
    """Which environment facts a piece of verifier prose is about."""
    return {fact for fact, pattern in _FACT_PATTERNS.items() if pattern.search(text or "")}


def facts_from_memory(memory_text: str) -> set[str]:
    """Read back the facts a previous learning step wrote down."""
    return set(re.findall(r"^ENV_FACT:\s*(\S+)\s*$", memory_text or "", re.MULTILINE))


# --------------------------------------------------------------------------
# Program templates (what the actor submits)
# --------------------------------------------------------------------------
PRELUDE = '''\
import csv, glob, json, os

CSV_BOM = {bom!r}
SNIFF_DELIMITER = {sniff!r}
AMOUNT_DECIMAL_COMMA = {decimal!r}
DATE_DMY = {dmy!r}


def read_rows(path):
    encoding = "utf-8-sig" if CSV_BOM else "utf-8"
    with open(path, newline="", encoding=encoding) as handle:
        header = handle.readline()
        handle.seek(0)
        delimiter = ","
        if SNIFF_DELIMITER and header.count(";") > header.count(","):
            delimiter = ";"
        return list(csv.DictReader(handle, delimiter=delimiter))


def to_amount(text):
    cleaned = (text or "").strip()
    # Detected per value, not per run: only the exports that actually use a
    # decimal comma get the European transform. Applying it globally would
    # turn "120.50" into 12050.0 -- the exact overgeneralisation the paper's
    # reconciliation step exists to catch.
    if AMOUNT_DECIMAL_COMMA and "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def to_month(text):
    cleaned = (text or "").strip()
    if DATE_DMY:
        parts = cleaned.split("/")
        if len(parts) == 3:
            return parts[2] + "-" + parts[1]
    return cleaned[:7]


def write_json(name, payload):
    os.makedirs("out", exist_ok=True)
    with open(name, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload))
'''


def _flags(facts: set[str]) -> dict[str, bool]:
    return {
        "bom": FACT_BOM in facts,
        "sniff": FACT_DELIMITER in facts,
        "decimal": FACT_DECIMAL in facts,
        "dmy": FACT_DATE in facts,
    }


def program_regions(facts: set[str]) -> str:
    """Practice p1 — surface the BOM by reading a column by name.

    The failure carries the header it actually saw, so the verifier can tell a
    mangled first column (BOM) from a header that never split (wrong delimiter)
    instead of guessing from the exception type alone.
    """
    return PRELUDE.format(**_flags(facts)) + '''
out = {}
for path in sorted(glob.glob("data/*.csv")):
    name = os.path.basename(path)
    try:
        rows = read_rows(path)
        if not rows:
            out[name] = []
            continue
        if "region" not in rows[0]:
            raise KeyError("no 'region' column; header read as " + repr(list(rows[0])))
        out[name] = sorted({row["region"] for row in rows})
    except Exception as exc:
        out[name] = "ERROR: " + type(exc).__name__ + ": " + str(exc)
write_json("out/regions.json", out)
'''


def program_sums(facts: set[str]) -> str:
    """Practice p2/p3 — surface the delimiter, then the decimal comma."""
    return PRELUDE.format(**_flags(facts)) + '''
out = {}
for path in sorted(glob.glob("data/*.csv")):
    name = os.path.basename(path)
    try:
        rows = read_rows(path)
        if rows and "amount" not in rows[0]:
            raise KeyError("no 'amount' column; header read as " + repr(list(rows[0])))
        total = 0.0
        seen = 0
        for row in rows:
            value = to_amount(row.get("amount"))
            if value is None:
                raise ValueError("unparseable amount: " + repr(row.get("amount")))
            total += value
            seen += 1
        out[name] = {"rows": seen, "total": round(total, 2)}
    except Exception as exc:
        out[name] = "ERROR: " + type(exc).__name__ + ": " + str(exc)
write_json("out/sums.json", out)
'''


def program_dates(facts: set[str]) -> str:
    """Practice p4 — surface the date convention."""
    return PRELUDE.format(**_flags(facts)) + '''
out = {}
for path in sorted(glob.glob("data/*.csv")):
    name = os.path.basename(path)
    try:
        months = sorted({to_month(row.get("date")) for row in read_rows(path)})
        out[name] = months
    except Exception as exc:
        out[name] = "ERROR: " + type(exc).__name__ + ": " + str(exc)
write_json("out/dates.json", out)
'''


def program_report(facts: set[str]) -> str:
    """The target: build out/report.json."""
    return PRELUDE.format(**_flags(facts)) + '''
totals = {}
months = []
parsed = 0
for path in sorted(glob.glob("data/*.csv")):
    try:
        rows = read_rows(path)
    except Exception:
        continue
    ok = True
    for row in rows:
        region = row.get("region")
        value = to_amount(row.get("amount"))
        if region is None or value is None:
            ok = False
            continue
        totals[region] = round(totals.get(region, 0.0) + value, 2)
        months.append(to_month(row.get("date")))
    if ok and rows:
        parsed += 1

write_json("out/report.json", {
    "month": max(months) if months else None,
    "source_files": parsed,
    "total_by_region": totals,
})
'''


# --------------------------------------------------------------------------
# Project catalogue (stands in for curriculum-authored instructions)
# --------------------------------------------------------------------------
PROJECTS = {
    "p-bom": {
        "instruction": (
            "For every CSV under `data/`, report the distinct values of its "
            "`region` column. Write `out/regions.json` mapping each filename to "
            "a sorted list of regions, or to an \"ERROR: ...\" string if the "
            "file cannot be read."
        ),
        "program": program_regions,
        "watch": "regions",
        "marker": "`region` column",
    },
    "p-delim": {
        "instruction": (
            "For every CSV under `data/`, sum the `amount` column. Write "
            "`out/sums.json` mapping each filename to an object with `rows` and "
            "`total`, or to an \"ERROR: ...\" string if any amount is "
            "unparseable."
        ),
        "program": program_sums,
        "watch": "sums",
        "marker": "sum the `amount` column",
    },
    "p-decimal": {
        "instruction": (
            "Re-check the money handling. For every CSV under `data/`, total the "
            "`amount` column and write `out/sums.json` mapping each filename to "
            "an object with `rows` and `total`. Every amount present in the "
            "files must be included at its true value."
        ),
        "program": program_sums,
        "watch": "sums",
        "marker": "money handling",
    },
    "p-dates": {
        "instruction": (
            "For every CSV under `data/`, report the distinct values of its "
            "`date` column as `YYYY-MM`. Write `out/dates.json` mapping each "
            "filename to a sorted list of months."
        ),
        "program": program_dates,
        "watch": "dates",
        "marker": "`date` column as `YYYY-MM`",
    },
}


# --------------------------------------------------------------------------
# Actor policy
# --------------------------------------------------------------------------
class ActorPolicy:
    """Writes programs, and later distills what they produced."""

    def __init__(self) -> None:
        self._practice_cursor = 0

    def __call__(self, messages: Sequence[Message]) -> str:
        last = messages[-1].content
        memory_text = "\n".join(m.content for m in messages if m.role == "user")
        facts = facts_from_memory(memory_text)
        known = facts_from_memory(memory_text)

        if "Learning step 1 of 2" in last:
            return self._distill(last, known)
        if "Learning step 2 of 2" in last:
            return "ACTION: done\nSUMMARY: reconciliation complete; no contradictions found."
        if "Learning diagnosis" in last:
            return (
                "The failures are all format conventions rather than logic errors. "
                "Reliable: the directory layout, the column names, the fact that every "
                "file is a regional export. Uncertain: whether any export uses a "
                "different date convention. A discriminating case would be an export "
                "from a fourth office, if one exists.\n\nACTION: done"
            )

        # --- work phase -------------------------------------------------
        project = self._current_project(messages)
        if last.lstrip().startswith("$ ("):
            return (
                "ACTION: done\n"
                "SUMMARY: ran the program and recorded its output; the artifact is "
                "on disk for inspection."
            )
        if project is not None:
            facts = self._facts_at_work_time(messages, project)
            return "```python\n" + project["program"](facts) + "```"

        # Target attempt.
        if "## Task" in last or "## Verifier findings" in last:
            return "```python\n" + program_report(facts) + "```"
        return "ACTION: done\nSUMMARY: nothing further to run."

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _current_project(messages: Sequence[Message]) -> dict | None:
        for message in messages:
            if message.role != "user":
                continue
            match = re.search(r"\*\*Project id:\*\*\s*(\S+)", message.content)
            if match and match.group(1) in PROJECTS:
                return PROJECTS[match.group(1)]
        return None

    @staticmethod
    def _facts_at_work_time(messages: Sequence[Message], project: dict) -> set[str]:
        """Facts as of the *opening* prompt, not the learning prompt.

        A branch cannot observe sibling work, so it must not be credited with
        facts another branch committed while it was running.  Reading only the
        opening message enforces that.
        """
        opening = next(
            (
                m.content
                for m in messages
                if m.role == "user" and "## Practice project" in m.content
            ),
            "",
        )
        return facts_from_memory(opening)

    def _distill(self, prompt: str, already_known: set[str]) -> str:
        findings = prompt.split("**Verifier findings:**", 1)[-1]
        discovered = facts_in(findings) - already_known
        if not discovered:
            return (
                "The verifier's findings name conventions already recorded in memory. "
                "No new claim is supported strongly enough to add.\n\nACTION: done"
            )
        blocks = []
        for fact in ALL_FACTS:
            if fact not in discovered:
                continue
            name, note = FACT_NOTES[fact]
            blocks.append(f"```memory:write env/{name}.md\nENV_FACT: {name}\n\n{note}\n```")
        blocks.append("ACTION: done")
        return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Verifier policy
# --------------------------------------------------------------------------
class VerifierPolicy:
    """Probes the real artifact tree, then judges against ground truth.

    The judgement is deliberately *diagnostic*: a FAIL names the mechanism, not
    just the symptom.  That is what the paper's distillation step consumes, and
    a verdict that only said "wrong totals" would leave nothing to learn from.
    """

    def __call__(self, messages: Sequence[Message]) -> str:
        last = messages[-1].content
        if last.lstrip().startswith("$ ("):
            return self._verdict_from_probe(last)
        return "```python\n" + self._probe(last) + "```"

    @staticmethod
    def _probe(request: str) -> str:
        watch = "report"
        if "This is a practice project." in request:
            for spec in PROJECTS.values():
                if spec["marker"] in request:
                    watch = spec["watch"]
                    break
        return (
            "import json, os\n"
            "out = {}\n"
            "for name in ['regions', 'sums', 'dates', 'report']:\n"
            "    path = 'out/' + name + '.json'\n"
            "    if os.path.exists(path):\n"
            "        out[name] = json.load(open(path))\n"
            "print(json.dumps({'watch': %r, 'artifacts': out}, indent=2))\n" % watch
        )

    def _verdict_from_probe(self, last: str) -> str:
        try:
            blob = json.loads(last.split("\n", 1)[1])
            watch = blob["watch"]
            artifacts = blob["artifacts"]
        except (ValueError, IndexError, KeyError):
            return "VERDICT: UNVERIFIED\nFINDINGS: probe output could not be parsed."

        judge = {
            "report": self._judge_report,
            "regions": self._judge_regions,
            "sums": self._judge_sums,
            "dates": self._judge_dates,
        }.get(watch)
        if judge is None:
            return "VERDICT: UNVERIFIED\nFINDINGS: unknown inspection target."
        return judge(artifacts.get(watch))

    # -- individual judgements -------------------------------------------
    @staticmethod
    def _judge_regions(payload) -> str:
        if not isinstance(payload, dict):
            return "VERDICT: FAIL\nFINDINGS: out/regions.json was not produced."
        broken = {k: v for k, v in payload.items() if isinstance(v, str) and v.startswith("ERROR")}
        if not broken:
            return "VERDICT: PASS\nFINDINGS: every file yielded its region list."

        # Distinguish the two ways a name lookup fails here: a header that still
        # carries the BOM leaves '\\ufeffregion', while a header that never split
        # leaves 'region;date;amount'.  The probe reports the header, so the
        # verdict can say which happened rather than guess.
        mangled = sorted(k for k, v in broken.items() if "\\ufeff" in v)
        if mangled:
            return (
                "VERDICT: FAIL\n"
                f"FINDINGS: {mangled[0]} could not be read: {broken[mangled[0]]}. The "
                "header names carry a leading '\\ufeff', which is a UTF-8 BOM. The "
                "file does contain a `region` column — the lookup failed because the "
                "BOM is glued to the first header. Open with encoding='utf-8-sig'."
            )
        first = sorted(broken)[0]
        return (
            "VERDICT: FAIL\n"
            f"FINDINGS: {first} could not be read: {broken[first]}. Its header did "
            "not split into columns, so the delimiter is not ','. The southern and "
            "eastern exports are semicolon-separated; detect the delimiter from the "
            "header line instead of assuming it."
        )

    @staticmethod
    def _judge_sums(payload) -> str:
        if not isinstance(payload, dict):
            return "VERDICT: FAIL\nFINDINGS: out/sums.json was not produced."

        for name in sorted(payload):
            entry = payload[name]
            if not isinstance(entry, str):
                continue
            if "KeyError" in entry:
                return (
                    "VERDICT: FAIL\n"
                    f"FINDINGS: {name} could not be read: {entry}. The header names "
                    "do not match, which happens when the file is not "
                    "comma-separated. Use a semicolon for the southern and eastern "
                    "exports — or sniff the delimiter from the header line."
                )
            if "unparseable amount" in entry:
                return (
                    "VERDICT: FAIL\n"
                    f"FINDINGS: {name} has an amount that float() cannot read: {entry}. "
                    "This export writes money in European convention — '.' is the "
                    "thousands separator and ',' is the decimal comma, as in "
                    "'1.234,56'. Strip '.' and replace ',' with '.' before float()."
                )
            return (
                "VERDICT: FAIL\n"
                f"FINDINGS: {name} could not be read: {entry}."
            )

        for name in sorted(payload):
            entry = payload[name]
            expected = EXPECTED.sums.get(name)
            if expected is None or not isinstance(entry, dict):
                continue
            total = entry.get("total")
            if isinstance(total, (int, float)) and abs(total - expected) > 0.005:
                return (
                    "VERDICT: FAIL\n"
                    f"FINDINGS: {name} sums to {total} but its true total is {expected}."
                )
        return "VERDICT: PASS\nFINDINGS: every file summed to its true total."

    @staticmethod
    def _judge_dates(payload) -> str:
        if not isinstance(payload, dict):
            return "VERDICT: FAIL\nFINDINGS: out/dates.json was not produced."
        for name in sorted(payload):
            months = payload[name]
            if not isinstance(months, list) or not months:
                continue
            bad = [m for m in months if not re.fullmatch(r"\d{4}-\d{2}", str(m))]
            if bad:
                return (
                    "VERDICT: FAIL\n"
                    f"FINDINGS: {name} reports {bad[0]!r}, which is not YYYY-MM. The "
                    "date column is DD/MM/YYYY everywhere under data/ — there is no "
                    "ISO 8601 date in the directory. Split on '/' and rebuild the "
                    "month as parts[2] + '-' + parts[1]."
                )
        return "VERDICT: PASS\nFINDINGS: every file yielded YYYY-MM months."

    @staticmethod
    def _judge_report(payload) -> str:
        if not isinstance(payload, dict):
            return (
                "VERDICT: FAIL\n"
                "FINDINGS: out/report.json was not produced or is not a JSON object."
            )
        required = {"month", "source_files", "total_by_region"}
        missing = required - set(payload)
        if missing:
            return (
                "VERDICT: FAIL\n"
                f"FINDINGS: out/report.json is missing required keys: {sorted(missing)}."
            )

        totals = payload.get("total_by_region") or {}
        expected_regions = set(EXPECTED.totals)
        if set(totals) != expected_regions:
            return (
                "VERDICT: FAIL\n"
                f"FINDINGS: total_by_region covers {sorted(totals)} but the exports "
                f"cover {sorted(expected_regions)}, so at least one export was not "
                "parsed at all. Its amounts are written as '1.234,56' — a thousands "
                "separator and a decimal comma — and float() reads only the '1'."
            )
        for region in sorted(EXPECTED.totals):
            expected = EXPECTED.totals[region]
            value = totals.get(region)
            if not isinstance(value, (int, float)) or abs(value - expected) > 0.005:
                return (
                    "VERDICT: FAIL\n"
                    f"FINDINGS: total_by_region[{region!r}] is {value}, but the true "
                    f"total is {expected}. The eastern export writes its amounts as "
                    "'1.234,56', so float() truncates them at the comma."
                )
        if payload.get("month") != EXPECTED.month:
            return (
                "VERDICT: FAIL\n"
                f"FINDINGS: month is {payload.get('month')!r}, but the most recent "
                f"date under data/ falls in {EXPECTED.month}. The date column is "
                "DD/MM/YYYY, so its first seven characters are not a month."
            )
        return (
            "VERDICT: PASS\n"
            "FINDINGS: out/report.json carries the required keys; every region total "
            "matches the exports at true value; source_files accounts for all three "
            "exports; month matches the most recent date."
        )


# --------------------------------------------------------------------------
# Curriculum policy
# --------------------------------------------------------------------------
class CurriculumPolicy:
    """Chooses the next experience from the last grounded outcome."""

    def __init__(self) -> None:
        self._pending = ["p-decimal", "p-dates"]
        self._wave = 0

    def __call__(self, messages: Sequence[Message]) -> str:
        prompt = messages[-1].content
        if "Phase 1 (Broad Recursive Self-exploration)" in prompt:
            return self._brs()
        return self._drs(prompt)

    def _brs(self) -> str:
        self._wave += 1
        if self._wave == 1:
            return _handoff(
                "PROJECTS",
                "Memory is empty. Two capabilities are prerequisites for any rollup "
                "over this directory: reading a column by name from each export, and "
                "reading money amounts. Both are prerequisites for the target, and "
                "neither requires touching the target itself.",
                [
                    _project("p-bom", PROJECTS["p-bom"]["instruction"]),
                    _project("p-delim", PROJECTS["p-delim"]["instruction"]),
                ],
            )
        return _handoff(
            "SATURATED",
            "Both prerequisite capabilities are recorded and the exports' structural "
            "conventions now appear in memory. Further broad practice would restate "
            "what is already retained; the remaining uncertainty concerns money and "
            "date conventions, which only a real attempt at the rollup will expose.",
            [],
        )

    def _drs(self, prompt: str) -> str:
        failed = re.search(r"\*\*Verdict:\*\*\s*(\w+)", prompt)
        verdict = failed.group(1) if failed else "UNKNOWN"
        practice_count = _practice_count(prompt)

        if verdict == "FAIL" and self._pending:
            next_id = self._pending.pop(0)
            return _handoff(
                "PROJECT",
                "The attempt got far enough to produce numbers, and the verifier "
                "established which requirement fails. Practice the specific "
                "capability the finding names rather than broadening.",
                [_project(next_id, PROJECTS[next_id]["instruction"])],
            )
        return _handoff(
            "READY_FOR_TARGET",
            f"No remaining practice has greater expected value than returning to the "
            f"target lifecycle (last verdict {verdict}, {practice_count} practice "
            f"project(s) so far). This is a readiness decision, not a correctness "
            f"verdict — the verifier decides correctness.",
            [],
        )


def _practice_count(prompt: str) -> int:
    match = re.search(r"Practice projects used so far\*{0,2}\s*\n+\s*(\d+)", prompt)
    return int(match.group(1)) if match else 0


def _handoff(decision: str, rationale: str, projects: list[dict]) -> str:
    payload = {"decision": decision, "rationale": rationale}
    if projects:
        payload["projects" if decision == "PROJECTS" else "project"] = (
            projects if decision == "PROJECTS" else projects[0]
        )
    return "```json\n" + json.dumps(payload, indent=2) + "\n```"


def _project(project_id: str, instruction: str) -> dict:
    """A curriculum handoff entry, fixtures included.

    "Each project has its own input fixtures, replayed under a common project
    path in an isolated environment" (Appendix A.1).  A practice project that
    needs the exports must ship them; a branch that starts from an empty
    workspace would otherwise pass trivially and teach nothing.
    """
    return {"id": project_id, "instruction": instruction, "fixtures": dict(FIXTURES)}


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------
def build_clients() -> RoleClients:
    """One scripted client per role, matching the paper's role bindings."""
    return RoleClients(
        actor=ScriptedClient(ActorPolicy(), name="scripted-actor"),
        verifier=ScriptedClient(VerifierPolicy(), name="scripted-verifier"),
        curriculum=ScriptedClient(CurriculumPolicy(), name="scripted-curriculum"),
    )

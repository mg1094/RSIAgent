"""The offline demo environment: a small, genuinely quirky data workspace.

The demo exists to prove the *protocol* runs end to end with no API key, no
network, and no VM — not to reproduce a benchmark number.  To make the run
meaningful rather than scripted theatre, the environment is real: the CSVs are
really written to disk, the actor's programs really execute, the verifier really
probes, and the evaluator really inspects the artifact.

Four quirks are planted, each of which a competent engineer would discover by
working in this environment and would otherwise rediscover forever:

===============  ==========================================================
``csv_bom``      ``data/sales_q1.csv`` starts with a UTF-8 BOM, so the first
                 header reads as ``\\ufeffdate`` unless opened ``utf-8-sig``.
``csv_semicolon`` ``data/sales_q2.csv`` and ``q3`` use ``;`` as the delimiter.
``amount_decimal_comma``  ``data/sales_q3.csv`` writes money as ``1.234,56``.
``date_dmy``     Every date is ``DD/MM/YYYY``, never ISO 8601.
===============  ==========================================================

The layout is chosen so the paper's ablation ordering falls out of it: the
BOM and delimiter quirks are discoverable by broad exploration, while the money
and date conventions only surface once a real attempt at the target fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from rsiagent.env.base import Environment
from rsiagent.tasks.spec import ScriptedEvaluator, TaskQuery

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
# Each export comes from one regional office and is internally consistent; the
# offices simply never agreed on a format.  The BOM below is a literal U+FEFF
# at the start of the file, so the quirk is visible in source.
SALES_NORTH = (
    "﻿region,date,amount\n"
    "north,05/01/2026,120.50\n"
    "north,12/01/2026,80.25\n"
    "north,19/01/2026,45.00\n"
    "north,26/02/2026,60.00\n"
)

SALES_SOUTH = (
    "region;date;amount\n"
    "south;03/03/2026;210.00\n"
    "south;15/03/2026;95.50\n"
    "south;28/03/2026;33.25\n"
)

SALES_EAST = (
    "region;date;amount\n"
    "east;07/03/2026;1.234,56\n"
    "east;21/03/2026;500,00\n"
    "east;26/03/2026;75,50\n"
)

README = """\
# Acme Analytics — regional sales exports

Quarterly exports land in `data/`. They are produced by three different
regional offices and have never been normalised.

The monthly rollup is built by `out/report.json`.
"""

FIXTURES: dict[str, str] = {
    "README.md": README,
    "data/sales_north.csv": SALES_NORTH,
    "data/sales_south.csv": SALES_SOUTH,
    "data/sales_east.csv": SALES_EAST,
}

# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
EXPECTED_TOTALS = {"north": 305.75, "south": 338.75, "east": 1810.06}
EXPECTED_MONTH = "2026-03"
EXPECTED_SOURCES = 3
TOLERANCE = 0.005

TARGET_INSTRUCTION = """\
Build the monthly regional rollup for Acme Analytics and write it to
`out/report.json` as JSON with exactly these keys:

- `month`: the reporting month as `YYYY-MM`, taken from the most recent `date`
  in the exports.
- `source_files`: how many of the CSV exports under `data/` you were able to
  parse.
- `total_by_region`: an object mapping each region name to the sum of its
  `amount` values, rounded to 2 decimals.

The rollup must cover every export under `data/`, and every amount must be
included at its true value.
"""

TASK = TaskQuery(
    id="acme-monthly-rollup",
    instruction=TARGET_INSTRUCTION,
    fixtures=FIXTURES,
)


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------
def _load_report(environment: Environment) -> dict | None:
    if not environment.exists("out/report.json"):
        return None
    try:
        blob = json.loads(environment.read_text("out/report.json"))
    except json.JSONDecodeError:
        return None
    return blob if isinstance(blob, dict) else None


def _report_parses(environment: Environment) -> float:
    report = _load_report(environment)
    if report is None:
        return 0.0
    required = {"month", "source_files", "total_by_region"}
    return 1.0 if required <= set(report) else 0.0


def _sources_declared(environment: Environment) -> float:
    report = _load_report(environment)
    if not report:
        return 0.0
    return 1.0 if report.get("source_files") == EXPECTED_SOURCES else 0.0


def _regions_exact(environment: Environment) -> float:
    report = _load_report(environment)
    if not report:
        return 0.0
    totals = report.get("total_by_region")
    if not isinstance(totals, dict):
        return 0.0
    return 1.0 if set(totals) == set(EXPECTED_TOTALS) else 0.0


def _region_check(region: str):
    def check(environment: Environment) -> float:
        report = _load_report(environment)
        if not report:
            return 0.0
        totals = report.get("total_by_region")
        if not isinstance(totals, dict):
            return 0.0
        value = totals.get(region)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return 0.0
        return 1.0 if abs(float(value) - EXPECTED_TOTALS[region]) <= TOLERANCE else 0.0

    return check


def _month_correct(environment: Environment) -> float:
    report = _load_report(environment)
    if not report:
        return 0.0
    return 1.0 if report.get("month") == EXPECTED_MONTH else 0.0


def build_evaluator() -> ScriptedEvaluator:
    """Seven equally weighted checks.

    Equal weighting mirrors the paper's convention that "each task has equal
    weight" applied one level down.  The check set is chosen so the score
    separates cleanly across the conditions the framework is meant to
    distinguish:

    ===========================  =====  ===================================
    Condition                    Score  What it knows
    ===========================  =====  ===================================
    w/o RSI                      1/7    nothing; every export fails to parse
    w/o DRS (BRS only)           4/7    BOM and delimiter
    Full RSI                     7/7    + decimal comma, + date convention
    ===========================  =====  ===================================

    The ordering is not arranged — it falls out of *when* each quirk becomes
    observable.  The BOM and the delimiter surface as soon as the actor tries
    to read the directory, so broad exploration finds them.  The money and date
    conventions only bite once a real attempt at the rollup produces
    plausible-looking wrong numbers, which is exactly the sequencing the paper
    describes for DRS.
    """
    return ScriptedEvaluator(
        {
            "report_parses": _report_parses,
            "source_files": _sources_declared,
            "regions": _regions_exact,
            "total_north": _region_check("north"),
            "total_south": _region_check("south"),
            "total_east": _region_check("east"),
            "month": _month_correct,
        },
        notes="Acme Analytics monthly rollup, 7 equally weighted checks",
    )


@dataclass
class Expected:
    """Ground truth a stand-in verifier compares against."""

    totals: dict[str, float]
    month: str
    row_counts: dict[str, int]
    sums: dict[str, float]


EXPECTED = Expected(
    totals=EXPECTED_TOTALS,
    month=EXPECTED_MONTH,
    row_counts={"sales_north.csv": 4, "sales_south.csv": 3, "sales_east.csv": 3},
    sums={"sales_north.csv": 305.75, "sales_south.csv": 338.75, "sales_east.csv": 1810.06},
)

"""Run configuration.

Mirrors Table A3 of the paper ("Reference agent configuration and exploration
settings").  Defaults here are the paper's OSWorld reference values so that a
default-constructed :class:`RunConfig` documents the reported setup; the
offline demo overrides the model sizes that matter for a laptop.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from .errors import ConfigurationError

StopPolicy = Literal["curriculum_review", "verifier_pass"]
"""How Deep Recursive Self-exploration decides it is finished.

``curriculum_review``
    Default.  A target PASS followed by *no additional practice* may complete
    DRS.  Any additional practice requires another target attempt.
``verifier_pass``
    Stops after a grounded target PASS and its memory consolidation, without a
    curriculum review.  Used by earlier runs in the paper.
"""

VERDICT_STATUSES = ("PASS", "FAIL", "UNVERIFIED")
PRACTICE_VERDICT_STATUSES = ("PASS", "FAIL")


@dataclass(frozen=True)
class RoleModelConfig:
    """Model binding for one agent role.

    The paper runs the actor on a different model from the verifier and
    curriculum agents, and notes that the verifier and curriculum agents "use
    separate contexts despite sharing a model".  Context separation is enforced
    in :mod:`rsiagent.agents.context`; this dataclass only names the model.
    """

    model: str
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 65_536
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExplorationConfig:
    """Exploration budgets and stopping rules (paper Table A3, Appendix C.2)."""

    # --- Stage 1: Broad Recursive Self-exploration -----------------------
    brs_project_budget: int = 8
    """Nominal project budget.  Checked only at complete-wave boundaries, so the
    realized project count can exceed it."""

    brs_concurrency: int = 4
    """Concurrency limit, *not* a required wave width."""

    brs_wave_width: int = 4
    """How many projects to ask the curriculum agent for in one wave.  The paper
    fixes the concurrency limit at four but leaves wave width to the curriculum
    agent; this is the ceiling the prompt states."""

    brs_max_waves: int = 16
    """Hard safety stop.  The paper has no wave cap; saturation is semantic."""

    brs_on_blocked: Literal["halt", "drop_branch"] = "halt"
    """What to do when a wave contains an unresolved branch.

    ``halt`` (default) matches the architecture note that "an incomplete or
    quarantined branch blocks the wave", stopping the lineage with no commits
    from that wave.  ``drop_branch`` continues with the grounded branches only,
    which is more forgiving but breaks the guarantee that every authored project
    contributed to the memory the curriculum agent reasoned over."""

    # --- Stage 2: Deep Recursive Self-exploration ------------------------
    stop_policy: StopPolicy = "curriculum_review"
    drs_max_practice_projects: int | None = None
    """``None`` means "until the curriculum agent stops asking".  The w/o-BRS
    ablation in Appendix C.5 sets this to 2."""

    drs_max_target_cycles: int = 8
    """Safety stop on target re-attempts within one DRS phase."""

    # --- Verification interface -----------------------------------------
    practice_verdicts: tuple[str, ...] = PRACTICE_VERDICT_STATUSES
    """Practice projects get one grounded PASS/FAIL and no same-project repair
    cycle.  The target interface additionally allows UNVERIFIED.

    "No repair cycle" means the *curriculum* does not re-run a project after a
    FAIL.  It does not forbid the actor from revising its own candidate after
    reading findings — that is the action–verification loop the harness runs,
    which the paper describes as core to the shared harness."""

    target_verdicts: tuple[str, ...] = VERDICT_STATUSES

    # --- Actor revision loops -------------------------------------------
    max_practice_revisions: int = 2
    """Actor revisions allowed per practice project after a FAIL."""

    max_target_revisions: int = 3
    """Actor revisions allowed per target attempt.  "The actor agent executes
    programs and revises its candidate using the verifier agent's findings"
    until "the verifier agent confirms that all task requirements have been
    satisfied" (Section 3.3)."""


@dataclass(frozen=True)
class ExecutionLimits:
    """Numerical guards from Appendix C.2, "Execution Safeguards".

    These bound *individual agent runs*; they are not a fixed number of RSI
    rounds.
    """

    target_iterations: int = 500
    practice_iterations: int = 2_000
    target_watchdog_s: float = 36_000.0
    practice_watchdog_s: float = 86_400.0
    program_timeout_s: float = 600.0
    allow_stall_role_switch: bool = True
    """The target harness permits one stall-triggered role swap of actor and
    verifier models when execution budget remains."""


@dataclass(frozen=True)
class RunConfig:
    """Everything one RSI lineage needs."""

    actor: RoleModelConfig
    verifier: RoleModelConfig
    curriculum: RoleModelConfig

    workspace: Path
    """Environment root.  Reset between independent attempts."""

    run_dir: Path
    """Artifacts: journals, memory checkpoints, evaluator output."""

    memory_root: Path | None = None
    """Canonical memory bank.  ``None`` derives ``run_dir/memory``."""

    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    limits: ExecutionLimits = field(default_factory=ExecutionLimits)

    stages: frozenset[str] = frozenset({"brs", "drs", "eval"})
    """Which lifecycle stages to run.

    This is the paper's stage-ablation switch (Figure 4, Appendix C.5):

    ==========================  ==========================================
    ``stages``                  Condition
    ==========================  ==========================================
    ``{"eval"}``                w/o RSI — empty memory, same harness
    ``{"brs", "eval"}``         w/o DRS — evaluate the BRS memory as-is
    ``{"drs", "eval"}``         w/o BRS — DRS from empty memory
    all three                   Full RSI
    ==========================  ==========================================

    ``eval`` is required: without it nothing is scored and the run has no
    result to report.
    """

    enable_memory: bool = True
    """``False`` is the paper's ``RSIAgent (w/o RSI)`` baseline: exploration and
    persistent memory disabled, same harness."""

    memory_writeback: bool = True
    """``False`` is frozen-memory evaluation (Phase 3): memory is read-only."""

    observer: RoleModelConfig | None = None
    """Model supplying visual observations.  The reference config uses the same
    model as the verifier."""

    user_channel: bool = False
    """Enables the actor's ``ask`` action against a simulated user."""

    seed: int | None = None

    def __post_init__(self) -> None:
        if self.exploration.brs_concurrency < 1:
            raise ConfigurationError("brs_concurrency must be >= 1")
        if self.exploration.brs_project_budget < 0:
            raise ConfigurationError("brs_project_budget must be >= 0")

        valid = {"brs", "drs", "eval"}
        unknown = set(self.stages) - valid
        if unknown:
            raise ConfigurationError(
                f"unknown stage(s) {sorted(unknown)}; expected a subset of {sorted(valid)}"
            )
        if "eval" not in self.stages:
            raise ConfigurationError(
                "stages must include 'eval': without it the lineage produces no score"
            )

        if not self.enable_memory and self.memory_writeback:
            raise ConfigurationError(
                "memory_writeback=True with enable_memory=False is contradictory"
            )
        if not self.enable_memory and self.stages & {"brs", "drs"}:
            raise ConfigurationError(
                "exploration stages require enable_memory=True; "
                "use stages=frozenset({'eval'}) for the w/o RSI baseline"
            )

    # -- derived paths ----------------------------------------------------
    @property
    def memory_path(self) -> Path:
        return self.memory_root if self.memory_root is not None else self.run_dir / "memory"

    @property
    def journal_path(self) -> Path:
        return self.run_dir / "journal.jsonl"

    @property
    def frozen_memory_path(self) -> Path:
        return self.run_dir / "frozen_memory"

    def replace(self, **changes: Any) -> "RunConfig":
        return replace(self, **changes)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": vars(self.actor),
            "verifier": vars(self.verifier),
            "curriculum": vars(self.curriculum),
            "observer": vars(self.observer) if self.observer else None,
            "workspace": str(self.workspace),
            "run_dir": str(self.run_dir),
            "memory_root": str(self.memory_root) if self.memory_root else None,
            "exploration": vars(self.exploration),
            "limits": vars(self.limits),
            "stages": sorted(self.stages),
            "enable_memory": self.enable_memory,
            "memory_writeback": self.memory_writeback,
            "user_channel": self.user_channel,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> "RunConfig":
        base = base_dir or Path.cwd()

        def _path(value: str | None, default: str) -> Path:
            raw = value or default
            p = Path(raw).expanduser()
            return p if p.is_absolute() else (base / p)

        def _role(key: str, default: str) -> RoleModelConfig:
            blob = data.get(key)
            if blob is None:
                return RoleModelConfig(model=default)
            if isinstance(blob, str):
                return RoleModelConfig(model=blob)
            return RoleModelConfig(**blob)

        observer_blob = data.get("observer")
        observer = None
        if observer_blob:
            observer = (
                RoleModelConfig(model=observer_blob)
                if isinstance(observer_blob, str)
                else RoleModelConfig(**observer_blob)
            )

        return cls(
            actor=_role("actor", "glm-5.3"),
            verifier=_role("verifier", "kimi-k3"),
            curriculum=_role("curriculum", "kimi-k3"),
            observer=observer,
            workspace=_path(data.get("workspace"), "workspace"),
            run_dir=_path(data.get("run_dir"), "runs/latest"),
            memory_root=(
                _path(data["memory_root"], "memory") if data.get("memory_root") else None
            ),
            exploration=ExplorationConfig(**data.get("exploration", {})),
            limits=ExecutionLimits(**data.get("limits", {})),
            stages=frozenset(data.get("stages", ("brs", "drs", "eval"))),
            enable_memory=data.get("enable_memory", True),
            memory_writeback=data.get("memory_writeback", True),
            user_channel=data.get("user_channel", False),
            seed=data.get("seed"),
        )


def load_config(path: str | Path) -> RunConfig:
    """Load a run configuration from JSON or YAML."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ConfigurationError(f"config not found: {p}")

    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise ConfigurationError(
                "PyYAML is required for YAML configs; use JSON or `pip install pyyaml`"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ConfigurationError(f"config root must be a mapping, got {type(data).__name__}")

    config = RunConfig.from_dict(data, base_dir=p.parent)
    _apply_env_overrides(config)
    return config


def _apply_env_overrides(config: RunConfig) -> None:
    """Allow ``RSIAGENT_WORKSPACE`` / ``RSIAGENT_RUN_DIR`` to relocate a run."""
    workspace = os.environ.get("RSIAGENT_WORKSPACE")
    run_dir = os.environ.get("RSIAGENT_RUN_DIR")
    if workspace:
        object.__setattr__(config, "workspace", Path(workspace).expanduser().resolve())
    if run_dir:
        object.__setattr__(config, "run_dir", Path(run_dir).expanduser().resolve())

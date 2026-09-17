"""RSIAgent — autonomous exploration for recursive self-improvement.

A training-free, multi-agent framework in which a Curriculum Agent, an Actor
Agent, and a Verifier Agent recursively explore an unfamiliar environment and
consolidate what they learn into reusable memory — without updating model
weights.

Reimplementation of:

    Zhu, Fan, Wang, Wu, Zhou, Huang.
    "RSIAgent: Autonomous Exploration for Recursive Self-improvement in New
    Environments." arXiv:2609.15364, 2026.

The entry point for the reference lifecycle (paper Algorithm A1) is
:func:`rsiagent.rsi.protocol.run_rsi`.
"""

from .config import (
    ExecutionLimits,
    ExplorationConfig,
    RoleModelConfig,
    RunConfig,
    load_config,
)
from .errors import (
    CheckpointError,
    ConfigurationError,
    InfrastructureFailure,
    MemoryIntegrityError,
    ModelResponseError,
    ProgramTimeout,
    RSIAgentError,
    UnverifiedOutcome,
)

__version__ = "0.1.0"

__all__ = [
    "CheckpointError",
    "ConfigurationError",
    "ExecutionLimits",
    "ExplorationConfig",
    "InfrastructureFailure",
    "MemoryIntegrityError",
    "ModelResponseError",
    "ProgramTimeout",
    "RSIAgentError",
    "RoleModelConfig",
    "RunConfig",
    "UnverifiedOutcome",
    "__version__",
    "load_config",
]

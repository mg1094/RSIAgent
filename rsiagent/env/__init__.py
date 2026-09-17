"""Environment abstraction and the local reference implementation."""

from .base import Environment, ExecResult, Observation, ProgramKind
from .local import LocalWorkspaceEnvironment

__all__ = [
    "Environment",
    "ExecResult",
    "LocalWorkspaceEnvironment",
    "Observation",
    "ProgramKind",
]

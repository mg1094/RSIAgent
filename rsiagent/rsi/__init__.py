"""The recursive self-improvement lifecycle (paper Algorithm A1)."""

from .phase1_brs import Branch, BroadExplorer, Phase1Result, WaveResult
from .phase2_drs import DeepRefiner, Phase2Result, TargetCycle
from .phase3_eval import FrozenMemoryEvaluator, Phase3Result
from .protocol import BaselineResult, RSIResult, RSIRunner, run_baseline

__all__ = [
    "BaselineResult",
    "Branch",
    "BroadExplorer",
    "DeepRefiner",
    "FrozenMemoryEvaluator",
    "Phase1Result",
    "Phase2Result",
    "Phase3Result",
    "RSIRunner",
    "RSIResult",
    "TargetCycle",
    "WaveResult",
    "run_baseline",
]

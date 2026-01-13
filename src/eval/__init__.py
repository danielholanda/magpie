"""
Evaluation modules for AIG-Kernel-Eval.

This module contains the core evaluation components:
- Evaluator: Main evaluation pipeline orchestrator
- Compiling: Kernel compilation
- Correctness: Correctness verification
- Performance: Performance measurement
"""

from .evaluator import (
    Evaluator,
    EvaluationState,
    BaseKind,
)
from .compiling import Compiling, CompilingResult
from .correctness import (
    Correctness,
    CorrectnessResult,
    MetricResult,
)
from .performance import Performance, PerformanceResult

__all__ = [
    # Evaluator
    "Evaluator",
    "EvaluationState",
    "BaseKind",
    # Compiling
    "Compiling",
    "CompilingResult",
    # Correctness
    "Correctness",
    "CorrectnessResult",
    "MetricResult",
    # Performance
    "Performance",
    "PerformanceResult",
]

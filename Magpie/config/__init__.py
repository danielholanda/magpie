"""
Configuration module for Magpie.

This module defines all configuration classes and enums used throughout
the evaluation framework.
"""

from .pipeline import (
    KernelType,
    EvalMode,
    PipelineConfig,
)
from .kernel import (
    KernelEvalConfig,
)
from .correctness import (
    CorrectnessMode,
    CorrectnessConfig,
    AlgorithmThresholds,
)
from .performance import (
    PerfBackend,
    PerformanceConfig,
)

__all__ = [
    # Pipeline configuration
    "KernelType",
    "EvalMode",
    "PipelineConfig",
    # Kernel evaluation configuration
    "KernelEvalConfig",
    # Correctness configuration
    "CorrectnessMode",
    "CorrectnessConfig",
    "AlgorithmThresholds",
    # Performance configuration
    "PerfBackend",
    "PerformanceConfig",
]


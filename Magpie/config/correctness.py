"""
Correctness evaluation configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class CorrectnessMode(Enum):
    """Correctness evaluation mode."""
    TESTCASE = auto()          # Run provided testcase command
    RESULT_COMPARISON = auto()  # Compare outputs between kernels (for compare mode)


@dataclass
class AlgorithmThresholds:
    """
    Thresholds for correctness comparison algorithms.
    """
    atol: float = 1e-5         # Absolute tolerance
    rtol: float = 1e-4         # Relative tolerance


@dataclass
class CorrectnessConfig:
    """
    Configuration for correctness evaluation.
    
    Attributes:
        mode: Correctness evaluation mode
        testcase_command: Command to run for testcase verification (for analyze mode)
        iteration_count: Number of iterations for correctness testing
        thresholds: Tolerance thresholds for result comparison
        check_nan: Whether to check for NaN values
        check_inf: Whether to check for Inf values
    """
    mode: CorrectnessMode = CorrectnessMode.TESTCASE
    testcase_command: Optional[List[str]] = None
    iteration_count: int = 1
    thresholds: AlgorithmThresholds = field(default_factory=AlgorithmThresholds)
    check_nan: bool = True
    check_inf: bool = True

    def has_testcase(self) -> bool:
        """Check if testcase command is provided."""
        return self.testcase_command is not None and len(self.testcase_command) > 0


"""
Evaluation modes for Magpie.

This module contains different evaluation modes:
- analyze_eval: Analyze individual kernels for correctness, performance, etc.
- compare_eval: Compare two or more kernel implementations
"""

from .analyze_eval import AnalyzeMode
from .compare_eval import CompareMode

__all__ = [
    "AnalyzeMode",
    "CompareMode",
]


"""
Compare mode for kernel comparison.

In compare mode:
- If testcase_command is provided, use it
- Otherwise, for PyTorch kernels, generate inputs and compare outputs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...config import (
    KernelType,
    EvalMode,
    PipelineConfig,
    KernelEvalConfig,
    CorrectnessConfig,
    CorrectnessMode,
    PerformanceConfig,
)
from ...eval import Evaluator, EvaluationState, BaseKind

logger = logging.getLogger(__name__)


@dataclass
class CompareConfig:
    """
    Configuration for compare mode.
    
    Attributes:
        baseline_index: Index of baseline kernel for comparison
        gpu_arch: GPU architecture
        check_performance: Whether to run performance profiling
        timeout_seconds: Timeout for profiling operations
        profiler_args: Additional arguments for the profiler
    """
    baseline_index: int = 0
    gpu_arch: str = "gfx90a"
    check_performance: bool = True
    timeout_seconds: float = 60.0
    profiler_args: List[str] = None
    
    def __post_init__(self):
        if self.profiler_args is None:
            self.profiler_args = []


@dataclass
class ComparisonResult:
    """
    Result of comparing kernel implementations.
    """
    kernel_results: List[EvaluationState] = field(default_factory=list)
    comparison_metrics: Dict[str, Any] = field(default_factory=dict)
    winner: Optional[int] = None
    rankings: List[Tuple[int, float]] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_results": [r.to_dict() for r in self.kernel_results],
            "comparison_metrics": self.comparison_metrics,
            "winner": self.winner,
            "rankings": self.rankings,
            "summary": self.summary
        }


class CompareMode:
    """
    Comparator for multiple kernel implementations.
    """
    
    def __init__(self, config: Optional[CompareConfig] = None):
        self.config = config or CompareConfig()
        
    def compare(
        self, 
        kernel_configs: List[KernelEvalConfig]
    ) -> ComparisonResult:
        """
        Compare multiple kernel implementations.
        
        Args:
            kernel_configs: List of kernel configurations to compare
            
        Returns:
            ComparisonResult with comparison results
        """
        if len(kernel_configs) < 2:
            raise ValueError("Need at least 2 kernels for comparison")
        
        logger.info(f"Comparing {len(kernel_configs)} kernels")
        
        # Evaluate all kernels
        results = []
        for i, cfg in enumerate(kernel_configs):
            logger.info(f"Evaluating kernel {i+1}/{len(kernel_configs)}: {cfg.kernel_id}")
            
            # Determine correctness mode
            if cfg.has_testcase():
                corr_mode = CorrectnessMode.TESTCASE
            else:
                corr_mode = CorrectnessMode.RESULT_COMPARISON
            
            # Build pipeline config
            pipeline_cfg = PipelineConfig(
                mode=EvalMode.COMPARE,
                kernel_type=cfg.kernel_type,
                gpu_arch=self.config.gpu_arch,
                correctness_config=CorrectnessConfig(mode=corr_mode),
                performance_config=PerformanceConfig(
                    enabled=self.config.check_performance,
                    kernel_type=cfg.kernel_type,
                    timeout_seconds=self.config.timeout_seconds,
                    profiler_args=self.config.profiler_args,
                ),
            )
            
            evaluator = Evaluator(pipeline_cfg)
            state = evaluator.evaluate(cfg)
            results.append(state)
        
        # Build comparison
        comparison = self._build_comparison(results, kernel_configs)
        self._log_summary(comparison)
        
        return comparison
    
    def _build_comparison(
        self, 
        results: List[EvaluationState],
        kernel_configs: List[KernelEvalConfig]
    ) -> ComparisonResult:
        """Build comparison from individual results."""
        comparison = ComparisonResult(kernel_results=results)
        
        # Check correctness
        correctness_results = []
        for r in results:
            is_correct = (
                r.correctness_state == BaseKind.SUCCESS and
                r.correctness_result is not None and
                r.correctness_result.success
            )
            correctness_results.append(is_correct)
        
        comparison.comparison_metrics["correctness"] = correctness_results
        comparison.comparison_metrics["all_correct"] = all(correctness_results)
        
        # For now, winner is the first kernel that passes correctness
        for i, correct in enumerate(correctness_results):
            if correct:
                comparison.winner = i
                break
        
        # Generate summary
        comparison.summary = self._generate_summary(comparison, kernel_configs)
        
        return comparison
    
    def _generate_summary(
        self, 
        comparison: ComparisonResult,
        kernel_configs: List[KernelEvalConfig]
    ) -> str:
        """Generate human-readable comparison summary."""
        lines = [
            f"Compared {len(comparison.kernel_results)} kernel implementations",
            ""
        ]
        
        # Correctness summary
        correct_count = sum(comparison.comparison_metrics.get("correctness", []))
        total_count = len(comparison.kernel_results)
        lines.append(f"Correctness: {correct_count}/{total_count} passed")
        
        # Individual results
        lines.append("")
        lines.append("Results:")
        for i, (cfg, result) in enumerate(zip(kernel_configs, comparison.kernel_results)):
            status = "✓" if comparison.comparison_metrics["correctness"][i] else "✗"
            lines.append(f"  {status} Kernel {i}: {cfg.kernel_id}")
        
        # Winner
        if comparison.winner is not None:
            lines.append("")
            lines.append(f"Winner: Kernel {comparison.winner} ({kernel_configs[comparison.winner].kernel_id})")
        
        return "\n".join(lines)
    
    def _log_summary(self, comparison: ComparisonResult) -> None:
        """Log comparison summary."""
        logger.info("Comparison complete")
        for line in comparison.summary.split("\n"):
            if line.strip():
                logger.info(f"  {line}")

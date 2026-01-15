"""
Analyze mode for single kernel analysis.

In analyze mode:
- A testcase command is required
- The kernel is compiled, testcase is run, and performance is measured
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ...config import (
    KernelType,
    EvalMode,
    PipelineConfig,
    KernelEvalConfig,
    CorrectnessConfig,
    CorrectnessMode,
    PerformanceConfig,
)
from ...eval import Evaluator, EvaluationState

logger = logging.getLogger(__name__)


@dataclass
class AnalyzeConfig:
    """
    Configuration for analyze mode.
    
    Attributes:
        kernel_type: Default kernel type
        gpu_arch: GPU architecture
        check_performance: Whether to run performance profiling
        timeout_seconds: Timeout for profiling operations
        profiler_args: Additional arguments for the profiler
    """
    kernel_type: KernelType = KernelType.HIP
    gpu_arch: str = "gfx942"
    check_performance: bool = True
    timeout_seconds: float = 60.0
    profiler_args: List[str] = None
    
    def __post_init__(self):
        if self.profiler_args is None:
            self.profiler_args = []


class AnalyzeMode:
    """
    Analyzer for individual kernel evaluation.
    
    Requires testcase_command to be provided in KernelEvalConfig.
    """
    
    def __init__(self, config: Optional[AnalyzeConfig] = None):
        self.config = config or AnalyzeConfig()
        
    def analyze(self, kernel_cfg: KernelEvalConfig) -> EvaluationState:
        """
        Analyze a single kernel.
        
        Args:
            kernel_cfg: Kernel configuration (must include testcase_command)
            
        Returns:
            EvaluationState with analysis results
        """
        # Validate that testcase is provided
        if not kernel_cfg.has_testcase():
            logger.error("Analyze mode requires testcase_command")
            state = EvaluationState()
            state.errors.append("Analyze mode requires testcase_command")
            return state
        
        # Build pipeline config for analyze mode
        pipeline_cfg = PipelineConfig(
            mode=EvalMode.ANALYZE,
            kernel_type=kernel_cfg.kernel_type,
            gpu_arch=self.config.gpu_arch,
            correctness_config=CorrectnessConfig(
                mode=CorrectnessMode.TESTCASE,
            ),
            performance_config=PerformanceConfig(
                enabled=self.config.check_performance,
                kernel_type=kernel_cfg.kernel_type,
                timeout_seconds=self.config.timeout_seconds,
                profiler_args=self.config.profiler_args,
            ),
        )
        
        evaluator = Evaluator(pipeline_cfg)
        state = evaluator.evaluate(kernel_cfg)
        
        self._log_summary(kernel_cfg, state)
        return state
    
    def analyze_batch(
        self, 
        kernel_configs: List[KernelEvalConfig]
    ) -> List[EvaluationState]:
        """Analyze multiple kernels."""
        results = []
        for i, cfg in enumerate(kernel_configs):
            logger.info(f"Analyzing kernel {i+1}/{len(kernel_configs)}: {cfg.kernel_id}")
            result = self.analyze(cfg)
            results.append(result)
        return results
    
    def _log_summary(self, kernel_cfg: KernelEvalConfig, state: EvaluationState) -> None:
        """Log analysis summary."""
        from ...eval import BaseKind
        
        logger.info(f"Analysis complete: {kernel_cfg.kernel_id}")
        logger.info(f"  Compiling: {state.compiling_state.name}")
        logger.info(f"  Correctness: {state.correctness_state.name}")
        logger.info(f"  Performance: {state.performance_state.name}")
        logger.info(f"  Score: {state.score:.2f}")


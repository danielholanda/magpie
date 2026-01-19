"""
Analyze mode for single kernel analysis.

In analyze mode:
- A testcase command is required
- The kernel is compiled, testcase is run, and performance is measured
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...config import (
    KernelType,
    EvalMode,
    PipelineConfig,
    KernelEvalConfig,
    CompilingConfig,
    CorrectnessConfig,
    CorrectnessMode,
    PerformanceConfig,
)
from ...config.performance import RocprofComputeConfig, NcuConfig
from ...eval import Evaluator, EvaluationState

logger = logging.getLogger(__name__)


@dataclass
class AnalyzeConfig:
    """
    Configuration for analyze mode.
    
    Attributes:
        kernel_type: Default kernel type
        gpu_arch: GPU architecture
        enable_default_compile: Enable default compilation when no compile_command
        check_performance: Whether to run performance profiling
        timeout_seconds: Timeout for profiling operations
        profiler_args: Additional arguments for the profiler (legacy)
        rocprof_config: rocprof-compute configuration dict
        ncu_config: ncu configuration dict
    """
    kernel_type: KernelType = KernelType.HIP
    gpu_arch: str = "gfx942"
    enable_default_compile: bool = False
    check_performance: bool = True
    timeout_seconds: float = 300.0
    profiler_args: List[str] = None
    rocprof_config: Dict[str, Any] = None
    ncu_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.profiler_args is None:
            self.profiler_args = []
        if self.rocprof_config is None:
            self.rocprof_config = {}
        if self.ncu_config is None:
            self.ncu_config = {}


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
        
        # Build rocprof-compute config if provided
        rocprof_cfg = None
        if self.config.rocprof_config:
            rocprof_cfg = RocprofComputeConfig(
                workload_dir=self.config.rocprof_config.get("workload_dir", "./workloads"),
                metric_blocks=self.config.rocprof_config.get("metric_blocks", ["1", "2", "5", "10", "11", "12", "14", "16", "17"]),
                no_roof=self.config.rocprof_config.get("no_roof", True),
                output_format=self.config.rocprof_config.get("output_format", "csv"),
                profile_args=self.config.rocprof_config.get("profile_args", []),
                analyze_args=self.config.rocprof_config.get("analyze_args", []),
            )
        
        # Build ncu config if provided
        ncu_cfg = None
        if self.config.ncu_config:
            ncu_cfg = NcuConfig(
                args=self.config.ncu_config.get("args", []),
                metrics=self.config.ncu_config.get("metrics", []),
            )
        
        # Build pipeline config for analyze mode
        pipeline_cfg = PipelineConfig(
            mode=EvalMode.ANALYZE,
            kernel_type=kernel_cfg.kernel_type,
            gpu_arch=self.config.gpu_arch,
            compiling_config=CompilingConfig(
                enable_default_compile=self.config.enable_default_compile,
            ),
            correctness_config=CorrectnessConfig(
                mode=CorrectnessMode.TESTCASE,
            ),
            performance_config=PerformanceConfig(
                enabled=self.config.check_performance,
                kernel_type=kernel_cfg.kernel_type,
                timeout_seconds=self.config.timeout_seconds,
                profiler_args=self.config.profiler_args,
                rocprof_config=rocprof_cfg,
                ncu_config=ncu_cfg,
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
        """
        Analyze multiple kernels sequentially.
        
        Args:
            kernel_configs: List of kernel configurations to analyze
            
        Returns:
            List of EvaluationState results
        """
        if not kernel_configs:
            return []
        
        logger.info(f"Analyzing {len(kernel_configs)} kernels sequentially")
        
        results = []
        for i, kernel_cfg in enumerate(kernel_configs):
            logger.info(f"Analyzing kernel {i+1}/{len(kernel_configs)}: {kernel_cfg.kernel_id}")
            state = self.analyze(kernel_cfg)
            results.append(state)
        
        return results
    
    def _log_summary(self, kernel_cfg: KernelEvalConfig, state: EvaluationState) -> None:
        """Log analysis summary."""
        logger.info(f"Analysis complete: {kernel_cfg.kernel_id}")
        logger.info(f"  Compiling: {state.compiling_state.name}")
        logger.info(f"  Correctness: {state.correctness_state.name}")
        logger.info(f"  Performance: {state.performance_state.name}")
        logger.info(f"  Score: {state.score:.2f}")

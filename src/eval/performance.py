"""
Performance evaluation module.

This module handles performance measurement of GPU kernels using
different backends based on kernel type:
- HIP kernels: rocprof-compute
- CUDA kernels: ncu (NVIDIA Nsight Compute)
"""

from __future__ import annotations

import logging
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..config import (
    KernelType,
    PipelineConfig,
    KernelEvalConfig,
    PerformanceConfig,
    PerfBackend,
)
from ..utils import get_updated_env

if TYPE_CHECKING:
    from .evaluator import EvaluationState

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """A single performance metric result."""
    name: str
    value: Optional[float]
    unit: str = ""


@dataclass
class PerformanceResult:
    """
    Result of performance evaluation.
    """
    success: bool
    metrics: List[MetricResult] = field(default_factory=list)
    raw_output: Optional[str] = None
    errors: Optional[str] = None


class Performance:
    """
    Performance evaluation handler.
    
    Uses different profiling backends based on kernel type:
    - HIP: rocprof-compute
    - CUDA: ncu (NVIDIA Nsight Compute)
    """
    
    def __init__(self, pipeline_cfg: PipelineConfig) -> None:
        """
        Initialize performance evaluator.
        """
        self.pipeline_cfg = pipeline_cfg
        self.perf_cfg = pipeline_cfg.performance_config
        
    def run(
        self, 
        eval_state: Any, 
        kernel_cfg: KernelEvalConfig
    ) -> PerformanceResult:
        """
        Run performance evaluation.
        
        Args:
            eval_state: Current evaluation state (contains compiled kernel info)
            kernel_cfg: Kernel configuration
            
        Returns:
            PerformanceResult with profiling metrics
        """
        if not self.perf_cfg.enabled:
            return PerformanceResult(success=True)
        
        # If custom prof_command is provided, use it instead of built-in backend
        if kernel_cfg.has_prof_command():
            return self._run_custom_profiler(kernel_cfg)
        
        backend = self.perf_cfg.get_backend()
        
        try:
            if backend == PerfBackend.ROCPROF_COMPUTE:
                return self._run_rocprof_compute(eval_state, kernel_cfg)
            elif backend == PerfBackend.NCU:
                return self._run_ncu(eval_state, kernel_cfg)
            else:
                return PerformanceResult(
                    success=True,
                    errors="No profiling backend configured"
                )
                
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return PerformanceResult(success=False, errors=str(e))

    def _run_custom_profiler(
        self,
        kernel_cfg: KernelEvalConfig
    ) -> PerformanceResult:
        """
        Run profiling using custom prof_command(s).
        
        This replaces the built-in profiler backend when prof_command is specified.
        Supports multiple commands executed in order.
        """
        commands = kernel_cfg.get_prof_commands()
        working_dir = kernel_cfg.working_dir
        env = get_updated_env(kernel_cfg.env)
        
        all_outputs = []
        
        try:
            for cmd_idx, cmd in enumerate(commands):
                logger.info(f"Running profiler command {cmd_idx+1}/{len(commands)}: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=working_dir,
                    timeout=self.perf_cfg.timeout_seconds
                )
                
                all_outputs.append(result.stdout)
                
                if result.returncode != 0:
                    cmd_info = f"command {cmd_idx+1}/{len(commands)}" if len(commands) > 1 else "command"
                    return PerformanceResult(
                        success=False,
                        raw_output="\n".join(all_outputs),
                        errors=f"Profiler {cmd_info} failed: {result.stderr or result.stdout}"
                    )
            
            # All commands succeeded
            return PerformanceResult(
                success=True,
                metrics=[MetricResult(
                    name="custom_profiling_complete",
                    value=1.0,
                    unit="bool"
                )],
                raw_output="\n".join(all_outputs)
            )
            
        except subprocess.TimeoutExpired:
            return PerformanceResult(
                success=False,
                errors=f"Custom profiler timed out after {self.perf_cfg.timeout_seconds}s"
            )
        except Exception as e:
            logger.error(f"Custom profiler failed: {e}")
            return PerformanceResult(success=False, errors=str(e))

    def _run_rocprof_compute(
        self, 
        eval_state: Any, 
        kernel_cfg: KernelEvalConfig
    ) -> PerformanceResult:
        """
        Run profiling using rocprof-compute (AMD GPUs).
        
        rocprof-compute is a performance analysis tool for AMD GPUs.
        """
        # Check if rocprof-compute is available
        if shutil.which("rocprof-compute") is None:
            return PerformanceResult(
                success=False,
                errors="rocprof-compute not found. Please install ROCm tools."
            )
        
        # Get the compiled executable
        if eval_state.compiling_result is None:
            return PerformanceResult(
                success=False,
                errors="No compiled kernel available for profiling"
            )
        
        executable = eval_state.compiling_result.output_file_path
        if executable is None:
            return PerformanceResult(
                success=False,
                errors="No executable path in compiling result"
            )
        
        working_dir = kernel_cfg.working_dir
        env = get_updated_env(kernel_cfg.env)
        
        # Build rocprof-compute command
        cmd = [
            "rocprof-compute",
            "profile",
            "--",
            executable,
            *self.perf_cfg.profiler_args
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=working_dir,
                timeout=self.perf_cfg.timeout_seconds
            )
            
            if result.returncode != 0:
                return PerformanceResult(
                    success=False,
                    raw_output=result.stdout,
                    errors=result.stderr or "rocprof-compute failed"
                )
            
            # Parse output and extract metrics
            metrics = self._parse_rocprof_output(result.stdout)
            
            return PerformanceResult(
                success=True,
                metrics=metrics,
                raw_output=result.stdout
            )
            
        except subprocess.TimeoutExpired:
            return PerformanceResult(
                success=False,
                errors=f"Profiling timed out after {self.perf_cfg.timeout_seconds}s"
            )
        except Exception as e:
            return PerformanceResult(success=False, errors=str(e))

    def _run_ncu(
        self, 
        eval_state: Any, 
        kernel_cfg: KernelEvalConfig
    ) -> PerformanceResult:
        """
        Run profiling using ncu (NVIDIA Nsight Compute).
        """
        # Check if ncu is available
        if shutil.which("ncu") is None:
            return PerformanceResult(
                success=False,
                errors="ncu not found. Please install NVIDIA Nsight Compute."
            )
        
        # Get the compiled executable
        if eval_state.compiling_result is None:
            return PerformanceResult(
                success=False,
                errors="No compiled kernel available for profiling"
            )
        
        executable = eval_state.compiling_result.output_file_path
        if executable is None:
            return PerformanceResult(
                success=False,
                errors="No executable path in compiling result"
            )
        
        working_dir = kernel_cfg.working_dir
        env = get_updated_env(kernel_cfg.env)
        
        # Build ncu command
        cmd = [
            "ncu",
            "--target-processes", "all",
            executable,
            *self.perf_cfg.profiler_args
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=working_dir,
                timeout=self.perf_cfg.timeout_seconds
            )
            
            if result.returncode != 0:
                return PerformanceResult(
                    success=False,
                    raw_output=result.stdout,
                    errors=result.stderr or "ncu failed"
                )
            
            # Parse output and extract metrics
            metrics = self._parse_ncu_output(result.stdout)
            
            return PerformanceResult(
                success=True,
                metrics=metrics,
                raw_output=result.stdout
            )
            
        except subprocess.TimeoutExpired:
            return PerformanceResult(
                success=False,
                errors=f"Profiling timed out after {self.perf_cfg.timeout_seconds}s"
            )
        except Exception as e:
            return PerformanceResult(success=False, errors=str(e))

    def _parse_rocprof_output(self, output: str) -> List[MetricResult]:
        """Parse rocprof-compute output and extract metrics."""
        metrics = []
        # TODO: Implement actual parsing based on rocprof-compute output format
        # For now, return raw output indicator
        metrics.append(MetricResult(
            name="profiling_complete",
            value=1.0,
            unit="bool"
        ))
        return metrics

    def _parse_ncu_output(self, output: str) -> List[MetricResult]:
        """Parse ncu output and extract metrics."""
        metrics = []
        # TODO: Implement actual parsing based on ncu output format
        # For now, return raw output indicator
        metrics.append(MetricResult(
            name="profiling_complete",
            value=1.0,
            unit="bool"
        ))
        return metrics

###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Correctness evaluation module.

This module handles correctness verification of GPU kernels.

- Analyze mode: Run testcase command and check pass/fail
- Compare mode: Compare outputs between multiple kernels
  - For PyTorch: Generate test inputs and compare outputs
  - For HIP/CUDA: Run kernels and compare stdout/results
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import random
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

from ..utils import get_updated_env
from ..config import (
    EvalMode,
    KernelType,
    PipelineConfig,
    KernelEvalConfig,
    CorrectnessConfig,
    CorrectnessMode,
)

# Optional torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

if TYPE_CHECKING:
    from .evaluator import EvaluationState

logger = logging.getLogger(__name__)


# =========================
# ------- Data Models -----
# =========================

@dataclass
class MetricResult:
    """A single metric outcome for one comparison."""
    name: str
    success: bool
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class CorrectnessResult:
    """Result of correctness evaluation."""
    success: bool
    metrics: List[MetricResult] = field(default_factory=list)
    errors: Optional[str] = None


# =========================
# ------- Correctness -----
# =========================

class Correctness:
    """
    Correctness evaluation orchestrator.
    
    - Analyze mode: Runs testcase command
    - Compare mode: Compares outputs between kernels
    """

    def __init__(self, pipeline_cfg: PipelineConfig) -> None:
        """
        Initialize correctness evaluator.
        
        Args:
            pipeline_cfg: Pipeline configuration
        """
        self.pipeline_cfg = pipeline_cfg
        self.corr_cfg = pipeline_cfg.correctness_config

    def run(
        self, 
        eval_state: Any, 
        kernel_cfg: KernelEvalConfig
    ) -> CorrectnessResult:
        """
        Run correctness evaluation.
        
        Args:
            eval_state: Current evaluation state
            kernel_cfg: Kernel configuration
            
        Returns:
            CorrectnessResult with pass/fail status
        """
        try:
            mode = self.pipeline_cfg.mode
            
            if mode == EvalMode.ANALYZE:
                return self._run_analyze_mode(eval_state, kernel_cfg)
            elif mode == EvalMode.COMPARE:
                return self._run_compare_mode(eval_state, kernel_cfg)
            else:
                return CorrectnessResult(
                    success=False,
                    errors=f"Unknown evaluation mode: {mode}"
                )
                
        except Exception as e:
            logger.error(f"Correctness evaluation failed: {e}")
            return CorrectnessResult(success=False, errors=str(e))

    def _run_analyze_mode(
        self, 
        eval_state: Any, 
        kernel_cfg: KernelEvalConfig
    ) -> CorrectnessResult:
        """
        Run analyze mode - execute testcase command.
        
        Requires testcase_command to be provided.
        """
        if not kernel_cfg.has_testcase():
            return CorrectnessResult(
                success=False,
                errors="Analyze mode requires testcase_command"
            )
        
        return self._run_testcase(kernel_cfg)

    def _run_compare_mode(
        self, 
        eval_state: Any, 
        kernel_cfg: KernelEvalConfig
    ) -> CorrectnessResult:
        """
        Run compare mode - compare outputs.
        
        If testcase command is provided, use it.
        Otherwise, for PyTorch kernels, generate inputs and compare outputs.
        """
        # If testcase is provided, use it
        if kernel_cfg.has_testcase():
            return self._run_testcase(kernel_cfg)
        
        # Otherwise, for PyTorch, generate inputs and compare
        if kernel_cfg.kernel_type == KernelType.PYTORCH:
            return self._run_pytorch_comparison(eval_state, kernel_cfg)
        
        # For HIP/CUDA without testcase, we need exec to compare outputs
        return CorrectnessResult(
            success=False,
            errors="Compare mode for HIP/CUDA requires testcase_command or executable comparison"
        )

    def _run_testcase(self, kernel_cfg: KernelEvalConfig) -> CorrectnessResult:
        """
        Run testcase command(s) and check result.
        
        Supports both single command and multiple commands executed in order.
        """
        commands = kernel_cfg.get_testcase_commands()
        working_dir = kernel_cfg.working_dir
        env = get_updated_env(kernel_cfg.env)
        
        n_iter = self.corr_cfg.iteration_count or 1
        all_metrics = []
        
        try:
            for i in range(n_iter):
                for cmd_idx, cmd in enumerate(commands):
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        env=env,
                        cwd=working_dir
                    )
                    
                    if result.returncode != 0:
                        errors = result.stderr if result.stderr else result.stdout
                        all_metrics.append(MetricResult(
                            name=f"testcase_cmd{cmd_idx+1}",
                            success=False,
                            value=float(result.returncode)
                        ))
                        cmd_info = f"command {cmd_idx+1}/{len(commands)}" if len(commands) > 1 else "command"
                        return CorrectnessResult(
                            success=False,
                            metrics=all_metrics,
                            errors=f"Testcase {cmd_info} failed (iteration {i+1}): {errors}"
                        )
                    
                    all_metrics.append(MetricResult(
                        name=f"testcase_cmd{cmd_idx+1}_iter{i+1}",
                        success=True
                    ))
            
            return CorrectnessResult(
                success=True,
                metrics=[MetricResult(name="testcase", success=True)]
            )
            
        except Exception as e:
            return CorrectnessResult(success=False, errors=str(e))

    def _run_pytorch_comparison(
        self, 
        eval_state: Any, 
        kernel_cfg: KernelEvalConfig
    ) -> CorrectnessResult:
        """
        Run PyTorch kernel comparison.
        
        Generate test inputs using get_inputs function and compare outputs.
        """
        if not HAS_TORCH:
            return CorrectnessResult(
                success=False,
                errors="PyTorch not available for comparison"
            )
        
        source_files = kernel_cfg.get_source_file_paths()
        if not source_files:
            return CorrectnessResult(
                success=False,
                errors="No source files provided"
            )
        
        try:
            # Load the module
            module = self._load_module(source_files[0])
            
            # Get input generation function
            get_inputs_fn = getattr(module, kernel_cfg.get_inputs_func, None)
            if get_inputs_fn is None:
                return CorrectnessResult(
                    success=False,
                    errors=f"Function '{kernel_cfg.get_inputs_func}' not found in module"
                )
            
            # Get model classes (assuming Model and NewModel pattern)
            model_class = getattr(module, "Model", None)
            
            if model_class is None:
                return CorrectnessResult(
                    success=False,
                    errors="Model class not found in module"
                )
            
            # Generate inputs and run comparison
            n_iter = self.corr_cfg.iteration_count or 1
            all_metrics = []
            
            for i in range(n_iter):
                seed = random.randint(1, 10000)
                torch.manual_seed(seed)
                
                # Generate inputs
                inputs = get_inputs_fn()
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    inputs = [
                        inp.cuda() if isinstance(inp, torch.Tensor) else inp 
                        for inp in inputs
                    ]
                
                # Get init inputs if available
                get_init_inputs_fn = getattr(module, kernel_cfg.get_init_inputs_func, None)
                init_inputs = get_init_inputs_fn() if get_init_inputs_fn else []
                
                # Create and run model
                if init_inputs:
                    model = model_class(*init_inputs)
                else:
                    model = model_class()
                
                if torch.cuda.is_available():
                    model = model.cuda()
                
                model.eval()
                with torch.no_grad():
                    output = model(*inputs)
                
                # Check for NaN/Inf
                if self.corr_cfg.check_nan and torch.isnan(output).any():
                    all_metrics.append(MetricResult(
                        name="nan_check", success=False
                    ))
                    return CorrectnessResult(
                        success=False,
                        metrics=all_metrics,
                        errors=f"NaN detected in output (iteration {i+1})"
                    )
                
                if self.corr_cfg.check_inf and torch.isinf(output).any():
                    all_metrics.append(MetricResult(
                        name="inf_check", success=False
                    ))
                    return CorrectnessResult(
                        success=False,
                        metrics=all_metrics,
                        errors=f"Inf detected in output (iteration {i+1})"
                    )
                
                all_metrics.append(MetricResult(
                    name=f"run_{i+1}", success=True
                ))
            
            return CorrectnessResult(
                success=True,
                metrics=all_metrics
            )
            
        except Exception as e:
            return CorrectnessResult(success=False, errors=str(e))

    def _load_module(self, file_path: str) -> ModuleType:
        """Load a Python module from file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        salt = hashlib.md5(str(p.resolve()).encode()).hexdigest()[:8]
        mod_name = f"_dyn_module_{p.stem}_{salt}"
        
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


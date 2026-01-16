"""
Main evaluator module.

This module contains the core Evaluator class that orchestrates the
compiling → correctness → performance evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from ..config import EvalMode, PipelineConfig, KernelEvalConfig
from .correctness import Correctness, CorrectnessResult
from .compiling import Compiling, CompilingResult
from .performance import Performance, PerformanceResult


class BaseKind(Enum):
    """Base status for evaluation stages."""
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class EvaluationState:
    """
    Evaluation state that holds the results of all evaluation stages.
    """
    # State of each evaluation step
    compiling_state: BaseKind = BaseKind.SUCCESS
    correctness_state: BaseKind = BaseKind.SUCCESS
    performance_state: BaseKind = BaseKind.SUCCESS
    errors: List[str] = field(default_factory=list)

    # Results of each evaluation step
    compiling_result: Optional[CompilingResult] = None
    correctness_result: Optional[CorrectnessResult] = None
    performance_result: Optional[PerformanceResult] = None

    # Overall score (0.0 to 1.0)
    score: float = 0.0

    # Additional metadata
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format."""
        return {
            "compiling_state": self.compiling_state.name,
            "correctness_state": self.correctness_state.name,
            "performance_state": self.performance_state.name,
            "errors": self.errors,
            "score": self.score,
            "compiling_result": {
                "success": self.compiling_result.success if self.compiling_result else False,
                "errors": self.compiling_result.errors if self.compiling_result else None,
            } if self.compiling_result else None,
            "correctness_result": {
                "success": self.correctness_result.success if self.correctness_result else False,
                "errors": self.correctness_result.errors if self.correctness_result else None,
            } if self.correctness_result else None,
            "performance_result": self.performance_result.to_dict() if self.performance_result else None,
            "extra": self.extra,
        }


class Evaluator:
    """
    Main evaluator implementing the evaluation pipeline.
    
    Pipeline: Compiling → Correctness → Performance
    """
    
    def __init__(self, pipeline_cfg: PipelineConfig) -> None:
        """
        Initialize the evaluator.
        
        Args:
            pipeline_cfg: Pipeline configuration
        """
        self.pipeline_cfg = pipeline_cfg
        self.compiling = Compiling(pipeline_cfg)
        self.correctness = Correctness(pipeline_cfg)
        self.performance = Performance(pipeline_cfg)

    def evaluate(self, kernel_cfg: KernelEvalConfig) -> EvaluationState:
        """
        Run the complete evaluation pipeline.
        
        Args:
            kernel_cfg: Kernel configuration
            
        Returns:
            EvaluationState with results from all stages
        """
        state = EvaluationState()
        state.extra["kernel_id"] = kernel_cfg.kernel_id
        state.extra["kernel_type"] = kernel_cfg.kernel_type.name

        # 1) Compiling (skip if no compile_command)
        state = self._compile(state, kernel_cfg)
        if state.compiling_state == BaseKind.FAILED:
            return state

        # 2) Correctness
        state = self._check_correctness(state, kernel_cfg)
        if state.correctness_state == BaseKind.FAILED:
            return state

        # 3) Performance (skip if no prof_command and profiling disabled)
        state = self._check_performance(state, kernel_cfg)

        # 4) Calculate score
        state = self._calculate_score(state)

        return state

    def _compile(
        self, 
        state: EvaluationState, 
        kernel_cfg: KernelEvalConfig
    ) -> EvaluationState:
        """Compile the kernel."""
        try:
            result = self.compiling.run(kernel_cfg)
            state.compiling_result = result
            
            if result is None:
                # No compilation needed (skipped)
                state.compiling_state = BaseKind.SKIPPED
            elif result.success:
                state.compiling_state = BaseKind.SUCCESS
            else:
                state.compiling_state = BaseKind.FAILED
                if result.errors:
                    state.errors.append(result.errors)
        except Exception as e:
            state.compiling_state = BaseKind.FAILED
            state.errors.append(f"Compilation error: {str(e)}")
            
        return state

    def _check_correctness(
        self, 
        state: EvaluationState, 
        kernel_cfg: KernelEvalConfig
    ) -> EvaluationState:
        """Check kernel correctness."""
        try:
            result = self.correctness.run(state, kernel_cfg)
            state.correctness_result = result
            
            if result.success:
                state.correctness_state = BaseKind.SUCCESS
            else:
                state.correctness_state = BaseKind.FAILED
                if result.errors:
                    state.errors.append(result.errors)
        except Exception as e:
            state.correctness_state = BaseKind.FAILED
            state.errors.append(f"Correctness error: {str(e)}")
            
        return state

    def _check_performance(
        self, 
        state: EvaluationState, 
        kernel_cfg: KernelEvalConfig
    ) -> EvaluationState:
        """Measure kernel performance."""
        try:
            result = self.performance.run(state, kernel_cfg)
            state.performance_result = result
            
            if result is None:
                # No profiling (skipped)
                state.performance_state = BaseKind.SKIPPED
            elif result.success:
                state.performance_state = BaseKind.SUCCESS
            else:
                state.performance_state = BaseKind.FAILED
                if result.errors:
                    state.errors.append(result.errors)
        except Exception as e:
            state.performance_state = BaseKind.FAILED
            state.errors.append(f"Performance error: {str(e)}")
            
        return state

    def _calculate_score(self, state: EvaluationState) -> EvaluationState:
        """Calculate overall evaluation score."""
        score = 0.0
        
        # If compiling failed (not skipped), score is 0
        if state.compiling_state == BaseKind.FAILED:
            state.score = 0.0
            return state
        
        # Correctness contributes 50%
        if state.correctness_state == BaseKind.SUCCESS:
            score += 0.5
        
        # Performance contributes 50% (if not skipped)
        if state.performance_state == BaseKind.SUCCESS:
            score += 0.5
        elif state.performance_state == BaseKind.SKIPPED:
            # If performance is skipped, correctness gets full weight
            if state.correctness_state == BaseKind.SUCCESS:
                score = 1.0
        
        state.score = score
        return state


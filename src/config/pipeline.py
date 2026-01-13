"""
Pipeline configuration for kernel evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional

from .correctness import CorrectnessConfig
from .performance import PerformanceConfig


class KernelType(Enum):
    """Kernel language type."""
    PYTORCH = auto()   # PyTorch kernel
    HIP = auto()       # HIP/ROCm kernel
    CUDA = auto()      # CUDA kernel


class EvalMode(Enum):
    """Evaluation mode."""
    ANALYZE = auto()   # Analyze kernel (requires testcase)
    COMPARE = auto()   # Compare multiple kernels


@dataclass
class PipelineConfig:
    """
    Main pipeline configuration (framework config).
    
    Attributes:
        mode: Evaluation mode (analyze or compare)
        kernel_type: Type of kernel (pytorch, hip, cuda)
        gpu_arch: Target GPU architecture (auto-detected if None)
        correctness_config: Configuration for correctness evaluation
        performance_config: Configuration for performance evaluation
        output_dir: Directory for output files
        verbose: Enable verbose logging
    """
    mode: EvalMode = EvalMode.ANALYZE
    kernel_type: KernelType = KernelType.HIP
    gpu_arch: Optional[str] = None  # Auto-detect if None
    correctness_config: Optional[CorrectnessConfig] = None
    performance_config: Optional[PerformanceConfig] = None
    output_dir: str = "./results"
    verbose: bool = False

    def __post_init__(self):
        if self.correctness_config is None:
            self.correctness_config = CorrectnessConfig()
        if self.performance_config is None:
            self.performance_config = PerformanceConfig(kernel_type=self.kernel_type)
        
        # Auto-detect GPU architecture if not specified
        if self.gpu_arch is None:
            self.gpu_arch = self._detect_gpu_arch()
    
    def _detect_gpu_arch(self) -> str:
        """Auto-detect GPU architecture."""
        try:
            from ..utils import detect_gpu, GPUVendor
            vendor, arch = detect_gpu()
            if arch:
                return arch
            raise RuntimeError("GPU detected but architecture is None")
        except Exception as e:
            raise RuntimeError(
                f"Failed to detect GPU architecture for kernel type '{self.kernel_type.name}'. "
                "Please specify 'gpu_arch' explicitly or ensure GPU tools (rocminfo/nvidia-smi) are available."
            ) from e

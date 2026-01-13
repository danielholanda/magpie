"""
Performance evaluation configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import KernelType


class PerfBackend(Enum):
    """Performance profiling backend."""
    ROCPROF_COMPUTE = auto()   # rocprof-compute for HIP/AMD GPUs
    NCU = auto()               # NVIDIA Nsight Compute for CUDA
    NONE = auto()              # No profiling


@dataclass
class PerformanceConfig:
    """
    Configuration for performance evaluation.
    
    Attributes:
        enabled: Whether to enable performance evaluation
        backend: Profiling backend (auto-selected based on kernel_type if not specified)
        kernel_type: Kernel type (used for auto-selecting backend)
        timeout_seconds: Maximum execution time per profiling run
        profiler_args: Additional arguments for the profiler (rocprof-compute or ncu)
    """
    enabled: bool = True
    backend: Optional[PerfBackend] = None
    kernel_type: Optional["KernelType"] = None
    timeout_seconds: float = 60.0
    profiler_args: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Auto-select backend based on kernel type if not specified
        if self.backend is None and self.kernel_type is not None:
            from .pipeline import KernelType
            if self.kernel_type == KernelType.HIP:
                self.backend = PerfBackend.ROCPROF_COMPUTE
            elif self.kernel_type == KernelType.CUDA:
                self.backend = PerfBackend.NCU
            else:
                self.backend = PerfBackend.NONE

    def get_backend(self) -> PerfBackend:
        """Get the performance backend."""
        return self.backend or PerfBackend.NONE

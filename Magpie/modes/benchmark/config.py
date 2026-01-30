###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Configuration classes for benchmark mode.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class BenchmarkFramework(Enum):
    """Supported benchmark frameworks."""
    VLLM = "vllm"
    SGLANG = "sglang"


@dataclass
class TorchProfilerConfig:
    """
    PyTorch Profiler configuration.
    
    Attributes:
        enabled: Whether torch_profiler is enabled (default: True)
    """
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"enabled": self.enabled}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TorchProfilerConfig":
        """Create from dictionary."""
        return cls(enabled=data.get("enabled", True))


@dataclass
class SystemProfilerConfig:
    """
    System-level profiler configuration (rocprof-compute / ncu).
    
    Attributes:
        enabled: Whether system profiler is enabled (default: False)
        profile_args: Additional arguments for profiler
    """
    enabled: bool = False
    profile_args: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "profile_args": self.profile_args,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemProfilerConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            profile_args=data.get("profile_args", []),
        )


@dataclass
class ProfilerConfig:
    """
    Complete profiler configuration.
    
    Attributes:
        torch_profiler: PyTorch profiler settings (default enabled)
        system_profiler: System profiler settings (default disabled)
    """
    torch_profiler: TorchProfilerConfig = field(default_factory=TorchProfilerConfig)
    system_profiler: SystemProfilerConfig = field(default_factory=SystemProfilerConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "torch_profiler": self.torch_profiler.to_dict(),
            "system_profiler": self.system_profiler.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfilerConfig":
        """Create from dictionary."""
        torch_cfg = data.get("torch_profiler", {})
        sys_cfg = data.get("system_profiler", {})
        return cls(
            torch_profiler=TorchProfilerConfig.from_dict(torch_cfg) if torch_cfg else TorchProfilerConfig(),
            system_profiler=SystemProfilerConfig.from_dict(sys_cfg) if sys_cfg else SystemProfilerConfig(),
        )


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark mode.
    
    Attributes:
        framework: Benchmark framework ("vllm" or "sglang")
        model: Model name or path (e.g., "meta-llama/Llama-2-7b-hf")
        precision: Model precision ("fp8", "fp16", "bf16", "fp4")
        envs: Environment variables for benchmark (TP, CONC, ISL, OSL, etc.)
        profiler: Profiler configuration
        docker_image: Override automatic image selection
        gpu_arch: GPU architecture (auto-detected if not specified)
        timeout_seconds: Benchmark timeout
        inferencemax_path: Path to InferenceMAX installation
        hf_cache_path: HuggingFace cache directory
        runner_type: Hardware runner type for InferenceMAX (e.g., "mi300x", "h100")
    """
    framework: str
    model: str
    precision: str = "fp8"
    
    # Environment variables for benchmark
    envs: Dict[str, Any] = field(default_factory=dict)
    
    # Profiler configuration
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    
    # Docker/execution settings
    docker_image: Optional[str] = None
    gpu_arch: Optional[str] = None
    timeout_seconds: float = 3600.0
    
    # Paths
    inferencemax_path: str = "/root/hao_workspace/InferenceMAX"
    hf_cache_path: Optional[str] = None
    
    # InferenceMAX specific
    runner_type: Optional[str] = None
    benchmark_script: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set defaults."""
        # Normalize framework name
        self.framework = self.framework.lower()
        if self.framework not in ["vllm", "sglang"]:
            raise ValueError(f"Unsupported framework: {self.framework}. Use 'vllm' or 'sglang'.")
        
        # Set default envs if not provided
        if not self.envs:
            self.envs = {
                "TP": 1,
                "CONC": 32,
                "ISL": 1024,
                "OSL": 512,
                "RANDOM_RANGE_RATIO": 0.5,
            }
        
        # Convert profiler dict to ProfilerConfig if needed
        if isinstance(self.profiler, dict):
            self.profiler = ProfilerConfig.from_dict(self.profiler)
    
    def get_env_vars(self) -> Dict[str, str]:
        """
        Get environment variables for InferenceMAX.
        
        Returns:
            Dictionary of environment variable names to values
        """
        env = {
            "MODEL": self.model,
            "PRECISION": self.precision,
        }
        
        # Add all envs as environment variables
        for key, value in self.envs.items():
            env[key.upper()] = str(value)
        
        # Add runner type if specified
        if self.runner_type:
            env["RUNNER_TYPE"] = self.runner_type
        
        return env
    
    def get_benchmark_script_name(self) -> str:
        """
        Determine the InferenceMAX benchmark script name.
        
        Returns:
            Script name like "dsr1_fp8_mi300x.sh"
        """
        if self.benchmark_script:
            return self.benchmark_script
        
        # Auto-generate based on config
        runner = self.runner_type or "mi300x"
        # Format: {exp_name}_{precision}_{runner}.sh
        # For now, use a generic experiment name
        return f"generic_{self.precision}_{runner}.sh"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "framework": self.framework,
            "model": self.model,
            "precision": self.precision,
            "envs": self.envs,
            "profiler": self.profiler.to_dict(),
            "docker_image": self.docker_image,
            "gpu_arch": self.gpu_arch,
            "timeout_seconds": self.timeout_seconds,
            "inferencemax_path": self.inferencemax_path,
            "hf_cache_path": self.hf_cache_path,
            "runner_type": self.runner_type,
            "benchmark_script": self.benchmark_script,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
        """Create from dictionary."""
        profiler_data = data.get("profiler", {})
        profiler = ProfilerConfig.from_dict(profiler_data) if profiler_data else ProfilerConfig()
        
        return cls(
            framework=data.get("framework", "sglang"),
            model=data.get("model", ""),
            precision=data.get("precision", "fp8"),
            envs=data.get("envs", {}),
            profiler=profiler,
            docker_image=data.get("docker_image"),
            gpu_arch=data.get("gpu_arch"),
            timeout_seconds=data.get("timeout_seconds", 3600.0),
            inferencemax_path=data.get("inferencemax_path", "/root/hao_workspace/InferenceMAX"),
            hf_cache_path=data.get("hf_cache_path"),
            runner_type=data.get("runner_type"),
            benchmark_script=data.get("benchmark_script"),
        )


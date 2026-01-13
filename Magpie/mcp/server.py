#!/usr/bin/env python3
"""
Magpie MCP Server

Single MCP server that exposes multiple GPU kernel evaluation tools.
Tools are automatically discovered by MCP clients.

Tools:
  - hardware_spec: Get GPU hardware specifications
  - analyze: Analyze kernel correctness and performance
  - compare: Compare multiple kernels
  - configure_gpu: Configure GPU power/frequency settings

Usage:
    python -m Magpie.mcp
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("magpie")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Tool 1: hardware_spec
# =============================================================================
@mcp.tool()
def hardware_spec(
    device_id: int = 0,
    include_all: bool = False,
) -> str:
    """
    Get GPU hardware specifications.
    
    Args:
        device_id: GPU device ID to query (default: 0)
        include_all: If True, returns info for all available GPUs
    
    Returns:
        JSON with GPU info: vendor, architecture, power, clocks, temperature, memory
    """
    from ..utils import GPUController, MultiGPUController, get_gpu_count
    
    try:
        if include_all:
            controller = MultiGPUController()
            all_info = controller.get_all_hardware_info()
            result = {
                "gpu_count": len(all_info),
                "gpus": {str(d): info.to_dict() for d, info in all_info.items()}
            }
        else:
            controller = GPUController(device_id=device_id)
            info = controller.get_hardware_info()
            result = {
                "gpu_count": get_gpu_count(),
                "gpu": info.to_dict()
            }
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool 2: analyze
# =============================================================================
@mcp.tool()
def analyze(
    kernel_path: str,
    testcase_command: str,
    kernel_type: str = "hip",
    working_dir: str = "",
    compile_command: str = "",
    check_performance: bool = True,
) -> str:
    """
    Analyze a GPU kernel for correctness and performance.
    
    Args:
        kernel_path: Path to kernel source file (.hip, .cu, .py)
        testcase_command: Command to run the test case
        kernel_type: "hip", "cuda", or "pytorch"
        working_dir: Working directory (default: kernel's parent dir)
        compile_command: Custom compile command (optional)
        check_performance: Run performance profiling (default: True)
    
    Returns:
        JSON with: compiling_state, correctness_state, performance_state, score, errors
    """
    from ..config import KernelType, KernelEvalConfig
    from ..modes import AnalyzeMode
    from ..modes.analyze_eval.analyzer import AnalyzeConfig
    
    try:
        # Parse kernel type
        type_map = {
            "hip": KernelType.HIP,
            "cuda": KernelType.CUDA,
            "pytorch": KernelType.PYTORCH,
            "torch": KernelType.PYTORCH,
        }
        ktype = type_map.get(kernel_type.lower(), KernelType.HIP)
        
        kernel_file = Path(kernel_path)
        kernel_config = KernelEvalConfig(
            kernel_id=kernel_file.stem,
            kernel_type=ktype,
            source_file_path=[str(kernel_file)],
            testcase_command=testcase_command.split(),
            compiling_command=compile_command.split() if compile_command else None,
            working_dir=working_dir or str(kernel_file.parent),
        )
        
        analyzer = AnalyzeMode(AnalyzeConfig(
            kernel_type=ktype,
            check_performance=check_performance,
        ))
        
        result = analyzer.analyze(kernel_config)
        
        return json.dumps({
            "kernel_id": kernel_config.kernel_id,
            "kernel_type": ktype.name,
            "compiling_state": result.compiling_state.name,
            "correctness_state": result.correctness_state.name,
            "performance_state": result.performance_state.name,
            "score": result.score,
            "errors": result.errors or [],
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "kernel_path": kernel_path})


# =============================================================================
# Tool 3: compare
# =============================================================================
@mcp.tool()
def compare(
    kernel_paths: List[str],
    kernel_type: str = "hip",
    testcase_command: str = "",
    baseline_index: int = 0,
    check_performance: bool = True,
) -> str:
    """
    Compare multiple GPU kernels for performance and correctness.
    
    Args:
        kernel_paths: List of kernel source file paths (minimum 2)
        kernel_type: "hip", "cuda", or "pytorch"
        testcase_command: Command to run the test case (optional)
        baseline_index: Index of baseline kernel for comparison (default: 0)
        check_performance: Run performance profiling (default: True)
    
    Returns:
        JSON with comparison results, ranking, and summary
    """
    from ..config import KernelType, KernelEvalConfig
    from ..modes import CompareMode
    from ..modes.compare_eval.comparator import CompareConfig
    
    try:
        if len(kernel_paths) < 2:
            return json.dumps({"error": "Compare requires at least 2 kernels"})
        
        type_map = {
            "hip": KernelType.HIP,
            "cuda": KernelType.CUDA,
            "pytorch": KernelType.PYTORCH,
        }
        ktype = type_map.get(kernel_type.lower(), KernelType.HIP)
        
        kernel_configs = []
        for i, path in enumerate(kernel_paths):
            kernel_file = Path(path)
            kernel_configs.append(KernelEvalConfig(
                kernel_id=f"kernel_{i}_{kernel_file.stem}",
                kernel_type=ktype,
                source_file_path=[str(kernel_file)],
                testcase_command=testcase_command.split() if testcase_command else None,
                working_dir=str(kernel_file.parent),
            ))
        
        comparator = CompareMode(CompareConfig(
            baseline_index=baseline_index,
            check_performance=check_performance,
        ))
        
        comparison = comparator.compare(kernel_configs)
        return json.dumps(comparison.to_dict(), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "kernel_paths": kernel_paths})


# =============================================================================
# Tool 4: configure_gpu
# =============================================================================
@mcp.tool()
def configure_gpu(
    device_ids: List[int] = None,
    power_limit_watts: int = None,
    gpu_clock_mhz: List[int] = None,
    mem_clock_mhz: List[int] = None,
    reset: bool = False,
) -> str:
    """
    Configure GPU hardware settings. Requires root/sudo permissions.
    
    Args:
        device_ids: GPU device IDs to configure (default: all GPUs)
        power_limit_watts: Power limit in watts (e.g., 300)
        gpu_clock_mhz: GPU clock range [min, max] MHz (e.g., [1500, 1800])
        mem_clock_mhz: Memory clock range [min, max] MHz
        reset: Reset GPUs to default settings
    
    Returns:
        JSON with configuration results for each GPU
    """
    from ..utils import MultiGPUController, MultiGPUConfig, GPUConfig
    
    try:
        controller = MultiGPUController(device_ids=device_ids)
        
        if reset:
            results = controller.reset_all()
            return json.dumps({
                "action": "reset",
                "results": {str(k): v for k, v in results.items()},
            }, indent=2)
        
        default_config = GPUConfig(
            power_limit_watts=power_limit_watts,
            gpu_clock_mhz=tuple(gpu_clock_mhz) if gpu_clock_mhz else None,
            mem_clock_mhz=tuple(mem_clock_mhz) if mem_clock_mhz else None,
        )
        
        multi_config = MultiGPUConfig(
            default_config=default_config,
            device_ids=device_ids,
            parallel=True,
        )
        
        results = controller.apply_config(multi_config)
        return json.dumps({
            "action": "configure",
            "config": {
                "power_limit_watts": power_limit_watts,
                "gpu_clock_mhz": gpu_clock_mhz,
                "mem_clock_mhz": mem_clock_mhz,
            },
            "results": {str(k): v for k, v in results.items()},
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    mcp.run()

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
  - discover_kernels: Discover analyzable kernels in a project
  - suggest_optimizations: Get optimization suggestions based on analysis results

Usage:
    python -m Magpie.mcp
"""

import json
import logging
import glob
import os
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("magpie")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_framework_config() -> Dict[str, Any]:
    """
    Load framework configuration from config.yaml.
    
    Searches for config.yaml in:
    1. Current directory
    2. Magpie package directory
    
    Returns:
        Configuration dictionary
    """
    # Try current directory first
    config_paths = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "Magpie" / "config.yaml",
        Path(__file__).parent.parent / "config.yaml",  # Magpie/config.yaml
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
    
    return {}


def _get_scheduler_config_from_yaml(environment: str = "local") -> "SchedulerConfig":
    """
    Get scheduler configuration from framework config.yaml.
    
    Args:
        environment: Execution environment override
        
    Returns:
        SchedulerConfig with settings from config.yaml
    """
    from ..core import SchedulerConfig, EnvironmentType
    
    config = _load_framework_config()
    sched_cfg = config.get("scheduler", {})
    
    # Get settings from config, with defaults
    max_workers = sched_cfg.get("max_workers", 1)
    docker_image = sched_cfg.get("docker_image", None)
    
    # Environment can be overridden by parameter
    env_type = EnvironmentType(environment.lower())
    
    return SchedulerConfig(
        environment_type=env_type,
        max_workers=max_workers,
        docker_image=docker_image,
    )


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
    environment: str = "local",
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
        environment: Execution environment "local" or "container"
    
    Returns:
        JSON with comprehensive analysis results including:
        - compiling_state, correctness_state, performance_state
        - score (0.0 - 1.0)
        - compiling_result: compilation details (if applicable)
        - correctness_result: testcase execution results
        - performance_result: detailed performance metrics
          - summary: key GPU metrics (VALU_FLOPs, MFMA_FLOPs, utilization, etc.)
          - kernels: per-kernel statistics (dispatch count, duration)
          - workload_dir: path to raw profiler data
    """
    from ..config import KernelType, KernelEvalConfig
    from ..core import Scheduler
    
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
        
        # Create scheduler with config from config.yaml
        scheduler_config = _get_scheduler_config_from_yaml(environment)
        scheduler = Scheduler(scheduler_config)
        
        if not scheduler.initialize():
            return json.dumps({"error": "Failed to initialize scheduler"})
        
        try:
            # Run analysis via scheduler
            task_result = scheduler.run_analyze(
                kernel_configs=[kernel_config],
                check_performance=check_performance,
            )
            
            if task_result.success and task_result.results:
                result = task_result.results[0]
                return json.dumps(_format_analysis_result(result, kernel_config, ktype), indent=2)
            else:
                return json.dumps({
                    "error": "Analysis failed",
                    "errors": task_result.errors,
                })
        finally:
            scheduler.shutdown()
            
    except Exception as e:
        return json.dumps({"error": str(e), "kernel_path": kernel_path})


def _format_analysis_result(result: dict, kernel_config, ktype) -> dict:
    """
    Format analysis result into a comprehensive response.
    
    Args:
        result: Raw result from analyzer (dict or object)
        kernel_config: Kernel configuration
        ktype: Kernel type enum
        
    Returns:
        Formatted result dictionary
    """
    if isinstance(result, dict):
        # Result is already a dict
        formatted = {
            "kernel_id": kernel_config.kernel_id,
            "kernel_type": ktype.name,
            "compiling_state": result.get("compiling_state", "UNKNOWN"),
            "correctness_state": result.get("correctness_state", "UNKNOWN"),
            "performance_state": result.get("performance_state", "UNKNOWN"),
            "score": result.get("score", 0.0),
            "errors": result.get("errors", []),
        }
        
        # Add compiling result if available
        compiling_result = result.get("compiling_result")
        if compiling_result:
            formatted["compiling_result"] = _format_compiling_result(compiling_result)
        
        # Add correctness result if available
        correctness_result = result.get("correctness_result")
        if correctness_result:
            formatted["correctness_result"] = _format_correctness_result(correctness_result)
        
        # Add performance result if available
        performance_result = result.get("performance_result")
        if performance_result:
            formatted["performance_result"] = _format_performance_result(performance_result)
        
        return formatted
    else:
        # Result is an object (EvaluationState)
        formatted = {
            "kernel_id": kernel_config.kernel_id,
            "kernel_type": ktype.name,
            "compiling_state": result.compiling_state.name if hasattr(result.compiling_state, 'name') else str(result.compiling_state),
            "correctness_state": result.correctness_state.name if hasattr(result.correctness_state, 'name') else str(result.correctness_state),
            "performance_state": result.performance_state.name if hasattr(result.performance_state, 'name') else str(result.performance_state),
            "score": result.score,
            "errors": result.errors or [],
        }
        
        if hasattr(result, 'compiling_result') and result.compiling_result:
            formatted["compiling_result"] = _format_compiling_result(result.compiling_result)
        
        if hasattr(result, 'correctness_result') and result.correctness_result:
            formatted["correctness_result"] = _format_correctness_result(result.correctness_result)
        
        if hasattr(result, 'performance_result') and result.performance_result:
            formatted["performance_result"] = _format_performance_result(result.performance_result)
        
        return formatted


def _format_compiling_result(result) -> dict:
    """Format compiling result."""
    if isinstance(result, dict):
        return {
            "success": result.get("success", False),
            "compile_time_seconds": result.get("compile_time_seconds"),
            "errors": result.get("errors"),
        }
    elif hasattr(result, 'to_dict'):
        return result.to_dict()
    else:
        return {
            "success": getattr(result, 'success', False),
            "compile_time_seconds": getattr(result, 'compile_time_seconds', None),
            "errors": getattr(result, 'errors', None),
        }


def _format_correctness_result(result) -> dict:
    """Format correctness result."""
    if isinstance(result, dict):
        return {
            "success": result.get("success", False),
            "errors": result.get("errors"),
        }
    elif hasattr(result, 'to_dict'):
        return result.to_dict()
    else:
        return {
            "success": getattr(result, 'success', False),
            "errors": getattr(result, 'errors', None),
        }


def _format_performance_result(result) -> dict:
    """
    Format performance result with summary metrics and kernel statistics.
    
    Returns a structured dict with:
    - success: bool
    - errors: str or None
    - workload_dir: path to raw profiler data
    - summary: key performance metrics (dict of metric_name -> {value, unit, peak, pct_of_peak})
    - kernels: list of kernel summaries (kernel_name, dispatch_count, duration_ns stats)
    """
    if isinstance(result, dict):
        return {
            "success": result.get("success", False),
            "errors": result.get("errors"),
            "workload_dir": result.get("workload_dir"),
            "summary": result.get("summary", {}),
            "kernels": result.get("kernels", []),
        }
    elif hasattr(result, 'to_dict'):
        return result.to_dict()
    else:
        # Handle PerformanceResult object directly
        formatted = {
            "success": getattr(result, 'success', False),
            "errors": getattr(result, 'errors', None),
            "workload_dir": getattr(result, 'workload_dir', None),
        }
        
        # Extract summary metrics
        if hasattr(result, 'get_summary_metrics'):
            formatted["summary"] = result.get_summary_metrics()
        elif hasattr(result, 'metrics'):
            summary = {}
            for m in result.metrics:
                summary[m.name] = {
                    "value": m.value,
                    "unit": m.unit,
                }
                if hasattr(m, 'peak') and m.peak is not None:
                    summary[m.name]["peak"] = m.peak
                if hasattr(m, 'pct_of_peak') and m.pct_of_peak is not None:
                    summary[m.name]["pct_of_peak"] = m.pct_of_peak
            formatted["summary"] = summary
        else:
            formatted["summary"] = {}
        
        # Extract kernel statistics
        if hasattr(result, 'get_kernel_summary'):
            formatted["kernels"] = result.get_kernel_summary()
        else:
            formatted["kernels"] = []
        
        return formatted


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
    environment: str = "local",
) -> str:
    """
    Compare multiple GPU kernels for performance and correctness.
    
    Args:
        kernel_paths: List of kernel source file paths (minimum 2)
        kernel_type: "hip", "cuda", or "pytorch"
        testcase_command: Command to run the test case (optional)
        baseline_index: Index of baseline kernel for comparison (default: 0)
        check_performance: Run performance profiling (default: True)
        environment: Execution environment "local" or "container"
    
    Returns:
        JSON with comparison results, ranking, and summary
    """
    from ..config import KernelType, KernelEvalConfig
    from ..core import Scheduler
    
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
        
        # Create scheduler with config from config.yaml
        scheduler_config = _get_scheduler_config_from_yaml(environment)
        scheduler = Scheduler(scheduler_config)
        
        if not scheduler.initialize():
            return json.dumps({"error": "Failed to initialize scheduler"})
        
        try:
            # Run comparison via scheduler
            task_result = scheduler.run_compare(
                kernel_configs=kernel_configs,
                baseline_index=baseline_index,
                check_performance=check_performance,
            )
            
            if task_result.success and task_result.results:
                comparison = task_result.results
                if isinstance(comparison, dict):
                    return json.dumps(comparison, indent=2)
                elif hasattr(comparison, 'to_dict'):
                    return json.dumps(comparison.to_dict(), indent=2)
                else:
                    return json.dumps({"result": str(comparison)}, indent=2)
            else:
                return json.dumps({
                    "error": "Comparison failed",
                    "errors": task_result.errors,
                })
        finally:
            scheduler.shutdown()
            
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
# Tool 5: discover_kernels
# =============================================================================
@mcp.tool()
def discover_kernels(
    project_path: str,
    kernel_type: str = "hip",
    include_tests: bool = True,
    include_examples: bool = True,
) -> str:
    """
    Discover analyzable GPU kernels in a project directory.
    
    Scans the project for kernel source files and attempts to find
    corresponding test binaries or executables.
    
    Args:
        project_path: Root path of the project to scan
        kernel_type: Type of kernels to find: "hip", "cuda", or "all"
        include_tests: Include test directories in search
        include_examples: Include example directories in search
    
    Returns:
        JSON with list of discovered kernels, each containing:
        - source_file: Path to kernel source
        - possible_binaries: List of potential test/example binaries
        - suggested_config: Suggested kernel configuration for analyze()
    """
    try:
        project = Path(project_path)
        if not project.exists():
            return json.dumps({"error": f"Project path does not exist: {project_path}"})
        
        # Define file patterns based on kernel type
        patterns = []
        if kernel_type in ("hip", "all"):
            patterns.extend(["**/*.hip", "**/*.cpp"])
        if kernel_type in ("cuda", "all"):
            patterns.extend(["**/*.cu", "**/*.cuh"])
        
        # Find source files
        discovered = []
        build_dirs = ["build", "bin", "out", "cmake-build-release", "cmake-build-debug"]
        
        for pattern in patterns:
            for source_file in project.glob(pattern):
                # Skip if not in relevant directories
                rel_path = str(source_file.relative_to(project))
                is_test = "test" in rel_path.lower()
                is_example = "example" in rel_path.lower()
                
                if not include_tests and is_test:
                    continue
                if not include_examples and is_example:
                    continue
                
                # Look for corresponding binaries
                stem = source_file.stem
                possible_binaries = []
                
                for build_dir in build_dirs:
                    build_path = project / build_dir
                    if build_path.exists():
                        # Look for binaries matching the source file name
                        for binary in build_path.glob(f"**/{stem}"):
                            if binary.is_file() and os.access(binary, os.X_OK):
                                possible_binaries.append(str(binary))
                        for binary in build_path.glob(f"**/*{stem}*"):
                            if binary.is_file() and os.access(binary, os.X_OK):
                                if str(binary) not in possible_binaries:
                                    possible_binaries.append(str(binary))
                
                # Generate suggested config
                suggested_config = {
                    "kernel_path": str(source_file),
                    "kernel_type": "hip" if source_file.suffix in (".hip", ".cpp") else "cuda",
                    "working_dir": str(project / "build") if (project / "build").exists() else str(project),
                }
                
                if possible_binaries:
                    suggested_config["testcase_command"] = possible_binaries[0]
                
                discovered.append({
                    "source_file": str(source_file),
                    "name": stem,
                    "is_test": is_test,
                    "is_example": is_example,
                    "possible_binaries": possible_binaries[:5],  # Limit to 5
                    "suggested_config": suggested_config,
                })
        
        # Sort by relevance (tests first, then examples)
        discovered.sort(key=lambda x: (not x["is_test"], not x["is_example"], x["name"]))
        
        return json.dumps({
            "project_path": str(project),
            "kernel_type": kernel_type,
            "total_found": len(discovered),
            "kernels": discovered[:50],  # Limit to 50 results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool 6: suggest_optimizations
# =============================================================================
@mcp.tool()
def suggest_optimizations(
    analysis_result: str,
) -> str:
    """
    Analyze performance results and suggest optimizations.
    
    Takes the JSON output from analyze() and provides actionable
    optimization suggestions based on the performance metrics.
    
    Args:
        analysis_result: JSON string from analyze() output
    
    Returns:
        JSON with optimization suggestions including:
        - bottlenecks: Identified performance bottlenecks
        - suggestions: Specific optimization recommendations
        - priority: Ranked list of optimizations by potential impact
    """
    try:
        data = json.loads(analysis_result)
        
        suggestions = []
        bottlenecks = []
        
        # Check if analysis was successful
        if data.get("performance_state") != "SUCCESS":
            return json.dumps({
                "error": "Performance analysis not available",
                "reason": data.get("errors", ["Unknown error"]),
            })
        
        perf = data.get("performance_result", {})
        summary = perf.get("summary", {})
        kernels = perf.get("kernels", [])
        
        # Analyze CU utilization
        active_cus = summary.get("Active CUs", {})
        if active_cus:
            cu_pct = active_cus.get("pct_of_peak", 0)
            if cu_pct < 50:
                bottlenecks.append({
                    "type": "low_cu_occupancy",
                    "severity": "high" if cu_pct < 25 else "medium",
                    "value": f"{cu_pct}% CU utilization",
                    "description": "Low Compute Unit occupancy indicates underutilization of GPU resources"
                })
                suggestions.append({
                    "category": "parallelism",
                    "suggestion": "Increase thread block size or launch more thread blocks",
                    "impact": "high",
                    "details": f"Current CU utilization is only {cu_pct}%. Consider increasing occupancy by adjusting block dimensions or using more registers efficiently."
                })
        
        # Analyze MFMA utilization
        mfma_util = summary.get("MFMA_Util", {})
        if mfma_util:
            mfma_pct = mfma_util.get("value", 0)
            if mfma_pct < 50:
                # Check if this is a matrix-heavy workload
                mfma_f16 = summary.get("MFMA_FLOPs_F16", {}).get("value", 0)
                mfma_bf16 = summary.get("MFMA_FLOPs_BF16", {}).get("value", 0)
                if mfma_f16 > 0 or mfma_bf16 > 0:
                    bottlenecks.append({
                        "type": "low_mfma_utilization",
                        "severity": "medium",
                        "value": f"{mfma_pct}% MFMA utilization",
                        "description": "Matrix operations not fully utilizing MFMA units"
                    })
                    suggestions.append({
                        "category": "matrix_ops",
                        "suggestion": "Optimize matrix tile sizes for MFMA instructions",
                        "impact": "high",
                        "details": "Consider using tile sizes that match MFMA instruction requirements (16x16, 32x32, etc.)"
                    })
        
        # Analyze VALU utilization
        valu_util = summary.get("VALU_Util", {})
        if valu_util:
            valu_pct = valu_util.get("value", 0)
            if valu_pct > 50:
                # High VALU but potentially low MFMA could indicate scalar bottleneck
                mfma_pct = mfma_util.get("value", 0) if mfma_util else 0
                if mfma_pct < 10:
                    suggestions.append({
                        "category": "vectorization",
                        "suggestion": "Replace scalar operations with matrix operations where possible",
                        "impact": "medium",
                        "details": f"High VALU utilization ({valu_pct}%) with low MFMA ({mfma_pct}%) suggests potential for matrix optimization"
                    })
        
        # Analyze memory bandwidth
        vmem_util = summary.get("VMEM_Util", {})
        if vmem_util:
            vmem_pct = vmem_util.get("value", 0)
            if vmem_pct > 70:
                bottlenecks.append({
                    "type": "memory_bound",
                    "severity": "high",
                    "value": f"{vmem_pct}% VMEM utilization",
                    "description": "Kernel is memory bandwidth limited"
                })
                suggestions.append({
                    "category": "memory",
                    "suggestion": "Optimize memory access patterns and use LDS caching",
                    "impact": "high",
                    "details": "Consider coalescing memory accesses, using shared memory (LDS), or reducing memory traffic through computation reuse"
                })
        
        # Analyze kernel dispatch overhead
        if kernels:
            short_kernels = [k for k in kernels if k.get("duration_ns", {}).get("avg", 0) < 10000]  # < 10µs
            if len(short_kernels) > len(kernels) * 0.5:
                bottlenecks.append({
                    "type": "kernel_launch_overhead",
                    "severity": "medium",
                    "value": f"{len(short_kernels)} short kernels (<10µs)",
                    "description": "Many short kernel launches can be dominated by launch overhead"
                })
                suggestions.append({
                    "category": "fusion",
                    "suggestion": "Fuse multiple short kernels into larger kernels",
                    "impact": "medium",
                    "details": "Consider kernel fusion to reduce launch overhead and improve data locality"
                })
        
        # Prioritize suggestions
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x.get("impact", "low"), 2))
        
        return json.dumps({
            "kernel_id": data.get("kernel_id", "unknown"),
            "overall_score": data.get("score", 0),
            "bottlenecks": bottlenecks,
            "suggestions": suggestions,
            "summary": {
                "total_bottlenecks": len(bottlenecks),
                "total_suggestions": len(suggestions),
                "high_impact_suggestions": len([s for s in suggestions if s.get("impact") == "high"]),
            }
        }, indent=2)
        
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool 7: create_kernel_config
# =============================================================================
@mcp.tool()
def create_kernel_config(
    kernel_id: str,
    kernel_path: str,
    testcase_command: str,
    kernel_type: str = "hip",
    working_dir: str = "",
    compile_command: str = "",
    output_path: str = "",
) -> str:
    """
    Create a kernel configuration YAML content for use with Magpie CLI.
    
    This tool generates a properly formatted kernel configuration that
    can be used with 'Magpie analyze --kernel-config <file>'.
    
    NOTE: This tool is READ-ONLY and does not write files to disk.
    The agent should use the returned config_content directly with
    the analyze() tool, or save it manually if needed.
    
    Args:
        kernel_id: Unique identifier for the kernel
        kernel_path: Path to the kernel source file
        testcase_command: Command to run the test case
        kernel_type: "hip", "cuda", or "pytorch"
        working_dir: Working directory for execution
        compile_command: Custom compile command (optional)
        output_path: DEPRECATED - file saving is disabled for safety
    
    Returns:
        JSON with:
        - config_content: The YAML configuration content
        - usage: Example command to use the config
    """
    try:
        import yaml
        
        # Build kernel config
        kernel_config = {
            "kernel": {
                "id": kernel_id,
                "type": kernel_type,
                "source_files": [kernel_path],
                "testcase_command": testcase_command,
            }
        }
        
        if working_dir:
            kernel_config["kernel"]["working_dir"] = working_dir
        
        if compile_command:
            kernel_config["kernel"]["compile_command"] = compile_command
        
        # Generate YAML content
        yaml_content = f"# Kernel Configuration for {kernel_id}\n"
        yaml_content += f"# Generated by Magpie MCP\n\n"
        yaml_content += yaml.dump(kernel_config, default_flow_style=False, sort_keys=False)
        
        result = {
            "config_content": yaml_content,
            "usage": f"python -m Magpie analyze --kernel-config <config_file>",
        }
        
        # Note: File saving is disabled by default to avoid unintended side effects.
        # The agent should use the config_content directly or save it manually.
        # To enable file saving, uncomment the code below:
        #
        # if output_path:
        #     output_file = Path(output_path)
        #     if output_file.exists():
        #         result["warning"] = f"File {output_path} already exists, not overwriting"
        #     else:
        #         output_file.parent.mkdir(parents=True, exist_ok=True)
        #         output_file.write_text(yaml_content)
        #         result["saved_to"] = str(output_file)
        #         result["usage"] = f"python -m Magpie analyze --kernel-config {output_path}"
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    mcp.run()

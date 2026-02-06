###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Magpie: GPU Kernel Evaluation Framework

Usage:
    # Run from project root
    python -m Magpie analyze kernel.hip -t "./test.sh"

    # Analyze with kernel config file
    python -m Magpie analyze --kernel-config my_kernel.yaml

    # Compare kernels
    python -m Magpie compare kernel1.hip kernel2.hip

    # Compare with kernel config file
    python -m Magpie compare --kernel-config kernels.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

from .config import KernelType, KernelEvalConfig
from .core import Scheduler, SchedulerConfig, EnvironmentType
from .eval import EvaluationState, BaseKind
from .utils import get_gpu_info

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def setup_logging(config: Dict[str, Any], verbose: bool = False) -> None:
    """Configure logging."""
    log_cfg = config.get("logging", {})
    level_str = "DEBUG" if verbose else log_cfg.get("level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)

    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_kernel_type(type_str: str) -> KernelType:
    """Parse kernel type from string."""
    type_map = {
        "hip": KernelType.HIP,
        "cuda": KernelType.CUDA,
        "pytorch": KernelType.PYTORCH,
        "torch": KernelType.PYTORCH,
        "py": KernelType.PYTORCH,
    }
    return type_map.get(type_str.lower(), KernelType.HIP)


def load_kernel_config(kernel_config_path: Path) -> List[KernelEvalConfig]:
    """
    Load kernel configuration from YAML file.

    Returns:
        List of KernelEvalConfig objects
    """
    data = load_yaml(kernel_config_path)
    configs = []

    # Single kernel
    if "kernel" in data:
        cfg = _parse_kernel_entry(data["kernel"])
        if cfg:
            configs.append(cfg)

    # Multiple kernels
    if "kernels" in data:
        for entry in data["kernels"]:
            cfg = _parse_kernel_entry(entry)
            if cfg:
                configs.append(cfg)

    return configs


def _parse_command_list(cmd_entry) -> Optional[List]:
    """
    Parse a command entry which can be:

    Single command formats:
    - A string: "make build" -> ["make", "build"]
    - A list of strings: ["make", "build"] -> ["make", "build"]

    Multiple commands format (list of lists):
    - [["make", "clean"], ["make", "build"]] -> [["make", "clean"], ["make", "build"]]
    - In YAML:
        compile_command:
          - ["cmake", "-B", "build"]
          - ["cmake", "--build", "build"]
      Or:
        compile_command:
          - - cmake
            - -B
            - build
          - - cmake
            - --build
            - build

    Returns:
        - Single command as List[str]
        - Multiple commands as List[List[str]]
        - None if no command
    """
    if cmd_entry is None:
        return None

    cmd_entry = _expand_env_vars(cmd_entry)

    if isinstance(cmd_entry, str):
        # Single string command: "make build" -> ["make", "build"]
        return cmd_entry.split()

    if isinstance(cmd_entry, list):
        if len(cmd_entry) == 0:
            return None

        # Check if it's a list of lists (multiple commands)
        # e.g., [["make", "clean"], ["make", "build"]]
        if isinstance(cmd_entry[0], list):
            return cmd_entry

        # It's a list of strings - treat as single command
        # e.g., ["make", "build"]
        if all(isinstance(item, str) for item in cmd_entry):
            return cmd_entry

    return None


def _expand_env_vars(value):
    """Recursively expand environment variables in strings."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(val) for key, val in value.items()}
    return value


def _parse_kernel_entry(entry: Dict[str, Any]) -> Optional[KernelEvalConfig]:
    """Parse a single kernel entry from config."""
    if not entry:
        return None

    kernel_type = parse_kernel_type(entry.get("type", "hip"))

    source_files = _expand_env_vars(entry.get("source_files", []))
    working_dir = _expand_env_vars(entry.get("working_dir"))
    env = _expand_env_vars(entry.get("env"))

    # Parse testcase command(s)
    testcase_cmd = _parse_command_list(entry.get("testcase_command"))

    # Parse compile command(s)
    compile_cmd = _parse_command_list(entry.get("compile_command"))

    # Parse prof command(s)
    prof_cmd = _parse_command_list(entry.get("prof_command"))

    return KernelEvalConfig(
        kernel_id=entry.get("id", "kernel"),
        kernel_type=kernel_type,
        source_file_path=source_files,
        working_dir=working_dir,
        env=env,
        testcase_command=testcase_cmd,
        compiling_command=compile_cmd,
        prof_command=prof_cmd,
        get_inputs_func=entry.get("get_inputs_func", "get_inputs"),
        get_init_inputs_func=entry.get("get_init_inputs_func", "get_init_inputs"),
    )


def _get_compiling_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get compiling configuration from framework config.

    Args:
        config: Framework config dict

    Returns:
        Dict with enable_default_compile
    """
    compile_cfg = config.get("compiling", {})
    return {
        "enable_default_compile": compile_cfg.get("enable_default_compile", False),
    }


def _get_performance_config(
    config: Dict[str, Any], kernel_type: KernelType
) -> Dict[str, Any]:
    """
    Get performance configuration from framework config.

    Args:
        config: Framework config dict
        kernel_type: Kernel type to determine which profiler args to use

    Returns:
        Dict with timeout_seconds, profiler_args, and rocprof_config/ncu_config
    """
    perf_cfg = config.get("performance", {})

    # Get timeout
    timeout = perf_cfg.get("timeout_seconds", 300.0)

    # Get profiler args and config based on kernel type
    profiler_args = []
    rocprof_config = {}
    ncu_config = {}

    if kernel_type == KernelType.HIP:
        rocprof_cfg = perf_cfg.get("rocprof_compute", {})

        # Build full rocprof config
        rocprof_config = {
            "workload_dir": rocprof_cfg.get("workload_dir", "./workloads"),
            "metric_blocks": rocprof_cfg.get(
                "metric_blocks", ["1", "2", "5", "10", "11", "12", "14", "16", "17"]
            ),
            "output_format": rocprof_cfg.get("output_format", "csv"),
            "profile_args": rocprof_cfg.get("profile_args", []),
            "analyze_args": rocprof_cfg.get("analyze_args", []),
        }
    elif kernel_type == KernelType.CUDA:
        ncu_cfg = perf_cfg.get("ncu", {})
        profiler_args = ncu_cfg.get("args", [])
        ncu_config = {
            "args": ncu_cfg.get("args", []),
            "metrics": ncu_cfg.get("metrics", []),
        }

    return {
        "timeout_seconds": timeout,
        "profiler_args": profiler_args,
        "rocprof_config": rocprof_config,
        "ncu_config": ncu_config,
    }


def _get_compare_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get compare configuration from framework config.

    Args:
        config: Framework config dict

    Returns:
        Dict with compare settings
    """
    return config.get("compare", {})


def _get_scheduler_config(config: Dict[str, Any], args) -> SchedulerConfig:
    """
    Get scheduler configuration from framework config and CLI args.

    Args:
        config: Framework config dict
        args: CLI arguments

    Returns:
        SchedulerConfig
    """
    sched_cfg = config.get("scheduler", {})

    # Determine environment type
    env_type_str = getattr(args, "environment", None) or sched_cfg.get(
        "environment", "local"
    )
    env_type = EnvironmentType(env_type_str.lower())

    # Get worker count
    max_workers = getattr(args, "workers", None) or sched_cfg.get("max_workers", 1)

    # Get GPU devices
    gpu_devices = sched_cfg.get("gpu_devices", [0])

    # Get docker image
    docker_image = getattr(args, "docker_image", None) or sched_cfg.get("docker_image")

    return SchedulerConfig(
        environment_type=env_type,
        max_workers=max_workers,
        gpu_devices=gpu_devices,
        docker_image=docker_image,
    )


def run_analyze(args, config: Dict[str, Any]) -> int:
    """Run analyze mode."""
    # Load kernel configs
    kernel_configs = []

    if args.kernel_config:
        # Load from kernel config file
        kernel_configs = load_kernel_config(args.kernel_config)
        if not kernel_configs:
            logger.error(f"No kernels found in {args.kernel_config}")
            return 1
    elif args.kernels:
        # Build from CLI arguments
        if not args.testcase:
            logger.error("Analyze mode requires --testcase")
            print("Error: --testcase is required for analyze mode")
            return 1

        kernel_type = parse_kernel_type(args.type)

        for path in args.kernels:
            if not path.exists():
                logger.error(f"Kernel not found: {path}")
                continue

            kernel_configs.append(
                KernelEvalConfig(
                    kernel_id=path.stem,
                    kernel_type=kernel_type,
                    source_file_path=[str(path)],
                    testcase_command=args.testcase.split(),
                    compiling_command=args.compile_cmd.split()
                    if args.compile_cmd
                    else None,
                    working_dir=str(path.parent),
                )
            )
    else:
        logger.error("No kernels specified")
        return 1

    # Get kernel type for configuration
    kernel_type = kernel_configs[0].kernel_type if kernel_configs else KernelType.HIP

    # Get config from framework config
    compile_settings = _get_compiling_config(config)
    perf_settings = _get_performance_config(config, kernel_type)

    # Create scheduler
    scheduler_config = _get_scheduler_config(config, args)
    scheduler = Scheduler(scheduler_config)

    if not scheduler.initialize():
        logger.error("Failed to initialize scheduler")
        return 1

    try:
        # Run analysis via scheduler
        result = scheduler.run_analyze(
            kernel_configs=kernel_configs,
            enable_default_compile=compile_settings["enable_default_compile"],
            check_performance=not args.no_perf,
            timeout_seconds=perf_settings["timeout_seconds"],
            profiler_args=perf_settings["profiler_args"],
            rocprof_config=perf_settings["rocprof_config"],
            ncu_config=perf_settings["ncu_config"],
        )

        # Print and save results
        if result.success and result.results:
            for kernel_cfg, state in zip(kernel_configs, result.results):
                # Reconstruct EvaluationState if needed
                if isinstance(state, dict):
                    state = _dict_to_eval_state(state)
                _print_result(kernel_cfg, state)

            _save_results(result.results, args.output_dir, "analyze")
        else:
            logger.error(f"Analysis failed: {result.errors}")
            return 1

        # Count failures
        failed = 0
        for state in result.results:
            if isinstance(state, dict):
                correctness = state.get("correctness_state", "UNKNOWN")
                if correctness != "SUCCESS":
                    failed += 1
            elif hasattr(state, "correctness_state"):
                if state.correctness_state != BaseKind.SUCCESS:
                    failed += 1

        return 1 if failed > 0 else 0

    finally:
        scheduler.shutdown()


def run_compare(args, config: Dict[str, Any]) -> int:
    """Run compare mode."""
    # Load kernel configs
    kernel_configs = []

    if args.kernel_config:
        kernel_configs = load_kernel_config(args.kernel_config)
    elif args.kernels:
        kernel_type = parse_kernel_type(args.type)

        for i, path in enumerate(args.kernels):
            if not path.exists():
                logger.error(f"Kernel not found: {path}")
                return 1

            kernel_configs.append(
                KernelEvalConfig(
                    kernel_id=f"kernel_{i}_{path.stem}",
                    kernel_type=kernel_type,
                    source_file_path=[str(path)],
                    testcase_command=args.testcase.split() if args.testcase else None,
                    working_dir=str(path.parent),
                )
            )

    if len(kernel_configs) < 2:
        logger.error("Compare mode requires at least 2 kernels")
        return 1

    # Get kernel type for configuration
    kernel_type = kernel_configs[0].kernel_type if kernel_configs else KernelType.HIP

    # Get config from framework config
    compile_settings = _get_compiling_config(config)
    perf_settings = _get_performance_config(config, kernel_type)
    compare_settings = _get_compare_config(config)

    # Create scheduler
    scheduler_config = _get_scheduler_config(config, args)
    scheduler = Scheduler(scheduler_config)

    if not scheduler.initialize():
        logger.error("Failed to initialize scheduler")
        return 1

    try:
        # Run comparison via scheduler
        result = scheduler.run_compare(
            kernel_configs=kernel_configs,
            baseline_index=args.baseline,
            enable_default_compile=compile_settings["enable_default_compile"],
            check_performance=not args.no_perf,
            timeout_seconds=perf_settings["timeout_seconds"],
            profiler_args=perf_settings["profiler_args"],
            rocprof_config=perf_settings["rocprof_config"],
            ncu_config=perf_settings["ncu_config"],
            compare_config=compare_settings,
        )

        # Print results
        if result.success and result.results:
            comparison = result.results

            print(f"\n{'=' * 60}")
            print("COMPARISON RESULTS")
            print(f"{'=' * 60}")

            if isinstance(comparison, dict):
                print(comparison.get("summary", "No summary available"))
            elif hasattr(comparison, "summary"):
                print(comparison.summary)

            print(f"{'=' * 60}\n")

            # Save results
            _save_comparison(comparison, args.output_dir)
        else:
            logger.error(f"Comparison failed: {result.errors}")
            return 1

        return 0

    finally:
        scheduler.shutdown()


def _dict_to_eval_state(state_dict: Dict[str, Any]) -> EvaluationState:
    """Convert a dictionary back to EvaluationState."""
    state = EvaluationState()

    if "compiling_state" in state_dict:
        state.compiling_state = BaseKind[state_dict["compiling_state"]]
    if "correctness_state" in state_dict:
        state.correctness_state = BaseKind[state_dict["correctness_state"]]
    if "performance_state" in state_dict:
        state.performance_state = BaseKind[state_dict["performance_state"]]
    if "score" in state_dict:
        state.score = state_dict["score"]
    if "errors" in state_dict:
        state.errors = state_dict["errors"]
    if "extra" in state_dict:
        state.extra = state_dict["extra"]

    return state


def _print_result(kernel_cfg: KernelEvalConfig, result: EvaluationState) -> None:
    """Print evaluation result."""
    print(f"\n{'=' * 60}")
    print(f"Kernel: {kernel_cfg.kernel_id}")
    print(f"Type: {kernel_cfg.kernel_type.name}")

    if isinstance(result, dict):
        print(f"Compiling: {result.get('compiling_state', 'UNKNOWN')}")
        print(f"Correctness: {result.get('correctness_state', 'UNKNOWN')}")
        print(f"Performance: {result.get('performance_state', 'UNKNOWN')}")
        print(f"Score: {result.get('score', 0.0):.2f}")
        errors = result.get("errors", [])
    else:
        print(f"Compiling: {result.compiling_state.name}")
        print(f"Correctness: {result.correctness_state.name}")
        print(f"Performance: {result.performance_state.name}")
        print(f"Score: {result.score:.2f}")
        errors = result.errors

    if errors:
        print("Errors:")
        for err in errors:
            print(f"  - {err}")
    print(f"{'=' * 60}")


def _save_results(results: List, output_dir: Path, mode: str) -> None:
    """Save results to file."""
    import json
    from datetime import datetime

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{mode}_results_{timestamp}.json"

    serialized_results: List[Any] = []
    for r in results:
        if isinstance(r, dict):
            serialized_results.append(r)
        elif hasattr(r, "to_dict"):
            serialized_results.append(r.to_dict())
        else:
            serialized_results.append(str(r))

    with open(output_file, "w") as f:
        json.dump(
            {"mode": mode, "timestamp": timestamp, "results": serialized_results},
            f,
            indent=2,
        )

    logger.info(f"Results saved to {output_file}")


def _save_comparison(comparison: Any, output_dir: Path) -> None:
    """Save comparison to file."""
    import json
    from datetime import datetime

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"compare_results_{timestamp}.json"

    # Extract comparison data
    if isinstance(comparison, dict):
        comparison_data = comparison
    elif hasattr(comparison, "to_dict"):
        comparison_data = comparison.to_dict()
    else:
        comparison_data = {"result": str(comparison)}

    # Use same format as analyze: mode, timestamp, results
    with open(output_file, "w") as f:
        json.dump(
            {
                "mode": "compare",
                "timestamp": timestamp,
                "results": {
                    "kernel_results": comparison_data.get("kernel_results", []),
                    "comparison_metrics": comparison_data.get("comparison_metrics", {}),
                    "winner": comparison_data.get("winner"),
                    "rankings": comparison_data.get("rankings", []),
                    "summary": comparison_data.get("summary", ""),
                },
            },
            f,
            indent=2,
        )

    logger.info(f"Comparison saved to {output_file}")


def load_benchmark_config(benchmark_config_path: Path) -> Dict[str, Any]:
    """
    Load benchmark configuration from YAML file.

    Returns:
        Dictionary with benchmark configuration
    """
    data = load_yaml(benchmark_config_path)
    return data.get("benchmark", {})


def run_benchmark(args, config: Dict[str, Any]) -> int:
    """Run benchmark mode."""
    from .modes.benchmark import BenchmarkMode, BenchmarkConfig
    
    # Build benchmark config
    benchmark_cfg = {}
    
    if args.benchmark_config:
        # Load from config file
        benchmark_cfg = load_benchmark_config(args.benchmark_config)
        if not benchmark_cfg:
            logger.error(f"No benchmark config found in {args.benchmark_config}")
            return 1
    elif args.framework and args.model:
        # Build from CLI arguments
        benchmark_cfg = {
            "framework": args.framework,
            "model": args.model,
            "precision": args.precision,
            "params": {
                "TP": args.tp,
                "CONC": args.concurrency,
                "ISL": args.input_len,
                "OSL": args.output_len,
                "RANDOM_RANGE_RATIO": 0.5,
            },
            "profiler": {
                "torch_profiler": {
                    "enabled": args.torch_profiler,
                },
                "system_profiler": {
                    "enabled": args.system_profiler,
                },
            },
            "docker_image": args.docker_image,
            "inferencemax_path": args.inferencemax_path,
            "benchmark_script": args.benchmark_script,
            "timeout_seconds": args.timeout,
        }
    else:
        logger.error("Benchmark mode requires either --benchmark-config or (framework and --model)")
        print("Error: Specify --benchmark-config or provide framework and --model")
        print("Example: python -m Magpie benchmark sglang --model meta-llama/Llama-2-7b-hf")
        return 1
    
    # Get benchmark settings from framework config
    bench_settings = config.get("benchmark", {})
    
    # Merge with framework config defaults (auto-clone handled in BenchmarkMode)
    if "inferencemax_path" not in benchmark_cfg or not benchmark_cfg["inferencemax_path"]:
        benchmark_cfg["inferencemax_path"] = bench_settings.get("inferencemax_path", "")
    
    # Create benchmark config object
    try:
        benchmark_config = BenchmarkConfig.from_dict(benchmark_cfg)
    except Exception as e:
        logger.error(f"Invalid benchmark configuration: {e}")
        return 1
    
    # Run benchmark directly (benchmark mode handles its own Docker execution)
    logger.info(f"Starting benchmark: {benchmark_config.framework} / {benchmark_config.model}")
    
    try:
        benchmarker = BenchmarkMode(
            config=benchmark_config,
            output_dir=str(args.output_dir),
        )
        result = benchmarker.run()
        
        # Print results
        print(result.get_summary())
        
        if result.success:
            logger.info(f"Benchmark completed successfully")
            logger.info(f"Results saved to: {result.workspace_dir}")
            return 0
        else:
            logger.error(f"Benchmark failed: {result.errors}")
            return 1
            
    except Exception as e:
        logger.exception(f"Benchmark failed with error: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Magpie: GPU Kernel Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Default config: use config.yaml relative to Magpie package
    default_config = Path(__file__).parent / "config.yaml"

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=default_config,
        help="Framework configuration file",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--gpu-info", action="store_true", help="Show detected GPU info and exit"
    )
    parser.add_argument(
        "--environment",
        "-e",
        type=str,
        choices=["local", "container"],
        help="Execution environment (default: local)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, help="Number of concurrent workers"
    )
    parser.add_argument(
        "--docker-image", type=str, help="Docker image for container environment"
    )

    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")

    # Analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze kernel(s) - requires testcase"
    )
    analyze_parser.add_argument(
        "kernels", type=Path, nargs="*", help="Kernel file(s) to analyze"
    )
    analyze_parser.add_argument(
        "--kernel-config", "-k", type=Path, help="Kernel configuration file"
    )
    analyze_parser.add_argument("--testcase", "-t", type=str, help="Testcase command")
    analyze_parser.add_argument(
        "--type",
        type=str,
        default="hip",
        choices=["hip", "cuda", "pytorch"],
        help="Kernel type",
    )
    analyze_parser.add_argument(
        "--compile-cmd", type=str, help="Custom compile command"
    )
    analyze_parser.add_argument(
        "--no-perf", action="store_true", help="Skip performance profiling"
    )
    analyze_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./results"),
        help="Output directory",
    )

    # Compare subcommand
    compare_parser = subparsers.add_parser("compare", help="Compare multiple kernels")
    compare_parser.add_argument(
        "kernels", type=Path, nargs="*", help="Kernel files to compare"
    )
    compare_parser.add_argument(
        "--kernel-config", "-k", type=Path, help="Kernel configuration file"
    )
    compare_parser.add_argument(
        "--testcase", "-t", type=str, help="Testcase command (optional)"
    )
    compare_parser.add_argument(
        "--type",
        type=str,
        default="hip",
        choices=["hip", "cuda", "pytorch"],
        help="Kernel type",
    )
    compare_parser.add_argument(
        "--baseline", type=int, default=0, help="Baseline kernel index"
    )
    compare_parser.add_argument(
        "--no-perf", action="store_true", help="Skip performance profiling"
    )
    compare_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./results"),
        help="Output directory",
    )

    # Benchmark subcommand
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run framework benchmark (vLLM/SGLang)"
    )
    benchmark_parser.add_argument(
        "framework",
        type=str,
        nargs="?",
        choices=["vllm", "sglang"],
        help="Framework to benchmark",
    )
    benchmark_parser.add_argument(
        "--benchmark-config", "-b", type=Path, help="Benchmark configuration file"
    )
    benchmark_parser.add_argument(
        "--model", "-m", type=str, help="Model name or path"
    )
    benchmark_parser.add_argument(
        "--precision", "-p", type=str, default="fp8",
        choices=["fp8", "fp16", "bf16", "fp4"],
        help="Model precision (default: fp8)"
    )
    benchmark_parser.add_argument(
        "--tp", type=int, default=1, help="Tensor parallel size"
    )
    benchmark_parser.add_argument(
        "--concurrency", type=int, default=32, help="Request concurrency"
    )
    benchmark_parser.add_argument(
        "--input-len", type=int, default=1024, help="Input sequence length"
    )
    benchmark_parser.add_argument(
        "--output-len", type=int, default=512, help="Output sequence length"
    )
    benchmark_parser.add_argument(
        "--torch-profiler", action="store_true",
        help="Enable torch profiler"
    )
    benchmark_parser.add_argument(
        "--system-profiler", action="store_true",
        help="Enable system profiler (rocprof/ncu)"
    )
    benchmark_parser.add_argument(
        "--docker-image", type=str, help="Override Docker image"
    )
    benchmark_parser.add_argument(
        "--inferencemax-path", type=str,
        default="",
        help="Path to InferenceMAX installation (auto-cloned if not specified)"
    )
    benchmark_parser.add_argument(
        "--benchmark-script", type=str,
        help="InferenceMAX benchmark script name"
    )
    benchmark_parser.add_argument(
        "--timeout", type=int, default=3600, help="Benchmark timeout in seconds"
    )
    benchmark_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./results"),
        help="Output directory",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle --gpu-info
    if args.gpu_info:
        info = get_gpu_info()
        print("GPU Information:")
        print(f"  Vendor: {info['vendor']}")
        print(f"  Architecture: {info['architecture']}")
        if info["detected"]:
            print(f"  Compiler: {info.get('compiler', 'N/A')}")
            print(f"  Profiler: {info.get('profiler', 'N/A')}")
        else:
            print("  (No GPU detected)")
        return 0

    if not args.mode:
        parser.print_help()
        return 0

    # Load framework config
    config = load_yaml(args.config)
    setup_logging(config, args.verbose)

    if args.mode == "analyze":
        return run_analyze(args, config)
    elif args.mode == "compare":
        return run_compare(args, config)
    elif args.mode == "benchmark":
        return run_benchmark(args, config)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

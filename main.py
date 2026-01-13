#!/usr/bin/env python3
"""
AIG-Kernel-Eval: GPU Kernel Evaluation Framework

Usage:
    # Analyze with CLI arguments
    python main.py analyze kernel.hip -t "./test.sh"
    
    # Analyze with kernel config file
    python main.py analyze --kernel-config my_kernel.yaml
    
    # Compare kernels
    python main.py compare kernel1.hip kernel2.hip
    
    # Compare with kernel config file
    python main.py compare --kernel-config kernels.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.config import KernelType, KernelEvalConfig
from src.eval import EvaluationState, BaseKind
from src.modes import AnalyzeMode, CompareMode
from src.modes.analyze_eval.analyzer import AnalyzeConfig
from src.modes.compare_eval.comparator import CompareConfig
from src.utils import detect_gpu, get_gpu_info

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
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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


def _parse_kernel_entry(entry: Dict[str, Any]) -> Optional[KernelEvalConfig]:
    """Parse a single kernel entry from config."""
    if not entry:
        return None
    
    kernel_type = parse_kernel_type(entry.get("type", "hip"))
    
    # Parse testcase command(s)
    testcase_cmd = _parse_command_list(entry.get("testcase_command"))
    
    # Parse compile command(s)
    compile_cmd = _parse_command_list(entry.get("compile_command"))
    
    # Parse prof command(s)
    prof_cmd = _parse_command_list(entry.get("prof_command"))
    
    return KernelEvalConfig(
        kernel_id=entry.get("id", "kernel"),
        kernel_type=kernel_type,
        source_file_path=entry.get("source_files", []),
        working_dir=entry.get("working_dir"),
        env=entry.get("env"),
        testcase_command=testcase_cmd,
        compiling_command=compile_cmd,
        prof_command=prof_cmd,
        get_inputs_func=entry.get("get_inputs_func", "get_inputs"),
        get_init_inputs_func=entry.get("get_init_inputs_func", "get_init_inputs"),
    )


def _get_performance_config(config: Dict[str, Any], kernel_type: KernelType) -> Dict[str, Any]:
    """
    Get performance configuration from framework config.
    
    Args:
        config: Framework config dict
        kernel_type: Kernel type to determine which profiler args to use
        
    Returns:
        Dict with timeout_seconds and profiler_args
    """
    perf_cfg = config.get("performance", {})
    
    # Get timeout
    timeout = perf_cfg.get("timeout_seconds", 60.0)
    
    # Get profiler args based on kernel type
    profiler_args = []
    if kernel_type == KernelType.HIP:
        rocprof_cfg = perf_cfg.get("rocprof_compute", {})
        profiler_args = rocprof_cfg.get("args", [])
    elif kernel_type == KernelType.CUDA:
        ncu_cfg = perf_cfg.get("ncu", {})
        profiler_args = ncu_cfg.get("args", [])
    
    return {
        "timeout_seconds": timeout,
        "profiler_args": profiler_args,
    }


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
            
            kernel_configs.append(KernelEvalConfig(
                kernel_id=path.stem,
                kernel_type=kernel_type,
                source_file_path=[str(path)],
                testcase_command=args.testcase.split(),
                compiling_command=args.compile_cmd.split() if args.compile_cmd else None,
                working_dir=str(path.parent),
            ))
    else:
        logger.error("No kernels specified")
        return 1
    
    # Get kernel type for configuration
    kernel_type = kernel_configs[0].kernel_type if kernel_configs else KernelType.HIP
    
    # Get performance config from framework config
    # Priority: args > kernel config > framework config
    perf_settings = _get_performance_config(config, kernel_type)
    
    # Create analyzer with merged configuration
    analyze_config = AnalyzeConfig(
        kernel_type=kernel_type,
        check_performance=not args.no_perf,  # args override
        timeout_seconds=perf_settings["timeout_seconds"],
        profiler_args=perf_settings["profiler_args"],
    )
    analyzer = AnalyzeMode(analyze_config)
    
    # Run analysis
    results = []
    for kernel_cfg in kernel_configs:
        result = analyzer.analyze(kernel_cfg)
        results.append((kernel_cfg, result))
        _print_result(kernel_cfg, result)
    
    # Save results
    if results:
        _save_results([r for _, r in results], args.output_dir, "analyze")
    
    failed = sum(1 for _, r in results if r.correctness_state != BaseKind.SUCCESS)
    return 1 if failed > 0 else 0


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
            
            kernel_configs.append(KernelEvalConfig(
                kernel_id=f"kernel_{i}_{path.stem}",
                kernel_type=kernel_type,
                source_file_path=[str(path)],
                testcase_command=args.testcase.split() if args.testcase else None,
                working_dir=str(path.parent),
            ))
    
    if len(kernel_configs) < 2:
        logger.error("Compare mode requires at least 2 kernels")
        return 1
    
    # Get kernel type for configuration
    kernel_type = kernel_configs[0].kernel_type if kernel_configs else KernelType.HIP
    
    # Get performance config from framework config
    # Priority: args > kernel config > framework config
    perf_settings = _get_performance_config(config, kernel_type)
    
    # Create comparator with merged configuration
    compare_config = CompareConfig(
        baseline_index=args.baseline,
        check_performance=not args.no_perf,  # args override
        timeout_seconds=perf_settings["timeout_seconds"],
        profiler_args=perf_settings["profiler_args"],
    )
    comparator = CompareMode(compare_config)
    
    # Run comparison
    comparison = comparator.compare(kernel_configs)
    
    # Print results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(comparison.summary)
    print(f"{'='*60}\n")
    
    # Save results
    _save_comparison(comparison, args.output_dir)
    
    return 0


def _print_result(kernel_cfg: KernelEvalConfig, result: EvaluationState) -> None:
    """Print evaluation result."""
    print(f"\n{'='*60}")
    print(f"Kernel: {kernel_cfg.kernel_id}")
    print(f"Type: {kernel_cfg.kernel_type.name}")
    print(f"Compiling: {result.compiling_state.name}")
    print(f"Correctness: {result.correctness_state.name}")
    print(f"Performance: {result.performance_state.name}")
    print(f"Score: {result.score:.2f}")
    if result.errors:
        print("Errors:")
        for err in result.errors:
            print(f"  - {err}")
    print(f"{'='*60}")


def _save_results(results: List[EvaluationState], output_dir: Path, mode: str) -> None:
    """Save results to file."""
    import json
    from datetime import datetime
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{mode}_results_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "mode": mode,
            "timestamp": timestamp,
            "results": [r.to_dict() for r in results]
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def _save_comparison(comparison: Any, output_dir: Path) -> None:
    """Save comparison to file."""
    import json
    from datetime import datetime
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"comparison_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(comparison.to_dict(), f, indent=2)
    
    logger.info(f"Comparison saved to {output_file}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="AIG-Kernel-Eval: GPU Kernel Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("config.yaml"),
        help="Framework configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--gpu-info",
        action="store_true",
        help="Show detected GPU info and exit"
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")
    
    # Analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze kernel(s) - requires testcase"
    )
    analyze_parser.add_argument(
        "kernels",
        type=Path,
        nargs="*",
        help="Kernel file(s) to analyze"
    )
    analyze_parser.add_argument(
        "--kernel-config", "-k",
        type=Path,
        help="Kernel configuration file"
    )
    analyze_parser.add_argument(
        "--testcase", "-t",
        type=str,
        help="Testcase command"
    )
    analyze_parser.add_argument(
        "--type",
        type=str,
        default="hip",
        choices=["hip", "cuda", "pytorch"],
        help="Kernel type"
    )
    analyze_parser.add_argument(
        "--compile-cmd",
        type=str,
        help="Custom compile command"
    )
    analyze_parser.add_argument(
        "--no-perf",
        action="store_true",
        help="Skip performance profiling"
    )
    analyze_parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./results"),
        help="Output directory"
    )
    
    # Compare subcommand
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple kernels"
    )
    compare_parser.add_argument(
        "kernels",
        type=Path,
        nargs="*",
        help="Kernel files to compare"
    )
    compare_parser.add_argument(
        "--kernel-config", "-k",
        type=Path,
        help="Kernel configuration file"
    )
    compare_parser.add_argument(
        "--testcase", "-t",
        type=str,
        help="Testcase command (optional)"
    )
    compare_parser.add_argument(
        "--type",
        type=str,
        default="hip",
        choices=["hip", "cuda", "pytorch"],
        help="Kernel type"
    )
    compare_parser.add_argument(
        "--baseline",
        type=int,
        default=0,
        help="Baseline kernel index"
    )
    compare_parser.add_argument(
        "--no-perf",
        action="store_true",
        help="Skip performance profiling"
    )
    compare_parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./results"),
        help="Output directory"
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
        if info['detected']:
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
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Remote entrypoint executed on Ray worker nodes.

This script is invoked by the Ray Job Submission API as::

    python -m Magpie.remote.entrypoint --config /path/to/config.json \
                                       --result-path /path/to/result.json

It reads the task configuration from shared NFS, executes the appropriate
mode (benchmark / analyze / compare), and writes the result JSON back to
shared NFS for the MCP server to retrieve.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("magpie.remote.entrypoint")


def _run_benchmark(mode_config: dict, ray_config: dict, task_id: str) -> dict:
    """Execute benchmark mode on the worker."""
    from Magpie.modes.benchmark import BenchmarkMode, BenchmarkConfig

    bench_cfg_dict = mode_config.get("benchmark_config", {})

    # On the Ray worker the benchmark runs locally (the worker IS the GPU node)
    # unless the user explicitly wants Docker-in-Docker.
    if bench_cfg_dict.get("run_mode") == "ray":
        bench_cfg_dict["run_mode"] = "local"

    # Override paths to point at shared storage
    shared = ray_config.get("shared_storage_path", "/shared_nfs/magpie")
    if not bench_cfg_dict.get("inferencemax_path"):
        bench_cfg_dict["inferencemax_path"] = f"{shared}/InferenceMAX"
    if not bench_cfg_dict.get("hf_cache_path"):
        bench_cfg_dict["hf_cache_path"] = f"{shared}/hf_cache"

    # Build result workspace inside the job directory
    job_dir = str(Path(ray_config.get("results_dir", f"{shared}/results")) / task_id)

    benchmark_config = BenchmarkConfig.from_dict(bench_cfg_dict)
    benchmarker = BenchmarkMode(benchmark_config, output_dir=job_dir)
    result = benchmarker.run(task_id=task_id)
    return result.to_dict()


def _run_analyze(mode_config: dict, ray_config: dict, task_id: str) -> dict:
    """Execute analyze mode on the worker."""
    from Magpie.modes import AnalyzeMode
    from Magpie.modes.analyze_eval.analyzer import AnalyzeConfig
    from Magpie.config import KernelEvalConfig, KernelType

    kernel_configs = []
    for cfg_data in mode_config.get("kernel_configs", []):
        if isinstance(cfg_data, dict):
            kernel_configs.append(KernelEvalConfig.from_dict(cfg_data))

    kernel_type = kernel_configs[0].kernel_type if kernel_configs else KernelType.HIP

    analyze_config = AnalyzeConfig(
        kernel_type=kernel_type,
        enable_default_compile=mode_config.get("enable_default_compile", False),
        check_performance=mode_config.get("check_performance", True),
        timeout_seconds=mode_config.get("timeout_seconds", 300.0),
        profiler_args=mode_config.get("profiler_args", []),
        rocprof_config=mode_config.get("rocprof_config", {}),
        ncu_config=mode_config.get("ncu_config", {}),
        metrix_config=mode_config.get("metrix_config", {}),
        gpu_arch=mode_config.get("gpu_arch"),
    )
    analyzer = AnalyzeMode(analyze_config)

    results = []
    for kcfg in kernel_configs:
        r = analyzer.analyze(kcfg)
        results.append(r.to_dict() if hasattr(r, "to_dict") else r)

    return {"task_id": task_id, "status": "completed", "results": results}


def _run_compare(mode_config: dict, ray_config: dict, task_id: str) -> dict:
    """Execute compare mode on the worker."""
    from Magpie.modes import CompareMode
    from Magpie.modes.compare_eval.comparator import CompareConfig
    from Magpie.config import KernelEvalConfig

    kernel_configs = []
    for cfg_data in mode_config.get("kernel_configs", []):
        if isinstance(cfg_data, dict):
            kernel_configs.append(KernelEvalConfig.from_dict(cfg_data))

    compare_cfg = mode_config.get("compare_config", {})
    compare_config = CompareConfig(
        baseline_index=mode_config.get("baseline_index", 0),
        enable_default_compile=mode_config.get("enable_default_compile", False),
        check_performance=mode_config.get("check_performance", True),
        timeout_seconds=mode_config.get("timeout_seconds", 300.0),
        profiler_args=mode_config.get("profiler_args", []),
        rocprof_config=mode_config.get("rocprof_config", {}),
        ncu_config=mode_config.get("ncu_config", {}),
        metrix_config=mode_config.get("metrix_config", {}),
        gpu_arch=mode_config.get("gpu_arch"),
        winner_strategy=compare_cfg.get("winner_strategy", "perf_score"),
        perf_weights_rocprof=compare_cfg.get("perf_weights_rocprof", {}),
        perf_weights_ncu=compare_cfg.get("perf_weights_ncu", {}),
        perf_weights_metrix=compare_cfg.get("perf_weights_metrix", {}),
    )
    comparator = CompareMode(compare_config)
    result = comparator.compare(kernel_configs)
    result_dict = result.to_dict() if hasattr(result, "to_dict") else result
    return {"task_id": task_id, "status": "completed", "results": result_dict}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Magpie remote entrypoint for Ray workers"
    )
    parser.add_argument(
        "--config", required=True, help="Path to the task config JSON on shared storage"
    )
    parser.add_argument(
        "--result-path", required=True, help="Path to write the result JSON"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    result_path = Path(args.result_path)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = json.loads(config_path.read_text())
    task_id = config.get("task_id", "unknown")
    mode_type = config.get("mode_type", "benchmark")
    mode_config = config.get("mode_config", {})
    ray_config = config.get("ray_config", {})

    logger.info(f"Starting remote execution: task={task_id} mode={mode_type}")
    start_time = time.time()

    try:
        # Set HF environment variables for model access
        shared = ray_config.get("shared_storage_path", "/shared_nfs/magpie")
        os.environ.setdefault("HF_HOME", f"{shared}/hf_cache")
        os.environ.setdefault("TRANSFORMERS_CACHE", f"{shared}/hf_cache/hub")

        if mode_type == "benchmark":
            result = _run_benchmark(mode_config, ray_config, task_id)
        elif mode_type == "analyze":
            result = _run_analyze(mode_config, ray_config, task_id)
        elif mode_type == "compare":
            result = _run_compare(mode_config, ray_config, task_id)
        else:
            result = {"error": f"Unknown mode_type: {mode_type}"}

        result["execution_time"] = time.time() - start_time

    except Exception as e:
        logger.exception(f"Remote execution failed: {e}")
        result = {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "execution_time": time.time() - start_time,
        }

    # Write result to shared NFS
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    logger.info(f"Result written to {result_path}")


if __name__ == "__main__":
    main()

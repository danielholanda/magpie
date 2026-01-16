"""
Performance evaluation module.

This module handles performance measurement of GPU kernels using
different backends based on kernel type:
- HIP kernels: rocprof-compute (two-stage: profile + analyze)
- CUDA kernels: ncu (NVIDIA Nsight Compute)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..config import (
    KernelType,
    PipelineConfig,
    KernelEvalConfig,
    PerformanceConfig,
    PerfBackend,
)
from ..config.performance import ROCPROF_KEY_METRICS, RocprofComputeConfig
from ..utils import get_updated_env

if TYPE_CHECKING:
    from .evaluator import EvaluationState

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """A single performance metric result."""
    name: str
    value: Optional[float]
    unit: str = ""
    peak: Optional[float] = None
    pct_of_peak: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
        }
        if self.peak is not None:
            d["peak"] = self.peak
        if self.pct_of_peak is not None:
            d["pct_of_peak"] = self.pct_of_peak
        return d


@dataclass
class KernelMetrics:
    """Performance metrics for a single kernel dispatch."""
    kernel_name: str
    dispatch_id: int
    metrics: List[MetricResult] = field(default_factory=list)
    duration_ns: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without raw metrics to reduce output size)."""
        return {
            "kernel_name": self.kernel_name,
            "dispatch_id": self.dispatch_id,
            "duration_ns": self.duration_ns,
        }


@dataclass
class PerformanceResult:
    """
    Result of performance evaluation.
    """
    success: bool
    metrics: List[MetricResult] = field(default_factory=list)
    kernel_metrics: List[KernelMetrics] = field(default_factory=list)
    raw_output: Optional[str] = None
    errors: Optional[str] = None
    command: Optional[str] = None  # The command that was executed (for debugging)
    workload_dir: Optional[str] = None  # rocprof-compute workload directory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "success": self.success,
            "errors": self.errors,
            "workload_dir": self.workload_dir,
            # Summary metrics in a flat format for easy access
            "summary": self.get_summary_metrics(),
            # Aggregated kernel statistics (unique kernels with dispatch count and avg duration)
            "kernels": self.get_kernel_summary(),
        }
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get a summary of key metrics across all kernels."""
        summary = {}
        for m in self.metrics:
            summary[m.name] = {
                "value": m.value,
                "unit": m.unit,
            }
            if m.peak is not None:
                summary[m.name]["peak"] = m.peak
            if m.pct_of_peak is not None:
                summary[m.name]["pct_of_peak"] = m.pct_of_peak
        return summary
    
    def get_kernel_summary(self) -> List[Dict[str, Any]]:
        """
        Get aggregated kernel statistics.
        
        Groups by kernel name, counts dispatches, and calculates avg/min/max duration.
        Filters out HIP runtime kernels (e.g., __amd_rocclr_*).
        """
        from collections import defaultdict
        
        # Group by kernel name
        kernel_stats: Dict[str, List[float]] = defaultdict(list)
        for km in self.kernel_metrics:
            # Filter out HIP runtime/internal kernels
            if km.kernel_name.startswith("__amd_rocclr_") or km.kernel_name.startswith("__hip_"):
                continue
            if km.duration_ns is not None:
                kernel_stats[km.kernel_name].append(km.duration_ns)
        
        # Build summary
        result = []
        for kernel_name, durations in kernel_stats.items():
            if durations:
                result.append({
                    "kernel_name": kernel_name,
                    "dispatch_count": len(durations),
                    "duration_ns": {
                        "avg": sum(durations) / len(durations),
                        "min": min(durations),
                        "max": max(durations),
                        "total": sum(durations),
                    }
                })
        
        # Sort by total duration (most time-consuming kernels first)
        result.sort(key=lambda x: x["duration_ns"]["total"], reverse=True)
        
        return result


class Performance:
    """
    Performance evaluation handler.
    
    Uses different profiling backends based on kernel type:
    - HIP: rocprof-compute (two-stage workflow)
    - CUDA: ncu (NVIDIA Nsight Compute)
    """
    
    def __init__(self, pipeline_cfg: PipelineConfig) -> None:
        """
        Initialize performance evaluator.
        """
        self.pipeline_cfg = pipeline_cfg
        self.perf_cfg = pipeline_cfg.performance_config
        
    def run(
        self, 
        eval_state: Any, 
        kernel_cfg: KernelEvalConfig
    ) -> Optional[PerformanceResult]:
        """
        Run performance evaluation.
        
        Logic:
        1. --no-perf (enabled=False) → Skip (return None)
        2. Has prof_command → Run custom profiler
        3. No prof_command but has testcase_command → Use built-in profiler on testcase
        4. No prof_command and no testcase_command → Skip (return None)
        
        Args:
            eval_state: Current evaluation state
            kernel_cfg: Kernel configuration
            
        Returns:
            PerformanceResult with profiling metrics, or None if skipped
        """
        # 1. If profiling is disabled (--no-perf), skip
        if not self.perf_cfg.enabled:
            return None
        
        # 2. If custom prof_command is provided, use it
        if kernel_cfg.has_prof_command():
            return self._run_custom_profiler(kernel_cfg)
        
        # 3. No prof_command - check if we have testcase_command for built-in profiler
        if not kernel_cfg.has_testcase():
            # 4. No testcase_command either, skip profiling
            return None
        
        # Use built-in profiler on testcase command
        backend = self.perf_cfg.get_backend()
        
        try:
            if backend == PerfBackend.ROCPROF_COMPUTE:
                return self._run_rocprof_compute_on_testcase(kernel_cfg)
            elif backend == PerfBackend.NCU:
                return self._run_ncu_on_testcase(kernel_cfg)
            else:
                # No profiling backend configured, skip
                return None
                
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return PerformanceResult(success=False, errors=str(e))

    def _run_custom_profiler(
        self,
        kernel_cfg: KernelEvalConfig
    ) -> PerformanceResult:
        """
        Run profiling using custom prof_command(s).
        
        This replaces the built-in profiler backend when prof_command is specified.
        Supports multiple commands executed in order.
        """
        commands = kernel_cfg.get_prof_commands()
        working_dir = kernel_cfg.working_dir
        env = get_updated_env(kernel_cfg.env)
        
        all_outputs = []
        
        try:
            for cmd_idx, cmd in enumerate(commands):
                logger.info(f"Running profiler command {cmd_idx+1}/{len(commands)}: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=working_dir,
                    timeout=self.perf_cfg.timeout_seconds
                )
                
                all_outputs.append(result.stdout)
                
                if result.returncode != 0:
                    cmd_info = f"command {cmd_idx+1}/{len(commands)}" if len(commands) > 1 else "command"
                    return PerformanceResult(
                        success=False,
                        raw_output="\n".join(all_outputs),
                        errors=f"Profiler {cmd_info} failed: {result.stderr or result.stdout}",
                        command=" ".join(cmd)
                    )
            
            # All commands succeeded
            return PerformanceResult(
                success=True,
                metrics=[MetricResult(
                    name="custom_profiling_complete",
                    value=1.0,
                    unit="bool"
                )],
                raw_output="\n".join(all_outputs)
            )
            
        except subprocess.TimeoutExpired:
            return PerformanceResult(
                success=False,
                errors=f"Custom profiler timed out after {self.perf_cfg.timeout_seconds}s"
            )
        except Exception as e:
            logger.error(f"Custom profiler failed: {e}")
            return PerformanceResult(success=False, errors=str(e))

    def _run_rocprof_compute_on_testcase(
        self, 
        kernel_cfg: KernelEvalConfig
    ) -> PerformanceResult:
        """
        Run profiling using rocprof-compute on the testcase command.
        
        rocprof-compute is a two-stage profiler:
        1. Profile stage: Collect hardware counters
           `rocprof-compute profile -b <blocks> -n <name> -o <dir> -- <command>`
        2. Analyze stage: Process data and generate metrics
           `rocprof-compute analyze -b <blocks> <workload_dir>`
        """
        # Check if rocprof-compute is available
        rocprof_path = shutil.which("rocprof-compute")
        if rocprof_path is None:
            return PerformanceResult(
                success=False,
                errors="rocprof-compute not found. Please install ROCm tools."
            )
        
        # Get testcase command(s)
        testcase_commands = kernel_cfg.get_testcase_commands()
        if not testcase_commands:
            return PerformanceResult(
                success=False,
                errors="No testcase command available for profiling"
            )
        
        working_dir = kernel_cfg.working_dir
        env = get_updated_env(kernel_cfg.env)
        rocprof_cfg = self.perf_cfg.rocprof_config
        
        # Create workload directory
        timestamp = int(time.time())
        kernel_name = kernel_cfg.kernel_id or "kernel"
        # Sanitize kernel name for directory
        safe_kernel_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in kernel_name)
        workload_name = f"{safe_kernel_name}_{timestamp}"
        
        # Use working directory as base for workload output
        workload_base_dir = Path(working_dir) / rocprof_cfg.workload_dir
        workload_base_dir.mkdir(parents=True, exist_ok=True)
        workload_path = workload_base_dir / workload_name
        
        all_outputs = []
        all_errors = []
        
        try:
            # ===== Stage 1: Profile =====
            for cmd_idx, testcase_cmd in enumerate(testcase_commands):
                # Build rocprof-compute profile command
                profile_args = rocprof_cfg.get_profile_args(
                    workload_name=f"{workload_name}_run{cmd_idx}",
                    output_dir=str(workload_base_dir)
                )
                
                cmd = [
                    "rocprof-compute",
                    "profile",
                    *profile_args,
                    "--",
                    *testcase_cmd,
                ]
                
                cmd_str = " ".join(cmd)
                logger.info(f"[Profile Stage] Running rocprof-compute on testcase {cmd_idx+1}/{len(testcase_commands)}")
                logger.debug(f"Command: {cmd_str}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=working_dir,
                    timeout=self.perf_cfg.timeout_seconds
                )
                
                all_outputs.append(f"=== Profile Run {cmd_idx+1} ===\n{result.stdout}")
                if result.stderr:
                    all_errors.append(f"=== Profile Run {cmd_idx+1} stderr ===\n{result.stderr}")
                
                if result.returncode != 0:
                    return PerformanceResult(
                        success=False,
                        raw_output="\n".join(all_outputs),
                        errors=f"rocprof-compute profile failed on testcase {cmd_idx+1}: {result.stderr or result.stdout}",
                        command=cmd_str,
                        workload_dir=str(workload_path)
                    )
            
            # Find the workload directory created by profile
            # rocprof-compute creates a directory named like: workload_name/
            actual_workload_dir = None
            for d in workload_base_dir.iterdir():
                if d.is_dir() and workload_name in d.name:
                    actual_workload_dir = d
                    break
            
            if actual_workload_dir is None:
                # Try to find any directory matching the pattern
                for d in workload_base_dir.iterdir():
                    if d.is_dir() and safe_kernel_name in d.name:
                        actual_workload_dir = d
                        break
            
            if actual_workload_dir is None:
                return PerformanceResult(
                    success=False,
                    raw_output="\n".join(all_outputs),
                    errors=f"Could not find workload directory in {workload_base_dir}",
                    workload_dir=str(workload_base_dir)
                )
            
            # rocprof-compute creates GPU-specific subdirectories (e.g., MI300X_A1/)
            # Find the GPU subdirectory containing the profiling data
            gpu_data_dir = None
            for subdir in actual_workload_dir.iterdir():
                if subdir.is_dir() and (subdir / "pmc_perf.csv").exists():
                    gpu_data_dir = subdir
                    break
            
            # Use the GPU data directory for analysis if found, otherwise use workload dir
            analyze_dir = gpu_data_dir if gpu_data_dir else actual_workload_dir
            
            # ===== Stage 2: Analyze =====
            # Create output directory for CSV files
            csv_output_dir = actual_workload_dir / "analysis"
            csv_output_dir.mkdir(parents=True, exist_ok=True)
            
            analyze_args = rocprof_cfg.get_analyze_args(
                str(analyze_dir), 
                output_dir=str(csv_output_dir)
            )
            analyze_cmd = [
                "rocprof-compute",
                "analyze",
                *analyze_args,
            ]
            
            analyze_cmd_str = " ".join(analyze_cmd)
            logger.info(f"[Analyze Stage] Running rocprof-compute analyze")
            logger.debug(f"Command: {analyze_cmd_str}")
            
            analyze_result = subprocess.run(
                analyze_cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=working_dir,
                timeout=self.perf_cfg.timeout_seconds
            )
            
            all_outputs.append(f"=== Analyze ===\n{analyze_result.stdout}")
            if analyze_result.stderr:
                all_errors.append(f"=== Analyze stderr ===\n{analyze_result.stderr}")
            
            if analyze_result.returncode != 0:
                return PerformanceResult(
                    success=False,
                    raw_output="\n".join(all_outputs),
                    errors=f"rocprof-compute analyze failed: {analyze_result.stderr or analyze_result.stdout}",
                    command=analyze_cmd_str,
                    workload_dir=str(actual_workload_dir)
                )
            
            # ===== Parse Results =====
            # Parse from both the GPU data directory (for raw metrics) and analysis output directory
            metrics, kernel_metrics = self._parse_rocprof_workload(
                gpu_data_dir or actual_workload_dir,
                csv_output_dir
            )
            
            return PerformanceResult(
                success=True,
                metrics=metrics if metrics else [MetricResult(
                    name="profiling_complete",
                    value=1.0,
                    unit="bool"
                )],
                kernel_metrics=kernel_metrics,
                raw_output="\n".join(all_outputs),
                workload_dir=str(actual_workload_dir)
            )
            
        except subprocess.TimeoutExpired:
            return PerformanceResult(
                success=False,
                errors=f"Profiling timed out after {self.perf_cfg.timeout_seconds}s",
                workload_dir=str(workload_path) if 'workload_path' in locals() else None
            )
        except Exception as e:
            logger.error(f"rocprof-compute failed: {e}", exc_info=True)
            return PerformanceResult(
                success=False, 
                errors=str(e),
                workload_dir=str(workload_path) if 'workload_path' in locals() else None
            )

    def _parse_rocprof_workload(
        self, 
        gpu_data_dir: Path,
        analysis_output_dir: Optional[Path] = None
    ) -> tuple[List[MetricResult], List[KernelMetrics]]:
        """
        Parse rocprof-compute workload directory and extract metrics.
        
        The GPU data directory typically contains:
        - pmc_perf.csv: Raw performance counters
        - sysinfo.csv: System information
        
        The analysis output directory contains:
        - CSV files generated by 'rocprof-compute analyze --save-dfs'
        
        Args:
            gpu_data_dir: Path to the GPU-specific data directory (e.g., MI300X_A1/)
            analysis_output_dir: Path to analysis output directory (for --save-dfs CSVs)
            
        Returns:
            Tuple of (aggregated metrics, per-kernel metrics)
        """
        metrics = []
        kernel_metrics = []
        
        gpu_data_dir = Path(gpu_data_dir)
        
        # Determine output directory for analysis results
        out_dir = Path(analysis_output_dir) if analysis_output_dir else gpu_data_dir
        
        # Parse raw pmc_perf.csv from GPU data directory
        pmc_perf_csv = gpu_data_dir / "pmc_perf.csv"
        if pmc_perf_csv.exists():
            try:
                raw_metrics = self._parse_pmc_perf_csv(pmc_perf_csv)
                kernel_metrics.extend(raw_metrics)
            except Exception as e:
                logger.warning(f"Failed to parse {pmc_perf_csv}: {e}")
        
        # Parse analysis output CSVs (if --save-dfs was used)
        if out_dir.exists():
            # Parse speed-of-light metrics
            sol_metrics = self._parse_speed_of_light_csv(out_dir)
            metrics.extend(sol_metrics)
            
            # Parse instruction mix metrics
            inst_metrics = self._parse_instruction_mix_csv(out_dir)
            metrics.extend(inst_metrics)
            
            # Parse memory/cache metrics
            mem_metrics = self._parse_memory_metrics_csv(out_dir)
            metrics.extend(mem_metrics)
        
        return metrics, kernel_metrics
    
    def _parse_pmc_perf_csv(self, csv_path: Path) -> List[KernelMetrics]:
        """
        Parse pmc_perf.csv which contains raw hardware counters per kernel dispatch.
        
        This file contains columns like:
        - Kernel_Name, Dispatch_ID, GPU_ID
        - Start_Timestamp, End_Timestamp
        - Various hardware counters (SQ_*, TCC_*, etc.)
        """
        kernel_metrics = []
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    kernel_name = row.get('Kernel_Name', row.get('KernelName', 'unknown'))
                    
                    try:
                        dispatch_id = int(row.get('Dispatch_ID', row.get('DispatchID', 0)))
                    except (ValueError, TypeError):
                        dispatch_id = 0
                    
                    km = KernelMetrics(
                        kernel_name=kernel_name,
                        dispatch_id=dispatch_id,
                        metrics=[]
                    )
                    
                    # Calculate duration from timestamps
                    try:
                        start_ts = float(row.get('Start_Timestamp', 0))
                        end_ts = float(row.get('End_Timestamp', 0))
                        if end_ts > start_ts:
                            km.duration_ns = end_ts - start_ts
                    except (ValueError, TypeError):
                        pass
                    
                    # Extract key hardware counters
                    counter_prefixes = ['SQ_', 'TCC_', 'TCP_', 'TA_', 'TD_', 'SQC_']
                    for key, value in row.items():
                        if any(key.startswith(prefix) for prefix in counter_prefixes):
                            try:
                                float_val = float(value) if value else None
                                if float_val is not None:
                                    km.metrics.append(MetricResult(
                                        name=key,
                                        value=float_val,
                                        unit="count"
                                    ))
                            except (ValueError, TypeError):
                                pass
                    
                    kernel_metrics.append(km)
                    
        except Exception as e:
            logger.warning(f"Failed to parse pmc_perf.csv: {e}")
        
        return kernel_metrics

    def _parse_kernel_top_csv(self, csv_path: Path) -> List[KernelMetrics]:
        """Parse pmc_kernel_top.csv for kernel execution info."""
        kernel_metrics = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kernel_name = row.get('Kernel_Name', row.get('KernelName', 'unknown'))
                dispatch_id = int(row.get('Dispatch_ID', row.get('DispatchID', 0)))
                
                km = KernelMetrics(
                    kernel_name=kernel_name,
                    dispatch_id=dispatch_id,
                    metrics=[]
                )
                
                # Extract duration if available
                for dur_key in ['Duration', 'Duration_ns', 'End_Timestamp', 'duration']:
                    if dur_key in row and row[dur_key]:
                        try:
                            km.duration_ns = float(row[dur_key])
                            break
                        except ValueError:
                            pass
                
                # Extract other numeric metrics
                for key, value in row.items():
                    if key not in ['Kernel_Name', 'KernelName', 'Dispatch_ID', 'DispatchID']:
                        try:
                            float_val = float(value) if value else None
                            if float_val is not None:
                                km.metrics.append(MetricResult(
                                    name=key,
                                    value=float_val,
                                    unit=""
                                ))
                        except (ValueError, TypeError):
                            pass
                
                kernel_metrics.append(km)
        
        return kernel_metrics
    
    def _parse_speed_of_light_csv(self, out_dir: Path) -> List[MetricResult]:
        """Parse speed-of-light metrics from analyze output."""
        metrics = []
        
        # rocprof-compute --save-dfs generates CSVs with pattern: <block_id>.<sub_id>_<Name>.csv
        # Look for system speed-of-light (block 2.1) and other speed-of-light files
        sol_patterns = [
            "2.1_Speed-of-Light.csv",       # System Speed-of-Light
            "11.1_Speed-of-Light.csv",      # Compute Pipeline Speed-of-Light
            "12.1_Speed-of-Light.csv",      # LDS Speed-of-Light
            "14.1_Speed-of-Light.csv",      # Scalar L1D Speed-of-Light
            "16.1_Speed-of-Light.csv",      # Vector L1D Speed-of-Light
            "17.1_Speed-of-Light.csv",      # L2 Cache Speed-of-Light
        ]
        
        for pattern in sol_patterns:
            sol_csv = out_dir / pattern
            if sol_csv.exists():
                metrics.extend(self._parse_metric_table_csv(sol_csv))
        
        # Also try glob for any speed-of-light files we might have missed
        for csv_file in out_dir.glob("*Speed-of-Light*.csv"):
            if csv_file.name not in sol_patterns:
                metrics.extend(self._parse_metric_table_csv(csv_file))
        
        return metrics
    
    def _parse_instruction_mix_csv(self, out_dir: Path) -> List[MetricResult]:
        """Parse instruction mix metrics from analyze output."""
        metrics = []
        
        # rocprof-compute --save-dfs generates CSVs with pattern: <block_id>.<sub_id>_<Name>.csv
        # Block 10 = Compute Units - Instruction Mix
        inst_patterns = [
            "10.1_Overall_Instruction_Mix.csv",
            "10.2_VALU_Arithmetic_Instr_Mix.csv",
            "10.3_VMEM_Instr_Mix.csv",
            "10.4_MFMA_Arithmetic_Instr_Mix.csv",
        ]
        
        for pattern in inst_patterns:
            inst_csv = out_dir / pattern
            if inst_csv.exists():
                metrics.extend(self._parse_metric_table_csv(inst_csv))
        
        # Also try glob for any instruction mix files
        for csv_file in out_dir.glob("*Instruction*Mix*.csv"):
            if csv_file.name not in inst_patterns:
                metrics.extend(self._parse_metric_table_csv(csv_file))
        
        return metrics
    
    def _parse_memory_metrics_csv(self, out_dir: Path) -> List[MetricResult]:
        """Parse memory/cache metrics from analyze output."""
        metrics = []
        
        # rocprof-compute --save-dfs generates CSVs with pattern: <block_id>.<sub_id>_<Name>.csv
        # Block 12 = LDS, Block 14 = Scalar L1D, Block 16 = Vector L1D, Block 17 = L2
        mem_patterns = [
            # LDS metrics (Block 12)
            "12.2_LDS_Stats.csv",
            # Scalar L1D Cache (Block 14)
            "14.2_Scalar_L1D_Cache_Accesses.csv",
            "14.3_Scalar_L1D_Cache_-_L2_Interface.csv",
            # Vector L1D Cache (Block 16)
            "16.2_L1D_Cache_Stalls_(%).csv",
            "16.3_L1D_Cache_Accesses.csv",
            "16.4_L1D_-_L2_Transactions.csv",
            "16.5_L1D_Addr_Translation.csv",
            # L2 Cache (Block 17)
            "17.2_L2_-_Fabric_Transactions.csv",
            "17.3_L2_Cache_Accesses.csv",
            "17.4_L2_Cache_Stalls.csv",
            "17.5_L2_-_Fabric_Interface_Stalls.csv",
            "17.6_L2_-_Fabric_Detailed_Transaction_Breakdown.csv",
        ]
        
        for pattern in mem_patterns:
            mem_csv = out_dir / pattern
            if mem_csv.exists():
                metrics.extend(self._parse_metric_table_csv(mem_csv))
        
        return metrics
    
    def _parse_metric_table_csv(self, csv_path: Path) -> List[MetricResult]:
        """
        Parse a rocprof-compute metric table CSV file.
        
        Expected format:
        Metric, Avg, Unit, Peak, Pct of Peak
        or
        Metric, Avg, Min, Max, Unit
        """
        metrics = []
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    metric_name = row.get('Metric', row.get('metric', ''))
                    if not metric_name:
                        continue
                    
                    # Try to get average value
                    value = None
                    for val_key in ['Avg', 'avg', 'Value', 'value', 'Average']:
                        if val_key in row and row[val_key]:
                            try:
                                value = float(row[val_key])
                                break
                            except ValueError:
                                pass
                    
                    if value is None:
                        continue
                    
                    # Get unit
                    unit = row.get('Unit', row.get('unit', ''))
                    
                    # Get peak value if available
                    peak = None
                    for peak_key in ['Peak', 'peak']:
                        if peak_key in row and row[peak_key]:
                            try:
                                peak = float(row[peak_key])
                                break
                            except ValueError:
                                pass
                    
                    # Get percent of peak if available
                    pct_of_peak = None
                    for pct_key in ['Pct of Peak', 'pct_of_peak', 'Pop', 'pop']:
                        if pct_key in row and row[pct_key]:
                            try:
                                pct_of_peak = float(row[pct_key])
                                break
                            except ValueError:
                                pass
                    
                    # Map to standardized metric name if in key metrics
                    display_name = ROCPROF_KEY_METRICS.get(metric_name, metric_name)
                    
                    metrics.append(MetricResult(
                        name=display_name,
                        value=value,
                        unit=unit,
                        peak=peak,
                        pct_of_peak=pct_of_peak
                    ))
        except Exception as e:
            logger.warning(f"Failed to parse metric table CSV {csv_path}: {e}")
        
        return metrics

    def _run_ncu_on_testcase(
        self, 
        kernel_cfg: KernelEvalConfig
    ) -> PerformanceResult:
        """
        Run profiling using ncu (NVIDIA Nsight Compute) on the testcase command.
        """
        # Check if ncu is available
        if shutil.which("ncu") is None:
            return PerformanceResult(
                success=False,
                errors="ncu not found. Please install NVIDIA Nsight Compute."
            )
        
        # Get testcase command(s)
        testcase_commands = kernel_cfg.get_testcase_commands()
        if not testcase_commands:
            return PerformanceResult(
                success=False,
                errors="No testcase command available for profiling"
            )
        
        working_dir = kernel_cfg.working_dir
        env = get_updated_env(kernel_cfg.env)
        ncu_cfg = self.perf_cfg.ncu_config
        
        all_outputs = []
        all_metrics = []
        
        try:
            for cmd_idx, testcase_cmd in enumerate(testcase_commands):
                # Build ncu command wrapping the testcase
                cmd = [
                    "ncu",
                    "--target-processes", "all",
                ]
                
                # Add metrics if specified
                if ncu_cfg.metrics:
                    cmd.extend(["--metrics", ",".join(ncu_cfg.metrics)])
                
                # Add kernel filter if specified
                if ncu_cfg.kernel_filter:
                    cmd.extend(["--kernel-name", ncu_cfg.kernel_filter])
                
                # Add custom args
                cmd.extend(ncu_cfg.args)
                cmd.extend(self.perf_cfg.profiler_args)
                
                cmd.extend(["--", *testcase_cmd])
                
                cmd_str = " ".join(cmd)
                logger.info(f"Running ncu on testcase {cmd_idx+1}/{len(testcase_commands)}")
                logger.debug(f"Command: {cmd_str}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=working_dir,
                    timeout=self.perf_cfg.timeout_seconds
                )
                
                all_outputs.append(result.stdout)
                
                if result.returncode != 0:
                    return PerformanceResult(
                        success=False,
                        raw_output="\n".join(all_outputs),
                        errors=f"ncu failed on testcase {cmd_idx+1}: {result.stderr or result.stdout}",
                        command=cmd_str
                    )
                
                # Parse output and extract metrics
                metrics = self._parse_ncu_output(result.stdout)
                all_metrics.extend(metrics)
            
            return PerformanceResult(
                success=True,
                metrics=all_metrics if all_metrics else [MetricResult(
                    name="profiling_complete",
                    value=1.0,
                    unit="bool"
                )],
                raw_output="\n".join(all_outputs)
            )
            
        except subprocess.TimeoutExpired:
            return PerformanceResult(
                success=False,
                errors=f"Profiling timed out after {self.perf_cfg.timeout_seconds}s",
                command=cmd_str if 'cmd_str' in locals() else None
            )
        except Exception as e:
            return PerformanceResult(
                success=False, 
                errors=str(e),
                command=cmd_str if 'cmd_str' in locals() else None
            )

    def _parse_ncu_output(self, output: str) -> List[MetricResult]:
        """Parse ncu output and extract metrics."""
        metrics = []
        
        # Parse ncu text output format
        # Example lines:
        #   Metric Name                          Unit          Value
        #   sm__cycles_elapsed.avg               cycle    123456.00
        
        lines = output.split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                # Try to detect metric lines (metric_name, unit, value)
                try:
                    # Check if last part is a number
                    value = float(parts[-1].replace(',', ''))
                    metric_name = parts[0]
                    unit = parts[-2] if len(parts) > 2 else ""
                    
                    metrics.append(MetricResult(
                        name=metric_name,
                        value=value,
                        unit=unit
                    ))
                except (ValueError, IndexError):
                    pass
        
        if not metrics:
            # Fallback: just indicate profiling completed
            metrics.append(MetricResult(
                name="profiling_complete",
                value=1.0,
                unit="bool"
            ))
        
        return metrics

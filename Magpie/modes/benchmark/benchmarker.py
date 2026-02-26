###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Core benchmarker for benchmark mode.

Orchestrates benchmark execution using InferenceMAX as backend.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import BenchmarkConfig
from .image_selector import ImageSelector
from .inferencemax import ensure_inferencemax_available
from .workspace import WorkspaceManager
from .result import BenchmarkResult, ResultParser
from .tracelens import TraceLensAnalyzer

from ...utils.gpu import detect_gpu, GPUVendor

logger = logging.getLogger(__name__)


class BenchmarkMode:
    """
    Benchmark mode for framework-level profiling.
    
    Uses InferenceMAX as backend to run vLLM/SGLang benchmarks
    inside Docker containers.
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        image_config_path: Optional[str] = None,
        output_dir: str = "./results",
    ):
        """
        Initialize benchmark mode.
        
        Args:
            config: Benchmark configuration
            image_config_path: Path to benchmark_images.yaml
            output_dir: Base directory for results
        """
        self.config = config
        self.image_selector = ImageSelector(image_config_path)
        self.workspace_mgr = WorkspaceManager(
            base_dir=output_dir,
            framework=config.framework,
        )
        self._task_id: Optional[str] = None
    
    def run(self, task_id: Optional[str] = None) -> BenchmarkResult:
        """
        Run benchmark.
        
        Args:
            task_id: Optional task identifier for container naming
        
        Returns:
            BenchmarkResult with metrics and profiling data
        """
        self._task_id = task_id or f"bench_{int(time.time())}"
        start_time = time.time()
        
        logger.info(f"Starting benchmark: {self.config.framework} / {self.config.model}")
        
        # 0. Ensure InferenceMAX is available (auto-clone if needed)
        try:
            self.config.inferencemax_path = ensure_inferencemax_available(
                self.config.inferencemax_path
            )
        except RuntimeError as e:
            result = BenchmarkResult()
            result.success = False
            result.errors.append(f"Failed to setup InferenceMAX: {e}")
            result.errors.append(f"Please clone manually: git clone https://github.com/SemiAnalysisAI/InferenceX.git")
            return result
        
        # 1. Copy Magpie generic scripts to InferenceMAX/benchmarks/
        self._prepare_benchmark_scripts()
        
        # 2. Determine runner type from GPU
        runner_type = self._get_runner_type()
        
        # 3. Find and validate benchmark script BEFORE container starts
        try:
            benchmark_script = self._get_benchmark_script(runner_type)
            logger.info(f"Selected benchmark script: {benchmark_script}")
        except FileNotFoundError as e:
            result = BenchmarkResult()
            result.success = False
            result.errors.append(str(e))
            return result
        
        # 4. Create workspace
        workspace = self.workspace_mgr.create(self.config.to_dict())
        
        # 5. Select Docker image
        docker_image = self._select_image()
        
        # 6. Build Docker command
        docker_cmd = self._build_docker_command(
            docker_image=docker_image,
            workspace=workspace,
            runner_type=runner_type,
        )
        
        # 7. Execute benchmark
        logger.info(f"Running benchmark in container with image: {docker_image}")
        logger.debug(f"Docker command: {' '.join(docker_cmd)}")
        
        result, stdout, stderr = self._execute_benchmark(docker_cmd, workspace)
        
        # 8. Collect results
        result.workspace_dir = str(workspace)
        result.execution_time = time.time() - start_time
        result.framework = self.config.framework
        result.model = self.config.model
        
        # Parse InferenceMAX output
        result_file = workspace / "inferencemax_result.json"
        if result_file.exists():
            parsed = ResultParser.parse_inferencemax_result(
                result_file,
                framework=self.config.framework,
                model=self.config.model,
            )
            # Merge parsed results
            result.throughput = parsed.throughput
            result.latency = parsed.latency
            result.raw_result = parsed.raw_result
            if parsed.errors:
                result.errors.extend(parsed.errors)
        else:
            # Result file not found - benchmark likely failed inside container
            result.success = False
            result.errors.append("inferencemax_result.json not found - benchmark may have failed inside container")
            if stderr:
                # Add last 500 chars of stderr for debugging
                result.errors.append(f"Container stderr (last 500 chars): {stderr[-500:]}")
        
        # Validate that we got actual results
        if result.success and not self._validate_results(result):
            result.success = False
            result.errors.append("Benchmark produced no valid throughput/latency metrics")
        
        # Parse torch trace if available
        if self.config.profiler.torch_profiler.enabled:
            torch_trace_dir = workspace / "torch_trace"
            kernels = ResultParser.parse_torch_trace(torch_trace_dir)
            result.kernel_summary = kernels
            # Get top 10 bottlenecks
            result.top_bottlenecks = [k.name for k in kernels[:10]]
            
            # Run TraceLens analysis if enabled
            # Note: TraceLens runs on the HOST (not in container) after benchmark completes.
            if self.config.profiler.tracelens.enabled:
                tracelens_result = self._run_tracelens_analysis(
                    torch_trace_dir, workspace
                )
                result.tracelens_analysis = tracelens_result
            
            # Run gap analysis if enabled
            if self.config.gap_analysis.enabled:
                gap_result = self._run_gap_analysis(torch_trace_dir, workspace)
                result.gap_analysis = gap_result
        
        # Save report
        self.workspace_mgr.save_report(result.to_dict())
        self.workspace_mgr.save_summary(result.get_summary())
        
        logger.info(f"Benchmark completed in {result.execution_time:.2f}s")
        
        return result
    
    def _select_image(self) -> str:
        """Select Docker image based on configuration."""
        return self.image_selector.select_image(
            framework=self.config.framework,
            gpu_arch=self.config.gpu_arch,
            override_image=self.config.docker_image,
        )
    
    def _get_runner_type(self) -> str:
        """Get InferenceMAX runner type."""
        if self.config.runner_type:
            return self.config.runner_type
        return self.image_selector.get_runner_type(self.config.gpu_arch)
    
    def _prepare_benchmark_scripts(self) -> None:
        """
        Copy Magpie generic benchmark scripts to InferenceMAX/benchmarks/.
        
        This allows using Magpie's generic scripts while still leveraging
        InferenceMAX's benchmark_lib.sh and other utilities.
        
        Note: Won't overwrite existing files to preserve InferenceMAX native scripts.
        """
        # Magpie scripts location: Magpie/scripts/benchmark/
        magpie_scripts = Path(__file__).parent.parent.parent / "scripts" / "benchmark"
        target_dir = Path(self.config.inferencemax_path) / "benchmarks"
        
        if not magpie_scripts.exists():
            logger.debug(f"Magpie scripts directory not found: {magpie_scripts}")
            return
        
        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all .sh scripts (don't overwrite existing)
        for script in magpie_scripts.glob("*.sh"):
            target_file = target_dir / script.name
            if not target_file.exists():
                shutil.copy2(script, target_file)
                # Make script executable
                target_file.chmod(0o755)
                logger.info(f"Copied Magpie script {script.name} to {target_dir}")
            else:
                logger.debug(f"Script {script.name} already exists in {target_dir}, skipping")
    
    def _validate_results(self, result: BenchmarkResult) -> bool:
        """
        Validate that benchmark produced meaningful results.
        
        Args:
            result: BenchmarkResult to validate
        
        Returns:
            True if results are valid, False otherwise
        """
        # Check if we have throughput metrics with actual values
        if result.throughput:
            if result.throughput.request_throughput > 0:
                return True
            if result.throughput.output_throughput > 0:
                return True
            if result.throughput.completed_requests > 0:
                return True
        
        # Check if we have latency metrics
        if result.latency:
            if result.latency.ttft_mean > 0 or result.latency.e2el_mean > 0:
                return True
        
        return False
    
    def _build_docker_command(
        self,
        docker_image: str,
        workspace: Path,
        runner_type: str,
    ) -> List[str]:
        """
        Build Docker run command.
        
        Args:
            docker_image: Docker image to use
            workspace: Workspace directory path
            runner_type: InferenceMAX runner type
        
        Returns:
            Docker command as list of strings
        """
        # Detect GPU vendor for device flags
        vendor, arch = detect_gpu()
        
        cmd = [
            "docker", "run", "--rm",
            "--ipc=host", "--shm-size=16g", "--network=host",
            "--name", f"magpie-benchmark-{self._task_id}",
        ]
        
        # Add GPU-specific flags
        if vendor == GPUVendor.AMD:
            cmd.extend([
                "--privileged",
                "--cap-add=CAP_SYS_ADMIN",
                "--device=/dev/kfd",
                "--device=/dev/dri",
                "--device=/dev/mem",
                "--cap-add=SYS_PTRACE",
                "--security-opt", "seccomp=unconfined",
            ])
        elif vendor == GPUVendor.NVIDIA:
            cmd.extend([
                "--gpus", "all",
            ])
        
        # HuggingFace cache mount
        hf_cache = self.config.hf_cache_path or os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(hf_cache):
            cmd.extend(["-v", f"{hf_cache}:/root/.cache/huggingface"])
        
        # InferenceMAX mount
        inferencemax_path = self.config.inferencemax_path
        if os.path.exists(inferencemax_path):
            cmd.extend(["-v", f"{inferencemax_path}:/workspace/InferenceMAX"])
        
        # Workspace mount (for results output)
        cmd.extend(["-v", f"{workspace}:/workspace/output"])
        
        # Environment variables
        env_vars = self.config.get_env_vars()
        env_vars["RESULT_FILENAME"] = "/workspace/output/inferencemax_result"
        env_vars["RESULT_DIR"] = "/workspace/output"
        env_vars["RUNNER_TYPE"] = runner_type
        
        # torch_profiler environment (matches official InferenceX: PROFILE=1)
        if self.config.profiler.torch_profiler.enabled:
            env_vars["PROFILE"] = "1"
            env_vars["VLLM_TORCH_PROFILER_DIR"] = "/workspace/output/torch_trace"
            env_vars["SGLANG_TORCH_PROFILER_DIR"] = "/workspace/output/torch_trace"

        
        # HuggingFace token from environment
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env_vars["HF_TOKEN"] = hf_token
        
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])
        
        # Working directory
        cmd.extend(["-w", "/workspace/InferenceMAX"])
        
        # Image and entrypoint - always override to bash for script compatibility
        cmd.extend(["--entrypoint", "/bin/bash"])
        cmd.append(docker_image)
        
        # Build the benchmark command
        benchmark_script = self._get_benchmark_script(runner_type)
        
        # With --entrypoint /bin/bash, pass -c as first arg
        cmd.extend([
            "-c",
            f"cd /workspace/InferenceMAX && bash {benchmark_script}"
        ])
        
        return cmd
    
    def _get_benchmark_script(self, runner_type: str) -> str:
        """
        Get InferenceMAX benchmark script path with 3-tier priority.
        
        Priority:
            1. User-specified script (benchmark_script config) - must exist
            2. InferenceMAX native scripts: {prefix}_{precision}_{runner}.sh
               - sglang -> dsr1_
               - vllm -> gptoss_
            3. Magpie generic scripts: {framework}_{runner}.sh
        
        Args:
            runner_type: Runner type (e.g., "mi300x", "h100", "b200")
        
        Returns:
            Relative path to benchmark script
            
        Raises:
            FileNotFoundError: If no suitable script found
        """
        benchmarks_dir = Path(self.config.inferencemax_path) / "benchmarks"
        
        # Priority 1: User-specified script (must exist)
        if self.config.benchmark_script:
            script_path = benchmarks_dir / self.config.benchmark_script
            if not script_path.exists():
                raise FileNotFoundError(
                    f"Specified benchmark_script not found: {script_path}\n"
                    f"Please ensure the file exists or remove the benchmark_script config."
                )
            logger.info(f"Using user-specified script: {self.config.benchmark_script}")
            return f"benchmarks/{self.config.benchmark_script}"
        
        # Priority 2: InferenceMAX native scripts
        # Mapping: sglang -> dsr1_, vllm -> gptoss_
        prefix_map = {"sglang": "dsr1", "vllm": "gptoss"}
        prefix = prefix_map.get(self.config.framework)
        
        if prefix:
            native_script = f"{prefix}_{self.config.precision}_{runner_type}.sh"
            if (benchmarks_dir / native_script).exists():
                logger.info(f"Using InferenceMAX native script: {native_script}")
                return f"benchmarks/{native_script}"
        
        # Priority 3: Magpie generic scripts
        generic_script = f"{self.config.framework}_{runner_type}.sh"
        if (benchmarks_dir / generic_script).exists():
            logger.info(f"Using Magpie generic script: {generic_script}")
            return f"benchmarks/{generic_script}"
        
        # No script found - raise error with helpful message
        raise FileNotFoundError(
            f"No benchmark script found for framework={self.config.framework}, "
            f"precision={self.config.precision}, gpu={runner_type}.\n"
            f"Expected one of:\n"
            f"  - {prefix}_{self.config.precision}_{runner_type}.sh (InferenceMAX native)\n"
            f"  - {generic_script} (Magpie generic)\n"
            f"Please create the script or specify benchmark_script in config."
        )
    
    def _execute_benchmark(self, cmd: List[str], workspace: Path) -> tuple:
        """
        Execute Docker benchmark command.
        
        Args:
            cmd: Docker command
            workspace: Workspace directory for saving logs
        
        Returns:
            Tuple of (BenchmarkResult, stdout, stderr)
        """
        result = BenchmarkResult()
        stdout = ""
        stderr = ""
        
        try:
            # Run Docker command
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            
            stdout = process.stdout or ""
            stderr = process.stderr or ""
            
            # Save logs to workspace for debugging
            self._save_container_logs(workspace, stdout, stderr)
            
            if process.returncode == 0:
                result.success = True
                logger.info("Benchmark completed successfully")
            else:
                result.success = False
                result.errors.append(f"Docker command failed with code {process.returncode}")
                if stderr:
                    result.errors.append(f"stderr: {stderr[:1000]}")
                logger.error(f"Benchmark failed: {process.returncode}")
                logger.error(f"stderr: {stderr[:500]}")
            
            # Log stdout for debugging
            if stdout:
                logger.debug(f"stdout (last 500 chars): {stdout[-500:]}")
                
        except subprocess.TimeoutExpired as e:
            result.errors.append(f"Benchmark timed out after {self.config.timeout_seconds}s")
            logger.error(f"Benchmark timed out")
            
            # Capture partial output if available
            if hasattr(e, 'stdout') and e.stdout:
                stdout = e.stdout if isinstance(e.stdout, str) else e.stdout.decode()
            if hasattr(e, 'stderr') and e.stderr:
                stderr = e.stderr if isinstance(e.stderr, str) else e.stderr.decode()
            
            self._save_container_logs(workspace, stdout, stderr)
            
            # Try to stop the container
            try:
                subprocess.run(
                    ["docker", "stop", f"magpie-benchmark-{self._task_id}"],
                    capture_output=True,
                    timeout=30,
                )
            except Exception:
                pass
                
        except Exception as e:
            result.errors.append(f"Benchmark execution error: {str(e)}")
            logger.exception(f"Benchmark execution failed: {e}")
        
        return result, stdout, stderr
    
    def _save_container_logs(self, workspace: Path, stdout: str, stderr: str) -> None:
        """
        Save container stdout/stderr to workspace for debugging.
        
        Args:
            workspace: Workspace directory
            stdout: Container stdout
            stderr: Container stderr
        """
        try:
            if stdout:
                stdout_file = workspace / "container_stdout.log"
                with open(stdout_file, 'w') as f:
                    f.write(stdout)
                logger.debug(f"Saved container stdout to {stdout_file}")
            
            if stderr:
                stderr_file = workspace / "container_stderr.log"
                with open(stderr_file, 'w') as f:
                    f.write(stderr)
                logger.debug(f"Saved container stderr to {stderr_file}")
        except Exception as e:
            logger.warning(f"Failed to save container logs: {e}")
    
    def _run_tracelens_analysis(
        self,
        torch_trace_dir: Path,
        workspace: Path,
    ) -> Dict[str, Any]:
        """
        Run TraceLens analysis on torch profiler traces.
        
        NOTE: This runs on the HOST machine (not in container) after the
        Docker benchmark completes. TraceLens will be auto-installed from
        https://github.com/AMD-AIG-AIMA/TraceLens.git if not present.
        
        Args:
            torch_trace_dir: Directory containing torch trace files
            workspace: Workspace directory for output
        
        Returns:
            TraceLens analysis results dictionary
        """
        logger.info("Running TraceLens analysis on host...")
        
        try:
            analyzer = TraceLensAnalyzer(self.config.profiler.tracelens)
            
            # Get number of ranks from TP config
            num_ranks = int(self.config.envs.get("TP", 8))
            
            results = analyzer.analyze(
                trace_dir=torch_trace_dir,
                output_dir=workspace,
                num_ranks=num_ranks,
            )
            
            if results.get("output_files"):
                logger.info(
                    f"TraceLens analysis complete: {len(results['output_files'])} output files"
                )
            
            if results.get("errors"):
                for error in results["errors"]:
                    logger.warning(f"TraceLens warning: {error}")
            
            return results
            
        except Exception as e:
            logger.exception(f"TraceLens analysis failed: {e}")
            return {"enabled": True, "error": str(e)}
    
    def _run_gap_analysis(
        self,
        torch_trace_dir: Path,
        workspace: Path,
    ) -> Dict[str, Any]:
        """
        Run gap analysis on torch profiler traces.
        
        Analyzes a time window of the trace to identify kernel-level
        bottlenecks and writes a CSV report.
        
        Args:
            torch_trace_dir: Directory containing torch trace files
            workspace: Workspace directory for output
        
        Returns:
            Gap analysis results dictionary
        """
        from .gap_analysis import GapAnalyzer

        logger.info("Running gap analysis on torch traces...")
        
        try:
            gap_dir = workspace / "gap_analysis"
            gap_dir.mkdir(parents=True, exist_ok=True)

            analyzer = GapAnalyzer(self.config.gap_analysis)
            result = analyzer.analyze(torch_trace_dir)
            
            # Write merged CSV
            csv_path = gap_dir / "gap_analysis.csv"
            result.to_csv(csv_path)
            logger.info(f"Wrote gap analysis CSV: {csv_path}")
            
            # Write per-rank CSVs if multiple ranks
            if len(result.rank_results) > 1:
                rank_paths = result.to_rank_csv(gap_dir)
                for rp in rank_paths:
                    logger.info(f"Wrote per-rank CSV: {rp}")
            
            ga_dict = result.to_dict()
            ga_dict["csv_path"] = str(csv_path)
            ga_dict["output_dir"] = str(gap_dir)
            
            if result.errors:
                for err in result.errors:
                    logger.warning(f"Gap analysis warning: {err}")
            else:
                n = len(result.merged_kernels)
                logger.info(f"Gap analysis complete: {n} kernels in CSV")
            
            return ga_dict
            
        except Exception as e:
            logger.exception(f"Gap analysis failed: {e}")
            return {"enabled": True, "error": str(e)}
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Stop any running container
        if self._task_id:
            try:
                subprocess.run(
                    ["docker", "stop", f"magpie-benchmark-{self._task_id}"],
                    capture_output=True,
                    timeout=30,
                )
            except Exception:
                pass


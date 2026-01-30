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
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import BenchmarkConfig
from .image_selector import ImageSelector
from .workspace import WorkspaceManager
from .result import BenchmarkResult, ResultParser

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
        
        # 1. Create workspace
        workspace = self.workspace_mgr.create(self.config.to_dict())
        
        # 2. Select Docker image
        docker_image = self._select_image()
        
        # 3. Determine runner type and benchmark script
        runner_type = self._get_runner_type()
        
        # 4. Build Docker command
        docker_cmd = self._build_docker_command(
            docker_image=docker_image,
            workspace=workspace,
            runner_type=runner_type,
        )
        
        # 5. Execute benchmark
        logger.info(f"Running benchmark in container with image: {docker_image}")
        logger.debug(f"Docker command: {' '.join(docker_cmd)}")
        
        result, stdout, stderr = self._execute_benchmark(docker_cmd, workspace)
        
        # 6. Collect results
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
            result.top_bottlenecks = [k.name for k in kernels[:10]]
        
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
        
        # torch_profiler environment
        if self.config.profiler.torch_profiler.enabled:
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
        
        # Image and entrypoint
        cmd.append(docker_image)
        
        # Build the benchmark command
        benchmark_script = self._get_benchmark_script(runner_type)
        
        # Use bash to run the benchmark
        cmd.extend([
            "/bin/bash", "-c",
            f"cd /workspace/InferenceMAX && bash {benchmark_script}"
        ])
        
        return cmd
    
    def _get_benchmark_script(self, runner_type: str) -> str:
        """
        Get InferenceMAX benchmark script path.
        
        Args:
            runner_type: Runner type (e.g., "mi300x", "h100")
        
        Returns:
            Relative path to benchmark script
        """
        if self.config.benchmark_script:
            return f"benchmarks/{self.config.benchmark_script}"
        
        # Try to find matching script
        # InferenceMAX convention: {exp_name}_{precision}_{runner}.sh
        inferencemax_path = Path(self.config.inferencemax_path)
        benchmarks_dir = inferencemax_path / "benchmarks"
        
        if benchmarks_dir.exists():
            # Look for scripts matching the runner
            pattern = f"*_{self.config.precision}_{runner_type}.sh"
            matches = list(benchmarks_dir.glob(pattern))
            
            if matches:
                # Return first match
                return f"benchmarks/{matches[0].name}"
        
        # Fallback: generic script name
        logger.warning(f"No matching benchmark script found, using generic")
        return f"benchmarks/generic_{self.config.precision}_{runner_type}.sh"
    
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


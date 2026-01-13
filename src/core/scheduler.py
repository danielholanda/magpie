"""
Workload scheduler for kernel evaluation.

This module handles workload scheduling, environment preparation,
and orchestration of evaluation tasks across available resources.
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config import KernelEvalConfig

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Type of execution environment."""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"


@dataclass
class WorkloadConfig:
    """
    Configuration for workload scheduling.
    
    Attributes:
        max_concurrent_tasks: Maximum number of tasks to run concurrently
        environment_type: Type of execution environment
        docker_image: Docker image to use (if applicable)
        gpu_devices: List of GPU device IDs to use
        retry_on_failure: Whether to retry failed tasks
        max_retries: Maximum number of retries per task
        pre_hooks: Functions to run before task execution
        post_hooks: Functions to run after task execution
    """
    max_concurrent_tasks: int = 1
    environment_type: EnvironmentType = EnvironmentType.LOCAL
    docker_image: Optional[str] = None
    gpu_devices: List[int] = field(default_factory=lambda: [0])
    retry_on_failure: bool = True
    max_retries: int = 3
    pre_hooks: List[Callable] = field(default_factory=list)
    post_hooks: List[Callable] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.environment_type, str):
            self.environment_type = EnvironmentType(self.environment_type)


class Scheduler:
    """
    Workload scheduler for managing kernel evaluation tasks.
    
    The scheduler handles:
    - Task queue management
    - Environment preparation (local, docker, etc.)
    - Pre/post processing hooks
    - Resource allocation and cleanup
    """

    def __init__(self, config: WorkloadConfig):
        """
        Initialize the scheduler.
        
        Args:
            config: Workload configuration
        """
        self.config = config
        self._task_queue: List[KernelEvalConfig] = []
        self._is_initialized = False
        self._available_gpus: List[int] = []

    def prepare_environment(self) -> bool:
        """
        Prepare the execution environment.
        
        Returns:
            True if environment is ready, False otherwise
        """
        logger.info(f"Preparing {self.config.environment_type.value} environment...")
        
        if self.config.environment_type == EnvironmentType.DOCKER:
            return self._prepare_docker_environment()
        elif self.config.environment_type == EnvironmentType.LOCAL:
            return self._prepare_local_environment()
        elif self.config.environment_type == EnvironmentType.KUBERNETES:
            return self._prepare_k8s_environment()
        
        return False

    def _prepare_local_environment(self) -> bool:
        """Prepare local execution environment."""
        # Verify GPU availability
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                gpus = result.stdout.strip().split("\n")
                self._available_gpus = list(range(len(gpus)))
                logger.info(f"Found {len(gpus)} GPU(s)")
                for i, gpu in enumerate(gpus):
                    logger.debug(f"  GPU {i}: {gpu.strip()}")
                self._is_initialized = True
                return True
            else:
                logger.warning("nvidia-smi failed, GPU may not be available")
                # Still allow running without GPU
                self._is_initialized = True
                return True
        except FileNotFoundError:
            logger.warning("nvidia-smi not found, GPU may not be available")
            self._is_initialized = True
            return True
        except subprocess.TimeoutExpired:
            logger.error("nvidia-smi timed out")
            return False

    def _prepare_docker_environment(self) -> bool:
        """Prepare Docker execution environment."""
        if not self.config.docker_image:
            logger.error("Docker image not specified")
            return False
        
        # Check if Docker is available
        try:
            subprocess.run(
                ["docker", "version"],
                capture_output=True,
                check=True,
                timeout=10
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Docker is not available")
            return False
        
        # Pull the image if needed
        logger.info(f"Pulling Docker image: {self.config.docker_image}")
        try:
            subprocess.run(
                ["docker", "pull", self.config.docker_image],
                capture_output=True,
                check=True,
                timeout=300
            )
            self._is_initialized = True
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull Docker image: {e}")
            return False

    def _prepare_k8s_environment(self) -> bool:
        """Prepare Kubernetes execution environment."""
        # Placeholder for Kubernetes support
        logger.warning("Kubernetes environment is not yet implemented")
        return False

    def add_task(self, kernel_cfg: KernelEvalConfig) -> None:
        """
        Add a kernel configuration to the execution queue.
        
        Args:
            kernel_cfg: Kernel configuration to add
        """
        self._task_queue.append(kernel_cfg)
        logger.debug(f"Added kernel {kernel_cfg.kernel_id} to queue")

    def add_tasks(self, kernel_configs: List[KernelEvalConfig]) -> None:
        """
        Add multiple kernel configurations to the execution queue.
        
        Args:
            kernel_configs: List of kernel configurations to add
        """
        for cfg in kernel_configs:
            self.add_task(cfg)

    def get_pending_tasks(self) -> List[KernelEvalConfig]:
        """
        Get list of pending tasks.
        
        Returns:
            List of kernel configurations in queue
        """
        return self._task_queue.copy()

    def clear_tasks(self) -> None:
        """Clear all pending tasks from queue."""
        self._task_queue.clear()

    def run_pre_hooks(self, kernel_cfg: KernelEvalConfig) -> None:
        """
        Run pre-execution hooks.
        
        Args:
            kernel_cfg: Kernel configuration about to be executed
        """
        for hook in self.config.pre_hooks:
            try:
                hook(kernel_cfg)
            except Exception as e:
                logger.warning(f"Pre-hook failed: {e}")

    def run_post_hooks(self, kernel_cfg: KernelEvalConfig, result: Any) -> None:
        """
        Run post-execution hooks.
        
        Args:
            kernel_cfg: Kernel configuration that was executed
            result: Result of the execution
        """
        for hook in self.config.post_hooks:
            try:
                hook(kernel_cfg, result)
            except Exception as e:
                logger.warning(f"Post-hook failed: {e}")

    def get_gpu_assignment(self) -> Optional[int]:
        """
        Get next available GPU for task execution.
        
        Returns:
            GPU device ID or None if no GPU available
        """
        if not self._available_gpus:
            return None
        # Simple round-robin assignment
        return self.config.gpu_devices[0] if self.config.gpu_devices else 0

    def is_initialized(self) -> bool:
        """Check if scheduler is initialized."""
        return self._is_initialized

    def cleanup(self) -> None:
        """Clean up scheduler resources."""
        self._task_queue.clear()
        self._is_initialized = False
        logger.info("Scheduler cleanup completed")

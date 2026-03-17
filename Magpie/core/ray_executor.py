###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Ray Job executor for remote benchmark / analyze / compare execution.

Submits tasks as Ray Jobs via the Ray Job Submission API.  Results are
exchanged through shared NFS storage; the executor never requires a local
GPU.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .executor import BaseExecutor, ExecutorConfig
from .task import Task, TaskResult, TaskStatus
from .job_store import JobStore, JobRecord

logger = logging.getLogger(__name__)

# Ray job status constants (mirrors ray.job_submission.JobStatus values)
_RAY_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "STOPPED"}


class RayJobExecutor(BaseExecutor):
    """
    Executor that submits tasks to a remote Ray cluster via the
    Ray Job Submission REST API.

    Prerequisites:
        * ``ray[default]`` must be installed (``pip install ray[default]``)
        * A Ray cluster must be reachable at the configured address
        * All nodes (MCP server + Ray workers) must share an NFS mount
    """

    def __init__(self, config: ExecutorConfig, ray_config: Any):
        """
        Args:
            config: Base executor configuration.
            ray_config: A ``RayConfig`` instance from
                ``Magpie.modes.benchmark.config``.
        """
        super().__init__(config)
        self._ray_config = ray_config
        self._client: Any = None  # ray.job_submission.JobSubmissionClient
        self._job_store: Optional[JobStore] = None
        # task_id -> ray_job_id mapping kept in memory for the current session
        self._job_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Connect to the Ray cluster and initialise the job store."""
        if self._is_running:
            logger.warning("RayJobExecutor is already running")
            return True

        try:
            from ray.job_submission import JobSubmissionClient
        except ImportError:
            logger.error(
                "ray[default] is not installed.  "
                "Run: pip install 'ray[default]'"
            )
            return False

        try:
            self._client = JobSubmissionClient(self._ray_config.cluster_address)
            logger.info(
                f"Connected to Ray cluster at {self._ray_config.cluster_address}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Ray cluster: {e}")
            return False

        try:
            self._job_store = JobStore(self._ray_config.job_store_path)
        except Exception as e:
            logger.error(f"Failed to initialise job store: {e}")
            return False

        self._is_running = True
        logger.info("RayJobExecutor started")
        return True

    def stop(self) -> None:
        """Disconnect and release resources."""
        if self._job_store:
            self._job_store.close()
            self._job_store = None
        self._client = None
        self._is_running = False
        self._pending_tasks.clear()
        self._job_map.clear()
        logger.info("RayJobExecutor stopped")

    # ------------------------------------------------------------------
    # Submit (async – returns immediately)
    # ------------------------------------------------------------------

    def submit(self, task: Task) -> str:
        """
        Submit a task as a Ray job.  Returns the Magpie task ID immediately.
        """
        if not self._is_running:
            raise RuntimeError("RayJobExecutor is not running")

        task.status = TaskStatus.RUNNING
        self._pending_tasks[task.task_id] = task

        ray_job_id = self._submit_ray_job(task)
        self._job_map[task.task_id] = ray_job_id

        logger.info(
            f"Task {task.task_id} submitted as Ray job {ray_job_id}"
        )
        return task.task_id

    # ------------------------------------------------------------------
    # Execute (synchronous – polls until done)
    # ------------------------------------------------------------------

    def execute(self, task: Task) -> TaskResult:
        """Submit and block until the Ray job completes."""
        self.submit(task)
        return self.wait_for_task(
            task.task_id, timeout=self.config.timeout_seconds
        )

    # ------------------------------------------------------------------
    # Waiting / polling
    # ------------------------------------------------------------------

    def wait_for_task(
        self, task_id: str, timeout: Optional[float] = None
    ) -> TaskResult:
        """Poll the Ray job until it reaches a terminal state."""
        if task_id not in self._job_map:
            raise ValueError(f"Task {task_id} not found")

        ray_job_id = self._job_map[task_id]
        deadline = time.time() + timeout if timeout else None
        poll_interval = 5.0

        while True:
            status_str = self._get_ray_status(ray_job_id)
            if self._job_store:
                self._job_store.update_status(ray_job_id, status_str)

            if status_str in _RAY_TERMINAL_STATUSES:
                break

            if deadline and time.time() >= deadline:
                logger.warning(f"Timeout waiting for Ray job {ray_job_id}")
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    errors=[f"Timeout after {timeout}s waiting for Ray job {ray_job_id}"],
                )

            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 30.0)

        # Map Ray status to Magpie TaskStatus
        if status_str == "SUCCEEDED":
            magpie_status = TaskStatus.COMPLETED
        elif status_str == "STOPPED":
            magpie_status = TaskStatus.CANCELLED
        else:
            magpie_status = TaskStatus.FAILED

        # Read result from shared storage
        record = self._job_store.get(ray_job_id) if self._job_store else None
        results = None
        errors: List[str] = []

        if record and record.result_path:
            result_path = Path(record.result_path)
            if result_path.exists():
                try:
                    results = json.loads(result_path.read_text())
                except Exception as e:
                    errors.append(f"Failed to read result: {e}")
            else:
                errors.append(f"Result file not found: {record.result_path}")

        if magpie_status == TaskStatus.FAILED:
            logs = self.get_job_logs(ray_job_id)
            if logs:
                errors.append(f"Ray job logs (last 2000 chars): {logs[-2000:]}")

        if task_id in self._pending_tasks:
            self._pending_tasks[task_id].status = magpie_status

        return TaskResult(
            task_id=task_id,
            status=magpie_status,
            results=results,
            errors=errors,
            metadata={"ray_job_id": ray_job_id},
        )

    def wait_all(self, timeout: Optional[float] = None) -> List[TaskResult]:
        """Wait for all submitted tasks."""
        results = []
        for task_id in list(self._job_map.keys()):
            result = self.wait_for_task(task_id, timeout)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Job management helpers (used by MCP tools)
    # ------------------------------------------------------------------

    def get_ray_job_id(self, task_id: str) -> Optional[str]:
        """Resolve a Magpie task ID to a Ray job ID."""
        return self._job_map.get(task_id)

    def get_job_status(self, ray_job_id: str) -> str:
        """Get the current status string of a Ray job."""
        return self._get_ray_status(ray_job_id)

    def get_job_logs(self, ray_job_id: str) -> str:
        """Fetch logs from a Ray job."""
        if not self._client:
            return ""
        try:
            return self._client.get_job_logs(ray_job_id)
        except Exception as e:
            logger.warning(f"Failed to get logs for {ray_job_id}: {e}")
            return f"[error fetching logs: {e}]"

    def cancel_job(self, ray_job_id: str) -> bool:
        """Cancel / stop a running Ray job."""
        if not self._client:
            return False
        try:
            self._client.stop_job(ray_job_id)
            if self._job_store:
                self._job_store.update_status(ray_job_id, "STOPPED")
            logger.info(f"Cancelled Ray job {ray_job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel Ray job {ray_job_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit_ray_job(self, task: Task) -> str:
        """Build the entrypoint, write config to NFS, and submit to Ray."""
        rc = self._ray_config
        job_dir = Path(rc.results_dir) / task.task_id
        job_dir.mkdir(parents=True, exist_ok=True)

        config_path = job_dir / "config.json"
        result_path = job_dir / "result.json"

        # Serialise the full task config to shared storage
        job_payload = {
            "task_id": task.task_id,
            "mode_type": task.mode_config.mode_type.value,
            "mode_config": task.to_dict()["mode_config"],
            "ray_config": rc.to_dict(),
        }
        config_path.write_text(json.dumps(job_payload, indent=2))

        # Build the entrypoint command
        entrypoint = (
            f"python -m Magpie.remote.entrypoint "
            f"--config {config_path} "
            f"--result-path {result_path}"
        )

        # Build runtime_env
        runtime_env: Dict[str, Any] = {}
        env_vars: Dict[str, str] = {
            "HF_HOME": rc.hf_cache_dir,
            "TRANSFORMERS_CACHE": f"{rc.hf_cache_dir}/hub",
        }
        if rc.multi_node:
            env_vars["RAY_ADDRESS"] = "auto"
            env_vars["MAGPIE_TOTAL_GPUS"] = str(rc.total_num_gpus)
        env_vars.update(rc.env_vars)
        runtime_env["env_vars"] = env_vars

        if rc.pip_packages:
            runtime_env["pip"] = rc.pip_packages

        # Submit via Ray Job Submission API
        submit_kwargs: Dict[str, Any] = {
            "entrypoint": entrypoint,
            "runtime_env": runtime_env,
            "metadata": {
                "magpie_task_id": task.task_id,
                "mode_type": task.mode_config.mode_type.value,
                **rc.metadata,
            },
        }
        if rc.entrypoint_num_gpus > 0:
            submit_kwargs["entrypoint_num_gpus"] = rc.entrypoint_num_gpus
        if rc.entrypoint_num_cpus > 0:
            submit_kwargs["entrypoint_num_cpus"] = rc.entrypoint_num_cpus

        ray_job_id = self._client.submit_job(**submit_kwargs)

        # Persist the record
        if self._job_store:
            self._job_store.add(JobRecord(
                ray_job_id=ray_job_id,
                magpie_task_id=task.task_id,
                mode_type=task.mode_config.mode_type.value,
                ray_cluster=rc.cluster_address,
                config_path=str(config_path),
                result_path=str(result_path),
                workspace_dir=str(job_dir),
                status="PENDING",
                metadata=submit_kwargs.get("metadata", {}),
            ))

        return ray_job_id

    def _get_ray_status(self, ray_job_id: str) -> str:
        """Query the Ray cluster for job status."""
        if not self._client:
            return "UNKNOWN"
        try:
            status = self._client.get_job_status(ray_job_id)
            return str(status.value) if hasattr(status, "value") else str(status)
        except Exception as e:
            logger.warning(f"Failed to get status for {ray_job_id}: {e}")
            return "UNKNOWN"

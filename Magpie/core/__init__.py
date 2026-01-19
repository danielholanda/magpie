"""
Core evaluation engine components.

This module provides:
- Scheduler: Task scheduling and executor management
- Executor: Task execution (Local, Container, future: Distributed)
- Task: Task and result definitions
"""

from .task import (
    Task,
    TaskResult,
    TaskStatus,
    ModeType,
    ModeConfig,
)
from .executor import (
    BaseExecutor,
    LocalExecutor,
    ContainerExecutor,
    ExecutorConfig,
    ExecutorType,
    create_executor,
)
from .scheduler import (
    Scheduler,
    SchedulerConfig,
    EnvironmentType,
)

__all__ = [
    # Task definitions
    "Task",
    "TaskResult",
    "TaskStatus",
    "ModeType",
    "ModeConfig",
    # Executor classes
    "BaseExecutor",
    "LocalExecutor",
    "ContainerExecutor",
    "ExecutorConfig",
    "ExecutorType",
    "create_executor",
    # Scheduler classes
    "Scheduler",
    "SchedulerConfig",
    "EnvironmentType",
]

"""
Execution engine for kernel evaluation.

This module provides a unified interface for running evaluations,
handling result saving, and generating summaries.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .scheduler import Scheduler, WorkloadConfig
from ..config import EvalMode, PipelineConfig, KernelEvalConfig
from ..eval import Evaluator, EvaluationState, BaseKind

logger = logging.getLogger(__name__)


@dataclass
class ExecutorConfig:
    """
    Configuration for the executor.
    
    Attributes:
        mode: Evaluation mode (analyze or compare)
        pipeline_config: Pipeline configuration for evaluation
        output_dir: Directory for output files
        output_format: Output format (json, csv, markdown)
        verbose: Enable verbose logging
        workload_config: Scheduler configuration
    """
    mode: EvalMode = EvalMode.ANALYZE
    pipeline_config: Optional[PipelineConfig] = None
    output_dir: Path = Path("./results")
    output_format: str = "json"
    verbose: bool = False
    workload_config: Optional[WorkloadConfig] = None

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if self.pipeline_config is None:
            self.pipeline_config = PipelineConfig()


class Executor:
    """
    Main execution engine for kernel evaluation.
    
    Supports:
    - ANALYZE mode: Evaluate one or more kernels with testcase
    - COMPARE mode: Compare multiple kernels
    """

    def __init__(self, config: ExecutorConfig):
        """
        Initialize the executor.
        
        Args:
            config: Executor configuration
        """
        self.config = config
        self._setup_logging()
        
        workload_config = config.workload_config or WorkloadConfig()
        self.scheduler = Scheduler(workload_config)
        
        # Initialize the evaluator
        self.evaluator = Evaluator(config.pipeline_config)
        
    def _setup_logging(self) -> None:
        """Configure logging based on verbosity setting."""
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def prepare(self) -> bool:
        """
        Prepare the execution environment.
        
        Returns:
            True if preparation succeeded, False otherwise
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        return self.scheduler.prepare_environment()

    def evaluate(self, kernel_cfg: KernelEvalConfig) -> EvaluationState:
        """
        Evaluate a single kernel.
        
        Args:
            kernel_cfg: Kernel configuration
            
        Returns:
            EvaluationState with evaluation results
        """
        logger.info(f"Evaluating kernel: {kernel_cfg.kernel_id}")
        return self.evaluator.evaluate(kernel_cfg)

    def evaluate_all(
        self, 
        kernel_configs: List[KernelEvalConfig]
    ) -> List[EvaluationState]:
        """
        Evaluate multiple kernels.
        
        Works for both ANALYZE and COMPARE modes.
        
        Args:
            kernel_configs: List of kernel configurations
            
        Returns:
            List of EvaluationState results
        """
        logger.info(f"Evaluating {len(kernel_configs)} kernels")
        
        results = []
        for i, kernel_cfg in enumerate(kernel_configs):
            logger.info(f"Processing kernel {i+1}/{len(kernel_configs)}: {kernel_cfg.kernel_id}")
            state = self.evaluate(kernel_cfg)
            results.append(state)
            
        # Save results
        self._save_results(results)
        
        return results

    def _save_results(self, results: List[EvaluationState]) -> None:
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.output_format == "json":
            output_file = self.config.output_dir / f"results_{timestamp}.json"
            output_data = {
                "mode": self.config.mode.name,
                "timestamp": timestamp,
                "results": [r.to_dict() for r in results],
                "summary": self._generate_summary(results)
            }
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Results saved to {output_file}")
            
        elif self.config.output_format == "csv":
            self._save_csv_results(results, timestamp)
            
        elif self.config.output_format == "markdown":
            self._save_markdown_results(results, timestamp)

    def _save_csv_results(
        self, 
        results: List[EvaluationState], 
        timestamp: str
    ) -> None:
        """Save results in CSV format."""
        import csv
        
        output_file = self.config.output_dir / f"results_{timestamp}.csv"
        
        fieldnames = [
            "kernel_id", "kernel_type", "compiling", "correctness", 
            "performance", "score", "errors"
        ]
        
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow({
                    "kernel_id": result.extra.get("kernel_id", "unknown"),
                    "kernel_type": result.extra.get("kernel_type", "unknown"),
                    "compiling": result.compiling_state.name,
                    "correctness": result.correctness_state.name,
                    "performance": result.performance_state.name,
                    "score": result.score,
                    "errors": "; ".join(result.errors) if result.errors else ""
                })
        
        logger.info(f"Results saved to {output_file}")

    def _save_markdown_results(
        self, 
        results: List[EvaluationState], 
        timestamp: str
    ) -> None:
        """Save results in Markdown format."""
        output_file = self.config.output_dir / f"results_{timestamp}.md"
        
        with open(output_file, "w") as f:
            f.write("# Evaluation Results\n\n")
            f.write(f"**Mode:** {self.config.mode.name}\n")
            f.write(f"**Timestamp:** {timestamp}\n\n")
            
            f.write("## Summary\n\n")
            summary = self._generate_summary(results)
            f.write(f"- Total Kernels: {summary['total']}\n")
            f.write(f"- Passed: {summary['passed']}\n")
            f.write(f"- Failed: {summary['failed']}\n")
            f.write(f"- Pass Rate: {summary['pass_rate']:.1%}\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("| # | Kernel | Compiling | Correctness | Performance | Score |\n")
            f.write("|---|--------|-----------|-------------|-------------|-------|\n")
            for i, result in enumerate(results):
                kernel_id = result.extra.get("kernel_id", "unknown")
                f.write(
                    f"| {i+1} | {kernel_id} | {result.compiling_state.name} | "
                    f"{result.correctness_state.name} | "
                    f"{result.performance_state.name} | "
                    f"{result.score:.2f} |\n"
                )
        
        logger.info(f"Results saved to {output_file}")

    def _generate_summary(self, results: List[EvaluationState]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        total = len(results)
        passed = sum(
            1 for r in results 
            if r.compiling_state == BaseKind.SUCCESS 
            and r.correctness_state == BaseKind.SUCCESS
        )
        failed = total - passed
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0
        }

    def cleanup(self) -> None:
        """Clean up executor resources."""
        self.scheduler.cleanup()
        logger.info("Executor cleanup completed")

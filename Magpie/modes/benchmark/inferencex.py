###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
InferenceX repository management.

Handles automatic cloning and validation of the InferenceX repository
for benchmark execution.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# InferenceX repository configuration
INFERENCEX_REPO_URL = "https://github.com/SemiAnalysisAI/InferenceX.git"
# Default directory:
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
INFERENCEX_DEFAULT_DIR = str(_PROJECT_ROOT / "InferenceX")

# Placeholder values that indicate the path is not configured
PLACEHOLDER_VALUES = {
    "YOUR_INFERENCEX_PATH",
    "",
}


class InferenceXManager:
    """
    Manages InferenceX repository for benchmark execution.
    
    Handles:
    - Validation of existing installation
    - Automatic cloning if not present
    - Path resolution for placeholder values
    """
    
    def __init__(
        self,
        repo_url: str = INFERENCEX_REPO_URL,
        default_dir: str = INFERENCEX_DEFAULT_DIR,
    ):
        """
        Initialize InferenceX manager.
        
        Args:
            repo_url: Git repository URL for InferenceX
            default_dir: Default directory to clone into
        """
        self.repo_url = repo_url
        self.default_dir = default_dir
    
    def ensure_available(self, configured_path: Optional[str] = None) -> str:
        """
        Ensure InferenceX is available at the specified path.
        
        If the path doesn't exist or is a placeholder value, automatically
        clone InferenceX from the repository.
        
        Args:
            configured_path: Configured path to InferenceX (may be None or placeholder)
            
        Returns:
            Valid path to InferenceX installation
            
        Raises:
            RuntimeError: If unable to clone or validate InferenceX
        """
        # Determine if path is a placeholder or not configured
        is_placeholder = self._is_placeholder(configured_path)
        
        # Check if configured path exists
        if not is_placeholder and configured_path and os.path.exists(configured_path):
            logger.debug(f"InferenceX found at: {configured_path}")
            return configured_path
        
        # Determine target directory
        target_dir = self.default_dir if is_placeholder or not configured_path else configured_path
        
        # Check if already exists at target directory
        if os.path.exists(target_dir):
            if self._validate_installation(target_dir):
                logger.info(f"InferenceX already exists at: {target_dir}")
                return target_dir
            else:
                logger.warning(f"Directory exists but doesn't appear to be InferenceX: {target_dir}")
                # Still return it, let downstream code handle validation
                return target_dir
        
        # Clone the repository
        return self._clone_repository(target_dir)
    
    def _is_placeholder(self, path: Optional[str]) -> bool:
        """Check if the path is a placeholder value."""
        if path is None:
            return True
        return path in PLACEHOLDER_VALUES
    
    def _validate_installation(self, path: str) -> bool:
        """
        Validate that the path contains a valid InferenceX installation.
        
        Args:
            path: Path to validate
            
        Returns:
            True if valid InferenceX installation
        """
        required_paths = [
            "benchmarks",  # InferenceX benchmark scripts directory
        ]
        
        for required in required_paths:
            if not os.path.exists(os.path.join(path, required)):
                return False
        
        return True
    
    def _clone_repository(self, target_dir: str) -> str:
        """
        Clone InferenceX repository.
        
        Args:
            target_dir: Directory to clone into
            
        Returns:
            Path to cloned repository
            
        Raises:
            RuntimeError: If clone fails
        """
        logger.info(f"InferenceX not found. Cloning from {self.repo_url}...")
        logger.info(f"Clone destination: {target_dir}")
        
        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(target_dir)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Clone the repository
            result = subprocess.run(
                ["git", "clone", self.repo_url, target_dir],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(f"Failed to clone InferenceX: {error_msg}")
                raise RuntimeError(f"git clone failed: {error_msg}")
            
            logger.info(f"Successfully cloned InferenceX to: {target_dir}")
            return target_dir
            
        except subprocess.TimeoutExpired:
            logger.error("InferenceX clone timed out after 5 minutes")
            raise RuntimeError("git clone timed out after 5 minutes")
        except FileNotFoundError:
            logger.error("git command not found. Please install git.")
            raise RuntimeError("git is not installed. Please install git first.")
        except OSError as e:
            logger.error(f"Failed to create directory or clone: {e}")
            raise RuntimeError(f"Failed to setup InferenceX: {e}")


# Module-level instance for convenience
_manager: Optional[InferenceXManager] = None


def get_manager() -> InferenceXManager:
    """Get the global InferenceX manager instance."""
    global _manager
    if _manager is None:
        _manager = InferenceXManager()
    return _manager


def ensure_inferencex_available(configured_path: Optional[str] = None) -> str:
    """
    Convenience function to ensure InferenceX is available.
    
    Args:
        configured_path: Configured path to InferenceX
        
    Returns:
        Valid path to InferenceX installation
    """
    return get_manager().ensure_available(configured_path)


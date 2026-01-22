###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Utility functions for Magpie.
"""

from .common import (
    get_updated_env,
    compile_hip,
)
from .gpu import (
    GPUVendor,
    GPUHardwareInfo,
    GPUConfig,
    GPUController,
    MultiGPUConfig,
    MultiGPUController,
    detect_gpu,
    get_gpu_info,
    get_gpu_count,
    list_gpus,
    load_gpu_config_from_dict,
    get_reset_after_benchmark,
)

__all__ = [
    "get_updated_env",
    "compile_hip",
    # Single GPU
    "GPUVendor",
    "GPUHardwareInfo",
    "GPUConfig",
    "GPUController",
    # Multi GPU
    "MultiGPUConfig",
    "MultiGPUController",
    # Functions
    "detect_gpu",
    "get_gpu_info",
    "get_gpu_count",
    "list_gpus",
    # Config loading
    "load_gpu_config_from_dict",
    "get_reset_after_benchmark",
]

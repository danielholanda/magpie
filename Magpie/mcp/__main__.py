#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Entry point for running Magpie MCP server as a module.

Usage:
    python -m Magpie.mcp
"""

from .server import mcp

if __name__ == "__main__":
    mcp.run()

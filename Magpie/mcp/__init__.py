"""
Magpie MCP (Model Context Protocol) Server.

This module provides MCP tools for GPU kernel evaluation that can be
used by AI agents and other MCP clients.

Available tools:
- hardware_spec: Get GPU hardware specifications
- analyze: Analyze kernel correctness and performance
- compare: Compare multiple kernels
- configure_gpu: Configure GPU power/frequency settings
"""

from .server import mcp

__all__ = ["mcp"]


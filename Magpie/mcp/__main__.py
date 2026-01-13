#!/usr/bin/env python3
"""
Entry point for running Magpie MCP server as a module.

Usage:
    python -m Magpie.mcp
"""

from .server import mcp

if __name__ == "__main__":
    mcp.run()


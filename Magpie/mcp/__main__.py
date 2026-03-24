#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Entry point for running Magpie MCP server as a module.

Usage:
    python -m Magpie.mcp                              # stdio (for MCP clients)
    python -m Magpie.mcp --transport streamable-http   # HTTP (for remote servers)
    python -m Magpie.mcp --transport sse               # SSE over HTTP

Environment variables:
    MAGPIE_HOST  - Bind host (default: 0.0.0.0)
    MAGPIE_PORT  - Bind port (default: 8000)
"""

import argparse

from .server import mcp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Magpie MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    args = parser.parse_args()

    mcp.run(transport=args.transport)

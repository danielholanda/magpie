#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Entry point for running Magpie MCP server as a module.

Usage:
    python -m Magpie.mcp                          # stdio (for MCP clients)
    python -m Magpie.mcp --transport sse           # SSE over HTTP
    python -m Magpie.mcp --transport streamable-http
    python -m Magpie.mcp --transport sse --host 0.0.0.0 --port 8000
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
    parser.add_argument("--host", type=str, help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Bind port (default: 8000)")
    args = parser.parse_args()

    if args.host:
        mcp.settings.host = args.host
        mcp.settings.transport_security = None
    if args.port:
        mcp.settings.port = args.port

    mcp.run(transport=args.transport)

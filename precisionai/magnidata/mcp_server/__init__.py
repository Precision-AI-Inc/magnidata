# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Read-only MCP server over MagniData's REST API, installed as the ``magnidata-mcp`` command."""

from precisionai.magnidata.mcp_server.client import DEFAULT_API_URL, DEMO_API_URL, MagniDataAPIError, MagniDataClient
from precisionai.magnidata.mcp_server.server import create_http_app, create_server, main

__all__ = [
    "DEFAULT_API_URL",
    "DEMO_API_URL",
    "MagniDataAPIError",
    "MagniDataClient",
    "create_http_app",
    "create_server",
    "main",
]

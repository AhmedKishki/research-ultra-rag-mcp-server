"""Shared stdio transport construction for the research MCP process."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastmcp.client.transports import StdioTransport

from .config import ResearchConfig


def create_research_transport(
    config: ResearchConfig,
    *,
    log_file: str | Path,
) -> StdioTransport:
    arguments = [
        "-m",
        "research_ultra_rag_mcp",
        "--project-root",
        str(config.project_root),
        "--source-directory",
        config.source_root.relative_to(config.project_root).as_posix(),
        "--vanilla-executable",
        str(config.vanilla_executable),
        "--model-cache-root",
        str(config.model_cache_root),
        "--dense-backend",
        config.dense_backend,
        "--log-level",
        config.log_level,
    ]
    if config.runtime_cache_root is not None:
        arguments.extend(["--runtime-cache-root", str(config.runtime_cache_root)])
    if config.runtime_root is not None:
        arguments.extend(["--runtime-root", str(config.runtime_root)])
    if config.offline:
        arguments.append("--offline")
    return StdioTransport(
        command=sys.executable,
        args=arguments,
        env=dict(os.environ),
        cwd=str(config.project_root),
        keep_alive=True,
        log_file=Path(log_file),
    )

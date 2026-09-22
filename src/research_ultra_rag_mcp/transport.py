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
    tool_detail: str | None = None,
) -> StdioTransport:
    """Start a private server for one research project.

    ``tool_detail`` defaults to the project's setting. The diagnostic consumers
    in this package — the browser UI, the terminal verifier, the bundle CLI, and
    the evaluation harness — pass ``full`` because they render, verify, or
    measure what an agent's lean answer deliberately leaves out.
    """

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
        "--tool-detail",
        tool_detail or config.tool_detail,
        "--log-level",
        config.log_level,
    ]
    if config.runtime_cache_root is not None:
        arguments.extend(["--runtime-cache-root", str(config.runtime_cache_root)])
    if config.runtime_root is not None:
        arguments.extend(["--runtime-root", str(config.runtime_root)])
    if config.embedding_threads is not None:
        arguments.extend(["--embedding-threads", str(config.embedding_threads)])
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

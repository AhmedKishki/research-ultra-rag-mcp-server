"""Shared stdio transport construction for the research MCP process."""

from __future__ import annotations

import sys
from pathlib import Path

from fastmcp.client.transports import StdioTransport

from .config import ResearchConfig, child_process_environment


def create_research_transport(
    config: ResearchConfig,
    *,
    log_file: str | Path,
    tool_detail: str | None = None,
) -> StdioTransport:
    """Start a private stdio server for one research project.

    ``tool_detail`` defaults to the project's setting; the diagnostic surfaces in
    this package pass ``full``.

    The child is a managed child: it is marked as one, it inherits no top-level UI
    setting, and it therefore cannot host a UI of its own.
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
        env=child_process_environment(),
        cwd=str(config.project_root),
        keep_alive=True,
        log_file=Path(log_file),
    )

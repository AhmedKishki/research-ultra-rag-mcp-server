"""Measure the lean agent-facing tool answers against the full-detail payload.

Every MCP tool answers with the lean projection from `tool_views.py` unless the
server runs with ``--tool-detail full``. This harness measures both, on one real
project, and reports the UTF-8 JSON size of each answer so the token budget is a
recorded number rather than an impression.

It starts one in-process server per tool and per detail mode, so it is a slow
manual measurement, not a test. It is read-only: `status`, `search`,
`list_sources`, and `get_passage` do not change a generation, and `list_sources`
only performs the idempotent source-ID registration it is documented to do.
`ingest` is deliberately not measured, because a measurement tool must never
build.

Usage:

    uv run python scripts/measure_tool_payloads.py --project /path/to/project
    uv run python scripts/measure_tool_payloads.py --project /path --offline \
        --query "your research question" --top-k 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.exceptions import ToolError

from research_ultra_rag_mcp.config import (
    FULL_TOOL_DETAIL,
    LEAN_TOOL_DETAIL,
    ResearchConfig,
    configured_source_directory,
    resolve_config,
)
from research_ultra_rag_mcp.server import create_server

DEFAULT_QUERY = "research evidence"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="measure_tool_payloads",
        description="Compare the lean and full-detail size of each tool answer.",
    )
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=f"Search query to measure (default: {DEFAULT_QUERY!r}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Passages to request from search (default: 6).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the vanilla runtime and model files to be cached already.",
    )
    parser.add_argument(
        "--model-cache-root",
        default=os.environ.get("RESEARCH_ULTRARAG_MODEL_CACHE_ROOT"),
    )
    parser.add_argument(
        "--runtime-root",
        default=os.environ.get("RESEARCH_ULTRARAG_RUNTIME_ROOT"),
        help="Absolute directory of the project's disposable derived state, if relocated.",
    )
    parser.add_argument(
        "--dense-backend",
        choices=("auto", "exact", "qdrant"),
        default=os.environ.get("RESEARCH_ULTRARAG_DENSE_BACKEND", "auto"),
    )
    return parser


def _payload_bytes(payload: Mapping[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


async def _answer(
    config: ResearchConfig,
    tool: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Call one tool against one in-process server and return its answer."""

    server = create_server(config)
    async with Client(server, timeout=1800, init_timeout=1800) as client:
        result = await client.call_tool(tool, arguments, timeout=1800)
    data = getattr(result, "data", result)
    if not isinstance(data, dict):
        raise TypeError(f"{tool} returned a non-object answer")
    return data


async def _measure(args: argparse.Namespace) -> int:
    project = args.project_root.expanduser().resolve()
    if not project.is_dir():
        raise RuntimeError(f"Project root is not a directory: {project}")
    if not 1 <= args.top_k <= 50:
        raise RuntimeError("--top-k must be between 1 and 50")

    base = resolve_config(
        project,
        source_directory=configured_source_directory(project),
        model_cache_root=args.model_cache_root,
        offline=args.offline,
        dense_backend=args.dense_backend,
        runtime_root=args.runtime_root,
    )
    calls: list[tuple[str, dict[str, Any]]] = [
        ("status", {}),
        ("list_sources", {}),
        ("search", {"query": args.query, "top_k": args.top_k}),
    ]

    print(f"project: {project}")
    print(f"generation: {base.project_name}")
    print(f"query: {args.query!r}  top_k={args.top_k}")
    print()
    print(f"{'tool':14s} {'lean bytes':>10s} {'full bytes':>10s} {'ratio':>6s}")
    failed = False
    for tool, arguments in calls:
        try:
            lean = await _answer(
                replace(base, tool_detail=LEAN_TOOL_DETAIL),
                tool,
                arguments,
            )
            full = await _answer(
                replace(base, tool_detail=FULL_TOOL_DETAIL),
                tool,
                arguments,
            )
        except (ToolError, OSError, RuntimeError, TypeError) as exc:
            # One tool that cannot answer must not hide the other rows: a
            # project without a generation still measures what it can.
            failed = True
            print(f"{tool:14s} {'-':>10s} {'-':>10s} {'-':>6s}  {exc}")
            continue
        lean_bytes = _payload_bytes(lean)
        full_bytes = _payload_bytes(full)
        print(
            f"{tool:14s} {lean_bytes:10d} {full_bytes:10d} "
            f"{lean_bytes / full_bytes:6.2f}"
        )
    return 1 if failed else 0


def main() -> None:
    args = _parser().parse_args()
    try:
        status = asyncio.run(_measure(args))
    except Exception as exc:
        raise SystemExit(f"Measurement failed: {exc}") from exc
    sys.exit(status)


if __name__ == "__main__":
    main()

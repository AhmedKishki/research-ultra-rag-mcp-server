"""Terminal verification client for one research project."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-ultra-rag-verify",
        description="Inspect, optionally ingest, and search a research MCP project.",
    )
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--query",
        default="research evidence",
        help="BM25 query used for the retrieval check.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Create and select a new generation before searching.",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require an already-installed vanilla UltraRAG runtime.",
    )
    return parser


async def _verify(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.expanduser().resolve()
    if not project.is_dir():
        raise RuntimeError(f"Project root is not a directory: {project}")
    if not 1 <= args.top_k <= 50:
        raise RuntimeError("--top-k must be between 1 and 50")

    executable = Path(sys.executable).parent / "research-ultra-rag-mcp"
    if not executable.is_file():
        raise RuntimeError(f"MCP executable was not found: {executable}")
    arguments = ["--project-root", str(project)]
    if args.offline:
        arguments.append("--offline")
    log_path = project / ".ultrarag" / "research" / "logs" / "verify-stderr.log"
    transport = StdioTransport(
        command=str(executable),
        args=arguments,
        log_file=log_path,
    )

    async with Client(transport, timeout=1800, init_timeout=1800) as client:
        tools = {tool.name for tool in await client.list_tools()}
        required = {"status", "ingest", "search"}
        if missing := required - tools:
            raise RuntimeError(f"MCP server is missing tools: {sorted(missing)}")

        before = (await client.call_tool("status", {})).data
        ingestion = None
        if args.ingest:
            ingestion = (
                await client.call_tool(
                    "ingest",
                    {
                        "chunk_size": args.chunk_size,
                        "chunk_overlap": args.chunk_overlap,
                    },
                    timeout=1800,
                )
            ).data
        elif not before.get("ready"):
            raise RuntimeError(
                "No knowledge base exists. Run again with --ingest to create one."
            )

        after = (await client.call_tool("status", {})).data
        search = (
            await client.call_tool(
                "search",
                {"query": args.query, "top_k": args.top_k},
                timeout=1800,
            )
        ).data
    return {
        "status": "passed",
        "project_root": str(project),
        "before": before,
        "ingestion": ingestion,
        "after": after,
        "search": search,
    }


def main() -> None:
    args = _parser().parse_args()
    try:
        result = asyncio.run(_verify(args))
    except Exception as exc:
        raise SystemExit(f"Verification failed: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

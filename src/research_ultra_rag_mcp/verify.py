"""Terminal verification client for one research project."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastmcp import Client

from .config import configured_source_directory, resolve_config
from .transport import create_research_transport


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-ultra-rag-verify",
        description="Inspect, optionally ingest, and search a research MCP project.",
    )
    parser.add_argument("project_root", type=Path)
    parser.add_argument(
        "--query",
        default="research evidence",
        help="Query used for the retrieval check.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--retrieval-method",
        choices=("hybrid", "bm25", "dense"),
        default="hybrid",
        help="Retrieval method to verify (default: hybrid).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Also verify the optional CPU cross-encoder reranker.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Create and select a new generation before searching.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="With --ingest, bypass all compatible document/chunk/vector reuse.",
    )
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the vanilla runtime and model files to be cached already.",
    )
    parser.add_argument(
        "--model-cache-root",
        default=os.environ.get("RESEARCH_ULTRARAG_MODEL_CACHE_ROOT"),
        help="Override the shared FastEmbed model cache.",
    )
    return parser


async def _verify(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.expanduser().resolve()
    if not project.is_dir():
        raise RuntimeError(f"Project root is not a directory: {project}")
    if not 1 <= args.top_k <= 50:
        raise RuntimeError("--top-k must be between 1 and 50")
    if args.force_recompute and not args.ingest:
        raise RuntimeError("--force-recompute requires --ingest")

    config = resolve_config(
        project,
        source_directory=configured_source_directory(project),
        model_cache_root=args.model_cache_root,
        offline=args.offline,
    )
    log_path = project / ".ultrarag" / "research" / "logs" / "verify-stderr.log"
    transport = create_research_transport(
        config,
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
                        "force_recompute": args.force_recompute,
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
                {
                    "query": args.query,
                    "top_k": args.top_k,
                    "retrieval_method": args.retrieval_method,
                    "rerank": args.rerank,
                },
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

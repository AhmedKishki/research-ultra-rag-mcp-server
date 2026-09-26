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
from .rerankers import RERANKER_MODEL_CHOICES
from .settings import (
    FULL_TOOL_DETAIL,
    MAXIMUM_WORK_BUDGET_SECONDS,
    describe_settings,
)
from .transport import create_research_transport

# How long the verifier waits for one tool call, for the same reason the UI does:
# it has to outlast the longest an ingest call may be told to run.
VERIFY_TOOL_TIMEOUT_SECONDS = MAXIMUM_WORK_BUDGET_SECONDS + 600


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
        "--ingest",
        action="store_true",
        help="Create and select a new generation before searching.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="With --ingest, bypass all compatible document/chunk/vector reuse.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the vanilla runtime and model files to be cached already.",
    )
    parser.add_argument(
        "--model-cache-root",
        default=None,
        help="Override the shared FastEmbed model cache.",
    )
    parser.add_argument(
        "--runtime-root",
        default=os.environ.get("RESEARCH_ULTRARAG_RUNTIME_ROOT"),
        help=(
            "Absolute directory for disposable derived state; use it when the "
            "project lives on slow storage."
        ),
    )
    parser.add_argument(
        "--embedding-threads",
        type=int,
        default=None,
        help=(
            "ONNX Runtime threads for the embedding model; unset lets the "
            "runtime decide"
        ),
    )
    parser.add_argument(
        "--dense-backend",
        choices=("auto", "exact", "qdrant"),
        default=None,
        help=(
            "Dense index backend for a new generation (default: auto, which is "
            "an exact scan below the documented corpus threshold)."
        ),
    )
    parser.add_argument(
        "--reranker-model",
        choices=RERANKER_MODEL_CHOICES,
        default=None,
        help=(
            "Reranker model the engine loads for every search; each choice is "
            "pinned to a revision."
        ),
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("RESEARCH_ULTRARAG_CONFIG"),
        help=(
            "Settings file layered between the project config and the "
            "environment. Omit to use only the packaged default, the per-user "
            "file, and the project's own config.toml."
        ),
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Override one setting for this invocation; repeat for more. "
            "--print-config lists every key."
        ),
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help=(
            "Print the merged settings with the layer each value came from, then exit."
        ),
    )
    return parser


async def _ingest_until_complete(
    client: Any,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    while True:
        ingestion = (
            await client.call_tool(
                "ingest",
                arguments,
                timeout=VERIFY_TOOL_TIMEOUT_SECONDS,
            )
        ).data
        if ingestion.get("status") != "in_progress":
            return ingestion


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
        dense_backend=args.dense_backend,
        runtime_root=args.runtime_root,
        embedding_threads=args.embedding_threads,
        reranker_model=args.reranker_model,
        config_path=args.config,
        settings_overrides=args.set_overrides,
    )
    if args.print_config:
        print(describe_settings(config.settings, config.settings_provenance))
        return None
    log_path = config.logs_root / "verify-stderr.log"
    transport = create_research_transport(
        config,
        log_file=log_path,
        # The verifier prints the complete payload for a human to inspect.
        tool_detail=FULL_TOOL_DETAIL,
    )

    async with Client(
        transport,
        timeout=VERIFY_TOOL_TIMEOUT_SECONDS,
        init_timeout=VERIFY_TOOL_TIMEOUT_SECONDS,
    ) as client:
        tools = {tool.name for tool in await client.list_tools()}
        required = {"status", "ingest", "search"}
        if missing := required - tools:
            raise RuntimeError(f"MCP server is missing tools: {sorted(missing)}")

        before = (await client.call_tool("status", {})).data
        ingestion = None
        if args.ingest:
            ingestion = await _ingest_until_complete(
                client,
                {"force_recompute": args.force_recompute},
            )
        elif not before.get("ready"):
            raise RuntimeError(
                "No knowledge base exists. Run again with --ingest to create one."
            )

        after = (await client.call_tool("status", {})).data
        search = (
            await client.call_tool(
                "search",
                {"query": args.query, "top_k": args.top_k},
                timeout=VERIFY_TOOL_TIMEOUT_SECONDS,
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
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

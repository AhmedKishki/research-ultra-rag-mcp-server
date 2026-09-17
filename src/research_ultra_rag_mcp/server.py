"""Project-scoped stdio MCP server for research knowledge bases."""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal, TypeVar

from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from .config import ConfigurationError, ResearchConfig, resolve_config
from .instructions import SERVER_INSTRUCTIONS
from .service import ResearchError, ResearchService
from .ultrarag import VanillaUltraRAG, create_vanilla_transport

SERVER_NAME = "research-ultra-rag-mcp"
SERVER_VERSION = "0.2.0"
T = TypeVar("T")


async def _tool_call(operation: Callable[[], Awaitable[T]]) -> T:
    try:
        return await operation()
    except ResearchError as exc:
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        raise ToolError(f"Research workflow failed: {exc}") from exc


def create_server(config: ResearchConfig) -> FastMCP[Any]:
    holder: dict[str, ResearchService] = {}

    @asynccontextmanager
    async def lifespan(_: FastMCP[Any]) -> AsyncIterator[dict[str, Any]]:
        transport = create_vanilla_transport(config)
        async with Client(
            transport,
            name=SERVER_NAME,
            timeout=1800,
            init_timeout=1800,
        ) as client:
            holder["service"] = ResearchService(config, VanillaUltraRAG(client))
            try:
                yield {}
            finally:
                holder.clear()

    app = FastMCP(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        instructions=SERVER_INSTRUCTIONS,
        lifespan=lifespan,
    )

    def service() -> ResearchService:
        instance = holder.get("service")
        if instance is None:
            raise ToolError("Research server is not initialized")
        return instance

    @app.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def status() -> dict[str, Any]:
        """Report source selection, current generation, and whether it is stale."""
        return await _tool_call(service().status)

    @app.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        }
    )
    async def ingest(
        chunk_size: int = 384,
        chunk_overlap: int = 64,
    ) -> dict[str, Any]:
        """Extract PDFs/EPUBs and create a BM25 plus dense generation.

        Markdown and all other formats are ignored. chunk_size is measured in
        GPT-2 tokens and is capped at 384 for the embedding model. Existing
        generations are retained; current changes only after both indexes pass.
        """

        return await _tool_call(
            lambda: service().ingest(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    @app.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def search(
        query: str,
        top_k: int = 8,
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        document_ids: list[str] | None = None,
        retrieval_method: Literal["hybrid", "bm25", "dense"] = "hybrid",
        rerank: bool = False,
    ) -> dict[str, Any]:
        """Search the current generation and return citable evidence.

        Optional filters require every requested category or keyword to be
        present. Hybrid is the default; BM25 and dense retrieval can be inspected
        separately. Optional CPU reranking is slower and lazily loads another
        local model. Returned passages include provenance and component ranks.
        """

        return await _tool_call(
            lambda: service().search(
                query,
                top_k=top_k,
                categories=categories,
                keywords=keywords,
                document_ids=document_ids,
                retrieval_method=retrieval_method,
                rerank=rerank,
            )
        )

    @app.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def list_sources(
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        """List indexed sources and their bibliographic metadata."""
        return await _tool_call(
            lambda: service().list_sources(
                categories=categories,
                keywords=keywords,
            )
        )

    @app.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def get_passage(
        chunk_id: str,
        context_chunks: int = 1,
    ) -> dict[str, Any]:
        """Return one retrieved passage with nearby chunks from the same source."""
        return await _tool_call(
            lambda: service().get_passage(
                chunk_id,
                context_chunks=context_chunks,
            )
        )

    @app.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def set_source_metadata(
        source_path: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Set reviewed metadata for one source-relative PDF or EPUB path.

        Supported fields are title, authors, year, doi, categories, and
        keywords. Run ingest afterward to create a generation using the update.
        """

        return await _tool_call(
            lambda: service().set_source_metadata(source_path, metadata)
        )

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=SERVER_NAME,
        description=(
            "Serve one project-scoped PDF/EPUB research knowledge base over stdio MCP."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("RESEARCH_ULTRARAG_PROJECT_ROOT"),
        required=os.environ.get("RESEARCH_ULTRARAG_PROJECT_ROOT") is None,
        help="Absolute research project root containing the sources directory.",
    )
    parser.add_argument(
        "--source-directory",
        default=os.environ.get("RESEARCH_ULTRARAG_SOURCE_DIRECTORY", "sources"),
        help="Project-relative source directory (default: sources).",
    )
    parser.add_argument(
        "--vanilla-executable",
        default=os.environ.get("VANILLA_ULTRARAG_MCP_EXECUTABLE"),
        help="Override the vanilla-ultra-rag-mcp executable.",
    )
    parser.add_argument(
        "--runtime-cache-root",
        default=os.environ.get("VANILLA_ULTRARAG_CACHE_ROOT"),
        help="Override the vanilla gateway's managed UltraRAG cache.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Require an installed vanilla runtime and already-cached embedding "
            "or reranker models."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warn", "error"),
        default="warn",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        config = resolve_config(
            args.project_root,
            source_directory=args.source_directory,
            vanilla_executable=args.vanilla_executable,
            runtime_cache_root=args.runtime_cache_root,
            offline=args.offline,
            log_level=args.log_level,
        )
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
    create_server(config).run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()

"""Project-scoped stdio MCP server for research knowledge bases."""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, NotRequired, TypeAlias, TypeVar

from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ConfigDict, Field
from typing_extensions import TypedDict

from .config import ConfigurationError, ResearchConfig, resolve_config
from .instructions import SERVER_INSTRUCTIONS
from .service import ResearchError, ResearchService
from .ultrarag import VanillaUltraRAG, create_vanilla_transport

SERVER_NAME = "research-ultra-rag-mcp"
SERVER_VERSION = "0.6.0"
T = TypeVar("T")


class SourceMetadataInput(TypedDict):
    """Reviewed bibliographic metadata accepted by set_source_metadata."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    title: NotRequired[
        Annotated[str, Field(description="Reviewed title of the source document.")]
    ]
    authors: NotRequired[
        Annotated[
            list[str],
            Field(description="Reviewed author names in preferred citation order."),
        ]
    ]
    year: NotRequired[
        Annotated[
            int | None,
            Field(
                description="Publication year from 1 through 9999, or null if unknown.",
                ge=1,
                le=9999,
            ),
        ]
    ]
    doi: NotRequired[
        Annotated[
            str,
            Field(
                description="Reviewed DOI, normally without a https://doi.org/ prefix."
            ),
        ]
    ]
    categories: NotRequired[
        Annotated[
            list[str],
            Field(description="Reviewed broad categories used to filter this source."),
        ]
    ]
    keywords: NotRequired[
        Annotated[
            list[str],
            Field(description="Reviewed specific keywords used to filter this source."),
        ]
    ]


ChunkSize: TypeAlias = Annotated[
    int,
    Field(
        description=(
            "Maximum GPT-2 token count per chunk; must be between 50 and 384."
        ),
        ge=50,
        le=384,
    ),
]
ChunkOverlap: TypeAlias = Annotated[
    int,
    Field(
        description=(
            "GPT-2 tokens repeated between adjacent chunks; must be non-negative "
            "and smaller than chunk_size."
        ),
        ge=0,
        le=383,
    ),
]
SearchQuery: TypeAlias = Annotated[
    str,
    Field(
        description="Research question, exact phrase, name, or concept to retrieve.",
        min_length=1,
    ),
]
TopK: TypeAlias = Annotated[
    int,
    Field(
        description="Maximum number of ranked evidence passages to return (1-50).",
        ge=1,
        le=50,
    ),
]
CategoryFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive category filters; a result must contain every supplied "
            "category. Omit or pass null for no category filter."
        )
    ),
]
KeywordFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive keyword filters; a result must contain every supplied "
            "keyword. Omit or pass null for no keyword filter."
        )
    ),
]
DocumentIdFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Document IDs to include; a result may match any supplied ID. Obtain IDs "
            "from list_sources or search. Omit or pass null for no document filter."
        )
    ),
]
RetrievalMethod: TypeAlias = Annotated[
    Literal["hybrid", "bm25", "dense"],
    Field(
        description=(
            "Retrieval mode: hybrid combines BM25 and dense results, bm25 favors "
            "exact terms, and dense favors semantic similarity."
        )
    ),
]
Rerank: TypeAlias = Annotated[
    bool,
    Field(
        description=(
            "Whether to apply the optional CPU cross-encoder reranker. This is slower "
            "and may download its pinned model on first use."
        )
    ),
]
ChunkId: TypeAlias = Annotated[
    str,
    Field(
        description="Exact chunk_id returned by search for the current generation.",
        min_length=1,
    ),
]
ContextChunks: TypeAlias = Annotated[
    int,
    Field(
        description=(
            "Number of neighboring chunks to return on each side of the requested "
            "passage (0-5)."
        ),
        ge=0,
        le=5,
    ),
]
SourcePath: TypeAlias = Annotated[
    str,
    Field(
        description=(
            "PDF or EPUB path relative to the configured sources directory, as shown "
            "by status or list_sources; absolute and escaping paths are rejected."
        ),
        min_length=1,
    ),
]
ReviewedMetadata: TypeAlias = Annotated[
    SourceMetadataInput,
    Field(
        description=(
            "Complete reviewed metadata override for the source. Supported fields are "
            "title, authors, year, doi, categories, and keywords; omitted fields remove "
            "their previous overrides."
        )
    ),
]
InclusionFlag: TypeAlias = Annotated[
    bool,
    Field(
        description=(
            "Set false to exclude the source from retrieval and future ingestion; "
            "set true to restore it. The original file is never changed."
        )
    ),
]
ExclusionReason: TypeAlias = Annotated[
    str | None,
    Field(
        description=(
            "Human-readable reason for the decision, such as identifying another "
            "file as the preferred copy. Required when included is false; omit or "
            "pass null when restoring a source."
        )
    ),
]


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
        chunk_size: ChunkSize = 384,
        chunk_overlap: ChunkOverlap = 64,
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
        query: SearchQuery,
        top_k: TopK = 8,
        categories: CategoryFilter = None,
        keywords: KeywordFilter = None,
        document_ids: DocumentIdFilter = None,
        retrieval_method: RetrievalMethod = "hybrid",
        rerank: Rerank = False,
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
        categories: CategoryFilter = None,
        keywords: KeywordFilter = None,
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
        chunk_id: ChunkId,
        context_chunks: ContextChunks = 1,
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
        source_path: SourcePath,
        metadata: ReviewedMetadata,
    ) -> dict[str, Any]:
        """Set reviewed metadata for one source-relative PDF or EPUB path.

        Supported fields are title, authors, year, doi, categories, and
        keywords. Run ingest afterward to create a generation using the update.
        """

        return await _tool_call(
            lambda: service().set_source_metadata(source_path, metadata)
        )

    @app.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def set_source_inclusion(
        source_path: SourcePath,
        included: InclusionFlag,
        reason: ExclusionReason = None,
    ) -> dict[str, Any]:
        """Include or exclude a PDF/EPUB from the project knowledge base.

        Use this after the agent or user has reviewed a source, including when a
        duplicate representation was found. Exclusion immediately blocks current
        search, source listing, and passage lookup, and future ingestion skips the
        source. The source file is never deleted or modified. Inclusion is
        reversible; re-ingest if the current generation does not contain it.
        """

        return await _tool_call(
            lambda: service().set_source_inclusion(
                source_path,
                included=included,
                reason=reason,
            )
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

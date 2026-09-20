"""Project-scoped stdio MCP server for research knowledge bases."""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, NotRequired, TypeAlias, TypedDict, TypeVar

from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ConfigDict, Field

from .config import (
    ConfigurationError,
    ResearchConfig,
    configured_source_directory,
    resolve_config,
)
from .instructions import SERVER_INSTRUCTIONS
from .service import ResearchError, ResearchService
from .ultrarag import VanillaUltraRAG, create_vanilla_transport

SERVER_NAME = "research-ultra-rag-mcp"
SERVER_VERSION = "0.11.0"
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
        description=(
            "Maximum total number of ranked evidence passages to return (1-50). "
            "This remains a passage budget in both passage and reference views."
        ),
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
ResultView: TypeAlias = Annotated[
    Literal["passages", "references"],
    Field(
        description=(
            "Response view: passages preserves the flat ranked passage list; "
            "references additionally groups the selected best passages by source "
            "reference without changing the total top_k passage budget."
        )
    ),
]
PassagesPerReference: TypeAlias = Annotated[
    int,
    Field(
        description=(
            "Maximum passages selected from one source reference in references "
            "view (1-5). The overall response still returns at most top_k passages."
        ),
        ge=1,
        le=5,
    ),
]
Rerank: TypeAlias = Annotated[
    bool,
    Field(
        description=(
            "Whether to apply the CPU cross-encoder reranker. It is on by default "
            "because it is the largest measured retrieval-quality gain; set it false "
            "for the lowest latency or to skip loading its model. When the pinned "
            "model cannot be loaded, the search returns the unranked candidate order "
            "and reports rerank_fallback."
        )
    ),
]
IncludeStaleness: TypeAlias = Annotated[
    bool,
    Field(
        description=(
            "Whether to check whether the selected generation is stale. "
            "Checking walks the source directory, so its cost grows with the "
            "collection. When false, the response reports stale=null and "
            "staleness_checked=false instead of a verdict."
        )
    ),
]
ForceRecompute: TypeAlias = Annotated[
    bool,
    Field(
        description=(
            "Set true to bypass document, chunk, and vector reuse and rebuild all "
            "derived content. The previous generation remains selected on failure."
        )
    ),
]
WorkBudgetSeconds: TypeAlias = Annotated[
    int,
    Field(
        description=(
            "Soft per-call ingestion work budget in seconds. Call ingest again "
            "when it returns status='in_progress'."
        ),
        ge=10,
        le=300,
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
            "PDF or EPUB path relative to the configured sources directory. Use "
            "list_sources.source_relative_path; absolute and escaping paths are "
            "rejected."
        ),
        min_length=1,
    ),
]
MetadataSourceId: TypeAlias = Annotated[
    str | None,
    Field(
        description=(
            "Stable project-scoped source_id returned by list_sources or search. "
            "Provide exactly one of source_id or source_path."
        ),
        min_length=1,
    ),
]
MetadataSourcePath: TypeAlias = Annotated[
    SourcePath | None,
    Field(
        description=(
            "Compatibility selector using list_sources.source_relative_path. "
            "Provide exactly one of source_id or source_path."
        )
    ),
]
ReviewedMetadata: TypeAlias = Annotated[
    SourceMetadataInput,
    Field(
        description=(
            "Complete reviewed metadata override for the source. Supported fields are "
            "title, authors, year, doi, categories, and keywords; omitted fields remove "
            "their previous overrides and an empty object restores all automatic "
            "values. Explicit empty values clear an automatically extracted field. "
            "For a source in the selected generation, the new values apply "
            "immediately without ingestion."
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
BundleName: TypeAlias = Annotated[
    str,
    Field(
        description=(
            "Filename of a .research-rag.zip archive already placed directly in "
            "the project's .research-rag/bundles directory. Paths are rejected."
        ),
        min_length=1,
    ),
]
ActivateImport: TypeAlias = Annotated[
    bool,
    Field(
        description=(
            "Whether to select the imported generation after all validation and "
            "local index reconstruction succeeds."
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


def create_server(
    config: ResearchConfig,
    *,
    ui_port: int | None = None,
) -> FastMCP[Any]:
    holder: dict[str, ResearchService] = {}
    ui_holder: dict[str, Any] = {}

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
            if ui_port is not None:
                # Imported lazily so a server that serves no UI never loads the
                # browser stack (uvicorn and starlette).
                from .ui import EmbeddedUi

                embedded = EmbeddedUi(config, port=ui_port)
                await embedded.start()
                ui_holder["ui"] = embedded
            try:
                yield {}
            finally:
                active_ui = ui_holder.pop("ui", None)
                if active_ui is not None:
                    await active_ui.stop()
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
        """Report project identity, staleness, required generation upgrades, and retained generations.

        generations lists every generation still on disk with its creation time,
        chunk and document counts, file count, and size, marks the one that is
        current, and reports the retained count and total bytes. A generation
        whose manifest is unreadable is reported with manifest_error instead of
        failing the call. Pruning is not offered; this only shows what a prune
        would consider.

        When this server was started with --ui-port, ui_url names the browser UI
        it is serving on loopback, ui_ready says whether it finished starting,
        and ui_error explains a port that was already in use. Without that
        option all three are null, false, and null, and no UI is running.
        """
        payload = await _tool_call(service().status)
        embedded = ui_holder.get("ui")
        if embedded is None:
            return {**payload, "ui_url": None, "ui_ready": False, "ui_error": None}
        return {
            **payload,
            "ui_url": embedded.url,
            "ui_ready": embedded.ready,
            "ui_error": embedded.error,
        }

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
        force_recompute: ForceRecompute = False,
        work_budget_seconds: WorkBudgetSeconds = 45,
    ) -> dict[str, Any]:
        """Create or refresh an immutable BM25 plus dense generation.

        Markdown and all other formats are ignored. chunk_size is measured in
        GPT-2 tokens and is capped at 384 for the embedding model. Compatible
        unchanged documents, chunks, and vectors are reused unless force_recompute
        is true. Complete BM25 and Qdrant indexes are still built for every changed
        generation. Work is checkpointed between bounded batches. If the result
        status is in_progress, call ingest again with the same settings. Existing
        generations are retained, and current changes only after both indexes pass
        verification.
        """

        return await _tool_call(
            lambda: service().ingest(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                force_recompute=force_recompute,
                work_budget_seconds=work_budget_seconds,
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
        result_view: ResultView = "passages",
        passages_per_reference: PassagesPerReference = 2,
        categories: CategoryFilter = None,
        keywords: KeywordFilter = None,
        document_ids: DocumentIdFilter = None,
        retrieval_method: RetrievalMethod = "hybrid",
        rerank: Rerank = True,
        include_staleness: IncludeStaleness = True,
    ) -> dict[str, Any]:
        """Search the current generation and return cleaned semantic evidence.

        Optional filters require every requested category or keyword to be
        present. Hybrid is the default; BM25 and dense retrieval can be inspected
        separately. CPU reranking is on by default because it is the largest
        measured quality gain, and it is slower and lazily loads another local
        model; set rerank=false to skip it. When that model cannot be loaded the
        unranked candidate order is returned and rerank_fallback explains why.
        The default passage view preserves the flat ranking. The
        reference view groups selected passages by source and caps passages from
        each reference while top_k remains the total passage budget. Results may
        be fewer than top_k when relevance gates reject weak candidates. Set
        include_staleness=false to skip the source-directory walk that produces
        the stale verdict; stale is then null. Returned
        text is not safe for direct quotation; use the original source path and
        locator.
        """

        return await _tool_call(
            lambda: service().search(
                query,
                top_k=top_k,
                result_view=result_view,
                passages_per_reference=passages_per_reference,
                categories=categories,
                keywords=keywords,
                document_ids=document_ids,
                retrieval_method=retrieval_method,
                rerank=rerank,
                include_staleness=include_staleness,
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
    async def list_sources(
        categories: CategoryFilter = None,
        keywords: KeywordFilter = None,
    ) -> dict[str, Any]:
        """List stable source IDs, indexed metadata, and saved overrides.

        This works before ingestion and durably registers each discovered
        source ID in the project catalog. ``known_sources`` keeps registered
        IDs addressable when originals are temporarily absent, while
        ``reviewed_metadata_sources`` makes every saved override inspectable.
        """
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
        """Return cleaned semantic context near one retrieved passage.

        Context preserves source and page/section provenance but is not safe for
        direct quotation. Open the original PDF or EPUB for exact wording.
        """
        return await _tool_call(
            lambda: service().get_passage(
                chunk_id,
                context_chunks=context_chunks,
            )
        )

    @app.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def set_source_metadata(
        metadata: ReviewedMetadata,
        source_id: MetadataSourceId = None,
        source_path: MetadataSourcePath = None,
    ) -> dict[str, Any]:
        """Set reviewed metadata for one identified PDF or EPUB source.

        Supported fields are title, authors, year, doi, categories, and
        keywords. For a source already in the selected generation, the update
        immediately affects source listings, search filters and results,
        citations, and neighboring passages without rebuilding the immutable
        indexes. Ingestion is required only when the source is absent from the
        selected generation. Identify the source with exactly one of source_id
        or source_path; the path form remains available for compatibility and
        uses list_sources.source_relative_path. Omitted metadata fields remove
        their previous overrides, an empty object restores all automatic
        values, and explicit empty values clear an automatically extracted
        field.
        """

        return await _tool_call(
            lambda: service().set_source_metadata(
                metadata=metadata,
                source_id=source_id,
                source_path=source_path,
            )
        )

    @app.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def set_source_inclusion(
        included: InclusionFlag,
        source_id: MetadataSourceId = None,
        source_path: MetadataSourcePath = None,
        reason: ExclusionReason = None,
    ) -> dict[str, Any]:
        """Include or exclude a PDF/EPUB from the project knowledge base.

        Use this after the agent or user has reviewed a source, including when a
        duplicate representation was found. Exclusion immediately blocks current
        search, source listing, and passage lookup, and future ingestion skips the
        source. The source file is never deleted or modified. Inclusion is
        reversible; re-ingest if the current generation does not contain it.
        Identify the source with exactly one of source_id or source_path.
        """

        return await _tool_call(
            lambda: service().set_source_inclusion(
                source_path=source_path,
                source_id=source_id,
                included=included,
                reason=reason,
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
    async def export_bundle() -> dict[str, Any]:
        """Export the fresh current generation and all original sources.

        The validated archive is written beneath .research-rag/bundles and is
        accompanied by a SHA-256 sidecar. Export is refused when the generation
        is stale or requires an upgrade. The user is responsible for the right
        to redistribute every included PDF and EPUB.
        """

        return await _tool_call(service().export_bundle)

    @app.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def import_bundle(
        bundle_name: BundleName,
        activate: ActivateImport = True,
    ) -> dict[str, Any]:
        """Validate a project bundle and reconstruct local BM25/Qdrant indexes.

        Existing source files are accepted only when their SHA-256 matches the
        bundled original; conflicting files are never overwritten. Bundled
        reviewed metadata and exclusions replace the project's portable copies.
        The current generation pointer changes last and only when activate is
        true.
        """

        return await _tool_call(
            lambda: service().import_bundle(bundle_name, activate=activate)
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
        default=os.environ.get("RESEARCH_ULTRARAG_SOURCE_DIRECTORY"),
        help=(
            "Project-relative source directory. Omit to reuse project.json, or "
            "use 'sources' for an uninitialized project."
        ),
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
        "--model-cache-root",
        default=os.environ.get("RESEARCH_ULTRARAG_MODEL_CACHE_ROOT"),
        help=(
            "Shared FastEmbed model cache (default: "
            "~/.cache/research-ultra-rag-mcp/models)."
        ),
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
        "--runtime-root",
        default=os.environ.get("RESEARCH_ULTRARAG_RUNTIME_ROOT"),
        help=(
            "Absolute directory for disposable derived state (generations, "
            "staging, logs). Defaults to <project>/.research-rag/runtime; point "
            "it at fast local storage when the project itself is on a slow disk. "
            "A relocated root is claimed by this project and refuses to be shared."
        ),
    )
    parser.add_argument(
        "--embedding-threads",
        type=int,
        default=os.environ.get("RESEARCH_ULTRARAG_EMBEDDING_THREADS"),
        help=(
            "ONNX Runtime threads for the embedding model; unset lets the "
            "runtime decide"
        ),
    )
    parser.add_argument(
        "--dense-backend",
        choices=("auto", "exact", "qdrant"),
        default=os.environ.get("RESEARCH_ULTRARAG_DENSE_BACKEND", "auto"),
        help=(
            "Dense index backend for new generations. 'auto' uses an exact scan "
            "over the portable vectors below the documented corpus threshold and "
            "the embedded ANN index above it."
        ),
    )
    parser.add_argument(
        "--ui-port",
        default=os.environ.get("RESEARCH_ULTRARAG_UI_PORT"),
        help=(
            "Also serve the local browser UI on this loopback port for the life "
            "of this server, reusing the resolved project, runtime, model, and "
            "offline settings. Omit to serve no UI."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warn", "error"),
        default="warn",
    )
    return parser


def _ui_port_from(raw: Any) -> int | None:
    """Normalize --ui-port, which may arrive from the environment as text."""

    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        port = int(raw)
    except (TypeError, ValueError):
        raise SystemExit("--ui-port must be an integer between 1 and 65535") from None
    if not 1 <= port <= 65535:
        raise SystemExit("--ui-port must be between 1 and 65535")
    return port


def main() -> None:
    args = _parser().parse_args()
    try:
        source_directory = args.source_directory or configured_source_directory(
            args.project_root
        )
        config = resolve_config(
            args.project_root,
            source_directory=source_directory,
            vanilla_executable=args.vanilla_executable,
            runtime_cache_root=args.runtime_cache_root,
            model_cache_root=args.model_cache_root,
            offline=args.offline,
            log_level=args.log_level,
            dense_backend=args.dense_backend,
            runtime_root=args.runtime_root,
            embedding_threads=args.embedding_threads,
        )
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
    create_server(config, ui_port=_ui_port_from(args.ui_port)).run(
        transport="stdio", show_banner=False
    )


if __name__ == "__main__":
    main()

"""Project-scoped stdio MCP server for research knowledge bases."""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, TypeAlias, TypeVar

from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from .config import (
    LEAN_TOOL_DETAIL,
    TOOL_DETAIL_MODES,
    ConfigurationError,
    ResearchConfig,
    configured_source_directory,
    is_managed_child,
    resolve_config,
)
from .instructions import SERVER_INSTRUCTIONS
from .service import ResearchError, ResearchService
from .tool_views import present_tool_response
from .ultrarag import VanillaUltraRAG, create_vanilla_transport
from .version import SERVER_VERSION

SERVER_NAME = "research-ultra-rag-mcp"
T = TypeVar("T")


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
KeywordFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive keyword filters; a result must contain every supplied "
            "keyword. Omit or pass null for no keyword filter."
        )
    ),
]
SourceIdFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Stable source IDs to include; a result may match any supplied ID. "
            "Obtain IDs from list_sources. A source ID survives a change to the "
            "file's bytes and changes when the file is renamed or moved. Omit or "
            "pass null to search every source."
        )
    ),
]
ExcludeSourceIdFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Stable source IDs to exclude; a result may not match any supplied ID. "
            "Omit or pass null to exclude nothing. Reviewed source exclusions "
            "always apply and cannot be undone here."
        )
    ),
]
CategoriesAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive 'any of' category filters; a result must contain at "
            "least one supplied category. Use it to search a set of corpus "
            "partitions in one call, and combine it with `categories` to require "
            "all of one set and any of another. Categories come from reviewed "
            "source metadata and `status.categories` lists the current inventory. "
            "Omit or pass null for no filter."
        )
    ),
]
ProjectsAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Reviewed project tags to match with 'any of' semantics; a result must "
            "carry at least one supplied project. Omit or pass null for no filter."
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
ChunkId: TypeAlias = Annotated[
    str,
    Field(
        description="Exact chunk_id returned by search for the current generation.",
        min_length=1,
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

    def _present(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a tool answer in this server's configured detail mode."""

        try:
            return present_tool_response(
                operation,
                payload,
                detail=config.tool_detail,
            )
        except ResearchError as exc:
            raise ToolError(str(exc)) from exc

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
        chunk and document counts, and size, marks the one that is
        current, and reports the retained count and total bytes; the full-detail
        payload adds each generation's file count and schema. A generation
        whose manifest is unreadable is reported with manifest_error instead of
        failing the call. Pruning is not offered; this only shows what a prune
        would consider.

        When the generation is stale, changes counts the added and modified
        sources, names the sources the generation has that the source directory
        no longer does, and reports whether reviewed metadata or exclusions
        moved. Available sources are counted rather than listed; list_sources
        is the inventory.

        When this server was started with --ui-port, ui_url names the browser UI
        it is serving on loopback, ui_ready says whether it finished starting,
        and ui_error explains a port it could not claim. Without that
        option all three are null, false, and null, and no UI is running.
        """
        payload = _present("status", await _tool_call(service().status))
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
        force_recompute: ForceRecompute = False,
    ) -> dict[str, Any]:
        """Create or refresh an immutable BM25 plus dense generation.

        Markdown and all other formats are ignored. Compatible unchanged
        documents, chunks, and vectors are reused unless force_recompute is true.
        Chunking uses this server's fixed settings. Work is checkpointed between
        bounded batches: if the result status is in_progress, call ingest again
        until it returns ready or unchanged. Existing generations are retained,
        and current changes only after both indexes pass verification.
        """

        return _present(
            "ingest",
            await _tool_call(lambda: service().ingest(force_recompute=force_recompute)),
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
        categories_any: CategoriesAnyFilter = None,
        projects_any: ProjectsAnyFilter = None,
        keywords: KeywordFilter = None,
        source_ids: SourceIdFilter = None,
        exclude_source_ids: ExcludeSourceIdFilter = None,
    ) -> dict[str, Any]:
        """Search the current generation and return cleaned semantic evidence.

        Retrieval is hybrid with CPU reranking, which is the largest measured
        quality gain, and every answer reports whether the reranker really ran;
        when its model cannot be loaded the unranked candidate order is returned
        and `rerank_fallback` explains why. A generation that predates dense
        support must be re-ingested first, and `status` says so.

        Optional filters narrow the corpus before ranking, so top_k is a budget
        inside the selection: `projects_any` and `categories_any` keep a result
        that carries at least one of the listed project tags or categories,
        `keywords` requires every listed keyword, `source_ids` includes named
        sources, and `exclude_source_ids` removes them. A source ID survives a
        change to a file's bytes and changes when the file is renamed or moved;
        the filename from list_sources works as well. Reviewed source exclusions
        always win, an unresolved caller-supplied ID is reported as
        `unresolved_source_ids` or `unresolved_exclude_source_ids`, and an
        include list that resolves to nothing is an error rather than an
        unfiltered result.

        Results may be fewer than top_k when relevance gates reject weak
        candidates. The answer names the source of every passage by filename,
        title, and authors, with a ready-to-use citation. Returned text is not
        safe for direct quotation; open the original at the returned locator.
        """

        return _present(
            "search",
            await _tool_call(
                lambda: service().search(
                    query,
                    top_k=top_k,
                    categories_any=categories_any,
                    projects_any=projects_any,
                    keywords=keywords,
                    source_ids=source_ids,
                    exclude_source_ids=exclude_source_ids,
                    retrieval_method="hybrid",
                    rerank=True,
                    include_staleness=True,
                )
            ),
        )

    @app.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
    )
    async def list_sources() -> dict[str, Any]:
        """List stable source IDs, inclusion state, and saved metadata overrides.

        This is the corpus inventory and works before ingestion. It durably
        registers each discovered source ID in the project catalog:
        `discovered_sources` carries every live PDF/EPUB with its inclusion and
        index state, `sources` carries the bibliography of what is searchable
        now, `excluded_sources` carries each exclusion with its reason, and
        `reviewed_metadata_sources` makes every saved override inspectable.
        """

        return _present(
            "list_sources",
            await _tool_call(lambda: service().list_sources()),
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
    ) -> dict[str, Any]:
        """Return cleaned semantic context near one retrieved passage.

        The passage is returned with one neighboring chunk on each side. Context
        preserves source and page/section provenance but is not safe for direct
        quotation. Open the original PDF or EPUB for exact wording.
        """
        return _present(
            "get_passage",
            await _tool_call(lambda: service().get_passage(chunk_id)),
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
        source_path: SourcePath,
        reason: ExclusionReason = None,
    ) -> dict[str, Any]:
        """Include or exclude a PDF/EPUB from the project knowledge base.

        Use this after the agent or user has reviewed a source, including when a
        duplicate representation was found. Exclusion immediately blocks current
        search, source listing, and passage lookup, and future ingestion skips the
        source. The source file is never deleted or modified. Inclusion is
        reversible; re-ingest if the current generation does not contain it.
        Identify the source by the filename that list_sources or search reports.
        """

        return _present(
            "set_source_inclusion",
            await _tool_call(
                lambda: service().set_source_inclusion(
                    source_path=source_path,
                    included=included,
                    reason=reason,
                )
            ),
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
        "--tool-detail",
        choices=TOOL_DETAIL_MODES,
        default=os.environ.get("RESEARCH_ULTRARAG_TOOL_DETAIL", LEAN_TOOL_DETAIL),
        help=(
            "Tool answer detail. 'lean' returns the fields an agent acts on. "
            "'full' returns the complete payload for developer debugging of "
            "retrieval or ingestion."
        ),
    )
    parser.add_argument(
        "--ui-port",
        default=None,
        help=(
            "Also serve the local browser UI on this loopback port for the life "
            "of this server, reusing the resolved project, runtime, model, and "
            "offline settings. Omit to serve no UI. Give it explicitly: it is a "
            "choice for the server you started, so a server this project starts "
            "for itself never serves a UI."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warn", "error"),
        default="warn",
    )
    return parser


def _ui_port_from(raw: Any) -> int | None:
    """Normalize --ui-port, which a caller may pass as text."""

    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        port = int(raw)
    except (TypeError, ValueError):
        raise SystemExit("--ui-port must be an integer between 1 and 65535") from None
    if not 1 <= port <= 65535:
        raise SystemExit("--ui-port must be between 1 and 65535")
    return port


def _ui_port_decision(raw: Any) -> int | None:
    """Validate --ui-port, and refuse it in a server another server started."""

    port = _ui_port_from(raw)
    if port is not None and is_managed_child():
        raise SystemExit(
            "This server was started by another server, so it does not serve the "
            "browser UI: drop --ui-port from this server's command line and let "
            "the server you started serve it."
        )
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
            tool_detail=args.tool_detail,
        )
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
    create_server(config, ui_port=_ui_port_decision(args.ui_port)).run(
        transport="stdio", show_banner=False
    )


if __name__ == "__main__":
    main()

"""Project-scoped stdio MCP server for research knowledge bases."""

from __future__ import annotations

import argparse
import os
import signal
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, TypeAlias, TypeVar

from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from .config import (
    ConfigurationError,
    ResearchConfig,
    apply_process_priority,
    configured_source_directory,
    declared_owner_pid,
    is_managed_child,
    resolve_config,
)
from .instructions import SERVER_INSTRUCTIONS
from .rerankers import RERANKER_MODEL_CHOICES
from .service import ResearchError, ResearchService
from .settings import TOOL_DETAIL_MODES, describe_settings
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
        description=("Maximum number of ranked evidence passages to return (1-50)."),
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
LanguagesAnyFilter: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description=(
            "Case-insensitive 'any of' language filters; a result must be written "
            "in at least one supplied ISO 639 code. A source carries the language "
            "detected while extracting it and the language a review set instead, "
            "and `status.languages` lists the current inventory. Use it to search "
            "one language of a mixed corpus. Omit or pass null for no filter."
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
        """Report readiness, the selected generation, and whether it is current.

        Call this first, and before telling the user their corpus is up to date.
        No generation means nothing can be searched yet; `stale` and
        `generation_upgrade_required` say what moved and whether to ingest again.
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
        """Build or refresh the generation that search reads, reusing compatible work.

        Writes persistent state and may download a model, so get the user's
        agreement first. A long build answers status=in_progress: call it again
        until it returns ready or unchanged, then report what changed. One call
        covers at most `ingestion.work_budget_seconds` of work, so a build larger
        than that budget needs repeated identical calls; a client that stops
        repeating them cannot finish it, and the project's own config is where the
        budget is raised. A rejected call means another process holds the project
        and its message names that build; progress that goes backwards is reported
        as `superseded_build`, because a changed corpus cannot resume the old one.
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
        top_k: TopK = 10,
        categories_any: CategoriesAnyFilter = None,
        projects_any: ProjectsAnyFilter = None,
        keywords: KeywordFilter = None,
        languages_any: LanguagesAnyFilter = None,
        source_ids: SourceIdFilter = None,
        exclude_source_ids: ExcludeSourceIdFilter = None,
    ) -> dict[str, Any]:
        """Retrieve evidence passages for a research question.

        Each passage gives its source filename, its authors, its position, and
        cleaned text. Quote only from the original at that locator. Ask for more
        passages before concluding that the corpus has nothing: the reranker
        reorders about twice as many candidates as top_k, so a low top_k hides
        candidates from it, and a question asked in different words is a
        different search rather than a narrower one.
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
                    languages_any=languages_any,
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
        """List the project's sources: filenames, inclusion, and index state.

        The corpus inventory, and the answer also works before any ingestion.
        Use the filename it reports to name a source in set_source_inclusion.
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
        """Return one passage with its immediate neighbors on each side.

        Use it to read around a hit. The text is still cleaned for retrieval, so
        quote from the original at the passage's locator.
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
        """Exclude a source from retrieval, or restore one, without touching the file.

        Act only after the agent or user has reviewed the source, and give an
        exclusion its reason. The decision binds the current retrieval at once
        and the next ingestion; re-ingest to drop an excluded source physically.
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
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Save reviewed bibliographic metadata for one source.

        Fields: title, authors, year, doi, language, categories, keywords, and
        project. `language` takes ISO 639 codes, one per language the source is
        written in, and a code BM25 has no stopword list for is accepted because
        the metadata describes the source rather than the index. An empty review
        clears the entry so automatic metadata applies again. The review is
        authoritative at read time, so it binds current retrieval without
        re-ingesting, and the same JSON file may be edited by hand.
        """

        return _present(
            "set_source_metadata",
            await _tool_call(
                lambda: service().set_source_metadata(
                    metadata=metadata,
                    source_path=source_path,
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
        default=None,
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
            "Dense index backend for new generations. 'auto' uses an exact scan "
            "over the portable vectors below the documented corpus threshold and "
            "the embedded ANN index above it."
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
        "--tool-detail",
        choices=TOOL_DETAIL_MODES,
        default=None,
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
        default=None,
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


def process_exists(pid: int) -> bool:
    """Whether one process is still there, without signalling it."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def watch_owner(interval: float = 1.0) -> None:
    """End this server when the process that started it goes away.

    A stdio server normally ends when its client closes the pipe and the read
    loop sees end-of-file. A server busy in a long ingestion is not reading
    anything, so a client that dies mid-build leaves the build running for a
    project nobody is watching, with a gateway below it: a browser UI killed
    after its launcher's grace period, or an editor that crashed, is enough.

    The owner is the process that started this one, which every spawner in this
    package names in the child's environment, so a child orphaned before it could
    even look at its own parent still knows whose it was. The parent is the
    fallback for a server started by hand. It is re-checked from a daemon thread,
    so no amount of work on the reading thread can starve the check. SIGTERM goes
    to this process rather than ``os._exit``, so whatever the framework installed
    to shut down still runs and the gateway child is not simply abandoned.
    """

    owner = declared_owner_pid() or os.getppid()
    if owner <= 1:
        # Nothing started this process in a way that can own it.
        return

    def check() -> None:
        while True:
            time.sleep(interval)
            if not process_exists(owner):
                os.kill(os.getpid(), signal.SIGTERM)
                return

    threading.Thread(target=check, name="owner-watchdog", daemon=True).start()


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
            reranker_model=args.reranker_model,
            config_path=args.config,
            settings_overrides=args.set_overrides,
        )
    except ConfigurationError as exc:
        raise SystemExit(str(exc)) from exc
    if args.print_config:
        print(describe_settings(config.settings, config.settings_provenance))
        return
    apply_process_priority(config.nice)
    watch_owner()
    create_server(config, ui_port=_ui_port_decision(args.ui_port)).run(
        transport="stdio", show_banner=False
    )


if __name__ == "__main__":
    main()

"""Research adapter for the shared local UltraRAG MCP browser interface."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import socket
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol

import uvicorn
from fastmcp import Client
from ui_ultra_rag_mcp import (
    AdapterFactory,
    SourceFile,
    UICapabilities,
    UIProfile,
    UIRequestError,
    run_ui,
)
from ui_ultra_rag_mcp import create_ui_app as create_shared_ui_app

from .config import (
    FULL_TOOL_DETAIL,
    ConfigurationError,
    ResearchConfig,
    configured_source_directory,
    resolve_config,
    resolve_source_reference,
)
from .sources import SourcePolicyError, scan_sources
from .transport import create_research_transport
from .version import version_label

if TYPE_CHECKING:
    from starlette.applications import Starlette

UI_NAME = "research-ultra-rag-ui"
MAX_ERROR_LENGTH = 1200

RESEARCH_UI_PROFILE = UIProfile(
    application_name="Research UltraRAG",
    version_label=version_label(),
    project_label="Project",
    project_fallback_name="Research project",
    navigation_label="Research views",
    source_types_label="PDF + EPUB sources",
    ingest_intro=(
        "All included PDFs and EPUBs will be extracted and indexed. The current "
        "generation remains active unless the complete build succeeds."
    ),
    ingest_busy_message=(
        "Building BM25 and dense indexes. This can take several minutes…"
    ),
    footer_text=(
        "Retrieved text is cleaned for semantic use. Open the original PDF or "
        "EPUB before quoting."
    ),
    result_text_label="Cleaned semantic text — not for direct quotation",
    copy_text_label="Copy semantic text",
    bundle_import_intro=(
        "Place the archive in this project's .research-rag/bundles directory, "
        "then enter its filename. Existing source files are never overwritten; "
        "bundled reviewed metadata and exclusions replace the local copies."
    ),
    bundle_export_warning=(
        "Export this generation? The bundle contains complete original PDF/EPUB "
        "works and derived text. You are responsible for redistribution rights."
    ),
    capabilities=UICapabilities(
        bundle_export=True,
        bundle_import=True,
        force_recompute=True,
        source_selection=True,
        category_partitions=True,
        project_metadata=True,
    ),
)


class ResearchToolClient(Protocol):
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        **kwargs: Any,
    ) -> Any: ...


class ResearchUIAdapter:
    """Map the shared UI contract to the public research MCP tools."""

    def __init__(self, config: ResearchConfig, client: ResearchToolClient) -> None:
        self.config = config
        self.client = client

    async def health(self) -> Mapping[str, Any]:
        return {
            "project_root": str(self.config.project_root),
            "source_root": str(self.config.source_root),
        }

    async def call(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        while True:
            try:
                result = await self.client.call_tool(
                    operation,
                    dict(arguments),
                    timeout=1800,
                    raise_on_error=True,
                )
            except Exception as exc:
                message = str(exc).strip() or exc.__class__.__name__
                raise UIRequestError(message[:MAX_ERROR_LENGTH]) from exc
            data = getattr(result, "data", result)
            if not isinstance(data, Mapping):
                raise UIRequestError(
                    f"Research tool {operation!r} returned a non-object response",
                    status_code=502,
                )
            if operation != "ingest" or data.get("status") != "in_progress":
                return data

    async def source_file(self, source_path: str) -> SourceFile:
        try:
            target = resolve_source_reference(self.config, source_path)
            scan = scan_sources(self.config)
        except (ConfigurationError, SourcePolicyError, ValueError) as exc:
            raise UIRequestError(str(exc)) from exc
        selected = next((item for item in scan.selected if item.path == target), None)
        if selected is None:
            raise UIRequestError("Source was not found", status_code=404)
        media_type = (
            "application/pdf"
            if selected.extension == ".pdf"
            else "application/epub+zip"
        )
        disposition = "inline" if selected.extension == ".pdf" else "attachment"
        return SourceFile(
            path=selected.path,
            media_type=media_type,
            filename=selected.path.name,
            content_disposition_type=disposition,
        )


def _adapter_factory(config: ResearchConfig) -> AdapterFactory:
    @asynccontextmanager
    async def adapter_context() -> AsyncIterator[ResearchUIAdapter]:
        transport = create_research_transport(
            config,
            log_file=config.logs_root / "research-ui-mcp-stderr.log",
            # The UI reads the complete payload: it renders the ranking scores
            # and the full source inventory.
            tool_detail=FULL_TOOL_DETAIL,
        )
        async with Client(
            transport,
            name=UI_NAME,
            timeout=1800,
            init_timeout=1800,
        ) as client:
            yield ResearchUIAdapter(config, client)

    return adapter_context


def create_ui_app(
    config: ResearchConfig,
    *,
    research_client: ResearchToolClient | None = None,
) -> Starlette:
    """Create the shared UI with the research MCP adapter."""
    if research_client is not None:
        return create_shared_ui_app(
            profile=RESEARCH_UI_PROFILE,
            adapter=ResearchUIAdapter(config, research_client),
        )
    return create_shared_ui_app(
        profile=RESEARCH_UI_PROFILE,
        adapter_factory=_adapter_factory(config),
    )


UI_HOST = "127.0.0.1"


def _port_is_available(host: str, port: int) -> bool:
    """Whether the loopback port can still be bound."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


class EmbeddedUi:
    """An opt-in browser UI served inside the MCP server process.

    The MCP server has already resolved the project, runtime root, model cache,
    dense backend, and offline mode, so hosting the UI here removes the mismatch
    a separately launched ``research-ultra-rag-ui`` can have, and the UI stops
    with the server instead of outliving it. The host is fixed to loopback, no
    browser is opened, and ``status`` reports ``ui_url``, ``ui_ready``, and
    ``ui_error`` so an agent can hand the URL to the user without pretending it
    started anything itself.

    The UI shares this process's event loop, so every blocking operation it
    triggers must keep going through ``asyncio.to_thread`` the way the service
    already does; a synchronous call on the loop would stall the UI it serves.
    """

    def __init__(self, config: ResearchConfig, *, port: int) -> None:
        if not 1 <= port <= 65535:
            raise ConfigurationError("--ui-port must be between 1 and 65535")
        self.config = config
        self.host = UI_HOST
        self.port = port
        self.error: str | None = None
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ready(self) -> bool:
        """Whether the UI is serving right now.

        uvicorn sets ``Server.started`` once and never clears it, so the live
        task is part of the test: after ``stop`` and after a bind failure the
        task is finished and the UI is not serving.
        """

        return (
            self._server is not None
            and bool(self._server.started)
            and self._task is not None
            and not self._task.done()
        )

    async def start(self, *, research_client: ResearchToolClient | None = None) -> None:
        """Serve the UI for the life of the MCP server."""

        if not _port_is_available(self.host, self.port):
            self.error = (
                f"Port {self.port} is already in use on {self.host}, so the UI was "
                "not started; choose another --ui-port."
            )
            return
        app = create_ui_app(self.config, research_client=research_client)
        self._server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )
        )
        self._task = asyncio.create_task(self._server.serve())

    async def stop(self) -> None:
        """Ask the UI to stop and wait for its task to finish."""

        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            task, self._task = self._task, None
            with contextlib.suppress(asyncio.CancelledError):
                await task


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=UI_NAME,
        description="Launch a local UI for one research UltraRAG project.",
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("RESEARCH_ULTRARAG_PROJECT_ROOT"),
        required=os.environ.get("RESEARCH_ULTRARAG_PROJECT_ROOT") is None,
    )
    parser.add_argument(
        "--source-directory",
        default=os.environ.get("RESEARCH_ULTRARAG_SOURCE_DIRECTORY"),
        help="Omit to reuse the initialized project's source-directory setting.",
    )
    parser.add_argument(
        "--vanilla-executable",
        default=os.environ.get("VANILLA_ULTRARAG_MCP_EXECUTABLE"),
    )
    parser.add_argument(
        "--runtime-cache-root",
        default=os.environ.get("VANILLA_ULTRARAG_CACHE_ROOT"),
    )
    parser.add_argument(
        "--model-cache-root",
        default=os.environ.get("RESEARCH_ULTRARAG_MODEL_CACHE_ROOT"),
    )
    parser.add_argument("--offline", action="store_true")
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
            "Dense index backend for a new generation (default: auto, which is "
            "an exact scan below the documented corpus threshold)."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warn", "error"),
        default="warn",
    )
    parser.add_argument(
        "--host",
        choices=("127.0.0.1", "localhost", "::1"),
        default="127.0.0.1",
        help="Loopback address only (default: 127.0.0.1).",
    )
    parser.add_argument("--port", type=int, default=5051)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
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
    run_ui(
        create_ui_app(config),
        host=args.host,
        port=args.port,
        log_level="warning" if args.log_level == "warn" else args.log_level,
    )


if __name__ == "__main__":
    main()

"""Research adapter for the shared local UltraRAG MCP browser interface."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import socket
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
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
    ConfigurationError,
    ResearchConfig,
    apply_process_priority,
    configured_source_directory,
    resolve_config,
    resolve_source_reference,
)
from .rerankers import RERANKER_MODEL_CHOICES
from .settings import (
    FULL_TOOL_DETAIL,
    MAXIMUM_WORK_BUDGET_SECONDS,
    describe_settings,
)
from .sources import SourcePolicyError, scan_sources
from .transport import create_research_transport
from .version import version_label

if TYPE_CHECKING:
    from starlette.applications import Starlette

UI_NAME = "research-ultra-rag-ui"
MAX_ERROR_LENGTH = 1200
# How long the adapter waits for one tool call. It has to outlast the longest an
# ingest call may be told to run, or this wrapper gives up first and the build it
# was waiting for carries on with nobody watching.
UI_TOOL_TIMEOUT_SECONDS = MAXIMUM_WORK_BUDGET_SECONDS + 600
# Matches uvicorn's own default accept backlog, so a claimed socket is in the
# state uvicorn would have created for itself.
_CLAIM_BACKLOG = 2048

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
    # The shared UI's neutral labels apply here: the quote rule is stated once
    # for a reader in README.md and once for an agent in the tool description,
    # not on every passage a browser renders.
    capabilities=UICapabilities(
        metadata=True,
        force_recompute=True,
        source_selection=True,
        category_partitions=True,
        project_metadata=True,
        metadata_filters=True,
        retrieval_modes=False,
        reranking=False,
        chunk_settings=False,
    ),
)


class ResearchToolClient(Protocol):
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        **kwargs: Any,
    ) -> Any: ...

    async def list_tools(self) -> Any: ...


class ResearchUIAdapter:
    """Map the shared UI contract to the public research MCP tools."""

    def __init__(self, config: ResearchConfig, client: ResearchToolClient) -> None:
        self.config = config
        self.client = client
        self._parameters: dict[str, set[str]] = {}

    async def health(self) -> Mapping[str, Any]:
        return {
            "project_root": str(self.config.project_root),
            "source_root": str(self.config.source_root),
        }

    async def _declared_parameters(self, operation: str) -> set[str]:
        """Return the parameter names one tool declares, read once per process.

        The pinned shared UI sends its full optional set, including fields a
        profile hides. Each tool's own signature is the contract, so this server
        forwards only what that tool declares.
        """

        declared = self._parameters.get(operation)
        if declared is not None:
            return declared
        try:
            tools = await self.client.list_tools()
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            raise UIRequestError(message[:MAX_ERROR_LENGTH]) from exc
        for tool in tools:
            if getattr(tool, "name", None) != operation:
                continue
            schema = getattr(tool, "inputSchema", None) or {}
            declared = set((schema.get("properties") or {}).keys())
            self._parameters[operation] = declared
            return declared
        raise UIRequestError(
            f"Research tool {operation!r} is not available",
            status_code=502,
        )

    async def call(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        declared = await self._declared_parameters(operation)
        forwarded = {key: value for key, value in arguments.items() if key in declared}
        while True:
            try:
                result = await self.client.call_tool(
                    operation,
                    forwarded,
                    timeout=UI_TOOL_TIMEOUT_SECONDS,
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
            timeout=UI_TOOL_TIMEOUT_SECONDS,
            init_timeout=UI_TOOL_TIMEOUT_SECONDS,
        ) as client:
            yield ResearchUIAdapter(config, client)

    return adapter_context


def create_ui_app(
    config: ResearchConfig,
    *,
    research_client: ResearchToolClient | None = None,
) -> Starlette:
    """Create the shared UI with the research MCP adapter."""

    # More than one project can serve a UI at the same time, so each one names
    # the project it serves instead of showing a generic label: a browser window
    # must be able to say which knowledge base it belongs to.
    profile = replace(
        RESEARCH_UI_PROFILE,
        project_fallback_name=config.project_name,
    )
    if research_client is not None:
        return create_shared_ui_app(
            profile=profile,
            adapter=ResearchUIAdapter(config, research_client),
        )
    return create_shared_ui_app(
        profile=profile,
        adapter_factory=_adapter_factory(config),
    )


UI_HOST = "127.0.0.1"


def _claim_loopback_port(host: str, port: int) -> socket.socket:
    """Bind and listen on the loopback port, and return the socket that holds it.

    The claim is the bind and the listen, not a probe followed by a bind, so two
    servers that start at the same time cannot both believe they hold the port:
    the loser gets an ``OSError`` here, before any server exists, and reports it
    through ``ui_error``. Listening is what makes the claim exclusive — a socket
    that is bound but not listening can still be bound again under
    ``SO_REUSEADDR``, which Linux uses to allow binding over a socket that is not
    accepting. ``SO_REUSEADDR`` is set anyway, matching what uvicorn sets for
    itself, so a port whose connections are still in ``TIME_WAIT`` after a stop is
    not mistaken for one another process holds.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(_CLAIM_BACKLOG)
    except OSError:
        sock.close()
        raise
    return sock


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

    The port is claimed by binding it before uvicorn is created, so a UI that
    cannot claim its port reports the reason through ``ui_error`` instead of
    failing after a successful probe, and the claim is released with the socket.
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
        self._socket: socket.socket | None = None

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

        try:
            claim = _claim_loopback_port(self.host, self.port)
        except OSError as exc:
            reason = exc.strerror or str(exc)
            self.error = (
                f"Port {self.port} is already in use on {self.host}, so the UI was "
                f"not started; choose another --ui-port ({reason})."
            )
            return
        self._socket = claim
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
        self._task = asyncio.create_task(self._server.serve(sockets=[claim]))
        self._task.add_done_callback(self._task_finished)

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        """Record a UI task that ended on its own, and release its claim.

        uvicorn can end the task by raising, which the probe-then-bind order used
        to hide behind an empty ``ui_error``; the reason is kept here so that
        ``status`` reports it instead of showing a dead UI with no explanation.
        """

        if not task.cancelled():
            failure = task.exception()
            if failure is not None:
                self.error = (
                    f"The UI on {self.host}:{self.port} stopped: "
                    f"{failure.__class__.__name__}: {failure}"
                )
        self._release_claim()

    def _release_claim(self) -> None:
        """Close the claimed socket, which uvicorn also closes on shutdown."""

        claim, self._socket = self._socket, None
        if claim is not None:
            with contextlib.suppress(OSError):
                claim.close()

    async def stop(self) -> None:
        """Ask the UI to stop and wait for its task to finish.

        A UI that fails must never take the MCP server down with it: whatever
        ended the task is recorded by ``_task_finished`` and reported through
        ``ui_error``, so it is not re-raised here.
        """

        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            task, self._task = self._task, None
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._release_claim()


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
        default=None,
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
        "--log-level",
        choices=("debug", "info", "warn", "error"),
        default=None,
    )
    parser.add_argument(
        "--host",
        choices=("127.0.0.1", "localhost", "::1"),
        default="127.0.0.1",
        help="Loopback address only (default: 127.0.0.1).",
    )
    parser.add_argument("--port", type=int, default=5051)
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
    run_ui(
        create_ui_app(config),
        host=args.host,
        port=args.port,
        log_level="warning" if args.log_level == "warn" else args.log_level,
    )


if __name__ == "__main__":
    main()

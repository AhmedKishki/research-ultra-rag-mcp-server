"""Research adapter for the shared local UltraRAG MCP browser interface."""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol

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
    configured_source_directory,
    resolve_config,
    resolve_source_reference,
)
from .sources import SourcePolicyError, scan_sources
from .transport import create_research_transport

if TYPE_CHECKING:
    from starlette.applications import Starlette

UI_NAME = "research-ultra-rag-ui"
MAX_ERROR_LENGTH = 1200

RESEARCH_UI_PROFILE = UIProfile(
    application_name="Research UltraRAG",
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

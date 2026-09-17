"""Local browser interface backed by the public research MCP tools."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import uvicorn
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from .config import (
    ConfigurationError,
    ResearchConfig,
    resolve_config,
    resolve_source_reference,
)
from .sources import SourcePolicyError, scan_sources

LOGGER = logging.getLogger(__name__)
STATIC_ROOT = Path(__file__).with_name("ui_static")
UI_NAME = "research-ultra-rag-ui"
MAX_ERROR_LENGTH = 1200


class ResearchToolClient(Protocol):
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        **kwargs: Any,
    ) -> Any: ...


def _research_transport(config: ResearchConfig) -> StdioTransport:
    arguments = [
        "-m",
        "research_ultra_rag_mcp",
        "--project-root",
        str(config.project_root),
        "--source-directory",
        config.source_root.relative_to(config.project_root).as_posix(),
        "--vanilla-executable",
        str(config.vanilla_executable),
        "--log-level",
        config.log_level,
    ]
    if config.runtime_cache_root is not None:
        arguments.extend(["--runtime-cache-root", str(config.runtime_cache_root)])
    if config.offline:
        arguments.append("--offline")
    return StdioTransport(
        command=sys.executable,
        args=arguments,
        env=dict(os.environ),
        cwd=str(config.project_root),
        keep_alive=True,
        log_file=config.logs_root / "research-ui-mcp-stderr.log",
    )


async def _tool_call(
    request: Request,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client: ResearchToolClient = request.app.state.research_client
    try:
        result = await client.call_tool(
            name,
            arguments or {},
            timeout=1800,
            raise_on_error=True,
        )
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise HTTPException(
            status_code=400,
            detail=message[:MAX_ERROR_LENGTH],
        ) from exc
    data = getattr(result, "data", result)
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=502,
            detail=f"Research tool {name!r} returned a non-object response",
        )
    return data


def _same_origin(request: Request) -> bool:
    if request.headers.get("sec-fetch-site", "").casefold() == "cross-site":
        return False
    origin = request.headers.get("origin")
    if not origin:
        return True
    parsed = urlsplit(origin)
    return parsed.scheme in {"http", "https"} and parsed.netloc == request.headers.get(
        "host", ""
    )


async def _json_body(request: Request) -> dict[str, Any]:
    if not _same_origin(request):
        raise HTTPException(status_code=403, detail="Cross-origin writes are blocked")
    content_type = request.headers.get("content-type", "").split(";", 1)[0]
    if content_type.casefold() != "application/json":
        raise HTTPException(
            status_code=415,
            detail="Write requests require application/json",
        )
    try:
        value = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON request") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return value


async def _index(_: Request) -> Response:
    return FileResponse(STATIC_ROOT / "index.html", media_type="text/html")


async def _asset(request: Request) -> Response:
    filename = request.path_params["filename"]
    allowed = {
        "app.css": "text/css",
        "app.js": "text/javascript",
    }
    media_type = allowed.get(filename)
    if media_type is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(STATIC_ROOT / filename, media_type=media_type)


async def _health(request: Request) -> Response:
    config: ResearchConfig = request.app.state.config
    return JSONResponse(
        {
            "status": "ok",
            "project_root": str(config.project_root),
            "source_root": str(config.source_root),
        }
    )


async def _status(request: Request) -> Response:
    return JSONResponse(await _tool_call(request, "status"))


def _query_list(request: Request, name: str) -> list[str] | None:
    values: list[str] = []
    for raw in request.query_params.getlist(name):
        values.extend(item.strip() for item in raw.split(",") if item.strip())
    return values or None


async def _sources(request: Request) -> Response:
    return JSONResponse(
        await _tool_call(
            request,
            "list_sources",
            {
                "categories": _query_list(request, "categories"),
                "keywords": _query_list(request, "keywords"),
            },
        )
    )


async def _search(request: Request) -> Response:
    body = await _json_body(request)
    allowed = {
        "query",
        "top_k",
        "categories",
        "keywords",
        "document_ids",
        "retrieval_method",
        "rerank",
    }
    unknown = set(body) - allowed
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported search fields: {', '.join(sorted(unknown))}",
        )
    return JSONResponse(await _tool_call(request, "search", body))


async def _passage(request: Request) -> Response:
    try:
        context_chunks = int(request.query_params.get("context_chunks", "1"))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="context_chunks must be an integer",
        ) from exc
    return JSONResponse(
        await _tool_call(
            request,
            "get_passage",
            {
                "chunk_id": request.path_params["chunk_id"],
                "context_chunks": context_chunks,
            },
        )
    )


async def _ingest(request: Request) -> Response:
    body = await _json_body(request)
    allowed = {"chunk_size", "chunk_overlap"}
    unknown = set(body) - allowed
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported ingestion fields: {', '.join(sorted(unknown))}",
        )
    return JSONResponse(await _tool_call(request, "ingest", body))


async def _set_metadata(request: Request) -> Response:
    body = await _json_body(request)
    if set(body) != {"source_path", "metadata"}:
        raise HTTPException(
            status_code=400,
            detail="Metadata requests require source_path and metadata",
        )
    return JSONResponse(await _tool_call(request, "set_source_metadata", body))


async def _set_inclusion(request: Request) -> Response:
    body = await _json_body(request)
    allowed = {"source_path", "included", "reason"}
    unknown = set(body) - allowed
    if unknown or not {"source_path", "included"}.issubset(body):
        raise HTTPException(
            status_code=400,
            detail="Inclusion requests require source_path and included",
        )
    return JSONResponse(await _tool_call(request, "set_source_inclusion", body))


async def _source_file(request: Request) -> Response:
    config: ResearchConfig = request.app.state.config
    raw_path = request.query_params.get("path", "").strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="Source path is required")
    try:
        target = resolve_source_reference(config, raw_path)
        scan = scan_sources(config)
    except (ConfigurationError, SourcePolicyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    selected = next((item for item in scan.selected if item.path == target), None)
    if selected is None:
        raise HTTPException(status_code=404, detail="Source was not found")
    media_type = (
        "application/pdf" if selected.extension == ".pdf" else "application/epub+zip"
    )
    disposition = "inline" if selected.extension == ".pdf" else "attachment"
    return FileResponse(
        selected.path,
        media_type=media_type,
        filename=selected.path.name,
        content_disposition_type=disposition,
    )


async def _security_headers(request: Request, call_next: Any) -> Response:
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'none'; object-src 'self'; script-src 'self'; "
        "style-src 'self'; connect-src 'self'; img-src 'self' data:"
    )
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


async def _http_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HTTPException)
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


async def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Research UI request failed", exc_info=exc)
    return JSONResponse(
        {"error": "Unexpected research UI failure; inspect the UI log."},
        status_code=500,
    )


def create_ui_app(
    config: ResearchConfig,
    *,
    research_client: ResearchToolClient | None = None,
) -> Starlette:
    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        app.state.config = config
        if research_client is not None:
            app.state.research_client = research_client
            yield
            return

        transport = _research_transport(config)
        async with Client(
            transport,
            name=UI_NAME,
            timeout=1800,
            init_timeout=1800,
        ) as client:
            app.state.research_client = client
            yield

    routes = [
        Route("/", _index),
        Route("/assets/{filename:str}", _asset),
        Route("/api/health", _health),
        Route("/api/status", _status),
        Route("/api/sources", _sources),
        Route("/api/search", _search, methods=["POST"]),
        Route("/api/passages/{chunk_id:str}", _passage),
        Route("/api/ingest", _ingest, methods=["POST"]),
        Route("/api/source-metadata", _set_metadata, methods=["POST"]),
        Route("/api/source-inclusion", _set_inclusion, methods=["POST"]),
        Route("/api/source-file", _source_file),
    ]
    app = Starlette(
        routes=routes,
        lifespan=lifespan,
        middleware=[Middleware(BaseHTTPMiddleware, dispatch=_security_headers)],
        exception_handlers={
            HTTPException: _http_error,
            Exception: _unexpected_error,
        },
    )
    app.state.config = config

    return app


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
        default=os.environ.get("RESEARCH_ULTRARAG_SOURCE_DIRECTORY", "sources"),
    )
    parser.add_argument(
        "--vanilla-executable",
        default=os.environ.get("VANILLA_ULTRARAG_MCP_EXECUTABLE"),
    )
    parser.add_argument(
        "--runtime-cache-root",
        default=os.environ.get("VANILLA_ULTRARAG_CACHE_ROOT"),
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
    uvicorn.run(
        create_ui_app(config),
        host=args.host,
        port=args.port,
        log_level="warning" if args.log_level == "warn" else args.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()

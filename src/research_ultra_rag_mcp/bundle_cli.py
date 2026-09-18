"""Terminal export/import client for portable Research RAG bundles."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastmcp import Client

from .config import configured_source_directory, resolve_config
from .sources import sha256_file
from .transport import create_research_transport


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-ultra-rag-bundle",
        description=(
            "Export or import a validated project generation. Exported archives "
            "contain complete original PDF/EPUB works; you are responsible for "
            "redistribution rights."
        ),
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("export", "import"):
        command = subparsers.add_parser(
            operation,
            help=(
                "Export the fresh current generation and all originals."
                if operation == "export"
                else "Validate a bundle and reconstruct local indexes."
            ),
        )
        command.add_argument(
            "--project-root",
            type=Path,
            required=True,
            help="Research project root.",
        )
        command.add_argument(
            "--offline",
            action="store_true",
            help="Require runtime and query-model files to be cached.",
        )
        command.add_argument(
            "--model-cache-root",
            default=os.environ.get("RESEARCH_ULTRARAG_MODEL_CACHE_ROOT"),
            help="Override the shared FastEmbed model cache.",
        )
        if operation == "import":
            command.add_argument(
                "--bundle",
                type=Path,
                required=True,
                help="Path to a .research-rag.zip archive.",
            )
            command.add_argument(
                "--no-activate",
                action="store_true",
                help=(
                    "Leave the generation pointer unchanged; originals and "
                    "reviewed project state are still imported."
                ),
            )
    return parser


def _stage_external_bundle(project: Path, source: Path) -> str:
    source = source.expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"Bundle is not a regular file: {source}")
    bundles = project / ".research-rag" / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    destination = bundles / source.name
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(
                f"A different bundle already uses this project-local name: {source.name}"
            )
    else:
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    source_sidecar = source.with_suffix(source.suffix + ".sha256")
    if source_sidecar.exists():
        if not source_sidecar.is_file() or source_sidecar.is_symlink():
            raise RuntimeError(
                f"Bundle sidecar is not a regular file: {source_sidecar}"
            )
        destination_sidecar = destination.with_suffix(destination.suffix + ".sha256")
        temporary_sidecar = destination_sidecar.with_name(
            f".{destination_sidecar.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            shutil.copy2(source_sidecar, temporary_sidecar)
            os.replace(temporary_sidecar, destination_sidecar)
        finally:
            temporary_sidecar.unlink(missing_ok=True)
    return destination.name


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.expanduser().resolve()
    if not project.is_dir():
        raise RuntimeError(f"Project root is not a directory: {project}")
    config = resolve_config(
        project,
        source_directory=configured_source_directory(project),
        model_cache_root=args.model_cache_root,
        offline=args.offline,
    )
    transport = create_research_transport(
        config,
        log_file=project / ".ultrarag" / "research" / "logs" / "bundle-stderr.log",
    )
    async with Client(transport, timeout=1800, init_timeout=1800) as client:
        if args.operation == "export":
            result = await client.call_tool("export_bundle", {}, timeout=1800)
        else:
            bundle_name = await asyncio.to_thread(
                _stage_external_bundle,
                project,
                args.bundle,
            )
            result = await client.call_tool(
                "import_bundle",
                {"bundle_name": bundle_name, "activate": not args.no_activate},
                timeout=1800,
            )
    if not isinstance(result.data, dict):
        raise TypeError("Bundle tool returned an invalid response")
    return result.data


def main() -> None:
    args = _parser().parse_args()
    try:
        result = asyncio.run(_run(args))
    except Exception as exc:
        raise SystemExit(f"Bundle operation failed: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

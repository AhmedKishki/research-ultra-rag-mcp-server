"""The core command line: one research project, no MCP client required.

The package has four surfaces over the same project state. `research-ultra-rag-mcp`
serves it over stdio MCP, `research-ultra-rag-ui` serves it to a browser,
`research-ultra-rag-verify` checks the MCP surface end to end, and this module is
the plainest one: it resolves a project, constructs the same `ResearchService`
the server exposes, and calls it in this process.

That makes the knowledge base usable with no MCP client, no agent, and no
long-running server — `init` creates a project, `ingest` builds it, `search`
reads it, and `ui` browses it. Answers are projected by the same `tool_views`
projector the server uses, so a command and its tool cannot drift apart.

Only a command that actually queries opens the vanilla gateway, and it is opened
on the first call rather than chosen from the command line: the BM25 index is
initialized through that gateway when a generation is loaded for querying.
Commands that only read local state — `init`, `config`, `ui`, `sources`,
`status`, `passage`, and the review commands — never start one, so they stay
quick and keep working on a machine where no UltraRAG runtime is installed yet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import signal
import subprocess
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

from fastmcp import Client

from .config import (
    ConfigurationError,
    ResearchConfig,
    apply_process_priority,
    configured_source_directory,
    resolve_config,
)
from .launcher import launcher_path, ui_launcher_state
from .rerankers import RERANKER_MODEL_CHOICES
from .service import ResearchService
from .settings import FULL_TOOL_DETAIL, TOOL_DETAIL_MODES, describe_settings
from .support import DEFAULT_RETRIEVAL_METHOD, RETRIEVAL_METHODS, ResearchError
from .tool_views import present_tool_response
from .ultrarag import VanillaUltraRAG, create_vanilla_transport

CLI_NAME = "research-ultra-rag"
DEFAULT_DEPTH = 10
# What a project's own .gitignore has to keep out of version control: the
# derived state that can be rebuilt, and the machine-local launcher. The
# descriptor, catalogs, and review files are small, portable, and worth keeping.
VERSION_CONTROL_NOTES = (
    ".research-rag/runtime/",
    ".research-rag/bin/",
    "open-ui.sh",
)
# A process the stop sweep may signal has to name one of these, so a shell or an
# editor that merely mentions the project path is never touched.
SERVICE_MARKERS = ("research_ultra_rag_mcp", "research-ultra-rag")
STOP_GRACE_SECONDS = 5.0


class _LazyGateway:
    """A vanilla-gateway client that connects when a command first reaches for it.

    Which operations need the gateway is not a property of the command line: the
    BM25 index is initialized through it when a generation is loaded for querying,
    so `search` needs one and `status` usually does not, and a command that
    recovers interrupted work may need one when nothing else does. Rather than
    predict that and start a process a command may never call, the client is
    opened by the first call. Commands that only read local state — `init`,
    `config`, `ui`, `sources`, `status`, `passage`, and the review commands —
    never open one.
    """

    def __init__(self, config: ResearchConfig) -> None:
        self._config = config
        self._stack: AsyncExitStack | None = None
        self._client: Any = None

    async def call_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if self._client is None:
            stack = AsyncExitStack()
            transport = create_vanilla_transport(self._config)
            self._client = await stack.enter_async_context(
                Client(transport, timeout=1800, init_timeout=1800)
            )
            self._stack = stack
        return await self._client.call_tool(name, *args, **kwargs)

    async def aclose(self) -> None:
        """Close the gateway if this command opened one, and never otherwise."""

        if self._stack is not None:
            await self._stack.aclose()


@asynccontextmanager
async def _service(config: ResearchConfig) -> AsyncIterator[ResearchService]:
    """Yield the service over a gateway that opens only if it is spoken to."""

    gateway = _LazyGateway(config)
    try:
        yield ResearchService(config, VanillaUltraRAG(gateway))
    finally:
        await gateway.aclose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=(
            "Run one project-scoped PDF/EPUB research knowledge base from a "
            "terminal, without an MCP client."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("RESEARCH_ULTRARAG_PROJECT_ROOT", "."),
        help="Project root holding .research-rag (default: the current directory).",
    )
    parser.add_argument(
        "--runtime-root",
        default=os.environ.get("RESEARCH_ULTRARAG_RUNTIME_ROOT"),
        help=(
            "Absolute directory for derived state, when the project itself is on "
            "slow storage. Omit to keep it under .research-rag."
        ),
    )
    parser.add_argument(
        "--model-cache-root",
        default=None,
        help="Shared model cache (default: the user cache for this application).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=None,
        help="Require the vanilla runtime and every model to be cached already.",
    )
    parser.add_argument(
        "--embedding-threads",
        type=int,
        default=None,
        help="ONNX Runtime threads for the embedding model; unset lets the runtime decide.",
    )
    parser.add_argument(
        "--dense-backend",
        choices=("auto", "exact", "qdrant"),
        default=None,
        help="Dense index backend for a new generation (default: auto).",
    )
    parser.add_argument(
        "--reranker-model",
        choices=RERANKER_MODEL_CHOICES,
        default=None,
        help="Cross-encoder the engine loads for a reranked search.",
    )
    parser.add_argument(
        "--detail",
        choices=TOOL_DETAIL_MODES,
        default=FULL_TOOL_DETAIL,
        help=(
            "Answer detail. 'full' prints everything, which is what a person "
            "reading a terminal wants; 'lean' prints exactly what an agent's "
            "tool answer would carry."
        ),
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("RESEARCH_ULTRARAG_CONFIG"),
        help="Extra settings file, layered above the per-user and project files.",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Override one setting for this command; repeat for more. This reaches "
            "every operation, including ingest and search, because the command "
            "resolves settings in this process."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    create = commands.add_parser(
        "init",
        help="Create a project, or add .research-rag to a directory you already have.",
    )
    create.add_argument(
        "--name",
        default=None,
        help=(
            "Name to record for this project. An existing project is renamed only "
            "when this is given; the stable project id never changes."
        ),
    )
    create.add_argument(
        "--sources",
        default=None,
        help="Project-relative source directory to create (default: sources).",
    )

    commands.add_parser(
        "status", help="Report readiness and what changed since the generation."
    )

    refresh = commands.add_parser(
        "ingest", help="Build or refresh the searchable generation."
    )
    refresh.add_argument(
        "--force-recompute",
        action="store_true",
        help="Rebuild every document, chunk, and vector instead of reusing compatible ones.",
    )

    find = commands.add_parser(
        "search", help="Retrieve evidence passages for a question."
    )
    find.add_argument(
        "query", help="Research question, exact phrase, name, or concept."
    )
    find.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_DEPTH,
        help=f"Maximum ranked passages to return (default: {DEFAULT_DEPTH}).",
    )
    find.add_argument(
        "--method",
        choices=sorted(RETRIEVAL_METHODS),
        default=DEFAULT_RETRIEVAL_METHOD,
        help="Retrieval half to rank with (default: hybrid, the tool's method).",
    )
    find.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the cross-encoder, which is on by default and is what search costs.",
    )
    find.add_argument(
        "--category",
        action="append",
        help="Keep results carrying any of these categories.",
    )
    find.add_argument(
        "--project",
        action="append",
        help="Keep results carrying any of these project tags.",
    )
    find.add_argument(
        "--keyword",
        action="append",
        help="Keep results carrying every one of these keywords.",
    )
    find.add_argument(
        "--language",
        action="append",
        help="Keep results written in any of these ISO 639 codes.",
    )
    find.add_argument(
        "--author",
        action="append",
        help="Keep results whose source has one of these names among its authors.",
    )
    find.add_argument(
        "--title",
        action="append",
        help="Keep results whose source title contains one of these phrases.",
    )
    find.add_argument(
        "--source-id", action="append", help="Search only these stable source ids."
    )
    find.add_argument(
        "--exclude-source-id",
        action="append",
        help="Search everything except these source ids.",
    )

    commands.add_parser(
        "sources", help="List the project's sources and their review state."
    )

    context = commands.add_parser(
        "passage", help="Read one passage and its immediate neighbours."
    )
    context.add_argument("chunk_id", help="Exact chunk id from a search result.")
    context.add_argument(
        "--context-chunks",
        type=int,
        default=1,
        help="Neighbours to include on each side (0-5, default: 1).",
    )

    for verb, reason_required in (("include", False), ("exclude", True)):
        change = commands.add_parser(
            verb,
            help=f"{verb.capitalize()} one source in retrieval without touching the file.",
        )
        change.add_argument(
            "source",
            nargs="?",
            help="Source path relative to the source directory.",
        )
        change.add_argument(
            "--source-id", help="The stable source id, instead of a path."
        )
        change.add_argument(
            "--reason",
            required=reason_required,
            help=(
                "Why the decision was made."
                if reason_required
                else "Why the decision was made, when it is worth recording."
            ),
        )

    review = commands.add_parser(
        "metadata",
        help="Save reviewed bibliographic metadata for one source.",
    )
    review.add_argument(
        "source", nargs="?", help="Source path relative to the source directory."
    )
    review.add_argument("--source-id", help="The stable source id, instead of a path.")
    review.add_argument("--title", help="Reviewed title.")
    review.add_argument(
        "--author", action="append", help="An author, in the order to list them."
    )
    review.add_argument("--year", type=int, help="Reviewed year of publication.")
    review.add_argument("--doi", help="Reviewed DOI.")
    review.add_argument(
        "--category", action="append", help="A category the work belongs to."
    )
    review.add_argument(
        "--keyword", action="append", help="A keyword the work carries."
    )
    review.add_argument(
        "--language",
        action="append",
        help="A language the work is written in, as an ISO 639 code.",
    )
    review.add_argument("--project", help="The project tag to file the work under.")
    review.add_argument(
        "--clear",
        action="store_true",
        help="Remove the review instead, so automatic metadata applies again.",
    )

    commands.add_parser(
        "config", help="Print the merged settings and where each value came from."
    )

    browser = commands.add_parser(
        "ui", help="Start, open, or stop this project's browser view."
    )
    browser.add_argument(
        "--open", action="store_true", help="Open it in the browser as well."
    )
    browser.add_argument(
        "--stop", action="store_true", help="Stop it, and its private server."
    )
    browser.add_argument("--port", type=int, help="Serve on a different port.")

    stop = commands.add_parser(
        "stop",
        help="Stop this project's browser view, and optionally every server of it.",
    )
    stop.add_argument(
        "--servers",
        action="store_true",
        help=(
            "Also stop every research process serving this project, including a "
            "server an MCP client started. The client, not this command, decides "
            "whether a server it owns comes back."
        ),
    )
    return parser


def _project_path(args: argparse.Namespace) -> Path:
    return Path(args.project_root).expanduser().resolve()


def _config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The command-line settings layer, shared by every command."""

    return {
        "runtime_root": args.runtime_root,
        "model_cache_root": args.model_cache_root,
        "offline": args.offline,
        "dense_backend": args.dense_backend,
        "embedding_threads": args.embedding_threads,
        "reranker_model": args.reranker_model,
        "config_path": args.config,
        "settings_overrides": args.set_overrides,
    }


def _command(project_root: Path, *arguments: str) -> str:
    """A copy-pasteable command naming this project, quoting what needs it."""

    parts = (CLI_NAME, "--project-root", str(project_root), *arguments)
    return " ".join(shlex.quote(part) for part in parts)


def _resolve(args: argparse.Namespace) -> ResearchConfig:
    """Resolve an existing project, reusing the source directory it recorded."""

    project = _project_path(args)
    return resolve_config(
        project,
        source_directory=configured_source_directory(project),
        **_config_kwargs(args),
    )


def _init(args: argparse.Namespace) -> dict[str, Any]:
    """Create a project, or attach the portable state to a directory that exists.

    Both are the same act: a project *is* a directory with `.research-rag` beside
    whatever is already there. Nothing this command does touches a file the user
    wrote, so pointing it at a repository that already holds work adds only the
    portable state, the source directory, and the browser launcher.
    """

    project = _project_path(args)
    created: list[str] = []
    if project.exists() and not project.is_dir():
        raise ConfigurationError(f"Project root is not a directory: {project}")
    if not project.exists():
        project.mkdir(parents=True)
        created.append("project_root")
    # An existing project records its own source directory, and a project that is
    # not there yet has no descriptor to read, so both are settled before the
    # descriptor is written.
    sources = args.sources or configured_source_directory(project)
    config = resolve_config(
        project,
        source_directory=sources,
        project_name=args.name,
        **_config_kwargs(args),
    )
    if not config.source_root.exists():
        config.source_root.mkdir(parents=True)
        created.append("source_root")
    return {
        "status": "ready",
        "project_root": str(config.project_root),
        "project_id": config.project_id,
        "project_name": config.project_name,
        "source_root": str(config.source_root),
        "created": created,
        "launcher": ui_launcher_state(config.project_root, config.portable_root),
        "keep_out_of_version_control": list(VERSION_CONTROL_NOTES),
        "next_steps": [
            f"Add PDF or EPUB sources to {config.source_root}.",
            _command(config.project_root, "ingest"),
            _command(config.project_root, "search", "your question"),
            _command(config.project_root, "ui", "--open"),
        ],
    }


def _ui(args: argparse.Namespace, config: ResearchConfig) -> None:
    """Hand the browser view to the project's own generated launcher.

    That launcher already owns the free-port choice, the pid file, and stopping
    the private server's whole process group, so this command reuses it rather
    than keeping a second copy of that logic here.
    """

    script = launcher_path(config.portable_root)
    if not script.is_file():
        raise ResearchError(
            f"This project has no UI launcher at {script}. "
            f"Run '{CLI_NAME} init' once to generate it."
        )
    command = [str(script)]
    if args.open:
        command.append("--open")
    if args.stop:
        command.append("--stop")
    if args.port is not None:
        command.extend(["--port", str(args.port)])
    status = subprocess.call(command)
    if status != 0:
        raise ResearchError(f"The UI launcher exited with status {status}.")


def _project_argument(arguments: list[str]) -> str | None:
    """Return the ``--project-root`` value in one command line, if it has one."""

    for index, argument in enumerate(arguments):
        if argument == "--project-root" and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith("--project-root="):
            return argument.split("=", 1)[1]
    return None


def _service_processes(
    project_root: Path,
    proc_root: Path = Path("/proc"),
) -> list[tuple[int, str]]:
    """Return the ``(pid, command)`` pairs serving this project.

    A process qualifies when its command line names a research entry point *and*
    this project as ``--project-root``. Both halves matter: the marker keeps a
    shell or an editor that merely mentions the path out of the sweep, and the
    path keeps another project's server out of it. A relative ``--project-root``
    is not matched, because resolving it would use this process's directory
    rather than the other one's; every launcher and client configuration passes
    an absolute path.
    """

    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []
    found: list[tuple[int, str]] = []
    for pid in sorted(int(entry.name) for entry in entries if entry.name.isdigit()):
        if pid == os.getpid():
            continue
        try:
            raw = (proc_root / str(pid) / "cmdline").read_bytes()
        except OSError:
            continue
        arguments = [
            part for part in raw.decode("utf-8", "replace").split("\0") if part
        ]
        if not any(
            marker in argument for argument in arguments for marker in SERVICE_MARKERS
        ):
            continue
        if _project_argument(arguments) != str(project_root):
            continue
        found.append((pid, " ".join(arguments)))
    return found


def _alive(pid: int) -> bool:
    """Whether this process still exists and could be signalled."""

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate(pids: list[int]) -> list[int]:
    """Ask these processes to stop and then insist; return the ones killed outright."""

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    deadline = time.monotonic() + STOP_GRACE_SECONDS
    while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
        time.sleep(0.1)
    forced = [pid for pid in pids if _alive(pid)]
    for pid in forced:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue
    return forced


def _stop(args: argparse.Namespace, config: ResearchConfig) -> dict[str, Any]:
    """Stop this project's browser view, and with ``--servers`` every process serving it.

    The launcher's own message is captured rather than printed, so a caller gets
    one JSON report on stdout and nothing else.
    """

    script = launcher_path(config.portable_root)
    report: dict[str, Any] = {"project_root": str(config.project_root)}
    if script.is_file():
        stopped = subprocess.run(
            [str(script), "--stop"],
            capture_output=True,
            text=True,
            check=False,
        )
        report["ui_launcher_status"] = stopped.returncode
        report["ui_launcher_output"] = stopped.stdout.strip()
    else:
        report["ui_launcher_status"] = None
    if not args.servers:
        return report
    found = _service_processes(config.project_root)
    report["servers"] = [{"pid": pid, "command": command} for pid, command in found]
    report["forced_pids"] = _terminate([pid for pid, _ in found])
    report["notes"] = [
        "A server your MCP client started belongs to that client: this command stops it, and the client decides whether it comes back.",
        "A build interrupted this way resumes from its checkpoint on the next ingest call.",
    ]
    return report


def _metadata_body(args: argparse.Namespace) -> dict[str, Any]:
    """Build the review body, refusing a request that asks for two things at once."""

    supplied: dict[str, Any] = {
        "title": args.title,
        "authors": args.author,
        "year": args.year,
        "doi": args.doi,
        "language": args.language,
        "categories": args.category,
        "keywords": args.keyword,
        "project": args.project,
    }
    if args.clear and any(value is not None for value in supplied.values()):
        raise ResearchError(
            "--clear removes the review so automatic metadata applies again, "
            "so it cannot be combined with a field."
        )
    if args.clear:
        return {}
    return {key: value for key, value in supplied.items() if value is not None}


async def _operate(
    args: argparse.Namespace,
    service: ResearchService,
) -> tuple[str, dict[str, Any]]:
    """Call the one operation this command names, and report its tool name."""

    command = args.command
    if command == "status":
        return "status", await service.status()
    if command == "ingest":
        return "ingest", await service.ingest(force_recompute=args.force_recompute)
    if command == "search":
        return "search", await service.search(
            args.query,
            top_k=args.top_k,
            categories_any=args.category,
            projects_any=args.project,
            keywords=args.keyword,
            languages_any=args.language,
            authors_any=args.author,
            titles_any=args.title,
            source_ids=args.source_id,
            exclude_source_ids=args.exclude_source_id,
            retrieval_method=args.method,
            rerank=not args.no_rerank,
        )
    if command == "sources":
        return "list_sources", await service.list_sources()
    if command == "passage":
        return "get_passage", await service.get_passage(
            args.chunk_id,
            context_chunks=args.context_chunks,
        )
    if command in {"include", "exclude"}:
        return "set_source_inclusion", await service.set_source_inclusion(
            args.source,
            source_id=args.source_id,
            included=command == "include",
            reason=args.reason,
        )
    if command == "metadata":
        return "set_source_metadata", await service.set_source_metadata(
            _metadata_body(args),
            source_path=args.source,
            source_id=args.source_id,
        )
    raise ResearchError(f"Unknown command: {command}")


async def _run(args: argparse.Namespace) -> dict[str, Any] | None:
    """Resolve the project, run the named command, and return what to print."""

    if args.command == "init":
        return _init(args)
    config = _resolve(args)
    apply_process_priority(config.nice)
    if args.command == "config":
        print(describe_settings(config.settings, config.settings_provenance))
        return None
    if args.command == "ui":
        _ui(args, config)
        return None
    if args.command == "stop":
        return _stop(args, config)
    async with _service(config) as service:
        tool, payload = await _operate(args, service)
    return present_tool_response(tool, payload, detail=args.detail)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        result = asyncio.run(_run(args))
    except (ConfigurationError, ResearchError, OSError, ValueError) as exc:
        raise SystemExit(f"{CLI_NAME}: {exc}") from exc
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import contextlib
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.testclient import TestClient

from research_ultra_rag_mcp.config import (
    MANAGED_CHILD_ENV,
    TOP_LEVEL_ONLY_ENV,
    child_process_environment,
    resolve_config,
)
from research_ultra_rag_mcp.server import _parser, _ui_port_decision, _ui_port_from
from research_ultra_rag_mcp.transport import create_research_transport
from research_ultra_rag_mcp.ui import EmbeddedUi, create_ui_app
from research_ultra_rag_mcp.ultrarag import create_vanilla_transport

TOOL_PARAMETERS: dict[str, set[str]] = {
    "status": set(),
    "ingest": {"force_recompute"},
    "search": {
        "query",
        "top_k",
        "categories_any",
        "projects_any",
        "keywords",
        "authors_any",
        "titles_any",
        "source_ids",
        "exclude_source_ids",
    },
    "list_sources": set(),
    "get_passage": {"chunk_id"},
    "set_source_inclusion": {"included", "reason", "source_path"},
    "set_source_metadata": {"source_path", "metadata"},
}


class FakeResearchClient:
    """Stand in for the MCP client, including the tool schemas it publishes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                name=name,
                inputSchema={"properties": {key: {} for key in parameters}},
            )
            for name, parameters in TOOL_PARAMETERS.items()
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        **_: Any,
    ) -> SimpleNamespace:
        self.calls.append((name, arguments))
        responses: dict[str, dict[str, Any]] = {
            "status": {
                "ready": True,
                "stale": False,
                "project_root": "/research/project",
                "project_name": "Research project",
                "source_root": "/research/project/sources",
                "generation_id": "generation-1",
                "created_at": "2026-09-17T12:00:00Z",
                "selected_source_count": 1,
                "indexed_source_count": 1,
                "searchable_source_count": 1,
                "excluded_source_count": 0,
                "chunk_count": 1,
                "hybrid_ready": True,
                "generation_upgrade_required": False,
                "upgrade_reasons": [],
                "available_retrieval_methods": ["bm25", "dense", "hybrid"],
                "default_retrieval_method": "hybrid",
            },
            "list_sources": {
                "ready": True,
                "source_count": 1,
                "sources": [
                    {
                        "document_id": "doc-1",
                        "source_path": "sources/evidence.pdf",
                        "source_relative_path": "evidence.pdf",
                        "format": "pdf",
                        "title": "Evidence",
                        "authors": ["Researcher"],
                        "year": 2026,
                        "doi": "",
                        "categories": ["theory"],
                        "keywords": ["evidence"],
                    }
                ],
                "excluded_source_count": 0,
                "excluded_sources": [],
            },
            "search": {
                "query": arguments.get("query", ""),
                "generation_id": "generation-1",
                "stale": False,
                "retrieval_method": arguments.get("retrieval_method", "hybrid"),
                "reranked": arguments.get("rerank", False),
                "result_count": 1,
                "hits": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "title": "Evidence",
                        "authors": ["Researcher"],
                        "year": 2026,
                        "source_path": "sources/evidence.pdf",
                        "categories": ["theory"],
                        "keywords": ["evidence"],
                        "locator": {
                            "type": "pdf_page",
                            "page": 1,
                            "page_label": "1",
                        },
                        "citation": "Researcher, Evidence (2026), p. 1",
                        "text": "A cleaned semantic passage.",
                        "direct_quote_safe": False,
                        "component_ranks": {"bm25": 1, "dense": 1},
                        "component_scores": {
                            "dense_cosine_similarity": 0.8,
                            "bm25": None,
                        },
                        "fusion_score": 0.03,
                        "rerank_score": None,
                    }
                ],
            },
            "get_passage": {
                "generation_id": "generation-1",
                "requested_chunk_id": "chunk-1",
                "context": [
                    {
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "source_path": "sources/evidence.pdf",
                        "title": "Evidence",
                        "locator": {"type": "pdf_page", "page": 1},
                        "citation": "Researcher, Evidence (2026), p. 1",
                        "text": "A cleaned semantic passage.",
                    }
                ],
            },
            "set_source_inclusion": {
                "source_relative_path": arguments.get("source_path"),
                "included": arguments.get("included"),
                "message": "Inclusion saved.",
            },
            "set_source_metadata": {
                "source_relative_path": arguments.get("source_path"),
                "metadata": arguments.get("metadata"),
                "message": "Reviewed metadata saved.",
            },
            "ingest": {
                "status": "ready",
                "generation_id": "generation-2",
                "chunk_count": 2,
            },
        }
        return SimpleNamespace(data=responses[name])


class BatchedIngestResearchClient(FakeResearchClient):
    def __init__(self) -> None:
        super().__init__()
        self.ingest_calls = 0

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        **kwargs: Any,
    ) -> SimpleNamespace:
        if name == "ingest":
            self.ingest_calls += 1
            if self.ingest_calls == 1:
                self.calls.append((name, arguments))
                return SimpleNamespace(
                    data={
                        "status": "in_progress",
                        "build_id": "pending-generation",
                        "phase": "embedding",
                    }
                )
        return await super().call_tool(name, arguments, **kwargs)


def _client(project: Path) -> tuple[TestClient, FakeResearchClient]:
    config = resolve_config(project, vanilla_executable=sys.executable)
    fake = FakeResearchClient()
    return TestClient(create_ui_app(config, research_client=fake)), fake


def test_ui_serves_workspace_and_read_apis(project: Path) -> None:
    client, fake = _client(project)
    with client:
        page = client.get("/")
        assert page.status_code == 200
        assert "UltraRAG MCP" in page.text
        assert "Evidence, with its provenance intact" not in page.text
        assert page.text.index('id="search-view"') < page.text.index(
            'id="status-cards"'
        )
        assert "default-src 'self'" in page.headers["content-security-policy"]

        css = client.get("/assets/app.css")
        javascript = client.get("/assets/app.js")
        assert css.status_code == 200
        assert css.headers["content-type"].startswith("text/css")
        assert javascript.status_code == 200
        assert "loadWorkspace" in javascript.text
        assert client.get("/assets/unknown.js").status_code == 404

        health = client.get("/api/health")
        profile = client.get("/api/ui")
        status = client.get("/api/status")
        sources = client.get("/api/sources?categories=theory,history")
        context = client.get("/api/passages/chunk-1?context_chunks=2")

    assert health.json()["project_root"] == str(project)
    assert profile.json()["application_name"] == "Research UltraRAG"
    assert profile.json()["capabilities"]["metadata"] is True
    assert profile.json()["capabilities"]["bundle_export"] is False
    assert profile.json()["capabilities"]["bundle_import"] is False
    assert profile.json()["capabilities"]["force_recompute"] is True
    assert profile.json()["capabilities"]["source_selection"] is True
    assert profile.json()["capabilities"]["category_partitions"] is True
    assert profile.json()["capabilities"]["project_metadata"] is True
    assert profile.json()["capabilities"]["metadata_filters"] is True
    assert profile.json()["capabilities"]["bibliographic_filters"] is True
    assert profile.json()["capabilities"]["retrieval_modes"] is False
    assert profile.json()["capabilities"]["reranking"] is False
    assert profile.json()["capabilities"]["chunk_settings"] is False
    # The adapter leaves the shared UI's neutral result label in place: the quote
    # rule is stated once for a reader and once for an agent, not per passage.
    assert "not for direct quotation" not in json.dumps(profile.json()).lower()
    assert status.json()["generation_id"] == "generation-1"
    assert sources.json()["sources"][0]["title"] == "Evidence"
    assert context.json()["requested_chunk_id"] == "chunk-1"
    assert ("status", {}) in fake.calls
    assert ("list_sources", {}) in fake.calls
    assert ("get_passage", {"chunk_id": "chunk-1"}) in fake.calls


def test_ui_forwards_search_and_the_surviving_mutations(project: Path) -> None:
    client, fake = _client(project)
    with client:
        search = client.post(
            "/api/search",
            json={"query": "research question", "top_k": 5},
        )
        narrowed = client.post(
            "/api/search",
            json={
                "query": "research question",
                "top_k": 3,
                "categories_any": ["Commodity fetishism"],
                "authors_any": ["Crawford"],
                "titles_any": ["Atlas of AI"],
                "source_ids": ["src_1"],
                "exclude_source_ids": ["src_2"],
            },
        )
        inclusion = client.post(
            "/api/source-inclusion",
            json={
                "source_path": "evidence.pdf",
                "included": False,
                "reason": "Reviewed duplicate",
            },
        )
        ingestion = client.post(
            "/api/ingest",
            json={"force_recompute": True},
        )
        saved_metadata = client.post(
            "/api/source-metadata",
            json={
                "source_path": "evidence.pdf",
                "metadata": {"categories": ["theory"]},
            },
        )
        retired_export = client.post("/api/bundles/export", json={})
        retired_import = client.post(
            "/api/bundles/import",
            json={
                "bundle_name": "research-generation.research-rag.zip",
                "activate": True,
            },
        )

    assert search.json()["hits"][0]["citation"].endswith("p. 1")
    assert inclusion.json()["included"] is False
    assert ingestion.json()["generation_id"] == "generation-2"
    # Metadata is a capability again; the bundle operations stay retired.
    assert saved_metadata.status_code == 200
    assert saved_metadata.json()["message"] == "Reviewed metadata saved."
    assert (
        "set_source_metadata",
        {"source_path": "evidence.pdf", "metadata": {"categories": ["theory"]}},
    ) in fake.calls
    assert retired_export.status_code == 404
    assert retired_import.status_code == 404
    assert ("search", {"query": "research question", "top_k": 5}) in fake.calls
    assert (
        "search",
        {
            "query": "research question",
            "top_k": 3,
            "categories_any": ["Commodity fetishism"],
            "authors_any": ["Crawford"],
            "titles_any": ["Atlas of AI"],
            "source_ids": ["src_1"],
            "exclude_source_ids": ["src_2"],
        },
    ) in fake.calls
    assert narrowed.json()["hits"][0]["citation"].endswith("p. 1")
    assert ("ingest", {"force_recompute": True}) in fake.calls


def test_ui_repeats_batched_ingestion_until_ready(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    fake = BatchedIngestResearchClient()
    app = create_ui_app(config, research_client=fake)

    with TestClient(app) as client:
        response = client.post("/api/ingest", json={})

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert fake.ingest_calls == 2


def test_ui_rejects_unsafe_writes_and_source_paths(project: Path) -> None:
    source = project / "sources" / "evidence.pdf"
    source.write_bytes(b"%PDF-1.4\n% test\n")
    client, _fake = _client(project)
    with client:
        served = client.get("/api/source-file?path=evidence.pdf")
        traversal = client.get("/api/source-file?path=../secret.pdf")
        non_json = client.post("/api/search", content=b"query=test")
        cross_origin = client.post(
            "/api/search",
            headers={"Origin": "https://example.com"},
            json={"query": "test"},
        )
        unknown = client.post(
            "/api/search",
            json={"query": "test", "unsupported": True},
        )

    assert served.status_code == 200
    assert served.headers["content-type"].startswith("application/pdf")
    assert traversal.status_code == 400
    assert non_json.status_code == 415
    assert cross_origin.status_code == 403
    assert unknown.status_code == 400


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def _wait_until_ready(embedded: EmbeddedUi) -> None:
    for _ in range(300):
        if embedded.ready:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"embedded UI never became ready: {embedded.error}")


def _get_json(url_host: str, port: int, path: str) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection(url_host, port, timeout=10)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()


async def _assert_embedded_ui_serves_and_stops(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    port = _free_loopback_port()
    embedded = EmbeddedUi(config, port=port)

    assert embedded.url == f"http://127.0.0.1:{port}"
    assert embedded.ready is False
    assert embedded.error is None

    await embedded.start(research_client=FakeResearchClient())
    try:
        await _wait_until_ready(embedded)
        # The HTTP call must not run on this loop: the embedded server shares it,
        # so a blocking request here would deadlock the very server it calls.
        status_code, payload = await asyncio.to_thread(
            _get_json, embedded.host, port, "/api/health"
        )
        assert status_code == 200
        assert payload["status"] == "ok"
    finally:
        await embedded.stop()

    assert embedded.ready is False


def test_embedded_ui_serves_loopback_and_stops(project: Path) -> None:
    asyncio.run(_assert_embedded_ui_serves_and_stops(project))


async def _assert_embedded_ui_reports_a_used_port(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = int(busy.getsockname()[1])
        embedded = EmbeddedUi(config, port=port)
        await embedded.start(research_client=FakeResearchClient())

    # A taken port is reported, never raised: the MCP server keeps serving.
    assert embedded.ready is False
    assert embedded.error is not None
    assert str(port) in embedded.error


def test_embedded_ui_reports_a_used_port(project: Path) -> None:
    asyncio.run(_assert_embedded_ui_reports_a_used_port(project))


def test_ui_port_validation() -> None:
    assert _ui_port_from(None) is None
    assert _ui_port_from("") is None
    assert _ui_port_from(5051) == 5051
    assert _ui_port_from("5051") == 5051

    for invalid in (0, 65536, -1, "not-a-port"):
        with pytest.raises(SystemExit):
            _ui_port_from(invalid)


def test_ui_port_has_no_environment_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RESEARCH_ULTRARAG_UI_PORT", "5151")

    args = _parser().parse_args(["--project-root", str(tmp_path)])

    # Serving a UI is the decision of whoever started this server, so no ambient
    # variable may hand one to a server this project starts for itself.
    assert args.ui_port is None


def test_a_managed_child_refuses_to_serve_a_ui(monkeypatch) -> None:
    monkeypatch.setenv(MANAGED_CHILD_ENV, "1")

    assert _ui_port_decision(None) is None
    with pytest.raises(SystemExit) as refusal:
        _ui_port_decision(5051)
    assert "started by another server" in str(refusal.value)

    monkeypatch.delenv(MANAGED_CHILD_ENV)
    assert _ui_port_decision(5051) == 5051


def test_child_environment_is_marked_and_drops_top_level_settings(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_ULTRARAG_UI_PORT", "5151")
    monkeypatch.delenv(MANAGED_CHILD_ENV, raising=False)

    environment = child_process_environment()

    for name in TOP_LEVEL_ONLY_ENV:
        assert name not in environment
    assert environment[MANAGED_CHILD_ENV] == "1"
    assert environment["PATH"] == os.environ["PATH"]


def test_child_transports_carry_the_marker_and_no_top_level_settings(
    project: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RESEARCH_ULTRARAG_UI_PORT", "5151")
    monkeypatch.delenv(MANAGED_CHILD_ENV, raising=False)
    config = resolve_config(project, vanilla_executable=sys.executable)

    research = create_research_transport(
        config, log_file=config.logs_root / "research-ui-mcp-stderr.log"
    )
    gateway = create_vanilla_transport(config)

    for transport in (research, gateway):
        assert transport.env is not None
        for name in TOP_LEVEL_ONLY_ENV:
            assert name not in transport.env
        assert transport.env[MANAGED_CHILD_ENV] == "1"
        # A UI port is never an argument a child receives either.
        assert "--ui-port" not in transport.args
    assert research.env["PATH"] == os.environ["PATH"]


async def _assert_ui_claim_is_exclusive_and_released(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    port = _free_loopback_port()
    first = EmbeddedUi(config, port=port)
    second = EmbeddedUi(config, port=port)

    await first.start(research_client=FakeResearchClient())
    try:
        await _wait_until_ready(first)
        # The claim is exclusive because it listens: a socket that is only bound
        # can still be bound again under SO_REUSEADDR, which is the trap a plain
        # probe-before-bind falls into.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with pytest.raises(OSError):
                probe.bind(("127.0.0.1", port))

        await second.start(research_client=FakeResearchClient())

        # The claim is the bind and the listen, so the loser of a simultaneous
        # start fails before it serves anything and says why instead of reporting
        # nothing: the message is the claim's, not a downstream uvicorn failure.
        assert second.ready is False
        assert second.error is not None
        assert "already in use" in second.error
        assert str(port) in second.error
    finally:
        await first.stop()

    # The verdict is not cached: a released port is claimable again.
    third = EmbeddedUi(config, port=port)
    await third.start(research_client=FakeResearchClient())
    try:
        await _wait_until_ready(third)
        assert third.ready is True
        assert third.error is None
    finally:
        await third.stop()


def test_embedded_ui_claim_is_exclusive_and_released(project: Path) -> None:
    asyncio.run(_assert_ui_claim_is_exclusive_and_released(project))


def _process_alive(pid: int) -> bool:
    """True while one process exists and is not a zombie."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat.rsplit(") ", 1)[1].split()[0] != "Z"


def test_a_server_ends_when_its_declared_owner_is_gone(tmp_path: Path) -> None:
    """A server must not outlive the process that started it.

    A busy stdio server does not read stdin until the phase it is in returns, so
    a client that dies in the middle of a build used to leave the build running
    for a project nobody was watching, with a gateway below it. Every spawner in
    this package names itself the owner in the child's environment, which is what
    lets the child end itself even when it is orphaned before it can look at its
    own parent: the parent here exits in the instant after spawning.

    The wait is generous and the marker tells the two possible failures apart.
    Reaching the watchdog means importing the whole server, which on a loaded
    machine reads hundreds of files off the disk it shares with every other
    process, so a child can take tens of seconds to get there; that is a slow
    start, not a broken watchdog, and the marker says which one happened.
    """

    started = tmp_path / "watchdog-started"
    code = (
        "from pathlib import Path;"
        "from research_ultra_rag_mcp.server import watch_owner;"
        "import time;"
        f"marker = Path({str(started)!r});"
        "watch_owner(0.2);"
        "marker.write_text('started');"
        "time.sleep(300)"
    )
    parent = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import subprocess, sys;"
                "from research_ultra_rag_mcp.config import child_process_environment;"
                f"child = subprocess.Popen([sys.executable, '-c', {code!r}],"
                " env=child_process_environment(),"
                " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);"
                "print(child.pid, flush=True)"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert parent.returncode == 0, parent.stderr
    child = int(parent.stdout.strip())
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and _process_alive(child):
            time.sleep(0.25)
        if _process_alive(child):
            if not started.is_file():
                pytest.fail(
                    "the child never reached watch_owner within 120 s, so its "
                    "interpreter was still starting rather than its watchdog "
                    "failing to end it"
                )
            pytest.fail("the orphaned server is still running after its owner exited")
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)


def test_a_server_with_a_living_owner_keeps_running() -> None:
    """The watchdog watches the owner, not the clock."""

    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from research_ultra_rag_mcp.server import watch_owner; import time; "
                "watch_owner(0.2); time.sleep(30)"
            ),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=child_process_environment(),
    )
    try:
        time.sleep(1.5)
        assert _process_alive(child.pid), "the watchdog stopped a live server"
    finally:
        child.terminate()
        child.wait(timeout=30)

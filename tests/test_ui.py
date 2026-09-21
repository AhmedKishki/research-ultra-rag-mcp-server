from __future__ import annotations

import asyncio
import http.client
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.testclient import TestClient

from research_ultra_rag_mcp.config import resolve_config
from research_ultra_rag_mcp.server import _ui_port_from
from research_ultra_rag_mcp.ui import EmbeddedUi, create_ui_app


class FakeResearchClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

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
            "set_source_metadata": {
                "source_path": arguments.get("source_path"),
                "metadata": arguments.get("metadata"),
                "changed": True,
                "effective_immediately": True,
                "requires_ingest": False,
                "generation_metadata_snapshot_outdated": True,
                "message": "Metadata saved and applied immediately.",
            },
            "set_source_inclusion": {
                "source_relative_path": arguments.get("source_path"),
                "included": arguments.get("included"),
                "message": "Inclusion saved.",
            },
            "ingest": {
                "status": "ready",
                "generation_id": "generation-2",
                "chunk_count": 2,
            },
            "export_bundle": {
                "bundle_name": "research-generation.research-rag.zip",
                "sha256": "abc123",
            },
            "import_bundle": {
                "generation_id": "generation-imported",
                "activated": arguments.get("activate", True),
                "message": "Bundle imported and selected.",
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
    assert profile.json()["capabilities"]["bundle_export"] is True
    assert profile.json()["capabilities"]["force_recompute"] is True
    assert profile.json()["capabilities"]["source_selection"] is True
    assert profile.json()["capabilities"]["category_partitions"] is True
    assert profile.json()["result_text_label"].startswith("Cleaned semantic text")
    assert status.json()["generation_id"] == "generation-1"
    assert sources.json()["sources"][0]["title"] == "Evidence"
    assert context.json()["requested_chunk_id"] == "chunk-1"
    assert ("status", {}) in fake.calls
    assert (
        "list_sources",
        {"categories": ["theory", "history"], "categories_any": None, "keywords": None},
    ) in fake.calls
    assert ("get_passage", {"chunk_id": "chunk-1", "context_chunks": 2}) in fake.calls


def test_ui_forwards_search_and_project_mutations(project: Path) -> None:
    client, fake = _client(project)
    with client:
        search = client.post(
            "/api/search",
            json={
                "query": "research question",
                "top_k": 5,
                "retrieval_method": "hybrid",
                "rerank": False,
            },
        )
        narrowed = client.post(
            "/api/search",
            json={
                "query": "research question",
                "top_k": 3,
                "categories_any": ["Commodity fetishism"],
                "source_ids": ["src_1"],
                "exclude_source_ids": ["src_2"],
            },
        )
        metadata = client.post(
            "/api/source-metadata",
            json={
                "source_path": "evidence.pdf",
                "metadata": {"categories": ["theory"]},
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
            json={
                "chunk_size": 384,
                "chunk_overlap": 64,
                "force_recompute": True,
            },
        )
        exported = client.post("/api/bundles/export", json={})
        imported = client.post(
            "/api/bundles/import",
            json={
                "bundle_name": "research-generation.research-rag.zip",
                "activate": True,
            },
        )

    assert search.json()["hits"][0]["citation"].endswith("p. 1")
    assert metadata.json()["effective_immediately"] is True
    assert metadata.json()["requires_ingest"] is False
    assert inclusion.json()["included"] is False
    assert ingestion.json()["generation_id"] == "generation-2"
    assert exported.json()["bundle_name"].endswith(".research-rag.zip")
    assert imported.json()["generation_id"] == "generation-imported"
    assert (
        "search",
        {
            "query": "research question",
            "top_k": 5,
            "retrieval_method": "hybrid",
            "rerank": False,
        },
    ) in fake.calls
    assert (
        "search",
        {
            "query": "research question",
            "top_k": 3,
            "categories_any": ["Commodity fetishism"],
            "source_ids": ["src_1"],
            "exclude_source_ids": ["src_2"],
        },
    ) in fake.calls
    assert narrowed.json()["hits"][0]["citation"].endswith("p. 1")
    assert ("export_bundle", {}) in fake.calls
    assert (
        "ingest",
        {
            "chunk_size": 384,
            "chunk_overlap": 64,
            "force_recompute": True,
        },
    ) in fake.calls
    assert (
        "import_bundle",
        {
            "bundle_name": "research-generation.research-rag.zip",
            "activate": True,
        },
    ) in fake.calls


def test_ui_repeats_batched_ingestion_until_ready(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    fake = BatchedIngestResearchClient()
    app = create_ui_app(config, research_client=fake)

    with TestClient(app) as client:
        response = client.post(
            "/api/ingest",
            json={"chunk_size": 384, "chunk_overlap": 64},
        )

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

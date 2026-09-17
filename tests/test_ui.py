from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from starlette.testclient import TestClient

from research_ultra_rag_mcp.config import resolve_config
from research_ultra_rag_mcp.ui import create_ui_app


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
                "source_root": "/research/project/sources",
                "generation_id": "generation-1",
                "created_at": "2026-09-17T12:00:00Z",
                "selected_source_count": 1,
                "indexed_source_count": 1,
                "searchable_source_count": 1,
                "excluded_source_count": 0,
                "chunk_count": 1,
                "hybrid_ready": True,
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
                        "text": "A citable passage.",
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
                        "text": "A citable passage.",
                    }
                ],
            },
            "set_source_metadata": {
                "source_path": arguments.get("source_path"),
                "metadata": arguments.get("metadata"),
                "requires_ingest": True,
                "message": "Metadata saved.",
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
        }
        return SimpleNamespace(data=responses[name])


def _client(project: Path) -> tuple[TestClient, FakeResearchClient]:
    config = resolve_config(project, vanilla_executable=sys.executable)
    fake = FakeResearchClient()
    return TestClient(create_ui_app(config, research_client=fake)), fake


def test_ui_serves_workspace_and_read_apis(project: Path) -> None:
    client, fake = _client(project)
    with client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Research UltraRAG" in page.text
        assert "default-src 'self'" in page.headers["content-security-policy"]

        css = client.get("/assets/app.css")
        javascript = client.get("/assets/app.js")
        assert css.status_code == 200
        assert css.headers["content-type"].startswith("text/css")
        assert javascript.status_code == 200
        assert "loadWorkspace" in javascript.text
        assert client.get("/assets/unknown.js").status_code == 404

        health = client.get("/api/health")
        status = client.get("/api/status")
        sources = client.get("/api/sources?categories=theory,history")
        context = client.get("/api/passages/chunk-1?context_chunks=2")

    assert health.json()["project_root"] == str(project)
    assert status.json()["generation_id"] == "generation-1"
    assert sources.json()["sources"][0]["title"] == "Evidence"
    assert context.json()["requested_chunk_id"] == "chunk-1"
    assert ("status", {}) in fake.calls
    assert (
        "list_sources",
        {"categories": ["theory", "history"], "keywords": None},
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
            json={"chunk_size": 384, "chunk_overlap": 64},
        )

    assert search.json()["hits"][0]["citation"].endswith("p. 1")
    assert metadata.json()["requires_ingest"] is True
    assert inclusion.json()["included"] is False
    assert ingestion.json()["generation_id"] == "generation-2"
    assert (
        "search",
        {
            "query": "research question",
            "top_k": 5,
            "retrieval_method": "hybrid",
            "rerank": False,
        },
    ) in fake.calls


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

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from conftest import write_pdf
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from research_ultra_rag_mcp.instructions import SERVER_INSTRUCTIONS


async def _assert_real_stdio_research_flow(project: Path) -> None:
    write_pdf(
        project / "sources" / "evidence.pdf",
        [
            (
                "The cobalt heron is evidence for a page-aware research result. "
                "Its habitat is the amber marsh, where labour and ecology meet."
            )
        ],
        title="Citable Evidence",
    )
    (project / "sources" / "notes.md").write_text(
        "cobalt heron derived notes must not be indexed",
        encoding="utf-8",
    )
    executable = Path(sys.executable).parent / "research-ultra-rag-mcp"
    transport = StdioTransport(
        command=str(executable),
        args=["--project-root", str(project)],
        log_file=project / "research-server-stderr.log",
    )

    async with Client(transport, timeout=1800, init_timeout=1800) as client:
        assert client.initialize_result is not None
        assert client.initialize_result.instructions == SERVER_INSTRUCTIONS
        tool_records = await client.list_tools()
        tools = {tool.name: tool for tool in tool_records}
        assert set(tools) == {
            "get_passage",
            "ingest",
            "list_sources",
            "search",
            "set_source_metadata",
            "status",
        }
        search_properties = tools["search"].inputSchema["properties"]
        assert search_properties["retrieval_method"]["default"] == "hybrid"
        assert set(search_properties["retrieval_method"]["enum"]) == {
            "hybrid",
            "bm25",
            "dense",
        }
        assert search_properties["rerank"]["default"] is False

        initial = await client.call_tool("status", {})
        assert initial.data["ready"] is False
        assert initial.data["selected_source_count"] == 1
        assert initial.data["ignored_extensions"] == {".md": 1}

        metadata = await client.call_tool(
            "set_source_metadata",
            {
                "source_path": "evidence.pdf",
                "metadata": {
                    "categories": ["research"],
                    "keywords": ["wetland"],
                },
            },
        )
        assert metadata.data["requires_ingest"] is True

        ingested = await client.call_tool(
            "ingest",
            {"chunk_size": 50, "chunk_overlap": 10},
            timeout=1800,
        )
        assert ingested.data["document_count"] == 1
        assert ingested.data["ignored_extensions"] == {".md": 1}
        assert ingested.data["default_retrieval_method"] == "hybrid"

        current = json.loads(
            (project / ".ultrarag" / "research" / "current.json").read_text(
                encoding="utf-8"
            )
        )
        generation_root = (
            project
            / ".ultrarag"
            / "research"
            / "generations"
            / current["generation_id"]
        )
        manifest = json.loads(
            (generation_root / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["retrieval"]["dense"]["point_count"] == 1
        assert (generation_root / "indexes" / "qdrant").is_dir()

        ready = await client.call_tool("status", {})
        assert ready.data["hybrid_ready"] is True

        result = await client.call_tool(
            "search",
            {"query": "cobalt heron amber marsh", "top_k": 1},
        )
        hit = result.data["hits"][0]
        assert hit["title"] == "Citable Evidence"
        assert hit["source_path"] == "sources/evidence.pdf"
        assert hit["locator"]["page"] == 1
        assert "cobalt heron" in hit["text"].lower()
        assert "notes" not in hit["text"].lower()
        assert result.data["retrieval_method"] == "hybrid"
        assert hit["component_ranks"] == {"bm25": 1, "dense": 1}
        assert hit["fusion_score"] is not None

        dense_result = await client.call_tool(
            "search",
            {
                "query": "wetland bird and work",
                "top_k": 1,
                "retrieval_method": "dense",
                "categories": ["research"],
                "keywords": ["wetland"],
            },
        )
        assert dense_result.data["hits"][0]["chunk_id"] == hit["chunk_id"]
        assert (
            dense_result.data["hits"][0]["component_scores"]["dense_cosine_similarity"]
            is not None
        )

        filtered_out = await client.call_tool(
            "search",
            {
                "query": "wetland bird",
                "top_k": 1,
                "retrieval_method": "dense",
                "categories": ["unrelated"],
            },
        )
        assert filtered_out.data["result_count"] == 0

        reranked = await client.call_tool(
            "search",
            {
                "query": "cobalt heron amber marsh",
                "top_k": 1,
                "retrieval_method": "hybrid",
                "rerank": True,
            },
            timeout=1800,
        )
        assert reranked.data["reranked"] is True
        assert reranked.data["hits"][0]["rerank_score"] is not None

    offline_transport = StdioTransport(
        command=str(executable),
        args=["--project-root", str(project), "--offline"],
        log_file=project / "research-server-offline-stderr.log",
    )
    async with Client(
        offline_transport,
        timeout=1800,
        init_timeout=1800,
    ) as offline_client:
        offline_result = await offline_client.call_tool(
            "search",
            {
                "query": "cobalt heron amber marsh",
                "top_k": 1,
                "rerank": True,
            },
            timeout=1800,
        )
        assert offline_result.data["retrieval_method"] == "hybrid"
        assert offline_result.data["hits"][0]["rerank_score"] is not None


@pytest.mark.integration
def test_real_vanilla_ultrarag_research_flow(project: Path) -> None:
    asyncio.run(_assert_real_stdio_research_flow(project))

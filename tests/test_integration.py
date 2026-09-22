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

# Answer-level keys that only the full-detail payload carries.
LEAN_ONLY_ABSENT = (
    "allowed_formats",
    "generation_root",
    "ignored_extensions",
    "last_build_metrics",
    "retrieval",
    "source_exclusion_revision",
    "ui_launcher",
    "version",
)

# The same rule one level down, for a returned passage.
LEAN_ONLY_HIT_KEYS = (
    "component_ranks",
    "component_scores",
    "doi",
    "document_id",
    "fusion_score",
    "metadata_provenance",
    "metadata_warnings",
    "rank",
    "rerank_score",
    "source_path",
    "text_fidelity",
    "year",
)


async def _ingest_until_complete(
    client: Client,
    arguments: dict[str, object],
):
    while True:
        result = await client.call_tool("ingest", arguments, timeout=1800)
        if result.data["status"] != "in_progress":
            return result


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
            "set_source_inclusion",
            "status",
        }
        assert tools["set_source_inclusion"].annotations is not None
        assert tools["set_source_inclusion"].annotations.destructiveHint is True
        expected_parameters = {
            "ingest": {
                "chunk_size",
                "chunk_overlap",
                "force_recompute",
                "work_budget_seconds",
            },
            "search": {
                "query",
                "top_k",
                "result_view",
                "passages_per_reference",
                "categories",
                "categories_any",
                "projects",
                "projects_any",
                "keywords",
                "document_ids",
                "source_ids",
                "exclude_source_ids",
                "retrieval_method",
                "rerank",
                "include_staleness",
            },
            "list_sources": {
                "categories",
                "categories_any",
                "projects",
                "projects_any",
                "keywords",
            },
            "get_passage": {"chunk_id", "context_chunks"},
            "set_source_inclusion": {
                "source_id",
                "source_path",
                "included",
                "reason",
            },
        }
        for tool_name, parameter_names in expected_parameters.items():
            properties = tools[tool_name].inputSchema["properties"]
            assert set(properties) == parameter_names
            assert all(properties[name].get("description") for name in properties)

        search_properties = tools["search"].inputSchema["properties"]
        assert search_properties["retrieval_method"]["default"] == "hybrid"
        assert set(search_properties["retrieval_method"]["enum"]) == {
            "hybrid",
            "bm25",
            "dense",
        }
        assert search_properties["rerank"]["default"] is True
        assert search_properties["result_view"]["default"] == "passages"
        assert set(search_properties["result_view"]["enum"]) == {
            "passages",
            "references",
        }
        passages_per_reference = search_properties["passages_per_reference"]
        assert passages_per_reference["default"] == 2
        assert passages_per_reference["minimum"] == 1
        assert passages_per_reference["maximum"] == 5
        assert search_properties["include_staleness"]["default"] is True
        work_budget = tools["ingest"].inputSchema["properties"]["work_budget_seconds"]
        assert work_budget["default"] == 45
        assert work_budget["minimum"] == 10
        assert work_budget["maximum"] == 300

        initial = await client.call_tool("status", {})
        assert initial.data["ready"] is False
        assert initial.data["selected_source_count"] == 1
        for key in LEAN_ONLY_ABSENT:
            assert key not in initial.data

        # Reviewed metadata is a hand-edited review-state file; it applies at
        # read time, so it is written directly and checked through the tools.
        review_state = project / ".research-rag" / "source-metadata.json"
        review_state.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sources": {
                        "evidence.pdf": {
                            "categories": ["research"],
                            "keywords": ["wetland"],
                        }
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        ingested = await _ingest_until_complete(
            client,
            {"chunk_size": 50, "chunk_overlap": 10},
        )
        assert ingested.data["status"] == "ready"
        assert ingested.data["generation_changed"] is True
        assert ingested.data["document_count"] == 1
        assert "phase_timings_seconds" not in ingested.data
        assert "embedding_model" not in ingested.data

        current = json.loads(
            (project / ".research-rag" / "runtime" / "current.json").read_text(
                encoding="utf-8"
            )
        )
        generation_root = (
            project
            / ".research-rag"
            / "runtime"
            / "generations"
            / current["generation_id"]
        )
        manifest = json.loads(
            (generation_root / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["retrieval"]["dense"]["point_count"] == 1
        # The default `auto` backend is the exact scan below the corpus
        # threshold, and the manifest records which backend owns the index.
        assert manifest["retrieval"]["dense"]["dense_backend"] == (
            "portable-exact-vectors"
        )
        assert manifest["files"]["dense_index"] == "indexes/vectors"
        assert (generation_root / "indexes" / "vectors" / "index.json").is_file()

        ready = await client.call_tool("status", {})
        assert set(ready.data["available_retrieval_methods"]) == {
            "bm25",
            "dense",
            "hybrid",
        }
        # An upgrade note appears only when an upgrade is actually required.
        assert "generation_upgrade_required" not in ready.data
        # No --ui-port was passed, so this server hosts no UI and says so.
        assert ready.data["ui_url"] is None
        assert ready.data["ui_ready"] is False
        assert ready.data["ui_error"] is None
        assert ready.data["retained_generation_count"] >= 1

        result = await client.call_tool(
            "search",
            {"query": "cobalt heron amber marsh", "top_k": 1},
        )
        initial_hit = result.data["hits"][0]
        source_id = initial_hit["source_id"]
        assert source_id.startswith("src_")
        assert initial_hit["title"] == "Citable Evidence"
        assert initial_hit["source_relative_path"] == "evidence.pdf"
        assert initial_hit["locator"]["page"] == 1
        assert "Citable Evidence" in initial_hit["citation"]
        assert "cobalt heron" in initial_hit["text"].lower()
        assert "notes" not in initial_hit["text"].lower()
        assert initial_hit["direct_quote_safe"] is False
        assert result.data["stale"] is False
        assert result.data["reranked"] is True
        for key in LEAN_ONLY_HIT_KEYS:
            assert key not in initial_hit

        manifest_before_metadata = (generation_root / "manifest.json").read_bytes()
        chunks_before_metadata = (
            generation_root / "chunks" / "chunks.jsonl"
        ).read_bytes()
        review_state.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sources": {
                        "evidence.pdf": {
                            "title": "Reviewed Marsh Evidence",
                            "authors": ["Field Researcher"],
                            "year": 2025,
                            "categories": ["corrected"],
                            "keywords": ["heron"],
                        }
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        assert (
            generation_root / "manifest.json"
        ).read_bytes() == manifest_before_metadata
        assert (
            generation_root / "chunks" / "chunks.jsonl"
        ).read_bytes() == chunks_before_metadata

        metadata_status = await client.call_tool("status", {})
        assert metadata_status.data["stale"] is False
        assert metadata_status.data["metadata_overlay_active"] is True
        assert "immediately" in metadata_status.data["message"]
        # Change lists appear only when the generation is stale, so a
        # metadata-only overlay is reported as an overlay, not as churn.
        assert "changes" not in metadata_status.data

        corrected_sources = await client.call_tool(
            "list_sources",
            {"categories": ["corrected"], "keywords": ["heron"]},
        )
        assert corrected_sources.data["source_count"] == 1
        assert corrected_sources.data["sources"][0]["source_id"] == source_id
        assert corrected_sources.data["sources"][0]["title"] == (
            "Reviewed Marsh Evidence"
        )

        corrected_result = await client.call_tool(
            "search",
            {
                "query": "cobalt heron amber marsh",
                "top_k": 1,
                "categories": ["corrected"],
                "keywords": ["heron"],
            },
        )
        hit = corrected_result.data["hits"][0]
        assert hit["chunk_id"] == initial_hit["chunk_id"]
        assert hit["title"] == "Reviewed Marsh Evidence"
        assert hit["authors"] == ["Field Researcher"]
        assert hit["citation"].startswith(
            "Field Researcher, Reviewed Marsh Evidence (2025)"
        )

        corrected_context = await client.call_tool(
            "get_passage",
            {"chunk_id": hit["chunk_id"], "context_chunks": 0},
        )
        corrected_passage = corrected_context.data["context"][0]
        assert corrected_passage["title"] == "Reviewed Marsh Evidence"
        assert corrected_passage["authors"] == ["Field Researcher"]
        assert "categories" not in corrected_passage
        assert "notice" not in corrected_context.data

        references_result = await client.call_tool(
            "search",
            {
                "query": "cobalt heron amber marsh",
                "top_k": 1,
                "result_view": "references",
                "passages_per_reference": 2,
            },
        )
        # The reference view answers with the groups instead of the flat list.
        assert "hits" not in references_result.data
        reference_group = references_result.data["reference_groups"][0]
        assert reference_group["source_id"] == source_id
        assert reference_group["title"] == "Reviewed Marsh Evidence"
        assert reference_group["passages"][0]["chunk_id"] == hit["chunk_id"]
        assert "document_id" not in reference_group
        assert "categories" not in reference_group

        dense_result = await client.call_tool(
            "search",
            {
                "query": "cobalt heron amber marsh labour ecology",
                "top_k": 1,
                "retrieval_method": "dense",
                "categories": ["corrected"],
                "keywords": ["heron"],
            },
        )
        assert dense_result.data["hits"][0]["chunk_id"] == hit["chunk_id"]
        assert "component_scores" not in dense_result.data["hits"][0]

        unchecked = await client.call_tool(
            "search",
            {
                "query": "cobalt heron amber marsh labour ecology",
                "top_k": 1,
                "include_staleness": False,
            },
        )
        # A null stale verdict means the freshness check was skipped; the
        # `staleness_checked` flag itself stays out of the lean answer.
        assert unchecked.data["stale"] is None
        assert "staleness_checked" not in unchecked.data
        assert unchecked.data["hits"][0]["chunk_id"] == hit["chunk_id"]

        filtered_out = await client.call_tool(
            "search",
            {
                "query": "wetland bird",
                "top_k": 1,
                "retrieval_method": "dense",
                "categories": ["unrelated"],
            },
        )
        assert filtered_out.data["hits"] == []

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
        assert "rerank_score" not in reranked.data["hits"][0]

        excluded = await client.call_tool(
            "set_source_inclusion",
            {
                "source_id": source_id,
                "included": False,
                "reason": "Agent-reviewed duplicate representation test.",
            },
        )
        assert excluded.data["effective_immediately"] is True
        assert (project / "sources" / "evidence.pdf").is_file()
        excluded_search = await client.call_tool(
            "search",
            {"query": "cobalt heron", "top_k": 1},
        )
        assert excluded_search.data["hits"] == []

        restored = await client.call_tool(
            "set_source_inclusion",
            {"source_id": source_id, "included": True},
        )
        assert restored.data["effective_immediately"] is True
        restored_search = await client.call_tool(
            "search",
            {"query": "cobalt heron", "top_k": 1},
        )
        assert len(restored_search.data["hits"]) == 1

    offline_transport = StdioTransport(
        command=str(executable),
        args=[
            "--project-root",
            str(project),
            "--offline",
            # Developer detail mode: the complete payload.
            "--tool-detail",
            "full",
        ],
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
        assert offline_result.data["filters"]["active_document_count"] == 1
        assert offline_result.data["withheld_candidates"]["total"] == 0
        assert (
            offline_result.data["hits"][0]["text_fidelity"] == "cleaned_semantic_text"
        )
        offline_status = await offline_client.call_tool("status", {})
        assert offline_status.data["generation_upgrade_required"] is False
        assert offline_status.data["retrieval"]["available_methods"] == [
            "bm25",
            "dense",
            "hybrid",
        ]


@pytest.mark.integration
def test_real_vanilla_ultrarag_research_flow(project: Path) -> None:
    asyncio.run(_assert_real_stdio_research_flow(project))

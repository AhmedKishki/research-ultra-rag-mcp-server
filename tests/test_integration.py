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
    "available_retrieval_methods",
    "default_retrieval_method",
    "categories",
    "generation_root",
    "generations",
    "ignored_extensions",
    "last_build_metrics",
    "projects",
    "retrieval",
    "source_exclusion_revision",
    "ui_launcher",
    "version",
)

# The same rule one level down, for a returned passage: a citation-ready
# reference and the handles that identify a source a second time belong to the
# full-detail payload alone.
LEAN_ONLY_HIT_KEYS = (
    "citation",
    "component_ranks",
    "component_scores",
    "direct_quote_safe",
    "doi",
    "document_id",
    "fusion_score",
    "metadata_provenance",
    "metadata_warnings",
    "rank",
    "rerank_score",
    "source_id",
    "source_path",
    "text_fidelity",
    "text_notes",
    "title",
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
            "set_source_metadata",
            "status",
        }
        assert tools["set_source_inclusion"].annotations is not None
        assert tools["set_source_inclusion"].annotations.destructiveHint is True
        expected_parameters = {
            "status": set(),
            "ingest": {"force_recompute"},
            "search": {
                "query",
                "top_k",
                "categories_any",
                "projects_any",
                "keywords",
                "languages_any",
                "authors_any",
                "titles_any",
                "source_ids",
                "exclude_source_ids",
            },
            "list_sources": set(),
            "get_passage": {"chunk_id"},
            "set_source_inclusion": {"source_path", "included", "reason"},
        }
        for tool_name, parameter_names in expected_parameters.items():
            properties = tools[tool_name].inputSchema["properties"]
            assert set(properties) == parameter_names
            assert all(properties[name].get("description") for name in properties)

        # The retired knobs are gone from the published surface, not hidden.
        for retired in (
            "retrieval_method",
            "rerank",
            "include_staleness",
            "result_view",
            "passages_per_reference",
            "document_ids",
            "categories",
            "projects",
            "chunk_size",
            "chunk_overlap",
            "work_budget_seconds",
            "context_chunks",
            "source_id",
        ):
            assert retired not in tools["search"].inputSchema["properties"]
            assert retired not in tools["ingest"].inputSchema["properties"]
            assert retired not in tools["list_sources"].inputSchema["properties"]
            assert retired not in tools["get_passage"].inputSchema["properties"]
            assert (
                retired not in tools["set_source_inclusion"].inputSchema["properties"]
            )

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

        ingested = await _ingest_until_complete(client, {})
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
        # This generation is hybrid-ready, so the answer says nothing about
        # methods at all: the tool cannot be asked for a different one.
        assert "hybrid_ready" not in ready.data
        assert "available_retrieval_methods" not in ready.data
        # An upgrade note appears only when an upgrade is actually required.
        assert "generation_upgrade_required" not in ready.data
        # No --ui-port was passed, so this server hosts no UI and says so.
        assert ready.data["ui_url"] is None
        assert ready.data["ui_ready"] is False
        assert ready.data["ui_error"] is None
        # What a prune would consider is the count and the bytes, never a list.
        assert ready.data["retained_generation_count"] >= 1

        # A stale status counts what the corpus gained and reports only what a
        # researcher has to act on; it never enumerates the available sources.
        added_source = project / "sources" / "added-later.pdf"
        write_pdf(added_source, ["Amber marsh evidence added after the build."])
        added_status = await client.call_tool("status", {})
        assert added_status.data["stale"] is True
        assert added_status.data["changes"] == {"added_source_count": 1}
        assert "added-later.pdf" not in json.dumps(added_status.data)

        removed_source = project / "sources" / "evidence.pdf"
        removed_bytes = removed_source.read_bytes()
        removed_source.unlink()
        missing_status = await client.call_tool("status", {})
        assert missing_status.data["changes"]["removed_sources"] == ["evidence.pdf"]
        assert "added-later.pdf" not in json.dumps(missing_status.data)
        removed_source.write_bytes(removed_bytes)
        added_source.unlink()
        settled_status = await client.call_tool("status", {})
        assert settled_status.data["stale"] is False
        assert "changes" not in settled_status.data

        result = await client.call_tool(
            "search",
            {"query": "cobalt heron amber marsh", "top_k": 1},
        )
        initial_hit = result.data["hits"][0]
        assert initial_hit["source_relative_path"] == "evidence.pdf"
        assert initial_hit["locator"] == {"page": 1}
        assert "cobalt heron" in initial_hit["text"].lower()
        assert "notes" not in initial_hit["text"].lower()
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

        corrected_sources = await client.call_tool("list_sources", {})
        assert corrected_sources.data["source_count"] == 1
        source_record = corrected_sources.data["sources"][0]
        # The stable ID is the inventory's business, not a search answer's.
        assert source_record["source_id"].startswith("src_")
        assert source_record["title"] == "Reviewed Marsh Evidence"

        corrected_result = await client.call_tool(
            "search",
            {
                "query": "cobalt heron amber marsh",
                "top_k": 1,
                "categories_any": ["corrected"],
                "keywords": ["heron"],
            },
        )
        hit = corrected_result.data["hits"][0]
        assert hit["chunk_id"] == initial_hit["chunk_id"]
        # The overlay reaches a search answer, and the inventory carries the
        # rest of the reviewed bibliography.
        assert hit["authors"] == ["Field Researcher"]
        assert hit["locator"] == {"page": 1}
        for key in LEAN_ONLY_HIT_KEYS:
            assert key not in hit

        corrected_context = await client.call_tool(
            "get_passage",
            {"chunk_id": hit["chunk_id"]},
        )
        corrected_passage = corrected_context.data["context"][0]
        assert corrected_passage["source_relative_path"] == "evidence.pdf"
        assert corrected_passage["authors"] == ["Field Researcher"]
        assert corrected_passage["locator"] == {"page": 1}
        assert "categories" not in corrected_passage
        assert corrected_context.data["requested_chunk_id"] == hit["chunk_id"]

        filtered_out = await client.call_tool(
            "search",
            {
                "query": "wetland bird",
                "top_k": 1,
                "categories_any": ["unrelated"],
            },
        )
        assert filtered_out.data["hits"] == []

        # The reviewed author and title are what a name filter matches, through
        # the real tool surface, and a substring is enough for either.
        by_name = await client.call_tool(
            "search",
            {
                "query": "cobalt heron amber marsh",
                "top_k": 1,
                "authors_any": ["field researcher"],
                "titles_any": ["marsh evidence"],
            },
        )
        assert by_name.data["hits"][0]["chunk_id"] == hit["chunk_id"]

        # A name no source carries is a filtered answer rather than a silent
        # corpus, so the answer names the filter it applied.
        unmatched_name = await client.call_tool(
            "search",
            {
                "query": "cobalt heron amber marsh",
                "top_k": 1,
                "authors_any": ["Nobody At All"],
            },
        )
        assert unmatched_name.data["hits"] == []
        assert unmatched_name.data["applied_filters"] == {
            "authors_any": ["nobody at all"]
        }

        # The extracted title is still reachable while a reviewed one exists,
        # because the filter reads the overlay rather than the source filename.
        extracted_title = await client.call_tool(
            "search",
            {
                "query": "cobalt heron amber marsh",
                "top_k": 1,
                "titles_any": ["citable evidence"],
            },
        )
        assert extracted_title.data["hits"] == []

        reranked = await client.call_tool(
            "search",
            {"query": "cobalt heron amber marsh", "top_k": 1},
            timeout=1800,
        )
        assert reranked.data["reranked"] is True
        assert "rerank_score" not in reranked.data["hits"][0]

        excluded = await client.call_tool(
            "set_source_inclusion",
            {
                "source_path": "evidence.pdf",
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
            {"source_path": "evidence.pdf", "included": True},
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
            {"query": "cobalt heron amber marsh", "top_k": 1},
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

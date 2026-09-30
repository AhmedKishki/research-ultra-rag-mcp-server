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
            "find_source",
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
            "find_source": {"query", "limit"},
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
            assert retired not in tools["find_source"].inputSchema["properties"]
            assert retired not in tools["get_passage"].inputSchema["properties"]
            assert (
                retired not in tools["set_source_inclusion"].inputSchema["properties"]
            )

        initial = await client.call_tool("status", {})
        # There is nothing to serve yet, and that is the one readiness fact worth
        # saying. The counts of what was found are the inventory the command line
        # and the workspace read, not something a readiness answer repeats.
        assert initial.data["ready"] is False
        assert "selected_source_count" not in initial.data
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
        # The verdict is stated, and the disk inventory is not part of it: an
        # agent that needs it is answered by the command line.
        assert ready.data["ready"] is True
        assert ready.data["stale"] is False
        assert "requires" not in ready.data
        assert "retained_generation_count" not in ready.data
        assert "retained_generation_bytes" not in ready.data

        # A stale status counts what the corpus gained and reports only what a
        # researcher has to act on; it never enumerates the available sources.
        added_source = project / "sources" / "added-later.pdf"
        write_pdf(added_source, ["Amber marsh evidence added after the build."])
        added_status = await client.call_tool("status", {})
        assert added_status.data["stale"] is True
        assert added_status.data["requires"] == ["ingest"]
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
        # Settled says so outright rather than by leaving the field out, and
        # nothing is required of the caller.
        assert settled_status.data["ready"] is True
        assert settled_status.data["stale"] is False
        assert "requires" not in settled_status.data
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
        assert "stale" not in result.data
        assert "reranked" not in result.data
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
        # A reviewed change is not staleness and needs no rebuild, so the verdict
        # says so and the message carries the overlay.
        assert metadata_status.data["stale"] is False
        assert "requires" not in metadata_status.data
        assert "metadata_overlay_active" not in metadata_status.data
        assert "immediately" in metadata_status.data["message"]
        # Change lists appear only when the generation is stale, so a
        # metadata-only overlay is reported as an overlay, not as churn.
        assert "changes" not in metadata_status.data

        corrected_sources = await client.call_tool("find_source", {"query": "marsh"})
        assert corrected_sources.data["match_count"] == 1
        source_record = corrected_sources.data["matches"][0]
        # The stable ID is the lookup's business, not a search answer's.
        assert source_record["source_id"].startswith("src_")
        assert source_record["title"] == "Reviewed Marsh Evidence"
        assert source_record["indexed_in_current_generation"] is True

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
        assert "requested_chunk_id" not in corrected_context.data

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
        # Reranking is not optional, so an answer never announces that it
        # happened; only a cross-encoder that did not run is news.
        assert "reranked" not in reranked.data
        assert "rerank_score" not in reranked.data["hits"][0]

        excluded = await client.call_tool(
            "set_source_inclusion",
            {
                "source_path": "evidence.pdf",
                "included": False,
                "reason": "Agent-reviewed duplicate representation test.",
            },
        )
        assert "effective_immediately" not in excluded.data
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
        assert "effective_immediately" not in restored.data
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

        # A resource read is the tool's answer, as JSON, so a client that reads
        # context as a resource sees exactly what a tool call would have returned.
        status_resource = await offline_client.read_resource("research://status")
        status_payload = json.loads(status_resource[0].text)
        assert status_payload["ready"] is True
        assert status_payload["generation_id"] == offline_status.data["generation_id"]
        # The surface serves one source lookup rather than the inventory, so a
        # client asks about a name and learns whether that source is searchable.
        found = await offline_client.call_tool("find_source", {"query": "evidence"})
        assert found.data["match_count"] == 1
        assert found.data["matches"][0]["source_relative_path"] == "evidence.pdf"
        assert found.data["matches"][0]["indexed_in_current_generation"] is True


@pytest.mark.integration
def test_the_handshake_does_not_wait_for_the_gateway(project: Path) -> None:
    """A client must see the tools before any retrieval work is attempted.

    The handshake used to wait for the vanilla gateway to start and for the whole
    retrieval stack to import, which is seconds of work no tool had asked for and
    longer than some clients wait before they report a server as unavailable. With
    a gateway that cannot start at all, the property is easy to see: connecting and
    listing tools must succeed, and the failure must arrive as the answer to the
    tool that needed the gateway rather than as a client that never connected.

    That answer must also be usable. `Connection closed` names the symptom and
    nothing else, so the failure carries the reason the gateway wrote to its own
    log, and the paths of the logs a reader has to open to see the rest.
    """

    async def exercise() -> None:
        write_pdf(
            project / "sources" / "evidence.pdf",
            ["The cobalt heron is evidence."],
        )
        # A gateway that exists and cannot serve: configuration resolution accepts
        # it, so the failure can only appear when a tool asks for it.
        broken = project / "broken-vanilla-gateway"
        broken.write_text(
            "#!/bin/sh\necho 'no runtime at /nowhere' >&2\nexit 3\n", encoding="utf-8"
        )
        broken.chmod(0o755)
        transport = StdioTransport(
            command=str(Path(sys.executable).parent / "research-ultra-rag-mcp"),
            args=[
                "--project-root",
                str(project),
                "--vanilla-executable",
                str(broken),
            ],
            log_file=project / "broken-gateway-stderr.log",
        )

        async with Client(transport, timeout=300, init_timeout=300) as client:
            assert client.initialize_result is not None
            tools = {tool.name for tool in await client.list_tools()}
            assert tools == {
                "get_passage",
                "ingest",
                "find_source",
                "search",
                "set_source_inclusion",
                "set_source_metadata",
                "status",
            }
            # Resource metadata is static, so a client that probes resources finds
            # them before any gateway exists; only reading one needs the service.
            resources = {str(item.uri) for item in await client.list_resources()}
            assert resources == {"research://status"}
            with pytest.raises(Exception) as failure:
                await client.call_tool("status", {})
        message = str(failure.value)
        assert "Research workflow failed" in message
        assert "The UltraRAG gateway could not start" in message
        # The reason is the line the gateway itself printed, not the symptom the
        # client would otherwise be left with.
        assert "no runtime at /nowhere" in message
        runtime_root = project / ".research-rag" / "runtime"
        assert str(runtime_root / "logs" / "vanilla-gateway-stderr.log") in message
        assert str(runtime_root / "ultrarag-runtime" / "logs") in message

    asyncio.run(exercise())


@pytest.mark.integration
def test_real_vanilla_ultrarag_research_flow(project: Path) -> None:
    asyncio.run(_assert_real_stdio_research_flow(project))

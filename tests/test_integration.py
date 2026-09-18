from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest
from conftest import write_pdf
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from research_ultra_rag_mcp.instructions import SERVER_INSTRUCTIONS


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
            "export_bundle",
            "get_passage",
            "import_bundle",
            "ingest",
            "list_sources",
            "search",
            "set_source_inclusion",
            "set_source_metadata",
            "status",
        }
        assert tools["set_source_metadata"].annotations is not None
        assert tools["set_source_metadata"].annotations.destructiveHint is True
        assert tools["set_source_inclusion"].annotations is not None
        assert tools["set_source_inclusion"].annotations.destructiveHint is True
        assert tools["import_bundle"].annotations is not None
        assert tools["import_bundle"].annotations.destructiveHint is True
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
                "categories",
                "keywords",
                "document_ids",
                "retrieval_method",
                "rerank",
            },
            "list_sources": {"categories", "keywords"},
            "get_passage": {"chunk_id", "context_chunks"},
            "set_source_metadata": {"source_path", "metadata"},
            "set_source_inclusion": {"source_path", "included", "reason"},
            "export_bundle": set(),
            "import_bundle": {"bundle_name", "activate"},
        }
        for tool_name, parameter_names in expected_parameters.items():
            properties = tools[tool_name].inputSchema["properties"]
            assert set(properties) == parameter_names
            assert all(properties[name].get("description") for name in properties)

        metadata_definition = tools["set_source_metadata"].inputSchema["properties"][
            "metadata"
        ]
        assert metadata_definition["additionalProperties"] is False
        assert set(metadata_definition["properties"]) == {
            "title",
            "authors",
            "year",
            "doi",
            "categories",
            "keywords",
        }
        assert all(
            field.get("description")
            for field in metadata_definition["properties"].values()
        )

        search_properties = tools["search"].inputSchema["properties"]
        assert search_properties["retrieval_method"]["default"] == "hybrid"
        assert set(search_properties["retrieval_method"]["enum"]) == {
            "hybrid",
            "bm25",
            "dense",
        }
        assert search_properties["rerank"]["default"] is False
        work_budget = tools["ingest"].inputSchema["properties"]["work_budget_seconds"]
        assert work_budget["default"] == 45
        assert work_budget["minimum"] == 10
        assert work_budget["maximum"] == 300

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

        ingested = await _ingest_until_complete(
            client,
            {"chunk_size": 50, "chunk_overlap": 10},
        )
        assert ingested.data["document_count"] == 1
        assert ingested.data["ignored_extensions"] == {".md": 1}
        assert ingested.data["default_retrieval_method"] == "hybrid"

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
        assert (generation_root / "indexes" / "qdrant").is_dir()

        ready = await client.call_tool("status", {})
        assert ready.data["hybrid_ready"] is True
        assert ready.data["generation_upgrade_required"] is False

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
        assert hit["direct_quote_safe"] is False
        assert hit["text_fidelity"] == "cleaned_semantic_text"
        assert result.data["requested_top_k"] == 1

        dense_result = await client.call_tool(
            "search",
            {
                "query": "cobalt heron amber marsh labour ecology",
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

        exported = await client.call_tool("export_bundle", {}, timeout=1800)
        assert exported.data["source_count"] == 1
        assert exported.data["bundle_name"].endswith(".research-rag.zip")
        exported_bundle = Path(exported.data["bundle_path"])
        reference_chunk_id = hit["chunk_id"]
        imported = await client.call_tool(
            "import_bundle",
            {"bundle_name": exported.data["bundle_name"], "activate": True},
            timeout=1800,
        )
        assert imported.data["status"] == "already_present"

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
        assert excluded_search.data["result_count"] == 0

        restored = await client.call_tool(
            "set_source_inclusion",
            {"source_path": "evidence.pdf", "included": True},
        )
        assert restored.data["effective_immediately"] is True
        restored_search = await client.call_tool(
            "search",
            {"query": "cobalt heron", "top_k": 1},
        )
        assert restored_search.data["result_count"] == 1

    imported_project = project.parent / "portable-import-project"
    imported_project.mkdir()
    imported_portable = imported_project / ".research-rag"
    imported_portable.mkdir()
    shutil.copy2(
        project / ".research-rag" / "project.json",
        imported_portable / "project.json",
    )
    imported_bundles = imported_portable / "bundles"
    imported_bundles.mkdir()
    copied_bundle = imported_bundles / exported_bundle.name
    shutil.copy2(exported_bundle, copied_bundle)
    shutil.copy2(
        exported_bundle.with_suffix(exported_bundle.suffix + ".sha256"),
        copied_bundle.with_suffix(copied_bundle.suffix + ".sha256"),
    )
    import_transport = StdioTransport(
        command=str(executable),
        args=["--project-root", str(imported_project)],
        log_file=imported_project / "research-import-stderr.log",
    )
    async with Client(
        import_transport,
        timeout=1800,
        init_timeout=1800,
    ) as imported_client:
        reconstructed = await imported_client.call_tool(
            "import_bundle",
            {"bundle_name": copied_bundle.name, "activate": True},
            timeout=1800,
        )
        assert reconstructed.data["status"] == "imported"
        imported_status = await imported_client.call_tool("status", {})
        assert imported_status.data["stale"] is False
        assert imported_status.data["generation_upgrade_required"] is False
        imported_search = await imported_client.call_tool(
            "search",
            {"query": "cobalt heron amber marsh", "top_k": 1},
            timeout=1800,
        )
        assert imported_search.data["hits"][0]["chunk_id"] == reference_chunk_id
        assert (imported_project / "sources" / "evidence.pdf").is_file()

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

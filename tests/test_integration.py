from __future__ import annotations

import asyncio
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
        tools = {tool.name for tool in await client.list_tools()}
        assert tools == {
            "get_passage",
            "ingest",
            "list_sources",
            "search",
            "set_source_metadata",
            "status",
        }

        initial = await client.call_tool("status", {})
        assert initial.data["ready"] is False
        assert initial.data["selected_source_count"] == 1
        assert initial.data["ignored_extensions"] == {".md": 1}

        ingested = await client.call_tool(
            "ingest",
            {"chunk_size": 50, "chunk_overlap": 10},
            timeout=1800,
        )
        assert ingested.data["document_count"] == 1
        assert ingested.data["ignored_extensions"] == {".md": 1}

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


@pytest.mark.integration
def test_real_vanilla_ultrarag_research_flow(project: Path) -> None:
    asyncio.run(_assert_real_stdio_research_flow(project))

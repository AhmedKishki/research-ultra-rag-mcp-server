"""Search-level source selection and category partitions."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from conftest import write_pdf
from test_service import FakeDenseBackend, FakeUltraRAG

from research_ultra_rag_mcp.config import resolve_config
from research_ultra_rag_mcp.service import ResearchError, ResearchService


async def _build_project(project: Path) -> tuple[ResearchService, dict[str, str]]:
    write_pdf(
        project / "sources" / "cobalt.pdf",
        ["Cobalt extraction labour evidence from the mine."],
    )
    write_pdf(
        project / "sources" / "waste.pdf",
        ["Electronic waste labour evidence from the landfill."],
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)
    listed = await service.list_sources()
    source_ids = {
        item["source_relative_path"]: item["source_id"] for item in listed["sources"]
    }
    assert set(source_ids) == {"cobalt.pdf", "waste.pdf"}
    return service, source_ids


def test_search_selects_and_excludes_sources_by_stable_id(project: Path) -> None:
    async def exercise() -> None:
        service, source_ids = await _build_project(project)
        cobalt = source_ids["cobalt.pdf"]
        waste = source_ids["waste.pdf"]

        everything = await service.search("labour evidence", top_k=10, rerank=False)
        assert {hit["source_path"] for hit in everything["hits"]} == {
            "sources/cobalt.pdf",
            "sources/waste.pdf",
        }
        assert everything["filters"]["source_ids"] == []
        assert everything["filters"]["exclude_source_ids"] == []
        assert everything["filters"]["active_document_count"] == 2

        included = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            source_ids=[cobalt],
        )
        assert {hit["source_path"] for hit in included["hits"]} == {
            "sources/cobalt.pdf"
        }
        assert included["filters"]["source_ids"] == [cobalt]
        assert included["filters"]["unknown_source_ids"] == []
        assert included["filters"]["active_document_count"] == 1

        excluded = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            exclude_source_ids=[waste],
        )
        assert {hit["source_path"] for hit in excluded["hits"]} == {
            "sources/cobalt.pdf"
        }
        assert excluded["filters"]["exclude_source_ids"] == [waste]
        assert excluded["filters"]["active_document_count"] == 1

    asyncio.run(exercise())


def test_search_reports_unknown_source_ids_and_refuses_an_empty_include(
    project: Path,
) -> None:
    async def exercise() -> None:
        service, source_ids = await _build_project(project)
        cobalt = source_ids["cobalt.pdf"]

        with pytest.raises(ResearchError, match="matched no document"):
            await service.search(
                "labour evidence",
                top_k=5,
                rerank=False,
                source_ids=["src_missing"],
            )

        mixed = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            source_ids=[cobalt, "src_missing"],
        )
        assert {hit["source_path"] for hit in mixed["hits"]} == {"sources/cobalt.pdf"}
        assert mixed["filters"]["unknown_source_ids"] == ["src_missing"]

        unknown_exclusion = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            exclude_source_ids=["src_missing"],
        )
        assert unknown_exclusion["filters"]["unknown_exclude_source_ids"] == [
            "src_missing"
        ]
        assert unknown_exclusion["filters"]["unknown_source_ids"] == []
        assert unknown_exclusion["filters"]["active_document_count"] == 2

    asyncio.run(exercise())


def test_reviewed_exclusion_wins_over_a_search_include(project: Path) -> None:
    async def exercise() -> None:
        service, source_ids = await _build_project(project)
        waste = source_ids["waste.pdf"]
        await service.set_source_inclusion(
            included=False,
            source_id=waste,
            reason="Reviewed duplicate for this test.",
        )
        assert (await service.status())["excluded_source_count"] == 1

        result = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            source_ids=[waste],
        )
        assert result["hits"] == []
        assert result["result_count"] == 0
        assert result["filters"]["active_document_count"] == 0
        assert result["filters"]["source_ids"] == [waste]

    asyncio.run(exercise())


def test_categories_partition_the_corpus_for_search_and_listing(project: Path) -> None:
    async def exercise() -> None:
        service, source_ids = await _build_project(project)
        assert (await service.status())["categories"] == []

        await service.set_source_metadata(
            metadata={"categories": ["Cobalt corpus"]},
            source_id=source_ids["cobalt.pdf"],
        )
        await service.set_source_metadata(
            metadata={"categories": ["Waste corpus"]},
            source_id=source_ids["waste.pdf"],
        )

        assert (await service.status())["categories"] == [
            {"category": "Cobalt corpus", "searchable_source_count": 1},
            {"category": "Waste corpus", "searchable_source_count": 1},
        ]

        one_partition = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            categories=["cobalt corpus"],
        )
        assert {hit["source_path"] for hit in one_partition["hits"]} == {
            "sources/cobalt.pdf"
        }
        assert one_partition["filters"]["categories_all"] == ["cobalt corpus"]
        assert one_partition["filters"]["categories_any"] == []

        union = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            categories_any=["cobalt corpus", "waste corpus"],
        )
        assert {hit["source_path"] for hit in union["hits"]} == {
            "sources/cobalt.pdf",
            "sources/waste.pdf",
        }
        assert union["filters"]["categories_any"] == ["cobalt corpus", "waste corpus"]

        contradictory = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            categories=["cobalt corpus"],
            categories_any=["waste corpus"],
        )
        assert contradictory["hits"] == []

        listed = await service.list_sources(categories_any=["waste corpus"])
        assert [item["source_path"] for item in listed["sources"]] == [
            "sources/waste.pdf"
        ]

    asyncio.run(exercise())


def test_source_selection_combines_with_categories(project: Path) -> None:
    async def exercise() -> None:
        service, source_ids = await _build_project(project)
        await service.set_source_metadata(
            metadata={"categories": ["Shared corpus"]},
            source_id=source_ids["cobalt.pdf"],
        )
        await service.set_source_metadata(
            metadata={"categories": ["Shared corpus"]},
            source_id=source_ids["waste.pdf"],
        )

        selected = await service.search(
            "labour evidence",
            top_k=10,
            rerank=False,
            categories=["shared corpus"],
            source_ids=[source_ids["waste.pdf"]],
        )
        assert {hit["source_path"] for hit in selected["hits"]} == {"sources/waste.pdf"}
        assert selected["filters"]["active_document_count"] == 1
        assert selected["filters"]["categories_all"] == ["shared corpus"]
        assert selected["filters"]["source_ids"] == [source_ids["waste.pdf"]]

    asyncio.run(exercise())

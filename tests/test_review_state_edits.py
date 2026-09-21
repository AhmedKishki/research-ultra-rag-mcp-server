"""Human-edited review state: the plain JSON files stay authoritative."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from conftest import write_pdf
from test_service import FakeDenseBackend, FakeUltraRAG

from research_ultra_rag_mcp.config import resolve_config
from research_ultra_rag_mcp.service import ResearchError, ResearchService


def _write_metadata_file(project: Path, sources: dict[str, dict[str, object]]) -> Path:
    """Write the reviewed-metadata file the way a person would: by hand."""

    path = project / ".research-rag" / "source-metadata.json"
    path.write_text(
        json.dumps({"schema_version": 1, "sources": sources}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


async def _service(project: Path) -> tuple[ResearchService, str]:
    write_pdf(
        project / "sources" / "article.pdf",
        ["Cobalt evidence for the hand-edited review state."],
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)
    return service, "article.pdf"


def test_hand_written_metadata_file_is_honoured(project: Path) -> None:
    async def exercise() -> None:
        service, relative = await _service(project)
        _write_metadata_file(
            project,
            {
                relative: {
                    "title": "Hand-edited title",
                    "authors": ["Hand Editor", "Second Author"],
                    "year": 2025,
                    "doi": "10.1000/hand",
                    "categories": ["marxism", "waste studies"],
                    "keywords": ["fetishism", "use value"],
                    "project": ["ai-and-fetishism"],
                }
            },
        )

        listed = await service.list_sources()
        source = listed["sources"][0]
        assert source["title"] == "Hand-edited title"
        assert source["authors"] == ["Hand Editor", "Second Author"]
        assert source["year"] == 2025
        assert source["doi"] == "10.1000/hand"
        assert source["categories"] == ["marxism", "waste studies"]
        assert source["keywords"] == ["fetishism", "use value"]
        assert source["project"] == ["ai-and-fetishism"]

        status = await service.status()
        assert (status["categories"], status["projects"]) == (
            [
                {"category": "marxism", "searchable_source_count": 1},
                {"category": "waste studies", "searchable_source_count": 1},
            ],
            [{"project": "ai-and-fetishism", "searchable_source_count": 1}],
        )
        assert status["metadata_overlay_active"] is True

        filtered = await service.search(
            "cobalt evidence",
            top_k=5,
            rerank=False,
            projects=["ai-and-fetishism"],
            categories=["marxism"],
            keywords=["use value"],
        )
        assert filtered["filters"]["projects_all"] == ["ai-and-fetishism"]
        assert len(filtered["hits"]) == 1
        assert filtered["hits"][0]["title"] == "Hand-edited title"
        assert filtered["hits"][0]["project"] == ["ai-and-fetishism"]

    asyncio.run(exercise())


def test_hand_written_metadata_typo_fails_loudly(project: Path) -> None:
    async def exercise() -> None:
        service, relative = await _service(project)
        _write_metadata_file(project, {relative: {"titel": "Misspelled field"}})

        with pytest.raises(ResearchError, match="Unsupported metadata fields: titel"):
            await service.status()

    asyncio.run(exercise())


def test_hand_written_metadata_rejects_a_bad_value_or_path(project: Path) -> None:
    async def exercise() -> None:
        service, relative = await _service(project)
        _write_metadata_file(project, {relative: {"year": "2025"}})
        with pytest.raises(ResearchError, match="'year' must be null or an integer"):
            await service.status()

        _write_metadata_file(project, {"../escape.pdf": {"title": "Outside"}})
        with pytest.raises(ResearchError, match="normalized and relative"):
            await service.status()

        (project / ".research-rag" / "source-metadata.json").write_text(
            json.dumps({"schema_version": 2, "sources": {}}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ResearchError, match="Unsupported metadata file"):
            await service.status()

    asyncio.run(exercise())

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from conftest import write_pdf

from research_ultra_rag_mcp.config import resolve_config
from research_ultra_rag_mcp.service import (
    ResearchError,
    ResearchService,
    _enrich_chunks,
)
from research_ultra_rag_mcp.storage import read_jsonl, write_jsonl


class FakeUltraRAG:
    def __init__(self) -> None:
        self.passages: list[str] = []
        self.initialized: tuple[Path, Path] | None = None

    async def chunk(
        self,
        input_path: Path,
        output_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        assert chunk_size > chunk_overlap
        units = read_jsonl(input_path)
        write_jsonl(
            output_path,
            (
                {
                    "id": index,
                    "doc_id": item["id"],
                    "title": item["title"],
                    "contents": item["contents"],
                }
                for index, item in enumerate(units)
            ),
        )

    async def build_bm25(self, chunks_path: Path, index_path: Path) -> None:
        index_path.mkdir(parents=True)
        (index_path / "fake-index.json").write_text("{}\n", encoding="utf-8")
        self.passages = [item["contents"] for item in read_jsonl(chunks_path)]
        self.initialized = (chunks_path, index_path)

    async def initialize_bm25(self, chunks_path: Path, index_path: Path) -> None:
        self.passages = [item["contents"] for item in read_jsonl(chunks_path)]
        self.initialized = (chunks_path, index_path)

    async def search_bm25(self, query: str, top_k: int) -> list[str]:
        terms = query.casefold().split()

        def score(passage: str) -> int:
            value = passage.casefold()
            return sum(value.count(term) for term in terms)

        return sorted(self.passages, key=score, reverse=True)[:top_k]


def test_empty_upstream_chunks_are_counted_without_losing_a_unit() -> None:
    document = {
        "document_id": "doc_test",
        "source_path": "sources/test.pdf",
        "title": "Test",
        "authors": [],
        "year": None,
        "doi": "",
        "categories": [],
        "keywords": [],
    }
    unit = {
        "id": "doc_test:pdf-page:000001",
        "document_id": "doc_test",
        "locator": {"type": "pdf_page", "page": 1, "page_label": "1"},
    }
    chunks, discarded = _enrich_chunks(
        [
            {"doc_id": unit["id"], "contents": "Evidence remains searchable."},
            {"doc_id": unit["id"], "contents": ""},
        ],
        [unit],
        [document],
    )
    assert len(chunks) == 1
    assert discarded == 1

    second_unit = {
        "id": "doc_test:pdf-page:000002",
        "document_id": "doc_test",
        "locator": {"type": "pdf_page", "page": 2, "page_label": "2"},
    }
    with pytest.raises(ResearchError, match="extraction units"):
        _enrich_chunks(
            [
                {"doc_id": unit["id"], "contents": ""},
                {"doc_id": second_unit["id"], "contents": "Other page evidence."},
            ],
            [unit, second_unit],
            [document],
        )


async def _assert_research_generation_and_structured_search(project: Path) -> None:
    write_pdf(
        project / "sources" / "article.pdf",
        [
            "Cobalt evidence about labour and artificial intelligence.",
            "Quartz material unrelated to the primary question.",
        ],
        title="Research Article",
    )
    (project / "sources" / "notes.md").write_text(
        "cobalt cobalt should never be indexed",
        encoding="utf-8",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    fake = FakeUltraRAG()
    service = ResearchService(config, fake)  # type: ignore[arg-type]

    metadata_result = await service.set_source_metadata(
        "article.pdf",
        {
            "authors": ["Researcher One"],
            "year": 2026,
            "categories": ["political economy"],
            "keywords": ["labour", "AI"],
        },
    )
    assert metadata_result["requires_ingest"] is True

    result = await service.ingest(chunk_size=100, chunk_overlap=10)
    assert result["document_count"] == 1
    assert result["pdf_count"] == 1
    assert result["epub_count"] == 0
    assert result["ignored_extensions"] == {".md": 1}
    assert result["discarded_empty_chunk_count"] == 0

    status = await service.status()
    assert status["ready"] is True
    assert status["stale"] is False

    sources = await service.list_sources(categories=["political economy"])
    assert sources["source_count"] == 1
    assert sources["sources"][0]["source_path"] == "sources/article.pdf"

    search = await service.search(
        "cobalt labour",
        top_k=1,
        categories=["political economy"],
    )
    hit = search["hits"][0]
    assert hit["source_path"] == "sources/article.pdf"
    assert hit["locator"]["type"] == "pdf_page"
    assert hit["locator"]["page"] == 1
    assert "Cobalt evidence" in hit["text"]
    assert "p. 1" in hit["citation"]
    assert "notes" not in json.dumps(search)

    passage = await service.get_passage(hit["chunk_id"], context_chunks=1)
    assert passage["requested_chunk_id"] == hit["chunk_id"]
    assert len(passage["context"]) == 2

    write_pdf(project / "sources" / "new.pdf", ["A newly added source."])
    stale = await service.status()
    assert stale["stale"] is True
    assert stale["changes"]["added"] == ["new.pdf"]


def test_research_generation_and_structured_search(project: Path) -> None:
    asyncio.run(_assert_research_generation_and_structured_search(project))

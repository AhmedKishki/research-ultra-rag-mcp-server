from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from conftest import write_pdf

from research_ultra_rag_mcp.config import resolve_config
from research_ultra_rag_mcp.dense import DenseSearchHit
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


class FakeDenseBackend:
    def __init__(self) -> None:
        self.chunks: list[dict[str, object]] = []
        self.fail_build = False

    def build(
        self,
        chunks: list[dict[str, object]],
        index_path: Path,
    ) -> dict[str, object]:
        if self.fail_build:
            raise RuntimeError("simulated dense-index failure")
        index_path.mkdir(parents=True)
        (index_path / "fake-qdrant.json").write_text("{}\n", encoding="utf-8")
        self.chunks = chunks
        return {
            "backend": "fake Qdrant",
            "embedding_model": "fake-embedding-model",
            "embedding_dimension": 3,
            "point_count": len(chunks),
        }

    def search(
        self,
        index_path: Path,
        query: str,
        top_k: int,
        *,
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> list[DenseSearchHit]:
        assert index_path.is_dir()
        category_filter = {item.casefold() for item in categories or []}
        keyword_filter = {item.casefold() for item in keywords or []}
        document_filter = set(document_ids or [])
        terms = query.casefold().split()
        scored: list[tuple[float, str]] = []
        for chunk in self.chunks:
            chunk_categories = {
                str(item).casefold() for item in chunk.get("categories", [])
            }
            chunk_keywords = {
                str(item).casefold() for item in chunk.get("keywords", [])
            }
            if category_filter and not category_filter.issubset(chunk_categories):
                continue
            if keyword_filter and not keyword_filter.issubset(chunk_keywords):
                continue
            if document_filter and chunk["document_id"] not in document_filter:
                continue
            text = str(chunk["embedding_text"]).casefold()
            score = float(sum(text.count(term) for term in terms))
            scored.append((score, str(chunk["chunk_id"])))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            DenseSearchHit(chunk_id=chunk_id, score=score)
            for score, chunk_id in scored[:top_k]
        ]

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [
            float(document.casefold().count(query.split()[0].casefold()))
            for document in documents
        ]


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
    dense = FakeDenseBackend()
    service = ResearchService(config, fake, dense=dense)  # type: ignore[arg-type]

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
    assert result["default_retrieval_method"] == "hybrid"
    assert set(result["available_retrieval_methods"]) == {"bm25", "dense", "hybrid"}

    status = await service.status()
    assert status["ready"] is True
    assert status["stale"] is False
    assert status["hybrid_ready"] is True
    assert status["hybrid_upgrade_required"] is False

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
    assert search["retrieval_method"] == "hybrid"
    assert hit["component_ranks"]["bm25"] == 1
    assert hit["component_ranks"]["dense"] == 1
    assert hit["fusion_score"] is not None
    assert "notes" not in json.dumps(search)

    dense_search = await service.search(
        "cobalt labour",
        top_k=1,
        categories=["political economy"],
        retrieval_method="dense",
        rerank=True,
    )
    assert dense_search["retrieval_method"] == "dense"
    assert dense_search["reranked"] is True
    assert dense_search["hits"][0]["rerank_score"] is not None

    passage = await service.get_passage(hit["chunk_id"], context_chunks=1)
    assert passage["requested_chunk_id"] == hit["chunk_id"]
    assert len(passage["context"]) == 2

    generation_root = Path(result["generation_root"])
    manifest_path = generation_root / "manifest.json"
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_manifest["schema_version"] = 1
    legacy_manifest["retrieval"] = {
        "backend": "UltraRAG BM25",
        "language": "en",
        "tokenizer": "default",
    }
    legacy_manifest["files"].pop("dense_index")
    manifest_path.write_text(
        json.dumps(legacy_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    legacy_status = await service.status()
    assert legacy_status["hybrid_upgrade_required"] is True
    with pytest.raises(ResearchError, match="does not support 'hybrid'"):
        await service.search("cobalt", top_k=1)
    legacy_search = await service.search(
        "cobalt",
        top_k=1,
        retrieval_method="bm25",
    )
    assert legacy_search["hits"][0]["retrieval_method"] == "bm25"

    write_pdf(project / "sources" / "new.pdf", ["A newly added source."])
    stale = await service.status()
    assert stale["stale"] is True
    assert stale["changes"]["added"] == ["new.pdf"]


def test_research_generation_and_structured_search(project: Path) -> None:
    asyncio.run(_assert_research_generation_and_structured_search(project))


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    ordered, scores = ResearchService._fuse_rankings(
        ["bm25-only", "shared"],
        ["shared", "dense-only"],
    )
    assert ordered[0] == "shared"
    assert scores["shared"] > scores["bm25-only"]


async def _assert_failed_dense_build_does_not_replace_current(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    dense = FakeDenseBackend()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=dense,
    )
    first = await service.ingest(chunk_size=50, chunk_overlap=10)
    dense.fail_build = True

    with pytest.raises(RuntimeError, match="simulated dense-index failure"):
        await service.ingest(chunk_size=50, chunk_overlap=10)

    current = json.loads(config.current_path.read_text(encoding="utf-8"))
    assert current["generation_id"] == first["generation_id"]
    failed = [
        path
        for path in config.generations_root.iterdir()
        if path.name != first["generation_id"]
    ]
    assert len(failed) == 1
    failure = json.loads((failed[0] / "failure.json").read_text(encoding="utf-8"))
    assert failure["error"] == "simulated dense-index failure"


def test_failed_dense_build_does_not_replace_current(project: Path) -> None:
    asyncio.run(_assert_failed_dense_build_does_not_replace_current(project))

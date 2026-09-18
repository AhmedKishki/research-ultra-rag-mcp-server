from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import write_epub, write_pdf

import research_ultra_rag_mcp.service as service_module
from research_ultra_rag_mcp.config import resolve_config
from research_ultra_rag_mcp.dense import (
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
    DenseSearchHit,
)
from research_ultra_rag_mcp.extraction import ExtractionError
from research_ultra_rag_mcp.service import (
    ResearchError,
    ResearchService,
    _enrich_chunks,
    _public_document,
)
from research_ultra_rag_mcp.storage import read_jsonl, write_jsonl

CORRUPT_TEXT = (
    "��ѪҶޜഝǄ䘉Ӌਁ ⧠ᢃ⹤Ҷἅ ൠ؞༽൷㜭ᡀ࣏ "
    "䖜රѪտᆵǃ୶ъㅹ儈ԧ٬ъᘱⲴ⡷䶒ਉһǄ൘ᡰᴹᵳ╄ਈᯩ "
    "䶒ˈབྷཊᮠ൪ൠ൘䗷 ৫ഋॱᒤѝ࿻㓸؍ᤱ⿱ᴹᡆޜᴹ኎ᙗǄ "
    "ᵜ᮷䘈ᇎ䇱ਁ ⧠ˈ䜘࠶൪ൠᡰᴹᵳ⭡⊑ḃර⿱㩕Աъࡂ䖜㠣᭯ "
    "ᓌˈ⭡᭯ ᓌ᢯ᣵޘ䜘؞༽䍴䠁ˈᴰ㓸䙐ᡀ⊑ḃ⋫⨶ᡀᵜ⽮Պॆ "
    "ˈᒦᕅਁ ⧟ຳнޜǄᴹ∂ᓏ⢙൪ൠ 䳮�"
)


class FakeUltraRAG:
    def __init__(self) -> None:
        self.passages: list[str] = []
        self.initialized: tuple[Path, Path] | None = None
        self.chunk_calls = 0
        self.bm25_build_calls = 0

    async def chunk(
        self,
        input_path: Path,
        output_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        assert chunk_size > chunk_overlap
        self.chunk_calls += 1
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
        self.bm25_build_calls += 1
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
        self.build_calls = 0
        self.embed_calls = 0
        self.embedded_text_count = 0
        self.upload_batches: list[tuple[int, int]] = []

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.embed_calls += 1
        self.embedded_text_count += len(texts)
        return np.ones((len(texts), 384), dtype=np.float32)

    def build_index(
        self,
        chunks: list[dict[str, object]],
        index_path: Path,
        vectors: np.ndarray,
    ) -> dict[str, object]:
        self.build_calls += 1
        if self.fail_build:
            raise RuntimeError("simulated dense-index failure")
        assert vectors.shape == (len(chunks), 384)
        index_path.mkdir(parents=True)
        (index_path / "fake-qdrant.json").write_text("{}\n", encoding="utf-8")
        self.chunks = chunks
        return {
            "backend": "fake Qdrant",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_model_revision": EMBEDDING_MODEL_REVISION,
            "embedding_dimension": 384,
            "point_count": len(chunks),
        }

    def initialize_index(self, index_path: Path, dimension: int) -> None:
        self.build_calls += 1
        if self.fail_build:
            raise RuntimeError("simulated dense-index failure")
        assert dimension == 384
        index_path.mkdir(parents=True)
        self.chunks = []

    def upload_index_batch(
        self,
        chunks: list[dict[str, object]],
        index_path: Path,
        vectors: np.ndarray,
        *,
        offset: int,
    ) -> None:
        assert index_path.is_dir()
        assert vectors.shape == (len(chunks), 384)
        assert offset == len(self.chunks)
        self.upload_batches.append((offset, len(chunks)))
        self.chunks.extend(chunks)

    def finalize_index(
        self,
        index_path: Path,
        *,
        expected_count: int,
        dimension: int,
    ) -> dict[str, object]:
        assert index_path.is_dir()
        assert dimension == 384
        assert len(self.chunks) == expected_count
        (index_path / "fake-qdrant.json").write_text("{}\n", encoding="utf-8")
        return {
            "backend": "fake Qdrant",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_model_revision": EMBEDDING_MODEL_REVISION,
            "embedding_dimension": 384,
            "point_count": expected_count,
        }

    def build(
        self,
        chunks: list[dict[str, object]],
        index_path: Path,
        vectors_path: Path | None = None,
    ) -> dict[str, object]:
        vectors = self.embed_texts([str(chunk["embedding_text"]) for chunk in chunks])
        if vectors_path is not None:
            vectors_path.parent.mkdir(parents=True, exist_ok=True)
            with vectors_path.open("wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
        return self.build_index(chunks, index_path, vectors)

    def build_from_vectors(
        self,
        chunks: list[dict[str, object]],
        index_path: Path,
        vectors_path: Path,
    ) -> dict[str, object]:
        vectors = np.load(vectors_path, allow_pickle=False)
        assert vectors.shape == (len(chunks), 384)
        return self.build_index(chunks, index_path, vectors)

    def search(
        self,
        index_path: Path,
        query: str,
        top_k: int,
        *,
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        document_ids: list[str] | None = None,
        excluded_document_ids: list[str] | None = None,
    ) -> list[DenseSearchHit]:
        assert index_path.is_dir()
        category_filter = {item.casefold() for item in categories or []}
        keyword_filter = {item.casefold() for item in keywords or []}
        document_filter = set(document_ids or [])
        excluded_document_filter = set(excluded_document_ids or [])
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
            if chunk["document_id"] in excluded_document_filter:
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


class ScoredDenseBackend(FakeDenseBackend):
    def __init__(self, score: float) -> None:
        super().__init__()
        self.score = score

    def search(
        self,
        index_path: Path,
        query: str,
        top_k: int,
        **_filters: object,
    ) -> list[DenseSearchHit]:
        assert index_path.is_dir()
        return [
            DenseSearchHit(chunk_id=str(chunk["chunk_id"]), score=self.score)
            for chunk in self.chunks[:top_k]
        ]


class CancelOnceUltraRAG(FakeUltraRAG):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_next_chunk = True

    async def chunk(
        self,
        input_path: Path,
        output_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        if self.cancel_next_chunk:
            self.cancel_next_chunk = False
            raise asyncio.CancelledError
        await super().chunk(
            input_path,
            output_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


class ManyChunkUltraRAG(FakeUltraRAG):
    async def chunk(
        self,
        input_path: Path,
        output_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        assert chunk_size > chunk_overlap
        self.chunk_calls += 1
        unit = read_jsonl(input_path)[0]
        write_jsonl(
            output_path,
            (
                {
                    "id": index,
                    "doc_id": unit["id"],
                    "title": unit["title"],
                    "contents": f"Research evidence passage number {index}.",
                }
                for index in range(130)
            ),
        )


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


def test_legacy_automatic_metadata_uses_runtime_corruption_guard() -> None:
    legacy = {
        "source_path": "sources/safe-name.pdf",
        "title": CORRUPT_TEXT,
        "authors": [CORRUPT_TEXT, "Ada Example"],
        "categories": [],
        "keywords": [],
        "doi": "",
        "metadata_provenance": {
            "title": "pdf_metadata",
            "authors": "pdf_metadata",
        },
        "metadata_warnings": [],
    }

    public = _public_document(legacy)

    assert public["title"] == "safe-name"
    assert public["authors"] == ["Ada Example"]
    assert "corrupt_extracted_title" in public["metadata_warnings"]
    assert "corrupt_extracted_authors" in public["metadata_warnings"]

    legacy["metadata_provenance"] = {
        "title": "reviewed_override",
        "authors": "reviewed_override",
    }
    reviewed = _public_document(legacy)
    assert reviewed["title"] == CORRUPT_TEXT
    assert reviewed["authors"][0] == CORRUPT_TEXT


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
    assert status["generation_upgrade_required"] is False
    assert status["model_cache_root"] == str(config.model_cache_root)
    assert status["last_build_metrics"]["created_vector_count"] == result["chunk_count"]

    sources = await service.list_sources(categories=["political economy"])
    assert sources["source_count"] == 1
    assert sources["sources"][0]["source_path"] == "sources/article.pdf"

    generation_root = Path(result["generation_root"])
    manifest = json.loads(
        (generation_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert "raw_extraction" not in manifest["files"]
    assert "raw_ultrarag_chunks" not in manifest["files"]
    assert not (generation_root / "work").exists()
    chunks_path = generation_root / "chunks" / "chunks.jsonl"
    stored_chunks = read_jsonl(chunks_path)
    stored_chunks[0]["text"] = (
        "Cobalt evidence about\nlabour and artificial\nintelligence."
    )
    write_jsonl(chunks_path, stored_chunks)

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
    assert hit["text"] == ("Cobalt evidence about labour and artificial intelligence.")
    assert "p. 1" in hit["citation"]
    assert search["retrieval_method"] == "hybrid"
    assert hit["component_ranks"]["bm25"] == 1
    assert hit["component_ranks"]["dense"] == 1
    assert hit["fusion_score"] is not None
    assert hit["direct_quote_safe"] is False
    assert hit["text_fidelity"] == "cleaned_semantic_text"
    assert hit["content_kind"] == "prose"
    assert hit["metadata_provenance"]["authors"] == "reviewed_override"
    assert search["requested_top_k"] == 1
    assert search["relevance_limited"] is False
    assert search["relevance_policy"]["dense_minimum_cosine_similarity"] == 0.72
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
    assert passage["context"][0]["text"] == hit["text"]

    manifest_path = generation_root / "manifest.json"
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_manifest["retrieval"]["relevance_gates"][
        "dense_minimum_cosine_similarity"
    ] = 0.71
    manifest_path.write_text(
        json.dumps(legacy_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    policy_status = await service.status()
    assert policy_status["generation_upgrade_required"] is True
    assert "retrieval_policy" in policy_status["upgrade_reasons"]

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
    assert legacy_status["generation_upgrade_required"] is True
    assert "generation_schema" in legacy_status["upgrade_reasons"]
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


async def _assert_agent_reviewed_source_exclusion(project: Path) -> None:
    preferred = project / "sources" / "preferred.pdf"
    duplicate = project / "sources" / "duplicate.pdf"
    write_pdf(preferred, ["Shared cobalt evidence for duplicate review."])
    write_pdf(duplicate, ["Shared cobalt evidence for duplicate review."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )

    first = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert first["source_file_count"] == 2
    assert first["document_count"] == 2

    initial = await service.search("shared cobalt evidence", top_k=10)
    duplicate_hit = next(
        hit for hit in initial["hits"] if hit["source_path"] == "sources/duplicate.pdf"
    )

    with pytest.raises(ResearchError, match="reason is required"):
        await service.set_source_inclusion("duplicate.pdf", included=False)
    with pytest.raises(ResearchError, match="escapes"):
        await service.set_source_inclusion(
            "../outside.pdf",
            included=False,
            reason="Invalid path must not be accepted.",
        )

    excluded = await service.set_source_inclusion(
        "duplicate.pdf",
        included=False,
        reason="Agent-reviewed duplicate; preferred.pdf is the retained copy.",
    )
    assert excluded["effective_immediately"] is True
    assert excluded["source_file_changed"] is False
    assert duplicate.is_file()

    status = await service.status()
    assert status["stale"] is True
    assert status["excluded_source_count"] == 1
    assert status["searchable_source_count"] == 1
    assert status["changes"]["source_exclusions_changed"] is True
    assert status["excluded_sources"][0]["indexed_in_current_generation"] is True

    search = await service.search("shared cobalt evidence", top_k=10)
    assert search["excluded_source_count"] == 1
    assert {hit["source_path"] for hit in search["hits"]} == {"sources/preferred.pdf"}
    sources = await service.list_sources()
    assert sources["source_count"] == 1
    assert sources["sources"][0]["source_path"] == "sources/preferred.pdf"
    with pytest.raises(ResearchError, match="currently excluded"):
        await service.get_passage(duplicate_hit["chunk_id"])

    rebuilt = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert rebuilt["source_file_count"] == 2
    assert rebuilt["excluded_source_count"] == 1
    assert rebuilt["document_count"] == 1
    assert rebuilt["reused_document_count"] == 1
    assert rebuilt["rebuilt_document_count"] == 0
    rebuilt_status = await service.status()
    assert rebuilt_status["stale"] is False
    assert (
        rebuilt_status["excluded_sources"][0]["indexed_in_current_generation"] is False
    )

    included = await service.set_source_inclusion(
        "duplicate.pdf",
        included=True,
    )
    assert included["effective_immediately"] is False
    assert included["generation_rebuild_recommended"] is True
    assert duplicate.is_file()
    assert (await service.status())["stale"] is True

    restored = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert restored["document_count"] == 2
    assert restored["excluded_source_count"] == 0
    assert restored["reused_document_count"] == 1
    assert restored["rebuilt_document_count"] == 1


def test_agent_reviewed_source_exclusion_is_immediate_and_reversible(
    project: Path,
) -> None:
    asyncio.run(_assert_agent_reviewed_source_exclusion(project))


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    ordered, scores = ResearchService._fuse_rankings(
        ["bm25-only", "shared"],
        ["shared", "dense-only"],
    )
    assert ordered[0] == "shared"
    assert scores["shared"] > scores["bm25-only"]
    assert scores["bm25-only"] > scores["dense-only"]


async def _assert_relevance_gates_allow_abstention(project: Path) -> None:
    write_pdf(
        project / "sources" / "brands.pdf",
        ["Lenovo ZTE Sony (Motorola) Microsoft Samsung Vodafone"],
        title="Brand Table",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)

    unrelated = await service.search("hello", top_k=8)
    assert unrelated["result_count"] == 0
    assert unrelated["relevance_limited"] is True
    assert unrelated["rejected_candidates"] == {
        "bm25_no_query_token_overlap": 1,
        "bm25_extraction_artifact": 0,
        "bm25_corrupt_text": 0,
        "dense_below_threshold": 1,
        "dense_extraction_artifact": 0,
        "dense_corrupt_text": 0,
    }

    relevant = await service.search("Lenovo", top_k=8)
    assert relevant["result_count"] == 1
    assert relevant["hits"][0]["match_kind"] == "hybrid"


def test_relevance_gates_allow_zero_results(project: Path) -> None:
    asyncio.run(_assert_relevance_gates_allow_abstention(project))


async def _assert_dense_threshold_boundary(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Semantic research material."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    dense = ScoredDenseBackend(0.7199)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=dense,
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)

    rejected = await service.search("unrelated", retrieval_method="dense", top_k=1)
    assert rejected["result_count"] == 0
    assert rejected["rejected_candidates"]["dense_below_threshold"] == 1

    dense.score = 0.72
    accepted = await service.search("unrelated", retrieval_method="dense", top_k=1)
    assert accepted["result_count"] == 1
    assert accepted["hits"][0]["match_kind"] == "semantic"


def test_dense_threshold_boundary(project: Path) -> None:
    asyncio.run(_assert_dense_threshold_boundary(project))


async def _assert_no_change_ingest_is_noop_and_force_rebuilds(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = FakeUltraRAG()
    dense = FakeDenseBackend()
    service = ResearchService(config, ultrarag, dense=dense)  # type: ignore[arg-type]

    first = await service.ingest(chunk_size=50, chunk_overlap=10)
    second = await service.ingest(chunk_size=50, chunk_overlap=10)
    forced = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        force_recompute=True,
    )

    assert first["generation_id"] == second["generation_id"]
    assert second["generation_changed"] is False
    assert second["rebuilt_document_count"] == 0
    assert second["created_vector_count"] == 0
    assert forced["generation_id"] != first["generation_id"]
    assert forced["generation_changed"] is True
    assert forced["reused_document_count"] == 0
    assert forced["created_vector_count"] == forced["chunk_count"]
    assert Path(first["generation_root"]).is_dir()
    assert Path(forced["generation_root"]).is_dir()
    assert ultrarag.chunk_calls == 2
    assert ultrarag.bm25_build_calls == 2
    assert dense.build_calls == 2
    current = json.loads(config.current_path.read_text(encoding="utf-8"))
    assert current["generation_id"] == forced["generation_id"]


def test_no_change_ingest_is_noop_and_force_rebuilds(project: Path) -> None:
    asyncio.run(_assert_no_change_ingest_is_noop_and_force_rebuilds(project))


async def _assert_selective_reuse_tracks_every_input_change(project: Path) -> None:
    first_path = project / "sources" / "first.pdf"
    second_path = project / "sources" / "second.pdf"
    write_pdf(first_path, ["Stable cobalt evidence."], title="First")
    config = resolve_config(project, vanilla_executable=sys.executable)
    dense = FakeDenseBackend()
    ultrarag = FakeUltraRAG()
    service = ResearchService(config, ultrarag, dense=dense)  # type: ignore[arg-type]

    first = await service.ingest(chunk_size=50, chunk_overlap=10)
    write_pdf(second_path, ["Amber evidence in a second source."], title="Second")
    added = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert added["generation_changed"] is True
    assert added["reused_document_count"] == 1
    assert added["rebuilt_document_count"] == 1
    assert added["reused_chunk_count"] == 1
    assert added["created_vector_count"] == 1

    previous_stat = second_path.stat()
    second_path.unlink()
    write_pdf(second_path, ["Cedar evidence in a second source."], title="Second")
    assert second_path.stat().st_size == previous_stat.st_size
    os.utime(
        second_path,
        ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns),
    )
    changed_bytes = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert changed_bytes["generation_changed"] is True
    assert changed_bytes["reused_document_count"] == 1
    assert changed_bytes["rebuilt_document_count"] == 1

    await service.set_source_metadata("first.pdf", {"categories": ["theory"]})
    metadata_changed = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert metadata_changed["rebuilt_document_count"] == 1
    assert metadata_changed["reused_document_count"] == 1
    assert metadata_changed["reused_vector_count"] == metadata_changed["chunk_count"]
    assert metadata_changed["created_vector_count"] == 0

    reuse_search = await service.search("evidence", top_k=8)
    forced = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        force_recompute=True,
    )
    full_search = await service.search("evidence", top_k=8)
    assert forced["created_vector_count"] == forced["chunk_count"]
    assert [item["chunk_id"] for item in reuse_search["hits"]] == [
        item["chunk_id"] for item in full_search["hits"]
    ]
    assert [item["rank"] for item in reuse_search["hits"]] == [
        item["rank"] for item in full_search["hits"]
    ]

    second_path.unlink()
    removed = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert removed["generation_changed"] is True
    assert removed["document_count"] == 1
    assert removed["reused_document_count"] == 1
    assert removed["rebuilt_document_count"] == 0
    assert removed["generation_id"] != first["generation_id"]


def test_selective_reuse_tracks_every_input_change(project: Path) -> None:
    asyncio.run(_assert_selective_reuse_tracks_every_input_change(project))


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
        await service.ingest(
            chunk_size=50,
            chunk_overlap=10,
            force_recompute=True,
        )

    current = json.loads(config.current_path.read_text(encoding="utf-8"))
    assert current["generation_id"] == first["generation_id"]
    assert [path.name for path in config.generations_root.iterdir()] == [
        first["generation_id"]
    ]
    assert not any(config.staging_root.iterdir())
    failures = list(config.failures_root.glob("*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["error"] == "simulated dense-index failure"


def test_failed_dense_build_does_not_replace_current(project: Path) -> None:
    asyncio.run(_assert_failed_dense_build_does_not_replace_current(project))


async def _assert_bounded_ingestion_resumes_after_service_restart(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = FakeUltraRAG()
    dense = FakeDenseBackend()
    monkeypatch.setattr(service_module, "MINIMUM_WORK_BUDGET_SECONDS", 0)

    first_service = ResearchService(  # type: ignore[arg-type]
        config,
        ultrarag,
        dense=dense,
    )
    result = await first_service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )
    assert result["status"] == "in_progress"
    assert result["phase"] == "source_hashing"
    assert not config.current_path.exists()

    status = await first_service.status()
    assert status["ingestion_progress"]["build_id"] == result["build_id"]

    resumed_service = ResearchService(  # type: ignore[arg-type]
        config,
        ultrarag,
        dense=dense,
    )
    for _ in range(20):
        result = await resumed_service.ingest(
            chunk_size=50,
            chunk_overlap=10,
            work_budget_seconds=0,
        )
        if result["status"] != "in_progress":
            break
        assert not config.current_path.exists()
    else:  # pragma: no cover - protects the bounded state machine from stalling.
        raise AssertionError("bounded ingestion did not finish")

    assert result["status"] == "ready"
    assert result["resumed"] is True
    assert config.current_path.exists()
    assert not any(config.staging_root.iterdir())
    assert (await resumed_service.status())["ingestion_progress"] is None


def test_bounded_ingestion_resumes_after_service_restart(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _assert_bounded_ingestion_resumes_after_service_restart(project, monkeypatch)
    )


async def _assert_selected_generation_remains_searchable_during_build(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable cobalt evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = FakeUltraRAG()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        ultrarag,
        dense=FakeDenseBackend(),
    )
    selected = await service.ingest(chunk_size=50, chunk_overlap=10)
    write_pdf(project / "sources" / "new.pdf", ["New amber evidence."])
    monkeypatch.setattr(service_module, "MINIMUM_WORK_BUDGET_SECONDS", 0)

    pending = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )
    search = await service.search("stable cobalt", top_k=1)
    status = await service.status()

    assert pending["status"] == "in_progress"
    assert search["generation_id"] == selected["generation_id"]
    assert search["hits"][0]["chunk_id"]
    assert status["generation_id"] == selected["generation_id"]
    assert status["ingestion_progress"]["build_id"] == pending["build_id"]


def test_selected_generation_remains_searchable_during_build(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _assert_selected_generation_remains_searchable_during_build(
            project,
            monkeypatch,
        )
    )


async def _assert_cancelled_ingestion_preserves_its_checkpoint(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = CancelOnceUltraRAG()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        ultrarag,
        dense=FakeDenseBackend(),
    )

    with pytest.raises(asyncio.CancelledError):
        await service.ingest(chunk_size=50, chunk_overlap=10)

    staging_roots = list(config.staging_root.iterdir())
    assert len(staging_roots) == 1
    checkpoint = json.loads(
        (staging_roots[0] / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["phase"] == "chunking"
    assert checkpoint["last_interruption"] == "cancelled"
    assert not config.current_path.exists()

    resumed = await service.ingest(chunk_size=50, chunk_overlap=10)
    assert resumed["status"] == "ready"
    assert resumed["resumed"] is True


def test_cancelled_ingestion_preserves_its_checkpoint(project: Path) -> None:
    asyncio.run(_assert_cancelled_ingestion_preserves_its_checkpoint(project))


async def _assert_incompatible_force_mode_supersedes_checkpoint(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    monkeypatch.setattr(service_module, "MINIMUM_WORK_BUDGET_SECONDS", 0)

    normal = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )
    forced = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        force_recompute=True,
        work_budget_seconds=0,
    )
    forced_resume = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        force_recompute=True,
        work_budget_seconds=0,
    )

    assert normal["build_id"] != forced["build_id"]
    assert forced_resume["build_id"] == forced["build_id"]
    failure = json.loads(
        (config.failures_root / f"{normal['build_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert "checkpoint superseded" in failure["error"]


def test_incompatible_force_mode_supersedes_checkpoint(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _assert_incompatible_force_mode_supersedes_checkpoint(project, monkeypatch)
    )


async def _assert_final_revalidation_catches_same_stat_mutation(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = project / "sources" / "article.pdf"
    write_pdf(source, ["Stable research evidence."])
    original_stat = source.stat()
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    monkeypatch.setattr(service_module, "MINIMUM_WORK_BUDGET_SECONDS", 0)

    for _ in range(30):
        result = await service.ingest(
            chunk_size=50,
            chunk_overlap=10,
            work_budget_seconds=0,
        )
        if (
            result["phase"] == "qdrant_indexing"
            and result["progress"]["completed"] == result["progress"]["total"]
        ):
            break
    else:  # pragma: no cover
        raise AssertionError("ingestion did not reach final source revalidation")

    original_bytes = source.read_bytes()
    mutated = bytearray(original_bytes)
    mutated[-1] = 32 if mutated[-1] != 32 else 10
    source.write_bytes(mutated)
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert source.stat().st_size == original_stat.st_size
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns

    revalidated = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )
    assert revalidated["build_id"] == result["build_id"]
    assert revalidated["phase"] == "source_revalidation"
    restarted = await service.ingest(
        chunk_size=50,
        chunk_overlap=10,
        work_budget_seconds=0,
    )
    assert restarted["status"] == "in_progress"
    assert restarted["build_id"] != result["build_id"]
    assert "Source bytes changed" in restarted["message"]
    assert not config.current_path.exists()


def test_final_revalidation_catches_same_stat_mutation(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _assert_final_revalidation_catches_same_stat_mutation(project, monkeypatch)
    )


async def _assert_bounded_and_single_call_builds_are_equivalent(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    single_project = project / "single"
    bounded_project = project / "bounded"
    (single_project / "sources").mkdir(parents=True)
    (bounded_project / "sources").mkdir(parents=True)
    source = single_project / "sources" / "evidence.pdf"
    write_pdf(
        source,
        ["First page cobalt evidence.", "Second page amber evidence."],
    )
    (bounded_project / "sources" / "evidence.pdf").write_bytes(source.read_bytes())

    single_config = resolve_config(single_project, vanilla_executable=sys.executable)
    bounded_config = resolve_config(bounded_project, vanilla_executable=sys.executable)
    single = ResearchService(  # type: ignore[arg-type]
        single_config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    bounded = ResearchService(  # type: ignore[arg-type]
        bounded_config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )
    single_result = await single.ingest(chunk_size=50, chunk_overlap=10)
    monkeypatch.setattr(service_module, "MINIMUM_WORK_BUDGET_SECONDS", 0)
    for _ in range(40):
        bounded_result = await bounded.ingest(
            chunk_size=50,
            chunk_overlap=10,
            work_budget_seconds=0,
        )
        if bounded_result["status"] == "ready":
            break
    else:  # pragma: no cover
        raise AssertionError("bounded equivalent build did not finish")

    single_root = Path(single_result["generation_root"])
    bounded_root = Path(bounded_result["generation_root"])
    assert read_jsonl(single_root / "corpus" / "extracted-units.jsonl") == read_jsonl(
        bounded_root / "corpus" / "extracted-units.jsonl"
    )
    assert read_jsonl(single_root / "chunks" / "chunks.jsonl") == read_jsonl(
        bounded_root / "chunks" / "chunks.jsonl"
    )
    assert np.array_equal(
        np.load(single_root / "portable" / "embeddings.npy", allow_pickle=False),
        np.load(bounded_root / "portable" / "embeddings.npy", allow_pickle=False),
    )


def test_bounded_and_single_call_builds_are_equivalent(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(
        _assert_bounded_and_single_call_builds_are_equivalent(project, monkeypatch)
    )


async def _assert_resumed_dense_batches_are_exactly_once(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = ManyChunkUltraRAG()
    dense = FakeDenseBackend()
    monkeypatch.setattr(service_module, "MINIMUM_WORK_BUDGET_SECONDS", 0)

    for _ in range(60):
        service = ResearchService(  # type: ignore[arg-type]
            config,
            ultrarag,
            dense=dense,
        )
        result = await service.ingest(
            chunk_size=50,
            chunk_overlap=10,
            work_budget_seconds=0,
        )
        if result["status"] == "ready":
            break
    else:  # pragma: no cover
        raise AssertionError("multi-batch ingestion did not finish")

    assert result["chunk_count"] == 130
    assert dense.embed_calls == 3
    assert dense.embedded_text_count == 130
    assert dense.upload_batches == [(0, 64), (64, 64), (128, 2)]
    assert len(dense.chunks) == 130
    assert len({str(chunk["chunk_id"]) for chunk in dense.chunks}) == 130


def test_resumed_dense_batches_are_exactly_once(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_assert_resumed_dense_batches_are_exactly_once(project, monkeypatch))


async def _assert_legacy_corrupt_chunks_are_rejected_immediately(
    project: Path,
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable research evidence."])
    config = resolve_config(project, vanilla_executable=sys.executable)
    ultrarag = FakeUltraRAG()
    dense = FakeDenseBackend()
    service = ResearchService(  # type: ignore[arg-type]
        config,
        ultrarag,
        dense=dense,
    )
    result = await service.ingest(chunk_size=50, chunk_overlap=10)
    generation_root = Path(result["generation_root"])
    chunks_path = generation_root / "chunks" / "chunks.jsonl"
    chunks = read_jsonl(chunks_path)
    chunk_id = str(chunks[0]["chunk_id"])
    clean_chunk = dict(chunks[0])
    clean_chunk["chunk_id"] = "clean-neighbor"
    clean_chunk["id"] = "clean-neighbor"
    clean_chunk["document_chunk_index"] = 0
    corrupt_chunk = dict(chunks[0])
    corrupt_chunk["document_chunk_index"] = 1
    for field in ("contents", "embedding_text", "text"):
        corrupt_chunk[field] = CORRUPT_TEXT
    chunks = [clean_chunk, corrupt_chunk]
    write_jsonl(chunks_path, chunks)
    ultrarag.passages = [CORRUPT_TEXT, str(clean_chunk["contents"])]
    dense.chunks = [corrupt_chunk]

    for retrieval_method in ("bm25", "dense", "hybrid"):
        search = await service.search(
            "evidence",
            top_k=8,
            retrieval_method=retrieval_method,
            rerank=True,
        )
        assert all(hit["chunk_id"] != chunk_id for hit in search["hits"])
        if retrieval_method in {"bm25", "hybrid"}:
            assert search["rejected_candidates"]["bm25_corrupt_text"] == 1
        if retrieval_method in {"dense", "hybrid"}:
            assert search["rejected_candidates"]["dense_corrupt_text"] == 1
    with pytest.raises(ResearchError, match="corrupt extracted text"):
        await service.get_passage(chunk_id)
    passage = await service.get_passage("clean-neighbor", context_chunks=1)
    assert [item["chunk_id"] for item in passage["context"]] == ["clean-neighbor"]


def test_legacy_corrupt_chunks_are_rejected_immediately(project: Path) -> None:
    asyncio.run(_assert_legacy_corrupt_chunks_are_rejected_immediately(project))


async def _assert_unreadable_source_fails_without_activation(project: Path) -> None:
    write_epub(project / "sources" / "broken.epub", CORRUPT_TEXT)
    config = resolve_config(project, vanilla_executable=sys.executable)
    service = ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )

    with pytest.raises(ExtractionError, match="no readable English-oriented text"):
        await service.ingest(chunk_size=50, chunk_overlap=10)

    assert not config.current_path.exists()
    assert not any(config.staging_root.iterdir())
    failures = list(config.failures_root.glob("*.json"))
    assert len(failures) == 1
    assert CORRUPT_TEXT not in failures[0].read_text(encoding="utf-8")


def test_unreadable_source_fails_without_activation(project: Path) -> None:
    asyncio.run(_assert_unreadable_source_fails_without_activation(project))

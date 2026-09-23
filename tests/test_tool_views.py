"""The agent-facing lean tool responses, and the full-detail debug path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from research_ultra_rag_mcp.config import (
    FULL_TOOL_DETAIL,
    LEAN_TOOL_DETAIL,
    ConfigurationError,
    resolve_config,
)
from research_ultra_rag_mcp.server import _parser
from research_ultra_rag_mcp.service import ResearchError
from research_ultra_rag_mcp.tool_views import present_tool_response

# Keys the service builds for ranking, extraction, and storage diagnostics. A
# lean answer must not carry one.
DIAGNOSTIC_KEYS = {
    "annotations",
    "available_retrieval_methods",
    "candidate_count",
    "candidate_depth",
    "candidate_distinct_reference_count",
    "component_ranks",
    "component_scores",
    "content_kind",
    "dense_fidelity",
    "dense_truncated",
    "distinct_reference_count",
    "doi",
    "document_id",
    "embedding_model",
    "embedding_model_revision",
    "embedding_token_count",
    "filters",
    "fusion",
    "fusion_score",
    "generation_root",
    "grouping",
    "grouping_limited",
    "grouping_skipped_candidate_count",
    "ignored_extensions",
    "last_build_metrics",
    "match_kind",
    "metadata_pending_source_paths",
    "metadata_provenance",
    "metadata_warnings",
    "passages_per_reference",
    "quality_flags",
    "rank",
    "rejected_candidates",
    "relevance_limited",
    "relevance_policy",
    "rerank_requested",
    "rerank_score",
    "reranker_model",
    "retrieval",
    "retrieval_method",
    "retrieval_rank",
    "source_path",
    "text_fidelity",
    "withheld_candidates",
    "year",
}


def test_lean_passage_shape() -> None:
    lean = present_tool_response("search", _search_payload(), detail=LEAN_TOOL_DETAIL)

    assert set(lean) == {"query", "generation_id", "stale", "reranked", "hits"}
    assert lean["hits"] == [
        {
            "chunk_id": "chk_one",
            "source_relative_path": "evidence.pdf",
            "authors": ["A. Researcher"],
            "locator": {"page": 3},
            "text": "cleaned semantic text",
        },
        {
            "chunk_id": "chk_two",
            "source_relative_path": "anonymous.pdf",
            "locator": {"section": "chapter.xhtml"},
            "text": "second passage",
        },
    ]

    # A passage is its source, its authors, where it sits, and its text. The
    # reference is deliberately not citation-ready, an identifier the filename
    # already gives is not repeated, a page label that repeats the physical page
    # is not a second way of saying the same thing, and neither the quote-safety
    # rule nor the advisory script note is repeated on every passage.
    for key in (
        "citation",
        "direct_quote_safe",
        "source_id",
        "text_notes",
        "title",
    ):
        assert key not in json.dumps(lean)


def test_lean_response_never_contains_a_diagnostic_key() -> None:
    for operation, payload in _every_tool_payload().items():
        lean = present_tool_response(operation, payload, detail=LEAN_TOOL_DETAIL)
        leaked = _keys(lean) & DIAGNOSTIC_KEYS
        assert not leaked, f"{operation} leaked {sorted(leaked)}"
        assert len(json.dumps(lean, ensure_ascii=False)) < len(
            json.dumps(payload, ensure_ascii=False)
        )


def test_full_detail_returns_the_service_payload_unchanged() -> None:
    payload = _search_payload()

    assert present_tool_response("search", payload, detail=FULL_TOOL_DETAIL) == payload


def test_search_reports_rerank_state_and_unknown_ids() -> None:
    payload = _search_payload(
        reranked=False,
        rerank_fallback={"reason": "reranker_model_unavailable", "effect": "unranked"},
        unknown_source_ids=["src_missing"],
    )
    lean = present_tool_response("search", payload, detail=LEAN_TOOL_DETAIL)

    assert lean["reranked"] is False
    assert lean["rerank_fallback"]["reason"] == "reranker_model_unavailable"
    assert lean["unresolved_source_ids"] == ["src_missing"]
    assert "unresolved_exclude_source_ids" not in lean


def test_search_omits_an_upgrade_note_that_is_not_required() -> None:
    lean = present_tool_response("search", _search_payload(), detail=LEAN_TOOL_DETAIL)

    assert "generation_upgrade_required" not in lean
    assert "reference_groups" not in lean

    required = present_tool_response(
        "search",
        _search_payload(generation_upgrade_required=True),
        detail=LEAN_TOOL_DETAIL,
    )
    assert required["generation_upgrade_required"] is True


def test_passage_context_is_lean_and_keeps_no_rank() -> None:
    payload = {
        "generation_id": "20260101T000000Z-abcdef",
        "requested_chunk_id": "chk_one",
        "context": [{key: value for key, value in _hit().items() if key != "rank"}],
        "notice": "Context is cleaned semantic text and is not quote-safe.",
    }
    lean = present_tool_response("get_passage", payload, detail=LEAN_TOOL_DETAIL)

    assert set(lean) == {"generation_id", "requested_chunk_id", "context"}
    assert "rank" not in lean["context"][0]
    assert "direct_quote_safe" not in lean["context"][0]
    assert "notice" not in lean


def test_status_lean_keeps_the_current_generation_and_no_inventory() -> None:
    payload = _status_payload()
    lean = present_tool_response("status", payload, detail=LEAN_TOOL_DETAIL)

    assert set(lean) == {
        "ready",
        "stale",
        "project_name",
        "generation_id",
        "created_at",
        "chunk_count",
        "discovered_source_count",
        "selected_source_count",
        "indexed_source_count",
        "searchable_source_count",
        "excluded_source_count",
        "retained_generation_count",
        "retained_generation_bytes",
        "message",
    }
    assert lean["generation_id"] == "20260101T000000Z-abcdef"
    assert lean["created_at"] == "2026-01-01T00:00:00Z"
    assert lean["chunk_count"] == 12
    assert lean["retained_generation_count"] == 1
    assert lean["retained_generation_bytes"] == 4096
    # A generation this tool can serve says nothing about methods.
    assert "hybrid_ready" not in lean

    # A generation that predates dense support is stated plainly, with the
    # upgrade advice, rather than as a list of methods to pick from.
    legacy = present_tool_response(
        "status",
        _status_payload(
            available_retrieval_methods=["bm25"],
            hybrid_ready=False,
            hybrid_upgrade_required=True,
            generation_upgrade_required=True,
            upgrade_reasons=["bm25_only_generation"],
        ),
        detail=LEAN_TOOL_DETAIL,
    )
    assert legacy["hybrid_ready"] is False
    assert legacy["generation_upgrade_required"] is True
    assert legacy["upgrade_reasons"] == ["bm25_only_generation"]
    assert "available_retrieval_methods" not in legacy

    # The retained generations, the categories, and the projects are inventories:
    # they answer a question of their own and belong to the full-detail payload,
    # so a status answer stays a statement about the selected generation.
    serialized = json.dumps(lean)
    for key in ("generations", "categories", "projects"):
        assert key not in serialized
        assert key in present_tool_response("status", payload, detail=FULL_TOOL_DETAIL)
    # A field whose value is the harmless default is left out entirely.
    assert "excluded_sources" not in lean
    assert "generation_upgrade_required" not in lean
    assert "upgrade_reasons" not in lean
    assert "metadata_overlay_active" not in lean
    assert "metadata_pending_source_paths" not in lean
    assert "metadata_pending_source_count" not in lean
    assert "ingestion_progress" not in lean
    assert "generation_root" not in lean
    assert "version" not in lean
    assert "restart_required" not in lean

    # A required restart is actionable state, so it survives the projection.
    restarting = present_tool_response(
        "status",
        _status_payload(
            version={
                "server": "0.15.0",
                "installed": "0.16.0",
                "ui": None,
                "restart_required": True,
            }
        ),
        detail=LEAN_TOOL_DETAIL,
    )
    assert restarting["restart_required"] is True
    assert "version" not in restarting


def test_status_counts_available_source_changes_and_names_missing_ones() -> None:
    current = present_tool_response(
        "status", _status_payload(), detail=LEAN_TOOL_DETAIL
    )

    assert "changes" not in current

    stale = present_tool_response(
        "status",
        _status_payload(
            stale=True,
            changes={
                "added": ["new.pdf", "another.pdf", "third.pdf"],
                "removed": ["gone.pdf"],
                "modified": ["edited.pdf"],
                "metadata_changed": True,
                "source_exclusions_changed": False,
            },
        ),
        detail=LEAN_TOOL_DETAIL,
    )
    assert stale["changes"] == {
        "added_source_count": 3,
        "modified_source_count": 1,
        "removed_sources": ["gone.pdf"],
        "metadata_changed": True,
    }
    # No available source is ever named in a status answer.
    assert "new.pdf" not in json.dumps(stale)
    assert "edited.pdf" not in json.dumps(stale)


def test_status_before_the_first_ingestion_states_what_is_missing() -> None:
    payload = {
        "ready": False,
        "stale": True,
        "project_name": "example",
        "discovered_source_count": 3,
        "selected_source_count": 3,
        "excluded_source_count": 0,
        "categories": [],
        "projects": [],
        "upgrade_reasons": [],
        "metadata_overlay_active": False,
        "metadata_pending_source_paths": [],
        "ingestion_progress": None,
        "message": "No knowledge-base generation exists; call ingest.",
    }
    lean = present_tool_response("status", payload, detail=LEAN_TOOL_DETAIL)

    assert lean["ready"] is False
    assert lean["stale"] is True
    assert lean["discovered_source_count"] == 3
    assert "generations" not in lean
    assert "categories" not in lean
    assert "projects" not in lean
    assert "retained_generation_count" not in lean
    assert "excluded_source_count" not in lean
    assert "metadata_overlay_active" not in lean
    assert lean["message"] == "No knowledge-base generation exists; call ingest."


def test_list_sources_lean_keeps_handles_and_overrides() -> None:
    lean = present_tool_response(
        "list_sources", _list_sources_payload(), detail=LEAN_TOOL_DETAIL
    )

    assert lean["sources"] == [
        {
            "source_id": "src_one",
            "source_relative_path": "evidence.pdf",
            "title": "Citable Evidence",
            "authors": ["A. Researcher"],
        }
    ]
    assert lean["discovered_sources"] == [
        {
            "source_id": "src_one",
            "source_relative_path": "evidence.pdf",
            "included": True,
            "indexed_in_current_generation": True,
        }
    ]
    assert lean["excluded_sources"] == [
        {
            "source_id": "src_two",
            "source_relative_path": "duplicate.pdf",
            "reason": "Reviewed duplicate.",
            "indexed_in_current_generation": False,
        }
    ]
    assert lean["reviewed_metadata_sources"] == [
        {
            "source_id": "src_one",
            "source_relative_path": "evidence.pdf",
            "metadata": {"title": "Citable Evidence"},
        }
    ]
    assert "known_sources" not in lean


def test_ingest_lean_discloses_anomalies_only_when_they_happened() -> None:
    lean = present_tool_response("ingest", _ingest_payload(), detail=LEAN_TOOL_DETAIL)

    assert lean["status"] == "ready"
    assert lean["generation_changed"] is True
    assert lean["reused_document_count"] == 58
    assert "discarded_corrupt_chunk_count" not in lean
    assert "withheld_chunk_reasons" not in lean
    assert "phase_timings_seconds" not in lean
    assert "embedding_model" not in lean

    loud = present_tool_response(
        "ingest",
        _ingest_payload(
            discarded_corrupt_chunk_count=2,
            withheld_chunk_count=1,
            withheld_chunk_reasons={"corrupt_text": {"count": 1}},
        ),
        detail=LEAN_TOOL_DETAIL,
    )
    assert loud["discarded_corrupt_chunk_count"] == 2
    assert loud["withheld_chunk_count"] == 1
    assert loud["withheld_chunk_reasons"] == {"corrupt_text": {"count": 1}}


def test_ingest_in_progress_keeps_resume_state() -> None:
    payload = {
        "status": "in_progress",
        "generation_changed": False,
        "build_id": "20260101T000000Z-abcdef",
        "phase": "embedding",
        "progress": {"completed": 64, "total": 128, "unit": "chunks"},
        "parameters": {"chunk_size": 384},
        "created_at": "2026-01-01T00:00:00Z",
        "checkpointed_at": "2026-01-01T00:00:05Z",
        "next_action": "call_ingest_again",
        "message": "Ingestion checkpoint saved; call ingest again.",
    }
    lean = present_tool_response("ingest", payload, detail=LEAN_TOOL_DETAIL)

    assert lean["status"] == "in_progress"
    assert lean["generation_changed"] is False
    assert lean["build_id"] == "20260101T000000Z-abcdef"
    assert lean["phase"] == "embedding"
    assert lean["progress"]["unit"] == "chunks"
    assert lean["next_action"] == "call_ingest_again"
    assert "parameters" not in lean
    assert "checkpointed_at" not in lean


def test_retired_tools_have_no_projection() -> None:
    for operation in ("set_source_metadata", "export_bundle", "import_bundle"):
        with pytest.raises(ResearchError):
            present_tool_response(operation, {}, detail=LEAN_TOOL_DETAIL)


def test_inclusion_response_is_lean() -> None:
    inclusion = present_tool_response(
        "set_source_inclusion",
        {
            "status": "changed",
            "source_id": "src_two",
            "source_relative_path": "duplicate.pdf",
            "source_path": "sources/duplicate.pdf",
            "included": False,
            "reason": "Reviewed duplicate.",
            "source_file_changed": False,
            "effective_immediately": True,
            "generation_rebuild_recommended": True,
            "message": "Source exclusion saved and enforced for current retrieval.",
        },
        detail=LEAN_TOOL_DETAIL,
    )
    assert set(inclusion) == {
        "status",
        "source_id",
        "source_relative_path",
        "included",
        "reason",
        "effective_immediately",
        "generation_rebuild_recommended",
        "message",
    }
    assert inclusion["included"] is False
    assert inclusion["reason"] == "Reviewed duplicate."
    assert "source_file_changed" not in inclusion
    assert "source_path" not in inclusion


def test_every_public_tool_has_a_lean_projection() -> None:
    for operation, payload in _every_tool_payload().items():
        lean = present_tool_response(operation, payload, detail=LEAN_TOOL_DETAIL)
        assert isinstance(lean, dict)


def test_unknown_tool_name_fails_loudly() -> None:
    with pytest.raises(ResearchError):
        present_tool_response("no_such_tool", {}, detail=LEAN_TOOL_DETAIL)


def test_tool_detail_defaults_to_lean_and_rejects_unknown_modes(project: Path) -> None:
    assert resolve_config(project, vanilla_executable=sys.executable).tool_detail == (
        LEAN_TOOL_DETAIL
    )
    normalized = resolve_config(
        project,
        vanilla_executable=sys.executable,
        tool_detail=" FULL ",
    )
    assert normalized.tool_detail == FULL_TOOL_DETAIL

    with pytest.raises(ConfigurationError):
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            tool_detail="chatty",
        )


def test_server_parser_reads_the_detail_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_ULTRARAG_PROJECT_ROOT", "/tmp/research-project")
    monkeypatch.delenv("RESEARCH_ULTRARAG_TOOL_DETAIL", raising=False)

    assert _parser().parse_args([]).tool_detail == LEAN_TOOL_DETAIL

    monkeypatch.setenv("RESEARCH_ULTRARAG_TOOL_DETAIL", "full")
    assert _parser().parse_args([]).tool_detail == FULL_TOOL_DETAIL
    assert _parser().parse_args(["--tool-detail", LEAN_TOOL_DETAIL]).tool_detail == (
        LEAN_TOOL_DETAIL
    )


def _keys(value: object) -> set[str]:
    """Collect every mapping key in a nested payload."""

    if isinstance(value, dict):
        found = set(value)
        for item in value.values():
            found |= _keys(item)
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found |= _keys(item)
        return found
    return set()


def _hit() -> dict[str, object]:
    """One full service passage, with every diagnostic field the service builds."""

    return {
        "rank": 1,
        "retrieval_rank": 1,
        "chunk_id": "chk_one",
        "document_id": "doc_one",
        "source_id": "src_one",
        "source_path": "sources/evidence.pdf",
        "source_relative_path": "evidence.pdf",
        "title": "Citable Evidence",
        "authors": ["A. Researcher"],
        "year": 2025,
        "doi": "10.1/example",
        "categories": ["research"],
        "keywords": ["wetland"],
        "project": ["ai-and-fetishism"],
        "locator": {"page": 3, "page_label": "3", "type": "pdf_page"},
        "citation": "A. Researcher, Citable Evidence (2025), p. 3",
        "text": "cleaned semantic text",
        "text_fidelity": "cleaned_semantic_text",
        "direct_quote_safe": False,
        "text_notes": [],
        "embedding_token_count": 120,
        "dense_truncated": False,
        "content_kind": "prose",
        "annotations": [],
        "quality_flags": [],
        "metadata_provenance": {"title": "reviewed_override"},
        "metadata_warnings": [],
        "match_kind": "hybrid",
        "retrieval_method": "hybrid",
        "component_ranks": {"bm25": 1, "dense": 1},
        "component_scores": {"dense_cosine_similarity": 0.81, "bm25": None},
        "fusion_score": 0.032,
        "rerank_score": 4.2,
    }


def _anonymous_hit() -> dict[str, object]:
    """A second passage from a source with no author, year, or DOI."""

    return {
        **_hit(),
        "rank": 2,
        "retrieval_rank": 4,
        "chunk_id": "chk_two",
        "document_id": "doc_two",
        "source_id": "src_two",
        "source_path": "sources/anonymous.pdf",
        "source_relative_path": "anonymous.pdf",
        "title": "anonymous",
        "authors": [],
        "year": None,
        "doi": "",
        "locator": {"section_index": 4, "href": "chapter.xhtml", "type": "epub"},
        "citation": "anonymous, section chapter.xhtml",
        "text": "second passage",
        "text_notes": ["non_latin_dominant"],
        "metadata_warnings": ["title_from_filename"],
        "component_ranks": {"bm25": None, "dense": 4},
        "match_kind": "semantic",
    }


def _search_payload(**overrides: object) -> dict[str, object]:
    filters: dict[str, object] = {
        "categories_all": [],
        "categories_any": [],
        "projects_all": [],
        "projects_any": [],
        "keywords_all": [],
        "document_ids": [],
        "source_ids": [],
        "exclude_source_ids": [],
        "unknown_source_ids": list(overrides.pop("unknown_source_ids", [])),
        "unknown_exclude_source_ids": [],
        "active_document_count": 2,
        "note": "Filters narrow the corpus before ranking.",
    }
    payload: dict[str, object] = {
        "query": "cobalt heron amber marsh",
        "generation_id": "20260101T000000Z-abcdef",
        "stale": False,
        "staleness_checked": True,
        "generation_upgrade_required": False,
        "excluded_source_count": 1,
        "filters": filters,
        "retrieval_method": "hybrid",
        "reranked": True,
        "rerank_requested": True,
        "rerank_fallback": None,
        "candidate_depth": 24,
        "candidate_count": 9,
        "candidate_distinct_reference_count": 2,
        "requested_top_k": 6,
        "fusion": {"method": "weighted_reciprocal_rank_fusion", "rrf_k": 60},
        "relevance_policy": {"dense_minimum_cosine_similarity": 0.72},
        "rejected_candidates": {"dense_below_threshold": 11},
        "withheld_candidates": {"policy": "corruption_evidence_only", "total": 0},
        "dense_fidelity": {"embedding_maximum_tokens": 512},
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_model_revision": "52398278842ec682c6f32300af41344b1c0b0bb2",
        "reranker_model": "Xenova/ms-marco-MiniLM-L-6-v2",
        "reranker_model_revision": "a09144355adeed5f58c8ed011d209bf8ee5a1fec",
        "result_count": 2,
        "distinct_reference_count": 2,
        "relevance_limited": False,
        "hits": [_hit(), _anonymous_hit()],
        "notice": "Returned text is cleaned for semantic retrieval.",
    }
    payload.update(overrides)
    return payload


def _status_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "ready": True,
        "stale": False,
        "project_root": "/project",
        "project_id": "121304a5-378c-4384-a0c8-9db4476cec47",
        "project_name": "example",
        "source_root": "/project/sources",
        "state_root": "/state",
        "runtime_root": None,
        "portable_root": "/project/.research-rag",
        "model_cache_root": "/cache/models",
        "version": {
            "server": "0.15.0",
            "installed": "0.15.0",
            "restart_required": False,
        },
        "ui_launcher": {"script_present": True, "link_state": "linked"},
        "generation_id": "20260101T000000Z-abcdef",
        "created_at": "2026-01-01T00:00:00Z",
        "discovered_source_count": 3,
        "selected_source_count": 2,
        "indexed_source_count": 2,
        "searchable_source_count": 2,
        "excluded_source_count": 1,
        "excluded_sources": [{"source_id": "src_two", "reason": "Reviewed duplicate."}],
        "chunk_count": 12,
        "categories": [{"category": "research", "searchable_source_count": 2}],
        "projects": [{"project": "example", "searchable_source_count": 2}],
        "allowed_formats": [".epub", ".pdf"],
        "ignored_extensions": {".md": 4},
        "default_retrieval_method": "hybrid",
        "available_retrieval_methods": ["bm25", "dense", "hybrid"],
        "hybrid_ready": True,
        "hybrid_upgrade_required": False,
        "generation_upgrade_required": False,
        "upgrade_reasons": [],
        "retrieval": {"available_methods": ["bm25", "dense", "hybrid"]},
        "last_build_metrics": {"reused_chunk_count": 12},
        "ingestion_progress": None,
        "source_exclusion_revision": "a" * 64,
        "metadata_revision": "b" * 64,
        "generation_metadata_revision": "c" * 64,
        "metadata_overlay_active": False,
        "metadata_pending_source_paths": [],
        "generation_metadata_snapshot_outdated": False,
        "changes": {
            "added": [],
            "removed": [],
            "modified": [],
            "metadata_changed": False,
            "source_exclusions_changed": False,
        },
        "generation_root": "/state/generations/20260101T000000Z-abcdef",
        "message": "Everything is current.",
        "generations": [
            {
                "generation_id": "20260101T000000Z-abcdef",
                "is_current": True,
                "created_at": "2026-01-01T00:00:00Z",
                "chunk_count": 12,
                "document_count": 2,
                "schema_version": 5,
                "file_count": 14,
                "size_bytes": 4096,
            }
        ],
        "retained_generation_count": 1,
        "retained_generation_bytes": 4096,
    }
    payload.update(overrides)
    return payload


def _list_sources_payload() -> dict[str, object]:
    return {
        "ready": True,
        "generation_id": "20260101T000000Z-abcdef",
        "source_count": 1,
        "sources": [
            {
                **_hit(),
                "sha256": "d" * 64,
                "mtime_ns": 1789753677157890786,
                "size": 103548,
                "format": "pdf",
                "extracted_units": 4,
                "empty_units": 0,
                "physical_pages": 4,
                "excluded_corrupt_unit_count": 0,
                "removed_repeated_margin_blocks": 0,
            }
        ],
        "discovered_source_count": 2,
        "discovered_sources": [
            {
                "source_id": "src_one",
                "source_relative_path": "evidence.pdf",
                "source_path": "sources/evidence.pdf",
                "format": "pdf",
                "included": True,
                "indexed_in_current_generation": True,
            }
        ],
        "known_source_count": 2,
        "known_sources": [
            {
                "source_id": "src_one",
                "source_relative_path": "evidence.pdf",
                "exists": True,
                "included": True,
                "indexed_in_current_generation": True,
                "has_reviewed_metadata": True,
            }
        ],
        "excluded_source_count": 1,
        "excluded_sources": [
            {
                "source_id": "src_two",
                "source_relative_path": "duplicate.pdf",
                "source_path": "sources/duplicate.pdf",
                "reason": "Reviewed duplicate.",
                "excluded_at": "2026-01-01T00:00:00Z",
                "exists": True,
                "indexed_in_current_generation": False,
            }
        ],
        "reviewed_metadata_source_count": 1,
        "reviewed_metadata_sources": [
            {
                "source_id": "src_one",
                "source_relative_path": "evidence.pdf",
                "source_path": "sources/evidence.pdf",
                "metadata": {"title": "Citable Evidence"},
                "indexed_in_current_generation": True,
            }
        ],
    }


def _ingest_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "ready",
        "generation_changed": True,
        "generation_id": "20260101T000000Z-abcdef",
        "generation_root": "/state/generations/20260101T000000Z-abcdef",
        "source_file_count": 63,
        "excluded_source_count": 4,
        "document_count": 59,
        "pdf_count": 58,
        "epub_count": 1,
        "extraction_unit_count": 14072,
        "chunk_count": 14072,
        "content_kind_counts": {"prose": 14000},
        "reused_document_count": 58,
        "rebuilt_document_count": 1,
        "reused_chunk_count": 14044,
        "rebuilt_chunk_count": 28,
        "created_vector_count": 28,
        "reused_vector_count": 14044,
        "discarded_empty_chunk_count": 0,
        "discarded_symbol_only_chunk_count": 0,
        "discarded_corrupt_chunk_count": 0,
        "excluded_corrupt_unit_count": 0,
        "dense_truncated_chunk_count": 0,
        "withheld_chunk_count": 0,
        "withheld_chunk_reasons": {},
        "ignored_extensions": {".md": 61},
        "default_retrieval_method": "hybrid",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "phase_timings_seconds": {"extraction": 2.1},
    }
    payload.update(overrides)
    return payload


def _every_tool_payload() -> dict[str, dict[str, object]]:
    return {
        "status": _status_payload(),
        "ingest": _ingest_payload(),
        "search": _search_payload(),
        "list_sources": _list_sources_payload(),
        "get_passage": {
            "generation_id": "20260101T000000Z-abcdef",
            "requested_chunk_id": "chk_one",
            "context": [_hit()],
            "notice": "Context is cleaned semantic text and is not quote-safe.",
        },
        "set_source_inclusion": {
            "status": "changed",
            "source_id": "src_two",
            "source_relative_path": "duplicate.pdf",
            "source_path": "sources/duplicate.pdf",
            "included": False,
            "reason": "Reviewed duplicate.",
            "source_file_changed": False,
            "effective_immediately": True,
            "generation_rebuild_recommended": True,
            "message": "Source exclusion saved and enforced for current retrieval.",
        },
    }

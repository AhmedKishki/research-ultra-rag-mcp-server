"""Projections from service payloads to MCP tool answers.

Every tool answers with the lean projection, which `present_tool_response`
applies. `--tool-detail full` is the developer detail mode: it returns the
service payload unchanged, for debugging retrieval and ingestion.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .config import FULL_TOOL_DETAIL, LEAN_TOOL_DETAIL, TOOL_DETAIL_MODES
from .service import ResearchError

__all__ = [
    "FULL_TOOL_DETAIL",
    "LEAN_TOOL_DETAIL",
    "TOOL_DETAIL_MODES",
    "lean_ingest",
    "lean_list_sources",
    "lean_passage",
    "lean_passage_context",
    "lean_search",
    "lean_source_inclusion",
    "lean_status",
    "present_tool_response",
]


def _copy(source: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Return an allowlisted copy; a key the source lacks stays absent."""

    return {key: source[key] for key in keys if key in source}


def _meaningful(value: Any) -> bool:
    """Whether a value carries information rather than a default."""

    if value is None or value is False:
        return False
    if isinstance(value, (str, bytes, list, tuple, dict, set)) and not value:
        return False
    return not (isinstance(value, int) and not isinstance(value, bool) and value == 0)


def _add(target: dict[str, Any], key: str, value: Any) -> None:
    """Set a key unless its value is empty, null, false, or zero."""

    if _meaningful(value):
        target[key] = value


def lean_passage(passage: Mapping[str, Any]) -> dict[str, Any]:
    """Return one passage: filename, title, authors, citation, locator, and text.

    The year and DOI live in the citation rather than beside it.
    """

    result: dict[str, Any] = {}
    for key in ("chunk_id", "source_id", "source_relative_path", "title", "authors"):
        _add(result, key, passage.get(key))
    result["locator"] = dict(passage.get("locator") or {})
    result["citation"] = passage.get("citation")
    result["text"] = passage.get("text")
    result["direct_quote_safe"] = bool(passage.get("direct_quote_safe"))
    _add(result, "text_notes", passage.get("text_notes"))
    return result


def lean_search(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a search answer: query, generation, freshness, reranking, evidence.

    `stale` and `reranked` are always present; the other optional fields appear
    only when they carry a value. The reference view answers with
    `reference_groups` instead of `hits`.
    """

    result: dict[str, Any] = _copy(payload, ("query", "generation_id", "stale"))
    result["reranked"] = bool(payload.get("reranked"))
    _add(result, "rerank_fallback", payload.get("rerank_fallback"))
    _add(
        result,
        "generation_upgrade_required",
        payload.get("generation_upgrade_required"),
    )
    filters = payload.get("filters") or {}
    _add(result, "unresolved_source_ids", filters.get("unknown_source_ids"))
    _add(
        result,
        "unresolved_exclude_source_ids",
        filters.get("unknown_exclude_source_ids"),
    )
    result["hits"] = [lean_passage(hit) for hit in payload.get("hits") or []]
    return result


def lean_passage_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the requested passage with its neighbouring passages."""

    result: dict[str, Any] = _copy(payload, ("generation_id", "requested_chunk_id"))
    result["context"] = [lean_passage(item) for item in payload.get("context") or []]
    return result


def lean_generation(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return one retained generation's identity, content, and disk cost."""

    result = _copy(
        record,
        (
            "generation_id",
            "is_current",
            "created_at",
            "chunk_count",
            "document_count",
            "size_bytes",
        ),
    )
    _add(result, "manifest_error", record.get("manifest_error"))
    return result


def lean_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return project state: readiness, freshness, counts, inventory, retention.

    Change lists appear only when the generation is stale, and `restart_required`
    only when the running process is older than the installed version.
    """

    result = _copy(payload, ("ready", "stale", "project_name"))
    for key in (
        "generation_id",
        "created_at",
        "chunk_count",
        "discovered_source_count",
        "selected_source_count",
        "indexed_source_count",
        "searchable_source_count",
        "excluded_source_count",
    ):
        _add(result, key, payload.get(key))
    _add(result, "categories", payload.get("categories"))
    _add(result, "projects", payload.get("projects"))
    _add(
        result,
        "available_retrieval_methods",
        payload.get("available_retrieval_methods"),
    )
    _add(
        result,
        "generation_upgrade_required",
        payload.get("generation_upgrade_required"),
    )
    _add(result, "upgrade_reasons", payload.get("upgrade_reasons"))
    _add(result, "metadata_overlay_active", payload.get("metadata_overlay_active"))
    pending_paths = payload.get("metadata_pending_source_paths")
    if pending_paths:
        result["metadata_pending_source_count"] = len(pending_paths)
    _add(result, "ingestion_progress", payload.get("ingestion_progress"))
    if payload.get("stale"):
        changes = payload.get("changes") or {}
        lean_changes: dict[str, Any] = {}
        for key, source_key in (
            ("added_source_count", "added"),
            ("modified_source_count", "modified"),
        ):
            count = len(changes.get(source_key) or [])
            if count:
                lean_changes[key] = count
        # Available sources are counted, never listed. A source the generation
        # has and the directory does not is named, because that is what a
        # researcher has to act on.
        _add(lean_changes, "removed_sources", changes.get("removed"))
        for key in ("metadata_changed", "source_exclusions_changed"):
            _add(lean_changes, key, changes.get(key))
        if lean_changes:
            result["changes"] = lean_changes
    version = payload.get("version") or {}
    _add(result, "restart_required", version.get("restart_required"))
    generations = payload.get("generations")
    if generations is not None:
        result["generations"] = [lean_generation(item) for item in generations]
        for key in ("retained_generation_count", "retained_generation_bytes"):
            result[key] = payload.get(key)
    result["message"] = payload.get("message")
    return result


def lean_ingest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return ingestion state: what changed, what was reused, what was discarded.

    The counters that report discarded, withheld, or densely truncated material
    appear only when they are not zero.
    """

    result = _copy(payload, ("status", "generation_changed"))
    for key in ("generation_id", "build_id", "phase", "progress"):
        _add(result, key, payload.get(key))
    for key in (
        "document_count",
        "chunk_count",
        "reused_document_count",
        "rebuilt_document_count",
        "reused_chunk_count",
        "rebuilt_chunk_count",
        "created_vector_count",
        "reused_vector_count",
        "discarded_empty_chunk_count",
        "discarded_symbol_only_chunk_count",
        "discarded_corrupt_chunk_count",
        "excluded_corrupt_unit_count",
        "dense_truncated_chunk_count",
        "withheld_chunk_count",
    ):
        _add(result, key, payload.get(key))
    _add(result, "withheld_chunk_reasons", payload.get("withheld_chunk_reasons"))
    _add(result, "next_action", payload.get("next_action"))
    result["message"] = payload.get("message")
    return result


def lean_source_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return one searchable source's handle, title, and authors."""

    result: dict[str, Any] = {}
    for key in ("source_id", "source_relative_path", "title", "authors"):
        _add(result, key, record.get(key))
    return result


def lean_list_sources(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return source handles, inclusion state, and saved metadata overrides.

    `sources` holds the bibliography of what is searchable now and
    `discovered_sources` every live PDF/EPUB with its index state, so a handle
    exists before the first ingestion.
    """

    result = _copy(payload, ("ready",))
    _add(result, "generation_id", payload.get("generation_id"))
    for key in (
        "source_count",
        "discovered_source_count",
        "excluded_source_count",
        "reviewed_metadata_source_count",
    ):
        _add(result, key, payload.get(key))
    result["sources"] = [
        lean_source_record(record) for record in payload.get("sources") or []
    ]
    result["discovered_sources"] = [
        _copy(
            record,
            (
                "source_id",
                "source_relative_path",
                "included",
                "indexed_in_current_generation",
            ),
        )
        for record in payload.get("discovered_sources") or []
    ]
    result["excluded_sources"] = [
        _copy(
            record,
            (
                "source_id",
                "source_relative_path",
                "reason",
                "indexed_in_current_generation",
            ),
        )
        for record in payload.get("excluded_sources") or []
    ]
    result["reviewed_metadata_sources"] = [
        _copy(record, ("source_id", "source_relative_path", "metadata"))
        for record in payload.get("reviewed_metadata_sources") or []
    ]
    return result


def lean_source_inclusion(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the inclusion decision, its reason, and its effect."""

    result = _copy(payload, ("status", "source_id", "source_relative_path"))
    for key in (
        "included",
        "reason",
        "effective_immediately",
        "generation_rebuild_recommended",
    ):
        if key in payload:
            result[key] = payload[key]
    result["message"] = payload.get("message")
    return result


_PROJECTORS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "status": lean_status,
    "ingest": lean_ingest,
    "search": lean_search,
    "list_sources": lean_list_sources,
    "get_passage": lean_passage_context,
    "set_source_inclusion": lean_source_inclusion,
}


def present_tool_response(
    operation: str,
    payload: Mapping[str, Any],
    *,
    detail: str,
) -> dict[str, Any]:
    """Return one tool answer in the configured detail mode."""

    if detail == FULL_TOOL_DETAIL:
        return dict(payload)
    projector = _PROJECTORS.get(operation)
    if projector is None:
        raise ResearchError(f"Unknown research tool: {operation}")
    return projector(payload)

"""High-level, project-scoped research knowledge-base workflow."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from filelock import AsyncFileLock
from filelock import Timeout as FileLockTimeout

from .artifact_lookup import (
    LOOKUP_HEALTH_FLAGS_KEY,
    LOOKUP_RELATIVE_PATH,
    ArtifactLookup,
    ArtifactLookupError,
    ensure_artifact_lookup,
)
from .config import ResearchConfig, resolve_source_reference
from .dense import (
    EXACT_BACKEND_NAME,
    QDRANT_BACKEND_NAME,
    DenseBackend,
    DenseSearchHit,
    DenseTokenAuditUnavailable,
    LocalQdrantDenseBackend,
    LocalVectorDenseBackend,
    RerankerUnavailable,
)
from .embeddings import EmbeddingModel
from .extraction import (
    CHUNK_FLAG_CORRUPT_TEXT,
    CHUNK_FLAG_EXTRACTION_ARTIFACT,
    ExtractionError,
    chunk_health_flags,
    extract_epub_spine_item,
    extract_scanned_pdf_pages,
    has_searchable_alphanumeric_content,
    normalize_inline_text,
    normalize_reading_text,
    pdf_page_count,
    prepare_epub_extraction,
    prepare_scanned_pdf,
    scan_pdf_pages,
    text_corruption_reasons,
    text_health_reasons,
    text_script_notes,
)
from .generation import (
    generation_artifacts_are_valid,
    load_reuse_snapshot,
    source_set_matches,
    value_fingerprint,
)
from .launcher import ui_launcher_state
from .rerankers import resolve_reranker_model
from .settings import SETTINGS_BY_KEY, EffectiveSettings
from .sources import (
    ALLOWED_SOURCE_EXTENSIONS,
    SourceFile,
    SourcePolicyError,
    SourceScan,
    normalize_metadata,
    scan_sources,
    sha256_file,
    stable_source_id,
)
from .storage import (
    StorageError,
    atomic_write_json,
    atomic_write_jsonl,
    directory_statistics,
    fsync_directories,
    fsync_directory,
    iter_jsonl,
    load_current_generation,
    load_metadata_overrides,
    load_source_catalog,
    load_source_exclusions,
    read_json,
    read_jsonl,
    write_handoff_jsonl,
    write_source_catalog,
    write_source_exclusions,
)
from .ultrarag import VanillaUltraRAG
from .version import version_block

SCHEMA_VERSION = 5
CLEANING_POLICY_VERSION = 3
EXTRACTION_POLICY_VERSION = 7
ARTIFACT_POLICY_VERSION = 3
INGESTION_CHECKPOINT_VERSION = 1
INGESTION_IDENTITY_POLICY_VERSION = 2
PENDING_ACTIVATION_VERSION = 1
METADATA_STORAGE_POLICY = "automatic_only_runtime_overlay_v1"
# The retrieval policy is fixed here — the tool offers exactly one way to
# search — while the numbers that shape it live in the settings file, so fusion
# weights, gates, batch sizes, and budgets are tunable without editing code.
DEFAULT_RETRIEVAL_METHOD = "hybrid"
RETRIEVAL_METHODS = frozenset({"bm25", "dense", "hybrid"})
# The generation-relative directory each dense backend writes its index into.
DENSE_INDEX_PATHS = {
    QDRANT_BACKEND_NAME: "indexes/qdrant",
    EXACT_BACKEND_NAME: "indexes/vectors",
}


def retrieval_policy_fingerprint(settings: EffectiveSettings) -> str:
    """Return the identity of the ranking policy these settings describe.

    The fusion constants and the relevance gates decide what a search returns,
    so their values are part of what a generation *is*: a generation that
    recorded a different policy than this process runs is not reusable, and the
    next ingestion builds a new generation instead of mixing two.
    """

    return value_fingerprint(
        {
            "default_method": DEFAULT_RETRIEVAL_METHOD,
            "available_methods": sorted(RETRIEVAL_METHODS),
            "bm25": {
                "language": settings.bm25_stopwords_language,
                "tokenizer": "default",
            },
            "fusion": {
                "method": "weighted_reciprocal_rank_fusion",
                "rrf_k": settings.rrf_k,
                "bm25_weight": settings.bm25_weight,
                "dense_weight": settings.dense_weight,
                "minimum_candidates": settings.minimum_candidates,
                "maximum_candidates": settings.maximum_candidates,
            },
            "relevance_gates": {
                "bm25_requires_query_token_overlap": True,
                "dense_minimum_cosine_similarity": (
                    settings.dense_minimum_cosine_similarity
                ),
            },
        }
    )


_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


class ResearchError(RuntimeError):
    """User-facing research workflow failure."""


class _SourceChangedDuringIngest(RuntimeError):
    """Internal signal that a staged input snapshot is no longer current."""


async def _atomic_to_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Let a write-side thread finish its atomic unit before propagating cancel."""

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _generation_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _source_work_key(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]


def _pdf_batch_count(page_count: int, batch_size: int) -> int:
    if page_count < 0 or batch_size <= 0:
        raise ValueError("PDF page and batch counts must be valid")
    return (page_count + batch_size - 1) // batch_size


def _source_stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _hash_with_stable_stat(path: Path) -> tuple[str, dict[str, int]]:
    before = _source_stat_identity(path)
    digest = sha256_file(path)
    after = _source_stat_identity(path)
    if before != after:
        raise OSError("Source changed while it was being hashed")
    return digest, after


def _source_inventory(scan: SourceScan) -> list[dict[str, Any]]:
    return [
        {
            "source_id": source.source_id,
            "source_path": source.project_relative_path,
            "source_relative_path": source.source_relative_path,
            "format": source.extension.removeprefix("."),
            "size": source.size,
            "mtime_ns": source.mtime_ns,
        }
        for source in scan.selected
    ]


def _checkpoint_identity(
    *,
    project_id: str,
    inventory: list[dict[str, Any]],
    exclusion_revision: str,
    baseline_generation_id: str | None,
    chunk_size: int,
    chunk_overlap: int,
    force_recompute: bool,
    retrieval_policy: str,
    embedding: EmbeddingModel,
) -> str:
    return value_fingerprint(
        {
            "project_id": project_id,
            "inventory": inventory,
            "exclusion_revision": exclusion_revision,
            "baseline_generation_id": baseline_generation_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "force_recompute": force_recompute,
            "generation_schema_version": SCHEMA_VERSION,
            "extraction_policy_version": EXTRACTION_POLICY_VERSION,
            "cleaning_policy_version": CLEANING_POLICY_VERSION,
            "artifact_policy_version": ARTIFACT_POLICY_VERSION,
            "retrieval_policy_fingerprint": retrieval_policy,
            "embedding_model": embedding.name,
            "embedding_model_revision": embedding.revision,
            "embedding_dimension": embedding.dimension,
            "ingestion_identity_policy_version": (INGESTION_IDENTITY_POLICY_VERSION),
            "metadata_storage_policy": METADATA_STORAGE_POLICY,
        }
    )


def _citation(document: dict[str, Any], locator: dict[str, Any]) -> str:
    authors = document.get("authors") or []
    creator = "; ".join(str(item) for item in authors) if authors else ""
    title = str(document.get("title") or document.get("source_path") or "Source")
    year = document.get("year")
    lead = creator or title
    if creator and title:
        lead = f"{creator}, {title}"
    if year:
        lead = f"{lead} ({year})"
    doi = normalize_inline_text(str(document.get("doi") or ""))
    if doi:
        lead = f"{lead}, doi:{doi.removeprefix('doi:')}"

    if locator.get("type") == "pdf_page":
        location = f"p. {locator.get('page_label') or locator.get('page')}"
    else:
        section = (
            locator.get("href_with_fragment")
            or locator.get("section_title")
            or locator.get("href")
        )
        location = (
            f"section {section}"
            if section
            else f"EPUB section {locator.get('section_index')}"
        )
    return f"{lead}, {location}"


def _content_tokens(value: str) -> set[str]:
    """Return meaningful Unicode word tokens used for lexical abstention."""

    return {
        token
        for match in _WORD.finditer(value.casefold())
        if (token := match.group(0)) not in _STOPWORDS
    }


def _normalized_filter(values: list[str] | None) -> set[str]:
    return {
        normalized.casefold()
        for value in values or []
        if (normalized := normalize_inline_text(str(value)))
    }


def _requested_ids(values: list[str] | None) -> list[str]:
    """Return caller-supplied IDs in order, without blanks or repeats."""

    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        item = str(value).strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _document_matches_metadata(
    document: dict[str, Any],
    *,
    keywords: set[str],
    categories_any: set[str] = frozenset(),
    projects_any: set[str] = frozenset(),
) -> bool:
    """Match one document against the reviewed-metadata filter layers.

    `project` records which project a source was gathered for, `categories` the
    branches it belongs to, and `keywords` the terms that identify it. A source
    normally carries one project, so that layer is a passthrough inside a
    one-project server and becomes meaningful when a corpus is copied or shared.
    """

    document_categories = _normalized_filter(document.get("categories"))
    document_keywords = _normalized_filter(document.get("keywords"))
    document_projects = _normalized_filter(document.get("project"))
    return (
        keywords.issubset(document_keywords)
        and (not categories_any or not categories_any.isdisjoint(document_categories))
        and (not projects_any or not projects_any.isdisjoint(document_projects))
    )


def _metadata_inventory(
    documents: dict[str, dict[str, Any]],
    excluded_document_ids: set[str],
    *,
    field: str,
    label: str,
) -> list[dict[str, Any]]:
    """Count searchable sources per reviewed value of one list-valued field.

    Reviewed values are free strings, so this is the inventory an agent uses to
    see a corpus partition (categories) or its project tags before searching. A
    source is counted once per value it carries, and reviewed exclusions are not
    counted.
    """

    display: dict[str, str] = {}
    counts: dict[str, int] = {}
    for document_id, document in documents.items():
        if document_id in excluded_document_ids:
            continue
        normalized = _normalized_filter(document.get(field))
        if not normalized:
            continue
        for raw in document.get(field) or []:
            value = normalize_inline_text(str(raw))
            if value and value.casefold() in normalized:
                display.setdefault(value.casefold(), value)
        for name in normalized:
            counts[name] = counts.get(name, 0) + 1
    return [
        {label: display.get(name, name), "searchable_source_count": counts[name]}
        for name in sorted(counts)
    ]


def _public_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return document metadata without extraction-related line wrapping."""

    result = dict(document)
    # These fields existed in older immutable generations but are internal
    # implementation details, not reviewed metadata or useful diagnostics.
    result.pop("metadata_confidence", None)
    result.pop("metadata_override_revision", None)
    for field in ("title", "doi"):
        result[field] = normalize_inline_text(str(result.get(field) or ""))
    for field in ("authors", "categories", "keywords", "project"):
        result[field] = [
            normalized
            for value in result.get(field) or []
            if (normalized := normalize_inline_text(str(value)))
        ]
    provenance = dict(result.get("metadata_provenance") or {})
    warnings = list(result.get("metadata_warnings") or [])
    if provenance.get("title") != "reviewed_override" and text_health_reasons(
        result["title"]
    ):
        result["title"] = Path(str(result.get("source_path") or "source")).stem
        warnings.append("corrupt_extracted_title")
    if provenance.get("authors") != "reviewed_override":
        clean_authors = [
            author for author in result["authors"] if not text_health_reasons(author)
        ]
        if len(clean_authors) != len(result["authors"]):
            warnings.append("corrupt_extracted_authors")
        result["authors"] = clean_authors
    result["metadata_warnings"] = list(dict.fromkeys(warnings))
    return result


def _reranker_revision(model: str) -> str:
    """Return the pinned revision of one supported reranker model."""

    return resolve_reranker_model(model)[1]


def _canonical_metadata_override(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize one override while preserving explicit empty reviewed values."""

    normalized = normalize_metadata(value)
    result: dict[str, Any] = {}
    for field in ("title", "doi"):
        if field in normalized:
            result[field] = normalize_inline_text(str(normalized[field]))
    for field in ("authors", "categories", "keywords", "project"):
        if field not in normalized:
            continue
        items: list[str] = []
        seen: set[str] = set()
        for raw in normalized[field]:
            item = normalize_inline_text(str(raw))
            key = item.casefold()
            if item and key not in seen:
                items.append(item)
                seen.add(key)
        result[field] = items
    if "year" in normalized:
        result["year"] = normalized["year"]
    return result


def _metadata_snapshot_changed(
    manifest: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> bool:
    """Compare portable metadata with a generation's observational snapshot."""

    stored_revision = manifest.get("metadata_revision")
    if stored_revision is None:
        return bool(metadata)
    return stored_revision != value_fingerprint(metadata)


def _effective_document_metadata(
    document: dict[str, Any],
    override: dict[str, Any],
    *,
    unknown_legacy_snapshot_mismatch: bool = False,
) -> dict[str, Any]:
    """Overlay current reviewed metadata without mutating generation artifacts.

    Generation documents retain the metadata snapshot used while building their
    immutable indexes.  The portable reviewed-metadata file is authoritative at
    read time, so corrections can take effect without rebuilding those indexes.

    Older generations do not retain the automatic value hidden by a reviewed
    bibliographic override.  If such an override is later removed, use a safe
    deterministic fallback instead of silently retaining the value the user
    removed.  A later ingestion can recover automatic metadata from the source.
    """

    normalized_override = _canonical_metadata_override(override)
    override_revision = value_fingerprint(normalized_override)
    stored_override_revision = document.get("metadata_override_revision")
    if stored_override_revision == override_revision:
        result = dict(document)
        result.pop("metadata_confidence", None)
        return result

    result = dict(document)
    result.pop("metadata_confidence", None)
    provenance = dict(result.get("metadata_provenance") or {})
    warnings = [
        str(item)
        for item in result.get("metadata_warnings") or []
        if item
        not in {
            "authors_missing",
            "title_from_filename",
            "automatic_metadata_unavailable_after_override_removal",
        }
    ]
    removed_reviewed_value = False
    unknown_legacy_snapshot = (
        unknown_legacy_snapshot_mismatch
        and not provenance
        and stored_override_revision != override_revision
    )

    fallbacks: dict[str, Any] = {
        "title": Path(str(result.get("source_path") or "source")).stem,
        "authors": [],
        "year": None,
        "doi": "",
    }
    for field in ("title", "authors", "year", "doi"):
        if field in normalized_override:
            result[field] = normalized_override[field]
            provenance[field] = "reviewed_override"
        elif provenance.get(field) == "reviewed_override" or unknown_legacy_snapshot:
            result[field] = fallbacks[field]
            provenance[field] = "filename" if field == "title" else "missing"
            removed_reviewed_value = True

    # Categories, keywords, and the project tag have no automatic extraction
    # source, so the current reviewed lists can be represented exactly even on
    # old generations.
    for field in ("categories", "keywords", "project"):
        result[field] = list(normalized_override.get(field, []))
        provenance[field] = (
            "reviewed_override" if field in normalized_override else "missing"
        )

    if not result.get("authors"):
        warnings.append("authors_missing")
    if provenance.get("title") == "filename":
        warnings.append("title_from_filename")
    if removed_reviewed_value:
        warnings.append("automatic_metadata_unavailable_after_override_removal")

    result["metadata_provenance"] = provenance
    result["metadata_warnings"] = list(dict.fromkeys(warnings))
    result["metadata_override_revision"] = override_revision
    return result


def _effective_documents(
    manifest: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return current read-time document metadata keyed by stable document ID."""

    legacy_snapshot_mismatch = manifest.get(
        "metadata_storage_policy"
    ) != METADATA_STORAGE_POLICY and _metadata_snapshot_changed(manifest, metadata)
    project_id = str(manifest.get("project_id") or "")
    result: dict[str, dict[str, Any]] = {}
    for stored in manifest.get("documents", []):
        relative = str(stored.get("source_relative_path") or "")
        document = _effective_document_metadata(
            stored,
            metadata.get(relative, {}),
            unknown_legacy_snapshot_mismatch=legacy_snapshot_mismatch,
        )
        if not document.get("source_id") and project_id and relative:
            document["source_id"] = stable_source_id(project_id, relative)
        result[str(document["document_id"])] = document
    return result


def _document_for_chunk(
    chunk: dict[str, Any],
    documents_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    document_id = str(chunk.get("document_id") or "")
    document = documents_by_id.get(document_id)
    if document is None:
        raise ResearchError(
            f"Current generation chunk references unknown document: {document_id}"
        )
    return document


def _chunk_text(chunk: dict[str, Any]) -> str:
    """Read canonical passage text with compatibility for older generations."""

    return str(
        chunk.get("contents") or chunk.get("text") or chunk.get("embedding_text") or ""
    )


def _public_passage(
    chunk: dict[str, Any],
    document: dict[str, Any],
) -> dict[str, Any]:
    """Project one immutable chunk and current document metadata for clients."""

    public_document = _public_document(document)
    locator = dict(chunk.get("locator") or {})
    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "source_id": public_document["source_id"],
        "source_path": public_document["source_path"],
        "source_relative_path": public_document.get("source_relative_path"),
        "title": public_document["title"],
        "authors": public_document["authors"],
        "year": public_document.get("year"),
        "doi": public_document["doi"],
        "categories": public_document["categories"],
        "keywords": public_document["keywords"],
        "project": public_document["project"],
        "locator": locator,
        "citation": normalize_inline_text(_citation(public_document, locator)),
        "text": normalize_reading_text(_chunk_text(chunk)),
        "text_fidelity": "cleaned_semantic_text",
        "direct_quote_safe": False,
        "text_notes": text_script_notes(_chunk_text(chunk)),
        "embedding_token_count": chunk.get("embedding_token_count"),
        "dense_truncated": chunk.get("dense_truncated"),
        "content_kind": str(chunk.get("content_kind") or "prose"),
        "annotations": list(chunk.get("annotations") or []),
        "quality_flags": list(chunk.get("quality_flags") or []),
        "metadata_provenance": dict(public_document.get("metadata_provenance") or {}),
        "metadata_warnings": list(public_document.get("metadata_warnings") or []),
    }


def _enrich_chunks(
    raw_chunks: list[dict[str, Any]],
    units: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int, int]:
    units_by_id = {str(item["id"]): item for item in units}
    documents_by_id = {str(item["document_id"]): item for item in documents}
    document_ordinals: defaultdict[str, int] = defaultdict(int)
    enriched: list[dict[str, Any]] = []
    discarded_empty_chunks = 0
    discarded_symbol_only_chunks = 0
    discarded_corrupt_chunks = 0

    for raw in raw_chunks:
        unit_id = str(raw.get("doc_id") or "")
        unit = units_by_id.get(unit_id)
        if unit is None:
            raise ResearchError(
                f"UltraRAG returned an unknown extraction unit: {unit_id}"
            )
        document_id = str(unit["document_id"])
        document = documents_by_id[document_id]
        text = normalize_reading_text(str(raw.get("contents") or ""))
        if not text:
            discarded_empty_chunks += 1
            continue
        if not has_searchable_alphanumeric_content(text):
            discarded_symbol_only_chunks += 1
            continue
        if text_corruption_reasons(text):
            discarded_corrupt_chunks += 1
            continue

        ordinal = document_ordinals[document_id]
        document_ordinals[document_id] += 1
        identity = f"{unit_id}\0{ordinal}\0{text}".encode()
        chunk_id = f"chk_{hashlib.sha256(identity).hexdigest()[:24]}"
        locator = dict(unit["locator"])
        enriched.append(
            {
                "id": chunk_id,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source_id": document["source_id"],
                "document_chunk_index": ordinal,
                "unit_id": unit_id,
                "locator": locator,
                "contents": text,
                "content_kind": str(unit.get("content_kind") or "prose"),
                "annotations": list(unit.get("annotations") or []),
                "quality_flags": list(unit.get("quality_flags") or []),
            }
        )

    represented = {item["document_id"] for item in enriched}
    missing = [
        item["source_path"]
        for item in documents
        if item["document_id"] not in represented
    ]
    if missing:
        raise ResearchError(
            "No searchable chunks were produced for: " + ", ".join(missing)
        )
    return (
        enriched,
        discarded_empty_chunks,
        discarded_symbol_only_chunks,
        discarded_corrupt_chunks,
    )


def _is_extraction_artifact(chunk: dict[str, Any]) -> bool:
    quality_flags = {str(item) for item in chunk.get("quality_flags", [])}
    return "extraction_artifact" in quality_flags or not (
        has_searchable_alphanumeric_content(_chunk_text(chunk))
    )


def _candidate_flags(chunk: dict[str, Any]) -> int:
    """Return the stored retrieval-rejection verdict for one candidate.

    The artifact lookup computes this when it is built, so a query does not
    rescan chunk text. A generation whose lookup predates the stored verdict
    falls back to computing the same flags here.
    """

    stored = chunk.get(LOOKUP_HEALTH_FLAGS_KEY)
    if isinstance(stored, int):
        return stored
    return chunk_health_flags(
        _chunk_text(chunk),
        quality_flags=chunk.get("quality_flags"),
    )


def _record_withheld(
    withheld: dict[str, dict[str, Any]],
    chunk: dict[str, Any],
    reasons: Sequence[str],
    *,
    limit: int,
) -> None:
    """Record why a candidate was withheld so the response can disclose it."""

    for reason in reasons:
        entry = withheld.setdefault(reason, {"count": 0, "example_chunk_ids": []})
        entry["count"] = int(entry["count"]) + 1
        examples = entry["example_chunk_ids"]
        if len(examples) < limit:
            examples.append(str(chunk["chunk_id"]))


def _record_embedding_token_counts(
    chunks: list[dict[str, Any]],
    count_tokens: Any,
    *,
    maximum_tokens: int,
) -> bool:
    """Record each built chunk's embedding token count and truncation flag.

    FastEmbed silently truncates input that exceeds the embedding model's limit,
    so the dense vector of such a chunk covers only a prefix while BM25 indexes
    the whole text. The audit makes that visible per chunk. Returning False means
    the tokenizer could not be inspected; the fields stay absent and the build
    metrics report the audit as unavailable rather than inventing a value.
    """

    if not chunks:
        return True
    texts = [_chunk_text(chunk) for chunk in chunks]
    try:
        counts = count_tokens(texts)
    except DenseTokenAuditUnavailable:
        return False
    if len(counts) != len(chunks):
        raise ResearchError(
            "The embedding tokenizer returned a different count than chunks"
        )
    for chunk, count in zip(chunks, counts, strict=True):
        chunk["embedding_token_count"] = int(count)
        chunk["dense_truncated"] = int(count) > maximum_tokens
    return True


class ResearchService:
    def __init__(
        self,
        config: ResearchConfig,
        ultrarag: VanillaUltraRAG,
        dense: DenseBackend | None = None,
    ) -> None:
        self.config = config
        self.ultrarag = ultrarag
        self._dense_backends: dict[str, DenseBackend]
        if dense is not None:
            # An injected backend serves every recorded kind, so deterministic
            # test doubles stand in for both real backends.
            self._dense_backends = {name: dense for name in DENSE_INDEX_PATHS}
        else:
            self._dense_backends = {
                QDRANT_BACKEND_NAME: LocalQdrantDenseBackend(
                    config.models_root,
                    offline=config.offline,
                    embedding_threads=config.embedding_threads,
                    reranker_model=config.reranker_model,
                    embedding_inference_batch_size=(
                        config.settings.embedding_inference_batch_size
                    ),
                ),
                EXACT_BACKEND_NAME: LocalVectorDenseBackend(
                    config.models_root,
                    offline=config.offline,
                    embedding_threads=config.embedding_threads,
                    reranker_model=config.reranker_model,
                    embedding_inference_batch_size=(
                        config.settings.embedding_inference_batch_size
                    ),
                ),
            }
        # The primary backend answers model-only calls; indexing, validation, and
        # retrieval resolve the backend that the generation itself recorded.
        self.dense = (
            self._dense_backends[EXACT_BACKEND_NAME]
            if config.dense_backend in {"auto", "exact"}
            else self._dense_backends[QDRANT_BACKEND_NAME]
        )
        # The ranking policy is part of a generation identity, so it is
        # computed once from the settings this process resolved.
        self.retrieval_policy_fingerprint = retrieval_policy_fingerprint(
            config.settings
        )
        self._lock = asyncio.Lock()
        self._project_lock = AsyncFileLock(
            config.state_root / "project.lock",
            timeout=1800,
        )
        self._loaded_generation: str | None = None

    @asynccontextmanager
    async def _operation(self) -> AsyncIterator[None]:
        """Serialize project access across MCP and UI server processes."""

        async with self._lock:
            try:
                async with self._project_lock:
                    yield
            except FileLockTimeout as exc:
                raise ResearchError(
                    "Timed out waiting for another research process to finish"
                ) from exc

    @staticmethod
    def _dense_backend_name(manifest: dict[str, Any] | None) -> str:
        """Return the dense backend a generation recorded.

        Generations written before the backend was recorded always used the
        embedded Qdrant index.
        """

        if manifest:
            dense = manifest.get("retrieval", {}).get("dense", {})
            recorded = dense.get("dense_backend")
            if isinstance(recorded, str) and recorded in DENSE_INDEX_PATHS:
                return recorded
        return QDRANT_BACKEND_NAME

    def _dense_for(self, manifest: dict[str, Any] | None) -> DenseBackend:
        """Return the dense backend that owns a generation's index."""

        return self._dense_backends[self._dense_backend_name(manifest)]

    def _select_build_dense_backend(self, chunk_count: int) -> str:
        """Choose the dense backend for a new generation.

        `auto` uses the exact scan while a linear scan is cheaper than
        maintaining an ANN index, and falls back above the documented threshold.
        """

        if self.config.dense_backend == "exact":
            return EXACT_BACKEND_NAME
        if self.config.dense_backend == "qdrant":
            return QDRANT_BACKEND_NAME
        if chunk_count <= self.config.settings.exact_backend_chunk_limit:
            return EXACT_BACKEND_NAME
        return QDRANT_BACKEND_NAME

    def _metadata(self) -> dict[str, dict[str, Any]]:
        try:
            values = load_metadata_overrides(self.config.metadata_path)
            normalized = {
                key: _canonical_metadata_override(value)
                for key, value in values.items()
            }
            return {key: value for key, value in normalized.items() if value}
        except (StorageError, SourcePolicyError) as exc:
            raise ResearchError(str(exc)) from exc

    def _source_exclusions(self) -> dict[str, dict[str, str]]:
        try:
            return load_source_exclusions(self.config.source_exclusions_path)
        except StorageError as exc:
            raise ResearchError(str(exc)) from exc

    def _source_catalog(self) -> dict[str, str]:
        try:
            return load_source_catalog(
                self.config.source_catalog_path,
                project_id=self.config.project_id,
            )
        except StorageError as exc:
            raise ResearchError(str(exc)) from exc

    def _sync_source_catalog(
        self,
        scan: SourceScan,
        current: tuple[Path, dict[str, Any]] | None,
        *,
        exclusions: dict[str, dict[str, str]] | None = None,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        """Durably retain every issued opaque source ID and its project path."""

        catalog = self._source_catalog()
        updated = dict(catalog)
        paths_to_ids = {relative: source_id for source_id, relative in catalog.items()}

        def register(relative: str, recorded_source_id: str | None = None) -> None:
            try:
                expected_source_id = stable_source_id(
                    self.config.project_id,
                    relative,
                )
            except SourcePolicyError as exc:
                raise ResearchError(str(exc)) from exc
            if Path(relative).suffix.casefold() not in ALLOWED_SOURCE_EXTENSIONS:
                raise ResearchError(
                    f"Source catalog contains an unsupported source path: {relative}"
                )
            if recorded_source_id and recorded_source_id != expected_source_id:
                raise ResearchError(
                    f"Source ID does not match its project path: {relative}"
                )
            existing_path = updated.get(expected_source_id)
            existing_id = paths_to_ids.get(relative)
            if existing_path not in {None, relative} or existing_id not in {
                None,
                expected_source_id,
            }:
                raise ResearchError(
                    f"Conflicting source identity in project catalog: {relative}"
                )
            updated[expected_source_id] = relative
            paths_to_ids[relative] = expected_source_id

        manifest = current[1] if current is not None else {}
        stored_records = manifest.get("source_files")
        if not isinstance(stored_records, list):
            stored_records = manifest.get("documents", [])
        for item in stored_records:
            if not isinstance(item, dict):
                continue
            relative = str(item.get("source_relative_path") or "")
            if relative:
                register(relative, str(item.get("source_id") or "") or None)
        for relative in set(
            self._source_exclusions() if exclusions is None else exclusions
        ) | set(self._metadata() if metadata is None else metadata):
            register(relative)
        for source in scan.selected:
            register(source.source_relative_path, source.source_id)

        if updated != catalog:
            write_source_catalog(
                self.config.source_catalog_path,
                updated,
                project_id=self.config.project_id,
            )
        return updated

    def _known_sources(
        self,
        scan: SourceScan,
        current: tuple[Path, dict[str, Any]] | None,
        *,
        exclusions: dict[str, dict[str, str]] | None = None,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return live and selected-generation sources keyed by relative path."""

        catalog = self._sync_source_catalog(
            scan,
            current,
            exclusions=exclusions,
            metadata=metadata,
        )
        manifest = current[1] if current is not None else {}
        source_directory = Path(
            str(
                manifest.get("source_directory")
                or self.config.source_root.relative_to(self.config.project_root)
            )
        )
        records: dict[str, dict[str, Any]] = {
            relative: {
                "source_id": source_id,
                "source_relative_path": relative,
                "source_path": (source_directory / relative).as_posix(),
                "format": Path(relative).suffix.removeprefix(".").casefold(),
                "exists": False,
                "indexed_in_current_generation": False,
            }
            for source_id, relative in catalog.items()
        }
        indexed_paths = {
            str(item.get("source_relative_path") or "")
            for item in manifest.get("documents", [])
            if isinstance(item, dict)
        }
        stored_records = manifest.get("source_files")
        if not isinstance(stored_records, list):
            stored_records = manifest.get("documents", [])
        for item in stored_records:
            if not isinstance(item, dict):
                continue
            relative = str(item.get("source_relative_path") or "")
            if not relative:
                continue
            records[relative] = {
                **item,
                "source_id": str(item.get("source_id") or "")
                or stable_source_id(self.config.project_id, relative),
                "source_relative_path": relative,
                "source_path": str(
                    item.get("source_path") or (source_directory / relative).as_posix()
                ),
                "exists": False,
                "indexed_in_current_generation": relative in indexed_paths,
            }
        # Keep persisted policy records addressable even after their files are
        # removed and before they have ever appeared in a generation.  This
        # makes every source_id returned by list_sources usable by the mutation
        # tools, rather than exposing an ID that cannot restore its own record.
        policy_paths = set(
            self._source_exclusions() if exclusions is None else exclusions
        ) | set(self._metadata() if metadata is None else metadata)
        for relative in policy_paths:
            records.setdefault(
                relative,
                {
                    "source_id": stable_source_id(self.config.project_id, relative),
                    "source_relative_path": relative,
                    "source_path": (source_directory / relative).as_posix(),
                    "format": Path(relative).suffix.removeprefix(".").casefold(),
                    "exists": False,
                    "indexed_in_current_generation": False,
                },
            )
        for source in scan.selected:
            existing = records.get(source.source_relative_path, {})
            records[source.source_relative_path] = {
                **existing,
                "source_id": source.source_id,
                "source_relative_path": source.source_relative_path,
                "source_path": source.project_relative_path,
                "format": source.extension.removeprefix("."),
                "exists": True,
                "indexed_in_current_generation": (
                    source.source_relative_path in indexed_paths
                ),
            }
        return records

    def _resolve_source_selector(
        self,
        *,
        source_id: str | None,
        source_path: str | None,
        scan: SourceScan,
        current: tuple[Path, dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Resolve exactly one stable ID or source-relative compatibility path."""

        if (source_id is None) == (source_path is None):
            raise SourcePolicyError("Provide exactly one of source_id or source_path")
        known = self._known_sources(scan, current)
        if source_id is not None:
            normalized_id = source_id.strip()
            if not normalized_id:
                raise SourcePolicyError("source_id must not be empty")
            matches = [
                record
                for record in known.values()
                if record.get("source_id") == normalized_id
            ]
            if not matches:
                raise SourcePolicyError(f"Unknown source_id: {normalized_id}")
            if len(matches) != 1:  # Defensive: deterministic IDs must be unique.
                raise SourcePolicyError(f"Ambiguous source_id: {normalized_id}")
            return matches[0]

        assert source_path is not None
        source = resolve_source_reference(self.config, source_path)
        relative = source.relative_to(self.config.source_root).as_posix()
        record = known.get(relative)
        if record is None:
            raise SourcePolicyError(
                f"Unknown PDF or EPUB source_relative_path: {source_path}"
            )
        return record

    @staticmethod
    def _excluded_document_ids(
        manifest: dict[str, Any],
        exclusions: dict[str, dict[str, str]],
    ) -> set[str]:
        return {
            str(document["document_id"])
            for document in manifest.get("documents", [])
            if str(document.get("source_relative_path")) in exclusions
        }

    def _document_ids_for_source_ids(
        self,
        manifest: dict[str, Any],
        source_ids: list[str],
    ) -> tuple[set[str], list[str]]:
        """Resolve stable source IDs to document IDs in the given generation.

        Returns the matched document IDs and the requested IDs that resolved to
        nothing here. A source absent from the selected generation, or renamed or
        moved since the generation was built, cannot resolve because a
        `source_id` is derived from the current normalized relative path.
        """

        if not source_ids:
            return set(), []
        by_source_id: dict[str, set[str]] = {}
        for document in manifest.get("documents", []):
            relative = str(document.get("source_relative_path") or "")
            document_id = str(document.get("document_id") or "")
            if not relative or not document_id:
                continue
            try:
                source_id = stable_source_id(self.config.project_id, relative)
            except SourcePolicyError:
                continue
            by_source_id.setdefault(source_id, set()).add(document_id)
        matched: set[str] = set()
        unknown: list[str] = []
        for requested in source_ids:
            document_ids = by_source_id.get(requested)
            if document_ids is None:
                unknown.append(requested)
            else:
                matched.update(document_ids)
        return matched, unknown

    def _exclusion_records(
        self,
        scan: SourceScan,
        exclusions: dict[str, dict[str, str]],
        manifest: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        scanned = {item.source_relative_path: item for item in scan.selected}
        indexed_paths = {
            str(document.get("source_relative_path"))
            for document in (manifest or {}).get("documents", [])
        }
        source_directory = Path(
            str(
                (manifest or {}).get("source_directory")
                or self.config.source_root.relative_to(self.config.project_root)
            )
        )
        records: list[dict[str, Any]] = []
        for relative, exclusion in exclusions.items():
            source = scanned.get(relative)
            records.append(
                {
                    "source_id": (
                        source.source_id
                        if source is not None
                        else stable_source_id(self.config.project_id, relative)
                    ),
                    "source_relative_path": relative,
                    "source_path": (
                        source.project_relative_path
                        if source is not None
                        else (source_directory / relative).as_posix()
                    ),
                    "reason": exclusion["reason"],
                    "excluded_at": exclusion["excluded_at"],
                    "exists": source is not None,
                    "indexed_in_current_generation": relative in indexed_paths,
                }
            )
        return records

    def _load_current_optional(self) -> tuple[Path, dict[str, Any]] | None:
        if not self.config.current_path.exists():
            return None
        try:
            return load_current_generation(self.config.state_root)
        except StorageError as exc:
            raise ResearchError(str(exc)) from exc

    def _load_ingestion_checkpoint(
        self,
    ) -> tuple[Path, dict[str, Any]] | None:
        candidates: list[tuple[Path, dict[str, Any]]] = []
        roots = list(self.config.staging_root.iterdir())
        if self._pending_activation_path.is_file():
            try:
                journal = read_json(self._pending_activation_path)
            except StorageError:
                journal = None
            build_id = (
                str(journal.get("build_id") or "") if isinstance(journal, dict) else ""
            )
            pending_root = self.config.generations_root / build_id
            if build_id and pending_root.is_dir():
                roots.append(pending_root)
        for root in sorted(set(roots)):
            checkpoint_path = root / "checkpoint.json"
            if not root.is_dir() or not checkpoint_path.is_file():
                continue
            try:
                checkpoint = read_json(checkpoint_path)
            except StorageError:
                continue
            if not isinstance(checkpoint, dict):
                continue
            if (
                checkpoint.get("schema_version") != INGESTION_CHECKPOINT_VERSION
                or checkpoint.get("project_id") != self.config.project_id
                or checkpoint.get("build_id") != root.name
            ):
                continue
            self._reconcile_checkpoint_progress(root, checkpoint)
            candidates.append((root, checkpoint))
        if not candidates:
            return None
        return max(candidates, key=lambda item: str(item[1].get("updated_at") or ""))

    @staticmethod
    def _reconcile_checkpoint_progress(
        root: Path,
        checkpoint: dict[str, Any],
    ) -> None:
        """Rebuild split progress counters from committed per-source state."""

        sources_root = root / "work" / "sources"
        if not sources_root.is_dir():
            return
        selected_paths = [
            str(relative) for relative in checkpoint.get("selected_source_paths", [])
        ]
        source_formats = {
            str(record.get("source_relative_path") or ""): str(
                record.get("format") or ""
            )
            for record in checkpoint.get("source_inventory", [])
            if isinstance(record, dict)
        }
        extracted_paths = {
            str(relative) for relative in checkpoint.get("extracted_source_paths", [])
        }
        extraction_total = 0
        extraction_completed = 0
        chunking_total = 0
        chunking_completed = 0

        try:
            for relative in selected_paths:
                state_path = sources_root / _source_work_key(relative) / "state.json"
                if not state_path.is_file():
                    continue
                state = read_json(state_path)
                if not isinstance(state, dict):
                    return

                if not state.get("reused"):
                    extraction_stage = str(state.get("extraction_stage") or "")
                    unit_total = int(state.get("total") or 0)
                    next_index = max(
                        0, min(int(state.get("next_index") or 0), unit_total)
                    )
                    source_format = source_formats.get(relative)
                    if source_format == "pdf":
                        page_batch_size = int(state.get("page_batch_size") or 1)
                        batch_total = _pdf_batch_count(
                            unit_total,
                            page_batch_size,
                        )
                        completed_batches = _pdf_batch_count(
                            next_index,
                            page_batch_size,
                        )
                        source_total = (batch_total * 2) + 2
                        if extraction_stage == "pdf_scan":
                            source_completed = completed_batches
                        elif extraction_stage == "pdf_prepare":
                            source_completed = batch_total
                        elif extraction_stage == "pdf_pages":
                            source_completed = batch_total + 1 + completed_batches
                        elif extraction_stage == "finalize":
                            source_completed = (batch_total * 2) + 1
                        elif extraction_stage == "complete":
                            source_completed = source_total - int(
                                relative not in extracted_paths
                            )
                        else:
                            return
                    elif source_format == "epub" and extraction_stage == "epub_items":
                        source_total = unit_total + 2
                        source_completed = next_index + 1
                    elif source_format == "epub" and extraction_stage == "finalize":
                        source_total = unit_total + 2
                        source_completed = unit_total + 1
                    elif source_format == "epub" and extraction_stage == "complete":
                        source_total = unit_total + 2
                        source_completed = source_total - int(
                            relative not in extracted_paths
                        )
                    else:
                        return
                    extraction_total += source_total
                    extraction_completed += min(source_completed, source_total)

                    if "chunking_work_total" in state:
                        source_chunk_total = int(state["chunking_work_total"])
                        chunking_total += source_chunk_total
                        chunking_completed += min(
                            int(state.get("chunked_unit_count") or 0),
                            source_chunk_total,
                        )
        except (OSError, StorageError, TypeError, ValueError):
            # Leave the durable checkpoint untouched. The normal state machine
            # will diagnose malformed per-source state instead of hiding it.
            return

        checkpoint["extraction_work_total"] = extraction_total
        checkpoint["extraction_work_completed"] = extraction_completed
        checkpoint["chunking_work_total"] = chunking_total
        checkpoint["chunking_work_completed"] = chunking_completed

    def _cleanup_invalid_staging_roots(self) -> None:
        """Remove uncommitted staging entries that cannot be resumed safely."""

        for root in sorted(self.config.staging_root.iterdir()):
            checkpoint: Any = None
            if root.is_dir() and not root.is_symlink():
                checkpoint_path = root / "checkpoint.json"
                if checkpoint_path.is_file() and not checkpoint_path.is_symlink():
                    try:
                        checkpoint = read_json(checkpoint_path)
                    except StorageError:
                        checkpoint = None
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("schema_version") == INGESTION_CHECKPOINT_VERSION
                and checkpoint.get("project_id") == self.config.project_id
                and checkpoint.get("build_id") == root.name
            ):
                continue

            diagnostic_id = (
                root.name
                if re.fullmatch(r"[A-Za-z0-9._-]+", root.name)
                else f"invalid-{_source_work_key(root.name)}"
            )
            atomic_write_json(
                self.config.failures_root / f"{diagnostic_id}-orphan.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "generation_id": root.name,
                    "failed_at": _utc_now(),
                    "phase": None,
                    "error": "Removed invalid or checkpointless staging state",
                    "resumable": False,
                },
            )
            if root.is_dir() and not root.is_symlink():
                shutil.rmtree(root)
            else:
                root.unlink(missing_ok=True)
        fsync_directory(self.config.staging_root)

    @staticmethod
    def _ingestion_progress(checkpoint: dict[str, Any]) -> dict[str, Any]:
        phase = str(checkpoint.get("phase") or "source_hashing")
        inventory = checkpoint.get("source_inventory", [])
        selected = checkpoint.get("selected_source_paths", [])
        if phase == "source_hashing":
            completed = len(checkpoint.get("source_digests", {}))
            total = len(inventory)
            unit = "sources"
        elif phase == "extraction":
            completed = int(checkpoint.get("extraction_work_completed") or 0)
            total = int(checkpoint.get("extraction_work_total") or 0)
            if not total:
                completed = len(checkpoint.get("extracted_source_paths", []))
                total = len(selected)
                unit = "sources"
            else:
                unit = "pdf_page_batches_or_epub_sections"
        elif phase == "chunking":
            completed = int(checkpoint.get("chunking_work_completed") or 0)
            total = int(checkpoint.get("chunking_work_total") or 0)
            if not total:
                completed = len(checkpoint.get("chunked_source_paths", []))
                total = len(selected)
                unit = "sources"
            else:
                unit = "extraction_units"
        elif phase == "embedding":
            completed = int(checkpoint.get("embedded_chunk_count") or 0)
            total = int(checkpoint.get("chunk_count") or 0)
            unit = "chunks"
        elif phase in {"dense_indexing", "qdrant_indexing"}:
            completed = int(
                checkpoint.get("dense_indexed_count")
                or checkpoint.get("qdrant_indexed_count")
                or 0
            )
            total = int(checkpoint.get("chunk_count") or 0)
            unit = "chunks"
        elif phase == "source_revalidation":
            completed = len(checkpoint.get("revalidation_digests", {}))
            total = len(inventory)
            unit = "sources"
        else:
            completed = int(phase in {"complete"})
            total = 1
            unit = "phase"
        return {
            "build_id": str(checkpoint["build_id"]),
            "phase": phase,
            "progress": {
                "completed": completed,
                "total": total,
                "unit": unit,
            },
            "parameters": dict(checkpoint.get("parameters") or {}),
            "created_at": checkpoint.get("created_at"),
            "checkpointed_at": checkpoint.get("updated_at"),
        }

    @staticmethod
    def _write_checkpoint(root: Path, checkpoint: dict[str, Any]) -> None:
        checkpoint["updated_at"] = _utc_now()
        atomic_write_json(root / "checkpoint.json", checkpoint)

    def _discard_checkpoint(
        self,
        root: Path,
        checkpoint: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        atomic_write_json(
            self.config.failures_root / f"{checkpoint['build_id']}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "generation_id": checkpoint["build_id"],
                "failed_at": _utc_now(),
                "phase": checkpoint.get("phase"),
                "error": reason,
                "resumable": False,
            },
        )
        shutil.rmtree(root, ignore_errors=True)
        fsync_directory(root.parent)

    @property
    def _pending_activation_path(self) -> Path:
        return self.config.state_root / "pending-activation.json"

    def _discard_invalid_activation_journal(
        self,
        journal: Any,
        *,
        reason: str,
    ) -> None:
        """Remove an unreadable journal without trusting its referenced paths."""

        build_id = (
            str(journal.get("build_id") or "") if isinstance(journal, dict) else ""
        )
        valid_build_id = bool(re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", build_id))
        project_matches = bool(
            isinstance(journal, dict)
            and journal.get("project_id") == self.config.project_id
        )
        if valid_build_id and project_matches:
            staging_root = self.config.staging_root / build_id
            checkpoint_path = staging_root / "checkpoint.json"
            try:
                checkpoint = (
                    read_json(checkpoint_path) if checkpoint_path.is_file() else None
                )
            except StorageError:
                checkpoint = None
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("project_id") == self.config.project_id
                and checkpoint.get("build_id") == build_id
            ):
                self._discard_checkpoint(staging_root, checkpoint, reason=reason)

        diagnostic_id = build_id if valid_build_id else uuid.uuid4().hex[:16]
        atomic_write_json(
            self.config.failures_root
            / f"{diagnostic_id}-invalid-activation-journal.json",
            {
                "schema_version": SCHEMA_VERSION,
                "generation_id": build_id or None,
                "failed_at": _utc_now(),
                "phase": "activation",
                "error": reason,
                "resumable": False,
            },
        )
        self._pending_activation_path.unlink(missing_ok=True)
        fsync_directory(self.config.state_root)

    @staticmethod
    async def _ensure_artifact_lookup(
        root: Path,
        manifest: dict[str, Any],
    ) -> ArtifactLookup:
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ResearchError("Generation manifest has no artifact mapping")
        try:
            chunks_path = root / str(files["chunks"])
            units_path = root / str(files["extracted_units"])
            return await _atomic_to_thread(
                ensure_artifact_lookup,
                chunks_path,
                units_path,
                root / LOOKUP_RELATIVE_PATH,
            )
        except (ArtifactLookupError, KeyError, OSError) as exc:
            raise ResearchError("Generation artifact lookup is unavailable") from exc

    async def _validate_generation_for_activation(
        self,
        root: Path,
        manifest: dict[str, Any],
    ) -> None:
        """Validate portable artifacts and both indexes before pointer changes."""

        lookup = await self._ensure_artifact_lookup(root, manifest)
        if not await _atomic_to_thread(
            generation_artifacts_are_valid,
            root,
            manifest,
            embedding=self.config.settings.embedding_facts,
        ):
            raise ResearchError("Generation portable artifacts failed validation")
        self._loaded_generation = None
        try:
            await self.ultrarag.initialize_bm25(
                root / str(manifest["files"]["chunks"]),
                root / str(manifest["files"]["bm25_index"]),
            )
            chunks_path = root / str(manifest["files"]["chunks"])
            probe = next(iter_jsonl(chunks_path))["contents"]
            bm25_probe = await self.ultrarag.search_bm25(str(probe), 1)
            stored_probe = (
                await _atomic_to_thread(lookup.chunks_by_contents, bm25_probe)
                if bm25_probe
                else {}
            )
            if not bm25_probe or not stored_probe.get(bm25_probe[0]):
                raise RuntimeError("BM25 validation probe returned no stored passage")
            await _atomic_to_thread(
                self._dense_for(manifest).validate_index,
                root / str(manifest["files"]["dense_index"]),
                expected_count=int(manifest["chunk_count"]),
                dimension=self.config.settings.embedding_dimension,
            )
        except Exception as exc:
            raise ResearchError(
                "Generation retrieval indexes failed validation"
            ) from exc

    async def _recover_pending_activation(
        self,
        *,
        expected_identity: str,
        scan: SourceScan,
        exclusions: dict[str, dict[str, str]],
        current: tuple[Path, dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        """Finish a compatible validated activation or discard its orphan."""

        journal_path = self._pending_activation_path
        if not journal_path.exists():
            return None
        try:
            journal = read_json(journal_path)
        except StorageError:
            self._discard_invalid_activation_journal(
                None,
                reason="Pending activation journal contained invalid JSON",
            )
            return None
        build_id = (
            str(journal.get("build_id") or "") if isinstance(journal, dict) else ""
        )
        if (
            not isinstance(journal, dict)
            or journal.get("schema_version") != PENDING_ACTIVATION_VERSION
            or journal.get("project_id") != self.config.project_id
            or not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", build_id)
        ):
            self._discard_invalid_activation_journal(
                journal,
                reason="Pending activation journal fields were invalid",
            )
            return None

        staging_root = self.config.staging_root / build_id
        generation_root = self.config.generations_root / build_id
        current_id = str(current[1]["generation_id"]) if current is not None else None
        if current_id == build_id:
            # The pointer was committed before the process stopped. Clean only
            # activation debris, then let normal ingest validate freshness and
            # repair artifacts if necessary.
            shutil.rmtree(generation_root / "work", ignore_errors=True)
            (generation_root / "checkpoint.json").unlink(missing_ok=True)
            journal_path.unlink(missing_ok=True)
            self._loaded_generation = None
            return None

        root = generation_root if generation_root.is_dir() else staging_root
        checkpoint_path = root / "checkpoint.json"
        manifest_path = root / "manifest.json"
        checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() else None
        manifest = read_json(manifest_path) if manifest_path.is_file() else None
        source_stats = journal.get("revalidation_stats")
        source_paths = {source.source_relative_path for source in scan.selected}
        try:
            source_state_matches = bool(
                isinstance(source_stats, dict)
                and set(source_stats) == source_paths
                and all(
                    _source_stat_identity(source.path)
                    == source_stats[source.source_relative_path]
                    for source in scan.selected
                )
            )
        except OSError:
            source_state_matches = False
        baseline_matches = current_id == journal.get("baseline_generation_id")
        valid = bool(
            isinstance(checkpoint, dict)
            and isinstance(manifest, dict)
            and checkpoint.get("build_id") == build_id
            and checkpoint.get("identity") == expected_identity
            and journal.get("identity") == expected_identity
            and manifest.get("generation_id") == build_id
            and manifest.get("project_id") == self.config.project_id
            and manifest.get("schema_version") == SCHEMA_VERSION
            and manifest.get("artifact_policy_version") == ARTIFACT_POLICY_VERSION
            and _source_inventory(scan) == journal.get("source_inventory")
            and value_fingerprint(exclusions)
            == journal.get("source_exclusion_revision")
            and source_state_matches
            and baseline_matches
        )
        if valid:
            manifest_digests = {
                str(item.get("source_relative_path") or ""): str(
                    item.get("sha256") or ""
                )
                for item in manifest.get("source_files", [])
                if isinstance(item, dict)
            }
            valid = manifest_digests == journal.get("source_digests")
        if valid:
            self._loaded_generation = None
            try:
                await self._validate_generation_for_activation(root, manifest)
            except (ResearchError, TypeError):
                valid = False
        if not valid:
            if isinstance(checkpoint, dict):
                self._discard_checkpoint(
                    root,
                    checkpoint,
                    reason=(
                        "Pending activation was superseded by changed inputs, "
                        "selected generation, or incomplete artifacts"
                    ),
                )
            journal_path.unlink(missing_ok=True)
            self._loaded_generation = None
            return None

        shutil.rmtree(root / "work", ignore_errors=True)
        if root == staging_root:
            os.replace(staging_root, generation_root)
            fsync_directory(self.config.generations_root)
            fsync_directory(self.config.staging_root)
        atomic_write_json(
            self.config.current_path,
            {"schema_version": 1, "generation_id": build_id},
        )
        (generation_root / "checkpoint.json").unlink(missing_ok=True)
        journal_path.unlink(missing_ok=True)
        self._loaded_generation = None
        return {
            "status": "ready",
            "generation_changed": True,
            "activation_recovered": True,
            "generation_id": build_id,
            "generation_root": str(generation_root),
            "message": "Recovered and selected the completed generation.",
        }

    @staticmethod
    def _remove_uncommitted_files(root: Path) -> None:
        """Remove files that are always recreated before their checkpoint commit."""

        for pattern in ("*.tmp", "raw-chunks.jsonl"):
            for path in root.rglob(pattern):
                if path.is_file() and not path.is_symlink():
                    path.unlink(missing_ok=True)
        checkpoint_path = root / "checkpoint.json"
        if checkpoint_path.is_file():
            checkpoint = read_json(checkpoint_path)
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("phase") == "bm25_indexing"
            ):
                shutil.rmtree(root / "indexes" / "bm25", ignore_errors=True)

    def _create_checkpoint(
        self,
        *,
        scan: SourceScan,
        exclusions: dict[str, dict[str, str]],
        exclusion_revision: str,
        baseline_generation_id: str | None,
        chunk_size: int,
        chunk_overlap: int,
        force_recompute: bool,
    ) -> tuple[Path, dict[str, Any]]:
        build_id = _generation_id()
        inventory = _source_inventory(scan)
        identity = _checkpoint_identity(
            project_id=self.config.project_id,
            inventory=inventory,
            exclusion_revision=exclusion_revision,
            baseline_generation_id=baseline_generation_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_recompute=force_recompute,
            retrieval_policy=self.retrieval_policy_fingerprint,
            embedding=self.config.settings.embedding_facts,
        )
        now = _utc_now()
        checkpoint: dict[str, Any] = {
            "schema_version": INGESTION_CHECKPOINT_VERSION,
            "build_id": build_id,
            "project_id": self.config.project_id,
            "identity": identity,
            "identity_policy_version": INGESTION_IDENTITY_POLICY_VERSION,
            "metadata_storage_policy": METADATA_STORAGE_POLICY,
            "phase": "source_hashing",
            "created_at": now,
            "updated_at": now,
            "baseline_generation_id": baseline_generation_id,
            "parameters": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "force_recompute": force_recompute,
            },
            "source_inventory": inventory,
            "selected_source_paths": [
                source.source_relative_path
                for source in scan.selected
                if source.source_relative_path not in exclusions
            ],
            "source_exclusion_revision": exclusion_revision,
            "source_digests": {},
            "extracted_source_paths": [],
            "chunked_source_paths": [],
            "embedded_chunk_count": 0,
            "dense_indexed_count": 0,
            "revalidation_digests": {},
            "revalidation_stats": {},
            "phase_timings_seconds": {},
        }
        root = self.config.staging_root / build_id
        root.mkdir(parents=False, exist_ok=False)
        fsync_directory(self.config.staging_root)
        self._write_checkpoint(root, checkpoint)
        return root, checkpoint

    def _generation_upgrade_reasons(self, manifest: dict[str, Any]) -> list[str]:
        """Return the policy mismatches a generation has, without touching disk.

        This is the part of a status report that depends only on the manifest and
        the current policy constants, so a caller that skipped the staleness check
        can still report whether the generation needs an upgrade.
        """

        retrieval = manifest.get("retrieval", {})
        dense_policy = retrieval.get("dense", {})
        fusion_policy = retrieval.get("fusion", {})
        relevance_policy = retrieval.get("relevance_gates", {})
        reasons: list[str] = []
        if int(manifest.get("schema_version") or 0) != SCHEMA_VERSION:
            reasons.append("generation_schema")
        if (
            int(manifest.get("extraction_policy_version") or 0)
            != EXTRACTION_POLICY_VERSION
        ):
            reasons.append("layout_extraction")
        if int(manifest.get("cleaning_policy_version") or 0) != CLEANING_POLICY_VERSION:
            reasons.append("semantic_cleaning")
        if int(manifest.get("artifact_policy_version") or 0) != ARTIFACT_POLICY_VERSION:
            reasons.append("generation_artifacts")
        if manifest.get("metadata_storage_policy") != METADATA_STORAGE_POLICY:
            reasons.append("metadata_storage")
        if manifest.get("project_id") != self.config.project_id:
            reasons.append("project_identity")
        if (
            dense_policy.get("embedding_model") != self.config.settings.embedding_model
            or dense_policy.get("embedding_model_revision")
            != self.config.settings.embedding_model_revision
            or dense_policy.get("embedding_dimension")
            != self.config.settings.embedding_dimension
        ):
            reasons.append("embedding_model")
        if (
            manifest.get("retrieval_policy_fingerprint")
            != self.retrieval_policy_fingerprint
            or fusion_policy.get("method") != "weighted_reciprocal_rank_fusion"
            or fusion_policy.get("rrf_k") != self.config.settings.rrf_k
            or fusion_policy.get("bm25_weight") != self.config.settings.bm25_weight
            or fusion_policy.get("dense_weight") != self.config.settings.dense_weight
            or relevance_policy.get("dense_minimum_cosine_similarity")
            != self.config.settings.dense_minimum_cosine_similarity
            or relevance_policy.get("bm25_requires_query_token_overlap") is not True
        ):
            reasons.append("retrieval_policy")
        return reasons

    def _status(
        self,
        current: tuple[Path, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Summarize the project, reusing a caller already-parsed pointer.

        `search` already holds the selected generation and its manifest, so it
        passes them in rather than making this read and parse them again.
        """

        try:
            scan = scan_sources(self.config)
        except SourcePolicyError as exc:
            raise ResearchError(str(exc)) from exc
        exclusions = self._source_exclusions()
        exclusion_revision = value_fingerprint(exclusions)
        metadata = self._metadata()
        metadata_revision = value_fingerprint(metadata)
        selected = tuple(
            source
            for source in scan.selected
            if source.source_relative_path not in exclusions
        )
        checkpoint_state = self._load_ingestion_checkpoint()
        ingestion_progress = (
            self._ingestion_progress(checkpoint_state[1])
            if checkpoint_state is not None
            else None
        )
        if current is None:
            current = self._load_current_optional()
        if current is None:
            if ingestion_progress is not None:
                message = (
                    "An ingestion checkpoint is available; call ingest again with "
                    "the same settings to continue it."
                )
            elif scan.selected and not selected:
                message = (
                    "No knowledge-base generation exists and all discovered sources "
                    "are excluded; include at least one source before ingesting."
                )
            else:
                message = "No knowledge-base generation exists; call ingest."
            return {
                "ready": False,
                "stale": bool(selected),
                "project_root": str(self.config.project_root),
                "project_id": self.config.project_id,
                "project_name": self.config.project_name,
                "source_root": str(self.config.source_root),
                "state_root": str(self.config.state_root),
                "runtime_root": (
                    str(self.config.runtime_root)
                    if self.config.runtime_root is not None
                    else None
                ),
                "portable_root": str(self.config.portable_root),
                "model_cache_root": str(self.config.model_cache_root),
                "version": version_block(),
                "ui_launcher": ui_launcher_state(
                    self.config.project_root,
                    self.config.portable_root,
                ),
                "discovered_source_count": len(scan.selected),
                "selected_source_count": len(selected),
                "excluded_source_count": len(exclusions),
                "categories": [],
                "projects": [],
                "excluded_sources": self._exclusion_records(scan, exclusions),
                "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
                "ignored_extensions": scan.ignored_extensions,
                "language": {
                    "corpus": self.config.settings.language_corpus,
                    "languages": list(self.config.settings.corpus_languages),
                    "bm25_stopwords": self.config.settings.bm25_stopwords_language,
                    "warning": self.config.settings.embedding_language_warning,
                },
                "source_exclusion_revision": exclusion_revision,
                "metadata_revision": metadata_revision,
                "generation_metadata_revision": None,
                "metadata_overlay_active": False,
                "metadata_pending_source_paths": sorted(metadata),
                "generation_metadata_snapshot_outdated": False,
                "default_retrieval_method": DEFAULT_RETRIEVAL_METHOD,
                "available_retrieval_methods": [],
                "generation_upgrade_required": False,
                "upgrade_reasons": [],
                "last_build_metrics": None,
                "ingestion_progress": ingestion_progress,
                "message": message,
            }

        generation_root, manifest = current
        manifest_source_files = manifest.get("source_files")
        if not isinstance(manifest_source_files, list):
            manifest_source_files = manifest.get("documents", [])
        current_documents = {
            str(item["source_relative_path"]): item for item in manifest_source_files
        }
        scanned = {item.source_relative_path: item for item in scan.selected}
        added = sorted(set(scanned) - set(current_documents))
        removed = sorted(set(current_documents) - set(scanned))
        modified: list[str] = []
        for relative in sorted(set(scanned) & set(current_documents)):
            source = scanned[relative]
            existing = current_documents[relative]
            if source.size != existing.get("size"):
                modified.append(relative)
                continue
            if source.mtime_ns != existing.get("mtime_ns"):
                recorded_digest = str(existing.get("sha256") or "")
                if not recorded_digest or sha256_file(source.path) != recorded_digest:
                    modified.append(relative)
        metadata_changed = _metadata_snapshot_changed(manifest, metadata)
        stored_exclusion_revision = manifest.get("source_exclusion_revision")
        source_exclusions_changed = (
            stored_exclusion_revision != exclusion_revision
            if stored_exclusion_revision is not None
            else bool(exclusions)
        )
        stale = bool(added or removed or modified or source_exclusions_changed)
        retrieval = manifest.get("retrieval", {})
        available_methods = retrieval.get("available_methods") or ["bm25"]
        hybrid_ready = "hybrid" in available_methods
        upgrade_reasons = self._generation_upgrade_reasons(manifest)
        indexed_source_paths = {
            str(document.get("source_relative_path") or "")
            for document in manifest.get("documents", [])
        }
        metadata_overlay_active = bool(set(metadata) & indexed_source_paths)
        metadata_pending_source_paths = sorted(set(metadata) - indexed_source_paths)
        excluded_document_ids = self._excluded_document_ids(manifest, exclusions)
        exclusion_records = self._exclusion_records(scan, exclusions, manifest)
        if ingestion_progress is not None:
            status_message = (
                "A new generation is in progress; the selected generation remains "
                "searchable. Call ingest again with the same settings to continue."
            )
        elif source_exclusions_changed:
            status_message = (
                "Source exclusions are already enforced by retrieval; run ingest "
                "to rebuild the stored indexes without excluded sources."
            )
        elif upgrade_reasons:
            status_message = (
                "The current generation uses an older extraction or storage schema; "
                "regenerate it with ingest."
            )
        elif added or removed or modified:
            status_message = (
                "Source files differ from the selected generation; call ingest to "
                "index the current source set."
            )
        elif metadata_pending_source_paths:
            status_message = (
                "Reviewed metadata is saved for sources outside the selected "
                "generation; include those sources if needed and ingest to index them."
            )
        elif metadata_changed:
            status_message = (
                "Reviewed metadata differs from the immutable generation snapshot "
                "and is being applied immediately at read time; ingestion is not "
                "required."
            )
        elif hybrid_ready:
            status_message = "Current generation supports hybrid retrieval."
        else:
            status_message = (
                "Current generation is BM25-only; run ingest to build its "
                "project-local dense index."
            )
        effective_documents = _effective_documents(manifest, metadata)
        return {
            "ready": True,
            "stale": stale,
            "project_root": str(self.config.project_root),
            "project_id": self.config.project_id,
            "project_name": self.config.project_name,
            "source_root": str(self.config.source_root),
            "state_root": str(self.config.state_root),
            "runtime_root": (
                str(self.config.runtime_root)
                if self.config.runtime_root is not None
                else None
            ),
            "portable_root": str(self.config.portable_root),
            "model_cache_root": str(self.config.model_cache_root),
            "version": version_block(),
            "ui_launcher": ui_launcher_state(
                self.config.project_root,
                self.config.portable_root,
            ),
            "generation_id": manifest["generation_id"],
            "created_at": manifest["created_at"],
            "discovered_source_count": len(scan.selected),
            "selected_source_count": len(selected),
            "indexed_source_count": manifest["document_count"],
            "searchable_source_count": sum(
                str(document["document_id"]) not in excluded_document_ids
                for document in manifest.get("documents", [])
            ),
            "excluded_source_count": len(exclusions),
            "excluded_sources": exclusion_records,
            "chunk_count": manifest["chunk_count"],
            "categories": _metadata_inventory(
                effective_documents,
                excluded_document_ids,
                field="categories",
                label="category",
            ),
            "projects": _metadata_inventory(
                effective_documents,
                excluded_document_ids,
                field="project",
                label="project",
            ),
            "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
            "ignored_extensions": scan.ignored_extensions,
            "language": {
                "corpus": self.config.settings.language_corpus,
                "languages": list(self.config.settings.corpus_languages),
                "bm25_stopwords": self.config.settings.bm25_stopwords_language,
                "warning": self.config.settings.embedding_language_warning,
            },
            "default_retrieval_method": (
                retrieval.get("default_method", "bm25") if hybrid_ready else "bm25"
            ),
            "available_retrieval_methods": available_methods,
            "hybrid_ready": hybrid_ready,
            "hybrid_upgrade_required": not hybrid_ready,
            "generation_upgrade_required": bool(upgrade_reasons),
            "upgrade_reasons": upgrade_reasons,
            "retrieval": retrieval,
            "last_build_metrics": manifest.get("build_metrics"),
            "ingestion_progress": ingestion_progress,
            "source_exclusion_revision": exclusion_revision,
            "metadata_revision": metadata_revision,
            "generation_metadata_revision": manifest.get("metadata_revision"),
            "metadata_overlay_active": metadata_overlay_active,
            "metadata_pending_source_paths": metadata_pending_source_paths,
            "generation_metadata_snapshot_outdated": metadata_changed,
            "changes": {
                "added": added,
                "removed": removed,
                "modified": modified,
                "metadata_changed": metadata_changed,
                "source_exclusions_changed": source_exclusions_changed,
            },
            "generation_root": str(generation_root),
            "message": status_message,
        }

    def _generation_inventory(
        self, current_generation_id: str | None
    ) -> dict[str, Any]:
        """Describe every retained generation on disk without validating it.

        These are exactly the directories a prune would consider, so `status`
        reports them with their size and file count. A generation whose manifest
        is missing, unreadable, or not JSON is reported with a ``manifest_error``
        instead of raising: the purpose is transparency about what occupies disk,
        and a status call must still answer when one retained generation is
        damaged. This runs only on the read-only ``status`` surface, never on the
        search path, because it walks each generation's files.
        """

        root = self.config.generations_root
        records: list[dict[str, Any]] = []
        if root.is_dir():
            for entry in root.iterdir():
                if entry.is_symlink() or not entry.is_dir():
                    continue
                record: dict[str, Any] = {
                    "generation_id": entry.name,
                    "is_current": entry.name == current_generation_id,
                }
                try:
                    manifest = read_json(entry / "manifest.json")
                except StorageError as exc:
                    record["manifest_error"] = str(exc)
                else:
                    if isinstance(manifest, dict):
                        record.update(
                            created_at=manifest.get("created_at"),
                            chunk_count=manifest.get("chunk_count"),
                            document_count=manifest.get("document_count"),
                            schema_version=manifest.get("schema_version"),
                        )
                    else:
                        record["manifest_error"] = "manifest.json is not a JSON object"
                file_count, size_bytes = directory_statistics(entry)
                record["file_count"] = file_count
                record["size_bytes"] = size_bytes
                records.append(record)
        records.sort(
            key=lambda item: (str(item.get("created_at") or ""), item["generation_id"]),
            reverse=True,
        )
        return {
            "generations": records,
            "retained_generation_count": len(records),
            "retained_generation_bytes": sum(
                int(item["size_bytes"]) for item in records
            ),
        }

    async def status(self) -> dict[str, Any]:
        async with self._operation():
            payload = await asyncio.to_thread(self._status)
            payload.update(
                await asyncio.to_thread(
                    self._generation_inventory,
                    payload.get("generation_id"),
                )
            )
            return payload

    async def set_source_inclusion(
        self,
        source_path: str | None = None,
        *,
        source_id: str | None = None,
        included: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Include or exclude one source without changing the source file."""

        async with self._operation():
            try:
                scan = scan_sources(self.config)
                current = self._load_current_optional()
                selected = self._resolve_source_selector(
                    source_id=source_id,
                    source_path=source_path,
                    scan=scan,
                    current=current,
                )
                relative = str(selected["source_relative_path"])

                exclusions = self._source_exclusions()
                previous = exclusions.get(relative)
                if included:
                    changed = previous is not None
                    exclusions.pop(relative, None)
                else:
                    normalized_reason = (reason or "").strip()
                    if not normalized_reason:
                        raise SourcePolicyError(
                            "A non-empty reason is required when excluding a source"
                        )
                    if (
                        not selected["exists"]
                        and not selected["indexed_in_current_generation"]
                        and previous is None
                    ):
                        raise SourcePolicyError(
                            "Only an existing or currently indexed PDF or EPUB can "
                            "be excluded: "
                            f"{relative}"
                        )
                    changed = (
                        previous is None or previous.get("reason") != normalized_reason
                    )
                    if changed:
                        exclusions[relative] = {
                            "reason": normalized_reason,
                            "excluded_at": _utc_now(),
                        }

                if changed:
                    write_source_exclusions(
                        self.config.source_exclusions_path,
                        exclusions,
                    )

                indexed = bool(selected["indexed_in_current_generation"])
            except (StorageError, SourcePolicyError, ValueError) as exc:
                raise ResearchError(str(exc)) from exc

            effective_immediately = not included or indexed
            if not changed:
                message = (
                    "Source is already included."
                    if included
                    else "Source is already excluded with this reason."
                )
            elif included and not indexed:
                message = (
                    "Source inclusion saved. Run ingest before it can appear in "
                    "search because it is absent from the current generation."
                )
            elif included:
                message = (
                    "Source inclusion saved and restored for current retrieval. "
                    "Run ingest to record the change in a new generation."
                )
            else:
                message = (
                    "Source exclusion saved and enforced for current retrieval. "
                    "The original file was not changed. Run ingest to rebuild the "
                    "indexes without it."
                )
            return {
                "status": "changed" if changed else "unchanged",
                "source_id": selected["source_id"],
                "source_relative_path": relative,
                "source_path": selected["source_path"],
                "included": included,
                "reason": None if included else exclusions[relative]["reason"],
                "source_file_changed": False,
                "effective_immediately": effective_immediately,
                "generation_rebuild_recommended": changed,
                "message": message,
            }

    @staticmethod
    def _source_artifact_root(staging_root: Path, relative_path: str) -> Path:
        return staging_root / "work" / "sources" / _source_work_key(relative_path)

    @staticmethod
    def _save_vector_batch(
        path: Path,
        vectors: np.ndarray[Any, np.dtype[np.float32]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _assemble_vector_batches(
        path: Path,
        batches_root: Path,
        total: int,
        embedding_batch_size: int,
        dimension: int,
    ) -> None:
        """Assemble portable vectors without allocating a second full matrix."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        vectors = None
        try:
            vectors = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=(total, dimension),
            )
            for offset in range(0, total, embedding_batch_size):
                batch = np.load(
                    batches_root / f"{offset:012d}.npy",
                    allow_pickle=False,
                    mmap_mode="r",
                )
                vectors[offset : offset + len(batch)] = batch
            vectors.flush()
            del vectors
            vectors = None
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            fsync_directory(path.parent)
        finally:
            if vectors is not None:
                del vectors
            temporary.unlink(missing_ok=True)

    def _in_progress_result(
        self,
        checkpoint: dict[str, Any],
        *,
        message: str | None = None,
    ) -> dict[str, Any]:
        progress = self._ingestion_progress(checkpoint)
        return {
            "status": "in_progress",
            "generation_changed": False,
            **progress,
            "next_action": "call_ingest_again",
            "message": message
            or "Ingestion checkpoint saved; call ingest again with the same settings.",
        }

    @staticmethod
    def _add_phase_time(
        checkpoint: dict[str, Any],
        phase: str,
        elapsed: float,
    ) -> None:
        timings = checkpoint.setdefault("phase_timings_seconds", {})
        timings[phase] = float(timings.get(phase) or 0.0) + elapsed

    async def _advance_ingestion(
        self,
        *,
        staging_root: Path,
        checkpoint: dict[str, Any],
        scan: SourceScan,
        exclusions: dict[str, dict[str, str]],
        current: tuple[Path, dict[str, Any]] | None,
        deadline: float,
    ) -> dict[str, Any]:
        parameters = checkpoint["parameters"]
        chunk_size = int(parameters["chunk_size"])
        chunk_overlap = int(parameters["chunk_overlap"])
        force_recompute = bool(parameters["force_recompute"])
        sources_by_path: dict[str, SourceFile] = {
            source.source_relative_path: source for source in scan.selected
        }
        selected_paths = [str(item) for item in checkpoint["selected_source_paths"]]
        snapshot = None
        if not force_recompute:
            snapshot = load_reuse_snapshot(
                current,
                schema_version=SCHEMA_VERSION,
                extraction_policy_version=EXTRACTION_POLICY_VERSION,
                cleaning_policy_version=CLEANING_POLICY_VERSION,
                artifact_policy_version=ARTIFACT_POLICY_VERSION,
                project_id=self.config.project_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                load_units=False,
                load_chunks=False,
                load_vectors=False,
                embedding=self.config.settings.embedding_facts,
            )
        units_cache: dict[str, list[dict[str, Any]]] = {}
        portable_vectors: np.ndarray[Any, np.dtype[np.float32]] | None = None

        def budget_expired() -> bool:
            return time.perf_counter() >= deadline

        def source_records() -> list[dict[str, Any]]:
            digests = checkpoint["source_digests"]
            return [
                {
                    **record,
                    "sha256": str(digests[record["source_relative_path"]]),
                    "included": record["source_relative_path"] not in exclusions,
                }
                for record in checkpoint["source_inventory"]
            ]

        def revalidation_stats_match(*, require_complete: bool = False) -> bool:
            recorded = checkpoint.get("revalidation_stats") or {}
            if not isinstance(recorded, dict):
                return False
            if require_complete and set(recorded) != set(sources_by_path):
                return False
            try:
                return all(
                    _source_stat_identity(sources_by_path[relative].path) == identity
                    for relative, identity in recorded.items()
                    if relative in sources_by_path
                ) and set(recorded).issubset(sources_by_path)
            except OSError:
                return False

        if checkpoint.get("revalidation_stats") and not revalidation_stats_match():
            raise _SourceChangedDuringIngest(
                "A source changed after its final ingestion hash"
            )

        while True:
            phase = str(checkpoint["phase"])
            if phase == "source_hashing":
                digests = checkpoint["source_digests"]
                pending = [
                    record
                    for record in checkpoint["source_inventory"]
                    if record["source_relative_path"] not in digests
                ]
                if pending:
                    record = pending[0]
                    source = sources_by_path[str(record["source_relative_path"])]
                    started = time.perf_counter()
                    try:
                        digest = await _atomic_to_thread(sha256_file, source.path)
                    except OSError as exc:
                        raise _SourceChangedDuringIngest(
                            "A source became unavailable during hashing"
                        ) from exc
                    self._add_phase_time(
                        checkpoint,
                        "source_hashing",
                        time.perf_counter() - started,
                    )
                    digests[source.source_relative_path] = digest
                    self._write_checkpoint(staging_root, checkpoint)
                    if budget_expired():
                        return self._in_progress_result(checkpoint)
                    continue

                records = source_records()
                if snapshot is not None and source_set_matches(
                    snapshot,
                    records,
                    exclusion_revision=str(checkpoint["source_exclusion_revision"]),
                    retrieval_policy_fingerprint=(self.retrieval_policy_fingerprint),
                    metadata_storage_policy=METADATA_STORAGE_POLICY,
                ):
                    manifest = snapshot.manifest
                    try:
                        await self._validate_generation_for_activation(
                            snapshot.root,
                            manifest,
                        )
                    except ResearchError as exc:
                        checkpoint["reuse_rejection_reason"] = str(exc)
                        snapshot = None
                    else:
                        self._loaded_generation = str(manifest["generation_id"])
                        shutil.rmtree(staging_root, ignore_errors=True)
                        return {
                            "status": "unchanged",
                            "generation_changed": False,
                            "generation_id": manifest["generation_id"],
                            "generation_root": str(snapshot.root),
                            "source_file_count": len(scan.selected),
                            "excluded_source_count": len(exclusions),
                            "document_count": manifest["document_count"],
                            "extraction_unit_count": manifest["extraction_unit_count"],
                            "chunk_count": manifest["chunk_count"],
                            "reused_document_count": manifest["document_count"],
                            "rebuilt_document_count": 0,
                            "reused_chunk_count": manifest["chunk_count"],
                            "rebuilt_chunk_count": 0,
                            "reused_vector_count": manifest["chunk_count"],
                            "created_vector_count": 0,
                            "phase_timings_seconds": checkpoint[
                                "phase_timings_seconds"
                            ],
                            "message": (
                                "Inputs match the selected generation; no build "
                                "was needed."
                            ),
                        }
                checkpoint["phase"] = "extraction"
                self._write_checkpoint(staging_root, checkpoint)
                continue

            if phase == "extraction":
                if snapshot is not None and not snapshot.units_by_document:
                    snapshot = load_reuse_snapshot(
                        current,
                        schema_version=SCHEMA_VERSION,
                        extraction_policy_version=EXTRACTION_POLICY_VERSION,
                        cleaning_policy_version=CLEANING_POLICY_VERSION,
                        artifact_policy_version=ARTIFACT_POLICY_VERSION,
                        project_id=self.config.project_id,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        load_units=True,
                        load_chunks=True,
                        load_vectors=False,
                        embedding=self.config.settings.embedding_facts,
                    )
                completed = set(checkpoint["extracted_source_paths"])
                relative = next(
                    (path for path in selected_paths if path not in completed),
                    None,
                )
                if relative is None:
                    checkpoint["phase"] = "chunking"
                    self._write_checkpoint(staging_root, checkpoint)
                    continue
                source = sources_by_path[relative]
                artifact_root = self._source_artifact_root(staging_root, relative)
                artifact_root.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                state_path = artifact_root / "state.json"
                state = read_json(state_path) if state_path.exists() else None
                if state is None:
                    previous_source = (
                        snapshot.source_files.get(relative)
                        if snapshot is not None
                        else None
                    )
                    previous_document = (
                        snapshot.documents.get(relative)
                        if snapshot is not None
                        else None
                    )
                    document_id = str(
                        (previous_document or {}).get("document_id") or ""
                    )
                    reusable = bool(
                        snapshot is not None
                        and previous_source is not None
                        and previous_document is not None
                        and previous_source.get("sha256")
                        == checkpoint["source_digests"][relative]
                        and snapshot.manifest.get("metadata_storage_policy")
                        == METADATA_STORAGE_POLICY
                        and document_id in snapshot.units_by_document
                        and document_id in snapshot.chunks_by_document
                    )
                    if (
                        reusable
                        and snapshot is not None
                        and previous_document is not None
                    ):
                        document = dict(previous_document)
                        document["size"] = source.size
                        document["mtime_ns"] = source.mtime_ns
                        units = [
                            dict(item)
                            for item in snapshot.units_by_document[document_id]
                        ]
                        atomic_write_json(artifact_root / "document.json", document)
                        atomic_write_jsonl(artifact_root / "units.jsonl", units)
                        state = {
                            "reused": True,
                            "extraction_stage": "complete",
                            "discarded_empty_chunks": 0,
                            "discarded_symbol_only_chunks": 0,
                            "discarded_corrupt_chunks": 0,
                        }
                        atomic_write_json(state_path, state)
                        checkpoint["extracted_source_paths"].append(relative)
                        checkpoint["reused_document_count"] = (
                            int(checkpoint.get("reused_document_count") or 0) + 1
                        )
                    elif source.extension == ".pdf":
                        total = await _atomic_to_thread(pdf_page_count, source)
                        state = {
                            "reused": False,
                            "extraction_stage": "pdf_scan",
                            "next_index": 0,
                            "total": total,
                            "page_batch_size": self.config.settings.pdf_page_batch_size,
                            "rejected_units": [],
                            "empty_units": 0,
                            "removed_repeated_margin_blocks": 0,
                            "discarded_empty_chunks": 0,
                            "discarded_symbol_only_chunks": 0,
                            "discarded_corrupt_chunks": 0,
                        }
                        checkpoint["extraction_work_total"] = (
                            int(checkpoint.get("extraction_work_total") or 0)
                            + (
                                _pdf_batch_count(
                                    total, self.config.settings.pdf_page_batch_size
                                )
                                * 2
                            )
                            + 2
                        )
                        atomic_write_json(state_path, state)
                    else:
                        document, total = await _atomic_to_thread(
                            prepare_epub_extraction,
                            source,
                            checkpoint["source_digests"][relative],
                        )
                        atomic_write_json(artifact_root / "document.json", document)
                        state = {
                            "reused": False,
                            "extraction_stage": "epub_items",
                            "next_index": 0,
                            "total": total,
                            "rejected_units": [],
                            "empty_units": 0,
                            "discarded_empty_chunks": 0,
                            "discarded_symbol_only_chunks": 0,
                            "discarded_corrupt_chunks": 0,
                        }
                        checkpoint["extraction_work_total"] = (
                            int(checkpoint.get("extraction_work_total") or 0)
                            + total
                            + 2
                        )
                        checkpoint["extraction_work_completed"] = (
                            int(checkpoint.get("extraction_work_completed") or 0) + 1
                        )
                        atomic_write_json(state_path, state)
                    self._add_phase_time(
                        checkpoint,
                        "extraction",
                        time.perf_counter() - started,
                    )
                    self._write_checkpoint(staging_root, checkpoint)
                    if budget_expired():
                        return self._in_progress_result(checkpoint)
                    continue

                extraction_stage = str(state["extraction_stage"])
                if extraction_stage == "complete":
                    document = read_json(artifact_root / "document.json")
                    units = read_jsonl(artifact_root / "units.jsonl")
                    if (
                        not isinstance(document, dict)
                        or not units
                        or any(
                            unit.get("document_id") != document.get("document_id")
                            or unit.get("source_id") != document.get("source_id")
                            for unit in units
                        )
                    ):
                        raise ResearchError(
                            f"Completed extraction artifacts are invalid: {relative}"
                        )
                    checkpoint["extracted_source_paths"].append(relative)
                    count_field = (
                        "reused_document_count"
                        if state.get("reused")
                        else "rebuilt_document_count"
                    )
                    checkpoint[count_field] = int(checkpoint.get(count_field) or 0) + 1
                    if not state.get("reused"):
                        checkpoint["extraction_work_completed"] = (
                            int(checkpoint.get("extraction_work_completed") or 0) + 1
                        )
                elif extraction_stage == "pdf_scan":
                    index = int(state["next_index"])
                    total = int(state["total"])
                    page_batch_size = int(state.get("page_batch_size") or 1)
                    if index < total:
                        page_scans = await _atomic_to_thread(
                            scan_pdf_pages,
                            source,
                            index,
                            min(page_batch_size, total - index),
                        )
                        expected_indices = list(
                            range(index, min(index + page_batch_size, total))
                        )
                        if [int(scan["page_index"]) for scan in page_scans] != (
                            expected_indices
                        ):
                            raise ResearchError(
                                f"PDF scan batch was incomplete: {relative}"
                            )
                        for page_scan in page_scans:
                            page_index = int(page_scan["page_index"])
                            atomic_write_json(
                                artifact_root / "page-scans" / f"{page_index:08d}.json",
                                page_scan,
                                fsync_parent=False,
                            )
                        fsync_directories([artifact_root / "page-scans"])
                        state["next_index"] = expected_indices[-1] + 1
                        checkpoint["extraction_work_completed"] = (
                            int(checkpoint.get("extraction_work_completed") or 0) + 1
                        )
                    else:
                        state["extraction_stage"] = "pdf_prepare"
                        state["next_index"] = 0
                    atomic_write_json(state_path, state)
                elif extraction_stage == "pdf_prepare":
                    page_scans = [
                        read_json(artifact_root / "page-scans" / f"{index:08d}.json")
                        for index in range(int(state["total"]))
                    ]
                    document, repeated = await _atomic_to_thread(
                        prepare_scanned_pdf,
                        source,
                        checkpoint["source_digests"][relative],
                        page_scans,
                    )
                    atomic_write_json(artifact_root / "document.json", document)
                    state["repeated_margins"] = repeated
                    state["extraction_stage"] = "pdf_pages"
                    state["next_index"] = 0
                    checkpoint["extraction_work_completed"] = (
                        int(checkpoint.get("extraction_work_completed") or 0) + 1
                    )
                    atomic_write_json(state_path, state)
                elif extraction_stage in {"pdf_pages", "epub_items"}:
                    index = int(state["next_index"])
                    total = int(state["total"])
                    if index >= total:
                        state["extraction_stage"] = "finalize"
                        atomic_write_json(state_path, state)
                        continue
                    document = read_json(artifact_root / "document.json")
                    if extraction_stage == "pdf_pages":
                        page_batch_size = int(state.get("page_batch_size") or 1)
                        end_index = min(index + page_batch_size, total)
                        page_scans = [
                            read_json(
                                artifact_root / "page-scans" / f"{page_index:08d}.json"
                            )
                            for page_index in range(index, end_index)
                        ]
                        page_batches = await _atomic_to_thread(
                            extract_scanned_pdf_pages,
                            source,
                            document,
                            page_scans,
                            list(state.get("repeated_margins") or []),
                        )
                        if [page_index for page_index, *_rest in page_batches] != list(
                            range(index, end_index)
                        ):
                            raise ResearchError(
                                f"PDF extraction batch was incomplete: {relative}"
                            )
                    else:
                        batch, empty = await _atomic_to_thread(
                            extract_epub_spine_item,
                            source,
                            document,
                            index,
                        )
                        page_batches = [(index, batch, empty, 0)]
                    for page_index, batch, empty, removed in page_batches:
                        retained: list[dict[str, Any]] = []
                        for unit in batch:
                            reasons = text_health_reasons(
                                str(unit.get("contents") or "")
                            )
                            if reasons:
                                state["rejected_units"].append(
                                    {
                                        "unit_id": str(unit.get("id") or ""),
                                        "locator": dict(unit.get("locator") or {}),
                                        "reasons": reasons,
                                    }
                                )
                            else:
                                retained.append(unit)
                        atomic_write_jsonl(
                            artifact_root / "unit-batches" / f"{page_index:08d}.jsonl",
                            retained,
                            fsync_parent=False,
                        )
                        state["empty_units"] = int(state.get("empty_units") or 0) + int(
                            empty
                        )
                        if removed:
                            state["removed_repeated_margin_blocks"] = (
                                int(state.get("removed_repeated_margin_blocks") or 0)
                                + removed
                            )
                    state["next_index"] = (
                        end_index if extraction_stage == "pdf_pages" else index + 1
                    )
                    checkpoint["extraction_work_completed"] = (
                        int(checkpoint.get("extraction_work_completed") or 0) + 1
                    )
                    atomic_write_json(state_path, state, fsync_parent=False)
                    fsync_directories([artifact_root / "unit-batches", artifact_root])
                elif extraction_stage == "finalize":
                    units = []
                    for index in range(int(state["total"])):
                        units.extend(
                            read_jsonl(
                                artifact_root / "unit-batches" / f"{index:08d}.jsonl"
                            )
                        )
                    if not units:
                        raise ExtractionError(
                            "Source produced no readable English-oriented text after "
                            f"unhealthy extraction units were excluded: {source.path}"
                        )
                    document = read_json(artifact_root / "document.json")
                    rejected = list(state.get("rejected_units") or [])
                    document["extracted_units"] = len(units)
                    document["empty_units"] = int(state.get("empty_units") or 0)
                    document["excluded_corrupt_unit_count"] = len(rejected)
                    document["excluded_corrupt_units"] = rejected
                    if source.extension == ".pdf":
                        document["removed_repeated_margin_blocks"] = int(
                            state.get("removed_repeated_margin_blocks") or 0
                        )
                    if rejected:
                        document["metadata_warnings"] = list(
                            dict.fromkeys(
                                [
                                    *document.get("metadata_warnings", []),
                                    "corrupt_extraction_units_excluded",
                                ]
                            )
                        )
                    atomic_write_json(artifact_root / "document.json", document)
                    atomic_write_jsonl(artifact_root / "units.jsonl", units)
                    state["extraction_stage"] = "complete"
                    atomic_write_json(state_path, state)
                    checkpoint["extracted_source_paths"].append(relative)
                    checkpoint["rebuilt_document_count"] = (
                        int(checkpoint.get("rebuilt_document_count") or 0) + 1
                    )
                    checkpoint["extraction_work_completed"] = (
                        int(checkpoint.get("extraction_work_completed") or 0) + 1
                    )
                else:
                    raise ResearchError(
                        f"Unsupported staged extraction phase: {extraction_stage}"
                    )
                self._add_phase_time(
                    checkpoint,
                    "extraction",
                    time.perf_counter() - started,
                )
                self._write_checkpoint(staging_root, checkpoint)
                if budget_expired():
                    return self._in_progress_result(checkpoint)
                continue

            if phase == "chunking":
                if snapshot is not None and not snapshot.chunks_by_document:
                    snapshot = load_reuse_snapshot(
                        current,
                        schema_version=SCHEMA_VERSION,
                        extraction_policy_version=EXTRACTION_POLICY_VERSION,
                        cleaning_policy_version=CLEANING_POLICY_VERSION,
                        artifact_policy_version=ARTIFACT_POLICY_VERSION,
                        project_id=self.config.project_id,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        load_units=True,
                        load_chunks=True,
                        load_vectors=False,
                        embedding=self.config.settings.embedding_facts,
                    )
                completed = set(checkpoint["chunked_source_paths"])
                relative = next(
                    (path for path in selected_paths if path not in completed),
                    None,
                )
                if relative is None:
                    checkpoint["phase"] = "assembly"
                    self._write_checkpoint(staging_root, checkpoint)
                    continue
                artifact_root = self._source_artifact_root(staging_root, relative)
                state = read_json(artifact_root / "state.json")
                document = read_json(artifact_root / "document.json")
                units = units_cache.get(relative)
                if units is None:
                    units = read_jsonl(artifact_root / "units.jsonl")
                    units_cache[relative] = units
                started = time.perf_counter()
                if bool(state.get("reused")) and snapshot is not None:
                    chunks = [
                        dict(item)
                        for item in snapshot.chunks_by_document[
                            str(document["document_id"])
                        ]
                    ]
                    discarded_empty = 0
                    discarded_symbol_only = 0
                    discarded_corrupt = 0
                    checkpoint["reused_chunk_count"] = int(
                        checkpoint.get("reused_chunk_count") or 0
                    ) + len(chunks)
                    state["chunking_stage"] = "complete"
                else:
                    chunked_count = int(state.get("chunked_unit_count") or 0)
                    if "chunking_work_total" not in state:
                        state["chunking_work_total"] = len(units)
                        checkpoint["chunking_work_total"] = int(
                            checkpoint.get("chunking_work_total") or 0
                        ) + len(units)
                        atomic_write_json(artifact_root / "state.json", state)
                        self._write_checkpoint(staging_root, checkpoint)
                        if budget_expired():
                            return self._in_progress_result(checkpoint)
                        continue
                    if chunked_count < len(units):
                        batch_units = units[
                            chunked_count : chunked_count
                            + self.config.settings.chunk_batch_units
                        ]
                        requested_ids = {str(unit["id"]) for unit in batch_units}
                        chunking_root = artifact_root / "chunking"
                        input_path = chunking_root / "unit.jsonl"
                        working_path = artifact_root / "raw-chunks.jsonl"
                        # A handoff file is rewritten before every call and
                        # deleted afterwards, so it needs visibility, not
                        # durability.
                        write_handoff_jsonl(input_path, batch_units)
                        working_path.unlink(missing_ok=True)
                        await self.ultrarag.chunk(
                            input_path,
                            working_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        raw_by_unit: defaultdict[str, list[dict[str, Any]]] = (
                            defaultdict(list)
                        )
                        for raw in read_jsonl(working_path):
                            raw_by_unit[str(raw.get("doc_id") or "")].append(raw)
                        working_path.unlink(missing_ok=True)
                        input_path.unlink(missing_ok=True)
                        unknown = [
                            unit_id
                            for unit_id in raw_by_unit
                            if unit_id not in requested_ids
                        ]
                        if unknown:
                            raise ResearchError(
                                "UltraRAG returned an unknown extraction unit: "
                                f"{unknown[0]}"
                            )
                        # Every unit keeps its own durable output file and its own
                        # index, written in unit order, so the ordinal assignment
                        # in `_enrich_chunks` and the resume boundary per unit are
                        # the same as when each unit had its own call.
                        for position, unit in enumerate(batch_units):
                            atomic_write_jsonl(
                                chunking_root / f"{chunked_count + position:08d}.jsonl",
                                raw_by_unit.get(str(unit["id"]), []),
                                fsync_parent=False,
                            )
                        state["chunked_unit_count"] = chunked_count + len(batch_units)
                        checkpoint["chunking_work_completed"] = int(
                            checkpoint.get("chunking_work_completed") or 0
                        ) + len(batch_units)
                        atomic_write_json(
                            artifact_root / "state.json",
                            state,
                            fsync_parent=False,
                        )
                        # The whole batch is on disk before the checkpoint that
                        # claims it is complete.
                        fsync_directories([chunking_root, artifact_root])
                        self._add_phase_time(
                            checkpoint,
                            "chunking",
                            time.perf_counter() - started,
                        )
                        self._write_checkpoint(staging_root, checkpoint)
                        if budget_expired():
                            return self._in_progress_result(checkpoint)
                        continue
                    raw_chunks: list[dict[str, Any]] = []
                    for index in range(len(units)):
                        raw_chunks.extend(
                            read_jsonl(
                                artifact_root / "chunking" / f"{index:08d}.jsonl"
                            )
                        )
                    (
                        chunks,
                        discarded_empty,
                        discarded_symbol_only,
                        discarded_corrupt,
                    ) = _enrich_chunks(raw_chunks, units, [document])
                    audited = await _atomic_to_thread(
                        _record_embedding_token_counts,
                        chunks,
                        self.dense.embedding_token_counts,
                        maximum_tokens=(self.config.settings.embedding_maximum_tokens),
                    )
                    checkpoint["dense_token_audit"] = (
                        "counted" if audited else "unavailable"
                    )
                    checkpoint["rebuilt_chunk_count"] = int(
                        checkpoint.get("rebuilt_chunk_count") or 0
                    ) + len(chunks)
                    state["chunking_stage"] = "complete"
                atomic_write_jsonl(artifact_root / "chunks.jsonl", chunks)
                state["discarded_empty_chunks"] = discarded_empty
                state["discarded_symbol_only_chunks"] = discarded_symbol_only
                state["discarded_corrupt_chunks"] = discarded_corrupt
                atomic_write_json(artifact_root / "state.json", state)
                self._add_phase_time(
                    checkpoint,
                    "chunking",
                    time.perf_counter() - started,
                )
                checkpoint["chunked_source_paths"].append(relative)
                self._write_checkpoint(staging_root, checkpoint)
                if budget_expired():
                    return self._in_progress_result(checkpoint)
                continue

            if phase == "assembly":
                started = time.perf_counter()
                documents: list[dict[str, Any]] = []
                discarded_empty_chunks = 0
                discarded_symbol_only_chunks = 0
                discarded_corrupt_chunks = 0
                for relative in selected_paths:
                    artifact_root = self._source_artifact_root(staging_root, relative)
                    document = read_json(artifact_root / "document.json")
                    state = read_json(artifact_root / "state.json")
                    if not isinstance(document, dict) or not isinstance(state, dict):
                        raise ResearchError("Invalid staged source artifacts")
                    documents.append(document)
                    discarded_empty_chunks += int(
                        state.get("discarded_empty_chunks") or 0
                    )
                    discarded_symbol_only_chunks += int(
                        state.get("discarded_symbol_only_chunks") or 0
                    )
                    discarded_corrupt_chunks += int(
                        state.get("discarded_corrupt_chunks") or 0
                    )
                extracted_path = staging_root / "corpus" / "extracted-units.jsonl"
                chunks_path = staging_root / "chunks" / "chunks.jsonl"

                record_counts = {"units": 0, "chunks": 0}

                def staged_records(
                    filename: str,
                    count_key: str,
                    counts: dict[str, int],
                ) -> Iterator[dict[str, Any]]:
                    for source_relative_path in selected_paths:
                        path = (
                            self._source_artifact_root(
                                staging_root,
                                source_relative_path,
                            )
                            / filename
                        )
                        for record in iter_jsonl(path):
                            counts[count_key] += 1
                            yield record

                atomic_write_jsonl(
                    extracted_path,
                    staged_records("units.jsonl", "units", record_counts),
                )
                atomic_write_jsonl(
                    chunks_path,
                    staged_records("chunks.jsonl", "chunks", record_counts),
                )
                batches_root = staging_root / "work" / "chunk-batches"
                offset = 0
                batch: list[dict[str, Any]] = []
                for chunk in iter_jsonl(chunks_path):
                    batch.append(chunk)
                    if len(batch) < self.config.settings.embedding_batch_size:
                        continue
                    atomic_write_jsonl(
                        batches_root / f"{offset:012d}.jsonl",
                        batch,
                        fsync_parent=False,
                    )
                    offset += len(batch)
                    batch = []
                if batch:
                    atomic_write_jsonl(
                        batches_root / f"{offset:012d}.jsonl",
                        batch,
                        fsync_parent=False,
                    )
                if batches_root.is_dir():
                    # One directory fsync covers every batch before embedding
                    # resumes from any of them.
                    fsync_directories([batches_root])
                checkpoint["document_count"] = len(documents)
                checkpoint["extraction_unit_count"] = record_counts["units"]
                checkpoint["chunk_count"] = record_counts["chunks"]
                checkpoint["discarded_empty_chunk_count"] = discarded_empty_chunks
                checkpoint["discarded_symbol_only_chunk_count"] = (
                    discarded_symbol_only_chunks
                )
                checkpoint["discarded_corrupt_chunk_count"] = discarded_corrupt_chunks
                checkpoint["excluded_corrupt_unit_count"] = sum(
                    int(document.get("excluded_corrupt_unit_count") or 0)
                    for document in documents
                )
                self._add_phase_time(
                    checkpoint,
                    "assembly",
                    time.perf_counter() - started,
                )
                checkpoint["phase"] = "embedding"
                self._write_checkpoint(staging_root, checkpoint)
                if budget_expired():
                    return self._in_progress_result(checkpoint)
                continue

            if phase == "embedding":
                if snapshot is not None and not snapshot.vectors_by_text:
                    snapshot = load_reuse_snapshot(
                        current,
                        schema_version=SCHEMA_VERSION,
                        extraction_policy_version=EXTRACTION_POLICY_VERSION,
                        cleaning_policy_version=CLEANING_POLICY_VERSION,
                        artifact_policy_version=ARTIFACT_POLICY_VERSION,
                        project_id=self.config.project_id,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        load_units=False,
                        load_chunks=True,
                        load_vectors=True,
                        embedding=self.config.settings.embedding_facts,
                    )
                offset = int(checkpoint.get("embedded_chunk_count") or 0)
                total = int(checkpoint["chunk_count"])
                if offset >= total:
                    checkpoint["phase"] = "vector_assembly"
                    self._write_checkpoint(staging_root, checkpoint)
                    continue
                batch = read_jsonl(
                    staging_root / "work" / "chunk-batches" / f"{offset:012d}.jsonl"
                )
                vectors = np.empty(
                    (len(batch), self.config.settings.embedding_dimension),
                    dtype=np.float32,
                )
                missing_positions: list[int] = []
                missing_texts: list[str] = []
                batch_texts = [_chunk_text(chunk) for chunk in batch]
                reusable_vectors = (
                    await _atomic_to_thread(
                        snapshot.vectors_for_texts,
                        batch_texts,
                    )
                    if snapshot is not None and not force_recompute
                    else {}
                )
                reused_count = 0
                for index, text in enumerate(batch_texts):
                    reusable_vector = reusable_vectors.get(text)
                    if reusable_vector is None:
                        missing_positions.append(index)
                        missing_texts.append(text)
                    else:
                        vectors[index] = reusable_vector
                        reused_count += 1
                started = time.perf_counter()
                if missing_texts:
                    created = await _atomic_to_thread(
                        self.dense.embed_texts,
                        missing_texts,
                    )
                    if created.shape != (
                        len(missing_positions),
                        self.config.settings.embedding_dimension,
                    ):
                        raise ResearchError(
                            "Dense backend returned an invalid embedding matrix"
                        )
                    for position, vector in zip(
                        missing_positions,
                        created,
                        strict=True,
                    ):
                        vectors[position] = vector
                batch_path = (
                    staging_root / "work" / "vector-batches" / f"{offset:012d}.npy"
                )
                self._save_vector_batch(batch_path, vectors)
                self._add_phase_time(
                    checkpoint,
                    "embedding",
                    time.perf_counter() - started,
                )
                checkpoint["embedded_chunk_count"] = offset + len(batch)
                checkpoint["reused_vector_count"] = (
                    int(checkpoint.get("reused_vector_count") or 0) + reused_count
                )
                checkpoint["created_vector_count"] = int(
                    checkpoint.get("created_vector_count") or 0
                ) + len(missing_positions)
                self._write_checkpoint(staging_root, checkpoint)
                if budget_expired():
                    return self._in_progress_result(checkpoint)
                continue

            if phase == "vector_assembly":
                started = time.perf_counter()
                total = int(checkpoint["chunk_count"])
                vectors_path = staging_root / "portable" / "embeddings.npy"
                await _atomic_to_thread(
                    self._assemble_vector_batches,
                    vectors_path,
                    staging_root / "work" / "vector-batches",
                    total,
                    self.config.settings.embedding_batch_size,
                    self.config.settings.embedding_dimension,
                )
                self._add_phase_time(
                    checkpoint,
                    "vector_assembly",
                    time.perf_counter() - started,
                )
                checkpoint["phase"] = "bm25_indexing"
                self._write_checkpoint(staging_root, checkpoint)
                if budget_expired():
                    return self._in_progress_result(checkpoint)
                continue

            if phase == "bm25_indexing":
                bm25_index_path = staging_root / "indexes" / "bm25"
                shutil.rmtree(bm25_index_path, ignore_errors=True)
                started = time.perf_counter()
                self._loaded_generation = None
                await self.ultrarag.build_bm25(
                    staging_root / "chunks" / "chunks.jsonl",
                    bm25_index_path,
                    language=self.config.settings.bm25_stopwords_language,
                )
                self._add_phase_time(
                    checkpoint,
                    "bm25_indexing",
                    time.perf_counter() - started,
                )
                checkpoint["phase"] = "dense_indexing"
                checkpoint["dense_indexed_count"] = 0
                self._write_checkpoint(staging_root, checkpoint)
                if budget_expired():
                    return self._in_progress_result(checkpoint)
                continue

            if phase in {"dense_indexing", "qdrant_indexing"}:
                if portable_vectors is None:
                    portable_vectors = np.load(
                        staging_root / "portable" / "embeddings.npy",
                        allow_pickle=False,
                        mmap_mode="r",
                    )
                total = int(checkpoint["chunk_count"])
                backend_name = str(checkpoint.get("dense_backend") or "")
                if backend_name not in DENSE_INDEX_PATHS:
                    # A checkpoint resumed from before the backend was recorded
                    # restarts this phase cleanly with the selected backend.
                    backend_name = self._select_build_dense_backend(total)
                    checkpoint["dense_backend"] = backend_name
                    checkpoint["dense_index_relative"] = DENSE_INDEX_PATHS[backend_name]
                    checkpoint["dense_indexed_count"] = 0
                    self._write_checkpoint(staging_root, checkpoint)
                backend = self._dense_backends[backend_name]
                dense_index_path = staging_root / DENSE_INDEX_PATHS[backend_name]
                offset = int(checkpoint.get("dense_indexed_count") or 0)
                if offset == 0:
                    shutil.rmtree(dense_index_path, ignore_errors=True)
                    await _atomic_to_thread(
                        backend.initialize_index,
                        dense_index_path,
                        self.config.settings.embedding_dimension,
                    )
                if offset < total:
                    batch = read_jsonl(
                        staging_root / "work" / "chunk-batches" / f"{offset:012d}.jsonl"
                    )
                    started = time.perf_counter()
                    await _atomic_to_thread(
                        backend.upload_index_batch,
                        batch,
                        dense_index_path,
                        portable_vectors[offset : offset + len(batch)],
                        offset=offset,
                    )
                    self._add_phase_time(
                        checkpoint,
                        "dense_indexing",
                        time.perf_counter() - started,
                    )
                    checkpoint["dense_indexed_count"] = offset + len(batch)
                    self._write_checkpoint(staging_root, checkpoint)
                    if budget_expired():
                        return self._in_progress_result(checkpoint)
                    continue
                dense_metadata = await _atomic_to_thread(
                    backend.finalize_index,
                    dense_index_path,
                    expected_count=total,
                    dimension=self.config.settings.embedding_dimension,
                )
                checkpoint["dense_metadata"] = dense_metadata
                checkpoint["phase"] = "source_revalidation"
                self._write_checkpoint(staging_root, checkpoint)
                continue

            if phase == "source_revalidation":
                revalidated = checkpoint["revalidation_digests"]
                pending = [
                    record
                    for record in checkpoint["source_inventory"]
                    if record["source_relative_path"] not in revalidated
                ]
                if pending:
                    record = pending[0]
                    source = sources_by_path[str(record["source_relative_path"])]
                    started = time.perf_counter()
                    try:
                        digest, stat_identity = await _atomic_to_thread(
                            _hash_with_stable_stat,
                            source.path,
                        )
                    except OSError as exc:
                        raise _SourceChangedDuringIngest(
                            "A source became unavailable during final validation"
                        ) from exc
                    revalidated[source.source_relative_path] = digest
                    checkpoint["revalidation_stats"][source.source_relative_path] = (
                        stat_identity
                    )
                    self._add_phase_time(
                        checkpoint,
                        "source_revalidation",
                        time.perf_counter() - started,
                    )
                    self._write_checkpoint(staging_root, checkpoint)
                    if budget_expired():
                        return self._in_progress_result(checkpoint)
                    continue
                if revalidated != checkpoint["source_digests"]:
                    raise _SourceChangedDuringIngest(
                        "Source bytes changed while ingestion was in progress"
                    )
                try:
                    final_scan = scan_sources(self.config)
                except SourcePolicyError as exc:
                    raise _SourceChangedDuringIngest(str(exc)) from exc
                if _source_inventory(final_scan) != checkpoint["source_inventory"]:
                    raise _SourceChangedDuringIngest(
                        "The source collection changed while ingestion was in progress"
                    )
                if value_fingerprint(self._source_exclusions()) != checkpoint.get(
                    "source_exclusion_revision"
                ):
                    raise _SourceChangedDuringIngest(
                        "Source exclusions changed while ingestion was in progress"
                    )
                selected_now = self._load_current_optional()
                selected_generation_id = (
                    str(selected_now[1]["generation_id"])
                    if selected_now is not None
                    else None
                )
                if selected_generation_id != checkpoint.get("baseline_generation_id"):
                    raise _SourceChangedDuringIngest(
                        "The selected generation changed while ingestion was in progress"
                    )
                checkpoint["phase"] = "finalizing"
                self._write_checkpoint(staging_root, checkpoint)
                continue

            if phase == "finalizing":
                if not revalidation_stats_match(require_complete=True):
                    raise _SourceChangedDuringIngest(
                        "A source changed after its final ingestion hash"
                    )
                documents = [
                    read_json(
                        self._source_artifact_root(staging_root, relative)
                        / "document.json"
                    )
                    for relative in selected_paths
                ]
                content_kinds: Counter[str] = Counter()
                withheld_chunk_count = 0
                withheld_chunk_reasons: Counter[str] = Counter()
                audited_chunk_count = 0
                dense_truncated_chunk_count = 0
                maximum_embedding_token_count = 0
                for item in iter_jsonl(staging_root / "chunks" / "chunks.jsonl"):
                    content_kinds[str(item.get("content_kind") or "prose")] += 1
                    reasons = text_corruption_reasons(str(item.get("contents") or ""))
                    if reasons:
                        withheld_chunk_count += 1
                        withheld_chunk_reasons.update(reasons)
                    token_count = item.get("embedding_token_count")
                    if isinstance(token_count, int) and not isinstance(
                        token_count, bool
                    ):
                        audited_chunk_count += 1
                        maximum_embedding_token_count = max(
                            maximum_embedding_token_count, token_count
                        )
                        if item.get("dense_truncated"):
                            dense_truncated_chunk_count += 1
                content_kind_counts = dict(sorted(content_kinds.items()))
                extraction_unit_count = int(checkpoint["extraction_unit_count"])
                chunk_count = int(checkpoint["chunk_count"])
                phase_timings = {
                    key: round(float(value), 6)
                    for key, value in checkpoint["phase_timings_seconds"].items()
                }
                build_metrics = {
                    "forced": force_recompute,
                    "resumed": bool(checkpoint.get("resume_count")),
                    "reused_document_count": int(
                        checkpoint.get("reused_document_count") or 0
                    ),
                    "rebuilt_document_count": int(
                        checkpoint.get("rebuilt_document_count") or 0
                    ),
                    "reused_chunk_count": int(
                        checkpoint.get("reused_chunk_count") or 0
                    ),
                    "rebuilt_chunk_count": int(
                        checkpoint.get("rebuilt_chunk_count") or 0
                    ),
                    "reused_vector_count": int(
                        checkpoint.get("reused_vector_count") or 0
                    ),
                    "created_vector_count": int(
                        checkpoint.get("created_vector_count") or 0
                    ),
                    "excluded_corrupt_unit_count": int(
                        checkpoint.get("excluded_corrupt_unit_count") or 0
                    ),
                    "discarded_symbol_only_chunk_count": int(
                        checkpoint.get("discarded_symbol_only_chunk_count") or 0
                    ),
                    "discarded_corrupt_chunk_count": int(
                        checkpoint.get("discarded_corrupt_chunk_count") or 0
                    ),
                    "withheld_chunk_count": withheld_chunk_count,
                    "withheld_chunk_reasons": dict(
                        sorted(withheld_chunk_reasons.items())
                    ),
                    "dense_token_audit": (
                        "counted"
                        if audited_chunk_count == chunk_count
                        else ("partial" if audited_chunk_count else "unavailable")
                    ),
                    "dense_audited_chunk_count": audited_chunk_count,
                    "dense_truncated_chunk_count": dense_truncated_chunk_count,
                    "embedding_maximum_tokens": self.config.settings.embedding_maximum_tokens,
                    "maximum_embedding_token_count": maximum_embedding_token_count,
                    "phase_timings_seconds": phase_timings,
                }
                records = source_records()
                dense_metadata = dict(checkpoint["dense_metadata"])
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "extraction_policy_version": EXTRACTION_POLICY_VERSION,
                    "cleaning_policy_version": CLEANING_POLICY_VERSION,
                    "artifact_policy_version": ARTIFACT_POLICY_VERSION,
                    "generation_id": checkpoint["build_id"],
                    "project_id": self.config.project_id,
                    "project_name": self.config.project_name,
                    "created_at": _utc_now(),
                    "source_directory": self.config.source_root.relative_to(
                        self.config.project_root
                    ).as_posix(),
                    "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
                    "ignored_extensions": scan.ignored_extensions,
                    # Reviewed metadata is a portable read-time overlay. This
                    # revision is observational and never fingerprints derived
                    # index contents or checkpoint compatibility.
                    "metadata_revision": value_fingerprint(self._metadata()),
                    "metadata_storage_policy": METADATA_STORAGE_POLICY,
                    "source_exclusion_revision": checkpoint[
                        "source_exclusion_revision"
                    ],
                    "retrieval_policy_fingerprint": (self.retrieval_policy_fingerprint),
                    "source_file_count": len(scan.selected),
                    "excluded_source_count": len(exclusions),
                    "document_count": len(documents),
                    "extraction_unit_count": extraction_unit_count,
                    "chunk_count": chunk_count,
                    "content_kind_counts": content_kind_counts,
                    "discarded_empty_chunk_count": int(
                        checkpoint.get("discarded_empty_chunk_count") or 0
                    ),
                    "discarded_symbol_only_chunk_count": int(
                        checkpoint.get("discarded_symbol_only_chunk_count") or 0
                    ),
                    "discarded_corrupt_chunk_count": int(
                        checkpoint.get("discarded_corrupt_chunk_count") or 0
                    ),
                    "excluded_corrupt_unit_count": int(
                        checkpoint.get("excluded_corrupt_unit_count") or 0
                    ),
                    "chunking": {
                        "backend": "UltraRAG token chunker",
                        "tokenizer": "gpt2",
                        "unit": "tokens",
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    },
                    "retrieval": {
                        "default_method": DEFAULT_RETRIEVAL_METHOD,
                        "available_methods": sorted(RETRIEVAL_METHODS),
                        "bm25": {
                            "backend": "UltraRAG BM25",
                            "language": "en",
                            "tokenizer": "default",
                        },
                        "dense": dense_metadata,
                        "fusion": {
                            "method": "weighted_reciprocal_rank_fusion",
                            "rrf_k": self.config.settings.rrf_k,
                            "bm25_weight": self.config.settings.bm25_weight,
                            "dense_weight": self.config.settings.dense_weight,
                            "minimum_candidates": self.config.settings.minimum_candidates,
                            "maximum_candidates": self.config.settings.maximum_candidates,
                        },
                        "relevance_gates": {
                            "bm25_requires_query_token_overlap": True,
                            "dense_minimum_cosine_similarity": (
                                self.config.settings.dense_minimum_cosine_similarity
                            ),
                        },
                        "reranker": {
                            "optional": True,
                            "runtime": "FastEmbed ONNX Runtime (CPU)",
                            "model": self.config.reranker_model,
                            "model_revision": _reranker_revision(
                                self.config.reranker_model
                            ),
                            "maximum_candidates": self.config.settings.rerank_max_candidates,
                        },
                    },
                    "documents": documents,
                    "source_files": records,
                    "excluded_sources": [
                        {
                            "source_id": source.source_id,
                            "source_relative_path": source.source_relative_path,
                            "source_path": source.project_relative_path,
                            **exclusions[source.source_relative_path],
                        }
                        for source in scan.selected
                        if source.source_relative_path in exclusions
                    ],
                    "files": {
                        "extracted_units": "corpus/extracted-units.jsonl",
                        "chunks": "chunks/chunks.jsonl",
                        "bm25_index": "indexes/bm25",
                        "dense_index": str(
                            checkpoint.get("dense_index_relative")
                            or DENSE_INDEX_PATHS[QDRANT_BACKEND_NAME]
                        ),
                        "portable_embeddings": "portable/embeddings.npy",
                    },
                    "build_metrics": build_metrics,
                }
                generation_root = self.config.generations_root / str(
                    checkpoint["build_id"]
                )
                if not revalidation_stats_match(require_complete=True):
                    raise _SourceChangedDuringIngest(
                        "A source changed while final artifacts were assembled"
                    )
                atomic_write_json(staging_root / "manifest.json", manifest)
                await self._validate_generation_for_activation(
                    staging_root,
                    manifest,
                )
                if not revalidation_stats_match(require_complete=True):
                    raise _SourceChangedDuringIngest(
                        "A source changed while final indexes were validated"
                    )
                atomic_write_json(
                    self._pending_activation_path,
                    {
                        "schema_version": PENDING_ACTIVATION_VERSION,
                        "project_id": self.config.project_id,
                        "build_id": checkpoint["build_id"],
                        "identity": checkpoint["identity"],
                        "baseline_generation_id": checkpoint.get(
                            "baseline_generation_id"
                        ),
                        "source_inventory": checkpoint["source_inventory"],
                        "source_digests": checkpoint["source_digests"],
                        "revalidation_stats": checkpoint["revalidation_stats"],
                        "source_exclusion_revision": checkpoint[
                            "source_exclusion_revision"
                        ],
                        "created_at": _utc_now(),
                    },
                )
                shutil.rmtree(staging_root / "work", ignore_errors=True)
                os.replace(staging_root, generation_root)
                fsync_directory(self.config.generations_root)
                fsync_directory(self.config.staging_root)
                atomic_write_json(
                    self.config.current_path,
                    {
                        "schema_version": 1,
                        "generation_id": checkpoint["build_id"],
                    },
                )
                (generation_root / "checkpoint.json").unlink()
                self._pending_activation_path.unlink(missing_ok=True)
                self._loaded_generation = None
                return {
                    "status": "ready",
                    "generation_changed": True,
                    "generation_id": checkpoint["build_id"],
                    "generation_root": str(generation_root),
                    "source_file_count": len(scan.selected),
                    "excluded_source_count": len(exclusions),
                    "excluded_sources": [
                        {
                            "source_id": source.source_id,
                            "source_relative_path": source.source_relative_path,
                            "source_path": source.project_relative_path,
                            **exclusions[source.source_relative_path],
                        }
                        for source in scan.selected
                        if source.source_relative_path in exclusions
                    ],
                    "document_count": len(documents),
                    "pdf_count": sum(item["format"] == "pdf" for item in documents),
                    "epub_count": sum(item["format"] == "epub" for item in documents),
                    "extraction_unit_count": extraction_unit_count,
                    "chunk_count": chunk_count,
                    "content_kind_counts": content_kind_counts,
                    "discarded_empty_chunk_count": int(
                        checkpoint.get("discarded_empty_chunk_count") or 0
                    ),
                    "discarded_symbol_only_chunk_count": int(
                        checkpoint.get("discarded_symbol_only_chunk_count") or 0
                    ),
                    "discarded_corrupt_chunk_count": int(
                        checkpoint.get("discarded_corrupt_chunk_count") or 0
                    ),
                    "excluded_corrupt_unit_count": int(
                        checkpoint.get("excluded_corrupt_unit_count") or 0
                    ),
                    "ignored_extensions": scan.ignored_extensions,
                    "empty_units": sum(int(item["empty_units"]) for item in documents),
                    "default_retrieval_method": DEFAULT_RETRIEVAL_METHOD,
                    "available_retrieval_methods": sorted(RETRIEVAL_METHODS),
                    "embedding_model": self.config.settings.embedding_model,
                    "embedding_model_revision": self.config.settings.embedding_model_revision,
                    **build_metrics,
                }

            raise ResearchError(f"Unsupported ingestion checkpoint phase: {phase}")

    async def ingest(
        self,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        force_recompute: bool = False,
        work_budget_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Create or refresh the generation that search reads.

        An omitted parameter comes from the merged settings, which is where the
        chunking values live; a caller that passes one gets it checked here.
        """

        settings = self.config.settings
        if chunk_size is None:
            chunk_size = settings.chunk_size
        if chunk_overlap is None:
            chunk_overlap = settings.chunk_overlap
        if work_budget_seconds is None:
            work_budget_seconds = settings.work_budget_seconds
        size_setting = SETTINGS_BY_KEY["chunking.size"]
        if not size_setting.minimum <= chunk_size <= size_setting.maximum:
            raise ResearchError(
                "chunk_size must be between "
                f"{size_setting.minimum:g} and {size_setting.maximum:g} GPT-2 tokens"
            )
        if not 0 <= chunk_overlap < chunk_size:
            raise ResearchError(
                "chunk_overlap must be non-negative and below chunk_size"
            )
        budget_setting = SETTINGS_BY_KEY["ingestion.work_budget_seconds"]
        # An engine caller may ask for 0 to checkpoint at the next boundary; the
        # configured setting keeps the higher floor, since a whole build with a
        # budget below it would only ever complete one unit per call.
        if not 0 <= work_budget_seconds <= budget_setting.maximum:
            raise ResearchError(
                f"work_budget_seconds must be between 0 and {budget_setting.maximum:g}"
            )

        async with self._operation():
            deadline = time.perf_counter() + work_budget_seconds
            self._cleanup_invalid_staging_roots()
            try:
                scan = scan_sources(self.config)
            except SourcePolicyError as exc:
                raise ResearchError(str(exc)) from exc
            if not scan.selected:
                raise ResearchError(
                    f"No PDF or EPUB sources found beneath {self.config.source_root}"
                )
            exclusions = self._source_exclusions()
            selected = tuple(
                source
                for source in scan.selected
                if source.source_relative_path not in exclusions
            )
            if not selected:
                raise ResearchError(
                    "All discovered PDF and EPUB sources are excluded; include at "
                    "least one source before ingesting"
                )
            current = self._load_current_optional()
            self._sync_source_catalog(
                scan,
                current,
                exclusions=exclusions,
            )
            baseline_generation_id = (
                str(current[1]["generation_id"]) if current is not None else None
            )
            exclusion_revision = value_fingerprint(exclusions)
            inventory = _source_inventory(scan)
            identity = _checkpoint_identity(
                project_id=self.config.project_id,
                inventory=inventory,
                exclusion_revision=exclusion_revision,
                baseline_generation_id=baseline_generation_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                force_recompute=force_recompute,
                retrieval_policy=self.retrieval_policy_fingerprint,
                embedding=self.config.settings.embedding_facts,
            )
            recovered = await self._recover_pending_activation(
                expected_identity=identity,
                scan=scan,
                exclusions=exclusions,
                current=current,
            )
            if recovered is not None:
                return recovered
            existing = self._load_ingestion_checkpoint()
            if existing is not None and existing[1].get("identity") != identity:
                self._discard_checkpoint(
                    existing[0],
                    existing[1],
                    reason="Ingestion inputs or parameters changed; checkpoint superseded",
                )
                existing = None
            if existing is None:
                staging_root, checkpoint = self._create_checkpoint(
                    scan=scan,
                    exclusions=exclusions,
                    exclusion_revision=exclusion_revision,
                    baseline_generation_id=baseline_generation_id,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    force_recompute=force_recompute,
                )
            else:
                staging_root, checkpoint = existing
                self._remove_uncommitted_files(staging_root)
                checkpoint["resume_count"] = (
                    int(checkpoint.get("resume_count") or 0) + 1
                )
                self._write_checkpoint(staging_root, checkpoint)

            try:
                return await self._advance_ingestion(
                    staging_root=staging_root,
                    checkpoint=checkpoint,
                    scan=scan,
                    exclusions=exclusions,
                    current=current,
                    deadline=deadline,
                )
            except _SourceChangedDuringIngest as exc:
                self._discard_checkpoint(
                    staging_root,
                    checkpoint,
                    reason=str(exc),
                )
                fresh_scan = scan_sources(self.config)
                fresh_exclusions = self._source_exclusions()
                if not fresh_scan.selected:
                    raise ResearchError(
                        "All PDF and EPUB sources disappeared during ingestion"
                    ) from exc
                if not any(
                    source.source_relative_path not in fresh_exclusions
                    for source in fresh_scan.selected
                ):
                    raise ResearchError(
                        "All discovered sources became excluded during ingestion"
                    ) from exc
                self._sync_source_catalog(
                    fresh_scan,
                    current,
                    exclusions=fresh_exclusions,
                )
                fresh_root, fresh = self._create_checkpoint(
                    scan=fresh_scan,
                    exclusions=fresh_exclusions,
                    exclusion_revision=value_fingerprint(fresh_exclusions),
                    baseline_generation_id=baseline_generation_id,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    force_recompute=force_recompute,
                )
                del fresh_root
                return self._in_progress_result(
                    fresh,
                    message=(
                        "Source bytes changed during ingestion; the incompatible "
                        "checkpoint was replaced. Call ingest again."
                    ),
                )
            except asyncio.CancelledError:
                checkpoint["last_interruption"] = "cancelled"
                self._remove_uncommitted_files(staging_root)
                self._write_checkpoint(staging_root, checkpoint)
                raise
            except TimeoutError:
                checkpoint["last_interruption"] = "timeout"
                self._remove_uncommitted_files(staging_root)
                self._write_checkpoint(staging_root, checkpoint)
                raise
            except Exception as exc:
                pending = (
                    read_json(self._pending_activation_path)
                    if self._pending_activation_path.is_file()
                    else None
                )
                if not (
                    isinstance(pending, dict)
                    and pending.get("build_id") == checkpoint.get("build_id")
                ):
                    self._discard_checkpoint(
                        staging_root,
                        checkpoint,
                        reason=str(exc),
                    )
                raise

    async def _ensure_loaded(
        self,
        generation_root: Path,
        manifest: dict[str, Any],
    ) -> None:
        generation_id = str(manifest["generation_id"])
        if self._loaded_generation == generation_id:
            return
        await self.ultrarag.initialize_bm25(
            generation_root / manifest["files"]["chunks"],
            generation_root / manifest["files"]["bm25_index"],
        )
        self._loaded_generation = generation_id

    @staticmethod
    def _matches_filters(
        chunk: dict[str, Any],
        documents_by_id: dict[str, dict[str, Any]],
        *,
        categories_any: set[str],
        keywords: set[str],
        projects_any: set[str],
        document_filter: set[str],
        excluded_document_ids: set[str],
    ) -> bool:
        document = _document_for_chunk(chunk, documents_by_id)
        return not (
            chunk["document_id"] in excluded_document_ids
            or (document_filter and chunk["document_id"] not in document_filter)
            or not _document_matches_metadata(
                document,
                keywords=keywords,
                categories_any=categories_any,
                projects_any=projects_any,
            )
        )

    async def _bm25_ranking(
        self,
        query: str,
        lookup: ArtifactLookup,
        total_chunk_count: int,
        documents_by_id: dict[str, dict[str, Any]],
        limit: int,
        *,
        categories_any: set[str],
        projects_any: set[str],
        keywords: set[str],
        document_filter: set[str],
        excluded_document_ids: set[str],
        withheld: dict[str, dict[str, Any]],
    ) -> tuple[list[str], dict[str, int], dict[str, dict[str, Any]]]:
        if limit <= 0:
            return (
                [],
                {
                    "no_query_token_overlap": 0,
                    "extraction_artifact": 0,
                    "corrupt_text": 0,
                },
                {},
            )
        filtered = bool(
            categories_any
            or keywords
            or projects_any
            or document_filter
            or excluded_document_ids
        )
        requested = min(
            total_chunk_count,
            max(limit * 4, self.config.settings.minimum_candidates),
        )
        query_tokens = _content_tokens(query)
        by_contents: dict[str, list[dict[str, Any]]] = {}
        loaded_contents: set[str] = set()
        # A candidate's verdict depends only on the chunk, so a repeat in a
        # widening iteration reuses it instead of rescanning the text.
        flags_cache: dict[str, int] = {}
        tokens_cache: dict[str, frozenset[str]] = {}
        while True:
            passages = await self.ultrarag.search_bm25(query, requested)
            missing_contents = [
                passage for passage in passages if passage not in loaded_contents
            ]
            if missing_contents:
                by_contents.update(
                    await asyncio.to_thread(
                        lookup.chunks_by_contents,
                        missing_contents,
                    )
                )
                loaded_contents.update(missing_contents)

            ranking: list[str] = []
            used: set[str] = set()
            resolved: dict[str, dict[str, Any]] = {}
            rejected = {
                "no_query_token_overlap": 0,
                "extraction_artifact": 0,
                "corrupt_text": 0,
            }
            for passage in passages:
                candidates = by_contents.get(passage)
                if not candidates:
                    raise ResearchError(
                        "UltraRAG returned a passage absent from the current chunk store"
                    )
                chunk = next(
                    (
                        item
                        for item in candidates
                        if str(item["chunk_id"]) not in used
                        and self._matches_filters(
                            item,
                            documents_by_id,
                            categories_any=categories_any,
                            keywords=keywords,
                            projects_any=projects_any,
                            document_filter=document_filter,
                            excluded_document_ids=excluded_document_ids,
                        )
                    ),
                    None,
                )
                if chunk is None:
                    continue
                chunk_id = str(chunk["chunk_id"])
                used.add(chunk_id)
                resolved[chunk_id] = chunk
                flags = flags_cache.get(chunk_id)
                if flags is None:
                    flags = _candidate_flags(chunk)
                    flags_cache[chunk_id] = flags
                if flags & CHUNK_FLAG_EXTRACTION_ARTIFACT:
                    rejected["extraction_artifact"] += 1
                    continue
                if flags & CHUNK_FLAG_CORRUPT_TEXT:
                    rejected["corrupt_text"] += 1
                    # Reason codes are recomputed only here, because the
                    # response discloses them and they are not stored.
                    _record_withheld(
                        withheld,
                        chunk,
                        text_corruption_reasons(_chunk_text(chunk)),
                        limit=self.config.settings.maximum_withheld_examples,
                    )
                    continue
                tokens = tokens_cache.get(chunk_id)
                if tokens is None:
                    tokens = frozenset(_content_tokens(_chunk_text(chunk)))
                    tokens_cache[chunk_id] = tokens
                if not query_tokens.intersection(tokens):
                    rejected["no_query_token_overlap"] += 1
                    continue
                ranking.append(chunk_id)
                if len(ranking) == limit:
                    break

            if (
                len(ranking) == limit
                or not filtered
                or requested >= total_chunk_count
                or len(passages) < requested
            ):
                return ranking, rejected, resolved
            requested = min(total_chunk_count, requested * 2)

    @staticmethod
    def _fuse_rankings(
        bm25_ranking: list[str],
        dense_ranking: list[str],
        *,
        rrf_k: int,
        bm25_weight: float,
        dense_weight: float,
        maximum_candidates: int,
    ) -> tuple[list[str], dict[str, float]]:
        scores: defaultdict[str, float] = defaultdict(float)
        component_ranks = (
            {chunk_id: rank for rank, chunk_id in enumerate(ranking, 1)}
            for ranking in (bm25_ranking, dense_ranking)
        )
        bm25_ranks, dense_ranks = component_ranks
        for chunk_id, rank in bm25_ranks.items():
            scores[chunk_id] += bm25_weight / (rrf_k + rank)
        for chunk_id, rank in dense_ranks.items():
            scores[chunk_id] += dense_weight / (rrf_k + rank)
        absent = maximum_candidates + 1
        ordered = sorted(
            scores,
            key=lambda chunk_id: (
                -scores[chunk_id],
                min(
                    bm25_ranks.get(chunk_id, absent),
                    dense_ranks.get(chunk_id, absent),
                ),
                chunk_id,
            ),
        )
        return ordered, dict(scores)

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        categories_any: list[str] | None = None,
        projects_any: list[str] | None = None,
        keywords: list[str] | None = None,
        source_ids: list[str] | None = None,
        exclude_source_ids: list[str] | None = None,
        retrieval_method: str = DEFAULT_RETRIEVAL_METHOD,
        rerank: bool = False,
        rerank_model: str | None = None,
        include_staleness: bool = True,
    ) -> dict[str, Any]:
        """Retrieve evidence.

        The public MCP tool defaults ``rerank`` to true because it is the
        largest measured quality gain (``MEASUREMENTS.md``); this lower-level API
        keeps the neutral default so internal callers and tests state what they
        want. When the reranker model cannot be loaded the search still succeeds
        with the unranked candidate order and reports ``rerank_fallback``.

        ``rerank_model`` names a reranker for this call alone, so one process can
        measure several models against the same generation. It defaults to the
        engine's configured model, which is what every tool call uses.
        """
        query = query.strip()
        if not query:
            raise ResearchError("query must not be empty")
        if not 1 <= top_k <= 50:
            raise ResearchError("top_k must be between 1 and 50")
        retrieval_method = retrieval_method.casefold().strip()
        if retrieval_method not in RETRIEVAL_METHODS:
            raise ResearchError("retrieval_method must be one of: bm25, dense, hybrid")
        if rerank_model is not None and not rerank:
            raise ResearchError("rerank_model requires rerank=True")
        applied_reranker = rerank_model or self.config.reranker_model
        try:
            applied_reranker_revision = _reranker_revision(applied_reranker)
        except ValueError as exc:
            raise ResearchError(str(exc)) from exc

        async with self._operation():
            current = self._load_current_optional()
            if current is None:
                raise ResearchError("No knowledge base exists; call ingest first")
            generation_root, manifest = current
            lookup = await self._ensure_artifact_lookup(generation_root, manifest)
            total_chunk_count = await asyncio.to_thread(lookup.chunk_count)
            if not total_chunk_count:
                raise ResearchError("The current generation has no chunks")
            metadata = self._metadata()
            documents_by_id = _effective_documents(manifest, metadata)
            chunks_by_id: dict[str, dict[str, Any]] = {}
            exclusions = self._source_exclusions()
            excluded_document_ids = self._excluded_document_ids(
                manifest,
                exclusions,
            )

            retrieval = manifest.get("retrieval", {})
            available_methods = set(retrieval.get("available_methods") or ["bm25"])
            if retrieval_method not in available_methods:
                raise ResearchError(
                    f"Current generation does not support {retrieval_method!r}; "
                    f"available methods: {', '.join(sorted(available_methods))}. "
                    "Run ingest to build a hybrid generation."
                )

            requested_source_ids = _requested_ids(source_ids)
            requested_exclude_source_ids = _requested_ids(exclude_source_ids)
            category_any_filter = _normalized_filter(categories_any)
            project_any_filter = _normalized_filter(projects_any)
            keyword_filter = _normalized_filter(keywords)
            source_include_document_ids, unknown_source_ids = (
                self._document_ids_for_source_ids(manifest, requested_source_ids)
            )
            source_exclude_document_ids, unknown_exclude_source_ids = (
                self._document_ids_for_source_ids(
                    manifest,
                    requested_exclude_source_ids,
                )
            )
            if requested_source_ids and not source_include_document_ids:
                raise ResearchError(
                    "source_ids matched no document in the current generation: "
                    f"{', '.join(unknown_source_ids)}. Use list_sources for current "
                    "IDs; a renamed or moved source receives a new source_id."
                )
            # Reviewed exclusions always win over a search-level exclusion, and a
            # search-level include can never re-admit an excluded source.
            excluded_document_ids = excluded_document_ids | source_exclude_document_ids
            document_filter = set(source_include_document_ids)

            dense_document_filter: set[str] | None = None
            metadata_filter_active = bool(
                category_any_filter or project_any_filter or keyword_filter
            )
            if metadata_filter_active:
                dense_document_filter = {
                    document_id
                    for document_id, document in documents_by_id.items()
                    if _document_matches_metadata(
                        document,
                        keywords=keyword_filter,
                        categories_any=category_any_filter,
                        projects_any=project_any_filter,
                    )
                }
            if document_filter:
                dense_document_filter = (
                    document_filter
                    if dense_document_filter is None
                    else dense_document_filter & document_filter
                )
            active_document_ids = {
                document_id
                for document_id, document in documents_by_id.items()
                if document_id not in excluded_document_ids
                and _document_matches_metadata(
                    document,
                    keywords=keyword_filter,
                    categories_any=category_any_filter,
                    projects_any=project_any_filter,
                )
                and (not document_filter or document_id in document_filter)
            }
            active_chunk_count = await asyncio.to_thread(
                lookup.chunk_count,
                (
                    active_document_ids
                    if metadata_filter_active
                    or document_filter
                    or excluded_document_ids
                    else None
                ),
            )
            candidate_depth = min(
                active_chunk_count,
                self.config.settings.maximum_candidates,
                max(self.config.settings.minimum_candidates, top_k * 4),
            )

            use_bm25 = retrieval_method in {"bm25", "hybrid"}
            use_dense = retrieval_method in {"dense", "hybrid"}
            if use_bm25:
                await self._ensure_loaded(generation_root, manifest)

            bm25_ranking: list[str] = []
            dense_hits: list[DenseSearchHit] = []
            withheld: dict[str, dict[str, Any]] = {}
            bm25_rejected = {
                "no_query_token_overlap": 0,
                "extraction_artifact": 0,
                "corrupt_text": 0,
            }

            async def search_dense() -> list[DenseSearchHit]:
                if candidate_depth == 0 or dense_document_filter == set():
                    return []
                return await asyncio.to_thread(
                    self._dense_for(manifest).search,
                    generation_root / manifest["files"]["dense_index"],
                    query,
                    candidate_depth,
                    # Keep Qdrant payloads lean. Translate current reviewed
                    # metadata filters to document IDs at query time so edits
                    # remain exact without rebuilding the dense index.
                    document_ids=sorted(dense_document_filter or []),
                    excluded_document_ids=sorted(excluded_document_ids),
                )

            if use_bm25 and use_dense:
                bm25_result, dense_hits = await asyncio.gather(
                    self._bm25_ranking(
                        query,
                        lookup,
                        total_chunk_count,
                        documents_by_id,
                        candidate_depth,
                        categories_any=category_any_filter,
                        keywords=keyword_filter,
                        projects_any=project_any_filter,
                        document_filter=document_filter,
                        excluded_document_ids=excluded_document_ids,
                        withheld=withheld,
                    ),
                    search_dense(),
                )
                bm25_ranking, bm25_rejected, bm25_chunks = bm25_result
                chunks_by_id.update(bm25_chunks)
            elif use_bm25:
                bm25_ranking, bm25_rejected, bm25_chunks = await self._bm25_ranking(
                    query,
                    lookup,
                    total_chunk_count,
                    documents_by_id,
                    candidate_depth,
                    categories_any=category_any_filter,
                    keywords=keyword_filter,
                    projects_any=project_any_filter,
                    document_filter=document_filter,
                    excluded_document_ids=excluded_document_ids,
                    withheld=withheld,
                )
                chunks_by_id.update(bm25_chunks)
            else:
                dense_hits = await search_dense()

            dense_chunks = await asyncio.to_thread(
                lookup.chunks_by_ids,
                [hit.chunk_id for hit in dense_hits],
            )
            chunks_by_id.update(dense_chunks)
            accepted_dense_hits: list[DenseSearchHit] = []
            dense_below_threshold = 0
            dense_quality_rejected = 0
            dense_corrupt_text_rejected = 0
            for dense_hit in dense_hits:
                chunk = chunks_by_id.get(dense_hit.chunk_id)
                if chunk is None:
                    raise ResearchError(
                        "The dense index returned a chunk absent from the current "
                        f"chunk store: {dense_hit.chunk_id}"
                    )
                dense_flags = _candidate_flags(chunk)
                if dense_flags & CHUNK_FLAG_EXTRACTION_ARTIFACT:
                    dense_quality_rejected += 1
                    continue
                if dense_flags & CHUNK_FLAG_CORRUPT_TEXT:
                    dense_corrupt_text_rejected += 1
                    _record_withheld(
                        withheld,
                        chunk,
                        text_corruption_reasons(_chunk_text(chunk)),
                        limit=self.config.settings.maximum_withheld_examples,
                    )
                    continue
                if not self._matches_filters(
                    chunk,
                    documents_by_id,
                    categories_any=category_any_filter,
                    keywords=keyword_filter,
                    projects_any=project_any_filter,
                    document_filter=document_filter,
                    excluded_document_ids=excluded_document_ids,
                ):
                    continue
                if (
                    dense_hit.score
                    < self.config.settings.dense_minimum_cosine_similarity
                ):
                    dense_below_threshold += 1
                    continue
                accepted_dense_hits.append(dense_hit)
            dense_hits = accepted_dense_hits
            dense_ranking = [hit.chunk_id for hit in dense_hits]
            dense_scores = {hit.chunk_id: hit.score for hit in dense_hits}
            bm25_ranks = {
                chunk_id: rank for rank, chunk_id in enumerate(bm25_ranking, 1)
            }
            dense_ranks = {
                chunk_id: rank for rank, chunk_id in enumerate(dense_ranking, 1)
            }
            if retrieval_method == "hybrid":
                ordered_ids, fusion_scores = self._fuse_rankings(
                    bm25_ranking,
                    dense_ranking,
                    rrf_k=self.config.settings.rrf_k,
                    bm25_weight=self.config.settings.bm25_weight,
                    dense_weight=self.config.settings.dense_weight,
                    maximum_candidates=self.config.settings.maximum_candidates,
                )
            elif retrieval_method == "bm25":
                ordered_ids = bm25_ranking
                fusion_scores = {}
            else:
                ordered_ids = dense_ranking
                fusion_scores = {}

            unknown_ids = [item for item in ordered_ids if item not in chunks_by_id]
            if unknown_ids:
                raise ResearchError(
                    "The retrieval index returned chunk IDs absent from the current "
                    f"chunk store: {unknown_ids[:3]}"
                )

            base_ranks = {
                chunk_id: rank for rank, chunk_id in enumerate(ordered_ids, 1)
            }
            rerank_scores: dict[str, float] = {}
            rerank_fallback: dict[str, Any] | None = None
            if rerank and ordered_ids:
                rerank_count = min(
                    len(ordered_ids),
                    self.config.settings.rerank_max_candidates,
                    max(top_k * 2, 10),
                )
                rerank_ids = ordered_ids[:rerank_count]
                rerank_tail = ordered_ids[rerank_count:]
                # A backend that answers tool calls keeps its configured model,
                # so the keyword is passed only when this call names another.
                rerank_kwargs = {"model": rerank_model} if rerank_model else {}
                try:
                    scores = await asyncio.to_thread(
                        self.dense.rerank,
                        query,
                        [_chunk_text(chunks_by_id[item]) for item in rerank_ids],
                        **rerank_kwargs,
                    )
                except RerankerUnavailable as exc:
                    # Requested reranking cannot run without its model, so the
                    # unranked candidate order is returned unchanged and the
                    # response discloses why instead of failing the search.
                    rerank_fallback = {
                        "reason": "reranker_model_unavailable",
                        "message": str(exc),
                        "effect": "unranked_candidate_order_returned",
                    }
                else:
                    rerank_scores = dict(zip(rerank_ids, scores, strict=True))
                    ordered_ids = (
                        sorted(
                            rerank_ids,
                            key=lambda chunk_id: (
                                -rerank_scores[chunk_id],
                                base_ranks[chunk_id],
                                chunk_id,
                            ),
                        )
                        + rerank_tail
                    )
            reranked_applied = bool(rerank_scores)

            candidate_count = len(ordered_ids)
            candidate_distinct_reference_count = len(
                {
                    str(
                        _document_for_chunk(chunks_by_id[item], documents_by_id)[
                            "source_id"
                        ]
                    )
                    for item in ordered_ids
                }
            )
            selected_ids = ordered_ids[:top_k]

            hits: list[dict[str, Any]] = []
            for rank, chunk_id in enumerate(selected_ids, 1):
                chunk = chunks_by_id[chunk_id]
                document = _document_for_chunk(chunk, documents_by_id)
                component_ranks = {
                    "bm25": bm25_ranks.get(chunk_id),
                    "dense": dense_ranks.get(chunk_id),
                }
                if (
                    component_ranks["bm25"] is not None
                    and component_ranks["dense"] is not None
                ):
                    match_kind = "hybrid"
                elif component_ranks["bm25"] is not None:
                    match_kind = "lexical"
                else:
                    match_kind = "semantic"
                hits.append(
                    {
                        "rank": rank,
                        "retrieval_rank": base_ranks[chunk_id],
                        **_public_passage(chunk, document),
                        "match_kind": match_kind,
                        "retrieval_method": retrieval_method,
                        "component_ranks": component_ranks,
                        "component_scores": {
                            "dense_cosine_similarity": dense_scores.get(chunk_id),
                            "bm25": None,
                        },
                        "fusion_score": fusion_scores.get(chunk_id),
                        "rerank_score": rerank_scores.get(chunk_id),
                    }
                )

            distinct_reference_count = len({str(hit["source_id"]) for hit in hits})
            relevance_limited = candidate_count < top_k

            if include_staleness:
                # Walking the source tree is the only per-request work here that
                # grows with the collection, so a caller that does not need a
                # freshness verdict can skip it.
                status = await asyncio.to_thread(self._status, current)
                stale: bool | None = status["stale"]
                upgrade_reasons = list(status["upgrade_reasons"])
            else:
                stale = None
                upgrade_reasons = self._generation_upgrade_reasons(manifest)
            return {
                "query": query,
                "generation_id": manifest["generation_id"],
                "stale": stale,
                "staleness_checked": include_staleness,
                "generation_upgrade_required": bool(upgrade_reasons),
                "excluded_source_count": len(exclusions),
                "filters": {
                    "categories_any": sorted(category_any_filter),
                    "projects_any": sorted(project_any_filter),
                    "keywords_all": sorted(keyword_filter),
                    "source_document_ids": sorted(document_filter),
                    "source_ids": requested_source_ids,
                    "exclude_source_ids": requested_exclude_source_ids,
                    "unknown_source_ids": unknown_source_ids,
                    "unknown_exclude_source_ids": unknown_exclude_source_ids,
                    "active_document_count": len(active_document_ids),
                    "note": (
                        "Filters narrow the corpus before ranking, so top_k counts "
                        "matches inside the selection. Reviewed source exclusions "
                        "always win: a source_ids entry for an excluded source stays "
                        "excluded. Unresolved IDs are reported in "
                        "unknown_source_ids and unknown_exclude_source_ids; an "
                        "include list that resolves to nothing is an error rather "
                        "than an unfiltered result."
                    ),
                },
                "retrieval_method": retrieval_method,
                "reranked": reranked_applied,
                "rerank_requested": rerank,
                "rerank_fallback": rerank_fallback,
                "candidate_depth": candidate_depth,
                "candidate_count": candidate_count,
                "candidate_distinct_reference_count": (
                    candidate_distinct_reference_count
                ),
                "requested_top_k": top_k,
                "fusion": (
                    {
                        "method": "weighted_reciprocal_rank_fusion",
                        "rrf_k": self.config.settings.rrf_k,
                        "bm25_weight": self.config.settings.bm25_weight,
                        "dense_weight": self.config.settings.dense_weight,
                    }
                    if retrieval_method == "hybrid"
                    else None
                ),
                "relevance_policy": {
                    "bm25_requires_query_token_overlap": True,
                    "dense_minimum_cosine_similarity": (
                        self.config.settings.dense_minimum_cosine_similarity
                    ),
                },
                "rejected_candidates": {
                    "bm25_no_query_token_overlap": bm25_rejected[
                        "no_query_token_overlap"
                    ],
                    "bm25_extraction_artifact": bm25_rejected["extraction_artifact"],
                    "bm25_corrupt_text": bm25_rejected["corrupt_text"],
                    "dense_below_threshold": dense_below_threshold,
                    "dense_extraction_artifact": dense_quality_rejected,
                    "dense_corrupt_text": dense_corrupt_text_rejected,
                },
                "withheld_candidates": {
                    "policy": "corruption_evidence_only",
                    "note": (
                        "Candidates withheld from this result for corruption "
                        "evidence. Script notes such as non_latin_dominant or "
                        "mixed_script_text appear per hit in text_notes and never "
                        "withhold a passage."
                    ),
                    "total": sum(int(entry["count"]) for entry in withheld.values()),
                    "reasons": {
                        reason: {
                            "count": int(entry["count"]),
                            "example_chunk_ids": list(entry["example_chunk_ids"]),
                        }
                        for reason, entry in sorted(withheld.items())
                    },
                    "flagged_passages_returned": sum(
                        1 for hit in hits if hit.get("text_notes")
                    ),
                },
                "dense_fidelity": {
                    "embedding_maximum_tokens": self.config.settings.embedding_maximum_tokens,
                    "audited_passages_returned": sum(
                        1
                        for hit in hits
                        if isinstance(hit.get("embedding_token_count"), int)
                    ),
                    "truncated_passages_returned": sum(
                        1 for hit in hits if hit.get("dense_truncated")
                    ),
                    "note": (
                        "FastEmbed truncates text beyond the embedding model "
                        "limit, so such a chunk is matched lexically but only "
                        "partly semantically. A null embedding_token_count means "
                        "the generation predates the ingestion audit."
                    ),
                },
                "embedding_model": self.config.settings.embedding_model
                if use_dense
                else None,
                "embedding_model_revision": (
                    self.config.settings.embedding_model_revision if use_dense else None
                ),
                "reranker_model": applied_reranker if reranked_applied else None,
                "reranker_model_revision": (
                    applied_reranker_revision if reranked_applied else None
                ),
                "result_count": len(hits),
                "distinct_reference_count": distinct_reference_count,
                "relevance_limited": relevance_limited,
                "hits": hits,
                "notice": (
                    "Returned text is cleaned for semantic retrieval and is not "
                    "quote-safe. Open the original PDF or EPUB at the supplied "
                    "locator for direct quotation."
                ),
            }

    async def list_sources(self) -> dict[str, Any]:
        async with self._operation():
            current = self._load_current_optional()
            try:
                scan = scan_sources(self.config)
            except SourcePolicyError as exc:
                raise ResearchError(str(exc)) from exc
            exclusions = self._source_exclusions()
            metadata = self._metadata()
            indexed_paths = {
                str(document.get("source_relative_path") or "")
                for document in (current[1].get("documents", []) if current else [])
                if isinstance(document, dict)
            }
            discovered_sources = [
                {
                    "source_id": source.source_id,
                    "source_relative_path": source.source_relative_path,
                    "source_path": source.project_relative_path,
                    "format": source.extension.removeprefix("."),
                    "included": source.source_relative_path not in exclusions,
                    "indexed_in_current_generation": (
                        source.source_relative_path in indexed_paths
                    ),
                }
                for source in scan.selected
            ]
            known_sources = self._known_sources(
                scan,
                current,
                exclusions=exclusions,
                metadata=metadata,
            )
            known_source_records = [
                {
                    "source_id": record["source_id"],
                    "source_relative_path": relative,
                    "exists": bool(record["exists"]),
                    "included": relative not in exclusions,
                    "indexed_in_current_generation": bool(
                        record["indexed_in_current_generation"]
                    ),
                    "has_reviewed_metadata": relative in metadata,
                }
                for relative, record in sorted(known_sources.items())
            ]
            reviewed_metadata_sources = [
                {
                    "source_id": known_sources[relative]["source_id"],
                    "source_relative_path": relative,
                    "source_path": known_sources[relative]["source_path"],
                    "metadata": override,
                    "indexed_in_current_generation": relative in indexed_paths,
                }
                for relative, override in sorted(metadata.items())
            ]
            if current is None:
                return {
                    "ready": False,
                    "source_count": 0,
                    "sources": [],
                    "discovered_source_count": len(discovered_sources),
                    "discovered_sources": discovered_sources,
                    "known_source_count": len(known_source_records),
                    "known_sources": known_source_records,
                    "excluded_source_count": len(exclusions),
                    "excluded_sources": self._exclusion_records(scan, exclusions),
                    "reviewed_metadata_source_count": len(reviewed_metadata_sources),
                    "reviewed_metadata_sources": reviewed_metadata_sources,
                }
            _generation_root, manifest = current
            documents_by_id = _effective_documents(manifest, metadata)
            sources = [
                _public_document(document)
                for document in documents_by_id.values()
                if document.get("source_relative_path") not in exclusions
            ]
            return {
                "ready": True,
                "generation_id": manifest["generation_id"],
                "source_count": len(sources),
                "sources": sources,
                "discovered_source_count": len(discovered_sources),
                "discovered_sources": discovered_sources,
                "known_source_count": len(known_source_records),
                "known_sources": known_source_records,
                "excluded_source_count": len(exclusions),
                "excluded_sources": self._exclusion_records(
                    scan,
                    exclusions,
                    manifest,
                ),
                "reviewed_metadata_source_count": len(reviewed_metadata_sources),
                "reviewed_metadata_sources": reviewed_metadata_sources,
            }

    async def get_passage(
        self,
        chunk_id: str,
        *,
        context_chunks: int = 1,
    ) -> dict[str, Any]:
        if not 0 <= context_chunks <= 5:
            raise ResearchError("context_chunks must be between 0 and 5")
        async with self._operation():
            current = self._load_current_optional()
            if current is None:
                raise ResearchError("No knowledge base exists; call ingest first")
            generation_root, manifest = current
            lookup = await self._ensure_artifact_lookup(generation_root, manifest)
            documents_by_id = _effective_documents(manifest, self._metadata())
            target = (await asyncio.to_thread(lookup.chunks_by_ids, [chunk_id])).get(
                chunk_id
            )
            if target is None:
                raise ResearchError(f"Unknown chunk_id: {chunk_id}")
            exclusions = self._source_exclusions()
            excluded_document_ids = self._excluded_document_ids(
                manifest,
                exclusions,
            )
            if target["document_id"] in excluded_document_ids:
                raise ResearchError(
                    "The source for this chunk is currently excluded from retrieval; "
                    "include the source before requesting its passage"
                )
            if _is_extraction_artifact(target):
                raise ResearchError(
                    "The requested chunk is an extraction artifact and is not "
                    "available; re-ingest to remove it from the generation"
                )
            if text_corruption_reasons(_chunk_text(target)):
                raise ResearchError(
                    "The requested chunk contains corrupt extracted text and is "
                    "not available; re-ingest to remove it from the generation"
                )
            document_chunks = await asyncio.to_thread(
                lookup.chunks_for_document,
                str(target["document_id"]),
            )
            same_document = [
                item
                for item in document_chunks
                if not _is_extraction_artifact(item)
                and not text_corruption_reasons(_chunk_text(item))
            ]
            target_position = next(
                index
                for index, item in enumerate(same_document)
                if item["chunk_id"] == chunk_id
            )
            start = max(0, target_position - context_chunks)
            end = min(len(same_document), target_position + context_chunks + 1)
            context = []
            document = _document_for_chunk(target, documents_by_id)
            for item in same_document[start:end]:
                context.append(_public_passage(item, document))
            return {
                "generation_id": manifest["generation_id"],
                "requested_chunk_id": chunk_id,
                "context": context,
                "notice": (
                    "Context is cleaned semantic text and is not quote-safe. Open "
                    "the original source at the supplied locator for quotations."
                ),
            }

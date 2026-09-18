"""High-level, project-scoped research knowledge-base workflow."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from filelock import AsyncFileLock
from filelock import Timeout as FileLockTimeout

from .bundle import (
    BundleError,
    export_generation_bundle,
    install_portable_state,
    install_staged_sources,
    stage_bundle,
)
from .config import ResearchConfig, resolve_source_reference
from .dense import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
    RERANKER_MODEL,
    RERANKER_MODEL_REVISION,
    DenseBackend,
    DenseSearchHit,
    LocalQdrantDenseBackend,
)
from .extraction import extract_sources, normalize_inline_text, normalize_reading_text
from .generation import (
    load_reuse_snapshot,
    source_set_matches,
    value_fingerprint,
)
from .sources import (
    ALLOWED_SOURCE_EXTENSIONS,
    SourcePolicyError,
    SourceScan,
    normalize_metadata,
    scan_sources,
    sha256_file,
)
from .storage import (
    StorageError,
    atomic_write_json,
    load_current_generation,
    load_metadata_overrides,
    load_source_exclusions,
    read_jsonl,
    write_jsonl,
    write_metadata_overrides,
    write_source_exclusions,
)
from .ultrarag import VanillaUltraRAG

SCHEMA_VERSION = 5
CLEANING_POLICY_VERSION = 1
EXTRACTION_POLICY_VERSION = 4
ARTIFACT_POLICY_VERSION = 1
DEFAULT_RETRIEVAL_METHOD = "hybrid"
RETRIEVAL_METHODS = frozenset({"bm25", "dense", "hybrid"})
RRF_K = 60
BM25_RRF_WEIGHT = 1.25
DENSE_RRF_WEIGHT = 1.0
DENSE_MINIMUM_COSINE_SIMILARITY = 0.72
MINIMUM_CANDIDATES = 20
MAXIMUM_CANDIDATES = 200
RERANK_MAX_CANDIDATES = 50
RETRIEVAL_POLICY_FINGERPRINT = value_fingerprint(
    {
        "default_method": DEFAULT_RETRIEVAL_METHOD,
        "available_methods": sorted(RETRIEVAL_METHODS),
        "bm25": {"language": "en", "tokenizer": "default"},
        "fusion": {
            "method": "weighted_reciprocal_rank_fusion",
            "rrf_k": RRF_K,
            "bm25_weight": BM25_RRF_WEIGHT,
            "dense_weight": DENSE_RRF_WEIGHT,
            "minimum_candidates": MINIMUM_CANDIDATES,
            "maximum_candidates": MAXIMUM_CANDIDATES,
        },
        "relevance_gates": {
            "bm25_requires_query_token_overlap": True,
            "dense_minimum_cosine_similarity": DENSE_MINIMUM_COSINE_SIMILARITY,
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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _generation_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


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
        section = locator.get("section_title") or locator.get("href")
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
        if len(token := match.group(0)) >= 2 and token not in _STOPWORDS
    }


def _public_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return document metadata without extraction-related line wrapping."""

    result = dict(document)
    for field in ("title", "doi"):
        result[field] = normalize_inline_text(str(result.get(field) or ""))
    for field in ("authors", "categories", "keywords"):
        result[field] = [
            normalized
            for value in result.get(field) or []
            if (normalized := normalize_inline_text(str(value)))
        ]
    return result


def _enrich_chunks(
    raw_chunks: list[dict[str, Any]],
    units: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    units_by_id = {str(item["id"]): item for item in units}
    documents_by_id = {str(item["document_id"]): item for item in documents}
    document_ordinals: defaultdict[str, int] = defaultdict(int)
    enriched: list[dict[str, Any]] = []
    discarded_empty_chunks = 0

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
                "document_chunk_index": ordinal,
                "unit_id": unit_id,
                "source_path": document["source_path"],
                "title": document["title"],
                "authors": document["authors"],
                "year": document["year"],
                "doi": document["doi"],
                "categories": document["categories"],
                "keywords": document["keywords"],
                "locator": locator,
                "citation": _citation(document, locator),
                "text": text,
                "contents": text,
                "embedding_text": text,
                "content_kind": str(unit.get("content_kind") or "prose"),
                "annotations": list(unit.get("annotations") or []),
                "quality_flags": list(unit.get("quality_flags") or []),
                "metadata_provenance": dict(document.get("metadata_provenance") or {}),
                "metadata_confidence": dict(document.get("metadata_confidence") or {}),
                "metadata_warnings": list(document.get("metadata_warnings") or []),
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
    represented_units = {item["unit_id"] for item in enriched}
    missing_units = [
        str(item["id"]) for item in units if item["id"] not in represented_units
    ]
    if missing_units:
        raise ResearchError(
            "No searchable chunks were produced for extraction units: "
            + ", ".join(missing_units)
        )
    return enriched, discarded_empty_chunks


class ResearchService:
    def __init__(
        self,
        config: ResearchConfig,
        ultrarag: VanillaUltraRAG,
        dense: DenseBackend | None = None,
    ) -> None:
        self.config = config
        self.ultrarag = ultrarag
        self.dense = dense or LocalQdrantDenseBackend(
            config.models_root,
            offline=config.offline,
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

    def _metadata(self) -> dict[str, dict[str, Any]]:
        try:
            values = load_metadata_overrides(self.config.metadata_path)
            return {key: normalize_metadata(value) for key, value in values.items()}
        except (StorageError, SourcePolicyError) as exc:
            raise ResearchError(str(exc)) from exc

    def _source_exclusions(self) -> dict[str, dict[str, str]]:
        try:
            return load_source_exclusions(self.config.source_exclusions_path)
        except StorageError as exc:
            raise ResearchError(str(exc)) from exc

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

    def _status(self) -> dict[str, Any]:
        try:
            scan = scan_sources(self.config)
        except SourcePolicyError as exc:
            raise ResearchError(str(exc)) from exc
        exclusions = self._source_exclusions()
        exclusion_revision = value_fingerprint(exclusions)
        selected = tuple(
            source
            for source in scan.selected
            if source.source_relative_path not in exclusions
        )
        current = self._load_current_optional()
        if current is None:
            if scan.selected and not selected:
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
                "portable_root": str(self.config.portable_root),
                "model_cache_root": str(self.config.model_cache_root),
                "discovered_source_count": len(scan.selected),
                "selected_source_count": len(selected),
                "excluded_source_count": len(exclusions),
                "excluded_sources": self._exclusion_records(scan, exclusions),
                "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
                "ignored_extensions": scan.ignored_extensions,
                "source_exclusion_revision": exclusion_revision,
                "default_retrieval_method": DEFAULT_RETRIEVAL_METHOD,
                "available_retrieval_methods": [],
                "generation_upgrade_required": False,
                "upgrade_reasons": [],
                "last_build_metrics": None,
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
        metadata_changed = value_fingerprint(self._metadata()) != manifest.get(
            "metadata_revision"
        )
        stored_exclusion_revision = manifest.get("source_exclusion_revision")
        source_exclusions_changed = (
            stored_exclusion_revision != exclusion_revision
            if stored_exclusion_revision is not None
            else bool(exclusions)
        )
        stale = bool(
            added
            or removed
            or modified
            or metadata_changed
            or source_exclusions_changed
        )
        retrieval = manifest.get("retrieval", {})
        available_methods = retrieval.get("available_methods") or ["bm25"]
        hybrid_ready = "hybrid" in available_methods
        upgrade_reasons: list[str] = []
        if int(manifest.get("schema_version") or 0) != SCHEMA_VERSION:
            upgrade_reasons.append("generation_schema")
        if (
            int(manifest.get("extraction_policy_version") or 0)
            != EXTRACTION_POLICY_VERSION
        ):
            upgrade_reasons.append("layout_extraction")
        if int(manifest.get("cleaning_policy_version") or 0) != CLEANING_POLICY_VERSION:
            upgrade_reasons.append("semantic_cleaning")
        if int(manifest.get("artifact_policy_version") or 0) != ARTIFACT_POLICY_VERSION:
            upgrade_reasons.append("generation_artifacts")
        if manifest.get("project_id") != self.config.project_id:
            upgrade_reasons.append("project_identity")
        dense_policy = retrieval.get("dense", {})
        fusion_policy = retrieval.get("fusion", {})
        relevance_policy = retrieval.get("relevance_gates", {})
        if (
            dense_policy.get("embedding_model") != EMBEDDING_MODEL
            or dense_policy.get("embedding_model_revision") != EMBEDDING_MODEL_REVISION
            or dense_policy.get("embedding_dimension") != EMBEDDING_DIMENSION
        ):
            upgrade_reasons.append("embedding_model")
        if (
            manifest.get("retrieval_policy_fingerprint") != RETRIEVAL_POLICY_FINGERPRINT
            or fusion_policy.get("method") != "weighted_reciprocal_rank_fusion"
            or fusion_policy.get("rrf_k") != RRF_K
            or fusion_policy.get("bm25_weight") != BM25_RRF_WEIGHT
            or fusion_policy.get("dense_weight") != DENSE_RRF_WEIGHT
            or relevance_policy.get("dense_minimum_cosine_similarity")
            != DENSE_MINIMUM_COSINE_SIMILARITY
            or relevance_policy.get("bm25_requires_query_token_overlap") is not True
        ):
            upgrade_reasons.append("retrieval_policy")
        excluded_document_ids = self._excluded_document_ids(manifest, exclusions)
        exclusion_records = self._exclusion_records(scan, exclusions, manifest)
        return {
            "ready": True,
            "stale": stale,
            "project_root": str(self.config.project_root),
            "project_id": self.config.project_id,
            "project_name": self.config.project_name,
            "source_root": str(self.config.source_root),
            "state_root": str(self.config.state_root),
            "portable_root": str(self.config.portable_root),
            "model_cache_root": str(self.config.model_cache_root),
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
            "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
            "ignored_extensions": scan.ignored_extensions,
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
            "source_exclusion_revision": exclusion_revision,
            "changes": {
                "added": added,
                "removed": removed,
                "modified": modified,
                "metadata_changed": metadata_changed,
                "source_exclusions_changed": source_exclusions_changed,
            },
            "generation_root": str(generation_root),
            "message": (
                "Source exclusions are already enforced by retrieval; run ingest "
                "to rebuild the stored indexes without excluded sources."
                if source_exclusions_changed
                else (
                    "The current generation uses an older extraction or storage "
                    "schema; regenerate it with ingest."
                    if upgrade_reasons
                    else (
                        "Current generation supports hybrid retrieval."
                        if hybrid_ready
                        else "Current generation is BM25-only; run ingest to build "
                        "its project-local dense index."
                    )
                )
            ),
        }

    async def status(self) -> dict[str, Any]:
        async with self._operation():
            return await asyncio.to_thread(self._status)

    async def set_source_metadata(
        self,
        source_path: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        async with self._operation():
            try:
                source = resolve_source_reference(self.config, source_path)
                if (
                    not source.is_file()
                    or source.is_symlink()
                    or source.suffix.lower() not in ALLOWED_SOURCE_EXTENSIONS
                ):
                    raise SourcePolicyError(
                        "Metadata can only be assigned to an existing PDF or EPUB "
                        f"source: {source_path}"
                    )
                relative = source.relative_to(self.config.source_root).as_posix()
                normalized = normalize_metadata(metadata)
                overrides = load_metadata_overrides(self.config.metadata_path)
                overrides[relative] = normalized
                write_metadata_overrides(self.config.metadata_path, overrides)
            except (StorageError, SourcePolicyError, ValueError) as exc:
                raise ResearchError(str(exc)) from exc
            return {
                "source_path": relative,
                "metadata": normalized,
                "requires_ingest": True,
                "message": "Metadata saved. Run ingest to create a new generation.",
            }

    async def set_source_inclusion(
        self,
        source_path: str,
        *,
        included: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Include or exclude one source without changing the source file."""

        async with self._operation():
            try:
                source = resolve_source_reference(self.config, source_path)
                relative = source.relative_to(self.config.source_root).as_posix()
                if source.suffix.lower() not in ALLOWED_SOURCE_EXTENSIONS:
                    raise SourcePolicyError(
                        "Source inclusion can only be changed for a PDF or EPUB: "
                        f"{source_path}"
                    )

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
                    if not source.is_file() or source.is_symlink():
                        raise SourcePolicyError(
                            "Only an existing regular PDF or EPUB can be excluded: "
                            f"{source_path}"
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

                current = self._load_current_optional()
                indexed = False
                if current is not None:
                    _generation_root, manifest = current
                    indexed = any(
                        document.get("source_relative_path") == relative
                        for document in manifest.get("documents", [])
                    )
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
                "source_relative_path": relative,
                "source_path": source.relative_to(self.config.project_root).as_posix(),
                "included": included,
                "reason": None if included else exclusions[relative]["reason"],
                "source_file_changed": False,
                "effective_immediately": effective_immediately,
                "generation_rebuild_recommended": changed,
                "message": message,
            }

    async def ingest(
        self,
        *,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        force_recompute: bool = False,
    ) -> dict[str, Any]:
        if not 50 <= chunk_size <= 384:
            raise ResearchError("chunk_size must be between 50 and 384 GPT-2 tokens")
        if not 0 <= chunk_overlap < chunk_size:
            raise ResearchError(
                "chunk_overlap must be non-negative and below chunk_size"
            )

        async with self._operation():
            build_started = time.perf_counter()
            try:
                scan = scan_sources(self.config)
            except SourcePolicyError as exc:
                raise ResearchError(str(exc)) from exc
            if not scan.selected:
                raise ResearchError(
                    f"No PDF or EPUB sources found beneath {self.config.source_root}"
                )

            metadata = self._metadata()
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

            phase_timings: dict[str, float] = {}
            phase_started = time.perf_counter()
            source_digests = await asyncio.to_thread(
                lambda: {
                    source.source_relative_path: sha256_file(source.path)
                    for source in scan.selected
                }
            )
            phase_timings["source_hashing"] = time.perf_counter() - phase_started
            metadata_digest = value_fingerprint(metadata)
            exclusion_digest = value_fingerprint(exclusions)
            source_records = [
                {
                    "source_path": source.project_relative_path,
                    "source_relative_path": source.source_relative_path,
                    "format": source.extension.removeprefix("."),
                    "size": source.size,
                    "mtime_ns": source.mtime_ns,
                    "sha256": source_digests[source.source_relative_path],
                    "included": source.source_relative_path not in exclusions,
                }
                for source in scan.selected
            ]

            current = self._load_current_optional()
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
                )
            if snapshot is not None and source_set_matches(
                snapshot,
                source_records,
                metadata_revision=metadata_digest,
                exclusion_revision=exclusion_digest,
                retrieval_policy_fingerprint=RETRIEVAL_POLICY_FINGERPRINT,
            ):
                manifest = snapshot.manifest
                phase_timings.update(
                    {
                        "extraction": 0.0,
                        "chunking": 0.0,
                        "embedding": 0.0,
                        "bm25_indexing": 0.0,
                        "qdrant_indexing": 0.0,
                        "total": time.perf_counter() - build_started,
                    }
                )
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
                    "phase_timings_seconds": phase_timings,
                    "message": "Inputs match the selected generation; no build was needed.",
                }

            reused_documents: dict[str, dict[str, Any]] = {}
            reused_units: dict[str, list[dict[str, Any]]] = {}
            reused_chunks: dict[str, list[dict[str, Any]]] = {}
            rebuild_sources = []
            if snapshot is not None:
                for source in selected:
                    relative = source.source_relative_path
                    previous_source = snapshot.source_files.get(relative)
                    previous_document = snapshot.documents.get(relative)
                    override_revision = value_fingerprint(metadata.get(relative, {}))
                    document_id = str(
                        (previous_document or {}).get("document_id") or ""
                    )
                    if (
                        previous_source is not None
                        and previous_document is not None
                        and previous_source.get("sha256") == source_digests[relative]
                        and previous_document.get("metadata_override_revision")
                        == override_revision
                        and snapshot.units_by_document.get(document_id)
                        and snapshot.chunks_by_document.get(document_id)
                    ):
                        reused_document = dict(previous_document)
                        reused_document["size"] = source.size
                        reused_document["mtime_ns"] = source.mtime_ns
                        reused_documents[relative] = reused_document
                        reused_units[relative] = [
                            dict(item)
                            for item in snapshot.units_by_document[document_id]
                        ]
                        reused_chunks[relative] = [
                            dict(item)
                            for item in snapshot.chunks_by_document[document_id]
                        ]
                    else:
                        rebuild_sources.append(source)
            else:
                rebuild_sources.extend(selected)

            generation_id = _generation_id()
            staging_root = self.config.staging_root / generation_id
            generation_root = self.config.generations_root / generation_id
            staging_root.mkdir(parents=False, exist_ok=False)
            extracted_path = staging_root / "corpus" / "extracted-units.jsonl"
            chunks_path = staging_root / "chunks" / "chunks.jsonl"
            bm25_index_path = staging_root / "indexes" / "bm25"
            dense_index_path = staging_root / "indexes" / "qdrant"
            vectors_path = staging_root / "portable" / "embeddings.npy"
            work_root = staging_root / "work"
            current_phase = "extraction"
            generation_installed = False

            try:
                phase_started = time.perf_counter()
                rebuilt_documents, rebuilt_units = await asyncio.to_thread(
                    extract_sources,
                    tuple(rebuild_sources),
                    metadata,
                    source_digests,
                )
                for document in rebuilt_documents:
                    relative = str(document["source_relative_path"])
                    document["metadata_override_revision"] = value_fingerprint(
                        metadata.get(relative, {})
                    )
                phase_timings["extraction"] = time.perf_counter() - phase_started

                current_phase = "chunking"
                phase_started = time.perf_counter()
                rebuilt_chunks: list[dict[str, Any]] = []
                discarded_empty_chunks = 0
                if rebuilt_units:
                    changed_units_path = work_root / "changed-units.jsonl"
                    raw_chunks_path = work_root / "ultrarag-chunks.jsonl"
                    write_jsonl(changed_units_path, rebuilt_units)
                    await self.ultrarag.chunk(
                        changed_units_path,
                        raw_chunks_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    raw_chunks = read_jsonl(raw_chunks_path)
                    rebuilt_chunks, discarded_empty_chunks = _enrich_chunks(
                        raw_chunks,
                        rebuilt_units,
                        rebuilt_documents,
                    )
                phase_timings["chunking"] = time.perf_counter() - phase_started

                rebuilt_documents_by_path = {
                    str(item["source_relative_path"]): item
                    for item in rebuilt_documents
                }
                rebuilt_units_by_document: dict[str, list[dict[str, Any]]] = (
                    defaultdict(list)
                )
                for unit in rebuilt_units:
                    rebuilt_units_by_document[str(unit["document_id"])].append(unit)
                rebuilt_chunks_by_document: dict[str, list[dict[str, Any]]] = (
                    defaultdict(list)
                )
                for chunk in rebuilt_chunks:
                    rebuilt_chunks_by_document[str(chunk["document_id"])].append(chunk)

                documents: list[dict[str, Any]] = []
                units: list[dict[str, Any]] = []
                chunks: list[dict[str, Any]] = []
                for source in selected:
                    relative = source.source_relative_path
                    if relative in reused_documents:
                        document = reused_documents[relative]
                        source_units = reused_units[relative]
                        source_chunks = reused_chunks[relative]
                    else:
                        document = rebuilt_documents_by_path[relative]
                        document_id = str(document["document_id"])
                        source_units = rebuilt_units_by_document[document_id]
                        source_chunks = rebuilt_chunks_by_document[document_id]
                    documents.append(document)
                    units.extend(source_units)
                    chunks.extend(source_chunks)

                write_jsonl(extracted_path, units)
                write_jsonl(chunks_path, chunks)

                current_phase = "embedding"
                phase_started = time.perf_counter()
                vectors = np.empty(
                    (len(chunks), EMBEDDING_DIMENSION),
                    dtype=np.float32,
                )
                missing_positions: list[int] = []
                missing_texts: list[str] = []
                vectors_by_text = (
                    snapshot.vectors_by_text
                    if snapshot is not None and not force_recompute
                    else {}
                )
                reused_vector_count = 0
                for index, chunk in enumerate(chunks):
                    embedding_text = str(chunk["embedding_text"])
                    reusable_vector = vectors_by_text.get(embedding_text)
                    if reusable_vector is None:
                        missing_positions.append(index)
                        missing_texts.append(embedding_text)
                    else:
                        vectors[index] = reusable_vector
                        reused_vector_count += 1
                if missing_texts:
                    created_vectors = await asyncio.to_thread(
                        self.dense.embed_texts,
                        missing_texts,
                    )
                    if created_vectors.shape != (
                        len(missing_positions),
                        EMBEDDING_DIMENSION,
                    ):
                        raise ResearchError(
                            "Dense backend returned an invalid embedding matrix"
                        )
                    for position, vector in zip(
                        missing_positions,
                        created_vectors,
                        strict=True,
                    ):
                        vectors[position] = vector
                vectors_path.parent.mkdir(parents=True, exist_ok=True)
                with vectors_path.open("wb") as handle:
                    np.save(handle, vectors, allow_pickle=False)
                phase_timings["embedding"] = time.perf_counter() - phase_started

                current_phase = "bm25_indexing"
                phase_started = time.perf_counter()
                await self.ultrarag.build_bm25(chunks_path, bm25_index_path)
                phase_timings["bm25_indexing"] = time.perf_counter() - phase_started

                current_phase = "qdrant_indexing"
                phase_started = time.perf_counter()
                dense_metadata = await asyncio.to_thread(
                    self.dense.build_index,
                    chunks,
                    dense_index_path,
                    vectors,
                )
                phase_timings["qdrant_indexing"] = time.perf_counter() - phase_started
                content_kind_counts = dict(
                    sorted(
                        Counter(str(item["content_kind"]) for item in chunks).items()
                    )
                )
                phase_timings["total"] = time.perf_counter() - build_started
                build_metrics = {
                    "forced": force_recompute,
                    "reused_document_count": len(reused_documents),
                    "rebuilt_document_count": len(rebuilt_documents),
                    "reused_chunk_count": sum(
                        len(items) for items in reused_chunks.values()
                    ),
                    "rebuilt_chunk_count": len(rebuilt_chunks),
                    "reused_vector_count": reused_vector_count,
                    "created_vector_count": len(missing_positions),
                    "phase_timings_seconds": {
                        key: round(value, 6) for key, value in phase_timings.items()
                    },
                }
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "extraction_policy_version": EXTRACTION_POLICY_VERSION,
                    "cleaning_policy_version": CLEANING_POLICY_VERSION,
                    "artifact_policy_version": ARTIFACT_POLICY_VERSION,
                    "generation_id": generation_id,
                    "project_id": self.config.project_id,
                    "project_name": self.config.project_name,
                    "created_at": _utc_now(),
                    "source_directory": self.config.source_root.relative_to(
                        self.config.project_root
                    ).as_posix(),
                    "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
                    "ignored_extensions": scan.ignored_extensions,
                    "metadata_revision": metadata_digest,
                    "source_exclusion_revision": exclusion_digest,
                    "retrieval_policy_fingerprint": RETRIEVAL_POLICY_FINGERPRINT,
                    "source_file_count": len(scan.selected),
                    "excluded_source_count": sum(
                        source.source_relative_path in exclusions
                        for source in scan.selected
                    ),
                    "document_count": len(documents),
                    "extraction_unit_count": len(units),
                    "chunk_count": len(chunks),
                    "content_kind_counts": content_kind_counts,
                    "discarded_empty_chunk_count": discarded_empty_chunks,
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
                            "rrf_k": RRF_K,
                            "bm25_weight": BM25_RRF_WEIGHT,
                            "dense_weight": DENSE_RRF_WEIGHT,
                            "minimum_candidates": MINIMUM_CANDIDATES,
                            "maximum_candidates": MAXIMUM_CANDIDATES,
                        },
                        "relevance_gates": {
                            "bm25_requires_query_token_overlap": True,
                            "dense_minimum_cosine_similarity": (
                                DENSE_MINIMUM_COSINE_SIMILARITY
                            ),
                        },
                        "reranker": {
                            "optional": True,
                            "runtime": "FastEmbed ONNX Runtime (CPU)",
                            "model": RERANKER_MODEL,
                            "model_revision": RERANKER_MODEL_REVISION,
                            "maximum_candidates": RERANK_MAX_CANDIDATES,
                        },
                    },
                    "documents": documents,
                    "source_files": source_records,
                    "excluded_sources": [
                        {
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
                        "dense_index": "indexes/qdrant",
                        "portable_embeddings": "portable/embeddings.npy",
                    },
                    "build_metrics": build_metrics,
                }
                shutil.rmtree(work_root, ignore_errors=True)
                atomic_write_json(staging_root / "manifest.json", manifest)
                os.replace(staging_root, generation_root)
                generation_installed = True
                atomic_write_json(
                    self.config.current_path,
                    {
                        # Pointer format is stable across generation schema upgrades.
                        "schema_version": 1,
                        "generation_id": generation_id,
                    },
                )
                self._loaded_generation = generation_id
            except Exception as exc:
                shutil.rmtree(staging_root, ignore_errors=True)
                if generation_installed:
                    shutil.rmtree(generation_root, ignore_errors=True)
                atomic_write_json(
                    self.config.failures_root / f"{generation_id}.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "generation_id": generation_id,
                        "failed_at": _utc_now(),
                        "phase": current_phase,
                        "error": str(exc),
                    },
                )
                raise

            return {
                "status": "ready",
                "generation_changed": True,
                "generation_id": generation_id,
                "generation_root": str(generation_root),
                "source_file_count": len(scan.selected),
                "excluded_source_count": sum(
                    source.source_relative_path in exclusions
                    for source in scan.selected
                ),
                "excluded_sources": [
                    source.source_relative_path
                    for source in scan.selected
                    if source.source_relative_path in exclusions
                ],
                "document_count": len(documents),
                "pdf_count": sum(item["format"] == "pdf" for item in documents),
                "epub_count": sum(item["format"] == "epub" for item in documents),
                "extraction_unit_count": len(units),
                "chunk_count": len(chunks),
                "content_kind_counts": content_kind_counts,
                "discarded_empty_chunk_count": discarded_empty_chunks,
                "ignored_extensions": scan.ignored_extensions,
                "empty_units": sum(int(item["empty_units"]) for item in documents),
                "default_retrieval_method": DEFAULT_RETRIEVAL_METHOD,
                "available_retrieval_methods": sorted(RETRIEVAL_METHODS),
                "embedding_model": EMBEDDING_MODEL,
                "embedding_model_revision": EMBEDDING_MODEL_REVISION,
                **build_metrics,
            }

    async def export_bundle(self) -> dict[str, Any]:
        """Export the selected fresh generation and all original sources."""

        async with self._operation():
            status = await asyncio.to_thread(self._status)
            if not status.get("ready"):
                raise ResearchError(
                    "No generation exists; call ingest before exporting"
                )
            if status.get("stale"):
                raise ResearchError(
                    "The current generation is stale; ingest before exporting so "
                    "the bundle and original sources describe the same collection"
                )
            if status.get("generation_upgrade_required"):
                raise ResearchError(
                    "The current generation requires regeneration before export"
                )
            current = self._load_current_optional()
            if (
                current is None
            ):  # Defensive: status above already established readiness.
                raise ResearchError(
                    "No generation exists; call ingest before exporting"
                )
            generation_root, manifest = current
            try:
                scan = scan_sources(self.config)
                return await asyncio.to_thread(
                    export_generation_bundle,
                    self.config,
                    generation_root,
                    manifest,
                    scan,
                )
            except (BundleError, SourcePolicyError) as exc:
                raise ResearchError(str(exc)) from exc

    async def import_bundle(
        self,
        bundle_name: str,
        *,
        activate: bool = True,
    ) -> dict[str, Any]:
        """Validate a portable bundle and reconstruct local search indexes."""

        name = Path(bundle_name)
        if name.is_absolute() or len(name.parts) != 1 or name.name != bundle_name:
            raise ResearchError(
                "bundle_name must name one file directly beneath .research-rag/bundles"
            )
        bundle_path = (self.config.bundles_root / name).resolve()
        try:
            bundle_path.relative_to(self.config.bundles_root)
        except ValueError as exc:
            raise ResearchError(
                "Bundle path escapes the portable bundle directory"
            ) from exc

        async with self._operation():
            staged = None
            try:
                staged = await asyncio.to_thread(
                    stage_bundle,
                    self.config,
                    bundle_path,
                    generation_schema_version=SCHEMA_VERSION,
                    extraction_policy_version=EXTRACTION_POLICY_VERSION,
                    cleaning_policy_version=CLEANING_POLICY_VERSION,
                    artifact_policy_version=ARTIFACT_POLICY_VERSION,
                    embedding_model=EMBEDDING_MODEL,
                    embedding_model_revision=EMBEDDING_MODEL_REVISION,
                    embedding_dimension=EMBEDDING_DIMENSION,
                )
                generation_id = str(staged.manifest["generation_id"])
                target = self.config.generations_root / generation_id
                already_present = target.exists()
                if already_present:
                    if not target.is_dir() or target.is_symlink():
                        raise BundleError(
                            "Existing generation destination is not a safe directory"
                        )
                    existing = json.loads(
                        (target / "manifest.json").read_text(encoding="utf-8")
                    )
                    if (
                        existing.get("generation_id") != generation_id
                        or existing.get("project_id") != self.config.project_id
                        or existing.get("schema_version") != SCHEMA_VERSION
                    ):
                        raise BundleError(
                            "Existing generation directory has a conflicting manifest"
                        )
                    for field in (
                        "extraction_policy_version",
                        "cleaning_policy_version",
                        "artifact_policy_version",
                        "source_directory",
                        "chunk_count",
                        "document_count",
                        "retrieval",
                        "files",
                    ):
                        if existing.get(field) != staged.manifest.get(field):
                            raise BundleError(
                                "Existing generation directory has conflicting "
                                f"{field} metadata"
                            )
                    for index_name in ("bm25_index", "dense_index"):
                        index_path = target / str(existing["files"][index_name])
                        if not index_path.is_dir() or index_path.is_symlink():
                            raise BundleError(
                                "Existing generation has a missing or unsafe "
                                f"{index_name}"
                            )
                    for relative, checksum in staged.descriptor["files"].items():
                        if not relative.startswith("generation/") or relative.endswith(
                            "/manifest.json"
                        ):
                            continue
                        generation_relative = relative.removeprefix("generation/")
                        existing_file = target / generation_relative
                        if (
                            not existing_file.is_file()
                            or existing_file.is_symlink()
                            or existing_file.stat().st_size != checksum["size"]
                            or sha256_file(existing_file) != checksum["sha256"]
                        ):
                            raise BundleError(
                                "Existing generation content conflicts with the bundle: "
                                f"{generation_relative}"
                            )
                else:
                    files = staged.manifest.get("files", {})
                    chunks_path = staged.generation_root / str(files["chunks"])
                    bm25_path = staged.generation_root / str(files["bm25_index"])
                    dense_path = staged.generation_root / str(files["dense_index"])
                    vectors_path = staged.generation_root / str(
                        files["portable_embeddings"]
                    )
                    chunks = read_jsonl(chunks_path)
                    await self.ultrarag.build_bm25(chunks_path, bm25_path)
                    await asyncio.to_thread(
                        self.dense.build_from_vectors,
                        chunks,
                        dense_path,
                        vectors_path,
                    )

                    await asyncio.to_thread(install_staged_sources, self.config, staged)
                    await asyncio.to_thread(install_portable_state, self.config, staged)
                    source_stats = {
                        item["path"]: (self.config.source_root / item["path"]).stat()
                        for item in staged.descriptor["sources"]
                    }
                    for record in staged.manifest.get("source_files", []):
                        relative = str(record.get("source_relative_path") or "")
                        if stat_result := source_stats.get(relative):
                            record["size"] = stat_result.st_size
                            record["mtime_ns"] = stat_result.st_mtime_ns
                    for document in staged.manifest.get("documents", []):
                        relative = str(document.get("source_relative_path") or "")
                        if stat_result := source_stats.get(relative):
                            document["size"] = stat_result.st_size
                            document["mtime_ns"] = stat_result.st_mtime_ns
                    atomic_write_json(
                        staged.generation_root / "manifest.json",
                        staged.manifest,
                    )
                    os.replace(staged.generation_root, target)

                if already_present:
                    await asyncio.to_thread(install_staged_sources, self.config, staged)
                    await asyncio.to_thread(install_portable_state, self.config, staged)

                if activate:
                    atomic_write_json(
                        self.config.current_path,
                        {"schema_version": 1, "generation_id": generation_id},
                    )
                self._loaded_generation = None
                return {
                    "status": "already_present" if already_present else "imported",
                    "generation_id": generation_id,
                    "activated": activate,
                    "generation_root": str(target),
                    "source_count": len(staged.descriptor["sources"]),
                    "contains_original_sources": True,
                    "message": (
                        "Bundle imported and selected."
                        if activate
                        else "Bundle imported without changing the selected generation."
                    ),
                }
            except (BundleError, StorageError, OSError, KeyError, ValueError) as exc:
                raise ResearchError(str(exc)) from exc
            finally:
                if staged is not None:
                    shutil.rmtree(staged.root, ignore_errors=True)

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
        *,
        categories: set[str],
        keywords: set[str],
        document_ids: set[str],
        excluded_document_ids: set[str],
    ) -> bool:
        chunk_categories = {
            str(item).casefold() for item in chunk.get("categories", [])
        }
        chunk_keywords = {str(item).casefold() for item in chunk.get("keywords", [])}
        return not (
            chunk["document_id"] in excluded_document_ids
            or (categories and not categories.issubset(chunk_categories))
            or (keywords and not keywords.issubset(chunk_keywords))
            or (document_ids and chunk["document_id"] not in document_ids)
        )

    async def _bm25_ranking(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        limit: int,
        *,
        categories: set[str],
        keywords: set[str],
        document_ids: set[str],
        excluded_document_ids: set[str],
    ) -> tuple[list[str], dict[str, int]]:
        if limit <= 0:
            return [], {"no_query_token_overlap": 0, "extraction_artifact": 0}
        filtered = bool(categories or keywords or document_ids or excluded_document_ids)
        requested = (
            len(chunks)
            if filtered
            else min(len(chunks), max(limit * 4, MINIMUM_CANDIDATES))
        )
        passages = await self.ultrarag.search_bm25(query, requested)
        by_contents: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in chunks:
            by_contents[str(item["contents"])].append(item)
        query_tokens = _content_tokens(query)
        ranking: list[str] = []
        used: set[str] = set()
        rejected = {"no_query_token_overlap": 0, "extraction_artifact": 0}
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
                        categories=categories,
                        keywords=keywords,
                        document_ids=document_ids,
                        excluded_document_ids=excluded_document_ids,
                    )
                ),
                None,
            )
            if chunk is None:
                continue
            if not self._matches_filters(
                chunk,
                categories=categories,
                keywords=keywords,
                document_ids=document_ids,
                excluded_document_ids=excluded_document_ids,
            ):
                continue
            chunk_id = str(chunk["chunk_id"])
            used.add(chunk_id)
            quality_flags = {str(item) for item in chunk.get("quality_flags", [])}
            if "extraction_artifact" in quality_flags:
                rejected["extraction_artifact"] += 1
                continue
            if not query_tokens.intersection(
                _content_tokens(str(chunk.get("text") or ""))
            ):
                rejected["no_query_token_overlap"] += 1
                continue
            ranking.append(chunk_id)
            if len(ranking) == limit:
                break
        return ranking, rejected

    @staticmethod
    def _fuse_rankings(
        bm25_ranking: list[str],
        dense_ranking: list[str],
    ) -> tuple[list[str], dict[str, float]]:
        scores: defaultdict[str, float] = defaultdict(float)
        component_ranks = (
            {chunk_id: rank for rank, chunk_id in enumerate(ranking, 1)}
            for ranking in (bm25_ranking, dense_ranking)
        )
        bm25_ranks, dense_ranks = component_ranks
        for chunk_id, rank in bm25_ranks.items():
            scores[chunk_id] += BM25_RRF_WEIGHT / (RRF_K + rank)
        for chunk_id, rank in dense_ranks.items():
            scores[chunk_id] += DENSE_RRF_WEIGHT / (RRF_K + rank)
        ordered = sorted(
            scores,
            key=lambda chunk_id: (
                -scores[chunk_id],
                min(
                    bm25_ranks.get(chunk_id, MAXIMUM_CANDIDATES + 1),
                    dense_ranks.get(chunk_id, MAXIMUM_CANDIDATES + 1),
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
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        document_ids: list[str] | None = None,
        retrieval_method: str = DEFAULT_RETRIEVAL_METHOD,
        rerank: bool = False,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ResearchError("query must not be empty")
        if not 1 <= top_k <= 50:
            raise ResearchError("top_k must be between 1 and 50")
        retrieval_method = retrieval_method.casefold().strip()
        if retrieval_method not in RETRIEVAL_METHODS:
            raise ResearchError("retrieval_method must be one of: bm25, dense, hybrid")

        async with self._operation():
            current = self._load_current_optional()
            if current is None:
                raise ResearchError("No knowledge base exists; call ingest first")
            generation_root, manifest = current
            chunks = read_jsonl(generation_root / manifest["files"]["chunks"])
            if not chunks:
                raise ResearchError("The current generation has no chunks")
            chunks_by_id = {str(item["chunk_id"]): item for item in chunks}
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

            normalized_categories = [
                item.strip() for item in categories or [] if item.strip()
            ]
            normalized_keywords = [
                item.strip() for item in keywords or [] if item.strip()
            ]
            normalized_document_ids = [
                item.strip() for item in document_ids or [] if item.strip()
            ]
            category_filter = {item.casefold() for item in normalized_categories}
            keyword_filter = {item.casefold() for item in normalized_keywords}
            document_filter = set(normalized_document_ids)
            active_chunk_count = sum(
                chunk["document_id"] not in excluded_document_ids for chunk in chunks
            )
            candidate_depth = min(
                active_chunk_count,
                MAXIMUM_CANDIDATES,
                max(MINIMUM_CANDIDATES, top_k * 4),
            )

            use_bm25 = retrieval_method in {"bm25", "hybrid"}
            use_dense = retrieval_method in {"dense", "hybrid"}
            if use_bm25:
                await self._ensure_loaded(generation_root, manifest)

            bm25_ranking: list[str] = []
            dense_hits: list[DenseSearchHit] = []
            bm25_rejected = {
                "no_query_token_overlap": 0,
                "extraction_artifact": 0,
            }

            async def search_dense() -> list[DenseSearchHit]:
                if candidate_depth == 0:
                    return []
                return await asyncio.to_thread(
                    self.dense.search,
                    generation_root / manifest["files"]["dense_index"],
                    query,
                    candidate_depth,
                    categories=normalized_categories,
                    keywords=normalized_keywords,
                    document_ids=normalized_document_ids,
                    excluded_document_ids=sorted(excluded_document_ids),
                )

            if use_bm25 and use_dense:
                bm25_result, dense_hits = await asyncio.gather(
                    self._bm25_ranking(
                        query,
                        chunks,
                        candidate_depth,
                        categories=category_filter,
                        keywords=keyword_filter,
                        document_ids=document_filter,
                        excluded_document_ids=excluded_document_ids,
                    ),
                    search_dense(),
                )
                bm25_ranking, bm25_rejected = bm25_result
            elif use_bm25:
                bm25_ranking, bm25_rejected = await self._bm25_ranking(
                    query,
                    chunks,
                    candidate_depth,
                    categories=category_filter,
                    keywords=keyword_filter,
                    document_ids=document_filter,
                    excluded_document_ids=excluded_document_ids,
                )
            else:
                dense_hits = await search_dense()

            accepted_dense_hits: list[DenseSearchHit] = []
            dense_below_threshold = 0
            dense_quality_rejected = 0
            for dense_hit in dense_hits:
                chunk = chunks_by_id.get(dense_hit.chunk_id)
                if chunk is None:
                    raise ResearchError(
                        "The dense index returned a chunk absent from the current "
                        f"chunk store: {dense_hit.chunk_id}"
                    )
                quality_flags = {str(item) for item in chunk.get("quality_flags", [])}
                if "extraction_artifact" in quality_flags:
                    dense_quality_rejected += 1
                    continue
                if dense_hit.score < DENSE_MINIMUM_COSINE_SIMILARITY:
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
            if rerank and ordered_ids:
                rerank_count = min(
                    len(ordered_ids),
                    RERANK_MAX_CANDIDATES,
                    max(top_k * 2, 10),
                )
                rerank_ids = ordered_ids[:rerank_count]
                scores = await asyncio.to_thread(
                    self.dense.rerank,
                    query,
                    [
                        str(
                            chunks_by_id[item].get("embedding_text")
                            or chunks_by_id[item]["text"]
                        )
                        for item in rerank_ids
                    ],
                )
                rerank_scores = dict(zip(rerank_ids, scores, strict=True))
                ordered_ids = sorted(
                    rerank_ids,
                    key=lambda chunk_id: (
                        -rerank_scores[chunk_id],
                        base_ranks[chunk_id],
                        chunk_id,
                    ),
                )

            hits: list[dict[str, Any]] = []
            for rank, chunk_id in enumerate(ordered_ids[:top_k], 1):
                chunk = chunks_by_id[chunk_id]
                public_document = _public_document(chunk)
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
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "title": public_document["title"],
                        "authors": public_document["authors"],
                        "year": chunk["year"],
                        "doi": public_document["doi"],
                        "source_path": chunk["source_path"],
                        "categories": public_document["categories"],
                        "keywords": public_document["keywords"],
                        "locator": chunk["locator"],
                        "citation": normalize_inline_text(str(chunk["citation"])),
                        "text": normalize_reading_text(str(chunk["text"])),
                        "text_fidelity": "cleaned_semantic_text",
                        "direct_quote_safe": False,
                        "content_kind": str(chunk.get("content_kind") or "prose"),
                        "annotations": list(chunk.get("annotations") or []),
                        "quality_flags": list(chunk.get("quality_flags") or []),
                        "metadata_provenance": dict(
                            chunk.get("metadata_provenance") or {}
                        ),
                        "metadata_confidence": dict(
                            chunk.get("metadata_confidence") or {}
                        ),
                        "metadata_warnings": list(chunk.get("metadata_warnings") or []),
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

            status = await asyncio.to_thread(self._status)
            return {
                "query": query,
                "generation_id": manifest["generation_id"],
                "stale": status["stale"],
                "generation_upgrade_required": status["generation_upgrade_required"],
                "excluded_source_count": len(exclusions),
                "retrieval_method": retrieval_method,
                "reranked": rerank,
                "candidate_depth": candidate_depth,
                "requested_top_k": top_k,
                "fusion": (
                    {
                        "method": "weighted_reciprocal_rank_fusion",
                        "rrf_k": RRF_K,
                        "bm25_weight": BM25_RRF_WEIGHT,
                        "dense_weight": DENSE_RRF_WEIGHT,
                    }
                    if retrieval_method == "hybrid"
                    else None
                ),
                "relevance_policy": {
                    "bm25_requires_query_token_overlap": True,
                    "dense_minimum_cosine_similarity": (
                        DENSE_MINIMUM_COSINE_SIMILARITY
                    ),
                },
                "rejected_candidates": {
                    "bm25_no_query_token_overlap": bm25_rejected[
                        "no_query_token_overlap"
                    ],
                    "bm25_extraction_artifact": bm25_rejected["extraction_artifact"],
                    "dense_below_threshold": dense_below_threshold,
                    "dense_extraction_artifact": dense_quality_rejected,
                },
                "embedding_model": EMBEDDING_MODEL if use_dense else None,
                "embedding_model_revision": (
                    EMBEDDING_MODEL_REVISION if use_dense else None
                ),
                "reranker_model": RERANKER_MODEL if rerank else None,
                "reranker_model_revision": (
                    RERANKER_MODEL_REVISION if rerank else None
                ),
                "result_count": len(hits),
                "relevance_limited": len(hits) < top_k,
                "hits": hits,
                "notice": (
                    "Returned text is cleaned for semantic retrieval and is not "
                    "quote-safe. Open the original PDF or EPUB at the supplied "
                    "locator for direct quotation."
                ),
            }

    async def list_sources(
        self,
        *,
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        async with self._operation():
            current = self._load_current_optional()
            if current is None:
                exclusions = self._source_exclusions()
                try:
                    scan = scan_sources(self.config)
                except SourcePolicyError as exc:
                    raise ResearchError(str(exc)) from exc
                return {
                    "ready": False,
                    "source_count": 0,
                    "sources": [],
                    "excluded_source_count": len(exclusions),
                    "excluded_sources": self._exclusion_records(scan, exclusions),
                }
            _generation_root, manifest = current
            exclusions = self._source_exclusions()
            category_filter = {item.casefold() for item in categories or []}
            keyword_filter = {item.casefold() for item in keywords or []}
            sources = []
            for document in manifest["documents"]:
                if document.get("source_relative_path") in exclusions:
                    continue
                document_categories = {
                    str(item).casefold() for item in document.get("categories", [])
                }
                document_keywords = {
                    str(item).casefold() for item in document.get("keywords", [])
                }
                if category_filter and not category_filter.issubset(
                    document_categories
                ):
                    continue
                if keyword_filter and not keyword_filter.issubset(document_keywords):
                    continue
                sources.append(_public_document(document))
            return {
                "ready": True,
                "generation_id": manifest["generation_id"],
                "source_count": len(sources),
                "sources": sources,
                "excluded_source_count": len(exclusions),
                "excluded_sources": [
                    {
                        "source_relative_path": relative,
                        **record,
                    }
                    for relative, record in exclusions.items()
                ],
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
            chunks = read_jsonl(generation_root / manifest["files"]["chunks"])
            target = next(
                (item for item in chunks if item.get("chunk_id") == chunk_id),
                None,
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
            same_document = sorted(
                (
                    item
                    for item in chunks
                    if item.get("document_id") == target["document_id"]
                ),
                key=lambda item: int(item["document_chunk_index"]),
            )
            target_position = next(
                index
                for index, item in enumerate(same_document)
                if item["chunk_id"] == chunk_id
            )
            start = max(0, target_position - context_chunks)
            end = min(len(same_document), target_position + context_chunks + 1)
            context = [
                {
                    "chunk_id": item["chunk_id"],
                    "document_id": item["document_id"],
                    "source_path": item["source_path"],
                    "title": _public_document(item)["title"],
                    "authors": _public_document(item)["authors"],
                    "year": item.get("year"),
                    "doi": _public_document(item)["doi"],
                    "locator": item["locator"],
                    "citation": normalize_inline_text(str(item["citation"])),
                    "text": normalize_reading_text(str(item["text"])),
                    "text_fidelity": "cleaned_semantic_text",
                    "direct_quote_safe": False,
                    "content_kind": str(item.get("content_kind") or "prose"),
                    "annotations": list(item.get("annotations") or []),
                    "quality_flags": list(item.get("quality_flags") or []),
                    "metadata_provenance": dict(item.get("metadata_provenance") or {}),
                    "metadata_confidence": dict(item.get("metadata_confidence") or {}),
                    "metadata_warnings": list(item.get("metadata_warnings") or []),
                }
                for item in same_document[start:end]
            ]
            return {
                "generation_id": manifest["generation_id"],
                "requested_chunk_id": chunk_id,
                "context": context,
                "notice": (
                    "Context is cleaned semantic text and is not quote-safe. Open "
                    "the original source at the supplied locator for quotations."
                ),
            }

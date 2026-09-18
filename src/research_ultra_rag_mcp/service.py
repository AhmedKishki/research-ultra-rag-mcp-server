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
from .extraction import (
    ExtractionError,
    extract_epub_spine_item,
    extract_scanned_pdf_page,
    normalize_inline_text,
    normalize_reading_text,
    pdf_page_count,
    prepare_epub_extraction,
    prepare_scanned_pdf,
    scan_pdf_page,
    text_corruption_reasons,
)
from .generation import (
    load_reuse_snapshot,
    source_set_matches,
    value_fingerprint,
)
from .sources import (
    ALLOWED_SOURCE_EXTENSIONS,
    SourceFile,
    SourcePolicyError,
    SourceScan,
    normalize_metadata,
    scan_sources,
    sha256_file,
)
from .storage import (
    StorageError,
    atomic_write_json,
    atomic_write_jsonl,
    load_current_generation,
    load_metadata_overrides,
    load_source_exclusions,
    read_json,
    read_jsonl,
    write_metadata_overrides,
    write_source_exclusions,
)
from .ultrarag import VanillaUltraRAG

SCHEMA_VERSION = 5
CLEANING_POLICY_VERSION = 1
EXTRACTION_POLICY_VERSION = 4
ARTIFACT_POLICY_VERSION = 2
INGESTION_CHECKPOINT_VERSION = 1
DEFAULT_WORK_BUDGET_SECONDS = 45
MINIMUM_WORK_BUDGET_SECONDS = 10
MAXIMUM_WORK_BUDGET_SECONDS = 300
EMBEDDING_BATCH_SIZE = 64
QDRANT_BATCH_SIZE = 64
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


def _source_inventory(scan: SourceScan) -> list[dict[str, Any]]:
    return [
        {
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
    metadata_revision: str,
    exclusion_revision: str,
    baseline_generation_id: str | None,
    chunk_size: int,
    chunk_overlap: int,
    force_recompute: bool,
) -> str:
    return value_fingerprint(
        {
            "project_id": project_id,
            "inventory": inventory,
            "metadata_revision": metadata_revision,
            "exclusion_revision": exclusion_revision,
            "baseline_generation_id": baseline_generation_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "force_recompute": force_recompute,
            "generation_schema_version": SCHEMA_VERSION,
            "extraction_policy_version": EXTRACTION_POLICY_VERSION,
            "cleaning_policy_version": CLEANING_POLICY_VERSION,
            "artifact_policy_version": ARTIFACT_POLICY_VERSION,
            "retrieval_policy_fingerprint": RETRIEVAL_POLICY_FINGERPRINT,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_model_revision": EMBEDDING_MODEL_REVISION,
            "embedding_dimension": EMBEDDING_DIMENSION,
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
    provenance = dict(result.get("metadata_provenance") or {})
    warnings = list(result.get("metadata_warnings") or [])
    if provenance.get("title") != "reviewed_override" and text_corruption_reasons(
        result["title"]
    ):
        result["title"] = Path(str(result.get("source_path") or "source")).stem
        warnings.append("corrupt_extracted_title")
    if provenance.get("authors") != "reviewed_override":
        clean_authors = [
            author
            for author in result["authors"]
            if not text_corruption_reasons(author)
        ]
        if len(clean_authors) != len(result["authors"]):
            warnings.append("corrupt_extracted_authors")
        result["authors"] = clean_authors
    result["metadata_warnings"] = list(dict.fromkeys(warnings))
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

    def _load_ingestion_checkpoint(
        self,
    ) -> tuple[Path, dict[str, Any]] | None:
        candidates: list[tuple[Path, dict[str, Any]]] = []
        for root in sorted(self.config.staging_root.iterdir()):
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
            candidates.append((root, checkpoint))
        if not candidates:
            return None
        return max(candidates, key=lambda item: str(item[1].get("updated_at") or ""))

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
                unit = "pages_or_sections"
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
        elif phase == "qdrant_indexing":
            completed = int(checkpoint.get("qdrant_indexed_count") or 0)
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
        metadata_revision: str,
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
            metadata_revision=metadata_revision,
            exclusion_revision=exclusion_revision,
            baseline_generation_id=baseline_generation_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_recompute=force_recompute,
        )
        now = _utc_now()
        checkpoint: dict[str, Any] = {
            "schema_version": INGESTION_CHECKPOINT_VERSION,
            "build_id": build_id,
            "project_id": self.config.project_id,
            "identity": identity,
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
            "metadata_revision": metadata_revision,
            "source_exclusion_revision": exclusion_revision,
            "source_digests": {},
            "extracted_source_paths": [],
            "chunked_source_paths": [],
            "embedded_chunk_count": 0,
            "qdrant_indexed_count": 0,
            "revalidation_digests": {},
            "phase_timings_seconds": {},
        }
        root = self.config.staging_root / build_id
        root.mkdir(parents=False, exist_ok=False)
        self._write_checkpoint(root, checkpoint)
        return root, checkpoint

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
        checkpoint_state = self._load_ingestion_checkpoint()
        ingestion_progress = (
            self._ingestion_progress(checkpoint_state[1])
            if checkpoint_state is not None
            else None
        )
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
            "ingestion_progress": ingestion_progress,
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
                "A new generation is in progress; the selected generation remains "
                "searchable. Call ingest again with the same settings to continue."
                if ingestion_progress is not None
                else (
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
            os.replace(temporary, path)
        finally:
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
        metadata: dict[str, dict[str, Any]],
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
            )

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
                    metadata_revision=str(checkpoint["metadata_revision"]),
                    exclusion_revision=str(checkpoint["source_exclusion_revision"]),
                    retrieval_policy_fingerprint=RETRIEVAL_POLICY_FINGERPRINT,
                ):
                    manifest = snapshot.manifest
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
                        "phase_timings_seconds": checkpoint["phase_timings_seconds"],
                        "message": (
                            "Inputs match the selected generation; no build was needed."
                        ),
                    }
                checkpoint["phase"] = "extraction"
                self._write_checkpoint(staging_root, checkpoint)
                continue

            if phase == "extraction":
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
                override_revision = value_fingerprint(metadata.get(relative, {}))
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
                        previous_source is not None
                        and previous_document is not None
                        and previous_source.get("sha256")
                        == checkpoint["source_digests"][relative]
                        and previous_document.get("metadata_override_revision")
                        == override_revision
                        and snapshot is not None
                        and snapshot.units_by_document.get(document_id)
                        and snapshot.chunks_by_document.get(document_id)
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
                            "rejected_units": [],
                            "empty_units": 0,
                            "removed_repeated_margin_blocks": 0,
                            "discarded_empty_chunks": 0,
                        }
                        checkpoint["extraction_work_total"] = (
                            int(checkpoint.get("extraction_work_total") or 0)
                            + (total * 2)
                            + 2
                        )
                        atomic_write_json(state_path, state)
                    else:
                        document, total = await _atomic_to_thread(
                            prepare_epub_extraction,
                            source,
                            metadata.get(relative, {}),
                            checkpoint["source_digests"][relative],
                        )
                        document["metadata_override_revision"] = override_revision
                        atomic_write_json(artifact_root / "document.json", document)
                        state = {
                            "reused": False,
                            "extraction_stage": "epub_items",
                            "next_index": 0,
                            "total": total,
                            "rejected_units": [],
                            "empty_units": 0,
                            "discarded_empty_chunks": 0,
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
                if extraction_stage == "pdf_scan":
                    index = int(state["next_index"])
                    if index < int(state["total"]):
                        page_scan = await _atomic_to_thread(
                            scan_pdf_page, source, index
                        )
                        atomic_write_json(
                            artifact_root / "page-scans" / f"{index:08d}.json",
                            page_scan,
                        )
                        state["next_index"] = index + 1
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
                        metadata.get(relative, {}),
                        checkpoint["source_digests"][relative],
                        page_scans,
                    )
                    document["metadata_override_revision"] = override_revision
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
                        batch, empty, removed = await _atomic_to_thread(
                            extract_scanned_pdf_page,
                            source,
                            document,
                            read_json(
                                artifact_root / "page-scans" / f"{index:08d}.json"
                            ),
                            list(state.get("repeated_margins") or []),
                        )
                        state["removed_repeated_margin_blocks"] = (
                            int(state.get("removed_repeated_margin_blocks") or 0)
                            + removed
                        )
                    else:
                        batch, empty = await _atomic_to_thread(
                            extract_epub_spine_item,
                            source,
                            document,
                            index,
                        )
                    retained: list[dict[str, Any]] = []
                    for unit in batch:
                        reasons = text_corruption_reasons(
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
                        artifact_root / "unit-batches" / f"{index:08d}.jsonl",
                        retained,
                    )
                    state["empty_units"] = int(state.get("empty_units") or 0) + int(
                        empty
                    )
                    state["next_index"] = index + 1
                    checkpoint["extraction_work_completed"] = (
                        int(checkpoint.get("extraction_work_completed") or 0) + 1
                    )
                    atomic_write_json(state_path, state)
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
                            f"corrupt extraction units were excluded: {source.path}"
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
                units = read_jsonl(artifact_root / "units.jsonl")
                started = time.perf_counter()
                if bool(state.get("reused")) and snapshot is not None:
                    chunks = [
                        dict(item)
                        for item in snapshot.chunks_by_document[
                            str(document["document_id"])
                        ]
                    ]
                    discarded = 0
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
                        input_path = artifact_root / "chunking" / "unit.jsonl"
                        working_path = artifact_root / "raw-chunks.jsonl"
                        atomic_write_jsonl(input_path, [units[chunked_count]])
                        working_path.unlink(missing_ok=True)
                        await self.ultrarag.chunk(
                            input_path,
                            working_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        atomic_write_jsonl(
                            artifact_root / "chunking" / f"{chunked_count:08d}.jsonl",
                            read_jsonl(working_path),
                        )
                        working_path.unlink(missing_ok=True)
                        input_path.unlink(missing_ok=True)
                        state["chunked_unit_count"] = chunked_count + 1
                        checkpoint["chunking_work_completed"] = (
                            int(checkpoint.get("chunking_work_completed") or 0) + 1
                        )
                        atomic_write_json(artifact_root / "state.json", state)
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
                    chunks, discarded = _enrich_chunks(raw_chunks, units, [document])
                    checkpoint["rebuilt_chunk_count"] = int(
                        checkpoint.get("rebuilt_chunk_count") or 0
                    ) + len(chunks)
                    state["chunking_stage"] = "complete"
                atomic_write_jsonl(artifact_root / "chunks.jsonl", chunks)
                state["discarded_empty_chunks"] = discarded
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
                units: list[dict[str, Any]] = []
                chunks: list[dict[str, Any]] = []
                discarded_empty_chunks = 0
                for relative in selected_paths:
                    artifact_root = self._source_artifact_root(staging_root, relative)
                    document = read_json(artifact_root / "document.json")
                    state = read_json(artifact_root / "state.json")
                    if not isinstance(document, dict) or not isinstance(state, dict):
                        raise ResearchError("Invalid staged source artifacts")
                    documents.append(document)
                    units.extend(read_jsonl(artifact_root / "units.jsonl"))
                    chunks.extend(read_jsonl(artifact_root / "chunks.jsonl"))
                    discarded_empty_chunks += int(
                        state.get("discarded_empty_chunks") or 0
                    )
                extracted_path = staging_root / "corpus" / "extracted-units.jsonl"
                chunks_path = staging_root / "chunks" / "chunks.jsonl"
                atomic_write_jsonl(extracted_path, units)
                atomic_write_jsonl(chunks_path, chunks)
                checkpoint["document_count"] = len(documents)
                checkpoint["extraction_unit_count"] = len(units)
                checkpoint["chunk_count"] = len(chunks)
                checkpoint["discarded_empty_chunk_count"] = discarded_empty_chunks
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
                chunks = read_jsonl(staging_root / "chunks" / "chunks.jsonl")
                offset = int(checkpoint.get("embedded_chunk_count") or 0)
                if offset >= len(chunks):
                    checkpoint["phase"] = "vector_assembly"
                    self._write_checkpoint(staging_root, checkpoint)
                    continue
                batch = chunks[offset : offset + EMBEDDING_BATCH_SIZE]
                vectors = np.empty((len(batch), EMBEDDING_DIMENSION), dtype=np.float32)
                missing_positions: list[int] = []
                missing_texts: list[str] = []
                reusable_vectors = (
                    snapshot.vectors_by_text
                    if snapshot is not None and not force_recompute
                    else {}
                )
                reused_count = 0
                for index, chunk in enumerate(batch):
                    reusable_vector = reusable_vectors.get(str(chunk["embedding_text"]))
                    if reusable_vector is None:
                        missing_positions.append(index)
                        missing_texts.append(str(chunk["embedding_text"]))
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
                        EMBEDDING_DIMENSION,
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
                vectors = np.empty((total, EMBEDDING_DIMENSION), dtype=np.float32)
                for offset in range(0, total, EMBEDDING_BATCH_SIZE):
                    batch = np.load(
                        staging_root / "work" / "vector-batches" / f"{offset:012d}.npy",
                        allow_pickle=False,
                    )
                    vectors[offset : offset + len(batch)] = batch
                vectors_path = staging_root / "portable" / "embeddings.npy"
                self._save_vector_batch(vectors_path, vectors)
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
                await self.ultrarag.build_bm25(
                    staging_root / "chunks" / "chunks.jsonl",
                    bm25_index_path,
                )
                self._add_phase_time(
                    checkpoint,
                    "bm25_indexing",
                    time.perf_counter() - started,
                )
                checkpoint["phase"] = "qdrant_indexing"
                checkpoint["qdrant_indexed_count"] = 0
                self._write_checkpoint(staging_root, checkpoint)
                if budget_expired():
                    return self._in_progress_result(checkpoint)
                continue

            if phase == "qdrant_indexing":
                chunks = read_jsonl(staging_root / "chunks" / "chunks.jsonl")
                vectors = np.load(
                    staging_root / "portable" / "embeddings.npy",
                    allow_pickle=False,
                )
                dense_index_path = staging_root / "indexes" / "qdrant"
                offset = int(checkpoint.get("qdrant_indexed_count") or 0)
                if offset == 0:
                    shutil.rmtree(dense_index_path, ignore_errors=True)
                    await _atomic_to_thread(
                        self.dense.initialize_index,
                        dense_index_path,
                        EMBEDDING_DIMENSION,
                    )
                if offset < len(chunks):
                    batch = chunks[offset : offset + QDRANT_BATCH_SIZE]
                    started = time.perf_counter()
                    await _atomic_to_thread(
                        self.dense.upload_index_batch,
                        batch,
                        dense_index_path,
                        vectors[offset : offset + len(batch)],
                        offset=offset,
                    )
                    self._add_phase_time(
                        checkpoint,
                        "qdrant_indexing",
                        time.perf_counter() - started,
                    )
                    checkpoint["qdrant_indexed_count"] = offset + len(batch)
                    self._write_checkpoint(staging_root, checkpoint)
                    if budget_expired():
                        return self._in_progress_result(checkpoint)
                    continue
                dense_metadata = await _atomic_to_thread(
                    self.dense.finalize_index,
                    dense_index_path,
                    expected_count=len(chunks),
                    dimension=EMBEDDING_DIMENSION,
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
                        revalidated[
                            source.source_relative_path
                        ] = await _atomic_to_thread(sha256_file, source.path)
                    except OSError as exc:
                        raise _SourceChangedDuringIngest(
                            "A source became unavailable during final validation"
                        ) from exc
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
                checkpoint["phase"] = "finalizing"
                self._write_checkpoint(staging_root, checkpoint)
                continue

            if phase == "finalizing":
                documents = [
                    read_json(
                        self._source_artifact_root(staging_root, relative)
                        / "document.json"
                    )
                    for relative in selected_paths
                ]
                units = read_jsonl(staging_root / "corpus" / "extracted-units.jsonl")
                chunks = read_jsonl(staging_root / "chunks" / "chunks.jsonl")
                content_kind_counts = dict(
                    sorted(
                        Counter(str(item["content_kind"]) for item in chunks).items()
                    )
                )
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
                    "metadata_revision": checkpoint["metadata_revision"],
                    "source_exclusion_revision": checkpoint[
                        "source_exclusion_revision"
                    ],
                    "retrieval_policy_fingerprint": RETRIEVAL_POLICY_FINGERPRINT,
                    "source_file_count": len(scan.selected),
                    "excluded_source_count": len(exclusions),
                    "document_count": len(documents),
                    "extraction_unit_count": len(units),
                    "chunk_count": len(chunks),
                    "content_kind_counts": content_kind_counts,
                    "discarded_empty_chunk_count": int(
                        checkpoint.get("discarded_empty_chunk_count") or 0
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
                    "source_files": records,
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
                generation_root = self.config.generations_root / str(
                    checkpoint["build_id"]
                )
                shutil.rmtree(staging_root / "work", ignore_errors=True)
                atomic_write_json(staging_root / "manifest.json", manifest)
                os.replace(staging_root, generation_root)
                (generation_root / "checkpoint.json").unlink()
                atomic_write_json(
                    self.config.current_path,
                    {
                        "schema_version": 1,
                        "generation_id": checkpoint["build_id"],
                    },
                )
                self._loaded_generation = str(checkpoint["build_id"])
                return {
                    "status": "ready",
                    "generation_changed": True,
                    "generation_id": checkpoint["build_id"],
                    "generation_root": str(generation_root),
                    "source_file_count": len(scan.selected),
                    "excluded_source_count": len(exclusions),
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
                    "discarded_empty_chunk_count": int(
                        checkpoint.get("discarded_empty_chunk_count") or 0
                    ),
                    "excluded_corrupt_unit_count": int(
                        checkpoint.get("excluded_corrupt_unit_count") or 0
                    ),
                    "ignored_extensions": scan.ignored_extensions,
                    "empty_units": sum(int(item["empty_units"]) for item in documents),
                    "default_retrieval_method": DEFAULT_RETRIEVAL_METHOD,
                    "available_retrieval_methods": sorted(RETRIEVAL_METHODS),
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding_model_revision": EMBEDDING_MODEL_REVISION,
                    **build_metrics,
                }

            raise ResearchError(f"Unsupported ingestion checkpoint phase: {phase}")

    async def ingest(
        self,
        *,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        force_recompute: bool = False,
        work_budget_seconds: int = DEFAULT_WORK_BUDGET_SECONDS,
    ) -> dict[str, Any]:
        if not 50 <= chunk_size <= 384:
            raise ResearchError("chunk_size must be between 50 and 384 GPT-2 tokens")
        if not 0 <= chunk_overlap < chunk_size:
            raise ResearchError(
                "chunk_overlap must be non-negative and below chunk_size"
            )
        if (
            not MINIMUM_WORK_BUDGET_SECONDS
            <= work_budget_seconds
            <= (MAXIMUM_WORK_BUDGET_SECONDS)
        ):
            raise ResearchError("work_budget_seconds must be between 10 and 300")

        async with self._operation():
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
            current = self._load_current_optional()
            baseline_generation_id = (
                str(current[1]["generation_id"]) if current is not None else None
            )
            metadata_revision = value_fingerprint(metadata)
            exclusion_revision = value_fingerprint(exclusions)
            inventory = _source_inventory(scan)
            identity = _checkpoint_identity(
                project_id=self.config.project_id,
                inventory=inventory,
                metadata_revision=metadata_revision,
                exclusion_revision=exclusion_revision,
                baseline_generation_id=baseline_generation_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                force_recompute=force_recompute,
            )
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
                    metadata_revision=metadata_revision,
                    exclusion_revision=exclusion_revision,
                    baseline_generation_id=baseline_generation_id,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    force_recompute=force_recompute,
                )
            else:
                staging_root, checkpoint = existing
                checkpoint["resume_count"] = (
                    int(checkpoint.get("resume_count") or 0) + 1
                )
                self._write_checkpoint(staging_root, checkpoint)

            deadline = time.perf_counter() + work_budget_seconds
            try:
                return await self._advance_ingestion(
                    staging_root=staging_root,
                    checkpoint=checkpoint,
                    scan=scan,
                    metadata=metadata,
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
                fresh_root, fresh = self._create_checkpoint(
                    scan=fresh_scan,
                    exclusions=fresh_exclusions,
                    metadata_revision=value_fingerprint(self._metadata()),
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
                self._discard_checkpoint(
                    staging_root,
                    checkpoint,
                    reason=str(exc),
                )
                raise

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
            return [], {
                "no_query_token_overlap": 0,
                "extraction_artifact": 0,
                "corrupt_text": 0,
            }
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
            if text_corruption_reasons(str(chunk.get("text") or "")):
                rejected["corrupt_text"] += 1
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
                "corrupt_text": 0,
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
            dense_corrupt_text_rejected = 0
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
                if text_corruption_reasons(str(chunk.get("text") or "")):
                    dense_corrupt_text_rejected += 1
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
                    "bm25_corrupt_text": bm25_rejected["corrupt_text"],
                    "dense_below_threshold": dense_below_threshold,
                    "dense_extraction_artifact": dense_quality_rejected,
                    "dense_corrupt_text": dense_corrupt_text_rejected,
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
            if text_corruption_reasons(str(target.get("text") or "")):
                raise ResearchError(
                    "The requested chunk contains corrupt extracted text and is "
                    "not available; re-ingest to remove it from the generation"
                )
            same_document = sorted(
                (
                    item
                    for item in chunks
                    if item.get("document_id") == target["document_id"]
                    and not text_corruption_reasons(str(item.get("text") or ""))
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

"""High-level, project-scoped research knowledge-base workflow."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import AsyncFileLock
from filelock import Timeout as FileLockTimeout

from .config import ResearchConfig, resolve_source_reference
from .dense import (
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
    RERANKER_MODEL,
    RERANKER_MODEL_REVISION,
    DenseBackend,
    DenseSearchHit,
    LocalQdrantDenseBackend,
)
from .extraction import extract_sources
from .sources import (
    ALLOWED_SOURCE_EXTENSIONS,
    SourcePolicyError,
    SourceScan,
    metadata_revision,
    normalize_metadata,
    scan_sources,
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

SCHEMA_VERSION = 3
DEFAULT_RETRIEVAL_METHOD = "hybrid"
RETRIEVAL_METHODS = frozenset({"bm25", "dense", "hybrid"})
RRF_K = 60
MINIMUM_CANDIDATES = 20
MAXIMUM_CANDIDATES = 200
RERANK_MAX_CANDIDATES = 50


class ResearchError(RuntimeError):
    """User-facing research workflow failure."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _generation_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _source_exclusion_revision(
    exclusions: dict[str, dict[str, str]],
) -> str:
    encoded = json.dumps(
        exclusions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _index_text(
    document: dict[str, Any],
    locator: dict[str, Any],
    chunk_id: str,
    text: str,
) -> str:
    metadata_lines = [
        f"Title: {document['title']}",
        f"Source: {document['source_path']}",
        f"Locator: {_citation(document, locator)}",
    ]
    if document.get("authors"):
        metadata_lines.append(f"Authors: {'; '.join(document['authors'])}")
    if document.get("categories"):
        metadata_lines.append(f"Categories: {'; '.join(document['categories'])}")
    if document.get("keywords"):
        metadata_lines.append(f"Keywords: {'; '.join(document['keywords'])}")
    metadata_lines.append(f"Chunk reference: {chunk_id}")
    return "\n".join(metadata_lines) + f"\n\nContent:\n{text}"


def _embedding_text(document: dict[str, Any], text: str) -> str:
    """Build semantic input without injecting locators or internal identifiers."""

    metadata_lines = [f"Title: {document['title']}"]
    if document.get("authors"):
        metadata_lines.append(f"Authors: {'; '.join(document['authors'])}")
    if document.get("categories"):
        metadata_lines.append(f"Categories: {'; '.join(document['categories'])}")
    if document.get("keywords"):
        metadata_lines.append(f"Keywords: {'; '.join(document['keywords'])}")
    return "\n".join(metadata_lines) + f"\n\nContent:\n{text}"


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
        text = str(raw.get("contents") or "").strip()
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
                "contents": _index_text(document, locator, chunk_id, text),
                "embedding_text": _embedding_text(document, text),
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
        exclusion_revision = _source_exclusion_revision(exclusions)
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
                "source_root": str(self.config.source_root),
                "state_root": str(self.config.state_root),
                "discovered_source_count": len(scan.selected),
                "selected_source_count": len(selected),
                "excluded_source_count": len(exclusions),
                "excluded_sources": self._exclusion_records(scan, exclusions),
                "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
                "ignored_extensions": scan.ignored_extensions,
                "source_exclusion_revision": exclusion_revision,
                "default_retrieval_method": DEFAULT_RETRIEVAL_METHOD,
                "available_retrieval_methods": [],
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
        modified = sorted(
            relative
            for relative in set(scanned) & set(current_documents)
            if scanned[relative].size != current_documents[relative].get("size")
            or scanned[relative].mtime_ns != current_documents[relative].get("mtime_ns")
        )
        metadata_changed = metadata_revision(self._metadata()) != manifest.get(
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
        excluded_document_ids = self._excluded_document_ids(manifest, exclusions)
        exclusion_records = self._exclusion_records(scan, exclusions, manifest)
        return {
            "ready": True,
            "stale": stale,
            "project_root": str(self.config.project_root),
            "source_root": str(self.config.source_root),
            "state_root": str(self.config.state_root),
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
            "retrieval": retrieval,
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
                    "Current generation supports hybrid retrieval."
                    if hybrid_ready
                    else "Current generation is BM25-only; run ingest to build its "
                    "project-local dense index."
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
    ) -> dict[str, Any]:
        if not 50 <= chunk_size <= 384:
            raise ResearchError("chunk_size must be between 50 and 384 GPT-2 tokens")
        if not 0 <= chunk_overlap < chunk_size:
            raise ResearchError(
                "chunk_overlap must be non-negative and below chunk_size"
            )

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
            documents, units = await asyncio.to_thread(
                extract_sources,
                selected,
                metadata,
            )
            generation_id = _generation_id()
            generation_root = self.config.generations_root / generation_id
            generation_root.mkdir(parents=False, exist_ok=False)
            extracted_path = generation_root / "corpus" / "extracted-units.jsonl"
            raw_chunks_path = generation_root / "chunks" / "ultrarag-chunks.jsonl"
            chunks_path = generation_root / "chunks" / "chunks.jsonl"
            bm25_index_path = generation_root / "indexes" / "bm25"
            dense_index_path = generation_root / "indexes" / "qdrant"

            try:
                write_jsonl(extracted_path, units)
                await self.ultrarag.chunk(
                    extracted_path,
                    raw_chunks_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                raw_chunks = read_jsonl(raw_chunks_path)
                chunks, discarded_empty_chunks = _enrich_chunks(
                    raw_chunks,
                    units,
                    documents,
                )
                write_jsonl(chunks_path, chunks)

                await self.ultrarag.build_bm25(chunks_path, bm25_index_path)
                dense_metadata = await asyncio.to_thread(
                    self.dense.build,
                    chunks,
                    dense_index_path,
                )
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "generation_id": generation_id,
                    "created_at": _utc_now(),
                    "source_directory": self.config.source_root.relative_to(
                        self.config.project_root
                    ).as_posix(),
                    "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
                    "ignored_extensions": scan.ignored_extensions,
                    "metadata_revision": metadata_revision(metadata),
                    "source_exclusion_revision": _source_exclusion_revision(exclusions),
                    "source_file_count": len(scan.selected),
                    "excluded_source_count": sum(
                        source.source_relative_path in exclusions
                        for source in scan.selected
                    ),
                    "document_count": len(documents),
                    "extraction_unit_count": len(units),
                    "chunk_count": len(chunks),
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
                            "method": "reciprocal_rank_fusion",
                            "rrf_k": RRF_K,
                            "minimum_candidates": MINIMUM_CANDIDATES,
                            "maximum_candidates": MAXIMUM_CANDIDATES,
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
                    "source_files": [
                        {
                            "source_path": source.project_relative_path,
                            "source_relative_path": source.source_relative_path,
                            "format": source.extension.removeprefix("."),
                            "size": source.size,
                            "mtime_ns": source.mtime_ns,
                            "included": source.source_relative_path not in exclusions,
                        }
                        for source in scan.selected
                    ],
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
                        "raw_ultrarag_chunks": "chunks/ultrarag-chunks.jsonl",
                        "chunks": "chunks/chunks.jsonl",
                        "bm25_index": "indexes/bm25",
                        "dense_index": "indexes/qdrant",
                    },
                }
                atomic_write_json(generation_root / "manifest.json", manifest)
                atomic_write_json(
                    self.config.current_path,
                    {
                        # Pointer format is unchanged; manifest schema is version 3.
                        "schema_version": 1,
                        "generation_id": generation_id,
                    },
                )
                self._loaded_generation = generation_id
            except Exception as exc:
                atomic_write_json(
                    generation_root / "failure.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "generation_id": generation_id,
                        "failed_at": _utc_now(),
                        "error": str(exc),
                    },
                )
                raise

            return {
                "status": "ready",
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
                "discarded_empty_chunk_count": discarded_empty_chunks,
                "ignored_extensions": scan.ignored_extensions,
                "empty_units": sum(int(item["empty_units"]) for item in documents),
                "default_retrieval_method": DEFAULT_RETRIEVAL_METHOD,
                "available_retrieval_methods": sorted(RETRIEVAL_METHODS),
                "embedding_model": EMBEDDING_MODEL,
                "embedding_model_revision": EMBEDDING_MODEL_REVISION,
            }

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
    ) -> list[str]:
        if limit <= 0:
            return []
        filtered = bool(categories or keywords or document_ids or excluded_document_ids)
        requested = len(chunks) if filtered else limit
        passages = await self.ultrarag.search_bm25(query, requested)
        by_contents = {str(item["contents"]): item for item in chunks}
        ranking: list[str] = []
        for passage in passages:
            chunk = by_contents.get(passage)
            if chunk is None:
                raise ResearchError(
                    "UltraRAG returned a passage absent from the current chunk store"
                )
            if not self._matches_filters(
                chunk,
                categories=categories,
                keywords=keywords,
                document_ids=document_ids,
                excluded_document_ids=excluded_document_ids,
            ):
                continue
            ranking.append(str(chunk["chunk_id"]))
            if len(ranking) == limit:
                break
        return ranking

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
        for ranks in (bm25_ranks, dense_ranks):
            for chunk_id, rank in ranks.items():
                scores[chunk_id] += 1.0 / (RRF_K + rank)
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
                bm25_ranking, dense_hits = await asyncio.gather(
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
            elif use_bm25:
                bm25_ranking = await self._bm25_ranking(
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

            chunks_by_id = {str(item["chunk_id"]): item for item in chunks}
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
                component_ranks = {
                    "bm25": bm25_ranks.get(chunk_id),
                    "dense": dense_ranks.get(chunk_id),
                }
                hits.append(
                    {
                        "rank": rank,
                        "retrieval_rank": base_ranks[chunk_id],
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "title": chunk["title"],
                        "authors": chunk["authors"],
                        "year": chunk["year"],
                        "doi": chunk["doi"],
                        "source_path": chunk["source_path"],
                        "categories": chunk["categories"],
                        "keywords": chunk["keywords"],
                        "locator": chunk["locator"],
                        "citation": chunk["citation"],
                        "text": chunk["text"],
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
                "excluded_source_count": len(exclusions),
                "retrieval_method": retrieval_method,
                "reranked": rerank,
                "candidate_depth": candidate_depth,
                "fusion": (
                    {"method": "reciprocal_rank_fusion", "rrf_k": RRF_K}
                    if retrieval_method == "hybrid"
                    else None
                ),
                "embedding_model": EMBEDDING_MODEL if use_dense else None,
                "embedding_model_revision": (
                    EMBEDDING_MODEL_REVISION if use_dense else None
                ),
                "reranker_model": RERANKER_MODEL if rerank else None,
                "reranker_model_revision": (
                    RERANKER_MODEL_REVISION if rerank else None
                ),
                "result_count": len(hits),
                "hits": hits,
                "notice": (
                    "Page and section locators come from text extraction. Verify "
                    "important quotations against the original source."
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
                sources.append(document)
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
                    key: item[key]
                    for key in (
                        "chunk_id",
                        "document_id",
                        "source_path",
                        "title",
                        "locator",
                        "citation",
                        "text",
                    )
                }
                for item in same_document[start:end]
            ]
            return {
                "generation_id": manifest["generation_id"],
                "requested_chunk_id": chunk_id,
                "context": context,
                "notice": "Verify important quotations against the original source.",
            }

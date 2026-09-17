"""High-level, project-scoped research knowledge-base workflow."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ResearchConfig, resolve_source_reference
from .extraction import extract_sources
from .sources import (
    ALLOWED_SOURCE_EXTENSIONS,
    SourcePolicyError,
    metadata_revision,
    normalize_metadata,
    scan_sources,
)
from .storage import (
    StorageError,
    atomic_write_json,
    load_current_generation,
    load_metadata_overrides,
    read_jsonl,
    write_jsonl,
    write_metadata_overrides,
)
from .ultrarag import VanillaUltraRAG

SCHEMA_VERSION = 1


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
    def __init__(self, config: ResearchConfig, ultrarag: VanillaUltraRAG) -> None:
        self.config = config
        self.ultrarag = ultrarag
        self._lock = asyncio.Lock()
        self._loaded_generation: str | None = None

    def _metadata(self) -> dict[str, dict[str, Any]]:
        try:
            values = load_metadata_overrides(self.config.metadata_path)
            return {key: normalize_metadata(value) for key, value in values.items()}
        except (StorageError, SourcePolicyError) as exc:
            raise ResearchError(str(exc)) from exc

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
        current = self._load_current_optional()
        if current is None:
            return {
                "ready": False,
                "stale": bool(scan.selected),
                "project_root": str(self.config.project_root),
                "source_root": str(self.config.source_root),
                "state_root": str(self.config.state_root),
                "selected_source_count": len(scan.selected),
                "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
                "ignored_extensions": scan.ignored_extensions,
                "message": "No knowledge-base generation exists; call ingest.",
            }

        generation_root, manifest = current
        current_documents = {
            str(item["source_relative_path"]): item
            for item in manifest.get("documents", [])
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
        stale = bool(added or removed or modified or metadata_changed)
        return {
            "ready": True,
            "stale": stale,
            "project_root": str(self.config.project_root),
            "source_root": str(self.config.source_root),
            "state_root": str(self.config.state_root),
            "generation_id": manifest["generation_id"],
            "created_at": manifest["created_at"],
            "selected_source_count": len(scan.selected),
            "indexed_source_count": manifest["document_count"],
            "chunk_count": manifest["chunk_count"],
            "allowed_formats": sorted(ALLOWED_SOURCE_EXTENSIONS),
            "ignored_extensions": scan.ignored_extensions,
            "changes": {
                "added": added,
                "removed": removed,
                "modified": modified,
                "metadata_changed": metadata_changed,
            },
            "generation_root": str(generation_root),
        }

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(self._status)

    async def set_source_metadata(
        self,
        source_path: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        async with self._lock:
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

    async def ingest(
        self,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 64,
    ) -> dict[str, Any]:
        if not 50 <= chunk_size <= 4000:
            raise ResearchError("chunk_size must be between 50 and 4000 words")
        if not 0 <= chunk_overlap < chunk_size:
            raise ResearchError(
                "chunk_overlap must be non-negative and below chunk_size"
            )

        async with self._lock:
            try:
                scan = scan_sources(self.config)
            except SourcePolicyError as exc:
                raise ResearchError(str(exc)) from exc
            if not scan.selected:
                raise ResearchError(
                    f"No PDF or EPUB sources found beneath {self.config.source_root}"
                )

            metadata = self._metadata()
            documents, units = await asyncio.to_thread(
                extract_sources,
                scan.selected,
                metadata,
            )
            generation_id = _generation_id()
            generation_root = self.config.generations_root / generation_id
            generation_root.mkdir(parents=False, exist_ok=False)
            extracted_path = generation_root / "corpus" / "extracted-units.jsonl"
            raw_chunks_path = generation_root / "chunks" / "ultrarag-chunks.jsonl"
            chunks_path = generation_root / "chunks" / "chunks.jsonl"
            index_path = generation_root / "indexes" / "bm25"

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

                await self.ultrarag.build_bm25(chunks_path, index_path)
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
                    "document_count": len(documents),
                    "extraction_unit_count": len(units),
                    "chunk_count": len(chunks),
                    "discarded_empty_chunk_count": discarded_empty_chunks,
                    "chunking": {
                        "backend": "UltraRAG token chunker",
                        "counter": "word",
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    },
                    "retrieval": {
                        "backend": "UltraRAG BM25",
                        "language": "en",
                        "tokenizer": "default",
                    },
                    "documents": documents,
                    "files": {
                        "extracted_units": "corpus/extracted-units.jsonl",
                        "raw_ultrarag_chunks": "chunks/ultrarag-chunks.jsonl",
                        "chunks": "chunks/chunks.jsonl",
                        "bm25_index": "indexes/bm25",
                    },
                }
                atomic_write_json(generation_root / "manifest.json", manifest)
                atomic_write_json(
                    self.config.current_path,
                    {
                        "schema_version": SCHEMA_VERSION,
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
                "document_count": len(documents),
                "pdf_count": sum(item["format"] == "pdf" for item in documents),
                "epub_count": sum(item["format"] == "epub" for item in documents),
                "extraction_unit_count": len(units),
                "chunk_count": len(chunks),
                "discarded_empty_chunk_count": discarded_empty_chunks,
                "ignored_extensions": scan.ignored_extensions,
                "empty_units": sum(int(item["empty_units"]) for item in documents),
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

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ResearchError("query must not be empty")
        if not 1 <= top_k <= 50:
            raise ResearchError("top_k must be between 1 and 50")

        async with self._lock:
            current = self._load_current_optional()
            if current is None:
                raise ResearchError("No knowledge base exists; call ingest first")
            generation_root, manifest = current
            await self._ensure_loaded(generation_root, manifest)
            chunks = read_jsonl(generation_root / manifest["files"]["chunks"])
            if not chunks:
                raise ResearchError("The current generation has no chunks")

            category_filter = {item.casefold() for item in categories or []}
            keyword_filter = {item.casefold() for item in keywords or []}
            document_filter = set(document_ids or [])
            filtered = bool(category_filter or keyword_filter or document_filter)
            candidate_k = len(chunks) if filtered else min(top_k, len(chunks))
            passages = await self.ultrarag.search_bm25(query, candidate_k)
            by_contents = {str(item["contents"]): item for item in chunks}

            hits: list[dict[str, Any]] = []
            for retrieval_rank, passage in enumerate(passages, 1):
                chunk = by_contents.get(passage)
                if chunk is None:
                    raise ResearchError(
                        "UltraRAG returned a passage absent from the current chunk store"
                    )
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
                hits.append(
                    {
                        "rank": len(hits) + 1,
                        "retrieval_rank": retrieval_rank,
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
                        "retrieval_method": "bm25",
                    }
                )
                if len(hits) == top_k:
                    break

            status = await asyncio.to_thread(self._status)
            return {
                "query": query,
                "generation_id": manifest["generation_id"],
                "stale": status["stale"],
                "retrieval_method": "bm25",
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
        async with self._lock:
            current = self._load_current_optional()
            if current is None:
                return {"ready": False, "source_count": 0, "sources": []}
            _generation_root, manifest = current
            category_filter = {item.casefold() for item in categories or []}
            keyword_filter = {item.casefold() for item in keywords or []}
            sources = []
            for document in manifest["documents"]:
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
            }

    async def get_passage(
        self,
        chunk_id: str,
        *,
        context_chunks: int = 1,
    ) -> dict[str, Any]:
        if not 0 <= context_chunks <= 5:
            raise ResearchError("context_chunks must be between 0 and 5")
        async with self._lock:
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

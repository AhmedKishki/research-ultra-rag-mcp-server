"""Compatibility checks and reusable state for immutable generations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dense import EMBEDDING_DIMENSION, EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION
from .storage import StorageError, read_jsonl


def value_fingerprint(value: Any) -> str:
    """Return a stable fingerprint for one JSON-compatible policy value."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generation_is_reusable(
    manifest: dict[str, Any],
    *,
    schema_version: int,
    extraction_policy_version: int,
    cleaning_policy_version: int,
    artifact_policy_version: int,
    project_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> bool:
    """Require exact processing and model compatibility before any reuse."""

    chunking = manifest.get("chunking", {})
    dense = manifest.get("retrieval", {}).get("dense", {})
    return bool(
        manifest.get("schema_version") == schema_version
        and manifest.get("extraction_policy_version") == extraction_policy_version
        and manifest.get("cleaning_policy_version") == cleaning_policy_version
        and manifest.get("artifact_policy_version") == artifact_policy_version
        and manifest.get("project_id") == project_id
        and chunking.get("backend") == "UltraRAG token chunker"
        and chunking.get("tokenizer") == "gpt2"
        and chunking.get("chunk_size") == chunk_size
        and chunking.get("chunk_overlap") == chunk_overlap
        and dense.get("embedding_model") == EMBEDDING_MODEL
        and dense.get("embedding_model_revision") == EMBEDDING_MODEL_REVISION
        and dense.get("embedding_dimension") == EMBEDDING_DIMENSION
    )


@dataclass(frozen=True, slots=True)
class ReuseSnapshot:
    """Validated reusable records from the selected generation."""

    root: Path
    manifest: dict[str, Any]
    source_files: dict[str, dict[str, Any]]
    documents: dict[str, dict[str, Any]]
    units_by_document: dict[str, list[dict[str, Any]]]
    chunks_by_document: dict[str, list[dict[str, Any]]]
    vectors_by_text: dict[str, np.ndarray[Any, np.dtype[np.float32]]]


def load_reuse_snapshot(
    current: tuple[Path, dict[str, Any]] | None,
    *,
    schema_version: int,
    extraction_policy_version: int,
    cleaning_policy_version: int,
    artifact_policy_version: int,
    project_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> ReuseSnapshot | None:
    """Load reusable records, or return None for any incompatible/corrupt state."""

    if current is None:
        return None
    root, manifest = current
    if not generation_is_reusable(
        manifest,
        schema_version=schema_version,
        extraction_policy_version=extraction_policy_version,
        cleaning_policy_version=cleaning_policy_version,
        artifact_policy_version=artifact_policy_version,
        project_id=project_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ):
        return None
    try:
        files = manifest["files"]
        units = read_jsonl(root / str(files["extracted_units"]))
        chunks = read_jsonl(root / str(files["chunks"]))
        vectors = np.load(root / str(files["portable_embeddings"]), allow_pickle=False)
    except (KeyError, OSError, ValueError, StorageError):
        return None
    if (
        vectors.dtype != np.float32
        or vectors.shape != (len(chunks), EMBEDDING_DIMENSION)
        or not np.isfinite(vectors).all()
    ):
        return None

    source_files = {
        str(item.get("source_relative_path")): item
        for item in manifest.get("source_files", [])
        if isinstance(item, dict) and item.get("source_relative_path")
    }
    documents = {
        str(item.get("source_relative_path")): item
        for item in manifest.get("documents", [])
        if isinstance(item, dict) and item.get("source_relative_path")
    }
    units_by_document: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        units_by_document.setdefault(str(unit.get("document_id")), []).append(unit)
    chunks_by_document: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        chunks_by_document.setdefault(str(chunk.get("document_id")), []).append(chunk)

    vectors_by_text: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
    for chunk, vector in zip(chunks, vectors, strict=True):
        text = str(chunk.get("embedding_text") or "")
        if text:
            vectors_by_text.setdefault(text, vector)
    return ReuseSnapshot(
        root=root,
        manifest=manifest,
        source_files=source_files,
        documents=documents,
        units_by_document=units_by_document,
        chunks_by_document=chunks_by_document,
        vectors_by_text=vectors_by_text,
    )


def source_set_matches(
    snapshot: ReuseSnapshot,
    source_records: list[dict[str, Any]],
    *,
    metadata_revision: str,
    exclusion_revision: str,
    retrieval_policy_fingerprint: str,
) -> bool:
    """Compare all source bytes and generation-affecting portable state."""

    if snapshot.manifest.get("metadata_revision") != metadata_revision:
        return False
    if snapshot.manifest.get("source_exclusion_revision") != exclusion_revision:
        return False
    if (
        snapshot.manifest.get("retrieval_policy_fingerprint")
        != retrieval_policy_fingerprint
    ):
        return False
    files = snapshot.manifest.get("files", {})
    for field in ("bm25_index", "dense_index"):
        relative = files.get(field)
        if not isinstance(relative, str) or not relative:
            return False
        if not (snapshot.root / relative).is_dir():
            return False
    expected = {
        str(item["source_relative_path"]): (
            str(item["sha256"]),
            bool(item["included"]),
        )
        for item in source_records
    }
    existing = {
        path: (str(item.get("sha256") or ""), bool(item.get("included")))
        for path, item in snapshot.source_files.items()
    }
    return expected == existing

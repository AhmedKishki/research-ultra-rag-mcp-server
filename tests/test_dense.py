from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research_ultra_rag_mcp.dense import (
    DenseTokenAuditUnavailable,
    LocalQdrantDenseBackend,
)


def _chunks(count: int) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": f"chunk-{index}",
            "document_id": "document-1",
            "source_path": "sources/evidence.pdf",
            "categories": [],
            "keywords": [],
        }
        for index in range(count)
    ]


def test_embedding_token_audit_reports_a_missing_model_cache(
    tmp_path: Path,
) -> None:
    # The audit must fail loudly to its caller instead of breaking ingestion, so
    # a missing or offline model cache raises a dedicated error the service
    # degrades from rather than an unrelated exception.
    backend = LocalQdrantDenseBackend(tmp_path / "models", offline=True)

    with pytest.raises(DenseTokenAuditUnavailable):
        backend.embedding_token_counts(["some evidence text"])


def test_embedding_token_audit_accepts_an_empty_batch(tmp_path: Path) -> None:
    backend = LocalQdrantDenseBackend(tmp_path / "models", offline=True)

    assert backend.embedding_token_counts([]) == []


def test_local_qdrant_batches_resume_without_skips_or_duplicates(
    tmp_path: Path,
) -> None:
    chunks = _chunks(130)
    vectors = np.ones((130, 384), dtype=np.float32)
    index_path = tmp_path / "qdrant"
    first_process = LocalQdrantDenseBackend(tmp_path / "models", offline=True)
    first_process.initialize_index(index_path, 384)
    first_process.upload_index_batch(
        chunks[:64],
        index_path,
        vectors[:64],
        offset=0,
    )

    resumed_process = LocalQdrantDenseBackend(tmp_path / "models", offline=True)
    resumed_process.upload_index_batch(
        chunks[64:128],
        index_path,
        vectors[64:128],
        offset=64,
    )
    resumed_process.upload_index_batch(
        chunks[128:],
        index_path,
        vectors[128:],
        offset=128,
    )
    # Replaying a committed-but-not-checkpointed batch is an idempotent upsert.
    resumed_process.upload_index_batch(
        chunks[64:128],
        index_path,
        vectors[64:128],
        offset=64,
    )

    metadata = resumed_process.finalize_index(
        index_path,
        expected_count=130,
        dimension=384,
    )
    assert metadata["point_count"] == 130

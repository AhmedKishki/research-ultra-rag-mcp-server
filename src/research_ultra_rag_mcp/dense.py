"""Project-local dense retrieval and optional CPU reranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_REVISION = "52398278842ec682c6f32300af41344b1c0b0bb2"
EMBEDDING_DIMENSION = 384
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_REVISION = "a09144355adeed5f58c8ed011d209bf8ee5a1fec"
COLLECTION_NAME = "research_chunks"


@dataclass(frozen=True, slots=True)
class DenseSearchHit:
    """One scored Qdrant result, identified by the canonical chunk ID."""

    chunk_id: str
    score: float


class DenseBackend(Protocol):
    """Boundary used by the service and deterministic test doubles."""

    def build(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors_path: Path | None = None,
    ) -> dict[str, Any]: ...

    def build_from_vectors(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors_path: Path,
    ) -> dict[str, Any]: ...

    def embed_texts(
        self, texts: list[str]
    ) -> np.ndarray[Any, np.dtype[np.float32]]: ...

    def build_index(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors: np.ndarray[Any, np.dtype[np.float32]],
    ) -> dict[str, Any]: ...

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
    ) -> list[DenseSearchHit]: ...

    def rerank(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]: ...


class LocalQdrantDenseBackend:
    """FastEmbed CPU vectors stored in an embedded, project-local Qdrant DB."""

    def __init__(self, model_cache_root: Path, *, offline: bool = False) -> None:
        self.model_cache_root = model_cache_root
        self.offline = offline
        self._embedding_model: TextEmbedding | None = None
        self._reranker: TextCrossEncoder | None = None

    def _embedder(self) -> TextEmbedding:
        if self._embedding_model is None:
            self.model_cache_root.mkdir(parents=True, exist_ok=True)
            self._embedding_model = TextEmbedding(
                model_name=EMBEDDING_MODEL,
                cache_dir=str(self.model_cache_root),
                cuda=False,
                local_files_only=self.offline,
                revision=EMBEDDING_MODEL_REVISION,
            )
        return self._embedding_model

    def _cross_encoder(self) -> TextCrossEncoder:
        if self._reranker is None:
            self.model_cache_root.mkdir(parents=True, exist_ok=True)
            self._reranker = TextCrossEncoder(
                model_name=RERANKER_MODEL,
                cache_dir=str(self.model_cache_root),
                cuda=False,
                local_files_only=self.offline,
                revision=RERANKER_MODEL_REVISION,
            )
        return self._reranker

    @staticmethod
    def _client(index_path: Path) -> QdrantClient:
        return QdrantClient(path=str(index_path))

    def build(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors_path: Path | None = None,
    ) -> dict[str, Any]:
        texts = [str(chunk["embedding_text"]) for chunk in chunks]
        vectors = self.embed_texts(texts)

        if vectors_path is not None:
            vectors_path.parent.mkdir(parents=True, exist_ok=True)
            with vectors_path.open("wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
        return self.build_index(chunks, index_path, vectors)

    def embed_texts(
        self,
        texts: list[str],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        embedded = list(self._embedder().passage_embed(texts, batch_size=64))
        if len(embedded) != len(texts):
            raise RuntimeError(
                "FastEmbed returned a different number of vectors than passages"
            )
        vectors = np.asarray(embedded, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(texts), EMBEDDING_DIMENSION):
            raise RuntimeError(
                f"Unexpected {EMBEDDING_MODEL} vector shape: {vectors.shape}"
            )
        if not np.isfinite(vectors).all():
            raise RuntimeError("FastEmbed returned non-finite embedding values")
        return vectors

    def build_from_vectors(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors_path: Path,
    ) -> dict[str, Any]:
        try:
            vectors = np.load(vectors_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Cannot load portable embeddings: {vectors_path}"
            ) from exc
        if vectors.dtype != np.float32:
            raise ValueError("Portable embeddings must use float32 values")
        if vectors.ndim != 2 or vectors.shape != (
            len(chunks),
            EMBEDDING_DIMENSION,
        ):
            raise ValueError(
                "Portable embedding dimensions do not match the chunk collection"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Portable embeddings contain non-finite values")
        return self.build_index(chunks, index_path, vectors)

    def build_index(
        self,
        chunks: list[dict[str, Any]],
        index_path: Path,
        vectors: np.ndarray[Any, np.dtype[np.float32]],
    ) -> dict[str, Any]:
        if not chunks:
            raise ValueError("Cannot build a dense index without chunks")
        if index_path.exists():
            raise ValueError(f"Dense index path already exists: {index_path}")
        dimension = int(vectors.shape[1])

        index_path.parent.mkdir(parents=True, exist_ok=True)
        client = self._client(index_path)
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            points = (
                models.PointStruct(
                    id=index,
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "source_path": chunk["source_path"],
                        "categories": [
                            str(value).casefold()
                            for value in chunk.get("categories", [])
                        ],
                        "keywords": [
                            str(value).casefold() for value in chunk.get("keywords", [])
                        ],
                    },
                )
                for index, (chunk, vector) in enumerate(
                    zip(chunks, vectors, strict=True)
                )
            )
            client.upload_points(
                collection_name=COLLECTION_NAME,
                points=points,
                batch_size=64,
                wait=True,
            )
            collection = client.get_collection(COLLECTION_NAME)
            point_count = int(collection.points_count or 0)
            if point_count != len(chunks):
                raise RuntimeError(
                    "Qdrant verification failed: "
                    f"expected {len(chunks)} points, found {point_count}"
                )
        finally:
            client.close()

        return {
            "backend": "Qdrant local mode",
            "collection": COLLECTION_NAME,
            "distance": "cosine",
            "embedding_runtime": "FastEmbed ONNX Runtime (CPU)",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_model_revision": EMBEDDING_MODEL_REVISION,
            "embedding_dimension": dimension,
            "point_count": len(chunks),
        }

    @staticmethod
    def _filter(
        *,
        categories: list[str] | None,
        keywords: list[str] | None,
        document_ids: list[str] | None,
        excluded_document_ids: list[str] | None,
    ) -> models.Filter | None:
        conditions: list[models.FieldCondition] = []
        conditions.extend(
            models.FieldCondition(
                key="categories",
                match=models.MatchValue(value=value.casefold()),
            )
            for value in categories or []
        )
        conditions.extend(
            models.FieldCondition(
                key="keywords",
                match=models.MatchValue(value=value.casefold()),
            )
            for value in keywords or []
        )
        if document_ids:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=document_ids),
                )
            )
        excluded_conditions: list[models.FieldCondition] = []
        if excluded_document_ids:
            excluded_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=excluded_document_ids),
                )
            )
        return (
            models.Filter(must=conditions, must_not=excluded_conditions)
            if conditions or excluded_conditions
            else None
        )

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
        if not index_path.is_dir():
            raise ValueError(f"Dense index is missing: {index_path}")
        query_vectors = list(self._embedder().query_embed(query))
        if len(query_vectors) != 1:
            raise RuntimeError("FastEmbed did not return exactly one query vector")

        client = self._client(index_path)
        try:
            if not client.collection_exists(COLLECTION_NAME):
                raise RuntimeError(f"Qdrant collection is missing: {COLLECTION_NAME}")
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vectors[0].tolist(),
                query_filter=self._filter(
                    categories=categories,
                    keywords=keywords,
                    document_ids=document_ids,
                    excluded_document_ids=excluded_document_ids,
                ),
                limit=top_k,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
        finally:
            client.close()

        hits: list[DenseSearchHit] = []
        for point in response.points:
            chunk_id = (point.payload or {}).get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise RuntimeError("Qdrant returned a point without a chunk_id")
            hits.append(DenseSearchHit(chunk_id=chunk_id, score=float(point.score)))
        return hits

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores = list(
            self._cross_encoder().rerank(
                query,
                documents,
                batch_size=32,
            )
        )
        if len(scores) != len(documents):
            raise RuntimeError(
                "FastEmbed reranker returned a different number of scores than passages"
            )
        return [float(score) for score in scores]

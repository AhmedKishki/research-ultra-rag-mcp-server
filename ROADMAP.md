# Research server roadmap

The first release deliberately establishes a reliable CPU research baseline
before adding more retrieval backends.

## Implemented baseline

- project-root and source-directory confinement;
- PDF/EPUB-only source selection;
- source hashing and immutable generations;
- PDF page and EPUB section provenance;
- reviewed bibliographic/category/keyword metadata;
- UltraRAG token chunking;
- UltraRAG CPU BM25 indexing and retrieval;
- structured evidence results and context retrieval;
- staleness reporting; and
- real stdio integration coverage.

## Next: semantic and hybrid retrieval

- pin a CPU sentence-transformer model;
- store embeddings and metadata payloads in project-local Qdrant;
- search BM25 and Qdrant independently;
- combine rankings with reciprocal-rank fusion;
- optionally rerank a bounded candidate set; and
- expose retrieval methods and component ranks in every result.

This belongs here rather than in `vanilla-ultra-rag-mcp-server` because it adds
new orchestration and result semantics.

## Next: ingestion lifecycle

- detect unchanged files by hash;
- reuse unchanged extracted units and embeddings;
- add explicit source removal and generation cleanup tools;
- provide generation listing and rollback;
- validate disk-space requirements before building; and
- add lock files for multiple server processes targeting one project.

## Next: stronger citations

- preserve character offsets within extraction units;
- distinguish physical PDF pages from printed page labels;
- store EPUB CFI or equivalent stable internal anchors where available;
- add an exact-quote verification tool;
- add optional OCR with page-level confidence; and
- export citations in common bibliographic styles without inventing metadata.

## Later

- research-focused UI;
- configurable multilingual BM25 tokenization;
- GPU embedding/reranking profiles;
- import/export and backup tooling;
- retrieval-quality evaluation sets; and
- optional cross-project federated search that preserves explicit boundaries.

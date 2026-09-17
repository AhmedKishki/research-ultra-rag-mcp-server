# Research server roadmap

The current release establishes a local CPU hybrid-retrieval baseline.

## Implemented baseline

- project-root and source-directory confinement;
- PDF/EPUB-only source selection;
- source hashing and immutable generations;
- PDF page and EPUB section provenance;
- reviewed bibliographic/category/keyword metadata;
- UltraRAG token chunking;
- UltraRAG CPU BM25 indexing and retrieval;
- pinned FastEmbed CPU embeddings;
- project-local embedded Qdrant indexes and metadata filtering;
- independently selectable BM25, dense, and hybrid retrieval;
- reciprocal-rank fusion with visible component ranks;
- opt-in bounded CPU cross-encoder reranking;
- structured evidence results and context retrieval;
- reversible agent-reviewed source exclusion without source-file deletion;
- staleness reporting; and
- real stdio integration coverage.

## Next: retrieval quality and efficiency

- create representative project-specific evaluation queries and relevance
  judgments;
- measure BM25, dense, hybrid, and reranked recall/precision;
- make fusion weights or rank constants configurable only if evaluation shows a
  repeatable benefit;
- reuse unchanged embeddings across immutable generations;
- add a multilingual embedding profile alongside multilingual BM25; and
- investigate GPU profiles without changing the CPU default.

## Next: ingestion lifecycle

- detect unchanged files by hash;
- reuse unchanged extracted units and embeddings;
- add generation cleanup tools;
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
- import/export and backup tooling;
- retrieval-quality evaluation sets; and
- optional cross-project federated search that preserves explicit boundaries.

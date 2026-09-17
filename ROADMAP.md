# Research server roadmap

This file records deferred work only. The current capabilities and operating
instructions belong in `README.md`, `AGENT_GUIDE.md`, and `AGENTS.md`.

## Retrieval quality and efficiency

- create representative project-specific evaluation queries and relevance
  judgments;
- measure BM25, dense, hybrid, and reranked recall/precision;
- make fusion weights or rank constants configurable only if evaluation shows a
  repeatable benefit;
- reuse unchanged embeddings across immutable generations;
- add a multilingual embedding profile alongside multilingual BM25; and
- investigate GPU profiles without changing the CPU default.

## Ingestion lifecycle

- detect unchanged files by hash;
- reuse unchanged extracted units and embeddings;
- add generation cleanup tools;
- provide generation listing and rollback; and
- validate disk-space requirements before building.

## Stronger citations

- preserve character offsets within extraction units;
- distinguish physical PDF pages from printed page labels;
- store EPUB CFI or equivalent stable internal anchors where available;
- add an exact-quote verification tool;
- add optional OCR with page-level confidence; and
- export citations in common bibliographic styles without inventing metadata.

## Other deferred work

- import/export and backup tooling;
- retrieval-quality evaluation sets; and
- optional cross-project federated search that preserves explicit boundaries.

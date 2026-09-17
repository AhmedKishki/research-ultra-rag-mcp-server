# Research server roadmap

This file records deferred work only. The current capabilities and operating
instructions belong in `README.md`, `AGENT_GUIDE.md`, and `AGENTS.md`.

## Future: project research memory

Add durable research memory only after defining a boundary that cannot confuse
agent-generated material with source evidence. The recommended design is a
separate project-local store beneath `.ultrarag/research/memory/`, never an
unlabelled addition to the PDF/EPUB passage index.

The memory model should distinguish at least:

- research notes and interpretations written by the user or agent;
- open questions, reading decisions, and project terminology;
- links to supporting or contradicting source passages; and
- transient conversation/session context, which should remain client-owned
  unless the user explicitly promotes it to project memory.

Required safeguards:

- an explicit write action—ordinary conversation must not be saved silently;
- visible author/origin, creation time, and last-edit time;
- user-readable listing, editing, export, and deletion;
- project confinement and no implicit cross-project retrieval;
- provenance links containing generation ID, document ID, chunk ID, locator,
  and quotation hash where a note depends on evidence;
- stale/orphan detection after a new generation changes or removes a linked
  passage;
- separate search results and labels for memory versus source passages; and
- a hard rule that memory is never quoted or cited as if it were a PDF/EPUB
  source.

Candidate tools are `remember_research_note`, `search_research_memory`,
`list_research_memory`, `update_research_memory`, and
`delete_research_memory`. Any UI should make the source/memory distinction
visually unmistakable and require confirmation for durable writes.

UltraRAG includes memory components intended for pipeline conversations. Before
reusing them here, evaluate whether their storage model supports the project
boundary, explicit writes, evidence links, deletion, and stale-reference checks
above. Reuse is desirable only if those guarantees can be preserved without
patching upstream source code.

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

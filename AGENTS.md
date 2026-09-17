# AGENTS.md

This is the engineering guide for AI coding agents working in
`research-ultra-rag-mcp-server`.

## Objective

Provide a high-level stdio MCP server for project-scoped research knowledge
bases built from original PDF and EPUB sources. The server helps an AI agent
retrieve citable evidence across a collection while preserving document
identity and source locators.

The package builds on the separately versioned
`vanilla-ultra-rag-mcp-server`. Never add research behavior to the vanilla
repository to support this project.

## Current compatibility baseline

- Package and command: `research-ultra-rag-mcp`
- Version: `0.2.0`
- Python: `>=3.11,<3.13`
- FastMCP: `3.4.0`
- Vanilla gateway commit: `d080b0c2c1172f029024149aee15d295cd8e0d14`
- Upstream UltraRAG: `0.3.0.2` at
  `3a709a2aea3fbe46acca59c422621c94b6e86857`

## Non-negotiable research contract

- Retain the prominent UltraRAG acknowledgement in `README.md`, the root
  `NOTICE`, upstream project links, license information, and the independent
  project disclaimer.
- Credit THUNLP, NEUIR, OpenBMB, AI9stars, and the upstream contributors using
  the wording supported by UltraRAG's own README. Do not imply endorsement.
- One server process serves exactly one configured project root.
- MCP tools must not accept arbitrary filesystem output paths.
- Ingestion selects only regular PDF and EPUB files beneath the configured
  sources directory.
- Reject source symlinks and path traversal.
- Store all derived state beneath `<project>/.ultrarag/research`.
- Never edit or write the original source documents.
- Do not switch `current.json` until a generation is completely indexed.
- Preserve deterministic document IDs, chunk IDs, source paths, and locators.
  Chunk IDs may change when content or chunking configuration changes; never
  imply that they are permanent across incompatible generations.
- Keep BM25 and dense indexes in the same immutable generation, and never select
  the generation unless both indexes validate successfully.
- Keep dense vectors and Qdrant payloads project-local; do not introduce a
  required external database service.
- Search results must separate passage text from metadata so an agent knows what
  may be quoted.
- Never claim that extracted text is a substitute for checking the original.
- Keep MCP stdout reserved for protocol messages.

## Architecture

```text
AI agent / MCP client
        │ stdio
        ▼
research-ultra-rag-mcp
        ├── project/source policy
        ├── PDF page extraction
        ├── EPUB section extraction
        ├── metadata and provenance
        ├── immutable generations
        ├── FastEmbed CPU embeddings
        ├── project-local Qdrant dense index
        ├── reciprocal-rank fusion
        ├── optional CPU cross-encoder reranking
        └── persistent stdio MCP -> vanilla-ultra-rag-mcp
                                  ├── UltraRAG corpus chunker
                                  └── UltraRAG BM25 retriever
```

Only the high-level research tools are exposed to the outer MCP client. The
vanilla gateway is an implementation dependency, not a second user-facing tool
surface within this server.

## Intentional differences from Vanilla RAG

The official UltraRAG Vanilla RAG pipeline includes benchmark loading, dense
retrieval, prompt rendering, model generation, answer extraction, and
evaluation. This server reuses UltraRAG's chunking and retrieval mechanics but
changes the boundary for research work:

- PDF/EPUB extraction, project isolation, immutable storage, metadata, and
  locators are implemented here.
- Retrieval combines UltraRAG CPU BM25 with a project-local FastEmbed/Qdrant
  dense index. This extension owns fusion, filters, scores, and provenance.
- `search` returns visible, structured evidence instead of anonymous passage
  strings.
- The calling AI agent is the generation stage and must cite the returned
  evidence; this server does not call UltraRAG generation internally.
- Benchmark loading, boxed-answer extraction, and automatic evaluation are not
  part of the interactive research flow.

Do not blur this boundary in documentation. Adding server-side answer
generation would be a deliberate research feature requiring its own API,
citation contract, tests, and user-visible model configuration.

## Repository map

- `src/research_ultra_rag_mcp/server.py`: CLI, MCP lifecycle, and six public
  tools.
- `config.py`: project boundary and executable validation.
- `sources.py`: allowlist, source discovery, hashing, and metadata validation.
- `extraction.py`: PDF pages and EPUB sections.
- `storage.py`: atomic JSON state and JSONL artifacts.
- `dense.py`: pinned FastEmbed models, local Qdrant indexing/filtering, and
  optional cross-encoder reranking.
- `ultrarag.py`: persistent client for vanilla UltraRAG tools.
- `service.py`: generation, indexing, status, filtering, and evidence workflow.
- `instructions.py`: guidance returned to MCP agents.
- `verify.py`: terminal MCP client for end-to-end project verification.
- `tests/`: unit and real stdio integration coverage.
- `ROADMAP.md`: explicitly deferred work.

## Storage model

The mutable pointer and reviewed metadata live at:

```text
<project>/.ultrarag/research/current.json
<project>/.ultrarag/research/source-metadata.json
```

Each build gets a unique directory under `generations/`. A failed generation
gets `failure.json` and never becomes current. Successful generations contain a
manifest, extraction units, raw UltraRAG chunks, enriched chunks, BM25 index,
and Qdrant index. Model weights are cached once per project under `models/`. Do
not treat generated files as source documents.

## Public MCP tools

- `status`: read-only source/current/staleness inspection.
- `ingest`: create and select a complete new generation.
- `search`: hybrid-by-default retrieval with selectable BM25/dense modes,
  optional reranking, and structured evidence.
- `list_sources`: inspect indexed documents and metadata.
- `get_passage`: retrieve neighboring chunks from the same document.
- `set_source_metadata`: update reviewed metadata for the next generation.

Tool docstrings and `SERVER_INSTRUCTIONS` are part of the agent-facing contract.
Update tests and documentation when changing them.

## Retrieval contract and recorded decisions

- Default method: `hybrid`; diagnostic methods: `bm25` and `dense`.
- Lexical path: pinned vanilla gateway -> UltraRAG BM25.
- Dense path: FastEmbed `BAAI/bge-small-en-v1.5`, ONNX Runtime CPU, 384
  dimensions, cosine distance, embedded Qdrant collection `research_chunks`.
  Artifact revision: `52398278842ec682c6f32300af41344b1c0b0bb2`.
- Fusion: reciprocal-rank fusion with `k=60`. Do not combine raw BM25 and cosine
  values; their scales are unrelated.
- Candidate depth: at least 20, normally `top_k * 4`, bounded at 200 and by the
  current chunk count.
- Chunking: UltraRAG token chunker with the GPT-2 tiktoken encoding, default and
  maximum 384 tokens, overlap 64. The cap reserves space for retrieval metadata
  within the embedding model's 512-token window; do not raise it without adding
  an explicit long-input strategy and tests.
- Optional reranker: FastEmbed
  `Xenova/ms-marco-MiniLM-L-6-v2`, CPU, lazily loaded, applied to at most 50
  candidates. Artifact revision:
  `a09144355adeed5f58c8ed011d209bf8ee5a1fec`. It must remain opt-in because of
  latency and its extra model.
- Qdrant owns only vectors and lookup/filter payloads. `chunks.jsonl` remains the
  canonical passage/provenance store.
- Qdrant is used instead of FAISS here because payload filtering and scored
  results are needed. FAISS remains an upstream vanilla capability. Milvus is
  intentionally not required because this server targets local project use.
- Category and keyword lists use AND semantics; document IDs use membership
  semantics. Store normalized filter values in Qdrant, but return reviewed
  values from the canonical chunk store.
- Raw BM25 scores are unavailable from the pinned UltraRAG tool. Report its
  rank, never synthesize a score. Dense/fusion/reranker scores are ranking
  signals, not calibrated confidence or truth probabilities.
- `text` is the only quote-safe passage field returned to clients.
  `embedding_text` is internal retrieval input and must not be exposed as a
  quotation.
- Schema-1 generations are BM25-only. Keep them usable when a caller explicitly
  requests `bm25`; require a new ingestion before dense or hybrid search.

Do not move the Qdrant implementation into the vanilla gateway or patch
UltraRAG for this feature. The research-specific integration deliberately lives
in this repository so vanilla can continue tracking upstream safely.

## Safe change rules

- Use `pathlib.Path`, type hints, and JSON-serializable tool results.
- Keep blocking extraction and filesystem scans outside the event loop.
- Serialize ingest/search/metadata operations with the service lock.
- Prefer new immutable generations to in-place index mutation.
- Validate a new artifact before updating a pointer to it.
- Keep model downloads lazy: the embedding model is needed during ingestion;
  the reranker model only when `rerank=true`. In offline mode, require an
  existing project-local cache instead of making a network request.
- Do not expose underlying vanilla tools through this server.
- Do not silently skip a selected PDF/EPUB that fails extraction; fail the new
  generation and leave the previous current generation intact.
- Empty upstream chunk records may be discarded only when their extraction unit
  still has at least one searchable chunk; record the discarded count.
- Retain explicit limitations when a feature is not implemented.

## Validation

Before finalizing a change, run:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q src tests
```

For source or retrieval changes, the integration test must still launch the
real vanilla stdio server, build BM25 and Qdrant indexes, run hybrid and dense
search, retrieve the known passage, and prove that a neighboring Markdown file
was excluded. It must then restart offline and repeat hybrid reranked search from
the caches. Unit tests must cover RRF and failure atomicity without depending on
model downloads.

For significant extraction changes, also test a representative real collection
without writing into its source directory.

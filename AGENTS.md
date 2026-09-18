# AGENTS.md

This is the engineering guide for AI coding agents working in
`research-ultra-rag-mcp-server`.

## Objective

Provide a high-level stdio MCP server for project-scoped research knowledge
bases built from original PDF and EPUB sources. The server helps an AI agent
retrieve cleaned semantic evidence across a collection while preserving
document identity and original-file locators. It also provides a research
adapter that connects the shared loopback-only `ui-ultra-rag-mcp` workspace to
the same public MCP tools and project state.

The package builds on the separately versioned
`vanilla-ultra-rag-mcp-server`. Never add research behavior to the vanilla
repository to support this project.

## Documentation responsibilities

- `README.md` is a standalone user manual: capability summary, operation,
  installation, MCP configuration, concrete usage, expected results, storage,
  and user-visible limitations. It must not compare or link to sibling
  MCP-server projects.
- `AGENT_GUIDE.md` is operational policy for an AI agent using the nine research
  tools. Do not put installation or contributor workflows there.
- `AGENTS.md` is this engineering contract. It may document internal dependency
  boundaries, but must not become a second user manual.
- `ROADMAP.md` contains only deferred work.
- `NOTICE` contains attribution and legal notices.

## Current compatibility baseline

- Package: `research-ultra-rag-mcp`
- Commands: `research-ultra-rag-mcp`, `research-ultra-rag-ui`,
  `research-ultra-rag-verify`, and `research-ultra-rag-bundle`
- Version: `0.8.0`
- Python: `>=3.11,<3.13`
- FastMCP: `3.4.0`
- Vanilla gateway commit: `05ae4b155d38a294260a36017f6429ce73b1641b`
- Shared UI commit: `fb569668c381efaa0089536de99928c77f8e7f31`
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
- Store portable identity/review state beneath `<project>/.research-rag` and all
  disposable runtime state beneath `<project>/.ultrarag/research`.
- Share only immutable model binaries through the configured user cache. Never
  place documents, metadata, chunks, vectors, indexes, bundles, logs, or query
  state in global storage.
- Never edit or write the original source documents.
- Keep source exclusions explicit, reversible, project-local, and immediately
  enforced by every retrieval surface. Do not add automatic duplicate guessing.
- Do not switch `current.json` until a generation is completely indexed.
- Preserve deterministic document IDs, chunk IDs, source paths, and locators.
  Chunk IDs may change when content or chunking configuration changes; never
  imply that they are permanent across incompatible generations.
- Keep BM25 and dense indexes in the same immutable generation, and never select
  the generation unless both indexes validate successfully.
- Keep dense vectors and Qdrant payloads project-local; do not introduce a
  required external database service.
- Search results must identify `text` as cleaned semantic text with
  `direct_quote_safe=false`; direct quotations must come from the original.
- Keep MCP stdout reserved for protocol messages.
- Keep the UI bound to loopback addresses. Do not add remote exposure or
  authentication assumptions without an explicit security design.
- Keep browser write endpoints same-origin, JSON-only, and constrained to the
  nine public MCP operations.
- Do not let UI code read or mutate generation artifacts directly. It must use
  the private stdio MCP client, except for safely serving an allowlisted
  original PDF or EPUB from the configured source root.
- Keep the shared UI dependency pinned by commit. Keep MCP transport, research
  tool mapping, and source authorization in this repository's adapter; do not
  copy the shared static workspace back into this package.

## Architecture

```text
AI agent / MCP client ── stdio ──────────────────────────┐
                                                         ▼
Local browser ── HTTP ── shared UI ── research adapter ── research-ultra-rag-mcp

research-ultra-rag-mcp
├── project/source policy
├── PDF page extraction
├── EPUB section extraction
├── metadata and provenance
├── portable project bundles
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
- `search` returns visible, structured semantic evidence instead of anonymous
  passage strings. It is explicitly not an exact-quotation surface.
- The calling AI agent is the generation stage and must cite the returned
  evidence; this server does not call UltraRAG generation internally.
- Benchmark loading, boxed-answer extraction, and automatic evaluation are not
  part of the interactive research flow.

Do not blur this boundary in documentation. Adding server-side answer
generation would be a deliberate research feature requiring its own API,
citation contract, tests, and user-visible model configuration.

## Repository map

- `src/research_ultra_rag_mcp/server.py`: CLI, MCP lifecycle, and nine public
  tools.
- `config.py`: project boundary, stable identity, and executable validation.
- `sources.py`: allowlist, source discovery, hashing, and metadata validation.
- `extraction.py`: layout-aware PDF/EPUB extraction and bibliographic identity.
- `bundle.py`: deterministic export and hostile-archive-safe import staging.
- `bundle_cli.py`: terminal export/import client.
- `storage.py`: atomic JSON state and JSONL artifacts.
- `dense.py`: pinned FastEmbed models, local Qdrant indexing/filtering, and
  optional cross-encoder reranking.
- `generation.py`: exact compatibility checks and validated reuse snapshots.
- `ultrarag.py`: persistent client for vanilla UltraRAG tools.
- `transport.py`: the single research stdio transport builder used by UI and
  terminal clients.
- `service.py`: generation, indexing, status, filtering, and evidence workflow.
- `instructions.py`: guidance returned to MCP agents.
- `ui.py`: shared-UI profile, private MCP client, public-tool mapping, and safe
  original-source authorization.
- `ui-ultra-rag-mcp` dependency: loopback HTTP host, constrained JSON API, and
  packaged dependency-free browser workspace.
- `verify.py`: terminal MCP client for end-to-end project verification.
- `tests/`: unit and real stdio integration coverage.
- `ROADMAP.md`: explicitly deferred work.

## Storage model

Portable, project-owned state lives at:

```text
<project>/.research-rag/project.json
<project>/.research-rag/source-metadata.json
<project>/.research-rag/source-exclusions.json
<project>/.research-rag/bundles/
```

`project.json` is authoritative for the stable project ID, name, and
project-relative source directory. CLI entrypoints reuse its source setting
when `--source-directory` is omitted; an explicit differing value must fail.

Disposable local state lives at:

```text
<project>/.ultrarag/research/current.json
<project>/.ultrarag/research/project.lock
```

Changed builds use a unique directory under `staging/`, then move a verified
generation beneath `generations/` before switching `current.json`. Successful
generations retain only the manifest, cleaned extraction units, final chunks,
portable float32 vectors, BM25 index, and Qdrant index. UltraRAG raw chunks are
temporary staging data, and raw coordinate records are not generated. Failed
builds remove heavy staging data and leave a small record under `failures/`.
Model binaries default to `~/.cache/research-ultra-rag-mcp/models` and are the
only cross-project shared state. `project.lock` serializes MCP and UI operations
across processes so no caller observes a partial index.

## Public MCP tools

- `status`: read-only source/current/staleness inspection.
- `ingest`: return the current generation for an exact no-op, or create and
  select a complete new generation with verified reuse; `force_recompute`
  bypasses reuse.
- `search`: hybrid-by-default retrieval with selectable BM25/dense modes,
  optional reranking, and structured evidence.
- `list_sources`: inspect indexed documents and metadata.
- `get_passage`: retrieve neighboring chunks from the same document.
- `set_source_metadata`: update reviewed metadata for the next generation.
- `set_source_inclusion`: immediately exclude or restore an agent/user-reviewed
  source without modifying the source file; rebuild later to align the indexes.
- `export_bundle`: export a fresh generation and all original sources beneath
  the project's portable bundle directory.
- `import_bundle`: validate a project-owned bundle, reconstruct BM25/Qdrant from
  chunks/vectors, install non-conflicting originals, and switch current last.

Tool docstrings and `SERVER_INSTRUCTIONS` are part of the agent-facing contract.
Update tests and documentation when changing them.

## Retrieval contract and recorded decisions

- Default method: `hybrid`; diagnostic methods: `bm25` and `dense`.
- Lexical path: pinned vanilla gateway -> UltraRAG BM25.
- Dense path: FastEmbed `BAAI/bge-small-en-v1.5`, ONNX Runtime CPU, 384
  dimensions, cosine distance, embedded Qdrant collection `research_chunks`.
  Artifact revision: `52398278842ec682c6f32300af41344b1c0b0bb2`.
- Fusion: weighted reciprocal-rank fusion with `k=60`, BM25 weight `1.25`, and
  dense weight `1.0`. Do not combine raw BM25 and cosine values; their scales
  are unrelated.
- Candidate depth: at least 20, normally `top_k * 4`, bounded at 200 and by the
  current chunk count.
- Chunking: UltraRAG token chunker with the GPT-2 tiktoken encoding, default and
  maximum 384 tokens, overlap 64. The cap stays below the embedding model's
  512-token input limit despite tokenizer differences; do not raise it without
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
- BM25 candidates require at least one non-stopword query token. Dense candidates
  require cosine similarity `>= 0.72`. Reject extraction artifacts before the
  optional reranker and permit fewer than `top_k`, including zero.
- `text` and internal `embedding_text` contain only cleaned semantic content;
  never inject paths, IDs, authors, citations, or repeated document titles into
  each indexed passage.
- Normalize layout wrapping before chunking, remove controls/soft hyphens, and
  join alphabetic line-end hyphen splits. Preserve all other wording and
  punctuation, but set `direct_quote_safe=false` on every public passage.
- Do not retain raw coordinate extraction. The untouched PDF/EPUB is the quote
  authority. Preserve legend-marker meaning in cleaned-unit annotations.
- Schema-1 generations are BM25-only. Keep them usable when a caller explicitly
  requests `bm25`; require a new ingestion before dense or hybrid search.
- Exclusions are path-based, stored outside generations, and applied to BM25,
  dense, source-list, and passage results immediately. Ingestion snapshots the
  exclusion revision and omits excluded documents from both indexes. Inclusion
  can only restore current retrieval immediately if the current generation
  still contains that source.
- Bundle import must reject traversal, symlinks/non-regular members, duplicate
  entries, checksum/schema/model failures, project-ID mismatch, unsafe manifest
  paths, and differing bytes at an existing source path. Never export live
  indexes, locks, logs, runtime files, or model caches. Import intentionally
  replaces reviewed metadata/exclusions with the validated bundled copies and
  must disclose that behavior.

## Metadata and extraction contract

- Resolve each field independently. PDF precedence is reviewed override,
  high-confidence visible front matter, validated embedded metadata, then the
  filename stem for title only. EPUB precedence is reviewed override, validated
  OPF metadata, visible title/byline, then the filename stem for title only.
- Keep automatic metadata rules generic and conservative. Never add a source-,
  title-, author-, or publisher-specific extraction exception to fix one
  document. Expose uncertainty through provenance and warnings, then use a
  reviewed `set_source_metadata` override for the exceptional document.
- Never infer authors from filenames. Reject DOI/URL/export-junk titles, move a
  detected DOI to its own field, and expose per-field provenance/confidence plus
  review warnings.
- Inspect the first five text-bearing PDF pages for identity. Keep physical page
  and available page-label locators.
- Use coordinate blocks to restore column order, remove repeated margins/page
  numbers, and distinguish prose, lists, tables, and figures. Do not infer visual
  relationships not expressed by captions, legends, or labels.
- Hash all discovered sources before ingestion decides whether work is needed.
  Exact source bytes, portable-state revisions, chunk settings, processing
  policies, and model fingerprints are required for a no-op.
- Reuse extraction units/chunks only for a source with matching bytes,
  per-source reviewed metadata, and processing fingerprints. Reuse a vector
  only when `embedding_text`, model revision, and dimension match exactly.
- Always reconstruct complete BM25 and Qdrant indexes for a changed generation;
  never update selected indexes in place. `force_recompute=true` disables all
  document, chunk, and vector reuse.

Do not move the Qdrant implementation into the vanilla gateway or patch
UltraRAG for this feature. The research-specific integration deliberately lives
in this repository so vanilla can continue tracking upstream safely.

## Safe change rules

- Use `pathlib.Path`, type hints, and JSON-serializable tool results.
- Keep blocking extraction and filesystem scans outside the event loop.
- Serialize every project operation with both the in-process service lock and
  the cross-process `project.lock`.
- Prefer new immutable generations to in-place index mutation.
- Validate a new artifact before updating a pointer to it.
- Keep model downloads lazy: the embedding model is needed during ingestion;
  the reranker model only when `rerank=true`. In offline mode, require an
  existing shared cache, with read-only fallback to an existing legacy
  project-local cache. Never migrate or delete that legacy cache automatically.
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

Research UI adapter changes must cover safe source-file resolution, forwarding
to the public MCP tools, and the real UI host against an existing project
without mutating its sources. Shared workspace, JSON validation, capability,
and same-origin changes belong in `ui-ultra-rag-mcp` and must pass that
package's own tests before updating the pinned commit here.

For source or retrieval changes, the integration test must still launch the
real vanilla stdio server, build BM25 and Qdrant indexes, run hybrid and dense
search, retrieve the known passage, and prove that a neighboring Markdown file
was excluded. It must then restart offline and repeat hybrid reranked search from
the caches. Unit tests must cover RRF and failure atomicity without depending on
model downloads. They must also cover no-op ingestion, additions, changes,
removals, reviewed metadata/exclusions, same-size/same-mtime byte changes,
forced regeneration, exact-text vector reuse, and lean final artifacts.

For significant extraction changes, also test a representative real collection
without writing into its source directory.

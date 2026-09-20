# AGENTS.md

This is the engineering guide for AI coding agents working in `research-ultra-rag-mcp-server`.

## Objective

Provide a high-level stdio MCP server for project-scoped research knowledge bases built from original PDF and EPUB sources. The server helps an AI agent retrieve cleaned semantic evidence across a collection while preserving document identity and original-file locators. It also provides a research adapter that connects the shared loopback-only `ui-ultra-rag-mcp` workspace to the same public MCP tools and project state.

The package builds on the separately versioned `vanilla-ultra-rag-mcp-server`. Never add research behavior to the vanilla repository to support this project.

## Documentation responsibilities

- `README.md` is a standalone user manual: capability summary, operation, installation, MCP configuration, concrete usage, expected results, storage, and user-visible limitations. It must not compare or link to sibling MCP-server projects.
- `AGENT_GUIDE.md` is operational policy for an AI agent using the nine research tools. Do not put installation or contributor workflows there.
- `AGENTS.md` is this engineering contract. It may document internal dependency boundaries, but must not become a second user manual.
- `ROADMAP.md` contains only deferred work.
- `PLAN.md` records review findings, their evidence, and the agreed strengthening sequence. It is a review and plan document, not a description of shipped behaviour.
- `TODO.md` is the actionable checklist derived from `PLAN.md`. It may repeat `PLAN.md` items as checkboxes but must not introduce capability that is absent from `PLAN.md` or `ROADMAP.md`.
- `NOTICE` contains attribution and legal notices.

## Presenting decisions to the user

Any question that needs a user choice must be presented as a numbered list of concrete options, never as an open question. For every decision:

- give each option a short identifier (`A`, `B`, `C`, …), a one-line statement of what it does, and the smallest change it requires;
- state pros and cons for each option separately, covering cost, risk, and the effect on the research contract;
- state whether each option is reversible and what undoing it would take;
- mark exactly one option as the recommendation and say why in one sentence;
- include an explicit "no change" or "decide later" option whenever work can proceed without an answer;
- keep options mutually exclusive, and complete enough that choosing one is sufficient to proceed;
- record the options in `PLAN.md` with the same identifiers before asking, and record the chosen option there afterwards, so the decision stays traceable;
- never implement a choice that changes generation artifacts, retrievable evidence, the portable-state contract, or on-disk layouts before the user has chosen it.

## Current compatibility baseline

- Package: `research-ultra-rag-mcp`
- Commands: `research-ultra-rag-mcp`, `research-ultra-rag-ui`, `research-ultra-rag-verify`, and `research-ultra-rag-bundle`
- Version: `0.11.0`
- Python: `>=3.11,<3.13`
- FastMCP: `3.4.0`
- Vanilla gateway commit: `05ae4b155d38a294260a36017f6429ce73b1641b`
- Shared UI commit: `fb569668c381efaa0089536de99928c77f8e7f31`
- Upstream UltraRAG: `0.3.0.2` at `3a709a2aea3fbe46acca59c422621c94b6e86857`

## Non-negotiable research contract

- Retain the prominent UltraRAG acknowledgement in `README.md`, the root `NOTICE`, upstream project links, license information, and the independent project disclaimer.
- Credit THUNLP, NEUIR, OpenBMB, AI9stars, and the upstream contributors using the wording supported by UltraRAG's own README. Do not imply endorsement.
- One server process serves exactly one configured project root.
- MCP tools must not accept arbitrary filesystem output paths.
- Ingestion selects only regular PDF and EPUB files beneath the configured sources directory.
- Reject source symlinks and path traversal.
- Store every project-owned research-RAG artifact beneath `<project>/.research-rag` by default: portable identity/review state at its root and all disposable derived state beneath `.research-rag/runtime`. An explicit `--runtime-root` may relocate derived state only, and only to an absolute path claimed by a marker naming this project's `project_id`; review state must stay in the project, and two projects must never share one runtime root.
- Share only immutable model binaries through the configured user cache. Never place documents, metadata, chunks, vectors, indexes, bundles, logs, or query state in global storage.
- Never edit or write the original source documents.
- Keep source exclusions explicit, reversible, project-local, and immediately enforced by every retrieval surface. Do not add automatic duplicate guessing.
- Do not switch `current.json` until a generation is completely indexed.
- Preserve deterministic source IDs, document IDs, chunk IDs, source paths, and locators. A project-scoped `source_id` is based on the normalized source-relative path, survives byte changes, and changes on rename/move. A `document_id` identifies a path/content version. Chunk IDs may change when content or chunking configuration changes; never imply that document or chunk IDs are permanent across incompatible generations.
- Keep BM25 and dense indexes in the same immutable generation, and never select the generation unless both indexes validate successfully.
- Never let a checkpoint, `current.json`, or any other commit point become durable before the artifacts it describes. Grouping directory fsyncs inside a unit is required and expected (`fsync_directories` before the checkpoint write); dropping the artifact write's own durability, the checkpoint's, or the ordering is not. A file written only to hand off to a peer process and rewritten before every use may skip fsync entirely.
- Keep dense vectors and any dense index project-local; do not introduce a required external database service.
- Record the dense backend in each generation manifest and dispatch retrieval from that record. Never rebuild an existing generation with a different backend, and never assume a fixed dense index directory name.
- Search results must identify `text` as cleaned semantic text with `direct_quote_safe=false`; direct quotations must come from the original.
- Keep MCP stdout reserved for protocol messages.
- Keep the UI bound to loopback addresses. Do not add remote exposure or authentication assumptions without an explicit security design.
- Keep browser write endpoints same-origin, JSON-only, and constrained to the nine public MCP operations.
- Do not let UI code read or mutate generation artifacts directly. It must use the private stdio MCP client, except for safely serving an allowlisted original PDF or EPUB from the configured source root.
- Keep the shared UI dependency pinned by commit. Keep MCP transport, research tool mapping, and source authorization in this repository's adapter; do not copy the shared static workspace back into this package.

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
├── project-local dense index (exact scan by default)
├── reciprocal-rank fusion
├── optional CPU cross-encoder reranking
└── persistent stdio MCP -> vanilla-ultra-rag-mcp
                          ├── UltraRAG corpus chunker
                          └── UltraRAG BM25 retriever
```

Only the high-level research tools are exposed to the outer MCP client. The vanilla gateway is an implementation dependency, not a second user-facing tool surface within this server.

## Intentional differences from Vanilla RAG

The official UltraRAG Vanilla RAG pipeline includes benchmark loading, dense retrieval, prompt rendering, model generation, answer extraction, and evaluation. This server reuses UltraRAG's chunking and retrieval mechanics but changes the boundary for research work:

- PDF/EPUB extraction, project isolation, immutable storage, metadata, and locators are implemented here.
- Retrieval combines UltraRAG CPU BM25 with a project-local FastEmbed dense index. This extension owns fusion, filters, scores, and provenance.
- `search` returns visible, structured semantic evidence instead of anonymous passage strings. It is explicitly not an exact-quotation surface.
- The calling AI agent is the generation stage and must cite the returned evidence; this server does not call UltraRAG generation internally.
- Benchmark loading, boxed-answer extraction, and automatic evaluation are not part of the interactive research flow.

Do not blur this boundary in documentation. Adding server-side answer generation would be a deliberate research feature requiring its own API, citation contract, tests, and user-visible model configuration.

## Repository map

- `src/research_ultra_rag_mcp/server.py`: CLI, MCP lifecycle, and nine public tools.
- `config.py`: project boundary, stable identity, and executable validation.
- `sources.py`: allowlist, source discovery, hashing, and metadata validation.
- `extraction.py`: layout-aware PDF/EPUB extraction and bibliographic identity.
- `artifact_lookup.py`: generation-local SQLite offsets for selective canonical chunk/unit reads and exact-text vector reuse without copied corpus text.
- `bundle.py`: deterministic export and hostile-archive-safe import staging.
- `bundle_cli.py`: terminal export/import client.
- `storage.py`: atomic JSON state and JSONL artifacts.
- `dense.py`: pinned FastEmbed models, both local dense backends (exact scan and embedded ANN) with document filtering, and optional cross-encoder reranking.
- `generation.py`: exact compatibility checks and validated reuse snapshots.
- `ultrarag.py`: persistent client for vanilla UltraRAG tools.
- `transport.py`: the single research stdio transport builder used by UI and terminal clients.
- `service.py`: generation, indexing, status, filtering, and evidence workflow.
- `instructions.py`: guidance returned to MCP agents.
- `ui.py`: shared-UI profile, private MCP client, public-tool mapping, and safe original-source authorization.
- `ui-ultra-rag-mcp` dependency: loopback HTTP host, constrained JSON API, and packaged dependency-free browser workspace.
- `verify.py`: terminal MCP client for end-to-end project verification.
- `tests/`: unit and real stdio integration coverage.
- `ROADMAP.md`: explicitly deferred work.

## Storage model

Portable, project-owned state lives at:

```text
<project>/.research-rag/project.json
<project>/.research-rag/source-catalog.json
<project>/.research-rag/source-metadata.json
<project>/.research-rag/source-exclusions.json
<project>/.research-rag/bundles/
```

`project.json` is authoritative for the stable project ID, name, and project-relative source directory. CLI entrypoints reuse its source setting when `--source-directory` is omitted; an explicit differing value must fail.

Disposable local state lives at:

```text
<project>/.research-rag/runtime/current.json
<project>/.research-rag/runtime/project.lock
```

`--runtime-root` (or `RESEARCH_ULTRARAG_RUNTIME_ROOT`) may relocate the whole runtime root, including `current.json` and `project.lock`, to an absolute path outside the project, which is how a project on slow storage keeps generations, staging, and index builds on a fast device. The root is claimed on first use by a `.research-ultra-rag-runtime.json` marker holding the owning `project_id`; a root with a different `project_id`, a non-empty root with no marker, a relative path, the project root, and a non-directory are all rejected with explicit messages. Review state in `.research-rag` never moves, and the legacy-location migration runs only for the default root.

`resolve_config` performs a guarded one-time move from the legacy `.ultrarag/research` location. Never create new research state there. Refuse to guess when both old and new locations contain runtime payloads. Migration must run only after older research MCP and UI processes have stopped.

Changed builds use a unique directory under `staging/`, then move a verified generation beneath `generations/` before switching `current.json`. Successful generations retain only the manifest, cleaned extraction units, final chunks, portable float32 vectors, a text-free SQLite offset lookup, BM25 index, and dense index selected for that generation. UltraRAG raw chunks are temporary staging data, and raw coordinate records are not generated. Bounded calls, cancellations, and timeouts retain an atomic `checkpoint.json` and only committed work; incompatible inputs supersede that checkpoint with a small diagnostic. Non-resumable failures remove heavy staging data and leave a small record under `failures/`. Model binaries default to `~/.cache/research-ultra-rag-mcp/models` and are the only cross-project shared state. `project.lock` serializes MCP and UI operations across processes so no caller observes a partial index.

## Public MCP tools

- `status`: read-only source/current/staleness inspection.
- `ingest`: return the current generation for an exact no-op, advance a checkpointed build and return `in_progress`, or select a complete new generation with verified reuse; `force_recompute` bypasses reuse but may resume its own matching checkpoint.
- `search`: hybrid-by-default retrieval with selectable BM25/dense modes, optional reranking, and passage-ranked or reference-grouped structured evidence.
- `list_sources`: inspect indexed documents and metadata, expose `discovered_sources` before ingestion, and idempotently register those stable IDs in the portable catalog so `known_sources` remains addressable after an original disappears. Its MCP read-only hint must remain false because this registration is a durable project-state write.
- `get_passage`: retrieve neighboring chunks from the same document.
- `set_source_metadata`: update authoritative reviewed metadata immediately for every retrieval surface when the source is in the selected generation; an unindexed source still requires ingestion. Require exactly one of `source_id` or `source_path`, preferring the ID for agent operations.
- `set_source_inclusion`: immediately exclude or restore an agent/user-reviewed source without modifying the source file; rebuild later to align the indexes. It uses the same exact-one-selector rule.
- `export_bundle`: export a fresh generation and all original sources beneath the project's portable bundle directory.
- `import_bundle`: validate a project-owned bundle, reconstruct BM25 and the recorded dense backend from chunks/vectors, and install non-conflicting originals plus portable state. Switch current last when activation is requested; otherwise keep the pointer while imported metadata/exclusions immediately govern matching retrieval.

Tool docstrings and `SERVER_INSTRUCTIONS` are part of the agent-facing contract. Update tests and documentation when changing them.

## Retrieval contract and recorded decisions

- Default method: `hybrid`; diagnostic methods: `bm25` and `dense`.
- Lexical path: pinned vanilla gateway -> UltraRAG BM25.
- Dense path: FastEmbed `BAAI/bge-small-en-v1.5`, ONNX Runtime CPU, 384 dimensions, cosine distance, and a project-local index the manifest records: the exact scan by default, or the embedded Qdrant collection `research_chunks` above the documented threshold. Artifact revision: `52398278842ec682c6f32300af41344b1c0b0bb2`.
- Fusion: weighted reciprocal-rank fusion with `k=60`, BM25 weight `1.25`, and dense weight `1.0`. Do not combine raw BM25 and cosine values; their scales are unrelated.
- Candidate depth: at least 20, normally `top_k * 4`, bounded at 200 and by the current chunk count.
- `top_k` is always a total returned-passage budget. The optional reference view scans the complete relevance-gated candidate ordering, admits at most two passages per `source_id` by default, and groups those passages without aggregating scores or rewarding documents for producing more chunks.
- Reference grouping is presentation-time MCP orchestration. It must not alter indexes or the default passage ordering. Keep its cap explicit, report both returned and candidate-pool reference counts, and preserve the unreranked candidate tail after a reranked prefix so useful references remain reachable.
- Chunking: UltraRAG token chunker with the GPT-2 tiktoken encoding, default and maximum 384 tokens, overlap 64. The cap stays below the embedding model's 512-token input limit despite tokenizer differences; do not raise it without an explicit long-input strategy and tests.
- Optional reranker: FastEmbed `Xenova/ms-marco-MiniLM-L-6-v2`, CPU, lazily loaded, applied to at most 50 candidates. Artifact revision: `a09144355adeed5f58c8ed011d209bf8ee5a1fec`. It must remain opt-in because of latency and its extra model.
- A dense index owns only vectors and identifiers (`chunk_id`, `document_id`, `source_id`). `chunks.jsonl` remains the canonical passage and locator store; document metadata and provenance remain canonical in the manifest, so filtering resolves current metadata to document IDs at query time.
- The generation-local SQLite artifact lookup contains only identifiers, ordinals, content hashes, byte offsets into canonical chunk/unit JSONL, and one integer retrieval verdict per chunk. It must never duplicate passage or extraction text. Use it for candidate retrieval, neighboring passages, document-scoped reuse, and exact-text vector reuse; reconstruct it from canonical artifacts when a legacy generation or portable bundle does not contain it.
- Precompute a retrieval verdict when the lookup is built and reject candidates from that stored value rather than rescanning text per query. The verdict is a bitmask over properties of the chunk alone (`chunk_health_flags`: corrupt text, extraction artifact), never a property of the query, and it must mirror the query-time check exactly so rejection counters and withheld disclosures cannot change. Keep a fallback that recomputes it from text when a lookup predates the column, and recompute reason codes only for a chunk the verdict flags as corrupt, because the response discloses them.
- Qdrant is used instead of FAISS here because payload filtering and scored results are needed. FAISS remains an upstream vanilla capability. Milvus is intentionally not required because this server targets local project use.
- Category and keyword lists use AND semantics; document IDs use membership semantics. Resolve current reviewed filters to matching document IDs, use those IDs for dense retrieval, and verify results against canonical documents rather than copying mutable metadata into a dense index.
- Raw BM25 scores are unavailable from the pinned UltraRAG tool. Report its rank, never synthesize a score. Dense/fusion/reranker scores are ranking signals, not calibrated confidence or truth probabilities.
- BM25 candidates require at least one non-stopword query token. Dense candidates require cosine similarity `>= 0.72`. Reject extraction artifacts before the optional reranker and permit fewer than `top_k`, including zero.
- Final chunk artifacts use one cleaned semantic-content field, `contents`; `text` and `embedding_text` are read-only legacy compatibility fallbacks. Never inject paths, authors, citations, or repeated document titles into each indexed passage. Public results project `contents` as `text`.
- Normalize layout wrapping before chunking, remove controls/soft hyphens, and join alphabetic line-end hyphen splits. Preserve all other wording and punctuation, but set `direct_quote_safe=false` on every public passage.
- Do not retain raw coordinate extraction. The untouched PDF/EPUB is the quote authority. Preserve legend-marker meaning in cleaned-unit annotations.
- Schema-1 generations are BM25-only. Keep them usable when a caller explicitly requests `bm25`; require a new ingestion before dense or hybrid search.
- Exclusions are path-based, stored outside generations, and applied to BM25, dense, source-list, and passage results immediately. Ingestion snapshots the exclusion revision and omits excluded documents from both indexes. Inclusion can only restore current retrieval immediately if the current generation still contains that source.
- Bundle import must reject traversal, symlinks/non-regular members, duplicate entries, checksum/schema/model failures, project-ID mismatch, unsafe manifest paths, and differing bytes at an existing source path. Never export live indexes, locks, logs, runtime files, or model caches. Import intentionally replaces reviewed metadata/exclusions with the validated bundled copies and must disclose that behavior.

## Metadata and extraction contract

- Resolve each field independently. Automatic PDF precedence is valid visible front matter, valid embedded metadata, then the filename stem for title only. Automatic EPUB precedence is valid OPF metadata, visible title/byline, then the filename stem for title only. Apply reviewed metadata afterward as the authoritative read-time per-field overlay.
- Keep automatic metadata rules generic and conservative. Never add a source-, title-, author-, or publisher-specific extraction exception to fix one document. Expose uncertainty through provenance and warnings, then use a reviewed `set_source_metadata` override for the exceptional document.
- Never infer authors from filenames. Reject DOI/URL/export-junk titles, move a detected DOI to its own field, and expose per-field provenance plus concrete review warnings. Provenance and those inspectable warnings are the complete uncertainty model.
- Apply the deterministic English-oriented text-health classifier to complete extraction units and automatically extracted titles/authors. Retain only locator/reason diagnostics for rejected units, never guessed repairs or their garbage text. Reviewed metadata remains authoritative. Apply the same guard at retrieval time for older generations.
- Withhold text only for corruption evidence: replacement characters, private-use or unassigned code points, or a known damaged encoding sequence. A single replacement character counts only with corroborating corruption evidence. Script mixing and non-Latin dominance are advisory `text_notes` that never withhold a unit, a chunk, or a passage, because English-language scholarship legitimately quotes other scripts.
- Fold only formula-font letters (Mathematical Alphanumeric Symbols) and the alphabetic presentation ligatures (fi, fl, ff) so a typed query can match the printed text. Leave every other character canonical: do not apply global NFKC, because it would also fold superscripts, subscripts, and symbols that carry meaning in citations and notation.
- Audit every built chunk against the embedding model's token limit, record the count and a truncation flag on the chunk record, and report the aggregate in build metrics and per hit. When the tokenizer cannot be inspected, record the audit as unavailable rather than failing the build or inventing a count.
- Treat the embedding inference batch size as a padding decision, not a throughput dial: FastEmbed pads every sequence to the longest member of its batch, so a larger batch makes short chunks pay for the longest one. Keep `EMBEDDING_INFERENCE_BATCH_SIZE` at 1 unless a measurement on the target corpus says otherwise, and record any change in `PLAN.md` §10.6. Do not raise it "to go faster" without measuring padded tokens, and keep `--embedding-threads` unset by default because its optimum is machine-specific.
- Disclose withholding instead of hiding it. Report reason codes, counts, and example chunk IDs in the search response, and record corpus-level withheld counts and reasons in the generation build metrics that `status` returns.
- Keep checkpoints durable at the finest practical granularity: one extracted document, one PDF scan batch, one extraction-unit batch, one embedding batch, and one index batch. Reduce the cost of each durable write rather than widening the resume granularity, so a crash never redoes more than one bounded batch. Each phase's batch size is a documented constant — `PDF_PAGE_BATCH_SIZE`, `CHUNK_BATCH_UNITS`, `EMBEDDING_BATCH_SIZE` — and it must stay bounded and small enough that redoing one batch is cheap. Never replace a bounded batch with a whole-phase commit.
- Exclude every nonempty chunk that contains no Unicode alphanumeric content, both while ingesting and at retrieval time for older generations. Text, numbers, and formulas containing at least one letter or digit are not classified as symbol-only; the other extraction-artifact rules still apply.
- Treat the portable reviewed-metadata file as an authoritative read-time overlay on the selected immutable generation. Apply it consistently to source listings, search results, reference groups, citations, neighboring passages, and category/keyword filters without rewriting generation files.
- Do not mark a selected generation stale merely because its metadata snapshot differs from the current reviewed overlay. Report that the overlay is active. A source absent from the selected generation still requires ingestion before any of its metadata can appear in retrieval.
- Do not copy mutable category or keyword values into a dense index. Translate current reviewed filters through selected-generation document IDs and verify them against the overlaid canonical documents and chunk records.
- Omitted fields remove their prior reviewed overrides. If an old generation cannot recover automatic bibliography hidden by a removed override, prefer a safe missing/filename fallback with an explicit warning; a later ingestion may recover automatic metadata from the original.
- Inspect the first five text-bearing PDF pages for identity. Keep physical page and available page-label locators.
- For EPUBs, retain the spine section identity and a deterministic XHTML block position for every semantic unit. Preserve an existing element ID or named anchor as an exact fragment when available; never label a synthetic block path as an EPUB CFI or imply quotation-level precision.
- Use coordinate blocks to restore column order, remove repeated margins/page numbers, and distinguish prose, lists, tables, and figures. Do not infer visual relationships not expressed by captions, legends, or labels.
- Hash all discovered sources before ingestion decides whether work is needed. Exact source bytes, source-exclusion decisions, chunk settings, processing policies, and model fingerprints are required for a no-op. Reviewed metadata is excluded from build/checkpoint identity because it is a read-time overlay.
- Reuse extraction units/chunks only for a source with matching bytes, automatic-metadata storage policy, and processing fingerprints. Generation artifacts retain automatic bibliography beneath the reviewed overlay. Reuse a vector only when canonical `contents`, model revision, and dimension match exactly. Legacy text fields may be read only to upgrade an older generation.
- Always reconstruct complete BM25 and dense indexes for a changed generation; never update selected indexes in place. `force_recompute=true` disables all document, chunk, and vector reuse.
- Check the soft work budget only between atomic units: source hashes, fixed eight-page PDF scan/extraction batches, EPUB spine sections, extraction-unit chunking, 64-text embedding batches, and 64-point dense uploads. Treat BM25 finalization as one restartable unit and re-hash all sources before activation.

Do not move the dense backend implementations into the vanilla gateway or patch UltraRAG for this feature. The research-specific integration deliberately lives in this repository so vanilla can continue tracking upstream safely.

## Safe change rules

- Use `pathlib.Path`, type hints, and JSON-serializable tool results.
- Keep blocking extraction and filesystem scans outside the event loop.
- Serialize every project operation with both the in-process service lock and the cross-process `project.lock`.
- Prefer new immutable generations to in-place index mutation.
- Validate a new artifact before updating a pointer to it.
- Keep model downloads lazy: the embedding model is needed during ingestion; the reranker model only when `rerank=true`. In offline mode, require an existing shared cache, with read-only fallback to an existing legacy project-local cache. Never migrate or delete that legacy cache automatically.
- Do not expose underlying vanilla tools through this server.
- Do not silently skip a selected PDF/EPUB that fails extraction; fail the new generation and leave the previous current generation intact.
- Empty, symbol-only, or corrupt upstream chunk records may be discarded when their source still has at least one searchable chunk; record each discarded count separately. Fail the build when filtering leaves a source with none.
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

Research UI adapter changes must cover safe source-file resolution, forwarding to the public MCP tools, and the real UI host against an existing project without mutating its sources. Shared workspace, JSON validation, capability, and same-origin changes belong in `ui-ultra-rag-mcp` and must pass that package's own tests before updating the pinned commit here.

For source or retrieval changes, the integration test must still launch the real vanilla stdio server, build BM25 and dense indexes, run hybrid and dense search, retrieve the known passage, and prove that a neighboring Markdown file was excluded. It must then restart offline and repeat hybrid reranked search from the caches. Unit tests must cover RRF and failure atomicity without depending on model downloads. They must also cover no-op ingestion, additions, changes, removals, reviewed metadata/exclusions, same-size/same-mtime byte changes, forced regeneration, exact-text vector reuse, and lean final artifacts. They must also prove that post-ingestion metadata corrections immediately affect every read surface and filter mode without modifying generation files.

For significant extraction changes, also test a representative real collection without writing into its source directory.

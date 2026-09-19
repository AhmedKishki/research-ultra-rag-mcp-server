# Research UltraRAG MCP — Review Findings and Strengthening Plan

Status: review complete; no strengthening change is approved or implemented yet.

Reviewed revision: `36ccdd5` (`main`) **plus the uncommitted working tree**
(22 modified/added files, `+5178 / −1036`). All findings describe the working
tree, because that is what runs today.

Review type: read-only. No source, project state, or dependency was modified
while producing this document. Measurements were taken against the live project
`/mnt/DATA/projects/ai-and-fetishism` using read-only paths only.

This document records observations, evidence, and a proposed sequence. The
actionable checklist derived from it lives in `TODO.md`. Deferred product work
that predates this review remains in `ROADMAP.md`.

## 1. Method

- Read `README.md` (657 lines), `AGENTS.md`, `AGENT_GUIDE.md`, `ROADMAP.md`,
  `NOTICE`, `pyproject.toml`, the CI workflow, and all 16 modules in `src/`
  (~11,084 lines, 301 functions).
- Ran `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`,
  AST-based function-size analysis, and targeted benchmarks for durable writes,
  retrieval quality gates, Qdrant build amplification, and embedding token
  limits.
- Inspected the pinned upstream: installed `ultrarag` `0.3.0.2`, the
  `vanilla-ultra-rag-mcp` gateway package, and the managed UltraRAG snapshot at
  `~/.cache/vanilla-ultra-rag-mcp/runtime/UltraRAG-3a709a2.../servers/corpus/`.
- Called the live `status` tool to obtain real build metrics.

## 2. What UltraRAG is, and what this server uses

UltraRAG (`OpenBMB/UltraRAG`, a joint project of THUNLP, NEUIR, OpenBMB,
AI9stars and contributors; Apache-2.0; pinned at `0.3.0.2` commit
`3a709a2aea3fbe46acca59c422621c94b6e86857`) is a modular RAG toolkit exposed as
independent MCP servers per stage: `corpus`, `retriever`, `reranker`,
`generation`, `prompt`, `benchmark`, `evaluation`, `memory`, `router`, `custom`.
The vanilla gateway pins the upstream tarball by SHA-256 and tree hash, then
runs the selected upstream servers as keep-alive stdio child processes behind
one namespaced gateway.

This server consumes a deliberately narrow slice of that surface:

| Used from UltraRAG | Deliberately not used |
|---|---|
| `corpus_chunk_documents` (chonkie `TokenChunker` + tiktoken `gpt2`) | UltraRAG parsing/MinerU extraction, office-format paths |
| `retriever_retriever_init`, `retriever_bm25_index`, `retriever_bm25_search` | UltraRAG FAISS/dense retriever, reranker, generation, prompt, benchmark, evaluation, memory, router |

The research layer re-implements extraction, chunk enrichment, embeddings
(FastEmbed `bge-small-en-v1.5`), the vector store (embedded Qdrant), fusion, and
citation *around* UltraRAG rather than modifying it. `AGENTS.md` forbids pushing
research behaviour into the vanilla repository, and the code respects that.
`README.md` and `NOTICE` carry the required attribution, upstream links, license
information, and the independent/unofficial disclaimer.

## 3. Does it do what it claims? — verified

| Claim | Verdict | Evidence |
|---|---|---|
| Exactly nine MCP tools; no arbitrary output paths | Pass | `create_server`; integration test asserts the exact tool set, parameter names, per-parameter descriptions, and `additionalProperties: false` |
| PDF/EPUB only, recursive, Markdown ignored, symlinks rejected | Pass | `ALLOWED_SOURCE_EXTENSIONS`; `scan_sources` rejects symlink/escape; live `ignored_extensions: {".md": 52}` |
| One process serves one project; project-scoped state | Pass | `resolve_config` containment checks for sources, portable root, runtime root |
| All state under `<project>/.research-rag`; only model binaries shared | Pass | `config.py` path properties; live roots match the documented tree exactly |
| Immutable generations; pointer switches only after both indexes pass | Pass | artifact validation + BM25 probe + Qdrant count/dimension check, then journal, then `os.replace`, then `current.json` last |
| Durable resumable checkpoints; budgeted calls return `in_progress` | Pass | `_advance_ingestion` phase machine; live `resume_count` accounting |
| Compatible reuse; `force_recompute` bypasses it | Pass | Live build reused 52/53 documents, 7670/8102 chunks, 8102/8102 vectors (`created_vector_count: 0`) |
| Exact input match returns `unchanged` with no new generation | Pass | `source_hashing` → `source_set_matches` → staging removed, `status: "unchanged"` |
| Reviewed metadata applies immediately without re-ingest | Pass | Read-time overlay; `metadata_storage_policy` excluded from generation identity; live `metadata_overlay_active: true` |
| Explicit, reversible exclusion enforced on every retrieval surface | Pass | Excluded IDs pushed into BM25 filtering *and* a Qdrant `must_not` filter; live project has 2 reasoned exclusions |
| English-oriented corrupt-text / symbol-only rejection, no invented text | Pass | `text_corruption_reasons` + `has_searchable_alphanumeric_content`, applied at ingest **and** retrieval |
| Hybrid = weighted RRF k=60, 1.25/1.0; dense ≥ 0.72 cosine | Pass | `_fuse_rankings`; echoed in `manifest.retrieval` and the response `fusion` block |
| Reference view caps passages per `source_id` inside `top_k` | Pass | Grouping after gates/fusion/rerank; graded fixture asserts nDCG 1.0 grouped vs 0.333 flat |
| Bundles validate paths, checksums, project ID, schemas, models | Pass | `stage_bundle` + `install_staged_sources`; export refuses stale/upgrade-required generations |
| Loopback-only shared UI reusing the same public tools | Pass | `--host` limited to loopback; adapter calls the nine tools; port 5051 |
| Tool annotations are honest | Pass | read-only for `status`/`search`/`get_passage`; non-idempotent write for `ingest`; destructive for `set_source_*`/`import_bundle` |
| `ruff check`, `ruff format --check`, `pytest` pass in CI | **Fail** | `ruff check` clean; **`ruff format --check .` flags `bundle.py:583` and `service.py:761`** → current tree would fail its own CI |
| Test suite passes | Pass | 134 passed in 70.58 s |

## 4. Design strengths to preserve

These are the properties that make the "does it do what it claims" answer a
clear yes. Every strengthening change below must be implemented *without*
weakening them.

- Multi-axis versioning (`schema_version`, `extraction_policy_version`,
  `cleaning_policy_version`, `artifact_policy_version`,
  `retrieval_policy_fingerprint`, `metadata_storage_policy`) so an older
  generation is flagged with specific `upgrade_reasons` while its read surfaces
  keep working. Verified live: four reasons reported, search still functional.
- Immutable-generation protocol: validate artifacts, probe BM25, verify Qdrant,
  write an activation journal, move the staging tree, and only then rewrite
  `current.json`.
- A text-free SQLite sidecar storing only IDs, content hashes, vector ordinals,
  and JSONL byte offsets, so no corpus text is duplicated.
- Reviewed metadata as a read-time overlay that is deliberately excluded from
  generation and checkpoint identity.
- Exclusions enforced on both BM25 and dense paths, and applied to legacy
  generations at retrieval time before re-ingestion removes them.
- Documentation tests that fail the build when README/agent guidance drifts
  from the implemented tool surface.
- Honest self-labelling: `direct_quote_safe: false`, "regression fixture, not a
  claim of project-specific retrieval quality", explicit "no automatic pruning".

## 5. Findings

### P0 — correctness and delivery risks

#### P0-1. CI is red on the current tree

`uv run ruff format --check .` reports `src/research_ultra_rag_mcp/bundle.py`
(line 583) and `src/research_ultra_rag_mcp/service.py` (line 761) as
unformatted. Both are cosmetic (string wrapping), but the CI workflow runs
`uv run ruff format --check .` before `pytest`, so the current tree cannot pass
its own pipeline, and the root `AGENTS.md` requires testing inside the child
repository before a parent pointer update.

#### P0-2. Silent embedding truncation is possible and unmeasured

Chunk sizing uses **GPT-2** tokens capped at 384, but
`EMBEDDING_MODEL = BAAI/bge-small-en-v1.5` has
`max_position_embeddings = 512` **WordPiece** tokens. The two tokenizers are not
proportional. Measured over all 8,102 chunks of the live corpus:

- GPT-2 tokens: mean 270, p95 384, **max 387**;
- bge WordPiece tokens: mean 263, p95 387, **max 3,417**;
- **8 chunks (0.10%) exceed 512 WordPiece tokens.**

FastEmbed truncates those inputs, so a chunk's dense vector can represent only
its prefix while BM25 indexes the whole `contents`. That is a silent
lexical/semantic asymmetry inside a retrieval system, and nothing in the
manifest, `status`, or `search` output reports it. The GPT-2 maximum of 387 also
shows the documented "must be between 50 and 384" is a requested maximum, not an
enforced one.

Implemented in the working tree: the dense backend exposes
`embedding_token_counts` through a dedicated non-truncating tokenizer, the
service audits every built chunk and records `embedding_token_count` with a
`dense_truncated` flag, build metrics report `dense_token_audit`,
`dense_audited_chunk_count`, `dense_truncated_chunk_count`,
`embedding_maximum_tokens`, and `maximum_embedding_token_count`, and each search
hit carries the per-chunk values plus a `dense_fidelity` summary. Re-verified on
the reference corpus through the shipped path: 8,102 chunks audited, **8 over the
limit, maximum 3,417 tokens**. The audit degrades to `unavailable` when the
model cache is missing rather than failing a build, and over-limit chunks are
flagged rather than split (D4/Decision 4 note in `TODO.md`).

#### P0-3. Nothing protects the chunk-size / embedding-limit invariant

All unit and integration tests use a fake dense backend, so no test would catch
a future chunker setting or embedding-model change that exceeds the model's
input limit. A property test over the real tokenizer pair (as used for P0-2)
belongs in `tests/`.

#### P0-4. The text-health policy silently withholds 29 chunks of the reference corpus, including quotations from other languages

Measured by running the current working-tree policy over all 8,102 chunks of the
live generation: **29 chunks (0.36%) are flagged**, across 10 documents.

| Reason combination | Chunks |
|---|---|
| `private_or_unassigned_characters` only | 15 |
| `replacement_characters` only | 5 |
| `replacement_characters` + `private_or_unassigned_characters` | 2 |
| `replacement_characters` + `private_or_unassigned_characters` + `non_latin_dominant` + `mixed_script_text` | 3 |
| `non_latin_dominant` only | 2 |
| `non_latin_dominant` + `mixed_script_text` | 1 |
| `replacement_characters` + `non_latin_dominant` | 1 |

Two consequences:

1. **26 of the 29 list corruption evidence**, and 25 of them are correctly
   withheld once the corroboration rule is applied: replacement characters,
   private-use or unassigned code points, and in one "afterlives of toxic waste
   sites" chunk a blend of roughly six rare scripts (CJK, Gurmukhi, Malayalam,
   Devanagari, Hangul, Cyrillic) that is the signature of a broken PDF font map.
2. **4 chunks are restored by the revised policy**: the two Russian bibliography
   entries from a Russian edition of *Capital* (for example
   `Покровскій], Василій [Иванович]`), one further script-mixed chunk, and the
   Greek quotation from Sophocles about money being the root of evil, whose
   single unreadable glyph no longer has corroborating corruption evidence. That
   quotation is exactly the kind of evidence this corpus exists to find.

Because the retrieval path re-runs the same gate, all 29 chunks are currently
**invisible to `search`** on the live project, and nothing in the response says
so beyond aggregate counters. For a tool whose purpose is finding evidence,
silent evidence loss is a correctness problem rather than a tuning preference.

Recommended fix, consistent with the English-only scope in section 12:

- keep rejection for genuine corruption evidence (`replacement_characters`,
  `private_or_unassigned_characters`, `known_mojibake`);
- demote `non_latin_dominant` and `mixed_script_text` from rejections to
  **flags** carried on the chunk and surfaced in search results, which restores
  the legitimate Cyrillic entries;
- report withheld counts and reasons in `search` and `status`, with a way to
  inspect the withheld chunks rather than only their count.

Implemented in the working tree (`EXTRACTION_POLICY_VERSION` 7): `_text_signals`
separates corruption evidence from advisory script notes, `text_script_notes`
reports the notes, every returned passage carries `text_notes`, each search
response discloses `withheld_candidates` with reason codes and example chunk IDs,
and generation build metrics record `withheld_chunk_count` with
`withheld_chunk_reasons` for `status`.

#### P0-5. Search normalization loses math-styled letters and presentation ligatures

The reference corpus is English, but its text is not folded for matching.
Measured over the same 8,102 chunks:

- **39 chunks** contain Mathematical Alphanumeric Symbols: `𝑀𝑗𝑀𝑗𝑗𝑀` is stored
  as those code points, so a plain-text BM25 query for `MjMjjM` cannot match it.
- **723 chunks** contain ligature or presentation forms: `ﬁ`, `ﬂ`, and `ﬀ`
  (U+FB01, U+FB02, U+FB00) remain distinct under NFC, so queries for `fi`, `fl`,
  and `ff` miss them.
- **6,895 chunks (85%)** contain at least one non-ASCII character, so this is
  not a corner case.

`normalize_reading_text` uses NFC, which performs canonical but not
compatibility folding. NFKC maps the math-styled letters to plain ASCII and the
presentation ligatures to their letter sequences, while leaving ordinary letters
such as `é`, `æ`, and `œ` intact.

Recommended: apply NFKC, or a targeted fold limited to the Mathematical
Alphanumeric and Alphabetic Presentation Forms blocks, in the text path; verify
on a judged query set that recall improves and nothing regresses; and note that
this changes stored `contents`, so it requires a `cleaning_policy_version` bump
and one regeneration.

Implemented in the working tree (`CLEANING_POLICY_VERSION` 3): a targeted fold
covers Mathematical Alphanumeric Symbols and the Alphabetic Presentation Forms
and runs before NFC, while superscripts, subscripts, accented letters, and
symbols stay canonical. Measured on the reference corpus: 751 chunks change
text, and no chunk retains a formula letter afterwards (39 before).

### P1 — measured efficiency findings

#### P1-1. `qdrant_indexing` is 99.1% of build wall-clock and is device-bound

Live build metrics for 8,102 chunks (`status.last_build_metrics`):

```
total             3067.35 s
  qdrant_indexing 3040.73 s   <-- 99.13%
  extraction        12.13 s
  chunking           9.11 s
  source_hashing     2.23 s
  bm25_indexing      1.77 s
  embedding          0.016 s   (all 8102 vectors reused)
```

Reproducing the same code pattern (one `QdrantClient` per 64-point batch on
embedded storage) on NVMe measured 8.6 s for 2,048 points — approximately
**34 s extrapolated for 8,102 points**, i.e. the live run was ~90× slower
(375 ms/point versus ~4 ms/point). Measured Qdrant write amplification is
**24.8×** (39 MB written for 1.6 MB of logical vectors); the live index is three
files totaling 33.5 MB of `storage.sqlite`. The project runtime lives on
`/dev/sda1` ext4 (`/mnt/DATA`, HDD class: 34.6 MB/s cold read of
`embeddings.npy`).

Conclusion: the dense build is not CPU-bound; it is bound by many small
synchronous SQLite/WAL commits on slow storage, multiplied by per-batch
open/close and 64-point transactions.

Reproduced on the bounded 20-source benchmark (section 10.4): the identical
1,073-point index cost **4.12 s on NVMe and 310.97 s on HDD (75.5×)**, i.e.
**3.8 ms versus 290 ms per point**, independently confirmed at 289 ms/point in a
second HDD build. Because the code path is identical, the difference is storage,
not logic — which means Step 2's batching changes are worth doing but the
*device* is the first-order factor.

#### P1-2. Chunking performs one MCP round trip per extraction unit

In the `chunking` phase each extraction unit gets its own input JSONL, one
`corpus_chunk_documents` call, one output write, two unlinks, a `state.json`
write, and a full checkpoint write. Measured durable-write cost on this machine:
`atomic_write_json` **2.65 ms** small / **6.40 ms** at 500 sources (83 KB), and
`atomic_write_jsonl` with 8 records **2.74 ms** — roughly **11–16 ms of pure
write overhead per unit**, before the RPC and chunker work. The benchmark
confirms the amplification: 501 units cost **21.41 s on NVMe versus 397.59 s on
HDD (18.6×)**, and chunking a single added document cost 5.71 s versus 179.02 s
(**31.4×**) — with no CPU-bound work involved, only durable writes and RPCs.

#### P1-3. The per-unit checkpoint rewrites the whole source inventory

`checkpoint.json` carries `source_inventory` and `source_digests` for every
source, and `_write_checkpoint` runs after every unit, giving
**O(units × sources)** serialization and fsync cost. Measured cost rose from
2.65 ms to 6.40 ms as the inventory grew to 500 sources. Per-source progress
already lives in `work/sources/<key>/state.json`, so the heavy data does not
need to be rewritten continuously. The benchmark shows the aggregate effect in
the `assembly` phase, which is otherwise trivial work: **0.14 s on NVMe versus
4.78 s on HDD (34×)** fresh, and 0.19 s versus 7.15 s (38×) for the incremental
add.

#### P1-4. Retrieval re-normalizes and re-scans every candidate on every query

Measured per 755-character candidate: `text_corruption_reasons` **0.362 ms**,
`has_searchable_alphanumeric_content` **0.079 ms**, `normalize_inline_text`
**0.080 ms**, `_content_tokens` **0.050 ms** — **0.551 ms per candidate**. At the
default depth (32 candidates) that is ~18 ms per query; at
`MAXIMUM_CANDIDATES = 200` (reference view) it is ~110 ms per query. The same
text is normalized three to four times per query, and `text_corruption_reasons`
calls `unicodedata.name()` for every alphabetic character.

Correction from measurement: replacing the 31-entry script table with cheap
code-point ranges was measured at only **1.4× faster** (0.330 ms → 0.236 ms per
755-character chunk), because `unicodedata.name()` is C-implemented. The
dominant remaining cost is the double normalization plus the per-character
`isalnum` and `category` scans. So the reason to touch this code is recall
fidelity (P0-4), not speed; the speed win comes from precomputing the verdict
once at build time (Step 3), which removes the entire cost from the query path.

#### P1-5. `_bm25_ranking`'s widening loop restarts all work

Each retry re-issues `search_bm25(query, requested)` with a doubled `requested`
and then rebuilds `ranking`, `used`, `resolved`, and `rejected` from scratch
over every returned passage, re-running `_matches_filters` and
`_content_tokens(_chunk_text(chunk))` on passages already accepted. Worst case
is O(n log n) redundant work whenever filters are selective.

#### P1-6. `_status` (a full source-tree walk) runs inside every `search`

`search` calls `self._status()` to populate `stale` and
`generation_upgrade_required`. Measured `scan_sources` on the live project:
**8.84 ms for 55 sources (~160 µs/file)**, i.e. linear in source count
(~160 ms at 1,000 sources, ~1.6 s at 10,000) on **every query**, while holding
the cross-process project lock. The same walk runs on `status`, `list_sources`,
`set_source_metadata`, and `set_source_inclusion`. `search` also re-parses the
manifest that `_status` has just parsed (0.82 ms for the live 101 KB manifest,
plus the separate parse `load_reuse_snapshot` performs during ingestion).

#### P1-7. Dense filtering materializes document-ID lists into every query

Category/keyword filters are translated into the full set of matching
`document_id` values and passed to Qdrant as `MatchAny`, so a broad filter ships
potentially thousands of IDs per query and the ID set is rebuilt per call.

#### P1-8. Duplicate hashing on bundle import

`stage_bundle` hashes every already-present destination source (~line 670) and
`install_staged_sources` hashes them again (~line 789): two extra full passes
over the original corpus on import.

#### P1-9. Smaller O(corpus)-per-request patterns

`ArtifactLookup._connect()` opens a fresh SQLite connection per method call
(three to four per `search`, more inside `_bm25_ranking`); `_effective_documents`
rebuilds the entire document-metadata map per request (used by every `search`,
every `get_passage`, and once per `set_source_metadata`); PDF page scanning
re-opens the document once per 8-page batch (~63 opens for a 500-page book).
Each is small today and linear in corpus size.

#### P1-10. Generation retention is unbounded and expensive

Each generation is a complete copy: the live project keeps **101 MB per
generation against a 135 MB source corpus**, with two generations retained
(202 MB) and no pruning. Compounded with P1-1, this deserves a P1 slot rather
than only the roadmap entry.

### P2 — maintainability, robustness, and contract clarity

#### P2-1. Monolithic methods

Measured by AST: `_advance_ingestion` **1,214 lines**, `search` **437**,
`bundle.stage_bundle` **347**, `server.create_server` **287**, `_status` **229**,
`import_bundle` **206**, `export_generation_bundle` **184**,
`_front_matter_identity` **172**, `generation_artifacts_are_valid` **141**,
`_recover_pending_activation` **133**. `_advance_ingestion` mixes nine phases
with per-format branching and repeated checkpoint/`budget_expired()` epilogues.
`ruff` complexity rules (`C901`) are not enabled, so nothing resists further
growth.

#### P2-2. Activation validation is all-or-nothing

`_validate_generation_for_activation` collapses artifacts, BM25 probe, and
Qdrant verification into one `except Exception` and reports
"Generation retrieval indexes failed validation" without saying which stage
failed, even though the appropriate remedy differs per stage.

#### P2-3. `stale` semantics are overloaded in the no-generation branch

With sources present and no generation, `_status` returns `ready: false,
stale: true` and no `changes` key, while `AGENT_GUIDE.md` step 4 instructs the
agent to explain the reported source or inclusion changes. There are none.

#### P2-4. Legacy generations are searchable but can never be re-promoted

Verified live: `generation_artifacts_are_valid` returns `False` immediately for
the selected generation because its `documents` records predate `source_id`
(schema 5, extraction policy 4, artifact policy 1, no
`metadata_storage_policy`). Search works; activation, reuse, and export do not.
This is correct behaviour, but "earlier successful generations remain usable"
reads more broadly in the README than the code delivers.

#### P2-5. `SCHEMA_VERSION` stays at 5 while four policy axes moved

The design works — four `upgrade_reasons` are reported correctly — but nothing
enforces that a future policy change also bumps a policy constant, and
`generation_is_reusable` silently degrades to "no reuse" if one is forgotten.

#### P2-6. Documentation precision

- `chunk_size` "must be between 50 and 384" is a requested maximum; chonkie
  produced a 387-GPT-2-token chunk in the live corpus (see P0-2).
- "An exact input match is a true no-op" is true about *writes*. It is still an
  O(corpus) read and validation pass: measured ~0.3 s warm for 8,102 chunks and
  4,016 units (SHA-256 of 48.1 MB JSONL 0.12 s, full JSON parse of every record
  0.22 s, vector finite scan negligible), plus re-hashing all 135 MB of sources.
- "Earlier successful generations remain usable" needs the P2-4 qualifier.

#### P2-7. Broad `except Exception` at two durability boundaries

`bundle.py:774` (`except Exception: rmtree; raise`) and `service.py:3446` can
mask programming errors in the failure record, weakening the diagnostics the
design otherwise invests in heavily. Most other uses sit at process/CLI
boundaries and are justified.

### P3 — evaluation and product gaps

- Retrieval quality is guarded only by a synthetic graded fixture with a fake
  dense backend. It validates the grouping *mechanism* (grouped nDCG 1.0 versus
  flat 0.333), and `README.md` correctly calls it a regression evaluation rather
  than a quality claim. No real-corpus relevance judgements exist, so BM25
  versus dense versus hybrid versus reranked quality is unmeasured and the
  constants (1.25 / 1.0 / k=60 / 0.72 / 384 / 64) are asserted rather than
  evaluated. The live 53-source, 8,102-chunk project is a ready candidate.
- No generation listing, pruning, or rollback, and no disk-space pre-check.
- English-only embedding and text-health heuristics are documented but untested
  against a non-English corpus.

## 6. Measurement appendix

Environment: Linux, Python 3.12, `/` on NVMe ext4, project runtime on
`/dev/sda1` ext4 (HDD class), `~/` model cache present (152 MB).

| Measurement | Result |
|---|---|
| Full test suite | 134 passed in 70.58 s |
| `ruff check .` | clean |
| `ruff format --check .` | 2 files would be reformatted |
| Live corpus | 55 discovered / 53 indexed sources, 8,102 chunks, 4,016 units |
| Live build total | 3,067.35 s (qdrant 3,040.73 s = 99.13%) |
| Qdrant writes | 24.8× amplification; ~34 s extrapolated on NVMe for 8,102 points |
| Cold read, `embeddings.npy` (12.4 MB) | 34.6 MB/s |
| `scan_sources` | 8.84 ms for 55 sources (~160 µs/file) |
| `load_current_generation` (manifest parse) | 0.82 ms for a 101 KB manifest |
| Corpus validation, warm | ~0.3 s for 8,102 chunks + 4,016 units + 12.4 MB vectors |
| `atomic_write_json` | 2.65 ms small; 6.40 ms at 500 sources (83 KB) |
| `atomic_write_jsonl` (8 records) | 2.74 ms |
| Retrieval gate cost | 0.551 ms per 755-character candidate |
| Embedding token limits | 8/8,102 chunks exceed 512 WordPiece tokens; GPT-2 max 387 |
| Retained generations | 2 × 101 MB against a 135 MB source corpus |

| Reference-corpus text study (all 8,102 live chunks, current policy) | Result |
|---|---|
| Chunks withheld by the text-health policy | 29 (0.36%) across 10 documents |
| Of those, with genuine corruption evidence | 26 |
| Of those, flagged only for script mixing | 3 (2 are legitimate Cyrillic bibliography entries) |
| Chunks containing any non-ASCII character | 6,895 (85%) |
| Chunks below a 0.75 Latin-letter ratio | 20 |
| Chunks containing Mathematical Alphanumeric Symbols | 39 |
| Chunks containing ligature or presentation forms | 723 |
| English-only script-classification speedup | 1.4× (0.330 ms → 0.236 ms per chunk) |

| Bounded 20-source benchmark (6.1 MB, 1,073 chunks, 501 units) | NVMe | HDD |
|---|---|---|
| Fresh build wall clock | 259.1 s | 1,344.7 s |
| Fresh build `chunking` | 21.41 s | 397.59 s |
| Fresh build `qdrant_indexing` | 4.12 s | 310.97 s |
| Qdrant cost per point | 3.8 ms | 290 ms |
| Single-source add wall clock | 96.1 s | 950.7 s |
| Single-source add `qdrant_indexing` (all 1,505 points) | 5.91 s | 434.78 s |
| Single-source add `chunking` | 5.71 s | 179.02 s |
| Reused documents / chunks / vectors in the add | 20 / 1,073 / 1,073 | 20 / 1,073 / 1,073 |
| Runtime footprint after two generations | 27 MB | 27 MB |

| Live-project retrieval (53 sources, 8,102 chunks) | Result |
|---|---|
| `verify` read-only run | `status: passed`, 5 hits from 3 references |
| Client init including gateway startup | 2.73 s |
| First (cold) hybrid search | 9.19 s |
| Warm hybrid search, `top_k=8` | 0.52–0.68 s |
| Reference view / reranked / BM25-only / dense-only | 1.03 / 2.27 / 0.11 / 0.50 s |
| `status` warm | 0.019 s |

## 7. Proposed execution order

Each step is independently testable and must leave the repository green. Any
change that alters generation artifacts requires a policy-version bump so
existing generations are flagged for regeneration rather than silently reused.

### Step 0 — make the tree shippable

Format the two files; confirm `ruff format --check .`, `ruff check .`, and
`pytest -q` all pass; commit inside the child repository before any parent
pointer update, per the collection-root `AGENTS.md`.

### Step 1 — close the retrieval-fidelity gap (P0-2, P0-3)

Compute the embedding-tokenizer length at enrich time; persist a
`dense_truncated` flag in the chunk record and the SQLite sidecar (with a
`LOOKUP_SCHEMA_VERSION` bump); split or flag over-limit chunks; report an
aggregate `dense_truncation_risk` in `ingest` metrics and `status`; add a
tokenizer-pair property test.

Validation: a corpus that previously contained 8 over-limit chunks reports 0
after re-ingestion, and legacy generations expose the flag without re-ingestion.

### Step 2 — the measured performance work (P1-1, P1-2, P1-3)

Reuse one `QdrantClient` across the whole index phase; make upload batches
time-boxed rather than a fixed 64 points; checkpoint once per bounded batch of
extraction units; split the immutable inventory from the mutable progress
journal; document fast-local-storage requirements for `.research-rag/runtime`.

Validation: re-ingest the live 53-source corpus and compare
`last_build_metrics.phase_timings_seconds` against the recorded baseline
(`qdrant_indexing` 3,040.73 s, `chunking` 9.11 s), asserting no change in reuse
counts or in the `unchanged` path.

### Step 2c — explicit derived-state location (P1-12)

Decide the `AGENTS.md` invariant question (D5) first. If the invariant is
amended, add and validate `--runtime-root`; if not, document the
fast-local-storage requirement in the README storage and limitations sections.

### Step 2d — embedding throughput (P1-13)

Measure the ONNX session configuration, padding behaviour, and thread count
against the recorded 0.16 s-per-chunk baseline, and only then decide whether an
accelerator or a smaller model profile is justified. This runs last in Step 2
because storage currently masks it.

### Step 3 — per-query CPU (P1-4, P1-5)

Precompute chunk health verdicts and token counts into the sidecar at build
time and filter on the stored flag at query time; accumulate BM25 widening
state incrementally instead of restarting; pass the already-parsed manifest into
`_status` instead of re-reading it.

Validation: a benchmark test showing candidate-gate cost no longer scales with
the number of queries, with the existing rejection counters unchanged.

### Step 4 — remove per-request O(corpus) work (P1-6, P1-7, P1-9)

Cache the staleness verdict behind a cheap directory signature (with a short TTL
or an explicit opt-out on `search`); move dense filtering server-side; reuse
SQLite connections; cache the effective-document map per manifest and metadata
revision.

Validation: an integration test asserting `search` performs no source-tree walk
when staleness is not requested.

### Step 5 — decomposition for safety (P2-1)

Split `_advance_ingestion` into one handler per phase behind a dispatch table,
then enable `ruff` `C901` with a threshold so the shape cannot regress.

Validation: the existing 134 tests pass unchanged, plus new per-phase tests for
`budget_expired` and cancellation at each boundary.

### Step 6 — real evaluation (P3)

Build a judged query set on the live corpus and record
precision/recall/nDCG for BM25, dense, hybrid, and reranked as a checked-in
report before any tuning of retrieval constants.

### Step 7 — contract polish and retention

Generation listing/pruning with an explicit keep-last-N policy; structured
activation-failure reporting; the `stale` semantics fix; the README precision
fixes; narrowing the two broad `except Exception` sites; and the `AGENTS.md`
documentation-responsibility entries for `PLAN.md` and `TODO.md`.

## 8. Capability verification: requested capabilities

### 8.1 Resumable ingestion checkpoints — already implemented

Every build writes `<project>/.research-rag/runtime/staging/<build-id>/checkpoint.json`
and advances a phase machine (`source_hashing`, `extraction`, `chunking`,
`embedding`, `vector_assembly`, `bm25_indexing`, `qdrant_indexing`,
`source_revalidation`, assembly). `status.ingestion_progress` reports the
unfinished build, `resume_count` counts resumptions, cancellation and timeouts
retain a resumable checkpoint, and a non-resumable failure leaves only a small
failure record. `research-ultra-rag-verify --ingest` and the UI adapter loop the
call until it returns `ready` or `unchanged`.

Limits to document:

- Resumption requires identical inputs and parameters. The checkpoint identity
  fingerprints the source inventory, digests, exclusion revision, chunk
  settings, policy versions, and model revisions, so changing a source or a
  parameter discards the checkpoint and starts a new build. `force_recompute`
  resumes its own matching checkpoint.
- The budget is a soft ceiling: one expensive page, the first model download, or
  a large Qdrant batch can exceed it.
- There is no explicit pause/resume or cancel tool; stopping is done by stopping
  the client, and resumption by calling `ingest` again.

### 8.2 Metadata editable after ingestion — already implemented

`set_source_metadata` writes a replace-all reviewed override that is overlaid at
read time, so `search`, `list_sources`, `get_passage`, citations, and
category/keyword filters use the current values immediately without rewriting
chunk, BM25, vector, or Qdrant files. `list_sources.reviewed_metadata_sources`
enumerates every saved override by stable `source_id`, and the loopback UI
exposes the same editing. Verified live: `metadata_overlay_active: true`,
`generation_metadata_snapshot_outdated: false`.

Limits to document:

- Overrides are per source; there is no bulk or wildcard editing.
- Removing an override can hide an automatic value in an older generation, which
  then falls back to a missing value (or the filename for title) with the
  warning `automatic_metadata_unavailable_after_override_removal`.
- Reviewed metadata is intentionally excluded from generation identity, so it
  never forces a rebuild and never invalidates reuse.

### 8.3 Ingesting one new source without re-ingesting everything — partially implemented

Derivation is incremental; indexing is not.

- **Reused per unchanged document:** extraction units, chunks, and exact-text
  vectors. Verified live: 52/53 documents, 7,670/8,102 chunks, and
  8,102/8,102 vectors reused with `created_vector_count: 0`.
- **Rebuilt in full for every changed generation:** both retrieval indexes.
  `build_bm25` runs over the complete staging corpus, and `qdrant_indexing`
  removes the index directory and re-uploads every chunk batch. There is no
  append, upsert, delete, or payload-update path anywhere in `src/`
  (grepped: no `update_points`, `upsert`, `delete_points`, or `set_payload`).
  `README.md` states this correctly: "The server always reconstructs complete
  new BM25 and Qdrant indexes for a changed generation."
- **No scoping control:** `ingest` always evaluates the whole project. There is
  no per-source or per-directory ingest parameter.

Consequence: adding a single source to the live 8,102-chunk project still costs
a complete Qdrant rebuild, which is the 3,040 s item in P1-1. Measured cost for
a bounded corpus appears in section 10. This is the single largest practical gap
for day-to-day research use, because adding one paper should not cost the same
as building the project from scratch.

### 8.4 Where models and derived state are allowed to live

| Artifact | Configurable today | Default | Device in this setup |
|---|---|---|---|
| Embedding/reranker model binaries | Yes: `--model-cache-root`, `RESEARCH_ULTRARAG_MODEL_CACHE_ROOT` | `~/.cache/research-ultra-rag-mcp/models` | home, already SSD |
| Vanilla/UltraRAG runtime snapshot | Yes: `--runtime-cache-root`, `VANILLA_ULTRARAG_CACHE_ROOT` | `~/.cache/vanilla-ultra-rag-mcp` | home, already SSD |
| Project chunks, units, vectors, BM25, Qdrant, staging, logs | **No** | `<project>/.research-rag/runtime` | wherever the project lives (HDD here) |
| Portable identity, metadata, exclusions, catalog, bundles | No (by design) | `<project>/.research-rag/` | project device |

So model flexibility is already implemented and already benefits from the SSD;
the model cache is **not** the bottleneck. What is fixed is the project's
*derived state*, which is where the 3,040 s went. Relocating it is currently
impossible even by symlink: `resolve_config` calls `.resolve()` on
`<project>/.research-rag/runtime` and then requires containment in the portable
root, so a symlinked runtime is rejected at startup with
`Research runtime escapes its project state: <target>`. Demonstrated:

```text
REJECTED symlinked runtime: Research runtime escapes its project state:
/tmp/rr-symlink-test/ssd-runtime2
```

## 9. New proposals (beyond the original P0–P3 list)

### P1-11. Incremental index construction for source additions and removals

- Keep a **full BM25 rebuild** (measured 1.77 s for 8,102 chunks; cheap), and add
  an **append path for Qdrant**, which is where the time goes.
- Prerequisites: stable point IDs derived from `chunk_id` (today's
  `id = offset + index` is positional, so it changes whenever the corpus
  changes); explicit deletion of points for removed or excluded sources; and a
  delta applied to a *copy* of the previous index so the active generation stays
  immutable.
- The new generation must still be complete and verified: compare expected point
  count and a point-ID digest, and fall back to a full rebuild when the delta is
  large (for example >40% of chunks changed) or the previous index is
  incompatible.
- Expected benefit: adding one paper to the live project becomes a directory
  copy plus a few dozen appended points instead of a 3,040 s rebuild.

### P1-12. Explicit, validated location for derived project state

- Add an opt-in `--runtime-root` / `RESEARCH_ULTRARAG_RUNTIME_ROOT` that places
  derived state on a named device, recorded in `.research-rag/project.json` so a
  moved or missing runtime root is detected rather than silently re-derived.
- Preserve the existing safety guarantees: one project per runtime root, never
  shared between projects, never global index/bundle/log storage, and a stable
  marker file so a mismatched root refuses to start instead of mixing projects.
- This requires an explicit decision (D5) because it conflicts with the
  current `AGENTS.md` invariant "Store every project-owned research-RAG artifact
  beneath `<project>/.research-rag`". The alternative that needs no code change
  is to require the project (or its `.research-rag`) to live on fast local
  storage, and to say so plainly in the README.

### P1-13. Embedding throughput is the remaining bottleneck on fast storage

Measured: 171.75 s to embed 1,073 chunks (about **0.16 s per chunk**) — 71.5% of
a fresh NVMe build — and 66.52 s for the 432 new vectors of an incremental add.
Because `EMBEDDING_BATCH_SIZE = 64` is already applied at the FastEmbed layer,
the remaining levers are the ONNX session's thread count and provider, the
padding strategy for variable-length chunks, and whether a GPU or a smaller
model profile is worth offering. This is worth measuring only after Step 2/2b,
because it is currently masked by storage costs on HDD projects.

## 10. Benchmark evidence (bounded 20-source corpora)

Method: the 20 smallest PDF/EPUB files of `ai-and-fetishism` (6.1 MB) were copied
into two scratch projects with identical content — one on the NVMe system disk,
one on the HDD data disk — and ingested through the public MCP `ingest` tool
while recording `last_build_metrics`. All runs used the default `chunk_size=384`,
`chunk_overlap=64`, `force_recompute=false`. The donor project was not modified,
and its model cache on the home SSD was used throughout.

### 10.1 Read-only verification of the live project

`research-ultra-rag-verify /mnt/DATA/projects/ai-and-fetishism --query
"commodity fetishism and artificial intelligence"` returned `"status":
"passed"`; hybrid retrieval returned 5 passages from 3 distinct references with
correct locators (for example "Between Material and Virtual Worlds: Fetishism
and the Discourse of Capitalism", p. 6), and all rejection counters were zero.
Warm in-session latency measured through the same stdio MCP surface:

| Operation | Time |
|---|---|
| Client init, including server and vanilla gateway startup | 2.73 s |
| First hybrid search (cold; loads the BM25 index into the retriever child) | 9.19 s |
| Warm hybrid search, `top_k=8` (three runs) | 0.52 / 0.68 / 0.53 s |
| Reference view, `top_k=8`, 2 per reference | 1.03 s |
| Reranked hybrid | 2.27 s |
| BM25 only | 0.11 s |
| Dense only | 0.50 s |
| `list_sources` (55 sources) | 0.12 s |
| `status` (warm) | 0.019 s |

Interpretation: retrieval latency is acceptable. The per-query `_status` walk
costs about 19 ms at 55 sources (roughly 3.5% of a warm hybrid query), so P1-6 is
a scaling risk rather than a present bottleneck. The CLI's 15.6 s end-to-end
figure is dominated by process startup and the cold BM25 load, not by the query.

### 10.2 Fresh build on NVMe (20 sources, 1,073 chunks, 501 units)

One `ingest` call; wall clock 259.1 s.

| Phase | Time | Share |
|---|---|---|
| embedding | 171.75 s | 71.5% |
| extraction | 43.05 s | 17.9% |
| chunking | 21.41 s | 8.9% |
| bm25_indexing | 4.40 s | 1.8% |
| qdrant_indexing | 4.12 s | 1.7% |
| assembly / vector_assembly / revalidation / hashing | 0.21 s | 0.1% |

On fast storage with a cold corpus, **embedding dominates**, and Qdrant is
negligible at **3.8 ms per point**. This is the opposite of the live project's
profile, where every vector was reused and Qdrant dominated at **375 ms per
point** (3,040.73 s / 8,102 points). The two together show that the Qdrant cost
is a *device* property amplified by the per-batch open/close pattern, not an
inherent CPU cost.

### 10.3 Adding one source to an existing generation on NVMe

The 21st smallest source (715 KB) was added and `ingest` was called once; wall
clock 96.1 s for a corpus growing from 1,073 to 1,505 chunks.

| Phase | Time |
|---|---|
| extraction | 14.93 s (one document only) |
| chunking | 5.71 s |
| embedding | 66.52 s (432 new vectors) |
| qdrant_indexing | 5.91 s (**all 1,505 points re-uploaded**) |
| bm25_indexing | 0.32 s (**all 1,505 chunks re-indexed**) |
| assembly / vector_assembly / revalidation / hashing | 0.25 s |

Reuse was exact for unchanged material: 20 documents, 1,073 chunks, and 1,073
vectors reused; 1 document, 432 chunks, and 432 vectors rebuilt.

Interpretation: the *derivation* half of the requested capability already works
— adding one paper did not re-extract, re-chunk, or re-embed the other twenty.
The *indexing* half does not: both indexes were rebuilt over the whole corpus.
On NVMe that costs only 6.2 s, but the same full rebuild on the live HDD project
is the measured 375 ms per point, so adding one paper to the 53-source project
still costs roughly the same order as rebuilding it from scratch. This is the
concrete justification for P1-11.

### 10.4 Same builds on the HDD data disk

Fresh build of the identical 20 sources (1,073 chunks, 501 units):

| Phase | NVMe | HDD | Ratio |
|---|---|---|---|
| source_hashing | 0.016 s | 0.026 s | 1.6× |
| extraction | 43.05 s | 251.32 s | 5.8× |
| chunking | 21.41 s | 397.59 s | **18.6×** |
| embedding | 171.75 s | 156.67 s | ~1× (CPU) |
| vector_assembly | 0.031 s | 0.101 s | 3.3× |
| bm25_indexing | 4.40 s | 3.82 s | ~1× |
| qdrant_indexing | 4.12 s | 310.97 s | **75.5×** |
| source_revalidation | 0.019 s | 0.175 s | 9.2× |
| assembly | 0.140 s | 4.777 s | 34× |
| phase sum | 240.1 s | 1,125.4 s | 4.7× |
| wall clock | 259.1 s (1 call) | 1,344.7 s (5 calls) | 5.2× |

Qdrant cost per point is **3.8 ms on NVMe (4.12 s / 1,073) versus 290 ms on HDD**
— a 76× device penalty, independently reproduced at **289 ms/point** in the
second HDD build (434.78 s / 1,505) and consistent with the live project's
**375 ms/point** (3,040.73 s / 8,102).

Adding the same single source to the existing generation:

| Phase | NVMe | HDD | Ratio |
|---|---|---|---|
| extraction (1 new document) | 14.93 s | 152.82 s | 10.2× |
| chunking | 5.71 s | 179.02 s | **31.4×** |
| embedding (432 new vectors) | 66.52 s | 71.23 s | ~1× (CPU) |
| qdrant_indexing (all 1,505 points) | 5.91 s | 434.78 s | **73.6×** |
| bm25_indexing (all 1,505 chunks) | 0.33 s | 0.33 s | ~1× |
| assembly | 0.186 s | 7.147 s | 38× |
| phase sum | 93.8 s | 845.7 s | 9.0× |
| wall clock | 96.1 s (1 call) | 950.7 s (4 calls) | 9.9× |

The unchanged 20 documents, 1,073 chunks, and 1,073 vectors were reused exactly
in both runs.

Conclusions:

1. On HDD, adding one paper to a 21-source project took **15.8 minutes**, of
   which 434.8 s (46%) was a single full Qdrant rebuild and 179.0 s (19%) was
   per-unit chunking — even though nothing else in the corpus changed. This is
   the strongest argument for P1-11.
2. Two phases are storage-bound amplification of patterns this plan already
   targets: `chunking` (18.6× fresh, 31.4× incremental) is the per-unit MCP round
   trip plus roughly four atomic writes per unit (P1-2), and `assembly` (34–38×)
   is dominated by the per-unit checkpoint rewrites (P1-3). Both are fixable
   without changing retrieval behaviour.
3. `embedding` is the only phase with the same cost on both devices, and it is
   the largest single cost of a fresh build on fast storage (71.5%). It scales
   with *new* chunks, so P1-11 also reduces it for incremental adds, and it is
   the natural next optimisation target after storage (batching, thread count,
   or an optional accelerator).
4. Storage footprint: a 6.1 MB source corpus produced a **27 MB runtime**
   (4.4× the corpus) while retaining two generations.

## 11. Options for cheap incremental adds — pros and cons

### 11.1 The stated priorities, in order

1. **Resumability first.** A long first build is acceptable; losing progress is
   not. This is already satisfied and should not be weakened: progress is
   durable per extracted document, per 8-page PDF scan batch, per extraction
   unit, per 64-vector embedding batch, per 64-point Qdrant batch, and per
   phase, and survives cancellation, timeout, and process restart.
2. **Adding one source must be cheap.** It should not feel like rebuilding the
   project.
3. **First-build speed is secondary.** Improve it only when it does not cost
   priority 1 or 2.

### 11.2 Where the time goes when one source is added (HDD, measured)

Adding one 715 KB source to a 21-source, 1,505-chunk project took 950.7 s:

| Component | Seconds | Share | Does it reprocess the whole corpus? |
|---|---|---|---|
| Qdrant index rebuild (all 1,505 points) | 434.8 | 45.7% | **yes — full rebuild** |
| chunking (new document's units only) | 179.0 | 18.8% | no |
| extraction (new document only) | 152.8 | 16.1% | no |
| embedding (432 new vectors) | 71.2 | 7.5% | no |
| per-call overhead across 4 ingest calls | ~105.0 | 11.0% | n/a |
| assembly and checkpoint rewrites | 7.1 | 0.8% | no |
| BM25 rebuild (all 1,505 chunks) | 0.3 | 0.0% | yes, but negligible |

So priority 2 is blocked by exactly one whole-corpus component — the Qdrant
rebuild — plus, on slow storage, the per-unit durable writes for the new
document itself (chunking 179 s and extraction 153 s, neither of which touches
the rest of the corpus). BM25's full rebuild is already a non-issue at
0.3–4.4 s.

### 11.3 Option 1 — Keep full rebuilds, relocate derived state to fast storage (P1-12)

Do nothing structural; require (or allow) `.research-rag/runtime` to live on an
SSD.

- **Pros:** smallest change; measured effect is large — the same single-source
  add drops from 950.7 s to 96.1 s and the live project's Qdrant phase from
  3,040 s to about 35 s; also improves every other asset (logs, staging,
  bundles); no change to the generation contract.
- **Cons:** requires the `AGENTS.md` invariant decision (D5); the
  project is no longer self-contained if the runtime is external; does not fix
  the scaling curve — every change still reprocesses all points; useless on
  machines without an SSD.

### 11.4 Option 2 — Keep full rebuilds, reduce durable writes (Step 2)

Keep the same work, but write far less: one `QdrantClient` for the whole index
phase, time-boxed upload batches, fewer atomic writes per extraction unit, and a
split between the immutable source inventory and the mutable progress journal.

- **Pros:** no architectural change and no new failure modes; helps every build,
  including the first; directly attacks the 18.6×–31.4× HDD penalty in
  `chunking`/`assembly`; independent of which dense backend is chosen.
- **Cons:** does not remove the whole-corpus Qdrant rebuild (45.7% of the add on
  HDD); the win size on HDD is unproven — my NVMe test of one-client-per-phase
  versus one-client-per-batch showed only 1.1×, so the benefit must be measured
  on HDD rather than assumed; two sub-variants exist and must be chosen:
  **2a** keep the per-unit durable state and only make each write cheaper, or
  **2b** keep unit-level state but fsync the checkpoint every N units or T
  seconds, which makes a crash redo at most those N units (bounded, small
  rework) in exchange for far fewer syncs.

### 11.5 Option 3 — Incremental Qdrant append (P1-11)

Give points stable IDs derived from `chunk_id`, copy the previous generation's
Qdrant directory, append the new points, delete points for removed sources.

- **Pros:** removes most of the 45.7% whole-corpus component; keeps the existing
  ANN backend and its payload filtering; scales to very large corpora.
- **Cons:** most complex option — needs a new ID scheme plus a compatibility or
  migration path for existing generations; deletion and re-inclusion semantics
  need explicit handling; the copy plus the delta still pays the device's
  per-point cost (290 ms/point on HDD), so a large add is still slow; it adds a
  second index-construction path that must be kept equivalent to the full
  rebuild forever.

### 11.6 Option 4 — Serve dense results from the portable vectors; drop the embedded vector database

The generation already stores `portable/embeddings.npy` (float32, one row per
chunk in stable chunk order) and the SQLite sidecar already maps content hashes
and chunk metadata to vector ordinals. Dense retrieval therefore needs nothing
more than a matrix–vector product and a top-k selection, and vector assembly is
already incremental (each batch is either reused or recomputed per text).
Qdrant's HNSW index adds nothing that a 12 MB matrix does not already provide.

- **Pros:** removes the single largest cost in the system — the whole-corpus
  rebuild that is 45.7% of an incremental add, 75.5× slower on HDD, and 99.1% of
  the live build; **resumability strictly improves**, because the phase that
  takes minutes and checkpoints every 64 points is replaced by a matrix layout
  that is already checkpointed per batch; removes a dependency, 24.8× write
  amplification, and a class of failure modes ("collection is missing", segment
  or lock corruption); makes dense scoring exact and therefore exactly
  reproducible; and it fits behind the existing `DenseBackend` protocol, so
  Qdrant can remain an optional backend for very large corpora.
- **Cons:** an O(chunks) scan per query instead of ANN, so it needs a documented
  size threshold and a fallback; changes the `dense_index` artifact in the
  manifest, so existing generations need a compatibility reader or one
  regeneration; filtering must be applied to candidate IDs in Python rather than
  pushed into the store (the server already resolves metadata filters to
  document IDs, so this is a small, contained change that must keep honoring
  exclusions).

Sizing check: the live project is 8,102 chunks × 384 float32 = **12.4 MB**, and a
full dot product over it is about **3 ms**. Even 100,000 chunks (150 MB) is about
40 ms per query. Below roughly 200,000 chunks the exact scan is comfortably
faster than the index it replaces, because the index build is what costs minutes
and the scan costs milliseconds.

### 11.7 Option 5 — Status quo: document the limits and change nothing structural

- **Pros:** zero risk, zero effort, and accurate documentation.
- **Cons:** does not meet priority 2 at all. Adding one paper to an HDD project
  continues to cost about 16 minutes.

### 11.8 Comparison

| Option | Meets priority 2? | Effect on priority 1 (resumability) | Effort | Risk | Contract impact |
|---|---|---|---|---|---|
| 1. Relocate runtime to SSD | Partly (950.7 s → 96.1 s measured) | none | tiny | low | needs the `AGENTS.md` invariant decision |
| 2a. Cheaper per-unit writes | No (still whole-corpus) | none | small | low | none |
| 2b. Coarser checkpoint cadence | No (still whole-corpus) | slight loss (bounded rework of N units) | small | low | none |
| 3. Incremental Qdrant append | Mostly (removes 45.7%) | none | large | medium-high | new ID scheme, migration, dual construction path |
| 4. Dense over portable vectors | **Yes** (removes the whole-corpus component entirely) | **improves** | medium | medium | `dense_index` artifact changes; compatibility path needed |
| 5. Status quo | No | none | none | none | none |

### 11.9 Recommendation

1. **Adopt Option 4 as the primary fix.** It is the only option that makes an
   incremental add genuinely cheap *and* strengthens resumability, and it removes
   a subsystem instead of adding one. Keep `LocalQdrantDenseBackend` behind the
   existing protocol for corpora above the documented threshold, so nothing is
   lost for future large collections.
2. **Adopt Option 2a unconditionally** (cheaper writes, same per-unit durable
   state). It helps the first build and the HDD cases that Option 4 does not
   touch (extraction 153 s and chunking 179 s of the measured add), and it does
   not trade away resumability. Measure before assuming the size of the win.
3. **Treat Option 1 as an immediate, zero-code stopgap for this workstation**
   (measured 950.7 s → 96.1 s) and as a documented requirement — not as the
   strategy, because it does not change the scaling curve.
4. **Keep Option 3 only if you decide to retain Qdrant as the primary dense
   backend.** It is more code than Option 4, keeps the 290 ms/point device
   penalty for the delta, and permanently adds a second construction path that
   must stay equivalent to a full rebuild.
5. **Reject Option 5**, but borrow its honesty: whatever is chosen, the actual
   limits belong in `README.md`.

### 11.10 Revised order of work for these priorities

1. Step 0 — green the tree.
2. Step 1 — embedding-limit fidelity (a correctness risk that affects every
   existing generation).
3. The storage-location decision (Option 1) and its documentation or option.
   Cheapest large win, no interaction with anything else.
4. Option 4, implemented as a second `DenseBackend` with an equivalence test:
   the same queries must return the same passages as the current Qdrant backend
   on the live corpus before anything is switched.
5. Option 2a write reduction, with a measured before/after on HDD.
6. Then the remaining steps in their existing order, with Step 2's Qdrant
   batching reduced to whatever still matters after Option 4.

## 12. Reference workload and design envelope

The design target is a corpus like `ai-and-fetishism`, declared by the user as
the representative case. Options, priorities, and defaults should be judged
against this envelope first, and only then against general-purpose RAG needs.

### 12.1 Reference workload (measured)

| Property | Value |
|---|---|
| Sources | 55 discovered, 53 indexed, 2 reviewed exclusions |
| Formats | 51 PDF, 2 EPUB |
| Source bytes | 135 MB |
| Extraction units | 4,016 |
| Chunks | 8,102 (about 150 per source) |
| Vectors | 8,102 × 384 float32 = 12.4 MB |
| Character | English-language humanities and social-science scholarship: journal articles, book chapters, working papers, agency reports; prose with footnotes, reference lists, tables, and short quotations in other scripts |
| Runtime footprint | 101 MB per generation |
| Query shape | natural-language research questions, occasional exact names and phrases |

### 12.2 Design envelope

- 25–250 English-primary sources;
- born-digital, text-layer PDFs (majority) plus a minority of EPUBs;
- 5,000–50,000 chunks and 10–80 MB of vectors;
- English prose with occasional short quotations in other scripts, plus
  footnote and reference apparatus;
- CPU-only, one machine, one user;
- occasional large changes (a new paper or a batch of them) rather than
  continuous writing.

### 12.3 Explicitly out of scope

- **Non-English-primary corpora and multilingual retrieval quality.** English is
  the supported language; the embedding and reranker models stay
  English-oriented, and no multilingual evaluation or tuning is planned.
- Scanned or image-only sources (no OCR is provided), and handwriting.
- Mathematical, statistical, and formula-heavy corpora (LaTeX or equation
  corpora) and code/documentation corpora.
- Very large corpora above the documented dense-search threshold, which the
  optional ANN backend exists to serve if that need ever appears.

### 12.4 What this scope licenses

- **Exact dense search (Option 4) as the default.** The target corpus is two
  orders of magnitude below the size at which an ANN index earns its keep, so
  the index build can be deleted rather than optimised.
- **Corruption-only text rejection for English**, with script mixing demoted
  from a rejection to a flag (see P0-4), and no investment in multilingual
  text-health heuristics.
- Treating symbol-only detection as a cheap guard rather than a feature.
- Removing multilingual items from `ROADMAP.md` rather than carrying them.
- Building the Step 6 evaluation set from this corpus instead of synthetic
  fixtures.

### 12.5 What this scope does not license

- **Weakening corruption detection.** Measured: 25 of the 29 chunks the revised
  policy reviews on this corpus are withheld for genuine corruption evidence
  (replacement characters or private-use/unassigned code points). The policy is
  working; the 4 restored chunks are legitimate evidence.
- **Assuming "English" means ASCII.** Measured: 6,895 of 8,102 chunks (85%)
  contain at least one non-ASCII character, so ASCII-only fast paths must be
  fallbacks, not the only path.

## 13. Decisions with options

Per the `AGENTS.md` rule "Presenting decisions to the user", every open decision
is recorded here as concrete options with pros and cons, one recommendation, and
a reversibility note. Record the chosen identifier beneath each decision once the
user decides.

| ID | Decision | Recommended | Blocks |
|---|---|---|---|
| D1 | Scope of this pass | C | everything |
| D2 | Resumability granularity | A | Step 2 |
| D3 | `stale` in the `search` response | B | Step 4 |
| D4 | When policy versions may be bumped | A, plus C for cosmetic changes | Steps 1b/1c |
| D5 | Where derived state may live | C now, B if first-class support is wanted | Option 1 |
| D6 | Dense backend | B | Step A |
| D7 | `ingest` scope control | A + C | nothing (not blocking) |
| D8 | Text-health rejection policy | B | Step 1b |
| D9 | Text normalization folding | C | Step 1c |

### D1 — Scope of this pass

Chosen: **C** (2026-09-19) — fidelity, storage location, Option 4, and Step 2a.

Context: the findings span fidelity defects (P0), measured performance (P1), and
hygiene (P2–P3). The declared priorities are resumability, then cheap incremental
adds, then first-build speed.

- **A. Fidelity only** — Step 0 plus Steps 1, 1b, and 1c.
  *Pros:* smallest change, lowest risk; restores evidence that is currently
  withheld (P0-4) and makes ligature and math-styled text matchable (P0-5);
  immediate research benefit. *Cons:* adding one paper still costs about 16
  minutes on HDD; the 45.7% whole-corpus rebuild remains.
- **B. A plus Option 1** (storage location, subject to D5).
  *Pros:* adds the largest measured single win for little effort, taking that
  same add from 950.7 s to 96.1 s. *Cons:* does not change the scaling curve;
  depends entirely on D5 being answered.
- **C. B plus Step A (Option 4) and Step 2a — recommended.**
  *Pros:* removes the whole-corpus dense rebuild, so an incremental add costs
  only the new document's own work; improves resumability by deleting the
  coarsest-grained phase; reduces write cost; satisfies both top priorities.
  *Cons:* medium effort; changes the `dense_index` artifact, so it needs a
  compatibility path and an equivalence test against the current backend.
- **D. Everything** — Steps 0 through 7.
  *Pros:* leaves no known issue unaddressed. *Cons:* mixes low-urgency hygiene
  (query CPU, decomposition, evaluation set, documentation polish) into the same
  pass, delaying validation of the fidelity fixes in real use.

Recommendation: **C**. It meets priorities 1 and 2 and defers only work that has
no bearing on them. Reversible: yes — each step is independently revertable; the
only hard-to-undo element is the `dense_index` artifact change, which D4 and D6
govern.

### D2 — Resumability granularity

Chosen: **A** (2026-09-19) — keep unit-level durable state; reduce only the cost of each write.

Context: resumability is priority 1. The current design is durable per document,
per 8-page scan batch, per extraction unit, per 64-vector batch, and per 64-point
index batch.

- **A. Keep unit-level durable state; reduce the cost of each write — recommended.**
  *Pros:* preserves the current recovery guarantee exactly; the HDD penalty in
  `chunking` and `assembly` is still reduced by removing redundant writes and
  shrinking the per-unit write. *Cons:* leaves some durability cost on the table,
  so a full HDD build stays slower than it could be.
- **B. Keep unit-level state, but fsync the checkpoint every N units or T seconds.**
  *Pros:* fewer syncs, so faster unit loops. *Cons:* a crash redoes up to N units
  of work; the recovery boundary becomes a tunable rather than a rule.
- **C. Coarser, batch-level checkpointing.**
  *Pros:* fastest unit loop. *Cons:* a crash redoes the whole batch, which is
  exactly what priority 1 argues against.

Recommendation: **A**. Reversible: yes — the cadence is a constant, not a schema.

### D3 — Should `search` return `stale`?

Chosen: **B** (2026-09-19) — add `include_staleness=false`, default `true`.

Context: every `search` calls `_status`, which walks the source tree (measured
8.84 ms for 55 sources, ~160 µs per file) and re-parses the manifest while
holding the project lock. At 55 sources this is about 3.5% of a warm query; at
thousands of sources it becomes the dominant per-query cost.

- **A. Keep the current behaviour.**
  *Pros:* every response is self-describing; no API change. *Cons:* O(sources)
  filesystem work on every query, growing linearly.
- **B. Add `include_staleness=false` (default `true`) — recommended.**
  *Pros:* fully backward compatible; gives agents and the UI an explicit way to
  skip the walk when they have already called `status`; cheap to implement.
  *Cons:* adds a parameter and a documented caller obligation.
- **C. Remove `stale` from `search` and report it only from `status`.**
  *Pros:* cleanest latency and the least work per query. *Cons:* breaking change
  for any client that reads `stale` from search results; `AGENT_GUIDE.md` already
  tells agents to call `status` first, so the practical loss is small.
- **D. Cache the verdict behind a directory signature with a short TTL.**
  *Pros:* keeps the field and removes most of the walk. *Cons:* a stale verdict
  is possible for the TTL window, which is a correctness-of-reporting risk in a
  tool that must be honest about what it searched.

Recommendation: **B**, with **D** as a later refinement if profiling warrants.
Reversible: yes.

### D4 — When may a policy version be bumped?

Chosen: **A + C** (2026-09-19) — bump for semantic changes; keep cosmetic changes compatible.

Context: Steps 1b and 1c change which chunks are retrievable and how text is
stored, so they interact with generation identity.

- **A. Bump whenever a change alters chunk records or stored text, and accept one
  regeneration — recommended.**
  *Pros:* simplest and most honest; no dual-format readers or writers; every
  generation has exactly one meaning. *Cons:* existing projects need one full
  rebuild, which is expensive on HDD today (mitigated by D1 option C and D5).
- **B. Never bump; make readers and writers accept both old and new records.**
  *Pros:* no regeneration for anyone. *Cons:* a permanent compatibility burden,
  more code paths, higher bug risk, and ambiguous chunk-identity semantics.
- **C. Bump only when retrieval semantics change; keep display-only changes
  compatible.**
  *Pros:* targeted and cheap where it is safe. *Cons:* requires a documented
  rule for what counts as semantic, and judgement on every change.

Recommendation: **A** for the semantic changes in Steps 1b and 1c, **C** for
purely cosmetic changes. Reversible: the bump can be reverted, but projects that
already regenerated keep the new artifacts.

### D5 — Where may derived state live?

Chosen: **C** (2026-09-19) — first-class validated `--runtime-root`, with an OS bind mount as the immediate stopgap.

Context: the project lives on an HDD; the runtime is 101 MB per generation and
the Qdrant build is 76× slower there than on the same machine's SSD.

- **A. Document the requirement** — the project, or at least `.research-rag`, must
  live on fast local storage.
  *Pros:* zero code and zero risk; keeps the invariant untouched. *Cons:* the
  user must relocate the project; HDD projects stay slow; nothing is enforced.
- **B. Bind-mount `<project>/.research-rag/runtime` to fast storage at the OS
  level.**
  *Pros:* works today with no code change; the path stays literally inside the
  project, so the invariant holds; no new failure modes in the server. *Cons:*
  Linux/ops-level and not portable; the server cannot validate it and cannot warn
  if a moved project loses the mount.
- **C. Add a first-class, validated `--runtime-root`, validated by a marker file
  and project ID — recommended as the durable answer.**
  *Pros:* portable, explicit, and detectable, with clear failure messages; works
  on every supported platform. *Cons:* requires amending the `AGENTS.md`
  invariant; adds a validation surface; must prevent two projects from sharing
  one root.

Recommendation: **C**, with **B** as the immediate stopgap. Reversible: B is
trivially reversible; C is reversible by removing the flag after moving state
back.

Outcome (2026-09-19, Step 2c): implemented as chosen. `--runtime-root` (or
`RESEARCH_ULTRARAG_RUNTIME_ROOT`) relocates the runtime root and nothing else:
`project.json`, reviewed metadata, exclusions, the source catalog, and bundles
stay in `<project>/.research-rag`. The first run claims the target by writing
`.research-ultra-rag-runtime.json` with the owning `project_id` and
`project_root`; every later run validates it and refuses a foreign owner, a
non-empty unmarked directory, a relative path, the project root or its
`.research-rag`, and a non-directory. One deviation from this plan's wording:
the root is recorded by a marker in the runtime root rather than in the portable
descriptor, so a project can be pointed at a different fast device without
rewriting portable identity, and a *shared* root is detected at the destination
where the risk actually is. The legacy `.ultrarag/research` migration runs only
for the default root. `status` reports the effective root as `runtime_root`
(`null` when default). The README keeps bind mount B documented as the
alternative, with its limitation stated: the server cannot detect it or warn
when a moved project loses the mount.

### D6 — Dense backend

Chosen: **B** (2026-09-19) — exact scan by default, `LocalQdrantDenseBackend` selectable above a documented threshold.

Context: the embedded Qdrant index build is 99.1% of the live build and 45.7% of
an incremental add, and the corpus it indexes is 12.4 MB.

- **A. Exact scan only; remove the Qdrant backend and dependency.**
  *Pros:* least code, one code path, nothing to keep equivalent. *Cons:* O(n) per
  query forever; no path for larger corpora; discards a working implementation.
- **B. Exact scan by default, with `LocalQdrantDenseBackend` selectable above a
  documented threshold — recommended.**
  *Pros:* meets priority 2 now, keeps an ANN path for future scale, reuses the
  existing `DenseBackend` protocol, and the optional backend stays covered by the
  existing suite. *Cons:* two backends to maintain, plus a documented threshold
  and an equivalence test.
- **C. Keep Qdrant primary and implement Option 3 (incremental append).**
  *Pros:* preserves ANN behaviour and payload filtering. *Cons:* highest
  complexity, permanently dual construction paths, and the delta still costs
  about 290 ms per point on HDD.

Recommendation: **B**. Reversible: yes — the backend is a policy field, not a
schema.

Outcome (2026-09-19, Step A part 2): implemented as chosen. The backend name is
stored per generation (`retrieval.dense.dense_backend`, with the index directory
in `files.dense_index`), dispatch reads that record, and manifests written before
the field resolve to `embedded-qdrant`. `--dense-backend auto|exact|qdrant`
(default `auto`, `RESEARCH_ULTRARAG_DENSE_BACKEND`) picks the build backend;
`auto` selects the exact scan at or below `EXACT_BACKEND_CHUNK_LIMIT`
(200,000 chunks) and the ANN backend above it. Measured on the live
`ai-and-fetishism` generation (8,102 chunks): exact index build **0.03 s** and
0.58 MB versus the recorded Qdrant baseline of **3,040.73 s**; dense query
latency 35–68 ms including query embedding; top-20 identical to a brute-force
cosine ranking for all five probe queries. The `qdrant_indexing` checkpoint phase
is renamed `dense_indexing`, and a checkpoint resumed under the old name restarts
that phase with the newly selected backend. Two consequences recorded rather than
hidden: 50k/100k latency figures are arithmetic extrapolations from the 8,102
measurement, not measurements, and the Step A gate ("adding one source performs
no whole-corpus dense index build") is **not** met yet — the build still
recreates the (now nearly free) index for every changed generation, which is
Step 2's incremental-append work.

### D7 — Does `ingest` need scope control?

Chosen: **C** (2026-09-19) — read-only `ingest` dry run.

Context: derivation is already incremental per document, so an operator never
has to re-extract everything; the whole-corpus cost is in indexing, which a scope
filter would not change.

- **A. No change.**
  *Pros:* no new surface, no new error paths. *Cons:* operators keep asking
  whether they must re-ingest everything, and keep believing they do.
- **B. Add an optional source filter (`source_ids` or source paths).**
  *Pros:* explicit control, useful for a targeted refresh. *Cons:* does not
  reduce index work, so it can mislead; adds validation, error paths, and
  questions about how a partial scope interacts with checkpoint identity.
- **C. Add a read-only `ingest` dry run that reports exactly what would change —
  recommended.**
  *Pros:* answers the real question (added, removed, and modified documents; new
  chunks and vectors; expected index work) without building anything; read-only
  and safe; complements `status.changes`. *Cons:* the work estimate is
  approximate.

Recommendation: **C**, with **B** only if a concrete operational need appears.
Reversible: yes. Not blocking any other step.

### D8 — Text-health rejection policy

Chosen: **B** (2026-09-19) — corruption-only rejection, script mixing as an advisory flag, and disclosed withholding.

Context: P0-4. On the reference corpus the current policy withholds 29 of 8,102
chunks, of which 26 carry genuine corruption evidence and 3 are flagged only for
script mixing.

- **A. Keep the current policy.**
  *Pros:* no change and no regeneration. *Cons:* three legitimate chunks stay
  permanently unretrievable, and every new source carries the same risk.
- **B. Reject only on genuine corruption evidence; demote `non_latin_dominant` and
  `mixed_script_text` to flags; report withheld counts and reasons —
  recommended.**
  *Pros:* restores the legitimate Cyrillic and Greek material while keeping all
  26 genuinely corrupt chunks withheld; makes withholding visible instead of
  silent; matches the English-only scope, which accepts short foreign-language
  quotations. *Cons:* small risk of admitting a font-garbage chunk whose only
  symptom was script mixing, mitigated by surfacing the flag for the agent to
  judge.
- **C. B plus a reviewed per-source "allow foreign-script content" override.**
  *Pros:* precise per-source control for genuinely multilingual sources. *Cons:*
  more review state and another override to reason about, for a case the
  envelope already excludes.
- **D. Flag-only: never withhold, only mark.**
  *Pros:* no silent loss, ever. *Cons:* admits real font garbage into both
  indexes, wasting candidate depth and lowering measured relevance.

Recommendation: **B**. Reversible: yes for retrieval; if it also changes what
extraction excludes at ingest, it is a policy change governed by D4.

### D9 — Text normalization folding

Chosen: **C** (2026-09-19) — targeted fold for Mathematical Alphanumeric Symbols and Alphabetic Presentation Forms.

Context: P0-5. 39 chunks use Mathematical Alphanumeric Symbols (`𝑀𝑗𝑀𝑗𝑗𝑀` for
`MjMjjM`) and 723 use ligature or presentation forms (`ﬁ`, `ﬂ`, `ﬀ`), none of
which plain-text BM25 queries can match under NFC.

- **A. Keep NFC.**
  *Pros:* no change and no regeneration. *Cons:* those chunks stay unmatchable by
  the spellings a researcher would actually type.
- **B. Apply full NFKC everywhere.**
  *Pros:* one change covers every compatibility form. *Cons:* wider blast radius —
  it folds superscripts, subscripts, and assorted symbols, so footnote markers and
  notation can change silently.
- **C. Targeted fold for Mathematical Alphanumeric Symbols and Alphabetic
  Presentation Forms, then NFC — recommended.**
  *Pros:* fixes exactly the two measured problems; leaves `é`, `æ`, `œ`,
  superscripts, and other symbols canonical. *Cons:* needs a documented list, and
  any future compatibility form must be added deliberately.
- **D. C plus a separate folded field used only for chunking and search, keeping
  `contents` canonical.**
  *Pros:* display text is untouched. *Cons:* two texts to keep in sync; changes
  the chunker input, the manifest, and reuse keys, which is more risk than the
  problem justifies.

Recommendation: **C**. Reversible: yes, but because it changes stored `contents`
it requires a `cleaning_policy_version` bump and one regeneration (D4).

### Recording a decision

When the user chooses, record it directly under the decision heading in this
section in the form `Chosen: <identifier> (<date>)`, with a one-line note of any
clarifying detail. Then mark the matching checklist item in `TODO.md` and add the
constraint to `AGENTS.md` if the choice changes a standing rule. Until a decision
that affects generation artifacts, retrievable evidence, the portable-state
contract, or on-disk layout is recorded here, the corresponding step stays
blocked.


## 14. Non-goals

- **No multilingual support or multilingual retrieval quality work.** English is
  the supported language for the reference workload in section 12; the
  multilingual roadmap items should be removed rather than carried.
- **No support target for non-English-primary corpora, OCR or scanned sources,
  handwriting, or mathematical/formula-heavy corpora.** These are outside the
  design envelope.
- No change to the pinned UltraRAG version, the vanilla gateway boundary, or the
  vanilla repository.
- No weakening of the generation/activation protocol, the portable-state
  contract, or the exclusion and metadata-overlay semantics.
- No weakening of corruption detection: only genuine corruption evidence
  (replacement characters, private-use or unassigned code points, known
  mojibake) may withhold a chunk.
- No document-, author-, or publisher-specific extraction exceptions.
- No automatic source deletion or automatic duplicate resolution.
- No silent relaxation of the `direct_quote_safe: false` contract.

# Research UltraRAG MCP — Strengthening Checklist

Derived from [`PLAN.md`](PLAN.md). Complete a step's gate before starting the
next one. Check an item only when its code, tests, and documentation are
complete and the repository is green.

## Working rules

- [ ] Keep every change inside `research-ultra-rag-mcp-server`; never add
      research behaviour to the vanilla repository.
- [ ] Run `uv run ruff format --check .`, `uv run ruff check .`, and
      `uv run pytest -q` before every commit.
- [ ] Commit and push inside this child repository before any parent pointer
      update, per the collection-root `AGENTS.md`.
- [ ] Preserve the generation/activation protocol, portable-state contract, and
      exclusion and metadata-overlay semantics exactly.
- [ ] Bump the correct policy version whenever a change alters generation
      artifacts, so existing projects are flagged for regeneration instead of
      silently reused.
- [ ] Add no dependency, background process, or external service without
      measured need.

## Prioritization (agreed order)

Priority order for this work: **1) the process must be resumable; 2) adding a
new source must not require re-ingesting everything; 3) first-build speed is
secondary.** The option analysis with pros and cons is `PLAN.md` §11; the choices
to confirm are D1 and D6 below.

Recommended sequence (`PLAN.md` §11.10): Step 0 → Step 1 (including 1b and 1c) →
storage-location decision (Option 1) → Step A (Option 4) with an equivalence test
→ Step 2 (Option 2a) → remaining steps.

## Target workload and scope (PLAN.md §12)

The reference corpus is an `ai-and-fetishism`-like collection and is the
yardstick for every default: 25–250 English-primary born-digital PDF/EPUB
sources, 5,000–50,000 chunks, CPU-only, humanities and social-science prose with
quotes, footnotes, and reference apparatus.

- [ ] Optimize and validate against that envelope first; do not add
      general-purpose machinery that the envelope does not need.
- [ ] Treat non-English-primary corpora, multilingual retrieval, OCR/scanned
      sources, and formula-heavy corpora as out of scope, and say so in
      `README.md`.
- [ ] Remove the multilingual items from `ROADMAP.md` instead of carrying them.
- [ ] Route any conflicting default to a decision rather than to code.

## Blocked on decisions

Options, pros and cons for every decision are in `PLAN.md` §13. Record the chosen
identifier in `PLAN.md` as `Chosen: <identifier> (<date>)`, then tick the box
here.

- [x] D1 — Scope of this pass: (a) fidelity only, (b) fidelity + storage
      location, **(c) fidelity + storage + Option 4 + Step 2a (recommended)**,
      (d) everything through Step 7.
- [x] D2 — Resumability granularity: **(a) keep unit-level durable state and
      reduce write cost (recommended)**, (b) unit-level state with an fsync
      cadence, (c) batch-level checkpoints.
- [x] D3 — `stale` in `search`: (a) keep as is, **(b) add
      `include_staleness=false` (recommended)**, (c) remove from `search`,
      (d) cache behind a directory signature.
- [x] D4 — Policy-version bumps: **(a) bump for semantic changes (recommended)**,
      (b) never bump, dual readers, (c) bump only for retrieval-semantic
      changes — recommended for cosmetic ones.
- [x] D5 — Derived-state location: (a) document the fast-storage requirement,
      (b) OS-level bind mount, **(c) first-class validated `--runtime-root`
      (recommended), with (b) as the immediate stopgap)**.
- [x] D6 — Dense backend: (a) exact scan only and drop Qdrant, **(b) exact scan
      by default with Qdrant selectable above a threshold (recommended)**,
      (c) keep Qdrant and implement Option 3.
- [x] D7 — `ingest` scope control: (a) no change, (b) source filter,
      **(c) read-only dry run (recommended)**.
- [x] D8 — Text-health rejection: (a) keep as is, **(b) corruption-only rejection
      with script mixing as a flag, plus reporting (recommended)**, (c) b plus a
      per-source override, (d) flag-only.
- [x] D9 — Text normalization: (a) keep NFC, (b) full NFKC, **(c) targeted fold
      for math-alphanumeric and presentation forms (recommended)**, (d) c with a
      separate folded search field.

## Step A — Dense backend: exact search over portable vectors (Option 4)

Only if D6 selects option B. This is the change that makes an incremental
add cheap without weakening resumability.

- [ ] Prototype a `LocalVectorDenseBackend` behind the existing `DenseBackend`
      protocol, reading `portable/embeddings.npy` and the artifact-lookup
      ordinals; no new dependency.
- [ ] Keep exclusion and document-ID filters enforced on candidate IDs, exactly
      as the Qdrant path does today.
- [ ] Add a compatibility path so existing generations with a Qdrant dense index
      still search, or flag them for one regeneration.
- [ ] Prove equivalence correctly: the ANN path is approximate, so the gate is
      that the exact backend matches brute-force ground truth, never returns a
      chunk the ANN path would have found at the same depth with a lower score,
      and does not reduce nDCG on the judged query set. Do not require
      bit-identical ordering against HNSW.
- [ ] Measure dense query latency at 8k, 50k, and 100k chunks and document the
      threshold above which an ANN backend is recommended.
- [ ] Keep `LocalQdrantDenseBackend` available and selectable for large corpora.
- [ ] Update every architecture description that names Qdrant as the dense store
      (`AGENTS.md` component and storage notes, README "How it works under the
      hood") and document the selection threshold.
- [ ] Record the decision as an architecture note, including why the ANN index
      was not needed at current corpus sizes.

Gate:

- [ ] Adding one source performs no whole-corpus dense index build, a crash
      mid-way remains resumable at the same granularity as today, and retrieval
      results are unchanged.

## Verification baseline (completed in the review)

- [x] Confirm resumable ingestion checkpoints are implemented, and record their
      limits, in `PLAN.md` §8.1.
- [x] Confirm post-ingestion metadata editing is implemented, and record its
      limits, in `PLAN.md` §8.2.
- [x] Establish that per-document derivation reuse is implemented but index
      rebuilding is not, in `PLAN.md` §8.3.
- [x] Establish that the model cache and UltraRAG runtime cache are already
      configurable and already on the SSD, while project derived state is fixed,
      in `PLAN.md` §8.4.
- [x] Demonstrate that a symlinked `.research-rag/runtime` is rejected at
      startup, so relocation needs a code change rather than a symlink.
- [x] Run the read-only verification and latency probe against the live
      `ai-and-fetishism` project.
- [x] Run bounded 20-source benchmarks on SSD and HDD project roots.
- [ ] Add `scripts/benchmark_ingest.py` so the benchmark evidence in `PLAN.md`
      §10 is reproducible inside the repository. A working version is staged
      outside the repositories at `~/rr_benchmark_ingest.py`; it must be
      parameterized on a donor source directory before it is committed.

Recorded results (details in `PLAN.md` §10):

- [x] Live-project read-only verification passes; warm hybrid search is
      0.52–0.68 s while the CLI's 15.6 s is startup and cold BM25 load.
- [x] Fresh 20-source build: 259.1 s on NVMe versus 1,344.7 s on HDD (5.2×).
- [x] Qdrant cost is 3.8 ms/point on NVMe versus 290 ms/point on HDD (76×),
      consistent with the live project's 375 ms/point.
- [x] Adding one source reuses everything unchanged (20 documents, 1,073 chunks,
      1,073 vectors) yet still rebuilds both indexes: 96.1 s on NVMe versus
      950.7 s (15.8 minutes) on HDD.
- [x] Embedding is the dominant fresh-build cost on fast storage (171.75 s,
      71.5%) at about 0.16 s per new chunk.
- [x] Text study of the reference corpus (all 8,102 chunks): the current policy
      withholds 29 chunks (0.36%), of which 26 carry genuine corruption evidence
      and 3 are flagged only for script mixing (2 are legitimate Cyrillic
      bibliography entries); 6,895 chunks (85%) contain non-ASCII characters;
      39 chunks use Mathematical Alphanumeric Symbols; 723 use ligature or
      presentation forms (`PLAN.md` P0-4 and P0-5).
- [x] The English-only script-classification simplification was measured at only
      1.4× faster, so it is not a speed argument.
- [x] Step 1b implemented and probed on the reference corpus: withheld chunks
      fell from 29 to 25, 4 legitimate chunks are restored (the Greek quotation
      and two Cyrillic bibliography entries among them), and 7 chunks carry
      advisory script notes instead of being discarded.
- [x] Step 1c implemented and probed: folding removes formula-font letters from
      all 39 affected chunks (751 chunks change text in total) while leaving
      accented letters, superscripts, and symbols untouched.
- [x] Step 1 implemented and probed on the reference corpus through the shipped
      path: all 8,102 chunks are audited, 8 exceed the 512-token embedding limit
      (maximum 3,417), and the audit degrades to `unavailable` without failing a
      build when the model cache is absent.

## Documentation baseline

- [x] Create `PLAN.md` with review findings, evidence, and the execution order.
- [x] Create `TODO.md` as the actionable checklist.
- [x] Add `PLAN.md` and `TODO.md` to the `AGENTS.md` documentation
      responsibilities list.
- [ ] Keep `ROADMAP.md` limited to deferred work and remove entries that this
      plan promotes (for example generation cleanup).
- [ ] Document the checkpoint, metadata-editing, and single-source-ingest limits
      from `PLAN.md` §8 in `README.md` and `AGENT_GUIDE.md`.

## Step 0 — Make the tree shippable

- [x] Run `uv run ruff format .` to fix `bundle.py` and `service.py`.
- [x] Confirm `ruff format --check .`, `ruff check .`, `pytest -q` all pass.
- [ ] Commit inside the child repository.

Gate:

- [ ] The CI workflow (`test.yml`) would pass on the current commit.

## Step 1 — Close the retrieval-fidelity gap (P0-2, P0-3)

- [x] Compute the embedding-tokenizer length for every built chunk during
      enrichment and persist it in the chunk record as `embedding_token_count`
      with a `dense_truncated` flag.
- [x] Keep the flag in the canonical chunk record instead of adding a sidecar
      column. Deviation from the original plan: the sidecar stores only IDs,
      hashes, and byte offsets, and every query path already loads the full
      canonical record, so a `LOOKUP_SCHEMA_VERSION` bump would have added
      migration churn with no query benefit.
- [x] Report over-limit chunks explicitly instead of splitting them. Splitting
      was rejected for now because it would change chunk identity, ordinals, and
      locators for 0.10% of the corpus; the flag plus the aggregate makes the
      limitation visible, and splitting stays deferred.
- [x] Report the aggregate in build metrics, which `status` returns as
      `last_build_metrics`: `dense_token_audit`, `dense_audited_chunk_count`,
      `dense_truncated_chunk_count`, `embedding_maximum_tokens`, and
      `maximum_embedding_token_count`. Each search hit carries the per-chunk
      values plus a `dense_fidelity` summary.
- [x] Keep legacy generations readable: an absent count means "predates the
      audit", reported as null rather than as a false zero.
- [x] Replace the original property test with what the decision requires: the
      audit detects a real over-limit text (verified on the reference corpus: 8
      chunks, maximum 3,417 tokens) and degrades safely with no model cache.
      A test asserting that no chunk exceeds the limit would fail by design,
      because the policy flags rather than splits.
- [x] Correct the README tool-reference wording so `chunk_size` reads as a
      requested maximum that the chunker can exceed slightly.

Gate:

- [x] The reference corpus reports its 8 over-limit chunks with an inspectable
      count, the audit degrades safely without a model cache, and every consumer
      passes the existing suite. Splitting remains deferred by decision.

## Step 1b — Stop silently withholding legitimate chunks (P0-4; needs D8)

Measured: the current policy withholds 29 of 8,102 reference-corpus chunks, and
retrieval enforces the same gate, so they are invisible to `search` today.

- [x] Restrict withholding to genuine corruption evidence:
      `replacement_characters`, `private_or_unassigned_characters`,
      `known_mojibake`.
- [x] Demote `non_latin_dominant` and `mixed_script_text` from rejection reasons
      to flags carried on the chunk and reported in results.
- [ ] Keep the English-only `_script_family` simplification, but note that it
      measured only 1.4× faster; the justification for this step is recall
      fidelity, not speed.
- [x] Assert that the two legitimate Cyrillic bibliography chunks and the Greek
      quotation chunk become retrievable again after the change.
- [x] Keep the withholding of the 26 genuinely corrupt chunks unchanged.
- [x] Report withheld counts and per-reason breakdowns in `search` and `status`,
      and provide a way to inspect the withheld chunks rather than only a count.
- [ ] Add a regression fixture derived from the reference corpus so a future
      policy change cannot silently re-withhold legitimate evidence.

Gate:

- [x] Legitimate multilingual quotations and bibliography entries are
      retrievable, genuine corruption is still withheld, and the withheld count
      is reported rather than silent.

## Step 1c — Fold math-styled letters and ligatures (P0-5; needs D9)

Measured: 39 chunks use Mathematical Alphanumeric Symbols (`𝑀𝑗𝑀𝑗𝑗𝑀` instead of
`MjMjjM`) and 723 chunks use ligature or presentation forms (`ﬁ`, `ﬂ`, `ﬀ`), all
of which are invisible to plain-text BM25 queries under NFC.

- [x] Apply NFKC, or a targeted fold for the Mathematical Alphanumeric and
      Alphabetic Presentation Forms blocks, in the text path.
- [x] Confirm ordinary letters (`é`, `æ`, `œ`) and citation markers are not
      damaged; record any symbol-folding side effects.
- [x] Bump `cleaning_policy_version` and document that one regeneration is
      required.
- [ ] Measure recall before and after on the judged query set, including queries
      that contain a folded sequence.

Gate:

- [ ] Queries for the plain-text spelling of a math-styled or ligature sequence
      match the chunk, and no judged query regresses.

## Step 2 — Cheaper writes without losing resumability (Option 2a; P1-1, P1-2, P1-3)

Priority: reduce work and durability cost **without** widening the resume
granularity. Every unit must still leave a durable, resumable state behind; a
crash must never redo more than one extraction unit.

- [ ] Keep one `QdrantClient` open for the entire `qdrant_indexing` phase
      instead of one client per batch, or drop the phase entirely if Step A is
      adopted.
- [ ] Replace the fixed 64-point upload batch with a time-boxed batch, with a
      single explicit verification before activation.
- [ ] If extraction units are batched for the chunker, write each unit's durable
      state before the batch call so a crash resumes at the last durable unit
      rather than at the batch start.
- [ ] Keep `checkpoint.json` at unit granularity, and make each write cheap by
      moving the immutable source inventory and digests into a write-once file so
      the per-unit write stays small and constant-size.
- [ ] Stop rewriting per-source artifacts that did not change.
- [ ] Add a test asserting that a crash mid-chunking redoes at most one unit.
- [x] Document the fast-local-storage requirement for `.research-rag/runtime`
      and the bind-mount stopgap in the README storage section, including the
      measured cost per point on each device (done ahead of Step 2 because it is
      the cheapest large win for an HDD-backed project).
- [ ] Verify that resume, cancellation, and timeout behaviour still produce a
      resumable checkpoint at every boundary.

Gate:

- [ ] `chunking` and `assembly` are measurably cheaper on HDD, reuse counts and
      the `unchanged` path are identical, and a crash mid-chunking redoes at
      most one extraction unit.

## Step 2b — Incremental Qdrant append (P1-11; only if D6 chooses option C)

- [ ] Assign stable Qdrant point IDs derived from `chunk_id` instead of the
      positional `offset + index`, with a migration or fallback for existing
      generations.
- [ ] Build a new generation's Qdrant index by copying the compatible previous
      index and appending the delta, leaving the active generation immutable.
- [ ] Delete points for removed or newly excluded sources through an explicit,
      recorded delta operation.
- [ ] Keep the full BM25 rebuild (measured at 0.32–4.40 s) and do not attempt an
      incremental BM25 path.
- [ ] Fall back to a full rebuild when the delta is large (for example >40% of
      chunks) or the previous index is incompatible.
- [ ] Verify the appended generation with an expected point count and a
      point-ID digest before activation.
- [ ] Prove equivalence: for a corpus change, an incremental build and a forced
      full rebuild must produce identical retrieval results for a fixed query
      set.

Gate:

- [ ] Adding one source to the bounded benchmark corpus costs seconds of
      indexing instead of a full rebuild, with retrieval output identical to the
      full-rebuild path.

## Step 2c — Explicit derived-state location (P1-12, needs D5)

- [ ] Decide whether the `AGENTS.md` "everything beneath
      `<project>/.research-rag`" invariant is amended or the storage requirement
      is documented instead.
- [ ] If amended: add `--runtime-root` / `RESEARCH_ULTRARAG_RUNTIME_ROOT` with
      containment, ownership, and marker validation.
- [ ] If amended: update the `AGENTS.md` invariant bullet that requires every
      project-owned artifact to live beneath `<project>/.research-rag`.
- [ ] Record the runtime root in `.research-rag/project.json` so a moved or
      missing root is detected rather than silently re-derived.
- [ ] Refuse to start when a runtime root belongs to another project or its
      marker does not match, so a shared fast disk cannot mix projects.
- [ ] Confirm that no bundle, log, or index content can leak between projects
      through a shared runtime root.
- [ ] If not amended: document the fast-local-storage requirement in the README
      storage and limitations sections.

Gate:

- [ ] Either a relocated runtime root works with all safety guarantees intact,
      or the storage requirement is unambiguously documented.

## Step 2d — Embedding throughput (P1-13)

- [ ] Establish the current baseline (0.16 s per new chunk, 71.5% of a fresh
      fast-storage build) as a repeatable benchmark.
- [ ] Measure the ONNX session thread count and provider configuration rather
      than assuming defaults.
- [ ] Evaluate chunk padding and sequence-length effects.
- [ ] Consider an optional accelerator or a smaller model profile only if the
      measurements justify it, keeping CPU as the default.
- [ ] Re-check the embedding model input limit after any change (P0-2).

Gate:

- [ ] Embedding cost per new chunk improves measurably with no change in
      retrieval results for the judged query set.

## Step 3 — Per-query CPU (P1-4, P1-5)

- [ ] Precompute the corrupt-text, symbol-only, and query-token verdicts at
      artifact-lookup build time and filter on the stored flags at query time.
- [ ] Cache parsed chunk tokens per chunk for the duration of a query.
- [ ] Keep retrieval-time guards for legacy generations that lack stored flags.
- [ ] Make `_bm25_ranking` accumulate accepted results across widening
      iterations and process only newly returned passages.
- [ ] Pass the already-parsed manifest into `_status` instead of re-reading it.

Gate:

- [ ] A benchmark test shows candidate-filter cost is independent of the number
      of queries, with all existing relevance rejection counters unchanged.

## Step 4 — Remove per-request O(corpus) work (P1-6, P1-7, P1-9)

- [ ] Cache the staleness verdict behind a cheap directory signature, with a TTL
      or an explicit opt-out on `search`.
- [ ] Stop the source-tree walk inside `search` (per D3).
- [ ] Push category/keyword filtering into the dense backend instead of
      materializing document-ID lists per query.
- [ ] Reuse SQLite connections within a request.
- [ ] Cache the effective-document map per manifest and metadata revision.

Gate:

- [ ] An integration test proves `search` performs no source-tree walk when
      staleness is not requested, and filter results are unchanged.

## Step 5 — Decomposition for safety (P2-1)

- [ ] Extract one handler per ingestion phase behind a dispatch table.
- [ ] Remove the duplicated checkpoint/`budget_expired` epilogues.
- [ ] Enable `ruff` `C901` with a threshold so method size cannot regress.
- [ ] Add per-phase tests for budget expiry and cancellation at each boundary.

Gate:

- [ ] The full existing suite passes unchanged, and no phase exceeds the
      configured complexity threshold.

## Step 6 — Real evaluation (P3)

- [ ] Write 30–50 representative research queries for the live corpus.
- [ ] Produce graded source-level relevance judgements for them.
- [ ] Record precision/recall/nDCG for BM25, dense, hybrid, and reranked.
- [ ] Check the judged set and report into the repository.
- [ ] Only after that, consider tuning constants (fusion weights, RRF k,
      dense similarity floor, candidate bounds) and require evidence for each.

Gate:

- [ ] No retrieval constant is changed without a recorded before/after result
      on the judged set.

## Step 7 — Contract polish and retention

- [ ] Add generation listing and pruning with an explicit keep-last-N policy
      and a preview of what will be deleted.
- [ ] Add a disk-space pre-check before building.
- [ ] Report which activation stage failed (artifacts, BM25 probe, dense index)
      instead of one collapsed error.
- [ ] Fix the no-generation `stale` semantics and update `AGENT_GUIDE.md` and
      the documentation test accordingly.
- [ ] Clarify in the README that legacy generations are readable but cannot be
      activated, reused, or exported.
- [ ] Add a test asserting every retrieval constant participates in
      `RETRIEVAL_POLICY_FINGERPRINT`.
- [ ] Narrow the two broad `except Exception` sites to preserve diagnostics.
- [ ] Add a non-English corpus smoke test for the documented English-only
      heuristics.

Gate:

- [ ] Retention is bounded and previewed, failures name their stage, and the
      README, `AGENT_GUIDE.md`, and `AGENTS.md` match the implementation.

## Release checklist

- [ ] `ruff format --check`, `ruff check`, and the full test suite pass.
- [ ] The real-stdio integration flow passes against the pinned vanilla gateway.
- [ ] Offline operation passes once the runtime and models are cached.
- [ ] README tool table, parameters, and defaults match the implementation.
- [ ] `AGENT_GUIDE.md` matches the server instructions and tool descriptions.
- [ ] `AGENTS.md` reflects the architecture, commands, and pinned commits.
- [ ] UltraRAG version/commit, license, attribution, and the
      independent-project disclaimer are correct.
- [ ] No project data, model binaries, logs, or generated indexes are committed.
- [ ] `README.md` states the supported workload and language scope, and claims no
      multilingual, OCR, or formula-corpus support.
- [ ] `ROADMAP.md` carries no multilingual item and no item that this plan has
      promoted into a step.
- [ ] Child repository changes are committed and pushed before the parent
      submodule pointer is updated.
- [ ] `git submodule status --recursive` and `git diff --check` are clean at the
      collection root.

# Plan: what we reviewed, what we changed, and why

This document is the working record behind `TODO.md`. It exists so that a person — not only an agent — can see what was investigated, what was measured, what was decided, and what is still open.

How to read it:

- **Section 3** lists everything the review found, in plain language, grouped by how much it matters.
- **Section 4** records the decisions that needed a choice, with the options and the reasoning.
- **Section 5** walks through the work in the order it was done, explaining what each change was for.
- **Section 6** holds the measurements. Every number quoted elsewhere in the repository comes from there.

Finding identifiers (`P0-2`, `P1-13`) and decision identifiers (`D6`, `D10`) are kept because code comments and other documents refer to them.

## 1. The project and the workload it is designed for

The server lets an AI agent search a personal collection of PDF and EPUB files and return evidence passages with provenance: cleaned semantic text, source path, locator, and metadata. It is built around UltraRAG for chunking and BM25, with a research layer around it for extraction, metadata, locators, and immutability. `README.md` is the user manual.

The review judged everything against one real corpus, the `ai-and-fetishism` project of 55 discovered and 53 indexed sources:

| Property | Value |
|---|---|
| Sources | 55 discovered, 53 indexed, 2 reviewed exclusions |
| Source bytes | 135 MB, mostly PDF plus 2 EPUB |
| Extraction units | 4,016 |
| Chunks | 8,102 (about 150 per source) |
| Vectors | 8,102 × 384 float32 = 12.4 MB |
| Runtime footprint | 101 MB per generation |
| Character | English humanities and social-science prose with footnotes, reference lists, tables, and short quotations in other scripts |

The design envelope is 25–250 English-primary sources, 5,000–50,000 chunks, and CPU-only execution. Deliberately out of scope: non-English-primary corpora, multilingual retrieval, OCR or scanned sources, handwriting, and formula-heavy corpora. Those limits are also stated in `README.md` so users are not surprised.

## 2. How the review was done

1. Read the source, the tests, and every public document.
2. Ran the server against the real project above and checked what it actually returned, rather than trusting the documentation.
3. Benchmarked two identical corpora on the two devices that matter — an NVMe SSD and the HDD where the real project lives — so device effects could be separated from code effects.
4. Measured individual operations in isolation when a guess about the cause did not survive contact with the numbers.
5. Stopped and asked before changing anything with a trade-off, rather than silently reinterpreting an earlier decision.
6. Recorded measurements even when they contradicted the plan's own reasoning.

## 3. What we found

Severity here means "how much it costs if left alone", not "how hard it is". Each finding also appears in `TODO.md` with its current status.

### 3.1 Correctness and fidelity (P0)

These were wrong or unverifiable, and they affected existing data.

| ID | Finding | What it meant |
|---|---|---|
| P0-1 | The test suite was red on the current tree | Nothing could be safely changed until the baseline was green. Fixed with formatting only. |
| P0-2 | 8 of 8,102 chunks exceeded the embedding model's 512-token limit | FastEmbed truncates silently, so those chunks were searchable by BM25 across their full text but only their beginning was searchable semantically. Nobody could see this. Fixed by auditing every chunk and reporting the count and a per-chunk flag. |
| P0-3 | Nothing enforced the relationship between chunk size and the embedding limit | A future chunk-size change could silently widen the truncation problem. The audit now makes it visible on every build. |
| P0-4 | The text-health policy withheld 29 of 8,102 chunks, and 4 of those were legitimate | Quotations in Greek and Cyrillic were being dropped from retrieval. The policy now withholds only for corruption evidence, reports script mixing as advisory `text_notes`, and discloses every withholding with counts and example chunk IDs. After the fix, 25 chunks are withheld instead of 29. |
| P0-5 | Search normalisation lost math-styled letters and presentation ligatures | A query typed as plain text could not match a PDF that used formula-font letters or `ﬁ`/`ﬂ`/`ﬀ`. Now folded to plain spellings, which changed 751 chunks for the better. |

### 3.2 Measured inefficiency (P1)

These were not incorrect, but they cost time or space out of proportion to the work being done. Numbers are in section 6.

| ID | Finding | What it meant |
|---|---|---|
| P1-1 | The embedded index build was 99.1% of a real build and 45.7% of adding one source | Adding a paper to an existing project felt like rebuilding it. Resolved by making the exact vector scan the default. |
| P1-2 | Chunking did one gateway round trip per extraction unit | On the HDD this was an 18–31× device penalty for work that has almost no CPU cost. |
| P1-3 | Each unit rewrote the whole ingestion checkpoint | The plan's assumption was that the file size mattered; measurement showed the cost is fsync latency, so the fix became fewer writes, not smaller ones. |
| P1-4 | Retrieval re-normalised and re-scanned every candidate on every query | About 0.55 ms per candidate, roughly 18 ms per query at default depth and 110 ms at reference-view depth. |
| P1-5 | The BM25 widening loop restarted all previous work each iteration | Only visible with selective filters, but wasteful in exactly the cases where filters are used. |
| P1-6 | `search` performed a full source-tree walk | 8.84 ms for 55 sources, growing linearly with source count, on every query. |
| P1-7 | Dense filtering materialised document-ID lists per query | Scales with corpus size on every query. |
| P1-8 | Bundle import hashed the same content twice | Wasted I/O during a rare but slow operation. |
| P1-9 | Several smaller per-request O(corpus) patterns | Cumulative cost on large projects. |
| P1-10 | Generation retention was unbounded | Generations are ~101 MB each and accumulate silently. |
| P1-11 | Index construction is not incremental | The core reason adding one source was expensive. |
| P1-12 | Derived state had to live inside the project | A project on a slow disk could not put indexes on a fast one without an OS-level bind mount. |
| P1-13 | Embedding was 71.5% of a fresh build on fast storage | The next bottleneck once storage was fixed. |

### 3.3 Maintainability and contract clarity (P2)

| ID | Finding | What it meant |
|---|---|---|
| P2-1 | Two methods carried most of the ingestion logic | Hard to review safely; raising the linter's complexity limit was not the answer. |
| P2-2 | Activation validation is all-or-nothing | One failing index hides every other diagnostic. |
| P2-3 | `stale` is overloaded in the no-generation branch | A reader cannot tell "sources exist but nothing is built" from "the build is out of date". |
| P2-4 | Legacy generations can be searched but never re-promoted | An older generation can pass validation yet cannot become current again. |
| P2-5 | The schema version stayed at 5 while four policy axes moved | Version numbers stopped describing the data. |
| P2-6 | A few documentation statements were imprecise | Small, but they were the kind a careful reader would trip over. |
| P2-7 | Two broad `except Exception` blocks sat at durability boundaries | They could hide real storage faults. |

### 3.4 Evaluation and product gaps (P3)

The server has no judged query set, so "does it retrieve well?" could only be answered by inspection. Recall and ranking quality are unmeasured, and there is no built-in way to review or prune retained generations.

## 4. Decisions

Each decision below was recorded *before* being acted on. Options are summarised; the chosen line states the answer.

### D1 — Scope of this pass
Options: fidelity only; fidelity plus storage; **fidelity, storage, the dense backend change, and the write-path work**; everything through Step 7. **Chosen: the third.** It addressed the two findings that affect data and the three that affect the reference workload, and left evaluation and refactoring for later. Reversible: yes, the remaining steps are additive.

### D2 — Resumability granularity
Options: **keep unit-level durable state and make each write cheaper**; keep unit-level state with a less frequent fsync; checkpoint at batch level. **Chosen: keep unit-level state.** Losing a long build is the failure people actually care about, so the granularity stays and the cost was attacked instead. Reversible: yes. Later amended once, deliberately, by D10.

### D3 — Should `search` return `stale`?
Options: keep as is; **add `include_staleness=false`, defaulting to `true`**; cached signature; flag-only. **Chosen: add the opt-out.** Callers who do not need freshness stop paying for the source walk without changing the default for callers who do. Reversible: yes.

### D4 — When may a policy version be bumped?
Options: **bump for semantic changes, keep cosmetic changes compatible**; bump for everything; never bump. **Chosen: semantic changes only.** Bumping for cosmetic edits would force needless rebuilds. Reversible: yes.

### D5 — Where may derived state live?
Options: document the fast-storage requirement; OS bind mount; **first-class validated `--runtime-root`, with the bind mount as the immediate stopgap**. **Chosen: `--runtime-root`.** Portable, validated, and it protects against two projects sharing one directory. Reversible: yes, drop the flag and move the data back.

### D6 — Dense backend
Options: exact scan only; **exact scan by default with the embedded index selectable above a documented size**; keep the embedded index and make it incremental. **Chosen: exact by default, embedded index available.** It removes the dominant cost at the target corpus sizes while keeping an ANN path for very large collections. Reversible: yes, it is a policy field, not a schema.

### D7 — Does `ingest` need scope control?
Options: no change; source filter; **read-only dry run**. **Chosen: dry run.** Derivation is already incremental per document, so the useful thing is to explain what would happen before it happens. Reversible: yes.

### D8 — Text-health rejection policy
Options: keep as is; **corruption-only rejection, script mixing as an advisory flag, and disclosed withholding**; the same with a per-source override; flag-only. **Chosen: corruption-only with disclosure.** It restores legitimate quotations without hiding the decision. Reversible: yes.

### D9 — Text normalisation folding
Options: keep NFC; full NFKC; **targeted fold of Mathematical Alphanumeric Symbols and Alphabetic Presentation Forms**; the fold plus a separate search field. **Chosen: the targeted fold.** Full NFKC would also fold superscripts, subscripts, and symbols that carry meaning in citations. Reversible: yes.

### D10 — Chunker batching versus the redo window
Options: one unit per call and drop the per-unit checkpoint write; **batch 16 extraction units per call**; leave chunking alone. **Chosen: batch 16.** It removes the round trip that dominated the phase. The cost is that a hard crash redoes at most one batch (16 units, a second or two of work) instead of exactly one unit, so D2's wording became "one bounded batch". Reversible: yes, one constant set back to 1.

## 5. The work, step by step

Each step states what it was for, what it changed, and how it ended. Steps 0–3 are done; 4–7 are planned.

### Step 0 — make the tree shippable
**Purpose.** Nothing can be measured or changed safely on a red tree. **Done.** Formatted four files; the suite went green. This is why every later number in this document is trustworthy.

### Step 1 — close the retrieval-fidelity gap
**Purpose.** Two findings changed what users could actually retrieve, so they came before any speed work.

**Step 1b — text-health policy (P0-4, D8, D9).** Split the internal text signals into *corruption* (which may withhold a passage) and *script notes* (advisory only), and made withholding visible: search responses now name the reason codes, counts, and example chunk IDs, and build metrics record corpus-level totals. Result on the real corpus: 29 withheld chunks became 25, restoring a Greek quotation, two Cyrillic bibliography entries, and one mixed-script chunk.

**Step 1c — normalisation folding (P0-5, D9).** Folded only formula-font letters (`𝑀` → `M`) and the ligatures `ﬁ`, `ﬂ`, `ﬀ`, before NFC. Result: 39 chunks using mathematical alphanumerics became 0, and 751 chunks in total changed for the better. Accents, superscripts, subscripts, and symbols are untouched.

**Step 1 — embedding audit (P0-2, P0-3).** Every chunk now records its embedding token count and whether its vector is truncated, using a non-truncating tokenizer loaded separately from the embedder. Per-hit and aggregate reporting follows. Confirmed the real numbers: 8 of 8,102 chunks exceed 512 tokens, the longest being 3,417. Over-limit chunks are *flagged, not split*, because splitting would change chunk identities and reuse behaviour.

### Step 1d — put derived state where you want it (P1-12, D5)
**Purpose.** The measured cost of index writes was ~290 ms per point on the HDD project versus ~4 ms on NVMe, so the device mattered more than any code change. **Done.** Documented the numbers and the bind-mount stopgap, then added `--runtime-root`: an absolute, validated location for derived state, claimed by a marker file naming its owning project. Portable review state stays in the project. `status` reports the effective location as `runtime_root`.

### Step A — exact dense search (P1-1, P1-11, D6)
**Purpose.** The embedded index was 99.1% of a real build for a 12.4 MB vector file that can simply be scanned. **Done.** Added an exact cosine-scan backend over the generation's portable vectors, and recorded in every manifest which backend built its index. Existing generations keep working because a manifest without the field resolves to the embedded backend. Measured on the real corpus: 0.03 s and 0.58 MB to build, versus 3,040.73 s for the embedded index, with the top 20 results identical to a brute-force ranking for every probe query. `--dense-backend` selects the backend; `auto` switches to the embedded index only above 200,000 chunks.

### Step 2a — make durability writes cheaper (P1-2, P1-3, D2)
**Purpose.** On the HDD, one atomic JSON write cost about 91 ms — 34 ms for the file fsync plus 57 ms for the directory fsync — and each unit performed several. **Done.** Grouped a unit's directory fsyncs into one, committed before the checkpoint that claims the unit complete, and stopped fsyncing a handoff file that exists only to be read by another process and is deleted straight after. Measured on a 64-unit HDD corpus: chunking 63.6 s → 26.3 s and 53.3 s → 35.8 s in the two orders, wall clock down 16–30%.

The plan's own proposal — shrinking the checkpoint file — was dropped because it would have saved about 0.2 ms per unit. The measurement, not the intuition, is recorded as the record in section 6.

### Step 2b — chunker batching (P1-2, D10)
**Purpose.** After Step 2a, chunking was still the largest phase because every extraction unit paid its own gateway round trip and its own checkpoint. **Done.** Up to 16 units now share one chunker call. Each returned chunk names its unit, so results are split back into the same per-unit files, in the same order, that a one-unit call produced; a unit the chunker omits still gets its own empty file, and an unknown unit still fails the build. Measured: chunking 25.6 s → 15.9 s and 29.0 s → 19.7 s, wall clock down 20–26%, with chunk identities asserted unchanged by test.

The plan predicted a larger gain (~3.8×); the measurement showed ~1.5×. The difference is the chunker's own work and the per-unit output files, which batching cannot remove. The estimate was wrong and the measurement replaced it.

### Step 2c — embedding throughput (P1-13)
**Purpose.** Embedding was 71.5% of a fresh build on fast storage. **Done.** Found that the inference batch size is really a *padding* decision: FastEmbed pads every sequence to the longest in its batch, so a batch of 64 made every short chunk as expensive as the longest one. The server now embeds one sequence per inference, which measured 23.7 chunks/s against 4.5 at a batch of 64, and returns bit-identical vectors, so no retrieval result can change. The machine-specific thread count became an option (`--embedding-threads`) rather than a guess.

### Step 3 — per-query work (P1-4, P1-5)
**Purpose.** Every query rescanned candidate text to decide whether a chunk was usable, at about 0.55 ms per candidate. **Done.** That verdict is a property of the chunk, not the query, so it is now computed once when the artifact lookup is built and stored as a small bitmask per chunk. The candidate gate measured 92.1 ms → 9.0 ms for 200 candidates (10.3×), with a fallback that recomputes the verdict for any lookup that predates the column. The query-dependent token check stays per query and is memoised, which also removed the widening loop's repeated work. `search` now hands its already-parsed manifest to the status summary instead of making it re-read one.

### Step 4 — per-request work (P1-6, D3)
**Purpose.** Every query re-compared the source directory with the generation to decide whether it was stale, a cost that grows with the collection rather than with the question. **Done.** D3 chose an explicit opt-out over a cache: `search` now takes `include_staleness` (default `true`), and when it is `false` the source-tree walk is skipped entirely and the response reports `stale=null` with `staleness_checked=false` rather than implying freshness. Re-measured on the reference project, the walk is 9.69 ms for 55 sources, so the opt-out is worth about 0.18 ms per source — roughly 176 ms per query at 1,000 sources. Upgrade reporting was split out into a manifest-only computation, so a caller that skips the walk still learns that the generation needs rebuilding. Three further items from the original list were dropped on measurement; section 6 has the numbers.

### Step 5 — decompose the ingestion logic (planned)
**Purpose.** The resumable ingestion loop is one very large method, which makes review risky. **To do.** Split it into per-phase handlers and turn the linter's complexity check back on.

### Step 6 — real evaluation (planned)
**Purpose.** Nobody has measured whether retrieval is *good*, only whether it works. **To do.** 30–50 judged queries against the real corpus, reporting BM25, dense, hybrid, and reranked quality.

### Step 7 — contract polish and retention (planned)
**Purpose.** Small correctness-of-meaning issues and unbounded disk use. **To do.** Generation listing and pruning, structured activation failure reports, clearer `stale` semantics (P2-3), an `ingest` dry run (D7), and narrowing the two broad exception handlers (P2-7).

## 6. Measurements

Every number quoted elsewhere comes from here. The `10.x` labels are kept unchanged so references in code and other documents still resolve.

The benchmark harness is `scripts/benchmark_write_pattern.py`. Run it against a device to reproduce the write-pattern, chunker-batching, and embedding numbers.

### 6.1 Baseline before this work

**Retrieval latency on the real project** (warm, through the MCP surface):

| Operation | Time |
|---|---|
| Client start-up, including the gateway | 2.73 s |
| First hybrid search (cold index load) | 9.19 s |
| Warm hybrid search, `top_k=8` | 0.52–0.68 s |
| Reference view, `top_k=8`, 2 per reference | 1.03 s |
| Reranked hybrid | 2.27 s |
| BM25 only | 0.11 s |
| Dense only | 0.50 s |
| `status` (warm) | 0.019 s |

**A 20-source corpus built from scratch** (1,073 chunks, 501 units):

| Phase | NVMe | HDD | Device ratio |
|---|---|---|---|
| embedding | 171.75 s (71.5%) | 156.67 s | ~1× (CPU-bound) |
| extraction | 43.05 s | 251.32 s | 5.8× |
| chunking | 21.41 s | 397.59 s | **18.6×** |
| embedded index build | 4.12 s | 310.97 s | **75.5×** |
| BM25 | 4.40 s | 3.82 s | ~1× |
| assembly | 0.14 s | 4.78 s | 34× |
| **wall clock** | **259.1 s** | **1,344.7 s** | 5.2× |

Adding one 715 KB source to that corpus:

| Phase | NVMe | HDD | Device ratio |
|---|---|---|---|
| extraction (1 new document) | 14.93 s | 152.82 s | 10.2× |
| chunking | 5.71 s | 179.02 s | **31.4×** |
| embedding (432 new vectors) | 66.52 s | 71.23 s | ~1× |
| embedded index build (all 1,505 points) | 5.91 s | 434.78 s | **73.6×** |
| assembly | 0.19 s | 7.15 s | 38× |
| **wall clock** | **96.1 s** | **950.7 s** | 9.9× |

What that table means: derivation was already incremental — the 20 unchanged documents, 1,073 chunks, and 1,073 vectors were reused exactly — but both indexes were rebuilt over the whole corpus, and on the HDD that dominated everything. Adding one paper to a 53-source project therefore cost about the same order as rebuilding it, which is why P1-1 and P1-11 led the list.

### 10.5 Per-unit write cost

Isolating one atomic write (27 KB payload) on the same two devices:

| Operation | HDD | NVMe |
|---|---|---|
| write + `os.replace`, no fsync | 0.40 ms | 0.36 ms |
| + file fsync | 33.95 ms | 1.62 ms |
| + directory fsync | 91.25 ms | 2.67 ms |
| 8 artifact files with one directory fsync | 803 ms/batch | 14.5 ms/batch |

**What it means.** On the HDD, a file fsync costs ~34 ms and a directory fsync ~57 ms, and payload size is irrelevant: a 17 KB checkpoint cost the same as a 35 KB one (192–264 ms with the original per-file pattern). This is why the fix became *fewer* fsyncs rather than *smaller* files, and why the planned write-once-inventory split was dropped: it would have saved ~0.2 ms per unit while adding another durability surface.

A/B on an identical 64-unit HDD corpus, real gateway, chunking, and embeddings, run in both orders because the second run of any pair sees a warmer cache:

| Phase | before → after | after → before |
|---|---|---|
| chunking | 63.59 → 26.26 s (−37.33) | 53.32 → 35.83 s (−17.49) |
| extraction | 46.90 → 38.58 s | 49.07 → 35.56 s |
| phase sum | 115.08 → 67.36 s | 105.59 → 73.98 s |
| wall clock | 170.07 → 118.56 s | 161.03 → 134.94 s |

Then the same measurement for chunker batching (D10):

| Phase | 1 unit per call → 16 | 16 → 1 unit per call |
|---|---|---|
| chunking | 25.56 → 15.88 s (−9.69) | 19.66 → 29.03 s (−9.36) |
| wall clock | 117.66 → 86.82 s | 105.84 → 132.03 s (−26.20) |

`assembly` stayed under 1.7 s in all four runs, so its change is inside the noise. That claim is deliberately *not* made.

### 10.6 Embedding throughput: the inference batch is a padding decision

FastEmbed pads every sequence to the longest member of its batch, and ONNX Runtime still computes the padded positions. Cost is therefore about `0.27 ms × padded tokens`, so a large batch makes every short chunk as expensive as the longest one in its batch.

Measured on 128 real chunk texts (mean 152 tokens, median 112, p90 330, max 846), 64 texts per configuration:

| Inference batch | 64 texts | chunks/s |
|---|---|---|
| 1 | 2.70 s | **23.68** |
| 4 | 8.13 s | 7.87 |
| 8 | 12.86 s | 4.98 |
| 16 | 12.08 s | 5.30 |
| 32 | 11.07 s | 5.78 |
| 64 | 14.27 s | 4.49 |
| 16, input sorted by length | 6.95 s | 9.21 |
| 64, input sorted by length | 13.31 s | 4.81 |

**What it means.** The token model predicts every row: batch 1 costs `sum(lengths) ≈ 9,728` tokens (2.6 s predicted, 2.70 s measured), batch 64 costs `64 × 846 = 54,144` (14.6 s predicted, 14.27 s measured). Sorting narrows each group but cannot beat a batch of one, which pads nothing. The previous `batch_size=64` — and FastEmbed's own default of 256 — were making this phase 1.5× to 4.5× more expensive than necessary.

Vector parity is exact rather than approximate: batch 1 and batch 64 returned **bit-identical** vectors (`max |delta| = 0.0`) on two independent 64-text slices, so no retrieval result can change.

Threads (ONNX intra-op and inter-op) at batch 1, same texts: runtime default 23.65 chunks/s, 1 → 9.31, 4 → 23.97, 8 → **31.66**, 16 → 20.03. The optimum is the physical core count (8 here), which is machine-specific, so `--embedding-threads` exists and defaults to unset. Setting it to 16 was *worse* than leaving it alone, which is why nothing is auto-detected.

End-to-end through the harness (90 chunks, variable page lengths, HDD, real gateway and model), both orders:

| Phase | batch 64 → 1 | batch 1 → 64 |
|---|---|---|
| embedding | 19.04 → 11.24 s (−41%) | 10.91 → 14.62 s (−25%) |
| wall clock | 119.32 → 109.15 s | 112.05 → 116.20 s |

### 10.7 Per-query candidate gate: precomputed verdicts

Measured on 200 real chunk texts, doing the work a query used to do per candidate:

| Stage | Before | After |
|---|---|---|
| artifact-lookup fetch including offset parse | 6.90 ms | 6.78 ms |
| corrupt-text scan | 58.32 ms | — moved to build time |
| searchable-content scan | 13.65 ms | — moved to build time |
| extraction-artifact check | 14.12 ms | — moved to build time |
| content-token extraction | 8.37 ms | 2.18 ms |
| **gate total** | **92.06 ms** | **8.96 ms** |

**What it means.** The gate is 10.3× cheaper at the reference view's 200 candidates, and about 1.4 ms instead of 15 ms at the default depth of 32. What remains is the sidecar read plus the query-dependent token check.

The verdict is stored as a bitmask per chunk, computed by one shared function that both the build and the query-time fallback use, so a rejection decision and the counters that report it cannot drift. Reason codes are recomputed only for a chunk the verdict flags as corrupt, because the response discloses them.

Cross-check on the real corpus after the change: the sidecar holds 8,102 verdicts — 8,073 healthy, 25 corrupt, 4 extraction artifacts — and the 25 matches the withheld-chunk count measured independently in Step 1b.

### 10.8 Per-request overhead: what was left alone, and why

Step 4 also proposed three smaller changes. Measuring them first is what removed them from the plan.

Measured on the reference project (53 indexed documents, 55 sources, 8,102 chunks), medians:

| Operation | Cost |
|---|---|
| source-tree walk (`scan_sources`, the staleness check) | **9.69 ms** for 55 sources |
| rebuilding the effective-document map | 0.615 ms |
| fingerprinting the metadata that would key a cached map | 0.013 ms |
| opening one artifact-lookup SQLite connection | 0.094 ms |
| artifact lookup built from scratch, plus one count query | 1.32 ms |
| the same count query on a warm instance | 0.319 ms |

**What it means.** The staleness walk is an order of magnitude larger than everything else in a query's fixed overhead, which is why it was the only one worth a public flag. The document map and connection reuse are each worth fractions of a millisecond at this size: caching the map would save ~0.6 ms per query while adding cross-request state that has to be invalidated correctly, and reusing connections would save ~0.09 ms per call while requiring one SQLite connection to be reachable from whichever thread serves the next call. Neither is a good trade against a correctness risk, and at the scale where they would matter — tens of thousands of documents — parsing the manifest itself would dominate both, so the real answer there is a persistent document-metadata index, not a per-process cache.

A cached staleness verdict behind a directory signature was rejected for a different reason: it would be wrong. The walk compares each source's recorded size and modification time, and a directory's own modification time does not change when a file inside it is replaced in place, so a signature-keyed cache would report a changed corpus as fresh. An explicit opt-out lets the caller choose; a cache would have made that choice for them, silently.

## 7. Deliberate limitations, and claims we are not making

Honesty about what has *not* been established matters as much as the numbers:

- **Retrieval quality is still unmeasured.** Everything here is about fidelity (does the right text survive?) and cost. Whether the ranking is good needs the judged query set in Step 6.
- **`assembly` is not measurably faster.** It stayed under 1.7 s in every run, so the change is inside the noise at the benchmark corpus size.
- **Dense latency above the current corpus size is extrapolated.** 35–68 ms per query was measured at 8,102 chunks. The 50k and 100k figures in earlier notes were arithmetic, not measurements; the `auto` backend switch is set at 200,000 chunks on that basis.
- **Over-limit chunks are flagged, not split.** Splitting them would change chunk identities, reuse behaviour, and therefore generation compatibility. The audit makes the limit visible instead.
- **Legacy generations are reused as they are.** A manifest written before the dense backend was recorded resolves to the embedded index, and a lookup written before the verdict column is rebuilt on first use. Nothing is regenerated silently.
- **Generation retention is unbounded.** Old generations keep their disk space until Step 7.
- **Two Qdrant-backend optimisations are open.** Keeping one client per phase and time-boxed upload batches only matter for corpora above 200,000 chunks now, so they are off the reference path rather than done.
- **The embedding thread count is left to the runtime.** The measured optimum is machine-specific, and a wrong guess measured *worse* than the default.

## 8. Where to look next

- `TODO.md` — the same work as an actionable checklist, with what is done and what is open.
- `AGENTS.md` and `AGENT_GUIDE.md` — the rules and invariants that keep this codebase coherent; they explain *how* to change things, not *what* was changed.
- `README.md` — the user manual.
- `scripts/benchmark_write_pattern.py` — reproduce the write, chunking, and embedding measurements on your own hardware.

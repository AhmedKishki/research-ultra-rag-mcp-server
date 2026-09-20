# TODO — what is done, what is left

`PLAN.md` explains *why* the work exists and records the measurements. This file is the actionable list. Read the plan first if a word here is unfamiliar.

How the list is organised:

- **Status at a glance** — where each piece of work stands.
- **Done** — finished work, with one line on what it achieved.
- **Open work** — what is left, grouped by area, each item saying what it would improve and how we would know it worked.
- **Decisions** — every decision that was made, and what was chosen.
- **Scope and verification** — the workload this server targets and how to re-check the current state.

## Status at a glance

| Area | Status |
|---|---|
| Step 0 — green the test suite | Done |
| Step 1 — retrieval fidelity (truncation, text policy, folding) | Done |
| Step 1d — derived state on fast storage | Done |
| Step A — exact dense search by default | Done |
| Step 2a — cheaper durability writes | Done |
| Step 2b — chunker batching | Done |
| Step 2c — embedding throughput | Done |
| Step 3 — per-query candidate gate | Done |
| Step 4 — per-request corpus work | Done, with two items dropped on evidence |
| Step 5 — split the ingestion loop for reviewability | Open |
| Step 6 — measure retrieval quality with judged queries | Open |
| Step 7 — contract polish and generation retention | Open |

## Done

**Correctness and fidelity**

- The test suite is green, so every later number rests on a working baseline.
- Chunks that exceed the embedding model's token limit are now counted and flagged per chunk and in aggregate (8 of 8,102 on the reference corpus), so silent truncation is visible.
- The text-health policy withholds a passage only for corruption evidence. Script mixing is advisory, withholding is disclosed with reason codes and example IDs, and 4 previously-lost legitimate chunks are retrievable again (29 withheld → 25).
- Formula-font letters and the ligatures `ﬁ`, `ﬂ`, `ﬀ` fold to plain spellings so typed queries match printed text; accents, superscripts, and symbols are left alone.

**Speed and cost**

- Dense search scans the generation's portable vectors exactly by default. The index build went from 3,040.73 s to 0.03 s on the reference corpus, with identical top-20 results against a brute-force check.
- Durability writes are grouped per work unit instead of one fsync per file, which on HDD cut chunking by 17–37 s in two measurement orders.
- Up to 16 extraction units share one chunker call, cutting chunking a further 9 s per 64 units with chunk identities unchanged.
- Embedding runs one sequence per inference instead of batches of 64, which is 1.5× to 4.5× faster depending on how varied chunk lengths are, with bit-identical vectors.
- The per-query candidate gate is 10.3× cheaper (92.06 ms → 8.96 ms at 200 candidates) because the usability verdict is computed once when the lookup is built instead of rescanned on every query.
- `--runtime-root` lets a project keep derived state on fast storage without an OS-level bind mount, with validation so two projects cannot share one root.
- `search` can skip the source-directory freshness check with `include_staleness=false`. That check is the only per-query cost that grows with the collection: measured at 9.69 ms for 55 sources (about 0.18 ms per source, so roughly 176 ms at 1,000 sources, though at this size it is too small to see in end-to-end latency). When the check is skipped the response says so, reporting `stale=null` and `staleness_checked=false` rather than implying the generation is current.

**Docs and tooling**

- `README.md` documents the options, the storage layout, and what each feature is for.
- `scripts/benchmark_write_pattern.py` reproduces the write, chunking, and embedding measurements on any machine.

### Step 4 — per-request work (closed)

Two items from this step were dropped on evidence rather than left open, so the next reader does not have to rediscover why.

- Pushing category and keyword filters into the dense backend, reusing SQLite connections, and caching the document map all turn out to be in the 0.1–0.6 ms range per query. On the reference corpus the document map costs 0.615 ms to rebuild and the fingerprint that would invalidate a cached copy costs 0.013 ms; one SQLite connection costs 0.094 ms and a search opens a handful, so reuse would save roughly half a millisecond while making one connection reachable from several threads, which is a correctness risk for no measurable gain. Filtering is already applied while candidate text is scanned in the default backend, so the push-down only matters for the embedded index — off the reference path below 200,000 chunks.
- A cached staleness verdict behind a directory signature was rejected as unsound, not merely slow. A directory's modification time does not change when a file inside it is replaced in place, so the cache could report "fresh" for a corpus that had changed. The caller's explicit opt-out is a correct choice where a fast cached answer would have been a wrong one.

## Open work

### Step 5 — make the ingestion loop reviewable

The resumable ingestion function is enormous (over a thousand lines), which makes every change to it risky to review.

What to do: split it into one handler per phase (hashing, extraction, chunking, assembly, embedding, indexing, revalidation) and turn the linter's complexity check back on.

**Done when:** the complexity check passes without exclusions and the existing resumability tests still pass unchanged.

### Step 6 — measure whether retrieval is actually good

Nothing here has measured *quality*: only whether the right text survives and what it costs. Without judged queries, "is hybrid better than BM25 here?" is opinion.

What to do: write 30–50 judged queries against the reference corpus and record BM25, dense, hybrid, and reranked quality.

**Done when:** the numbers exist and any default that the results contradict is either changed or explicitly justified.

### Step 7 — contract polish and disk retention

Small items that affect meaning or disk use:

- List and prune generations. They are ~101 MB each and currently accumulate forever.
- Report activation failures structurally instead of one all-or-nothing message.
- Make `stale` mean one thing in the no-generation branch (it currently doubles as "ready to build").
- Add an `ingest` dry run that explains what would change without writing.
- Narrow the two broad exception handlers at durability boundaries so real storage faults are not swallowed.
- Two Qdrant-backend-only optimisations remain open and are off the reference path because that backend now applies only above 200,000 chunks: one client per phase, and time-boxed upload batches.
- Decide and add a repository licence. There is currently no `LICENSE` file and no licence field in `pyproject.toml`, even though this server uses Apache-2.0 UltraRAG and depends on third-party models. `NOTICE` covers upstream attribution but not the licence of this repository's own code, so the terms a user receives today are unstated. This is a decision for the owner, not something to be guessed at.

## Decisions

All decisions are settled. `PLAN.md` section 4 has the options and reasoning.

| ID | Question | Chosen |
|---|---|---|
| D1 | Scope of this pass | Fidelity, storage location, exact dense search, write-path work |
| D2 | Resumability granularity | Keep unit-level durable state; make each write cheaper |
| D3 | Should `search` return `stale`? | Add `include_staleness=false`, default `true` |
| D4 | When to bump a policy version | Semantic changes only, not cosmetic ones |
| D5 | Where derived state may live | Validated `--runtime-root`, bind mount as a stopgap |
| D6 | Dense backend | Exact scan by default, embedded index above a size threshold |
| D7 | `ingest` scope control | Read-only dry run |
| D8 | Text-health rejection | Corruption only, with disclosure; script mixing is advisory |
| D9 | Text normalisation | Fold only formula letters and the three ligatures |
| D10 | Chunker batching | 16 extraction units per call, redo bounded to one batch |

## Scope and verification

**Target workload.** 25–250 English-primary born-digital sources, 5,000–50,000 chunks, CPU only, humanities and social-science prose with footnotes and reference apparatus. Out of scope: non-English-primary corpora, multilingual retrieval, OCR or scanned sources, handwriting, and formula-heavy corpora.

**Re-check the current state:**

```bash
uv run pytest -q              # full suite
uv run ruff check .           # lint
uv run ruff format --check .  # formatting
```

**Re-produce a measurement:**

```bash
uv run python scripts/benchmark_write_pattern.py --root /path/on/target/disk
```

**Verify against a real project (read-only):**

```bash
uv run research-ultra-rag-verify /path/to/project --query "your question"
```

**What was verified on the real corpus:** retrieval returns the expected passages with correct locators; the sidecar's stored verdicts (8,073 healthy, 25 corrupt, 4 artifacts) match the independently measured withheld count; and the dense backend's top-20 matches a brute-force ranking.

# Measurements

Every number in this file is a current, reproducible property of this server and the workload it targets. It describes no past state of the code: history lives in git. Three harnesses produce all of it, and each drives the real gateway, the real models, and the public tool surface:

```bash
uv run python scripts/benchmark_write_pattern.py --root /path/on/target/disk
uv run python scripts/evaluate_retrieval.py --project /path/to/project --offline
uv run python scripts/measure_tool_payloads.py --project /path/to/project --offline
```

## 1. Workload and design envelope

Built and measured for 25–250 English-primary born-digital PDF/EPUB sources, 5,000–50,000 chunks, on CPU only, with humanities and social-science prose that carries footnotes, reference lists, tables, and short quotations in other scripts. Out of scope, and not claimed: non-English-primary corpora, multilingual retrieval, OCR or scanned material, handwriting, and formula-heavy corpora.

### Reference corpus

| Property | Value |
|---|---|
| Sources | 57 discovered, 55 indexed, 2 reviewed exclusions |
| Source material | 135 MB of PDF and EPUB originals |
| Documents | 55 |
| Extraction units | 9,546 |
| Chunks | 13,158 (about 239 per source) |
| Vectors | 13,158 × 384 float32 = 20.2 MB |
| Generation on disk | 68 MB in 14 files |
| Generation schema | 5 |
| Dense index | portable float32 vectors, exact cosine scan |
| Character | English humanities and social-science prose with footnotes, reference lists, tables, and short quotations in other scripts |

## 2. Ingestion cost

### Full rebuild of the reference corpus on NVMe storage, CPU only

| Phase | Time |
|---|---|
| source hashing | 1.7 s |
| extraction | 281.9 s |
| chunking | 51.3 s |
| embedding (13,158 chunks) | 668.4 s |
| BM25 index | 2.2 s |
| dense index (exact scan) | 0.07 s |
| assembly | 1.3 s |
| **total** | **about 1,007 s (17 minutes)** |

Embedding dominates, and it is CPU-bound: phase timings are only weakly device-dependent. The same build reports 8 of 13,158 chunks above the embedding model's 512-token input limit (the largest auditing at 3,417 of that tokenizer's tokens, because the chunker counts GPT-2 tokens and the embedding model counts its own). They are counted and flagged per chunk rather than split, because splitting would change chunk identities and reuse without a long-input strategy. The build also discarded 1 corrupt chunk and excluded 16 corrupt extraction units, each disclosed with its locator and reason.

### What durability costs, and why writes are grouped

One atomic write, 17–35 KB payload, measured on both devices:

| Operation | HDD | NVMe |
|---|---|---|
| write + `os.replace`, no fsync | 0.40 ms | 0.36 ms |
| + file fsync | 33.95 ms | 1.62 ms |
| + directory fsync | 91.25 ms | 2.67 ms |
| 8 artifact files sharing one directory fsync | 803 ms/batch | 14.5 ms/batch |

Payload size is irrelevant on the HDD: a 17 KB checkpoint costs the same as a 35 KB one. Cost follows the number of durability operations, so a unit's artifacts are written together and committed with one directory fsync before the checkpoint that claims the unit is complete. Grouping cut chunking by 17–37 s on a 64-unit HDD corpus. `assembly` stayed under 1.7 s in every run, which is inside the machine's noise, so it is not claimed as an improvement.

### Chunking in batches of 16 extraction units

| Phase | One unit per call | 16 units per call |
|---|---|---|
| chunking | 25.56 s | 15.88 s |
| wall clock | 117.66 s | 86.82 s |

Chunk identities are identical either way. A crash redoes at most one batch of 16 units.

### Why the dense index is an exact scan by default

| Backend | Full rebuild of the reference corpus |
|---|---|
| embedded ANN index (Qdrant local mode) | 3,040.73 s and a 105 MB generation, measured on the project's HDD |
| exact scan of the generation's portable vectors | 0.07 s of index work and a 68 MB generation, on NVMe |

Three things decide this. The embedded build is dominated by per-point device cost (75.5× slower on that HDD than the same build on NVMe). It returns neither chunk identity nor scores through the upstream API, which the research contract needs. And the exact scan reads the portable float32 vectors the generation already stores, so it needs no separate index and returns the same top 20 as a brute-force ranking. Above 200,000 chunks the exact scan stops being the right default and the embedded backend is chosen instead; that threshold is arithmetic from measured exact-scan cost, not a measurement at that size.

### Embedding: the inference batch is a padding decision

FastEmbed pads every sequence to the longest member of its batch, and ONNX Runtime still computes the padded positions, so cost is about `0.27 ms × padded tokens`. Measured on 128 real chunk texts (mean 152 tokens, median 112, p90 330, max 846), 64 texts per configuration:

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

The token model predicts every row: a batch of one costs `sum(lengths) ≈ 9,728` tokens (2.6 s predicted, 2.70 s measured) while a batch of 64 costs `64 × 846 = 54,144` (14.6 s predicted, 14.27 s measured). Sorting narrows each group and still cannot beat a batch of one, which pads nothing. `EMBEDDING_INFERENCE_BATCH_SIZE` is therefore 1, and the vectors are identical either way: batch 1 and batch 64 returned bit-identical floats (`max |delta| = 0.0`) on two independent 64-text slices.

Threads (ONNX intra-op and inter-op) at batch 1 on the same texts: runtime default 23.65 chunks/s, 1 → 9.31, 4 → 23.97, 8 → **31.66**, 16 → 20.03. The optimum was the physical core count, which is machine-specific, so `--embedding-threads` is unset by default: setting it to 16 measured *worse* than leaving it alone, and nothing is auto-detected.

## 3. Candidate gating and per-query overhead

### The candidate gate

A chunk's usability verdict is a property of the chunk, not of the query, so it is computed once when a generation's artifact lookup is built and stored as a small bitmask per chunk. Measured on 200 real chunk texts, per query:

| Stage | Stored verdict | Recomputed per query |
|---|---|---|
| artifact-lookup fetch including offset parse | 6.78 ms | 6.90 ms |
| corrupt-text scan | at build time | 58.32 ms |
| searchable-content scan | at build time | 13.65 ms |
| extraction-artifact check | at build time | 14.12 ms |
| content-token extraction (query-dependent) | 2.18 ms | 8.37 ms |
| **gate total** | **8.96 ms** | **92.06 ms** |

That is 10.3× cheaper at the 200 candidates a reference-view query can reach, and about 1.4 ms instead of 15 ms at the default depth of 32. One shared function computes the bitmask for both the build and the query-time fallback, so a rejection decision and the counter that reports it cannot drift. Reason codes are recomputed only for a chunk the verdict flags as corrupt, because the response discloses them, and a generation whose lookup predates the column recomputes the verdict from text.

### Fixed overhead per query

Measured on the reference project, medians:

| Operation | Cost |
|---|---|
| source-tree walk (`scan_sources`, the staleness check) | **9.69 ms** for 55 sources |
| rebuilding the effective-document map | 0.615 ms |
| fingerprinting the metadata that would key a cached map | 0.013 ms |
| opening one artifact-lookup SQLite connection | 0.094 ms |
| artifact lookup built from scratch plus one count query | 1.32 ms |
| the same count query on a warm instance | 0.319 ms |

The staleness walk is an order of magnitude larger than everything else and is the only term that grows with the number of source files: about 0.18 ms per source, so roughly 176 ms at 1,000 sources. That is why `include_staleness=false` exists. Nothing else here is cached, deliberately: caching the document map saves about 0.6 ms per query while adding cross-request state that must be invalidated correctly, and reusing SQLite connections saves about 0.09 ms while requiring one connection to be reachable from whichever thread serves the next call. At the scale where either would matter, parsing the manifest dominates both, so the right answer there is a persistent document-metadata index rather than a per-process cache. A staleness verdict cached behind a directory signature was rejected because it would be wrong: a directory's own modification time does not change when a file inside it is replaced in place, so such a cache would report a changed corpus as fresh. At this corpus size the flag does not change how fast a search feels — a warm hybrid query measured 567 ms with the check and 583 ms without it, inside the machine's run-to-run variance — and it is not claimed to.

### Tool answer size

`scripts/measure_tool_payloads.py` reports the JSON UTF-8 size of one lean answer and of the same call in `--tool-detail full`. Measured on the reference corpus with its derived state cleared — 63 discovered sources, 59 selected, 4 excluded, 59 reviewed-metadata entries, no selected generation — hybrid default, reranking enabled:

| Tool | Lean answer | Full-detail answer | Ratio |
|---|---|---|---|
| `status` | 392 bytes | 9,877 bytes | 0.04 |
| `list_sources` | 38,144 bytes | 72,503 bytes | 0.53 |
| `search`, `top_k=6` | not re-measured | — | — |

Two things dominate. A lean `status` is small because it reports the pending-review count, not the 59 reviewed paths the full payload lists, because it counts added and modified sources instead of listing them while naming only the ones that went missing, and because it describes the selected generation instead of inventorying anything: the retained generations, the categories and the projects are full-detail readers, so no status answer lists them however large the corpus grows. `list_sources` is the largest lean answer and stays largest because `reviewed_metadata_sources` echoes every saved override: that is also the only place the keyword vocabulary is visible, since a status answer inventories no vocabulary at all.

`search` needs a built generation, which the cleared measurement state did not have. Re-run on the reference project once it had one — 59 indexed sources, 14,072 chunks, 5 retained generations, hybrid retrieval with reranking, `top_k=6`, query "the multiplication of labour in the data supply chain":

| Tool | Lean answer | Full-detail answer | Ratio |
|---|---|---|---|
| `status` | 605 bytes | 9,410 bytes | 0.06 |
| `list_sources` | 54,981 bytes | 149,155 bytes | 0.37 |
| `search`, `top_k=6` | 10,691 bytes | 23,155 bytes | 0.46 |

A lean `status` is larger than the cleared-state figure above because it carries the current generation, its counts, and the retained count and bytes; it is also smaller than the earlier built-state measurement of 2,664 bytes, which predates moving the retained-generation, category and project inventories to the full-detail payload. A lean `list_sources` is larger than its cleared-state figure because the reviewed-metadata overlay now holds every saved override. A lean `search` at `top_k=6` is mostly the cleaned evidence itself: the accounting that used to surround it — component ranks and scores, fusion and reranker values, embedding token counts, candidate and rejection counts, withheld candidates, model identifiers and revision fingerprints, per-field provenance, and the applied-filter echo — is now returned only under `--tool-detail full`, and a passage itself carries only its source, its authors, its position, and its text, so the citation, the resolved title, the stable source ID, a page label that merely repeats the physical page, the quote-safety flag that would repeat once per passage, and the advisory script note are full-detail material too. The pre-lean reference figure of 10,799 lean bytes against 19,831 full, recorded before the per-passage `rank`, `year`, `doi`, and `metadata_warnings` fields were removed, is superseded by these numbers.

The returned passage text is about 6 kB of both search answers, so a search ratio is bounded by how much evidence was requested. The two source lists grow with the number of sources and `search` grows with `top_k` and passage length, so absolute bytes track the corpus while the ratio carries over.

## 4. Retrieval quality and query latency

Measured through `ResearchService.search`, the engine the `search` tool calls, on the current generation (13,158 chunks, 55 documents): 32 judged queries over 19 target passages, `include_staleness=false`, 160 searches in 107 s (mean 0.604 s), after a discarded 10.44 s warm-up. `scripts/evaluate_retrieval.py` produces these tables and a per-query JSON report; `evaluation/README.md` documents the judged set and its protocol.

### Per-mode latency

| Mode | Mean per search | Mean passages returned |
|---|---|---|
| BM25 | 0.084 s | 10.0 |
| Dense | 0.120 s | 5.3 |
| Hybrid | 0.173 s | 10.0 |
| Hybrid + rerank | 2.295 s | 10.0 |

Dense and hybrid are this cheap because the index is the exact scan: an embedded ANN index costs 0.5–0.6 s per query at this corpus size, since every query reopens it.

### Accuracy at `top_k=10`

`succ@k` is the share of queries whose judged passage is inside the first `k`; `doc@k` is the share where at least the right document appears, which separates a ranking miss from a coverage miss; `overlap` is the mean share of a query's content words that its judged passage contains.

| Mode | succ@1 | succ@3 | succ@10 | MRR | nDCG@10 | doc@10 | overlap | mean returned |
|---|---|---|---|---|---|---|---|---|
| BM25 | 59.4% | 75.0% | 84.4% | 0.671 | 0.713 | 87.5% | 0.659 | 10.0 |
| Dense | 46.9% | 53.1% | 62.5% | 0.513 | 0.539 | 71.9% | 0.659 | 5.3 |
| Hybrid | 65.6% | 78.1% | 84.4% | 0.714 | 0.745 | 90.6% | 0.659 | 10.0 |
| Hybrid + rerank | **81.2%** | **84.4%** | **87.5%** | **0.836** | **0.846** | **93.8%** | 0.659 | 10.0 |

### Accuracy by query class at `top_k=10`

| Mode / class | n | succ@1 | succ@3 | succ@10 | MRR | doc@10 |
|---|---|---|---|---|---|---|
| BM25 / quote | 11 | 90.9% | 100.0% | 100.0% | 0.939 | 100.0% |
| BM25 / paraphrase | 11 | 36.4% | 45.5% | 54.5% | 0.422 | 63.6% |
| BM25 / entity | 10 | 50.0% | 80.0% | 100.0% | 0.651 | 100.0% |
| Dense / quote | 11 | 63.6% | 63.6% | 81.8% | 0.657 | 90.9% |
| Dense / paraphrase | 11 | 54.5% | 54.5% | 54.5% | 0.545 | 72.7% |
| Dense / entity | 10 | 20.0% | 40.0% | 50.0% | 0.320 | 50.0% |
| Hybrid / quote | 11 | 81.8% | 100.0% | 100.0% | 0.879 | 100.0% |
| Hybrid / paraphrase | 11 | 54.5% | 54.5% | 54.5% | 0.545 | 72.7% |
| Hybrid / entity | 10 | 60.0% | 80.0% | 100.0% | 0.718 | 100.0% |
| Hybrid + rerank / quote | 11 | 100.0% | 100.0% | 100.0% | 1.000 | 100.0% |
| Hybrid + rerank / paraphrase | 11 | 63.6% | 63.6% | 63.6% | 0.636 | 81.8% |
| Hybrid + rerank / entity | 10 | 80.0% | 90.0% | 100.0% | 0.875 | 100.0% |

### Deep pass, hybrid at `top_k=50`

succ@50 90.6%, MRR 0.717, nDCG@50 0.759, doc@50 100.0%. Paraphrase reach rises from 54.5% at 10 to 72.7% at 50.

### What these numbers support

- **Reranking is the largest single quality gain**: first-position success rises from 65.6% to 81.2%, MRR from 0.714 to 0.836, and document-level success from 90.6% to 93.8%, for 2.295 s per query against 0.173 s. It is the only configuration that returns the judged passage first for all 11 quote queries, and it is on by default for that reason.
- **Hybrid orders better than BM25 alone** — succ@1 65.6% against 59.4%, MRR 0.714 against 0.671, doc@10 90.6% against 87.5% — while reach at 10 is identical at 84.4%. Choosing BM25 for its speed gives up ordering quality, not coverage.
- **Dense alone is the weakest mode** at 62.5% within 10 results, and returns fewer passages (5.3) because the 0.72 cosine gate rejects most of its candidates. Its worst class is entity queries (50% within 10, MRR 0.320): a proper noun needs the words to match, which is what BM25 is for.
- **Paraphrase is the hardest class for every mode** — 54.5% for BM25, dense, and hybrid, and 63.6% after reranking. Four of the eleven paraphrase queries are missed by every mode at `top_k=10`; two of those return the right document at the wrong rank, and depth recovers class reach to 72.7% at 50. The gap is part depth and part ordering, not the relevance gates.
- **The judged set behaves as designed**: mean query-to-target content-word overlap is 0.976 for quote queries, 0.327 for paraphrase, and 0.676 for entity queries, so the class labels describe what they claim to.

### What these numbers do not establish

- The judgments are known-item and single-annotator. A mode that returns a different passage making the same point is scored as a miss, and no pooled judgment exists, so no true recall figure is claimed.
- They come from one English-primary corpus and one generation, not from a benchmark suite.
- Fusion weights are deliberately untouched: with 32 queries the hybrid lead at rank 1 is about two queries, which cannot separate a real weight effect from noise.

### The reranker model: the default against `jinaai/jina-reranker-v1-turbo-en`

Every accuracy number above was measured with the default cross-encoder. The model is an engine setting rather than a search option, so the harness can measure a second one over the same judged queries in one run:

```bash
uv run python scripts/evaluate_retrieval.py --project /mnt/data/my-project \
    --modes hybrid,hybrid+rerank \
    --reranker-model Xenova/ms-marco-MiniLM-L-6-v2 \
    --reranker-model jinaai/jina-reranker-v1-turbo-en
```

Measured on the current generation (14,072 chunks, 59 documents) with 30 of the 32 judged queries at `top_k=10` and no deep pass: 90 searches in 191 s. Two queries are missing because they judge target `t12`, whose source the reviewer excluded on 2026-09-20 as superseded by the Duke reprint in "STUART HALL, SELECTED WRITINGS ON RACE AND DIFFERENCE.pdf"; the current generation no longer holds it, so the run names it (`--skip-targets t12`) and reports 30 evaluated queries rather than quietly resolving fewer targets.

| Reranker | succ@1 | succ@3 | succ@10 | MRR | nDCG@10 | doc@10 | mean s | p50 s | max s |
|---|---|---|---|---|---|---|---|---|---|
| none (hybrid) | 66.7% | 80.0% | 86.7% | 0.728 | 0.762 | 90.0% | 0.19 | 0.19 | 0.51 |
| `Xenova/ms-marco-MiniLM-L-6-v2` (default) | **83.3%** | **86.7%** | **90.0%** | **0.858** | **0.869** | **93.3%** | **2.34** | 2.30 | 3.07 |
| `jinaai/jina-reranker-v1-turbo-en` | 73.3% | 86.7% | 90.0% | 0.794 | 0.821 | 93.3% | 3.47 | 3.26 | 11.69 |

By query class, at rank 1:

| Reranker / class | n | succ@1 | succ@3 | MRR |
|---|---|---|---|---|
| ms-marco / quote | 10 | 100.0% | 100.0% | 1.000 |
| ms-marco / paraphrase | 10 | 70.0% | 70.0% | 0.700 |
| ms-marco / entity | 10 | 80.0% | 100.0% | 0.875 |
| jina turbo / quote | 10 | 90.0% | 90.0% | 0.917 |
| jina turbo / paraphrase | 10 | 50.0% | 70.0% | 0.567 |
| jina turbo / entity | 10 | 80.0% | 100.0% | 0.900 |

- **The default stays, and the measurement is why.** jina turbo reaches the same depth — succ@10 90.0% and doc@10 93.3% are identical — but it puts the judged passage first 22 times out of 30 against ms-marco's 25, and it costs about half again as much per query (3.47 s against 2.34 s). It is therefore neither the better nor the cheaper default here. Its one gain is entity ordering, where MRR rises to 0.900 from 0.875.
- **The two rerankers disagree about rank on only 4 of 30 queries**, and ms-marco is the better of the two on 3 of them: `q06` (quote, rank 1 against 6), `q16` and `q18` (paraphrase, rank 1 against 3 each). jina turbo wins `q27` (entity, rank 2 against 4). A 30-query set can support "the default is not worse", not a fine-grained model ranking; treat the split as directional.
- **The slower model is the smaller one here**, which is an ONNX-export property rather than a parameter to tune: both models run through the same FastEmbed cross-encoder class with the runtime's default thread count, and the first jina turbo call in the run (model load included) is the 11.69 s maximum in the table.
- **Changing the model is an operator decision, not a per-search one**: `--reranker-model` (or `RESEARCH_ULTRARAG_RERANKER_MODEL`) changes every search the server answers, and `search(rerank_model=...)` changes it for one engine call. Re-measure before trusting either on a different corpus, because the ordering above is specific to these queries.

### Reply depth: what the caller's top_k buys

The reranker reorders `min(candidates, rerank_max_candidates, max(top_k * 2, 10))` passages, so the depth a caller asks for is also the depth the ranking reaches: at `top_k=8` only sixteen candidates are ever reranked, and the cap of fifty never binds at any depth measured here. That makes `top_k` the cheapest quality lever this server has.

Measured on the current generation with reranked hybrid, 30 of the 32 judged queries and no deep pass, with the lean answer size of one search at the same depth:

| `top_k` | succ@1 | succ@3 | succ@k | MRR | nDCG@k | doc@k | lean answer | full detail |
|---|---|---|---|---|---|---|---|
| 8 | 80.0% | 83.3% | 86.7% | 0.825 | 0.835 | 90.0% | 7,448 B | 22,151 B |
| 10 | **83.3%** | 86.7% | 90.0% | **0.858** | 0.869 | 93.3% | 8,050 B | 25,808 B |
| 12 | **83.3%** | 86.7% | 90.0% | **0.858** | 0.869 | 93.3% | 10,448 B | — |
| 15 | **83.3%** | 86.7% | **93.3%** | **0.861** | **0.877** | **96.7%** | 13,523 B | 38,532 B |

- **The plateau starts at ten, and eight is the only depth measured below it.** One query out of thirty separates 8 from the rest, so the difference is directional rather than decisive on this set; it is consistent across every metric and the mechanism is visible in the window arithmetic above, since 8 reranks sixteen candidates and 10 reranks twenty.
- **The tool default moved from 8 to the smallest depth that keeps the measured quality, which is 10.** That costs 602 bytes of lean answer for the gain, and it aligns the default with the depth every published number was taken at. Asking for 15 costs 5,473 bytes more than 10 and buys reach (`doc@k` 93.3% to 96.7%), which is worth it when a first answer is thin and the caller can afford the context.
- **Payload grows by roughly 750 to 870 bytes per passage** in a lean answer, and by about 1.6 kB per passage with `--tool-detail full`, so depth is cheap in lean mode and more expensive when a developer debugging session asks for everything.
- **`rerank_max_candidates` and the candidate window are not levers at this corpus size.** Raising the cap to 100 or 150, and the window to 50 minimum and 600 maximum, changed no ranking decision at all, because the window formula never reaches them. They stay settings because a larger corpus could reach them, not because they moved anything here.

## 5. Current limits

- **Retrieval quality is measured, not settled.** The numbers above describe findability of one designated passage per query on one corpus; pooled judgments, a second annotator, and a second corpus are open work in `TODO.md`.
- **Dense cost above this corpus size is extrapolated.** The 200,000-chunk switch to the embedded backend is arithmetic from the measured exact-scan cost, not a measurement at that size.
- **Over-limit chunks are flagged, not split.** Splitting would change chunk identities and reuse behaviour, so the build counts and flags them instead.
- **Older generations are read as they are.** A schema-1 generation stays BM25-only until it is re-ingested, and a lookup written before the retrieval-verdict column is recomputed on first use rather than regenerated silently.
- **Generation retention is unbounded.** Nothing prunes generations; `status` reports what each one occupies and removal is manual.
- **The embedding thread count is left to the runtime.** The measured optimum is machine-specific, and forcing the value measured worse than the default on the reference machine.
- **The embedded backend is unoptimised for corpora far above this one.** One client per phase and time-boxed upload batches are unbuilt, and both only matter above the 200,000-chunk threshold where that backend is selected at all; nothing here measures that size, so the cost above it is arithmetic rather than a reading.
- **Extraction is English-oriented.** The embedding model, the text-health policy, and the chunker target English-primary prose; other scripts appear as quotations inside it and are marked advisory rather than withheld.
- **`assembly` cost sits inside the machine's noise.** It stayed under 1.7 s in every measured run, so no claim is made about it.

## What a ranking change costs the corpus

The reuse snapshot never depended on the ranking policy. `generation_is_reusable` compares the schema, extraction, cleaning and artifact policy versions, the project id, the chunk size and overlap, and the embedding model — and nothing else — so weights, gates, caps and the reranked window cannot affect what a corpus holds. Measured on the German reference corpus, changing `retrieval.rrf_k` from 60 to 30 and re-ingesting:

| Quantity | Result |
|---|---|
| new generation | yes (`generation_changed: true`) |
| documents reused / rebuilt | 13 / **0** |
| chunks reused / rebuilt | 9,237 / **0** |
| vectors reused / created | 9,237 / **0** |
| phase timings | extraction 9.4 s, chunking 6.0 s, embedding 35.3 s, bm25 2.2 s, assembly 20.6 s |

About two minutes end to end against a full build's seventeen, which is what makes a retrieval experiment that reaches a new baseline cheap to apply.

What *did* depend on the policy was the staging checkpoint identity: it included the retrieval-policy fingerprint, so editing a ranking value discarded a build **in progress**, which is why an interrupted build could not be resumed across a ranking edit. The policy was removed from that identity (`INGESTION_IDENTITY_POLICY_VERSION` 3), which touches only disposable staging — a published generation records its policy in its manifest and was never identified by this fingerprint — and a test now pins that a ranking change reuses every chunk and vector.

## Pseudo-relevance feedback, measured and its shortfall diagnosed

`retrieval.prf` was implemented as the TODO described: search the lexical half once, mine terms from the leading passages, search again with them, and use the second ranking for the fusion. Turned on and off against the same judged set and the same generation, with the harness reporting every mode:

| Configuration | succ@1 | succ@3 | succ@k | MRR | nDCG | doc@k | mean s |
|---|---|---|---|---|---|---|---|
| `prf=false` (shipped default) | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% | 2.37 |
| `prf=true`, 5 leaders, 8 terms | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% | 1.62 |

No quality metric moved, and the latency difference is run order rather than the feature: the second run followed the first and met warm caches, while the expansion can only add a pass. The feature itself is live — the payload reports what it added, and a direct search with `RESEARCH_ULTRARAG_RETRIEVAL_PRF=true` returned `prf_terms: [about, between, have, can, change, conflicts, data, digital]`.

That list is the finding. Term selection ranks candidates by how many of the leading passages contain them, and the only filter is `_content_tokens`, whose stopword set is bm25s's 33-word English list. Words like *about*, *between*, *have* and *can* clear it, so the expansion adds terms that appear everywhere and discriminate nothing, which is why eight of them changed no ranking decision. The technique has not been shown to be worthless; this *selection rule* has been shown to be too weak, and the fix is a corpus-rarity weight — the classic term-selection criterion — rather than leader frequency alone.

Until that is tried, the default stays off and the refinement is recorded in `TODO.md`.

# TODO

The open work on this server, in the order it is intended to be attempted: one tier at a time, each measured before the next starts. Tier 0 is applied; Tier 1 is next. Everything listed here is unimplemented. Current behaviour is in `README.md`, current facts and limits are in `MEASUREMENTS.md`, capabilities are in `FEATURES.md`, and product ideas that are not in scope yet are in `ROADMAP.md`. Finished work lives in git history, not in this file.

## The measurement this plan rests on

Every tier below is judged against one baseline, taken on the reference project with the default settings, 30 of the 32 judged queries (`t12` judges a source the reviewer excluded), reranked hybrid at `top_k=10`:

| Configuration | succ@1 | succ@3 | succ@k | MRR | nDCG | doc@k |
|---|---|---|---|---|---|---|
| default (rerank cap 50, candidates 20/200) | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% |
| `rerank_max_candidates=100` | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% |
| candidates 50/600, `rerank_max_candidates=150` | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% |
| `dense_weight=1.5` | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% |
| `top_k=15` | 83.3% | 86.7% | **93.3%** | **0.861** | **0.877** | **96.7%** |

Four of those variants changed no ranking decision at all, and the reason is in the engine: the reranked window is `min(candidates, rerank_max_candidates, max(top_k * 2, 10))`, so at `top_k=10` only twenty candidates are ever reranked and the cap of fifty never binds. Raising `top_k` widened that window and recovered part of the paraphrase loss; the depth ladder built on that observation is in `MEASUREMENTS.md`, and the tool default moved from 8 to 10 as the smallest depth that keeps the plateau.

Two costs frame the tiers. A full rebuild of the reference corpus is about 1,007 s (17 minutes), of which embedding is 668 s, and `runtime.embedding_threads=8` measured 31.66 chunks/s against the runtime's default 23.65. A reranked search costs 2.34 s mean and 3.07 s maximum, so the 10 s budget has roughly four times the current cost available for accuracy.

Two facts rule out a lever that looks attractive. No paraphrase miss is withheld by a relevance gate — `withheld_total` is zero for every judged query that is not answered at rank 1 — so the cosine gate is not the paraphrase bottleneck. And three of ten paraphrase queries return the judged passage nowhere in the top ten while depth at fifty recovers class reach to 72.7%, so those failures are ordering or semantic match rather than recall.

## Tier 1 — ordering and reranking: query-time only, no rebuild

- **Make the reranked window a real setting.** The window is hard-coded as `max(top_k * 2, 10)`, which is why `rerank_max_candidates` could be raised to 150 with no effect. Give the window its own setting — a multiple of `top_k`, or an explicit floor — so that a deeper window can be measured at all, then measure it.
- **Try the stronger reranker against the deeper window.** `BAAI/bge-reranker-base` is already pinned in the registry (1.04 GB, query-time only). The budget arithmetic: about 25-30 candidates with that model is roughly 3-4 s, while 150 candidates with the current model is roughly 6.5 s. Measure which of the two trades wins on the judged set; the 10 s ceiling allows one of them, not both.
- **Guard: leave the fusion weights alone.** The hybrid lead over BM25 at rank 1 is about two queries out of 32, and `dense_weight=1.5` measured identically to the default, so a weight change is indistinguishable from noise until the judged set is larger (Tier 4).
- **Investigate the paraphrase gap, now that the gate is ruled out.** Three of ten paraphrase queries miss the judged passage entirely within ten results while nothing is withheld, and depth at fifty recovers most of that loss, so the candidates are orderable: the ranking or the semantic match is at fault, not the gates or the candidate window. This is the item Tiers 2 and 3 exist to serve.

## Tier 2 — one rebuild each (15-25 minutes with 8 threads)

- **Measure the embedding models the registry now offers.** `dense.embedding_model` selects among pinned models, and each carries its own dimension, token limit, covered languages, and prefixes, so the comparison is a re-ingestion per model rather than a code change: `BAAI/bge-base-en-v1.5` (768-d, 0.21 GB, MIT) and `mixedbread-ai/mxbai-embed-large-v1` (1024-d, 0.64 GB, Apache-2.0) for English, `jinaai/jina-embeddings-v2-base-de` (768-d, 0.32 GB, Apache-2.0) for German, and `intfloat/multilingual-e5-large` (1024-d, 2.24 GB, MIT) for one model across languages. The German model is also the first candidate for a corpus in that language, where the English default cannot help: `language.corpus` and `status` report the mismatch but only a model change fixes it.
- **Measure chunk size and overlap.** Both are settings now: `chunking.size` of 256, 384, or 512 against `chunking.overlap` of 48, 64, or 128. One rebuild per variant, no code.
- **Add contextual chunk headers.** Prepend the document title and the section or page to the text that is *embedded*, never to the text that is returned, so that returned text stays quote-clean. A chunk of 384 tokens rarely names the work it belongs to and a plain-language question rarely repeats the author's words; this is the cheapest known fix for exactly that mismatch, for the cost of one rebuild and a small change to the embedding input.
- **Calibrate the cosine gate last.** It is a setting, and the evidence says it is not the paraphrase bottleneck, so revisit it only after the embedding model and the chunk context have been measured.

## Tier 3 — code, no rebuild

- **Weight pseudo-relevance terms by corpus rarity.** `retrieval.prf` is implemented and measured (see `MEASUREMENTS.md`): it changed no ranking decision, because selection ranks by leader frequency and the stopword filter is bm25s's 33-word English list, so terms like *about*, *between* and *have* are mined. Rank candidates by inverse document frequency over the generation instead — a document-frequency table can be built once per loaded generation and cached, which keeps the cost off the query path — then re-measure before deciding whether the default moves.
- **A `--set` override does not reach the server a tool spawns.** `research-ultra-rag-verify --set retrieval.rrf_k=30 --ingest` reports `unchanged`: `--set` overrides the CLI process's own configuration, while ingest and search run in the spawned server, which resolves its own settings from the project, the environment and the defaults. An experiment driven that way silently measures the defaults. Environment variables (`RESEARCH_ULTRARAG_*`) propagate, which is what the measurements above used. Either forward the overrides into the child environment or say plainly in the settings documentation which channel each entry point honours.
- **Split the resumable ingestion loop into per-phase handlers, then enable `C901`.** Measured with `ruff check --select C901`, which is not enabled today: 23 functions exceed the default threshold of 10, across 9 modules. The loop dominates — `ResearchService._advance_ingestion` is **107**, against `search` 35, `generation_artifacts_are_valid` 23, `_front_matter_identity` 21, `ingest` 20, `_status` 18, `_reconcile_checkpoint_progress` 17, `extract_epub_spine_item` 17 — so one method is ten times the limit and five times the next worst, and the rest are a tail that the split does not touch. Enable the check only after the split, because a check that ignores its own worst offender teaches nothing.

Analysing the loop before touching it turned up three obstacles the line count hides. **Three phase blocks call nested closures** — `source_records` (`source_hashing`, `finalizing`), `staged_records` (`assembly`), `revalidation_stats_match` (`finalizing`) — and a handler cannot reach a closure defined inside the loop, so those become methods or state first. **The phases share loop-local state**: extraction, chunking, embedding and finalizing read `current`, `selected_paths`, `chunk_size`, `chunk_overlap`, `sources_by_path`, `force_recompute` and `exclusions`; `dense_indexing` reads `portable_vectors`, written by `vector_assembly`; chunking writes `units_cache`. That partly-mutated shared state is the real work — the split is the creation of an ingestion state object that handlers take and return, not a move of text. **And extracting a small phase does not lower `C901` on the loop**: each block becomes a dispatcher with a phase test and a budget test, so the two state-free phases (`vector_assembly`, `bm25_indexing`) trade one decision for two and the measured number was 107 before and after. The gain is in the large blocks, whose internal branching the loop is actually paying for.

Mechanics, verified: the handler body is the block body dedented by 8 (block sits at 12, body at 16, method body 8); a `continue` targeting the loop's own `while` becomes `return None`, while one inside a nested loop stays; `Callable` must be added to the `collections.abc` import, which does not have it today.

Order: (1) add the state object and move the three closures onto it; (2) extract the seven state-carrying phases, one commit each, verified by the suite and by re-ingesting layoul-thesis to confirm 9,237 chunks and 9,237 vectors stay reused with 0 rebuilt; (3) extract the two state-free phases; (4) enable `C901` and work the remaining tail.

## Tier 4 — the evidence base the tiers above depend on

- **Pooled relevance judgments.** The current set is known-item — one designated passage per query, judged by a single annotator — so a passage that makes the same point scores as a miss and true recall is not claimed. Collect every candidate from every mode and judge the pool.
- **Keep the measurement current and wider.** The published numbers come from one corpus and one generation; re-run `scripts/evaluate_retrieval.py` when the corpus, the extraction policy, or a retrieval default changes, and extend the set to a second corpus and to filtered queries.
- **Re-measure the `search` row of the tool-answer table** after the next reference build; `MEASUREMENTS.md` carries the pre-trim figure and says why.
- **Grow the judged set from real questions**, if the privacy of a query log can be settled: a log of questions actually asked is the only source of judgments for the questions this server is really used for.

## Tier 5 — coverage and corpus-level retrieval (larger work)

- **Run OCR before ingestion for scanned sources.** Scanned PDFs are excluded today, which is material a research corpus often needs; OCR adds extraction time and no query latency.
- **Section-aware chunking for PDFs, with parent-document retrieval.** Retrieve the chunk, return the enclosing section. This is the fix with the most headroom for plain-language questions, and the largest change on this list.
- **Decide how `list_sources` should expose the keyword vocabulary.** `reviewed_metadata_sources` echoes every saved override and is now the largest lean answer, but it is the only place an agent can discover which keywords exist. Either report keyword counts in `status` beside categories and projects, or reduce the review list to handles and leave vocabulary discovery to the filters.

## Correctness, lifecycle, and interfaces (not tiered; do when each one hurts)

- **The standalone UI launcher does not take its private server down with it.** Killing `research-ultra-rag-ui` with `SIGTERM` releases the port but leaves the private stdio server it started running with `ppid 1`, together with a defunct gateway beneath it, so every stop leaks a server, a gateway and their UltraRAG children until they are reaped by hand. Reproduce by starting the launcher, recording `pgrep -P <pid>` for its child, killing the launcher, and confirming `ps -o pid,ppid,stat -p <child>` still shows it alive. The generated per-project launcher only works around it by starting the UI in its own session and stopping the whole process group; a direct `research-ultra-rag-ui` invocation still leaks, and the launcher cannot reap a private server started by a UI it did not launch. The server-hosted UI (`--ui-port`) is unaffected.
- **Prune retained generations.** `status` lists each one with its size; removal is manual today and deletes data, so it needs a reviewed design: which generation, an explicit confirmation step, and never the current one.
- **Deliberate rollback** to a retained generation, instead of only ever moving forward.
- **Check free disk space before a build starts.**
- **An `ingest` dry run** that reports what would change, and what would be reused, without writing anything.
- **Structured activation failures** instead of one all-or-nothing message.
- **One meaning for `stale`** in the no-generation branch, where it currently doubles as "ready to build".
- **Narrow the two broad `except Exception` handlers** at durability boundaries so a real storage fault cannot be swallowed.
- **Surface the retained-generation inventory in the UI.** `status` reports `generations` and `retained_generation_bytes`; the pinned status view renders none of them.
- **Warn about a runtime-root mismatch in the standalone UI launcher,** which starts its own server and does not read an MCP client configuration, so it can silently show a different generation than the agent. The server-hosted UI (`--ui-port`) is unaffected.

- **Merge the stopword lists of a mixed corpus.** `language.corpus` can name several languages, but BM25 still filters one list, chosen by `language.bm25_stopwords`. bm25s itself accepts a list of stopwords, so the union of two lists is expressible; it needs a check that the pinned runtime passes a list through `bm25.lang` unchanged, and then a measurement showing the union beats a single list on a corpus that really is mixed.
- **Arabic needs a stopword source before it can be indexed.** bm25s ships no Arabic list, so `language.corpus = "ar"` is refused while settings are read instead of at the end of a build. Adding it means either an explicit stopword list in an option, or an empty-list setting that tells BM25 to filter nothing, plus an Arabic-capable model in the pinned embedding table.

## Verification

```bash
uv run pytest -q                        # unit and integration suite
uv run ruff check .                     # lint
uv run ruff format --check .            # formatting
uv run research-ultra-rag-verify /path/to/project --query "your question"
uv run python scripts/benchmark_write_pattern.py --root /path/on/target/disk
uv run python scripts/evaluate_retrieval.py --project /path/to/project --offline
uv run python scripts/measure_tool_payloads.py --project /path/to/project --offline
```

A tier is done when its change is in git history, its measurement is in `MEASUREMENTS.md` when it moved a number, and the suite and lint above are clean.

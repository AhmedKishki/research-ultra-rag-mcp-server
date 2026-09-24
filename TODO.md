# TODO

The open work on this server, in the order it is intended to be attempted: one tier at a time, each measured before the next starts. Everything listed here is unimplemented. Current behaviour is in `README.md`, current facts and limits are in `MEASUREMENTS.md`, capabilities are in `FEATURES.md`, and product ideas that are not in scope yet are in `ROADMAP.md`. Finished work lives in git history, not in this file.

## The measurement this plan rests on

Every tier below is judged against one baseline, taken on the reference project with the default settings, 30 of the 32 judged queries (`t12` judges a source the reviewer excluded), reranked hybrid at `top_k=10`:

| Configuration | succ@1 | succ@3 | succ@k | MRR | nDCG | doc@k |
|---|---|---|---|---|---|---|
| default (rerank cap 50, candidates 20/200) | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% |
| `rerank_max_candidates=100` | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% |
| candidates 50/600, `rerank_max_candidates=150` | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% |
| `dense_weight=1.5` | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% |
| `top_k=15` | 83.3% | 86.7% | **93.3%** | **0.861** | **0.877** | **96.7%** |

Four of those variants changed no ranking decision at all, and the reason is in the engine: the reranked window is `min(candidates, rerank_max_candidates, max(top_k * 2, 10))`, so at `top_k=10` only twenty candidates are ever reranked and the cap of fifty never binds. Raising `top_k` widened that window and recovered part of the paraphrase loss, which makes the caller's own depth request the cheapest lever measured so far.

Two costs frame the tiers. A full rebuild of the reference corpus is about 1,007 s (17 minutes), of which embedding is 668 s, and `runtime.embedding_threads=8` measured 31.66 chunks/s against the runtime's default 23.65. A reranked search costs 2.34 s mean and 3.07 s maximum, so the 10 s budget has roughly four times the current cost available for accuracy.

Two facts rule out a lever that looks attractive. No paraphrase miss is withheld by a relevance gate — `withheld_total` is zero for every judged query that is not answered at rank 1 — so the cosine gate is not the paraphrase bottleneck. And three of ten paraphrase queries return the judged passage nowhere in the top ten while depth at fifty recovers class reach to 72.7%, so those failures are ordering or semantic match rather than recall.

## Tier 0 — free accuracy: no rebuild, no added latency

- **Say what to do after a weak first search.** A search that misses is almost always one that asked too little of the engine. State the pattern in the `search` description and in `SERVER_INSTRUCTIONS`: ask for more passages (`top_k` up to 15-20), and ask the question again in different words rather than accepting a thin answer. Measured: `top_k=15` alone lifted `succ@k` from 90.0% to 93.3% and `doc@k` from 93.3% to 96.7%.
- **Decide the tool default for `top_k`.** It is 8 today, which makes the reranked window 16. Re-measure the answer size with `scripts/measure_tool_payloads.py` at 8 against 15 before changing it: the gain above may be worth the extra payload, and the lean-answer rule means the default should be the smallest depth that keeps the measured quality.
- **Publish the settings worth copying.** `README.md`'s Settings section should name the values that measurably pay — the thread count above for rebuilds, and whichever retrieval values win in Tier 1 — so a new project does not rediscover them.

## Tier 1 — ordering and reranking: query-time only, no rebuild

- **Make the reranked window a real setting.** The window is hard-coded as `max(top_k * 2, 10)`, which is why `rerank_max_candidates` could be raised to 150 with no effect. Give the window its own setting — a multiple of `top_k`, or an explicit floor — so that a deeper window can be measured at all, then measure it.
- **Try the stronger reranker against the deeper window.** `BAAI/bge-reranker-base` is already pinned in the registry (1.04 GB, query-time only). The budget arithmetic: about 25-30 candidates with that model is roughly 3-4 s, while 150 candidates with the current model is roughly 6.5 s. Measure which of the two trades wins on the judged set; the 10 s ceiling allows one of them, not both.
- **Guard: leave the fusion weights alone.** The hybrid lead over BM25 at rank 1 is about two queries out of 32, and `dense_weight=1.5` measured identically to the default, so a weight change is indistinguishable from noise until the judged set is larger (Tier 4).
- **Investigate the paraphrase gap, now that the gate is ruled out.** Three of ten paraphrase queries miss the judged passage entirely within ten results while nothing is withheld, and depth at fifty recovers most of that loss, so the candidates are orderable: the ranking or the semantic match is at fault, not the gates or the candidate window. This is the item Tiers 2 and 3 exist to serve.

## Tier 2 — one rebuild each (15-25 minutes with 8 threads)

- **Upgrade the embedding model.** The pinned `BAAI/bge-small-en-v1.5` is the smallest option in a registry of thirty. The two candidates worth measuring, with the cost of fetching them: `BAAI/bge-base-en-v1.5` (768-d, 0.21 GB, MIT) and `mixedbread-ai/mxbai-embed-large-v1` (1024-d, 0.64 GB, Apache-2.0). A change of dimension builds a new index, so this is a re-ingestion rather than a switch.
- **Measure chunk size and overlap.** Both are settings now: `chunking.size` of 256, 384, or 512 against `chunking.overlap` of 48, 64, or 128. One rebuild per variant, no code.
- **Add contextual chunk headers.** Prepend the document title and the section or page to the text that is *embedded*, never to the text that is returned, so that returned text stays quote-clean. A chunk of 384 tokens rarely names the work it belongs to and a plain-language question rarely repeats the author's words; this is the cheapest known fix for exactly that mismatch, for the cost of one rebuild and a small change to the embedding input.
- **Calibrate the cosine gate last.** It is a setting, and the evidence says it is not the paraphrase bottleneck, so revisit it only after the embedding model and the chunk context have been measured.

## Tier 3 — code, no rebuild

- **Split the ranking policy from the reuse fingerprint.** `source_set_matches` in `generation.py` returns `False` when the retrieval-policy fingerprint differs, so changing a ranking value — a weight, a gate, a cap — invalidates the reuse snapshot and forces a full re-chunk and re-embed even though chunk boundaries and vectors are unchanged. Until the ranking policy is separated from the identity of the corpus, every Tier 1 experiment that reaches a new baseline costs a 17-minute rebuild to apply.
- **Pseudo-relevance feedback on the lexical side.** Expand the BM25 query with terms taken from an initial top-ranked set. No model, no credential, roughly +0.1 s, and measured against the judged set like everything else.
- **Split the resumable ingestion loop into per-phase handlers** and enable `C901` with a documented threshold. The `ingest` method in `service.py` is roughly 1,100 lines, which makes reviewing a change to it risky; no complexity check is enabled today.

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

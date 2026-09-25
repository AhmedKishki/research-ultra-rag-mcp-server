# TODO

Open work, cheapest first. A finished item leaves this file: git history is the archive. Behaviour is in `README.md`, capabilities in `FEATURES.md`, numbers and limits in `MEASUREMENTS.md`, deferred ideas in `ROADMAP.md`.

Cost orders the file. A measurement costs one harness run; a code change costs no rebuild; an experiment that changes the corpus costs a rebuild of the reference corpus, 15–25 minutes with 8 threads.

## Minutes: a measurement, or a small change

- [ ] **Measure a shallower reranked window.** Test. 20, 30, 40–77 and 50 rank the judged passage identically while 50 costs 2.5× more, so nothing says the default window is the cheapest one that holds quality. `retrieval.rerank_window_multiple` and `retrieval.rerank_window_floor` make it expressible.
- [ ] **Calibrate the cosine gate.** Test. It is a setting, and the evidence says it is not the paraphrase bottleneck, so measure it after the semantic levers.
- [ ] **Surface the retained-generation inventory in the UI.** Feature. `status` reports `generations` and `retained_generation_bytes`; the pinned status view shows none of them.
- [ ] **Warn about a runtime-root mismatch in the standalone UI launcher.** Feature. It starts its own server without reading an MCP client configuration, so it can show a different generation than an agent. `--ui-port` is unaffected.
- [ ] **Give `stale` one meaning.** Feature. In the no-generation branch it doubles as "ready to build".
- [ ] **Narrow the two broad `except Exception` handlers** at durability boundaries, so a storage fault cannot be swallowed. Feature.
- [ ] **Report activation failures as structured values** instead of one all-or-nothing message. Feature.
- [ ] **Check free disk space before a build starts.** Feature.
- [ ] **Decide how `list_sources` exposes the keyword vocabulary.** Feature. `reviewed_metadata_sources` is the largest lean answer and the only place an agent can discover which keywords exist; either report counts in `status` or reduce the list to handles.

## Hours: code, still no rebuild

- [ ] **Stop the standalone UI launcher leaking its private server.** Feature. `SIGTERM` to `research-ultra-rag-ui` frees the port and leaves the private stdio server running under `ppid 1`, with a defunct gateway below it. The generated launcher stops the process group instead; a direct invocation does not, and no launcher can reap a server started by a UI it did not launch.
- [ ] **Make `--set` reach the surfaces that spawn a server.** Feature. `research-ultra-rag` resolves settings in process, so `--set` works there. `research-ultra-rag-verify` and the UI spawn a child that resolves its own, so `--set retrieval.rrf_k=30 --ingest` measures the defaults instead. Forward the overrides into the child environment, or document which channel each entry point honours.
- [ ] **Prune retained generations.** Feature. Removal is manual and deletes data, so it needs which generation, a confirmation step, and never the current one.
- [ ] **Roll back to a retained generation deliberately**, instead of only moving forward. Feature.
- [ ] **An `ingest` dry run** that reports what would change and what would be reused, and writes nothing. Feature.
- [ ] **Weight pseudo-relevance terms by corpus rarity.** Feature. `retrieval.prf` is implemented and changes no ranking decision, because selection ranks by leader frequency and the stopword filter is bm25s's 33-word English list. Rank by inverse document frequency over a per-generation document-frequency table, cached, then measure.
- [ ] **Merge the stopword lists of a mixed corpus.** Feature. `language.corpus` can name several languages while BM25 filters the one list in `language.bm25_stopwords`. bm25s accepts a list, so the union is expressible: check that the pinned runtime passes a list through `bm25.lang`, then measure the union against a single list on a mixed corpus.
- [ ] **An Arabic stopword source.** Feature. bm25s ships no Arabic list, so `language.corpus = "ar"` is refused while settings are read. Add an explicit list option, or an empty list that filters nothing, plus an Arabic-capable model in the pinned table.
- [ ] **Split the resumable ingestion loop into per-phase handlers, then enable `C901`.** Feature. `_advance_ingestion` is 1,320 lines at complexity 107, against 35 for the next worst function in the package; three of its phase blocks call closures defined inside it and eight read loop-local state, so the split is an ingestion state object that handlers take and return. Accept on a green suite and a re-ingest that reuses every chunk and vector. The map and the transformation rules are in git history.
- [ ] **A CPU reserve, so a build leaves cores free.** Feature. `runtime.embedding_threads` is a thread count, not a promise about the machine, and cutting threads costs build throughput — 8 threads measured 31.66 chunks/s against the default's 23.65 — where `runtime.nice` costs none. A reserve expresses the intent directly, as physical cores minus the reserve, and needs a measurement to price it.

## One rebuild each: 15–25 minutes per experiment

These are the levers for the paraphrase gap: three of ten judged paraphrase queries miss the judged passage entirely within the top ten, while nothing is withheld.

- [ ] **Add contextual chunk headers.** Feature. Prepend the title and the section to the text that is embedded, never to the text that is returned, so returned text stays quote-clean.
- [ ] **Measure chunk size and overlap.** Test, one rebuild per variant. `chunking.size` of 256, 384, or 512 against `chunking.overlap` of 48, 64, or 128.
- [ ] **Measure the embedding models the registry offers.** Test, one rebuild per model: `BAAI/bge-base-en-v1.5` and `mixedbread-ai/mxbai-embed-large-v1` for English, `jinaai/jina-embeddings-v2-base-de` for German, `intfloat/multilingual-e5-large` across languages. The German model is the first candidate for a corpus in that language, where the English default cannot help.
- [ ] **Section-aware chunking with parent-document retrieval.** Feature, the largest change here. Retrieve the chunk and return the enclosing section; it needs section structure preserved through extraction and a second level in the generation.

## Days: work that no rebuild bounds

- [ ] **Pooled relevance judgments.** Test. The set is known-item — one designated passage per query, one annotator — so a passage making the same point scores as a miss. Collect every candidate from every mode and judge the pool.
- [ ] **Grow the judged set from real questions**, if the privacy of a query log can be settled. Test.
- [ ] **Re-measure the `search` row of the tool-answer table** after the next reference build. Test; `MEASUREMENTS.md` carries the pre-trim figure and says why.
- [ ] **Keep the measurement current and wider.** Test. Re-run `scripts/evaluate_retrieval.py` when the corpus, the extraction policy, or a retrieval default changes, and add a second corpus and filtered queries.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run research-ultra-rag --project-root /path/to/project status
uv run research-ultra-rag-verify /path/to/project --query "your question"
uv run python scripts/benchmark_write_pattern.py --root /path/on/target/disk
uv run python scripts/evaluate_retrieval.py --project /path/to/project --offline
uv run python scripts/measure_tool_payloads.py --project /path/to/project --offline
```

An item is done when the change is committed, its measurement is in `MEASUREMENTS.md` if it moved a number, and the commands above are clean.


# TODO

The work that is not done: **features to add**, then **measurements and tests to run**. Each item says what it costs, because that is what decides the order they can be afforded in — a query-time change needs no rebuild, a model or chunking experiment needs one rebuild of the reference corpus, and a code change needs neither.

A finished item leaves this file; git history is the archive. Current behaviour is in `README.md`, capabilities and their provenance in `FEATURES.md`, numbers and limits in `MEASUREMENTS.md`, and deferred product ideas in `ROADMAP.md`.

## Features to add

### No rebuild

- **Close the paraphrase gap.** Three of ten judged paraphrase queries return the judged passage nowhere in the top ten while nothing is withheld, and depth at fifty recovers most of that loss, so the candidates are orderable: the fault is the ordering or the semantic match, not a gate. The levers are the semantic ones below — the embedding model and contextual chunk headers — measured against a judged set wide enough to separate them.
- **Surface the retained-generation inventory in the UI.** `status` reports `generations` and `retained_generation_bytes`; the pinned status view renders none of them, so a person working in the browser cannot see what retained generations occupy.
- **Warn about a runtime-root mismatch in the standalone UI launcher.** It starts its own server and does not read an MCP client configuration, so it can silently show a different generation than an agent in the same project sees. The server-hosted UI (`--ui-port`) is unaffected.

### One rebuild

- **Add contextual chunk headers.** Prepend the document title and the section or page to the text that is *embedded*, never to the text that is returned, so returned text stays quote-clean. A chunk of 384 tokens rarely names the work it belongs to, and a plain-language question rarely repeats the author's words; this is the cheapest known fix for exactly that mismatch, for one rebuild and a small change to the embedding input.
- **Section-aware chunking for PDFs, with parent-document retrieval.** Retrieve the chunk, return the enclosing section. This is the fix with the most headroom for plain-language questions — it targets the paraphrase gap directly — and the largest change here, since it needs the section structure preserved through extraction and a second level in the generation, so treat it as code work that happens to require a rebuild to measure.

### Code changes

- **Split the resumable ingestion loop into per-phase handlers, then enable `C901`.** `ResearchService._advance_ingestion` is one 1,320-line method at cyclomatic complexity 107, against 35 for the next worst function in the package, and three of its phase blocks call closures defined inside it while eight read loop-local state — so the split is the creation of an ingestion state object that handlers take and return, not a move of text. Accept when the suite is green and re-ingesting the reference corpus still reports every chunk and vector reused with none rebuilt. The map, the transformation rules, and why a shallow extraction changes nothing are in git history.
- **Weight pseudo-relevance terms by corpus rarity.** `retrieval.prf` is implemented and measured: it changed no ranking decision, because selection ranks by leader frequency and the stopword filter is bm25s's 33-word English list, so terms like *about*, *between* and *have* are mined. Rank candidates by inverse document frequency over the generation instead — a document-frequency table can be built once per loaded generation and cached, which keeps the cost off the query path — then measure before deciding whether the default moves.
- **Make `--set` reach the two surfaces that spawn a server.** `research-ultra-rag` resolves settings in its own process, so `--set` reaches every operation there, including `ingest` and `search`. `research-ultra-rag-verify` and the browser UI cannot, because they spawn a stdio server that resolves its own settings from the project, the environment and the defaults: `research-ultra-rag-verify --set retrieval.rrf_k=30 --ingest` reports `unchanged` and an experiment driven that way silently measures the defaults. Either forward the parent's overrides into the child environment or say in the settings documentation which channel each entry point honours.
- **Stop the standalone UI launcher from leaking its private server.** Killing `research-ultra-rag-ui` with `SIGTERM` releases the port but leaves the private stdio server it started running with `ppid 1`, together with a defunct gateway beneath it, so every stop leaks a server, a gateway and their UltraRAG children until they are reaped by hand. The generated per-project launcher works around it by starting the UI in its own session and stopping the whole process group, which a direct invocation still does not, and it cannot reap a private server started by a UI it did not launch. The server-hosted UI (`--ui-port`) is unaffected.
- **Prune retained generations.** `status` lists each one with its size; removal is manual today and deletes data, so it needs a reviewed design: which generation, an explicit confirmation step, and never the current one.
- **Deliberate rollback** to a retained generation, instead of only ever moving forward.
- **Check free disk space before a build starts.**
- **An `ingest` dry run** that reports what would change, and what would be reused, without writing anything.
- **Structured activation failures** instead of one all-or-nothing message.
- **One meaning for `stale`** in the no-generation branch, where it currently doubles as "ready to build".
- **Narrow the two broad `except Exception` handlers** at durability boundaries so a real storage fault cannot be swallowed.

### Language and coverage

- **Merge the stopword lists of a mixed corpus.** `language.corpus` can name several languages, but BM25 still filters one list, chosen by `language.bm25_stopwords`. bm25s itself accepts a list of stopwords, so the union of two lists is expressible; it needs a check that the pinned runtime passes a list through `bm25.lang` unchanged, and then a measurement showing the union beats a single list on a corpus that really is mixed.
- **Arabic needs a stopword source before it can be indexed.** bm25s ships no Arabic list, so `language.corpus = "ar"` is refused while settings are read instead of at the end of a build. Adding it means either an explicit stopword list in an option, or an empty-list setting that tells BM25 to filter nothing, plus an Arabic-capable model in the pinned embedding table.
- **Decide how `list_sources` should expose the keyword vocabulary.** `reviewed_metadata_sources` echoes every saved override and is the largest lean answer, but it is the only place an agent can discover which keywords exist. Either report keyword counts in `status` beside categories and projects, or reduce the review list to handles and leave vocabulary discovery to the filters.

## Measurements and tests to run

- **Measure a narrower reranked window.** No rebuild. The shipped default reranks twenty candidates at `top_k=10`, and 20, 30, 40–77 and 50 all rank the judged passage identically while the wider ones cost 2.5× more, so a *shallower* window is the only direction left that could cut the 1.8 s rerank without giving up quality. `retrieval.rerank_window_multiple` and `retrieval.rerank_window_floor` are what make it expressible.
- **Measure the embedding models the registry offers.** One rebuild per model. `dense.embedding_model` selects among pinned models, and each carries its own dimension, token limit, covered languages, and prefixes: `BAAI/bge-base-en-v1.5` (768-d, 0.21 GB, MIT) and `mixedbread-ai/mxbai-embed-large-v1` (1024-d, 0.64 GB, Apache-2.0) for English, `jinaai/jina-embeddings-v2-base-de` (768-d, 0.32 GB, Apache-2.0) for German, and `intfloat/multilingual-e5-large` (1024-d, 2.24 GB, MIT) for one model across languages. The German model is also the first candidate for a corpus in that language, where the English default cannot help: `language.corpus` and `status` report the mismatch but only a model change fixes it.
- **Measure chunk size and overlap.** One rebuild per variant. Both are settings now: `chunking.size` of 256, 384, or 512 against `chunking.overlap` of 48, 64, or 128.
- **Calibrate the cosine gate last.** No rebuild. It is a setting, and the evidence says it is not the paraphrase bottleneck, so revisit it only after the embedding model and the chunk context have been measured.
- **Pooled relevance judgments.** Annotation work. The current set is known-item — one designated passage per query, judged by a single annotator — so a passage that makes the same point scores as a miss and true recall is not claimed. Collect every candidate from every mode and judge the pool.
- **Grow the judged set from real questions,** if the privacy of a query log can be settled: a log of questions actually asked is the only source of judgments for the questions this server is really used for.
- **Re-measure the `search` row of the tool-answer table** after the next reference build; `MEASUREMENTS.md` carries the pre-trim figure and says why.
- **Keep the measurement current and wider.** The published numbers come from one corpus and one generation; re-run `scripts/evaluate_retrieval.py` when the corpus, the extraction policy, or a retrieval default changes, and extend the set to a second corpus and to filtered queries.

## Verification

```bash
uv run pytest -q                        # unit and integration suite
uv run ruff check .                     # lint
uv run ruff format --check .            # formatting
uv run research-ultra-rag --project-root /path/to/project status
uv run research-ultra-rag-verify /path/to/project --query "your question"
uv run python scripts/benchmark_write_pattern.py --root /path/on/target/disk
uv run python scripts/evaluate_retrieval.py --project /path/to/project --offline
uv run python scripts/measure_tool_payloads.py --project /path/to/project --offline
```

An item is done when its change is in git history, its measurement is in `MEASUREMENTS.md` when it moved a number, and the suite and lint above are clean.


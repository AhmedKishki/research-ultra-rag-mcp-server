# Reference retrieval evaluation

This directory holds the judged query set behind the retrieval-quality numbers in `MEASUREMENTS.md`, and the harness that produces them is `scripts/evaluate_retrieval.py`.

The set is deliberately small, honest, and reproducible: 32 queries over 19 passages of one real research corpus, judged by inspection of the extracted text, measured through the public MCP `search` tool. It is a starting point that can answer "is hybrid better than BM25 here?" — not a benchmark suite.

## The judged set

`ai-and-fetishism-queries.json` (schema version 1) contains two parts.

- `targets` — one entry per judged passage: the source path, the locator, a verbatim `snippet`, the `chunk_id_at_measurement`, and a note saying what the passage says. The snippet is an identity key, not a quote: it exists so the target can be re-resolved after re-ingestion, when chunk IDs and chunk boundaries legitimately change. The source path is matched against the generation manifest's documents, and `document_id` is a fallback, so a target also survives a source being renamed.
- `queries` — 32 queries, each judged relevant to exactly one target, in three classes:
  - `quote` (11): a remembered phrasing of the passage, the way a researcher recalls words they read.
  - `paraphrase` (11): the same 11 targets, asked as a question that avoids the author's vocabulary.
  - `entity` (10): a named person, project, place, or concept, asked about specifically.

The `quote` and `paraphrase` classes deliberately share targets, so the two query styles are compared on identical passages. The harness reports the mean content-word overlap between each query and its target passage, which makes it checkable how lexical each class actually is rather than a claim.

## Protocol and its limits

Judgments are known-item: one designated relevant chunk per query. That measures **findability of a passage**, and it under-credits a mode that returns a different passage making the same point. The harness therefore reports `doc@k` (target document retrieved) next to chunk-level success, so a chunk-level miss inside the right document is not confused with not finding the document at all.

The judgments are single-annotator and were written from the extracted text of the measured generation. There is no inter-annotator agreement, no pooled recall, and no judgment of the passages the systems returned that were not the target. Pooled judgments — every candidate from every mode judged for relevance, which is what proper recall measurement needs — are open work in `TODO.md`.

Two further limits are worth stating. Relevance gates can legitimately return fewer than `top_k`, so a miss can mean "rejected by a gate" rather than "ranked low"; the harness records rejected and withheld counts per run for that reason. And the numbers describe the generation the report names, so re-run the harness when the corpus, the extraction policy, or a retrieval default changes.

## Running it

```bash
uv run python scripts/evaluate_retrieval.py --project /path/to/project
uv run python scripts/evaluate_retrieval.py --project /path/to/project --validate-only
uv run python scripts/evaluate_retrieval.py --project /path/to/project --limit 5 --modes bm25,hybrid
```

Defaults: modes `bm25,dense,hybrid,hybrid+rerank` at `top_k=10`, plus a deep pass for `hybrid` at `top_k=50`, all 32 queries. `--offline` works when the runtime and both model caches are already present. Every mode passes `rerank` explicitly, so these numbers do not depend on the tool default; reranking is the default, so the `hybrid+rerank` row is what an ordinary search returns.

The harness never writes inside the project. It calls `status` and `search` over stdio, and reads the generation's canonical `chunks.jsonl` and `manifest.json` **read-only** to resolve judged targets; that read is measurement-only and is not part of the retrieval path. Searches pass `include_staleness=false`, so no result in the report depends on a freshness verdict.

## Output

The console prints two aligned tables (primary depth and deep pass) with, per mode and per class: `succ@1`, `succ@3`, `succ@k`, `MRR`, `nDCG@k`, `doc@k`, mean query-to-target overlap, and mean returned passages.

A full JSON report is written beside the judged set as `ai-and-fetishism-queries-report.json` with every per-query run, the resolved targets, the retrieval configuration recorded in the generation, and the timing. Reports are generated artifacts and are ignored by git; regenerate one instead of editing it.

## Extending the set

Add targets by copying an exact snippet out of the corpus text and confirming that `--validate-only` resolves it to exactly one chunk. Ambiguity is refused rather than guessed, so a snippet that spans a chunk overlap, or that appears twice in a document, must be extended or replaced before it can be judged. Adding queries only requires a `query_id`, a `class`, a `target_id`, and the query text.

Authorization over the corpus is unchanged by this directory: the original PDF and EPUB files remain the quote authority, and the snippets here are short identity keys used to resolve a judged passage.

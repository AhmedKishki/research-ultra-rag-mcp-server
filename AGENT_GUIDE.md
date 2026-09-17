# Agent guide

This file is for AI agents using `research-ultra-rag-mcp` as a research tool.

## Purpose

Use this server to find citable evidence across one project's original PDF and
EPUB collection. The server retrieves passages; you remain responsible for
comparison, synthesis, uncertainty, and the final written answer.

## Normal workflow

1. Call `status` before substantive research.
2. If no generation exists, tell the user that `ingest` creates persistent data,
   computes embeddings, and may download a model; obtain their agreement before
   calling it.
3. If `stale=true`, explain the reported changes. The old generation remains
   searchable; ask whether the user wants a new ingestion.
4. Call `search` with the actual research question. Leave
   `retrieval_method="hybrid"` for ordinary use.
5. Inspect several hits rather than treating rank 1 as automatically correct.
   Use `get_passage` when a quotation needs surrounding context.
6. Quote only the returned `text`. Cite the returned `citation`, `source_path`,
   and `locator`, and advise checking important quotations in the original file.
7. If multiple files appear to represent the same source, do not count them as
   independent support. Compare their metadata and passages, explain the issue,
   and use `set_source_inclusion` when the user asks to retain one copy and
   exclude another.

## Choosing retrieval

- `hybrid` (default): use for normal research and broad evidence discovery. It
  combines UltraRAG BM25 and semantic Qdrant rankings with reciprocal-rank
  fusion.
- `bm25`: use for exact terms, names, distinctive phrases, or quotation lookup.
- `dense`: use to inspect conceptual matches that may use different wording.
- `rerank=true`: use when the result set needs more precision or initial hybrid
  ordering is weak. It is slower on CPU and loads a second model on first use.

Use category, keyword, or document filters only when the user requests a scoped
search. Every requested category and keyword must be present. A result only
needs to match one of the supplied document IDs.

## Reading a hit

- `text`: extracted source passage; the only field that may be quoted.
- `citation`, `source_path`, `locator`: provenance to cite and verify.
- `chunk_id`: stable handle for `get_passage`.
- `component_ranks.bm25` / `.dense`: position in each independent rank list;
  `null` means that component did not retrieve the chunk in its candidate set.
- `component_scores.dense_cosine_similarity`: dense ranking signal.
- `fusion_score`: RRF ranking signal for hybrid results.
- `rerank_score`: cross-encoder ranking signal when reranking was requested.

None of the scores is a probability, truth assessment, or citation. Do not
compare scores across different queries. The server cannot provide a raw BM25
score, so it reports the BM25 rank and returns its score as `null`.

## Source and generation rules

- Only regular `.pdf` and `.epub` files beneath the configured `sources/`
  directory are indexed. Markdown is deliberately ignored.
- Each successful ingestion creates a complete immutable BM25 + Qdrant
  generation. The prior generation remains available on disk.
- Use a `chunk_id` only with the generation that returned it. IDs are
  deterministic for unchanged content and chunking settings, but can change
  after source, chunking, or incompatible server-version changes.
- A status response with `hybrid_upgrade_required=true` refers to an older
  BM25-only generation. Search it with `retrieval_method="bm25"`, or ask the
  user before ingesting a new hybrid generation.
- Use `set_source_metadata` only for user-reviewed title, author, year, DOI,
  category, and keyword values. A new ingestion is required to apply changes.
- The server does not automatically identify duplicates. Use
  `set_source_inclusion(source_path, included=false, reason=...)` only for an
  agent/user-reviewed decision. It immediately removes that source from search,
  source listings, and passage lookup; it never deletes or edits the PDF/EPUB.
  Run ingestion later to rebuild the indexes without it.
- Restore a source with `included=true`. Restoration is immediate if its chunks
  remain in the current generation; otherwise a new ingestion is required.
- Never invent a page, section, author, date, DOI, quotation, or relevance score.

## Recommended response pattern

For each material claim, identify the supporting passage and cite its returned
locator. Distinguish direct quotation from paraphrase. When sources disagree,
show the disagreement rather than merging them into a false consensus. If the
retrieval results do not support the requested claim, say so and suggest a
broader query or another retrieval method.

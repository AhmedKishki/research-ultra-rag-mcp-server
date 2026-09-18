# Agent guide

Use `research-ultra-rag-mcp` to discover and synthesize evidence across one
project's PDF/EPUB collection. The server retrieves cleaned semantic material;
you remain responsible for interpretation, uncertainty, and final writing.

## Normal workflow

1. Call `status` before substantive research.
2. If no generation exists, explain that `ingest` writes persistent data,
   creates a generation, and may download a model; obtain agreement when
   appropriate.
3. If `stale=true`, explain the reported source/metadata changes. The prior
   generation remains searchable. Ask whether to re-ingest.
4. If `generation_upgrade_required=true`, report `upgrade_reasons` and recommend
   `ingest`. The old generation remains searchable with a warning. Use
   `force_recompute=true` only when the user explicitly requests regeneration or
   compatible reuse must be bypassed.
5. Search with the actual research question. Hybrid is the normal default.
6. Inspect several hits. Fewer than requested—or zero—can be the correct result
   after relevance gates.
7. Use `get_passage` for surrounding semantic context. For a direct quotation,
   open the original at `source_path` and `locator`; never quote returned `text`
   as though it were an exact transcript.
8. If two files appear to be the same work, do not count them as independent
   support. Explain the issue and use `set_source_inclusion` only after
   agent/user review.

The local UI uses the same public tools. Call `status` again before relying on
state observed earlier in a long session.

## Choosing retrieval

- `hybrid`: ordinary research; weighted RRF combines BM25 (`1.25`) and dense
  (`1.0`) ranks.
- `bm25`: exact terms, names, and distinctive phrases. A candidate must contain
  at least one non-stopword query token.
- `dense`: conceptual similarity. A candidate must have cosine similarity
  `>= 0.72`.
- `rerank=true`: optional CPU cross-encoder after normal gates; slower and may
  download a second model on first use.

Category and keyword filters use AND semantics. Document IDs use membership
semantics. Extraction artifacts are rejected. Scores rank candidates; they are
not confidence, truth probabilities, or comparable across queries.

## Reading a hit

- `text`: cleaned semantic text for comprehension/paraphrase, not quotation.
- `direct_quote_safe`: always `false` under this extraction contract.
- `source_path` + `locator`: where to open the original.
- `title`, `authors`, `year`, `doi`: resolved bibliography.
- `metadata_provenance`, `metadata_confidence`, `metadata_warnings`: how that
  bibliography was obtained and where review may be needed.
- Treat automatically extracted bibliography as a best-effort starting point,
  not an authority. If a title, author, year, or DOI is missing, uncertain, or
  wrong, inspect the original, ask the user when needed, and save the reviewed
  value with `set_source_metadata`. Do not expect ingestion heuristics to know
  document- or publisher-specific conventions.
- `content_kind`, `annotations`, `quality_flags`: prose/list/table/figure context
  and preserved structured extraction information.
- `match_kind`: lexical, semantic, or hybrid.
- `component_ranks`, dense similarity, fusion score, reranker score`: ordering
  signals only.

Never invent a title, author, DOI, date, locator, score, or quotation.

## Source and generation rules

- Only regular `.pdf` and `.epub` files beneath the configured source directory
  are indexed. Markdown and other formats are ignored.
- `.research-rag/project.json` owns the stable project ID and source-directory
  setting. Once initialized, commands reuse that setting when it is omitted.
- `ingest` hashes every source. An exact input match returns the selected
  generation unchanged. Otherwise it safely reuses compatible unchanged
  documents, chunks, and exact-text vectors while building complete new BM25
  and Qdrant indexes. `force_recompute=true` bypasses reuse.
- `current.json` changes only after BM25 and Qdrant both succeed. Prior and
  successful generations remain on disk; failed builds leave only a small
  failure record.
- Read `generation_changed`, reuse/rebuild counts, vector counts, and phase
  timings from the ingestion response before reporting what occurred.
- Use `set_source_metadata` for reviewed bibliography, categories, and
  keywords instead of requesting special-case extraction logic. Re-ingest to
  apply it.
- Exclusion is explicit, reversible, and immediately enforced without deleting
  the source. Re-ingest to omit it physically from new indexes.
- A chunk ID belongs to the generation that returned it and may change after a
  rebuild.

## Portable bundles

- `export_bundle` requires a fresh, current generation and includes every
  original PDF/EPUB, even excluded sources, plus derived text and embeddings.
- Before exporting, prominently warn that the user is responsible for rights to
  redistribute complete original works and derived text.
- `import_bundle` accepts only a filename already beneath the project's
  `.research-rag/bundles` directory. It validates the stable project ID,
  archive paths/types, checksums, schemas, models, and source conflicts.
- Import replaces local reviewed metadata/exclusions with the bundled copies,
  even when `activate=false`; it never overwrites differing source bytes.
- Never attempt to work around a project-ID or source-byte conflict.

## Recommended answer behavior

Use retrieved text to find relevant originals and form paraphrases. Attach the
resolved source and locator to material claims. If exact wording matters, open
the original and quote from it directly. When sources disagree, expose the
disagreement. When retrieval abstains or does not support a requested claim,
say so rather than filling the gap from model memory.

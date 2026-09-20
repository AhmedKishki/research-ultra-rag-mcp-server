# Agent guide

Use `research-ultra-rag-mcp` to discover and synthesize evidence across one project's PDF/EPUB collection. The server retrieves cleaned semantic material; you remain responsible for interpretation, uncertainty, and final writing.

## Normal workflow

1. Call `status` before substantive research.
2. Call `list_sources` when you need source handles. Even before ingestion, `discovered_sources` lists live PDFs/EPUBs and their stable `source_id` values and idempotently registers them in portable project state. `known_sources` retains lean identity records for registered paths even when originals disappear, while `reviewed_metadata_sources` lists every saved override by the same stable ID. Prefer those IDs for metadata and inclusion decisions.
3. If no generation exists, explain that `ingest` writes persistent data, creates a generation, and may download a model; obtain agreement when appropriate.
4. If `stale=true`, explain the reported source or inclusion changes. The prior generation remains searchable. Ask whether to re-ingest. A true `metadata_overlay_active` is not staleness: reviewed metadata is already effective without rebuilding the generation. Metadata listed under `metadata_pending_source_paths` belongs to sources outside the selected generation and becomes effective after those sources are included and ingested.
5. If `generation_upgrade_required=true`, report `upgrade_reasons` and recommend `ingest`. The old generation remains searchable with a warning. Use `force_recompute=true` only when the user explicitly requests regeneration or compatible reuse must be bypassed.
6. When `ingest` returns `status="in_progress"`, repeat it with identical chunk settings and force mode until it returns `ready` or `unchanged`. Report the build ID, phase, and completed/total progress when useful. A selected prior generation remains searchable during the staged build.
7. Search with the actual research question. Hybrid is the normal default.
8. Inspect several hits. Fewer than requested—or zero—can be the correct result after relevance gates.
9. Use `get_passage` for surrounding semantic context. For a direct quotation, open the original at `source_path` and `locator`; never quote returned `text` as though it were an exact transcript.
10. If two files appear to be the same work, do not count them as independent support. Explain the issue and use `set_source_inclusion` only after agent/user review.

The local UI uses the same public tools. Call `status` again before relying on state observed earlier in a long session.

## Choosing retrieval

- `hybrid`: ordinary research; weighted RRF combines BM25 (`1.25`) and dense (`1.0`) ranks.
- `bm25`: exact terms, names, and distinctive phrases. A candidate must contain at least one non-stopword query token.
- `dense`: conceptual similarity. A candidate must have cosine similarity `>= 0.72`.
- `rerank=true`: optional CPU cross-encoder after normal gates; slower and may download a second model on first use.
- `result_view="passages"`: preserve the global passage ranking.
- `result_view="references"`: use the same ranked candidate pool but cap each `source_id` at `passages_per_reference` (default `2`) within the unchanged total `top_k` passage budget. Use this when several useful references matter more than several passages from one reference.
- `include_staleness=true` (default) makes each search re-compare the source directory with the generation to report `stale`. That comparison grows with the number of sources, so pass `include_staleness=false` for follow-up searches in a session where you have already called `status` and only need evidence. The response then reports `stale=null` and `staleness_checked=false`, which means the freshness was not checked, not that the generation is current. Call `status` before telling a user whether their corpus is up to date.

Category and keyword filters use AND semantics. Document IDs use membership semantics. Extraction artifacts and nonempty chunks containing no Unicode alphanumeric content are rejected. Text, numbers, and formulas containing at least one letter or digit are not symbol-only. A candidate is withheld only for corruption evidence: replacement characters, private-use or unassigned code points, or a known damaged encoding sequence. Script mixing never withholds a passage; it appears in per-hit `text_notes`, and `withheld_candidates` reports the reason codes and example chunk IDs for anything that was withheld. Scores rank candidates; they are not confidence, truth probabilities, or comparable across queries.

## Reading a hit

- `text`: cleaned semantic text for comprehension/paraphrase, not quotation.
- `text_notes`: advisory script notes such as `non_latin_dominant` or `mixed_script_text`. They never mean the passage was unusable; they warn that it mixes scripts, so read its locator before treating it as English prose.
- `direct_quote_safe`: always `false` under this extraction contract.
- `source_id`: the preferred project-scoped handle for metadata and inclusion edits. It survives content changes but changes when the source is renamed or moved.
- `document_id`: the content/path-version identity used by the selected generation. Do not treat it as the durable edit handle.
- `source_path` + `locator`: where to open the original.
- `distinct_reference_count`: how many separate indexed references contributed returned passages. In reference view, inspect `reference_groups` for each reference's best admitted passages.
- `relevance_limited` / `grouping_limited`: distinguish a candidate-pool shortfall from a reference cap that prevents filling the passage budget.
- `title`, `authors`, `year`, `doi`: resolved bibliography.
- `metadata_provenance` and `metadata_warnings`: which concrete source supplied each field and which missing, conflicting, fallback, or corrupt values need review.
- Treat automatically extracted bibliography as a best-effort starting point, not an authority. If a title, author, year, or DOI is missing, uncertain, or wrong, inspect the original, ask the user when needed, and save the reviewed value with `set_source_metadata`. Supply exactly one selector; prefer `source_id`, with `source_path` retained for compatibility. For an indexed source, the complete override immediately updates this metadata, citations, filters, source listings, and neighboring passages. Check `effective_immediately` and `requires_ingest` in the response; ingestion is needed only if the source is absent from the selected generation. The metadata object replaces the complete override: `{}` restores all automatic values, while an explicit empty value clears that automatic field. Do not expect ingestion heuristics to know document- or publisher-specific conventions.
- `content_kind`, `annotations`, `quality_flags`: prose/list/table/figure context and preserved structured extraction information.
- `match_kind`: lexical, semantic, or hybrid.
- `component_ranks`, dense similarity, fusion score, reranker score`: ordering signals only.

Never invent a title, author, DOI, date, locator, score, or quotation.

## Source and generation rules

- Only regular `.pdf` and `.epub` files beneath the configured source directory are indexed. Markdown and other formats are ignored.
- `.research-rag/project.json` owns the stable project ID and source-directory setting. Once initialized, commands reuse that setting when it is omitted.
- A `source_id` combines that project identity with the source-relative path, so replacing the file's bytes preserves it and renaming/moving the file changes it. A `document_id` identifies one path/content version and can change between generations.
- `ingest` hashes every source. An exact input match returns the selected generation unchanged. Otherwise it safely reuses compatible unchanged documents, chunks, and exact-text vectors while building complete new BM25 and dense indexes. Work is durably checkpointed between bounded units; `status.ingestion_progress` reports an unfinished build. `force_recompute=true` bypasses reuse but may resume its own matching checkpoint.
- `current.json` changes only after BM25 and the dense index both succeed. Prior and successful generations remain on disk. Cancellation and timeout retain a resumable checkpoint; incompatible inputs supersede it with a small diagnostic, and non-resumable failures leave only a small failure record.
- Corrupt extraction units are omitted whole, with locator/reason diagnostics but without retained garbage text. Retrieval also withholds corrupt chunks from older generations and discloses the reason codes and example chunk IDs in `withheld_candidates`. Never reconstruct, repair, or invent rejected wording.
- Script mixing and non-Latin dominance are advisory `text_notes`, never withhold reasons. A foreign-language quotation inside an English source is evidence; cite its locator instead of discarding it.
- Formula-font letters (`𝑀` → `M`) and the `ﬁ`, `ﬂ`, `ﬀ` ligatures are folded to their plain spellings during cleaning, so a typed query matches the printed text. Accented letters, superscripts, and subscripts are unchanged.
- Ingestion records each chunk's embedding token count. `dense_truncated: true` means the text exceeds the embedding model's limit, so the dense vector covers only a prefix while BM25 still matches the whole text; treat such a hit as lexically reliable and semantically partial, and read its locator. A null count means the generation predates the audit; re-ingest to populate it.
- Symbol-only chunks are omitted during ingestion and guarded at retrieval for older generations. A chunk with at least one Unicode letter or digit is not symbol-only, including an alphanumeric formula or numeric content; other extraction-artifact checks still apply.
- Read `generation_changed`, reuse/rebuild counts, vector counts, and phase timings from the ingestion response before reporting what occurred.
- Use `set_source_metadata` for reviewed bibliography, categories, and keywords instead of requesting special-case extraction logic. It is a replace-all override: omitted fields remove prior reviewed values. An indexed source updates immediately without changing its chunk IDs or immutable index files. If an old generation cannot recover automatic bibliography hidden by a removed override, it returns a safe fallback and warning until a later ingestion can extract the automatic value again.
- Exclusion is explicit, reversible, and immediately enforced without deleting the source. Call `set_source_inclusion` with exactly one selector, preferably `source_id`; re-ingest to omit it physically from new indexes.
- A chunk ID belongs to the generation that returned it and may change after a rebuild.

## Portable bundles

- `export_bundle` requires a fresh, current generation and includes every original PDF/EPUB, even excluded sources, plus derived text and embeddings.
- Before exporting, prominently warn that the user is responsible for rights to redistribute complete original works and derived text.
- `import_bundle` accepts only a filename already beneath the project's `.research-rag/bundles` directory. It validates the stable project ID, archive paths/types, checksums, schemas, models, and source conflicts.
- Import replaces local reviewed metadata/exclusions with the bundled copies, even when `activate=false`; compatible metadata immediately overlays the still-selected generation, while imported exclusions immediately govern its retrieval surfaces. Both also govern future ingestion. The response discloses the unchanged pointer and live portable-state replacement. Import never overwrites differing source bytes.
- Never attempt to work around a project-ID or source-byte conflict.

## Recommended answer behavior

Use retrieved text to find relevant originals and form paraphrases. Attach the resolved source and locator to material claims. If exact wording matters, open the original and quote from it directly. When sources disagree, expose the disagreement. When retrieval abstains or does not support a requested claim, say so rather than filling the gap from model memory.

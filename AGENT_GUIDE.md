# Agent guide

Use `research-ultra-rag-mcp` to discover and synthesize evidence across one project's PDF/EPUB collection. The server retrieves cleaned semantic material; you remain responsible for interpretation, uncertainty, and final writing.

## Normal workflow

1. Call `status` before substantive research. It also lists every retained generation with its counts and size (`generations`, `retained_generation_bytes`), which is how you answer questions about disk use. The server never prunes generations, so never claim it did: tell the user which generations exist and that removing one is a manual decision, and only after stopping every process that might be using it. If `status` reports a `ui_url`, this server is serving the browser UI there and you may give that URL to the user; `ui_ready: false` with `ui_error` means it could not bind its port. You cannot start or stop that UI — it follows the server's lifetime — so never claim you did. If it reports `restart_required`, the running server is older than the installed version: tell the user to restart it in their MCP client.
2. Call `list_sources` when you need source handles. Even before ingestion, `discovered_sources` lists live PDFs/EPUBs with their stable `source_id`, inclusion state, and index state, and idempotently registers them in portable project state. `reviewed_metadata_sources` lists every saved override by the same stable ID, and a registered path whose original is missing is listed as `known_sources` in the full-detail payload. Prefer those stable IDs for metadata and inclusion decisions.
3. If no generation exists, explain that `ingest` writes persistent data, creates a generation, and may download a model; obtain agreement when appropriate.
4. If `stale=true`, explain the reported source or inclusion changes. The prior generation remains searchable. Ask whether to re-ingest. A true `metadata_overlay_active` is not staleness: reviewed metadata is already effective without rebuilding the generation. Metadata listed under `metadata_pending_source_paths` belongs to sources outside the selected generation and becomes effective after those sources are included and ingested.
5. If `generation_upgrade_required=true`, report `upgrade_reasons` and recommend `ingest`. The old generation remains searchable with a warning. Use `force_recompute=true` only when the user explicitly requests regeneration or compatible reuse must be bypassed.
6. When `ingest` returns `status="in_progress"`, repeat it with identical chunk settings and force mode until it returns `ready` or `unchanged`. Report the build ID, phase, and completed/total progress when useful. A selected prior generation remains searchable during the staged build.
7. Search with the actual research question. Hybrid is the normal default.
8. Inspect several hits. Fewer than requested—or zero—can be the correct result after relevance gates.
9. Use `get_passage` for surrounding semantic context. For a direct quotation, open the original at `source_relative_path` and `locator`; never quote returned `text` as though it were an exact transcript.
10. If two files appear to be the same work, do not count them as independent support. Explain the issue and use `set_source_inclusion` only after agent/user review.

The local UI uses the same public tools. Call `status` again before relying on state observed earlier in a long session.

## Choosing retrieval

- `hybrid`: ordinary research; weighted RRF combines BM25 (`1.25`) and dense (`1.0`) ranks.
- `bm25`: exact terms, names, and distinctive phrases. A candidate must contain at least one non-stopword query token.
- `dense`: conceptual similarity. A candidate must have cosine similarity `>= 0.72`.
- `rerank=true`: CPU cross-encoder reranking, which is **on by default** and runs after the normal gates; slower (about 2.3 s per warm query against 0.17 s for unreranked hybrid) and it may download a second model on first use. Pass `rerank=false` for the lowest latency or when the model is unavailable. On the judged set in `MEASUREMENTS.md` it is the largest measured gain, lifting first-position success from 66% to 81% and entity questions from 60% to 80%. When the response contains `rerank_fallback`, the model could not be loaded and the order you received is the plain unranked candidate order.
- `result_view="passages"`: preserve the global passage ranking.
- `result_view="references"`: use the same ranked candidate pool but cap each `source_id` at `passages_per_reference` (default `2`) within the unchanged total `top_k` passage budget. Use this when several useful references matter more than several passages from one reference.
- `include_staleness=true` (default) makes each search re-compare the source directory with the generation to report `stale`. That comparison grows with the number of sources, so pass `include_staleness=false` for follow-up searches in a session where you have already called `status` and only need evidence. The response then reports `stale: null`, which means the freshness was not checked, not that the generation is current. Call `status` before telling a user whether their corpus is up to date.

## What a tool answer contains

Every tool answer is lean by default, because your context belongs to the research rather than to the server's internals. A search gives you the passages plus four facts: `query`, `generation_id`, `stale`, and `reranked`. Anything you ask about how retrieval scored, filtered, or withheld a candidate — component ranks, fusion and reranker scores, dense similarity, gate rejections with their counts, withheld candidate reason codes and example chunk IDs, and how long each ingestion phase took — is not part of that answer. If a user asks such a question, tell them to start the server with `--tool-detail full` (`RESEARCH_ULTRARAG_TOOL_DETAIL=full`), which returns the complete payload for a human to read. Never present that debug payload as evidence either way, and never invent a score, a rank, or a withheld count.

Two conventions make the lean answer readable:

- A field that is absent means "nothing to report", not "unknown": no unresolved ID, no ingestion required, no non-empty warning list. Do not ask for a field that is not there, and do not treat its absence as a problem to solve.
- `stale` and `reranked` are always present. `stale: null` means the check was skipped; `reranked: false` with `rerank_fallback` means the requested reranker could not run and you received the unranked candidate order.

Category and keyword filters use AND semantics. Document IDs use membership semantics; they name a content/path version inside one generation and are not reported in a lean answer, so prefer `source_id` for selection. Extraction artifacts and nonempty chunks containing no Unicode alphanumeric content are rejected. Text, numbers, and formulas containing at least one letter or digit are not symbol-only. A candidate is withheld only for corruption evidence: replacement characters, private-use or unassigned code points, or a known damaged encoding sequence. Script mixing never withholds a passage; it appears in per-hit `text_notes`.

## Reading a hit

- `text`: cleaned semantic text for comprehension and paraphrase, never quotation.
- `text_notes`: advisory script notes such as `non_latin_dominant` or `mixed_script_text`, present only when there are any. They never mean the passage was unusable; they warn that it mixes scripts, so read its locator before treating it as English prose.
- `direct_quote_safe`: always `false` under this extraction contract.
- `source_id`: the preferred project-scoped handle for metadata and inclusion edits. It survives content changes but changes when the source is renamed or moved.
- `source_relative_path` + `locator`: where to open the original.
- `rank`: this passage's position in the returned answer. It is an ordering within this one search, not a quality score and not comparable between searches.
- `title`, `authors`, `year`, `doi`: resolved bibliography, present only when it was resolved. `citation` is the ready-to-use form of those fields plus the locator.
- `metadata_warnings`, when present: which automatic value needs review, such as a title that fell back to the filename or conflicting candidates.
- In the reference view, read `reference_groups`: each entry names one reference and the passages selected from it, with each passage keeping its `rank`. That view answers with the groups instead of `hits`, so nothing is returned twice. `passages_per_reference` caps each source, and `top_k` remains the total passage budget.
- Treat automatically extracted bibliography as a best-effort starting point, not an authority. If a title, author, year, or DOI is missing, uncertain, or wrong, inspect the original, ask the user when needed, and save the reviewed value with `set_source_metadata`. Supply exactly one selector; prefer `source_id`, with `source_path` retained for compatibility. For an indexed source, the complete override immediately updates this metadata, citations, filters, source listings, and neighboring passages. Check `effective_immediately` and `requires_ingest` in the response; ingestion is needed only if the source is absent from the selected generation. The metadata object replaces the complete override: `{}` restores all automatic values, while an explicit empty value clears that automatic field. Do not expect ingestion heuristics to know document- or publisher-specific conventions.
- Per-field provenance (`metadata_provenance`), a passage's `document_id`, `match_kind`, `content_kind`, extraction `annotations` and `quality_flags`, embedding token counts, and the component scores are all part of the full-detail payload. Ask the user to enable `--tool-detail full` if one of them is genuinely needed for debugging; do not guess.

Never invent a title, author, DOI, date, locator, score, or quotation.

## Source and generation rules

- Only regular `.pdf` and `.epub` files beneath the configured source directory are indexed. Markdown and other formats are ignored.
- `.research-rag/project.json` owns the stable project ID and source-directory setting. Once initialized, commands reuse that setting when it is omitted.
- A `source_id` combines that project identity with the source-relative path, so replacing the file's bytes preserves it and renaming/moving the file changes it. A `document_id` identifies one path/content version and can change between generations; it stays in the full-detail payload rather than in the lean answers.
- `ingest` hashes every source. An exact input match returns the selected generation unchanged. Otherwise it safely reuses compatible unchanged documents, chunks, and exact-text vectors while building complete new BM25 and dense indexes. Work is durably checkpointed between bounded units; `status.ingestion_progress` reports an unfinished build. `force_recompute=true` bypasses reuse but may resume its own matching checkpoint.
- `current.json` changes only after BM25 and the dense index both succeed. Prior and successful generations remain on disk. Cancellation and timeout retain a resumable checkpoint; incompatible inputs supersede it with a small diagnostic, and non-resumable failures leave only a small failure record.
- Corrupt extraction units are omitted whole, with locator/reason diagnostics but without retained garbage text. Retrieval also withholds corrupt chunks from older generations. The full-detail payload reports both the corpus-level counts recorded at ingestion and the reason codes with example chunk IDs for one search; a lean answer reports neither, so never guess them. Never reconstruct, repair, or invent rejected wording.
- Script mixing and non-Latin dominance are advisory `text_notes`, never withhold reasons. A foreign-language quotation inside an English source is evidence; cite its locator instead of discarding it.
- Formula-font letters (`𝑀` → `M`) and the `ﬁ`, `ﬂ`, `ﬀ` ligatures are folded to their plain spellings during cleaning, so a typed query matches the printed text. Accented letters, superscripts, and subscripts are unchanged.
- Ingestion records each chunk's embedding token count and whether its vector covers only the beginning of its text. A non-zero `dense_truncated_chunk_count` in the ingestion response means some chunks run past the embedding model's limit, so their dense vector covers a prefix while BM25 still matches the whole text: treat those passages as lexically reliable and semantically partial, and read their locators. Per-passage counts and flags are full-detail only.
- Symbol-only chunks are omitted during ingestion and guarded at retrieval for older generations. A chunk with at least one Unicode letter or digit is not symbol-only, including an alphanumeric formula or numeric content; other extraction-artifact checks still apply.
- Read `generation_changed`, `status`, and the reuse/rebuild and vector counts from the ingestion response before reporting what occurred. Phase timings live in the full-detail payload.
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

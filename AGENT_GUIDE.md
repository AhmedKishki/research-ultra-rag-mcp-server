# Agent guide

Use `research-ultra-rag-mcp` to discover and synthesize evidence across one project's PDF/EPUB collection. The server retrieves cleaned semantic material; you remain responsible for interpretation, uncertainty, and final writing.

## Normal workflow

1. Call `status` before substantive research. It reports what a prune would consider as `retained_generation_count` and `retained_generation_bytes`, which is how you answer questions about disk use; the per-generation list is a `--tool-detail full` reader, so never report generations you have not been shown. The server never prunes generations, so never claim it did: tell the user how many are retained and what they occupy, and that removing one is a manual decision, and only after stopping every process that might be using it. If `status` reports a `ui_url`, this server is serving the browser UI there and you may give that URL to the user; `ui_ready: false` with `ui_error` means it could not claim its port, and that message says why. You cannot start or stop that UI — it follows the server's lifetime — so never claim you did. If it reports `restart_required`, the running server is older than the installed version: tell the user to restart it in their MCP client.
2. Call `list_sources` when you need source handles. Even before ingestion, `discovered_sources` lists live PDFs/EPUBs with their stable `source_id`, inclusion state, and index state, and idempotently registers them in portable project state. `reviewed_metadata_sources` lists every saved override by the same stable ID. Prefer those stable IDs for metadata and inclusion decisions.
3. If no generation exists, explain that `ingest` writes persistent data, creates a generation, and may download a model; obtain agreement when appropriate.
4. If `stale=true`, explain the reported changes: how many sources were added and modified (`changes.added_source_count`, `changes.modified_source_count`), by name which sources are gone from the directory (`changes.removed_sources`), and whether reviews or exclusions moved (`metadata_changed`, `source_exclusions_changed`). A status answer never names the sources that are still available; call `list_sources` when you need the inventory. The prior generation remains searchable. Ask whether to re-ingest. A true `metadata_overlay_active` is not staleness: reviewed metadata is already effective without rebuilding the generation. A non-zero `metadata_pending_source_count` counts reviewed sources outside the selected generation; their reviews take effect after those sources are included and ingested.
5. If `generation_upgrade_required=true`, report `upgrade_reasons` and recommend `ingest`. The old generation remains searchable with a warning. Use `force_recompute=true` only when the user explicitly requests regeneration or compatible reuse must be bypassed.
6. When `ingest` returns `status="in_progress"`, repeat it with identical chunk settings and force mode until it returns `ready` or `unchanged`. Report the build ID, phase, and completed/total progress when useful. A selected prior generation remains searchable during the staged build.
7. Search with the actual research question. Hybrid is the normal default.
8. Inspect several hits. Fewer than requested—or zero—can be the correct result after relevance gates.
9. Use `get_passage` for surrounding semantic context. For a direct quotation, open the original at `source_relative_path` and `locator`; never quote returned `text` as though it were an exact transcript.
10. If two files appear to be the same work, do not count them as independent support. Explain the issue and use `set_source_inclusion` only after agent/user review.

The local UI uses the same public tools. Call `status` again before relying on state observed earlier in a long session.

## How retrieval is chosen for you

- Hybrid retrieval with CPU cross-encoder reranking, always. BM25 (weighted `1.25`) and dense (`1.0`) ranks are fused, candidates are gated (BM25 needs at least one non-stopword query token, dense needs cosine `>= 0.72`), and the reranker then reorders at most 50 candidates. On the judged set in `MEASUREMENTS.md` this is the best of the measured modes, so there is no mode to choose.
- It is the slow path by design: about 2.3 s per warm query against 0.17 s for unranked hybrid, and it may download a second model on first use. Which model reranks is the server's setting and not yours: reranking is always on, and there is no parameter to change or disable it. `MEASUREMENTS.md` records how the supported models compare. When the response contains `rerank_fallback`, the model could not be loaded and the order you received is the plain unranked candidate order.
- Every search re-compares the source directory with the generation to report `stale`. That comparison grows with the number of sources but is always done, so `stale` is never null through this tool. Call `status` when you need the detail behind a stale verdict.
- A thin answer is a reason to ask again, not to conclude: the reranker reorders about twice the number of candidates you request, so raising `top_k` (15 is a good next step, 20 for a hard question) deepens the ranking as well as the answer, and the same question in different words is a different search. Reserve "the corpus has no answer" for after both.
- Those numbers are operator settings, not per-call ones: the fusion weights, the relevance gates, the candidate caps, chunk size, and the reranker model come from the server's own settings file (or `--set` on its command line). Nothing you pass as a tool argument changes them.
- `status` reports `hybrid_ready: false` when the selected generation predates dense support, which is the one case where this search cannot serve it: report it, recommend `ingest`, and do not look for a way to ask for another retrieval method, because the tool offers none.

## Reading an answer

A search answers with `query`, `generation_id`, `stale`, `reranked`, and its passages. A field that is absent means there is nothing to report — no unresolved ID, no required ingestion, no warning — not that its value is unknown. `reranked: false` with `rerank_fallback` means the reranker did not run and the order you received is the plain candidate order.

- `text`: cleaned semantic text for comprehension and paraphrase, never quotation.
- `source_relative_path` + `locator`: where to open the original, and how the source is named for `set_source_inclusion`. The locator is the position alone — a page, carrying `page_label` only where the printed label differs from the physical page, or a section for an EPUB. A stable `source_id` is not repeated in an answer; read it from `list_sources` when a filter needs one.
- `authors`: the resolved author list, present only when it was resolved. A passage carries no title and no citation: a citation-ready reference is `--tool-detail full` material, and the reviewed bibliography is in `list_sources`.
- `chunk_id`: this passage's handle for `get_passage`.
- A stale `status` carries `changes`: counts for added and modified sources, `removed_sources` naming what is gone from the directory, and the review and exclusion flags. Available sources are counted there rather than listed; `list_sources` is the inventory, and `status` never becomes a corpus listing.

Categories and projects are matched with any-of semantics (`categories_any`, `projects_any`: a result must carry at least one listed value), and keywords are matched with all-of semantics. A status answer does not inventory the vocabulary, because that is a question of its own: take it from the user, from the project's `source-metadata.json`, or from `list_sources`. Extraction artifacts and nonempty chunks containing no Unicode alphanumeric content are rejected. Text, numbers, and formulas containing at least one letter or digit are not symbol-only. A candidate is withheld only for corruption evidence: replacement characters, private-use or unassigned code points, or a known damaged encoding sequence. Script mixing never withholds a passage.

- Treat automatically extracted bibliography as a best-effort starting point, not an authority. If a title, author, year, or DOI is missing, uncertain, or wrong, inspect the original and tell the user, who corrects it in `.research-rag/source-metadata.json`; there is no metadata tool and you must not edit that file yourself. Reviewed values are authoritative at read time, so a correction reaches listings, filters, and search answers with no re-ingestion, while a source absent from the selected generation takes effect after the next ingestion. Do not expect ingestion heuristics to know document- or publisher-specific conventions.

Never invent a title, author, DOI, date, locator, score, or quotation.

## Source and generation rules

- Only regular `.pdf` and `.epub` files beneath the configured source directory are indexed. Markdown and other formats are ignored.
- `.research-rag/project.json` owns the stable project ID and source-directory setting. Once initialized, commands reuse that setting when it is omitted.
- A `source_id` combines that project identity with the source-relative path, so replacing the file's bytes preserves it and renaming/moving the file changes it. A `document_id` identifies one path/content version, changes between generations, and is not reported in an answer.
- `ingest` hashes every source. An exact input match returns the selected generation unchanged. Otherwise it safely reuses compatible unchanged documents, chunks, and exact-text vectors while building complete new BM25 and dense indexes. Work is durably checkpointed between bounded units; `status.ingestion_progress` reports an unfinished build. `force_recompute=true` bypasses reuse but may resume its own matching checkpoint.
- `current.json` changes only after BM25 and the dense index both succeed. Prior and successful generations remain on disk. Cancellation and timeout retain a resumable checkpoint; incompatible inputs supersede it with a small diagnostic, and non-resumable failures leave only a small failure record.
- Corrupt extraction units are omitted whole, with locator/reason diagnostics but without retained garbage text. Retrieval also withholds corrupt chunks from older generations, and the ingestion response reports those counts. Never reconstruct, repair, or invent rejected wording.
- Script mixing and non-Latin dominance are never withhold reasons. A foreign-language quotation inside an English source is evidence; cite its locator instead of discarding it.
- Formula-font letters (`𝑀` → `M`) and the `ﬁ`, `ﬂ`, `ﬀ` ligatures are folded to their plain spellings during cleaning, so a typed query matches the printed text. Accented letters, superscripts, and subscripts are unchanged.
- Ingestion records each chunk's embedding token count and whether its vector covers only the beginning of its text. A non-zero `dense_truncated_chunk_count` in the ingestion response means some chunks run past the embedding model's limit, so their dense vector covers a prefix while BM25 still matches the whole text: treat those passages as lexically reliable and semantically partial, and read their locators.
- Symbol-only chunks are omitted during ingestion and guarded at retrieval for older generations. A chunk with at least one Unicode letter or digit is not symbol-only, including an alphanumeric formula or numeric content; other extraction-artifact checks still apply.
- Read `generation_changed`, `status`, and the reuse/rebuild and vector counts from the ingestion response before reporting what occurred.
- Metadata review is a file edit, not a tool call. The project's `.research-rag/source-metadata.json` holds a complete override per source: omitting a field stops overriding it, `{}` restores every automatic value, and an explicitly empty value clears one field. It is authoritative at the next read wherever the source is in the selected generation. Never write that file yourself; report what looks wrong and let the user correct it.
- Exclusion is explicit, reversible, and immediately enforced without deleting the source. Call `set_source_inclusion` with the filename that `list_sources` reports; re-ingest to omit it physically from new indexes.
- A chunk ID belongs to the generation that returned it and may change after a rebuild.
- A project moves by copying its directory: `sources/` plus `.research-rag/`. `runtime/` is disposable and rebuilds on the next ingestion.

## Recommended answer behavior

Use retrieved text to find relevant originals and form paraphrases. Attach the resolved source and locator to material claims. If exact wording matters, open the original and quote from it directly. When sources disagree, expose the disagreement. When retrieval abstains or does not support a requested claim, say so rather than filling the gap from model memory.

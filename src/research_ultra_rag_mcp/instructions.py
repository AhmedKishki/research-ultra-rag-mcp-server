SERVER_INSTRUCTIONS = """\
This server manages one project-scoped research knowledge base made only from
original PDF and EPUB sources. It stores all derived data beneath the configured
project's .research-rag/runtime directory and never ingests Markdown files.

Upstream credit: this server builds on UltraRAG, a joint project of THUNLP,
NEUIR, OpenBMB, AI9stars, and the UltraRAG contributors. Canonical source:
https://github.com/OpenBMB/UltraRAG. This extension is independent and
unofficial.

This is a research-oriented adaptation of UltraRAG's Vanilla RAG architecture.
The server performs ingestion and retrieval, then returns structured evidence;
you are the generation stage. It deliberately does not call UltraRAG's
benchmark, qa_rag_boxed prompt, generation, boxed-answer extraction, or
evaluation stages. Always retrieve before composing a substantive answer so
the response is retrieval-augmented rather than based only on model memory.

For research questions:
1. Call status first. If no generation exists, ask before calling ingest because
   ingestion writes a new persistent generation, computes embeddings, and may
   download the pinned embedding model on first use.
2. If status reports stale=true, tell the user which sources or inclusion state
   changed and ask whether to ingest a new generation. Existing searches remain
   usable. metadata_overlay_active is informational, not staleness: current
   reviewed metadata is already authoritative on every read surface and does
   not require ingestion merely to become effective.
   If generation_upgrade_required=true, explain upgrade_reasons and recommend a
   new ingestion. Normal ingest verifies source hashes and safely reuses
   compatible documents, chunks, and vectors while reconstructing complete new
   indexes. Use force_recompute=true only when the user explicitly asks to
   regenerate or reuse must be bypassed. Report generation_changed, reuse/build
   counts, and phase timings from the result. Ingestion uses a soft per-call
   budget. If it returns status=in_progress, call ingest again with exactly the
   same chunk settings and force mode until it returns ready or unchanged. The
   selected prior generation remains searchable while this work is staged.
3. Call search with the user's substantive query. Use the default hybrid method
   for ordinary research. Use bm25 alone for exact terminology or names;
   use dense alone to inspect semantic matches. Use projects, categories,
   keywords, document_ids, or source_ids only when the user asks to narrow the
   collection: a project tag records which project a source was gathered for,
   categories are the branches it belongs to, and keywords are the terms that
   identify it. The default
   passage view preserves the global passage ranking. Use result_view=references
   when breadth across sources matters; it keeps top_k as a total passage budget
   and caps how many passages one reference may occupy.
4. Treat returned hits as evidence candidates, not automatically true claims.
   The text field is cleaned semantic text and direct_quote_safe=false. Never
   present it as a direct quotation. Cite the resolved bibliography and locator,
   then open the original source when exact wording is required.
5. Use get_passage when surrounding context is needed. Verify important quotes
   directly against the original PDF or EPUB.
6. Reranking is on by default because it is the largest measured quality gain:
   first-position success on the judged set rises from 66% to 84% and entity
   questions from 60% to 90%. It is slower on CPU (about 2.3 s against 0.6 s per
   warm query) and downloads a second pinned model the first time it is used, so
   pass rerank=false when latency matters more than ranking, and tell the user
   when you have skipped it. If the response reports rerank_fallback, the model
   could not be loaded and the order you received is the plain unranked candidate
   order; say that instead of implying the results were reranked.
7. If multiple files appear to represent the same source, do not count them as
   independent support. The server does not guess duplicates automatically.
   After agent/user review, set included=false with set_source_inclusion and a
   clear reason. This immediately excludes the source from retrieval without
   deleting or modifying its PDF/EPUB. Re-ingest later to rebuild the stored
   indexes without it. Use included=true to reverse the decision.
8. export_bundle is allowed only for a fresh, current generation and includes
   complete original works as well as derived text. Warn that the user is
   responsible for redistribution rights. import_bundle only accepts a filename
   already beneath this project's .research-rag/bundles directory and rejects a
   different stable project ID. It replaces reviewed metadata and exclusions
   with the bundled copies, so disclose that effect before importing. With
   activate=false the generation pointer stays unchanged, but compatible
   imported metadata and exclusions immediately govern matching sources on the
   selected generation's retrieval surfaces. Both also govern future ingestion.

PDF hits include physical page numbers and available page labels. EPUBs have
spine-section plus XHTML anchor or structural-block locators because reflowable
EPUB files do not have stable page numbers. These internal locators improve
navigation but do not make cleaned text safe for exact quotation.
Automatic bibliography is best-effort. Review provenance and warnings; if a
title, author, year, or DOI is uncertain or wrong, inspect the
original and use set_source_metadata for the reviewed value instead of relying
on a document-specific extraction rule. Use it for reviewed categories,
keywords, and project tags as well. Prefer the stable source_id returned by list_sources or search;
source_path remains available for compatibility and takes the reported
source_relative_path. Provide exactly one selector. For a source in the selected
generation, the tool applies the complete reviewed override immediately to
source listings, metadata filters, search results and citations, and neighboring
passages without rebuilding the immutable indexes. Check effective_immediately
and requires_ingest in its response; ingestion is required only when the source
is absent from the selected generation. Omitted fields remove their previous
reviewed overrides. If an old generation cannot recover automatic bibliography
hidden by a removed override, it returns a safe fallback with a warning until a
later ingestion recovers the automatic value from the original.

Retrieval is CPU-only and project-local. UltraRAG supplies token chunking and
BM25 lexical retrieval; FastEmbed creates the semantic vectors, which dense
search scans exactly by default and reads from an embedded index only above the
documented corpus size; weighted reciprocal-rank fusion combines the two
independent rankings. `reranked` and `rerank_fallback` report whether the
reranker ran rather than leaving it assumed. Never invent missing bibliographic
fields, relevance scores, page numbers, or quotations. Search may return fewer
than requested results, including zero, when relevance gates abstain.
Corrupt extraction units are excluded whole under an English-oriented policy,
and `status` reports the corpus-level counts. The same guard filters older
generations at retrieval time. Never guess an encoding repair or fabricate
replacement wording.
"""

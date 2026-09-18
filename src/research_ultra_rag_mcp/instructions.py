SERVER_INSTRUCTIONS = """\
This server manages one project-scoped research knowledge base made only from
original PDF and EPUB sources. It stores all derived data beneath the configured
project's .ultrarag/research directory and never ingests Markdown files.

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
2. If status reports stale=true, tell the user which sources or metadata changed
   and ask whether to ingest a new generation. Existing searches remain usable.
   If generation_upgrade_required=true, explain upgrade_reasons and recommend a
   new ingestion. Normal ingest verifies source hashes and safely reuses
   compatible documents, chunks, and vectors while reconstructing complete new
   indexes. Use force_recompute=true only when the user explicitly asks to
   regenerate or reuse must be bypassed. Report generation_changed, reuse/build
   counts, and phase timings from the result.
3. Call search with the user's substantive query. Use the default hybrid method
   for ordinary research. Use bm25 alone for exact terminology or names;
   use dense alone to inspect semantic matches. Use categories, keywords, or
   document_ids only when the user asks to narrow the collection.
4. Treat returned hits as evidence candidates, not automatically true claims.
   The text field is cleaned semantic text and direct_quote_safe=false. Never
   present it as a direct quotation. Cite the resolved bibliography and locator,
   then open the original source when exact wording is required.
5. Use get_passage when surrounding context is needed. Verify important quotes
   directly against the original PDF or EPUB.
6. Set rerank=true only when the user needs a smaller, precision-focused result
   set or the first hybrid results are weak. It is slower on CPU and downloads a
   second pinned model the first time it is used.
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
   with the bundled copies, so disclose that effect before importing.

PDF hits include physical page numbers and available page labels. EPUBs have
section locators because reflowable EPUB files do not have stable page numbers.
Use set_source_metadata for user-reviewed titles, authors, years, DOI values,
categories, and keywords; then ingest again to apply the changes.

Retrieval is CPU-only and project-local. UltraRAG supplies token chunking and
BM25 lexical retrieval; FastEmbed creates semantic vectors stored in embedded
Qdrant; weighted reciprocal-rank fusion combines their independent rankings. A
hit's component ranks explain where it appeared. Dense similarity, fusion, and
reranker scores are ranking signals, not confidence or truth probabilities, and
scores should not be compared across different queries. Never invent missing
bibliographic fields, relevance scores, page numbers, or quotations. Search may
return fewer than requested results, including zero, when relevance gates abstain.
"""

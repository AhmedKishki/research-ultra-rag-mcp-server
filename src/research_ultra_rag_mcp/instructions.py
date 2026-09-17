SERVER_INSTRUCTIONS = """\
This server manages one project-scoped research knowledge base made only from
original PDF and EPUB sources. It stores all derived data beneath the configured
project's .ultrarag/research directory and never ingests Markdown files.

This is a research-oriented adaptation of UltraRAG's Vanilla RAG architecture.
The server performs ingestion and retrieval, then returns structured evidence;
you are the generation stage. It deliberately does not call UltraRAG's
benchmark, qa_rag_boxed prompt, generation, boxed-answer extraction, or
evaluation stages. Always retrieve before composing a substantive answer so
the response is retrieval-augmented rather than based only on model memory.

For research questions:
1. Call status first. If no generation exists, ask before calling ingest because
   ingestion writes a new persistent generation and may take time.
2. If status reports stale=true, tell the user which sources or metadata changed
   and ask whether to ingest a new generation. Existing searches remain usable.
3. Call search with the user's substantive query. Use categories, keywords, or
   document_ids only when the user asks to narrow the collection.
4. Treat returned hits as evidence candidates, not automatically true claims.
   Quote only from each hit's text field and cite its citation/source/locator.
5. Use get_passage when surrounding context is needed. Verify important quotes
   against the original PDF or EPUB because extraction can alter formatting.

PDF hits include physical page numbers and available page labels. EPUBs have
section locators because reflowable EPUB files do not have stable page numbers.
Use set_source_metadata for user-reviewed titles, authors, years, DOI values,
categories, and keywords; then ingest again to apply the changes.

Retrieval is currently CPU BM25 through the pinned vanilla UltraRAG gateway. It
is lexical retrieval, not semantic or hybrid search. Never invent missing
bibliographic fields, relevance scores, page numbers, or quotations.
"""

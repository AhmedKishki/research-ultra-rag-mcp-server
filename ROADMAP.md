# Roadmap

Product ideas that are deliberately not being built yet. Nothing here is scheduled, and each would need a decision before it was started. Work that is already in scope is in `TODO.md`, current behaviour is in `README.md`, and current facts and limits are in `MEASUREMENTS.md`.

The server targets English-primary born-digital PDF and EPUB sources of 5,000–50,000 chunks on CPU. Anything outside that envelope — other-language corpora, OCR'd or scanned material, handwriting, formula-heavy documents — is a different product rather than a roadmap step, so it is not listed here.

## Upstream reuse

- Re-evaluate adopting UltraRAG's dense index backends (`retriever_init(index_backend="faiss"|"qdrant"|"milvus")`) and its reranking components once an upstream release returns identifiers and scores under query-time metadata filters, and offers a reranker that is CPU-only, offline, revision-pinned, and free of any server or credential.
- If those criteria are met, the work is a component swap behind the `DenseBackend` boundary plus a new manifest backend name, and the source and retrieval integration coverage in `AGENTS.md` must accept it before any generation selects it.

## Citations and quotation

- Character offsets inside extraction units, so a hit can point at a span instead of a whole unit.
- The printed page label distinguished from the physical page, wherever a document carries both.
- An exact-quote verification tool: the missing piece between cleaned semantic text and text that is safe to quote.
- Citation export in common bibliographic styles, without inventing any metadata.

## Scale and operations

- A persistent document-metadata index, which a corpus of tens of thousands of sources would need before per-process caches of the document map or of the staleness verdict are worth their invalidation risk.
- Pushing category and keyword filtering into the embedded dense index, which matters only above the 200,000-chunk threshold where that backend is selected.
- Incremental dense-index construction for very large collections, where the embedded backend currently rebuilds its index for every changed generation.

## Other

- Optional backup profiles that exclude originals, for users who already store their PDFs elsewhere.
- Optional cross-project search that keeps each project's boundary explicit rather than merging indexes into one.

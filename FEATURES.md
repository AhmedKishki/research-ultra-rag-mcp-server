# Features

This document lists what the server can do, and — just as importantly — where each capability comes from. Some of it is UltraRAG, some is built on top of UltraRAG, and some is only planned. It also compares this server with a different, more general MCP RAG server, because "which one should I use?" depends on the job.

Three labels are used throughout:

- **UltraRAG** — the capability comes from the upstream framework this project pins, and this server uses it as it is.
- **Added here** — the research layer around UltraRAG. This is where most of the product lives.
- **Planned** — described in `ROADMAP.md` or `TODO.md`, not built yet. Nothing in this document claims otherwise.

For measurements behind the performance-related features, see `PLAN.md`. For how to use them, see `README.md`.

## 1. What UltraRAG provides, and what this server actually uses

UltraRAG is much larger than the part this server needs. Its gateway exposes around 78 tools and 26 prompts across corpus, retrieval, reranking, prompt, generation, routing, memory, benchmark, and evaluation components, with several vector-database backends and an upstream web interface.

This server deliberately uses only three upstream capabilities, and owns everything else itself:

| UltraRAG capability | Used here? | What this server does with it |
|---|---|---|
| GPT-2 token chunking (`corpus_chunk_documents`) | Yes | Splits extraction units into token chunks with a configured size and overlap. Several units share one call for speed; each unit still gets its own durable output. |
| BM25 lexical index and search (`retriever_retriever_init`, `retriever_bm25_index`, `retriever_bm25_search`) | Yes | Supplies the lexical half of retrieval, in English, on CPU. |
| FAISS, Qdrant and Milvus dense backends | No | The dense path is built here instead: FastEmbed embeddings plus either an exact scan of the portable vectors or an embedded Qdrant index, chosen per generation and recorded in its manifest. |
| Reranking components | No | This server uses its own optional CPU cross-encoder, applied to at most 50 candidates. |
| Prompt assembly and answer generation | No | The connected agent generates. The server returns evidence, not prose. |
| Routing, memory, benchmark, evaluation components | No | Not part of this server's contract. Retrieval-quality evaluation is planned as its own work (see section 4). |
| Web-search retrieval | No | Out of scope: this server answers from a project's own documents only. |
| Upstream web interface | No | This server ships its own local UI for its own nine tools. |

Two consequences are worth stating plainly.

First, **the chunker and BM25 are upstream, but the parameters and the surrounding policy are not**: chunk size, overlap, which units are chunked, how results are filtered, and what is disclosed about rejected text are all decided here.

Second, **pinning a small upstream surface is a feature, not a gap**. It means an UltraRAG upgrade can be evaluated against three well-understood call sites instead of dozens, and that this server never depends on an upstream service, credential, GPU, or vector database being available.

## 2. Features added on top of UltraRAG

### Ingestion and text quality

| Feature | What it does | Why it matters |
|---|---|---|
| PDF and EPUB ingestion | Walks the source directory recursively and accepts regular `.pdf` and `.epub` files only. | Other formats are ignored on purpose, so a stray `notes.md` never enters the knowledge base. |
| Layout-aware extraction | Reads PDFs page by page and EPUBs by spine section, keeping structure and discarding repeated page furniture. | Reading order and paragraphs survive; running headers do not become fake content. |
| Original-file locators | Every unit and chunk points back to a PDF page (with the printed page label when available) or an EPUB section. | You can open the source and check the passage. Locators are navigation aids, not quote offsets. |
| Bibliographic metadata with provenance | Resolves title, authors, year, and DOI per document, recording where each value came from and any warnings. | Automatic metadata is provisional, and the response says so instead of presenting a guess as fact. |
| Reviewed metadata overlay | Corrections saved outside the generation apply immediately to listings, filters, citations, search results, and neighbouring passages. | Fixing a wrong author does not require re-ingesting, and does not rewrite immutable data. |
| Corrupt-text rejection with disclosure | Rejects a page or section only when its text shows strong evidence of a broken character map, then reports reason codes, counts, and example chunk IDs. | Bad text leaves the index, and you are told what was removed rather than silently losing material. |
| Script mixing as an advisory note | A quotation in Greek, Cyrillic, or any other script stays retrievable and is returned with `text_notes`. | English-language scholarship quotes other languages; withholding those passages was a real defect that this fixes. |
| Targeted normalisation folding | Formula-font letters (`𝑀` becomes `M`) and the presentation ligatures `ﬁ`, `ﬂ`, and `ﬀ` fold to plain spellings. Accents, superscripts, subscripts, and symbols are left alone. | A typed query matches printed text, without destroying notation that carries meaning. |
| Symbol-only chunk exclusion | Nonempty chunks containing no letters or digits are dropped. | Extraction artefacts do not pollute results, while numbers and formulas are unaffected. |
| Embedding-token audit | Every chunk records its embedding token count and whether its vector covers only a prefix. | Silent truncation becomes visible per hit and in aggregate instead of quietly degrading dense search. |
| Compatible-material reuse | Unchanged documents keep their extracted units and chunks; identical chunk text keeps its vector. | Adding one source costs a fraction of a rebuild. |
| Resumable ingestion | Long builds are checkpointed; a call that exceeds its soft time budget returns `in_progress` and you repeat it. Cancellation, timeout, and restart lose at most one bounded batch. | A rebuild of a large collection does not have to succeed in one sitting. |
| Explicit regeneration | `force_recompute` bypasses all reuse while still resuming its own checkpoint. | You can rebuild from scratch deliberately rather than by accident. |

### Retrieval

| Feature | What it does | Why it matters |
|---|---|---|
| BM25 lexical search (UltraRAG) | Matches the words you typed. | Exact names, terms, and phrases. Also the honest baseline when a semantic result surprises you. |
| Dense semantic search (added here) | FastEmbed `bge-small-en-v1.5` embeddings, 384 dimensions, CPU, revision-pinned. | Finds passages phrased differently from your question. |
| Hybrid search (UltraRAG plus added here) | Weighted reciprocal-rank fusion of the two rankings, with an opt-out to inspect either signal alone. | The default that behaves well on real questions. |
| Exact dense scan by default (added here) | Dense search scans the generation's portable float32 vectors directly; an embedded index is used above a documented corpus size. The manifest records which backend built the generation. | Nothing to build or keep in sync at normal sizes, and results are exactly reproducible. |
| Metadata filters (added here) | Narrow by category, keyword, or document. | Keeps a search inside the part of the collection you care about. |
| Reference-grouped results (added here) | `result_view="references"` caps how many passages each source contributes while keeping `top_k` as the total. | One prolific book chapter cannot fill the answer. |
| Optional reranking (added here) | A CPU cross-encoder reorders up to 50 candidates, opt-in. | A second pass when the candidate set is close. |
| Relevance gates with abstention (added here) | Weak candidates are dropped, so a search can legitimately return fewer results than `top_k`, including none, and the response explains which gate limited it. | An honest empty answer beats a confident irrelevant one. |
| Precomputed usability verdict (added here) | The decision "is this candidate structurally unusable?" is computed once when the lookup is built and stored per chunk. | The per-query gate measured 10.3× cheaper without changing a single rejection decision. |

### Project model, review, and durability

| Feature | What it does | Why it matters |
|---|---|---|
| One project per server process | A `--project-root` defines the boundary; all state lives under `<project>/.research-rag`. | Two projects cannot read each other's documents or indexes. |
| Portable versus derived state | Reviewed decisions (`project.json`, metadata, exclusions, catalog, bundles) are portable. Generations, staging, logs, and locks are derived and rebuildable. | You can back up what a human decided, and regenerate the rest. |
| Reviewed inclusions and exclusions | A source can be excluded as a duplicate and later restored. The original file is never deleted or modified. | Review stays reversible, and the server never destroys evidence. |
| Immutable generations | Each build produces a new generation. The active pointer moves only after both indexes validate. | A failed or interrupted build leaves the previous generation searchable. |
| Relocatable derived state | `--runtime-root` puts indexes, staging, and logs on a chosen disk, claimed by a marker naming its owning project. | A project on a slow disk can keep its working files on a fast one, without an OS-level bind mount. |
| Portable bundles | `export_bundle` and `import_bundle` move a project — originals, review state, cleaned artifacts, embeddings — with path, checksum, project-ID, and embedding-compatibility validation. Indexes are rebuilt on import. | A project can be moved, shared, or archived without hand-copying internals. |

### Interfaces and operational transparency

| Feature | What it does | Why it matters |
|---|---|---|
| Nine focused MCP tools | `status`, `ingest`, `search`, `list_sources`, `get_passage`, `set_source_metadata`, `set_source_inclusion`, `export_bundle`, `import_bundle`. | A small, reviewable surface for an agent, instead of upstream's ~78 tools. |
| Local evidence UI | A loopback-only browser workspace over the same nine tools and the same project state. | You can inspect and correct the knowledge base without an agent in the loop. |
| Terminal verifier | A read-only status and search check, with optional ingestion and forced recomputation. | Fast confidence that an installation works, and a scriptable smoke test. |
| Staleness and upgrade reporting | `status` says whether the selected generation is stale, and why, and whether a policy upgrade is required. | You know when a search is answering from older material, and why. |
| Build metrics and failure records | Phase timings, reuse counts, rejection counts, and truncation totals are recorded per generation; non-resumable failures leave a small record. | Slow or surprising builds can be diagnosed without guessing. |
| Documented performance characteristics | Measured, reproducible numbers for index builds, durability writes, embedding, and the query gate, plus a benchmark script to reproduce them on your hardware. | Claims can be checked instead of trusted. |

## 3. What this server deliberately does not do

These are choices, not missing pieces. Each one would change what the server is:

- **It does not generate answers.** It returns evidence candidates with provenance; the agent and the researcher interpret them.
- **It does not provide quote-safe transcripts.** Returned text is cleaned for retrieval, and every result says `direct_quote_safe: false`. Open the original.
- **It does not decide what is true, and never deletes or excludes sources on its own.** Duplicate and metadata decisions are reviewed and reversible.
- **It does not run OCR.** Scanned PDFs need OCR first; password-protected PDFs are rejected.
- **It is not multilingual.** Embeddings and the text-health policy are English-oriented, and non-English-primary corpora are outside the design envelope. This is stated in `README.md` and in `PLAN.md`.
- **It does not require an external service.** No hosted embedding API, no vector database server, no credentials. Models are downloaded once and cached locally.
- **It exposes no MCP resources.** Its surface is tools, with the local UI covering human inspection.

## 4. Planned additions

Everything here is deferred work, recorded in `ROADMAP.md` and `TODO.md`. None of it is claimed as present.

### Retrieval quality

- A judged query set for the reference corpus, so "is hybrid better than BM25 here?" has a measured answer rather than an assumption.
- Recall and ranking measurements for BM25, dense, hybrid, and reranked modes.
- Configurable fusion weights *only* if that measurement shows a repeatable benefit.

### Ingestion lifecycle

- Listing, inspecting, and pruning retained generations, which are roughly 101 MB each today and accumulate.
- Deliberate rollback to an earlier generation.
- A disk-space check before a build starts.
- Splitting the resumable ingestion loop into per-phase handlers, so review of that code is safer.

### Per-request efficiency

- Caching the staleness verdict behind a cheap directory signature, with an `include_staleness=false` opt-out for `search` so a caller can skip the source-tree walk.
- Pushing category and keyword filtering into the dense backend instead of materialising document-ID lists per query, plus SQLite connection reuse and a cached document map.

### Citations and quotation

- Character offsets inside extraction units, so a hit can point at a span rather than a whole unit.
- Distinguishing a PDF's physical page from its printed label more explicitly.
- An exact-quote verification tool — the missing piece between "cleaned semantic text" and "safe to quote".
- Citation export in common bibliographic styles, without inventing metadata.

### Scale and operations

- Incremental dense-index construction for very large collections. This matters only above the exact-scan threshold, where the embedded backend currently rebuilds its index for each changed generation.
- Optional backup profiles that exclude originals, for users who store their PDFs elsewhere.
- Optional cross-project search that keeps each project's boundary explicit.

## 5. Comparison with another MCP RAG server

The comparison below is with [`mcp-rag-server`](https://github.com/kwanLeeFrmVi/mcp-rag-server), a general-purpose RAG server written in TypeScript. It is a different tool for a different job, and it is deliberately *not* part of this collection: this repository's collection-level comparison covers its own sibling servers only.

Read this as two profiles rather than a scoreboard. It reflects that project's README at the time of writing; check its repository for the current state.

| Aspect | This server | `mcp-rag-server` |
|---|---|---|
| What it is for | Research evidence over scholarly PDF and EPUB collections, with provenance you can check | General RAG context for an LLM, over plain text files |
| Language and install | Python, `uv sync --frozen` from this repository | Node.js, `npm install -g mcp-rag-server` or `npx` |
| Document formats | Regular `.pdf` and `.epub` only | `.txt`, `.md`, `.json`, `.jsonl`, `.csv` |
| Chunking | UltraRAG GPT-2 token chunking, configurable size and overlap | Character count (`CHUNK_SIZE`, default 500) |
| Embeddings | FastEmbed CPU model pinned by revision, downloaded once, no service needed | An external embedding API (OpenAI, Ollama, Granite, or Nomic) over HTTP, Ollama by default |
| Retrieval | BM25, dense, and hybrid, with metadata filters, optional reranking, reference-grouped results, and relevance gates that can abstain | Dense nearest-chunk retrieval, top `k` default 15 |
| Dense store | The generation's portable vectors scanned exactly, or an embedded index above a size threshold | Local SQLite vector store (LangChain `LibSQLVectorStore`) |
| Metadata, citations, locators | Resolved title, authors, year, DOI with provenance and warnings, plus original-file locators and a reviewed-metadata overlay | Not part of the documented feature set |
| Index lifecycle | Immutable generations, validated before activation, resumable builds, reusable per-document and per-chunk work | Sequential indexing with progress reporting, plus per-document and whole-index removal |
| Cancellation and restart behaviour | Checkpointed: a build resumes where it stopped | Progress is reported; resumability is not documented |
| Interface | Nine tools, a local evidence UI, and a terminal verifier; no MCP resources | Five tools and four MCP resources (`rag://documents`, `rag://document/{path}`, `rag://query-document/{chunks}/{query}`, `rag://embedding/status`) |
| Human review | Reviewed metadata corrections and reversible exclusions in the UI | Not part of the documented feature set |
| Retrieval evaluation | Not yet measured — planned | Not part of the documented feature set |
| Licence | No licence file is present in this repository; `NOTICE` records UltraRAG and third-party attribution | MIT |

Which to choose:

- **Choose `mcp-rag-server`** if you want the shortest path from "I have notes in Markdown or CSV" to "my LLM can see them" — a single `npx` command, an existing Ollama or hosted embedding endpoint, and no concept of projects, generations, or citations to learn.
- **Choose this server** if your material is PDFs and EPUBs, if you need to cite and verify what you found, if you want the knowledge base to survive interruption and updates without being rebuilt by hand, and if you would rather not depend on an external embedding service.
- **Use both** if that is what the work needs. They share no state and can run side by side: one for quick text context, one for a citable document collection.

# research-ultra-rag-mcp-server

Use an AI agent to build and query a project-owned research knowledge base from
PDF and EPUB sources.

## Credit to UltraRAG

This project is a research-oriented extension built on
[`OpenBMB/UltraRAG`](https://github.com/OpenBMB/UltraRAG) through
[`vanilla-ultra-rag-mcp-server`](https://github.com/AhmedKishki/vanilla-ultra-rag-mcp-server).
UltraRAG's upstream team describes it as a joint project of
[`THUNLP`](https://nlp.csai.tsinghua.edu.cn/) at Tsinghua University,
[`NEUIR`](https://neuir.github.io/) at Northeastern University,
[`OpenBMB`](https://www.openbmb.cn/home), and
[`AI9stars`](https://github.com/AI9Stars), together with the
[`UltraRAG contributors`](https://github.com/OpenBMB/UltraRAG/graphs/contributors).
UltraRAG supplies the MCP architecture, corpus chunking, and BM25 retrieval on
which this server's research workflow depends.

The transitive upstream baseline is UltraRAG `0.3.0.2` at commit
[`3a709a2`](https://github.com/OpenBMB/UltraRAG/tree/3a709a2aea3fbe46acca59c422621c94b6e86857),
licensed under the
[`Apache License 2.0`](https://github.com/OpenBMB/UltraRAG/blob/3a709a2aea3fbe46acca59c422621c94b6e86857/LICENSE.txt)
with the upstream copyright notice `Copyright 2023 OpenBMB`.

This is an independent project. It is not an official UltraRAG release and is
not affiliated with or endorsed by OpenBMB or the other upstream organizations.
See [`NOTICE`](NOTICE) for complete attribution and a suggested software
citation.

Hybrid retrieval additionally uses the Apache-2.0
[`Qdrant Python client`](https://github.com/qdrant/qdrant-client) in local mode
and Apache-2.0 [`FastEmbed`](https://github.com/qdrant/fastembed). Model files
come from [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)
(MIT) and
[`Xenova/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2)
(Apache-2.0). Those projects are dependencies, not project authors or endorsers.

This server is for research work that requires searching across many sources,
recovering the passages that support a claim, comparing authors, and citing the
original works. It adds a safe, simple research workflow over
[`vanilla-ultra-rag-mcp-server`](https://github.com/AhmedKishki/vanilla-ultra-rag-mcp-server),
which remains an unmodified gateway to UltraRAG.

Version 0.2 is CPU-only and uses hybrid retrieval: UltraRAG BM25 plus local
FastEmbed vectors in project-local Qdrant, combined with reciprocal-rank fusion.

## Relationship to Vanilla RAG

UltraRAG's official Vanilla RAG pipeline loads benchmark questions, retrieves
passages, renders an UltraRAG prompt, calls an UltraRAG generation backend,
extracts boxed answers, and evaluates them. The vanilla MCP repository exposes
that upstream workflow unchanged.

This server deliberately adapts the architecture for interactive research:

| Vanilla RAG responsibility | Research server adaptation |
|---|---|
| Generic corpus and index paths | One enforced project with PDF/EPUB sources and immutable generations |
| Dense retrieval through a selectable index backend | CPU hybrid retrieval: UltraRAG BM25 plus project-local Qdrant |
| Anonymous passage strings | Structured evidence with source identity, metadata, and page/section locators |
| UltraRAG prompt and generation backend | The connected AI agent reads the evidence and performs synthesis |
| Benchmark answer extraction and scoring | Human-facing quotations, citations, comparison, and source checking |

RAG therefore spans the MCP boundary: `research-ultra-rag-mcp` supplies the
retrieved, citable context and the connected AI agent is the generation stage.
The research server does not currently expose an `answer` tool or call
UltraRAG's generation server internally. This keeps the evidence visible and
lets the user choose the agent/model that writes from it.

## What it provides

- one stdio MCP server named `research-ultra-rag-mcp`;
- one enforced project root per server instance;
- PDF and EPUB ingestion only—Markdown is never indexed;
- PDF page locators and EPUB section locators;
- stable document and chunk identifiers;
- user-reviewed authors, year, DOI, categories, and keywords;
- persistent, immutable knowledge-base generations;
- CPU semantic embeddings stored in project-local Qdrant;
- BM25, dense, and hybrid search modes, with hybrid as the default;
- reciprocal-rank fusion and optional CPU cross-encoder reranking;
- structured search results containing passage text and provenance;
- metadata and document filters; and
- simple tools designed for an AI agent rather than low-level pipeline calls.

## Requirements

- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Internet access during installation and first ingestion

The current implementation is tested on Linux.

## Install

```bash
git clone https://github.com/AhmedKishki/research-ultra-rag-mcp-server.git
cd research-ultra-rag-mcp-server
uv sync --frozen
```

The package pins a tested version of `vanilla-ultra-rag-mcp`, which in turn pins
and verifies UltraRAG `0.3.0.2`. You do not need an UltraRAG clone.

First ingestion downloads the approximately 67 MB FastEmbed ONNX artifact for
`BAAI/bge-small-en-v1.5` from
[`qdrant/bge-small-en-v1.5-onnx-q`](https://huggingface.co/qdrant/bge-small-en-v1.5-onnx-q)
at revision `52398278842ec682c6f32300af41344b1c0b0bb2`. The optional first
`rerank=true` search downloads the approximately 80 MB
`Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder at revision
`a09144355adeed5f58c8ed011d209bf8ee5a1fec`. Both exact revisions remain beneath
the project's `.ultrarag/research/models/` directory and run on CPU.

## Prepare a research project

Keep original sources under the project's `sources` directory:

```text
my-research-project/
├── sources/
│   ├── article.pdf
│   ├── book.epub
│   └── notes.md       # allowed to exist, but deliberately ignored
└── .ultrarag/         # created and managed by this server
```

Only regular `.pdf` and `.epub` files are selected. Other formats and symbolic
links are not ingested.

## Add it to an MCP client

Use one configuration entry per research project. Replace both paths with
absolute paths:

```json
{
  "mcpServers": {
    "research-ultra-rag-mcp": {
      "command": "/ABSOLUTE/PATH/TO/research-ultra-rag-mcp-server/.venv/bin/research-ultra-rag-mcp",
      "args": [
        "--project-root",
        "/ABSOLUTE/PATH/TO/MY-RESEARCH-PROJECT"
      ],
      "env": {},
      "disabled": false,
      "autoApprove": [],
      "timeout": 1800
    }
  }
}
```

A copyable template is available in
[`mcp_settings.example.json`](mcp_settings.example.json).

Keep `autoApprove` empty initially because ingestion writes persistent derived
data and can take time. Running the executable directly produces no prompt; a
stdio MCP server waits for an MCP client.

After the vanilla runtime and embedding model have been cached, add `--offline`
to `args` to prohibit runtime downloads. An offline `rerank=true` search also
requires that this project has previously cached the reranker model.

## Use it through an agent

For the first build, ask:

> Check the research knowledge-base status. If it has not been built, ingest the
> PDF and EPUB sources. Then search for evidence about commodity fetishism and
> artificial intelligence. Return the strongest passages with their source and
> page or section locator.

For later questions, ask normally:

> Search the research knowledge base for sources connecting AI supply chains to
> labour exploitation. Compare the relevant passages and cite every claim.

The server tells the agent to check `status`, use hybrid search by default,
search before answering, quote only returned passage text, cite the returned
provenance, and never invent a page number or bibliographic field. The
agent—not this MCP server—writes the final research response from those results.
See [AGENT_GUIDE.md](AGENT_GUIDE.md) for the complete agent-facing workflow and
[AGENTS.md](AGENTS.md) for the engineering contract.

## MCP tools

| Tool | Purpose |
|---|---|
| `status` | Show selected sources, current generation, and source changes |
| `ingest` | Extract PDFs/EPUBs and atomically select a new BM25 + Qdrant generation |
| `search` | Return BM25, dense, or hybrid evidence with metadata, ranks, and locators |
| `list_sources` | List indexed sources, optionally filtered by metadata |
| `get_passage` | Retrieve a hit with neighboring chunks for context |
| `set_source_metadata` | Save reviewed metadata for one source |

The agent does not need to coordinate UltraRAG's low-level initialization,
chunking, indexing, and reload calls. This server performs that lifecycle.

## What ingestion creates

```text
my-research-project/.ultrarag/research/
├── current.json
├── source-metadata.json
├── logs/
├── models/
├── ultrarag-runtime/
└── generations/
    └── 20260917T120000Z-ab12cd34/
        ├── manifest.json
        ├── corpus/extracted-units.jsonl
        ├── chunks/ultrarag-chunks.jsonl
        ├── chunks/chunks.jsonl
        └── indexes/
            ├── bm25/
            └── qdrant/
```

Each successful ingestion creates a new immutable generation. `current.json`
changes only after extraction, chunking, and indexing all succeed. Previous
generations are retained for inspection and recovery.

The default chunk is 384 GPT-2 tokens with 64-token overlap. The server caps
`chunk_size` at 384 to leave room for title and reviewed metadata within the
embedding model's 512-token input window and reduce silent truncation.

- `manifest.json` records every source, its SHA-256 hash, metadata, and counts.
- `extracted-units.jsonl` stores PDF pages or EPUB sections before chunking.
- `chunks.jsonl` stores quote-safe passage text, retrieval text, and complete
  provenance.
- `indexes/bm25/` contains UltraRAG's persistent lexical-search index.
- `indexes/qdrant/` contains that generation's dense vectors and filter
  payloads. It is embedded local storage; no Qdrant daemon is required.
- `models/` caches the pinned CPU models shared by generations of this project.

The manifest also records empty extraction units and any empty chunk records
discarded from UltraRAG's chunker. Ingestion fails if a selected source or a
non-empty page/section ends up with no searchable chunk.

The original files under `sources/` remain the authoritative sources.

## Search results

A hit contains structured evidence rather than an anonymous string:

```json
{
  "chunk_id": "chk_...",
  "document_id": "doc_...",
  "title": "The Fetishism of AI",
  "authors": ["Author Name"],
  "year": 2024,
  "source_path": "sources/the-fetishism-of-ai.pdf",
  "locator": {
    "type": "pdf_page",
    "page": 12,
    "page_label": "12"
  },
  "citation": "Author Name, The Fetishism of AI (2024), p. 12",
  "text": "The retrieved passage...",
  "retrieval_method": "hybrid",
  "component_ranks": {
    "bm25": 2,
    "dense": 1
  },
  "component_scores": {
    "dense_cosine_similarity": 0.73,
    "bm25": null
  },
  "fusion_score": 0.0325,
  "rerank_score": null
}
```

PDF page numbers are physical PDF pages; `page_label` records the document's
available label. Reflowable EPUBs do not have stable page numbers, so EPUB hits
use section titles, file references, and spine positions.

Only `text` is source passage text suitable for quotation. Ranking fields help
inspect retrieval; they are not confidence scores or evidence that a claim is
true. Dense similarity and reranker scores should not be compared across
different queries. UltraRAG's BM25 API returns ranked passages rather than raw
BM25 scores, so this server reports the BM25 component rank and leaves its raw
score as `null`.

## Retrieval design and why

The default `search` call uses `retrieval_method="hybrid"`:

1. UltraRAG BM25 ranks chunks by exact lexical overlap. It is strong for names,
   quotations, technical terms, and rare phrases.
2. FastEmbed runs the revision-pinned English `BAAI/bge-small-en-v1.5` model on CPU.
   Qdrant ranks its vectors by cosine similarity, which can recover conceptually
   related passages that use different wording.
3. The server combines the two rank lists with reciprocal-rank fusion (RRF,
   `k=60`). RRF uses rank positions because BM25 and cosine values are not on a
   comparable numerical scale.
4. If `rerank=true`, the revision-pinned CPU cross-encoder reranks at most 50 of
   the best candidates. This can improve precision but adds latency, so it is
   opt-in.

Search modes are deliberately visible:

- `hybrid` is the normal research default and usually offers the safest recall;
- `bm25` is useful when exact words, names, citations, or quotations matter; and
- `dense` is useful for inspecting semantic matches independently.

Qdrant was selected over FAISS for this research extension because it persists
metadata payloads and applies category, keyword, and document filters during
dense search while still running in-process with a project-local path. FAISS is
an excellent lightweight vector index and remains available through the vanilla
UltraRAG gateway, but metadata filtering and result bookkeeping would have to be
implemented separately here. Milvus is better suited to a separately operated,
larger-scale service and would add unnecessary infrastructure for the initial
local CPU use case.

The Qdrant integration lives only in this repository. It does not patch
UltraRAG or `vanilla-ultra-rag-mcp-server`, so upstream synchronization remains
safe. UltraRAG still performs corpus chunking and BM25 indexing/search.

Metadata filtering has explicit AND semantics: a hit must contain every
requested category and every requested keyword, and its document ID must be in
the requested document list. BM25 is filtered against canonical chunk metadata;
Qdrant applies the same constraints to its payloads before dense ranking.

Generations created by version 0.1 are still searchable with
`retrieval_method="bm25"`. `status` reports `hybrid_upgrade_required=true` for
them. Run `ingest` once to create a new immutable hybrid generation; the old
generation is retained. Chunk IDs are deterministic for the same extracted
content and chunking configuration, but the version-0.2 tokenization change can
produce new IDs; always use IDs from the current generation with `get_passage`.

## Add reviewed metadata

Ask the agent, for example:

> Set the metadata for `article.pdf`: authors Ahmed Example and Sam Researcher,
> year 2025, category political economy, and keywords AI and labour. Then ingest
> a new generation.

Supported fields are:

```json
{
  "title": "Reviewed title",
  "authors": ["Author One", "Author Two"],
  "year": 2025,
  "doi": "10.example/example",
  "categories": ["political economy"],
  "keywords": ["AI", "labour"]
}
```

Metadata is saved separately and applied when the next generation is ingested.

## Project isolation

Unlike the vanilla gateway, this server does not accept arbitrary corpus or
index paths through its MCP tools. It reads from the configured project
`sources/` directory and writes beneath `.ultrarag/research/`.

Run a separate server entry for each project. Do not point two simultaneously
running server entries at the same project root.

## Current limitations

- The pinned embedding and reranker models are English-oriented.
- CPU embedding and optional reranking are slower than a GPU implementation;
  the first use also downloads the relevant model.
- A changed source currently requires a complete new generation; incremental
  updates are planned.
- Scanned PDFs require OCR before ingestion.
- Password-protected PDFs are rejected.
- Extraction can alter spacing and hyphenation, so important quotations must be
  checked against the original.
- BM25 filtering requires ranking the full local chunk set before filtering;
  Qdrant filters dense candidates before ranking.
- There is no research UI yet.
- Failed and previous generations are retained; automatic cleanup is not yet
  provided.

See [ROADMAP.md](ROADMAP.md) for the next stages.

## Test

Verify a real project from the terminal and create its first generation:

```bash
uv run research-ultra-rag-verify \
  /absolute/path/to/my-research-project \
  --ingest \
  --query "commodity fetishism artificial intelligence"
```

Later checks can omit `--ingest` to search the current generation without
writing a new one. Use `--retrieval-method bm25` or `dense` to inspect one
component, and `--rerank` to exercise the optional cross-encoder. The command
prints the status, ingestion summary, and structured search result as JSON.

Run the repository tests with:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

The test suite includes a real stdio integration test covering PDF selection,
Markdown exclusion, UltraRAG chunking, BM25 indexing, FastEmbed CPU embedding,
project-local Qdrant indexing, hybrid RRF, dense search, and page-aware
retrieval. It also restarts the complete stack offline and repeats a hybrid,
reranked search from cached state. This is an executable system check, not a
claim that every retrieved passage is substantively relevant; retrieval quality
should also be evaluated against representative questions from the actual
research project.

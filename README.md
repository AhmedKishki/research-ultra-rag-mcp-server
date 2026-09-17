# research-ultra-rag-mcp-server

Use an AI agent to build and query a project-owned research knowledge base from
PDF and EPUB sources.

This server is for research work that requires searching across many sources,
recovering the passages that support a claim, comparing authors, and citing the
original works. It adds a safe, simple research workflow over
[`vanilla-ultra-rag-mcp-server`](https://github.com/AhmedKishki/vanilla-ultra-rag-mcp-server),
which remains an unmodified gateway to UltraRAG.

The initial release is CPU-only and uses UltraRAG's BM25 retriever.

## Relationship to Vanilla RAG

UltraRAG's official Vanilla RAG pipeline loads benchmark questions, retrieves
passages, renders an UltraRAG prompt, calls an UltraRAG generation backend,
extracts boxed answers, and evaluates them. The vanilla MCP repository exposes
that upstream workflow unchanged.

This server deliberately adapts the architecture for interactive research:

| Vanilla RAG responsibility | Research server adaptation |
|---|---|
| Generic corpus and index paths | One enforced project with PDF/EPUB sources and immutable generations |
| Dense retrieval by default | Local CPU BM25 retrieval for the initial release |
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
- structured search results containing passage text and provenance;
- metadata and document filters; and
- simple tools designed for an AI agent rather than low-level pipeline calls.

## Requirements

- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Internet access during installation and the first server launch

The current implementation is tested on Linux.

## Install

```bash
git clone https://github.com/AhmedKishki/research-ultra-rag-mcp-server.git
cd research-ultra-rag-mcp-server
uv sync --frozen
```

The package pins a tested version of `vanilla-ultra-rag-mcp`, which in turn pins
and verifies UltraRAG `0.3.0.2`. You do not need an UltraRAG clone.

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

## Use it through an agent

For the first build, ask:

> Check the research knowledge-base status. If it has not been built, ingest the
> PDF and EPUB sources. Then search for evidence about commodity fetishism and
> artificial intelligence. Return the strongest passages with their source and
> page or section locator.

For later questions, ask normally:

> Search the research knowledge base for sources connecting AI supply chains to
> labour exploitation. Compare the relevant passages and cite every claim.

The server tells the agent to check `status`, search before answering, quote only
returned passage text, cite the returned provenance, and never invent a page
number or bibliographic field. The agent—not this MCP server—writes the final
research response from those results.

## MCP tools

| Tool | Purpose |
|---|---|
| `status` | Show selected sources, current generation, and source changes |
| `ingest` | Extract all PDFs/EPUBs and atomically select a new BM25 generation |
| `search` | Return ranked evidence with source metadata and locators |
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
├── ultrarag-runtime/
└── generations/
    └── 20260917T120000Z-ab12cd34/
        ├── manifest.json
        ├── corpus/extracted-units.jsonl
        ├── chunks/ultrarag-chunks.jsonl
        ├── chunks/chunks.jsonl
        └── indexes/bm25/
```

Each successful ingestion creates a new immutable generation. `current.json`
changes only after extraction, chunking, and indexing all succeed. Previous
generations are retained for inspection and recovery.

- `manifest.json` records every source, its SHA-256 hash, metadata, and counts.
- `extracted-units.jsonl` stores PDF pages or EPUB sections before chunking.
- `chunks.jsonl` stores the searchable text and its complete provenance.
- `indexes/bm25/` contains UltraRAG's persistent lexical-search index.

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
  "retrieval_method": "bm25"
}
```

PDF page numbers are physical PDF pages; `page_label` records the document's
available label. Reflowable EPUBs do not have stable page numbers, so EPUB hits
use section titles, file references, and spine positions.

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

- Retrieval is lexical CPU BM25, not dense or hybrid retrieval.
- A changed source currently requires a complete new generation; incremental
  updates are planned.
- Scanned PDFs require OCR before ingestion.
- Password-protected PDFs are rejected.
- Extraction can alter spacing and hyphenation, so important quotations must be
  checked against the original.
- Metadata filters are applied to BM25 results after ranking.
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
writing a new one. The command prints the status, ingestion summary, and
structured search result as JSON.

Run the repository tests with:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

The test suite includes a real stdio integration test covering PDF selection,
Markdown exclusion, UltraRAG chunking, BM25 indexing, and page-aware retrieval.

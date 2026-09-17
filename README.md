# research-ultra-rag-mcp-server

Build and search a project-local research knowledge base from PDF and EPUB
sources through an AI agent.

The server is designed for evidence-based research: it retrieves passages from
multiple works, preserves source metadata and page or section locators, and
returns structured results that an agent can quote and cite.

## Built on UltraRAG

This project uses the corpus chunking and BM25 retrieval provided by
[`OpenBMB/UltraRAG`](https://github.com/OpenBMB/UltraRAG), pinned to version
`0.3.0.2` at commit
[`3a709a2`](https://github.com/OpenBMB/UltraRAG/tree/3a709a2aea3fbe46acca59c422621c94b6e86857).
UltraRAG is a joint project of THUNLP, NEUIR, OpenBMB, AI9stars, and its
contributors, and is licensed under Apache-2.0.

This is an independent project and is not an official UltraRAG release. See
[`NOTICE`](NOTICE) for complete attribution and dependency licenses.

## What the server offers

- one stdio MCP server named `research-ultra-rag-mcp`;
- one isolated knowledge base per configured project root;
- enforced PDF and EPUB ingestion—Markdown and other formats are ignored;
- PDF page locators and EPUB section locators;
- readable passage text with extraction-related line wrapping removed;
- reviewed title, author, year, DOI, category, and keyword metadata;
- reversible, agent-reviewed source exclusion without deleting original files;
- immutable knowledge-base generations with stable source provenance;
- CPU BM25, dense, and hybrid retrieval;
- project-local Qdrant storage with no database service to run;
- optional CPU cross-encoder reranking; and
- seven high-level tools intended for direct use by an AI agent; and
- a local research UI backed by those same MCP tools and project data.

The server retrieves evidence but does not generate a final answer. The
connected AI agent compares the returned passages and writes the response.

## How it works

```text
PDF/EPUB files
    │
    ├─ omit sources explicitly excluded by the agent/user
    ├─ extract text by PDF page or EPUB section
    ├─ split text into overlapping chunks with UltraRAG
    ├─ build an UltraRAG BM25 lexical index
    ├─ build FastEmbed vectors in local Qdrant
    └─ save provenance and reviewed metadata
             │
             ▼
       hybrid search with RRF
             │
             ▼
 citable passages returned through MCP
```

Hybrid search is the default. BM25 finds exact terms and distinctive phrases;
dense retrieval finds conceptually related wording. Reciprocal-rank fusion
(RRF) combines the independent rankings without treating their incompatible
raw scores as equivalent. Optional reranking applies a cross-encoder to a
bounded candidate set.

## Requirements

- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Linux for the currently tested setup
- Internet access during installation and first ingestion

## Install

```bash
git clone https://github.com/AhmedKishki/research-ultra-rag-mcp-server.git
cd research-ultra-rag-mcp-server
uv sync --frozen
```

No separate UltraRAG checkout or Qdrant service is required. The first
ingestion downloads an approximately 67 MB CPU embedding model. The first
search with `rerank=true` downloads an additional approximately 80 MB reranker.
Exact model revisions are recorded in each generation manifest.

## Prepare a project

Put original documents in a `sources` directory:

```text
my-research-project/
├── sources/
│   ├── article.pdf
│   ├── book.epub
│   └── notes.md       # permitted, but never indexed
└── .ultrarag/         # created automatically
```

Files are discovered recursively. Symbolic links and every extension other
than `.pdf` and `.epub` are excluded.

## Launch the research UI

From this repository, point the UI at the same project root used by your MCP
client:

```bash
uv run research-ultra-rag-ui \
  --project-root /absolute/path/to/my-research-project
```

Open [http://127.0.0.1:5051](http://127.0.0.1:5051). The UI immediately reads
the project's current generation; it does not create a second knowledge base
and does not require re-ingestion.

The basic browser workspace is supplied by the separately versioned
[`ui-ultra-rag-mcp`](https://github.com/AhmedKishki/ui-ultra-rag-mcp) package.
It is installed automatically at the revision pinned by this repository; you
do not need to clone or start it separately. This project retains the research
adapter, MCP subprocess lifecycle, and source-file safety policy.

The UI can:

- inspect readiness, staleness, generation identity, and source counts;
- search with hybrid, BM25, or dense retrieval and optional CPU reranking;
- copy passages and citations, inspect neighboring passages, and open the
  original PDF or download the original EPUB;
- browse and filter indexed sources;
- edit reviewed metadata and explicitly exclude or restore a source; and
- create a new immutable generation.

It binds to the local machine only. The browser talks to a private stdio
instance of this MCP server, so the UI and an AI agent use the same seven tools
and receive the same results. The UI is an evidence workspace, not a chatbot:
it does not generate an answer or treat retrieval scores as truth.

You may keep the UI open while the configured MCP server is running in Cline.
Project operations are serialized with a lock; if one process is ingesting,
searches and writes from the other wait for it to finish.

## Configure your MCP client

Replace both paths with absolute paths:

```json
{
  "mcpServers": {
    "research-ultra-rag-mcp": {
      "command": "/ABSOLUTE/PATH/TO/research-ultra-rag-mcp-server/.venv/bin/research-ultra-rag-mcp",
      "args": [
        "--project-root",
        "/ABSOLUTE/PATH/TO/my-research-project"
      ],
      "env": {},
      "disabled": false,
      "autoApprove": [],
      "timeout": 1800
    }
  }
}
```

A copyable configuration is provided in
[`mcp_settings.example.json`](mcp_settings.example.json). Keep `autoApprove`
empty initially because ingestion creates persistent data and may take time.

The executable is a stdio server. Running it directly shows no interactive
prompt because it waits for an MCP client.

## Use it

For the first ingestion, ask your agent:

> Check the research knowledge-base status. If no generation exists, ingest the
> PDF and EPUB sources. Then search for evidence about commodity fetishism and
> artificial intelligence. Return the strongest passages with their source and
> page or section locator.

For later research:

> Search the knowledge base for sources connecting AI supply chains to labour
> exploitation. Compare the relevant passages and cite every claim.

To add reviewed metadata:

> Set the metadata for `article.pdf`: authors Ahmed Example and Sam Researcher,
> year 2025, category political economy, and keywords AI and labour. Then ingest
> a new generation.

Metadata changes take effect after the next ingestion.

If search reveals that two files represent the same source, ask the agent to
retain the preferred copy and exclude the other one:

> Exclude `duplicate-copy.pdf` from the knowledge base because it duplicates
> `preferred-copy.pdf`. Do not delete either source file.

The exclusion affects `search`, `list_sources`, and `get_passage` immediately.
It is recorded for future ingestions and can be reversed by setting the source
back to included. Run `ingest` afterward when you want a new immutable
generation whose stored indexes no longer contain the excluded source.

## MCP tools

| Tool | Purpose |
|---|---|
| `status` | Report whether a generation exists and whether sources changed |
| `ingest` | Extract all PDFs/EPUBs and atomically create new BM25 and Qdrant indexes |
| `search` | Return ranked evidence using hybrid, BM25, or dense retrieval |
| `list_sources` | List indexed sources and reviewed metadata |
| `get_passage` | Return a passage with neighboring chunks for context |
| `set_source_metadata` | Save reviewed metadata for the next generation |
| `set_source_inclusion` | Exclude or restore one source without changing its file |

`search` accepts optional `categories`, `keywords`, and `document_ids` filters.
Every requested category and keyword must be present. A result must match one
of the supplied document IDs.

## Expected search result

The following is representative; scores and identifiers vary by corpus and
query:

```json
{
  "query": "commodity fetishism artificial intelligence",
  "generation_id": "20260917T120000Z-ab12cd34",
  "retrieval_method": "hybrid",
  "reranked": false,
  "result_count": 1,
  "hits": [
    {
      "rank": 1,
      "chunk_id": "chk_7e1a...",
      "document_id": "doc_a812...",
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
      "text": "The retrieved source passage...",
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
  ]
}
```

Only `text` is source passage text suitable for quotation. The locator and
citation identify where to verify it in the original. Ranking scores are not
truth probabilities and should not be compared across different queries.

PDF locators use physical PDF pages and available page labels. Reflowable EPUBs
use section titles, file references, and spine positions because they do not
have stable page numbers.

Layout line breaks introduced by PDF or EPUB extraction are removed from
returned passages. Existing generations benefit at response time. A new
ingestion stores the normalized text in the generation itself. This does not
perform OCR or rewrite source wording, so important quotations still need to be
checked against the original document.

## Choose a retrieval mode

- `hybrid` is the default and is recommended for ordinary research.
- `bm25` emphasizes exact words, names, quotations, and technical terms.
- `dense` emphasizes conceptual similarity and related wording.
- `rerank=true` can improve final ordering but is slower on CPU.

The embedding model is `BAAI/bge-small-en-v1.5`. The optional reranker is
`Xenova/ms-marco-MiniLM-L-6-v2`. Both are revision-pinned and English-oriented.

## Project storage

All derived knowledge-base data is stored beneath the configured project:

```text
my-research-project/.ultrarag/research/
├── current.json
├── project.lock
├── source-metadata.json
├── source-exclusions.json
├── models/
├── logs/
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
changes only after extraction and both indexes succeed. Previous and failed
generations are retained. Original files under `sources/` are never modified.
`source-exclusions.json` records reversible source-level decisions separately
from the immutable generations.

Use a separate server configuration for each project. Multiple local processes,
such as Cline and the research UI, may use the same project root because access
is serialized through `project.lock`. A long ingestion blocks other operations
for that project until it completes.

Generations created by version 0.1 remain searchable with
`retrieval_method="bm25"`. When `status` reports
`hybrid_upgrade_required=true`, run `ingest` to create a hybrid generation.

## Verify from the terminal

Create a generation and run a real MCP search:

```bash
uv run research-ultra-rag-verify \
  /absolute/path/to/my-research-project \
  --ingest \
  --query "commodity fetishism artificial intelligence"
```

Later checks can omit `--ingest`. Use `--retrieval-method bm25` or `dense` to
inspect one component, and `--rerank` to exercise the optional reranker. The
command prints status, ingestion details, and search results as JSON.

After the runtime and models are cached, add `--offline` to the MCP server
arguments or verifier command to prohibit downloads.

## Current limitations

- Scanned PDFs require OCR before ingestion.
- Password-protected PDFs are rejected.
- Extraction may alter spacing or hyphenation; verify important quotations in
  the original source.
- The embedding and reranker models are English-oriented.
- Source changes currently require a complete new generation.
- Duplicate identification is intentionally left to the agent and user; the
  server does not guess whether similar files are the same source.
- CPU embedding and reranking are slower than GPU-backed alternatives.
- There is no automatic generation cleanup yet.

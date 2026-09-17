# AGENTS.md

This is the engineering guide for AI coding agents working in
`research-ultra-rag-mcp-server`.

## Objective

Provide a high-level stdio MCP server for project-scoped research knowledge
bases built from original PDF and EPUB sources. The server helps an AI agent
retrieve citable evidence across a collection while preserving document
identity and source locators.

The package builds on the separately versioned
`vanilla-ultra-rag-mcp-server`. Never add research behavior to the vanilla
repository to support this project.

## Current compatibility baseline

- Package and command: `research-ultra-rag-mcp`
- Version: `0.1.0`
- Python: `>=3.11,<3.13`
- FastMCP: `3.4.0`
- Vanilla gateway commit: `404839c10aef254a49380695fa3b2bb9c2b1c95f`
- Upstream UltraRAG: `0.3.0.2` at
  `3a709a2aea3fbe46acca59c422621c94b6e86857`

## Non-negotiable research contract

- One server process serves exactly one configured project root.
- MCP tools must not accept arbitrary filesystem output paths.
- Ingestion selects only regular PDF and EPUB files beneath the configured
  sources directory.
- Reject source symlinks and path traversal.
- Store all derived state beneath `<project>/.ultrarag/research`.
- Never edit or write the original source documents.
- Do not switch `current.json` until a generation is completely indexed.
- Preserve stable document IDs, chunk IDs, source paths, and locators.
- Search results must separate passage text from metadata so an agent knows what
  may be quoted.
- Never claim that extracted text is a substitute for checking the original.
- Keep MCP stdout reserved for protocol messages.

## Architecture

```text
AI agent / MCP client
        │ stdio
        ▼
research-ultra-rag-mcp
        ├── project/source policy
        ├── PDF page extraction
        ├── EPUB section extraction
        ├── metadata and provenance
        ├── immutable generations
        │
        ▼ persistent stdio MCP
vanilla-ultra-rag-mcp
        ├── UltraRAG corpus chunker
        └── UltraRAG BM25 retriever
```

Only the high-level research tools are exposed to the outer MCP client. The
vanilla gateway is an implementation dependency, not a second user-facing tool
surface within this server.

## Repository map

- `src/research_ultra_rag_mcp/server.py`: CLI, MCP lifecycle, and six public
  tools.
- `config.py`: project boundary and executable validation.
- `sources.py`: allowlist, source discovery, hashing, and metadata validation.
- `extraction.py`: PDF pages and EPUB sections.
- `storage.py`: atomic JSON state and JSONL artifacts.
- `ultrarag.py`: persistent client for vanilla UltraRAG tools.
- `service.py`: generation, indexing, status, filtering, and evidence workflow.
- `instructions.py`: guidance returned to MCP agents.
- `verify.py`: terminal MCP client for end-to-end project verification.
- `tests/`: unit and real stdio integration coverage.
- `ROADMAP.md`: explicitly deferred work.

## Storage model

The mutable pointer and reviewed metadata live at:

```text
<project>/.ultrarag/research/current.json
<project>/.ultrarag/research/source-metadata.json
```

Each build gets a unique directory under `generations/`. A failed generation
gets `failure.json` and never becomes current. Successful generations contain a
manifest, extraction units, raw UltraRAG chunks, enriched chunks, and BM25
index. Do not treat generated files as source documents.

## Public MCP tools

- `status`: read-only source/current/staleness inspection.
- `ingest`: create and select a complete new generation.
- `search`: BM25 retrieval with structured evidence.
- `list_sources`: inspect indexed documents and metadata.
- `get_passage`: retrieve neighboring chunks from the same document.
- `set_source_metadata`: update reviewed metadata for the next generation.

Tool docstrings and `SERVER_INSTRUCTIONS` are part of the agent-facing contract.
Update tests and documentation when changing them.

## Safe change rules

- Use `pathlib.Path`, type hints, and JSON-serializable tool results.
- Keep blocking extraction and filesystem scans outside the event loop.
- Serialize ingest/search/metadata operations with the service lock.
- Prefer new immutable generations to in-place index mutation.
- Validate a new artifact before updating a pointer to it.
- Do not expose underlying vanilla tools through this server.
- Do not silently skip a selected PDF/EPUB that fails extraction; fail the new
  generation and leave the previous current generation intact.
- Empty upstream chunk records may be discarded only when their extraction unit
  still has at least one searchable chunk; record the discarded count.
- Retain explicit limitations when a feature is not implemented.

## Validation

Before finalizing a change, run:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q src tests
```

For source or retrieval changes, the integration test must still launch the
real vanilla stdio server, build a BM25 index, retrieve the known passage, and
prove that a neighboring Markdown file was excluded.

For significant extraction changes, also test a representative real collection
without writing into its source directory.

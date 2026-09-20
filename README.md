## Purpose

This server lets an AI agent search a personal collection of PDF and EPUB files and hand back the passages that matter, with enough provenance to check them.

It is built for research work where you need to find relationships between sources, compare authors, or locate passages that support, qualify, or contradict a claim. It returns **evidence candidates**, not conclusions. You and your agent stay responsible for reading, judging, and writing.

> **Retrieved text is cleaned semantic text, not a quote-safe transcript.** Open the original PDF or EPUB at the returned locator before using a direct quotation.

The server is built on [UltraRAG](https://github.com/OpenBMB/UltraRAG), whose corpus chunking and BM25 retrieval it uses. Full credit and licensing details appear at the end of this document.

## What the server provides

Plain-language summary of every capability, and why it is there:

- **An MCP server for AI agents, plus a local browser UI.** Agents call tools over stdio; you can also click through the same features in a browser. Both operate on the same project state.
- **Ingests regular `.pdf` and `.epub` files only.** It walks the source directory recursively. Markdown and every other format are ignored, so a stray `notes.md` in the folder never enters the knowledge base.
- **Resolves bibliographic metadata and locators.** Titles, authors, years, DOIs, per-field provenance and warnings, and a locator that points back into the original file (PDF page, or EPUB section). You can always find the page behind a passage.
- **Four ways to search.** BM25 lexical search for exact names and phrases, dense semantic search for meaning, hybrid search (the default) for normal use, and an optional CPU reranker that reorders a candidate set. Metadata filters narrow results by category, keyword, or document. A freshness check can be skipped per query when a caller only needs evidence.
- **Optional source-diverse results.** `result_view="references"` caps how many passages any single source can contribute, so one book chapter cannot fill the whole answer.
- **Reviewed metadata that applies immediately.** Fix a wrong author or year and the change shows up in listings, citations, filters, and results at once — without re-ingesting or rewriting the generation.
- **Reversible source inclusion.** Exclude a duplicate source, later restore it. The original file is never deleted or modified.
- **Portable project bundles.** Export a project (PDFs, review state, and the cleaned artifacts) as one archive and import it on another machine.
- **Resumable ingestion.** Every call has a soft time budget. If it runs out, the server returns a checkpointed `in_progress` result and you simply call it again. Client timeouts, cancellations, and restarts lose at most one small batch of work.
- **Text-health policy with disclosure.** Corrupt text (broken character maps) and chunks with no readable letters or digits are excluded, and the response tells you why and where. A quotation in another language or script is *not* excluded — it is returned with an advisory `text_notes` entry instead.
- **An embedding-fidelity audit.** Each chunk records how many embedding tokens it contains and whether its vector covers only the beginning of its text, so silent truncation is visible per hit and in aggregate.
- **Immutable, project-local generations.** A build creates a new generation and only becomes active after both indexes pass validation. A failed build leaves the previous generation searchable.
- **Cheap updates.** Adding or changing a source reuses the unchanged documents, chunks, and vectors, then rebuilds only the indexes. Dense search reads the generation's portable vectors directly rather than maintaining a separate index, so there is nothing to rebuild or keep in sync.

For the full inventory — which capabilities come from UltraRAG, which are added on top of it, what is deliberately excluded, what is only planned, and how this server compares with a general-purpose MCP RAG server — see [`FEATURES.md`](FEATURES.md).

It does not decide what is true, delete or exclude sources on its own, generate answers on the server, or provide exact quotation transcripts. Its job is to return structured evidence for an agent and researcher to assess.

## Install once

Requirements are Python 3.11 or 3.12, Linux for the currently tested setup, and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/AhmedKishki/research-ultra-rag-mcp-server.git
cd research-ultra-rag-mcp-server
uv sync --frozen
```

There is nothing else to install: no separate UltraRAG checkout, no external vector database service, and no per-project virtual environment. The single `.venv` in this repository serves every project.

The pinned UltraRAG runtime is installed automatically. The embedding model is downloaded on first ingestion, and the optional reranker on its first use. Model binaries are shared at `~/.cache/research-ultra-rag-mcp/models`, while all document data stays inside each project.

Useful settings, all optional:

| Setting | What it does |
|---|---|
| `--model-cache-root` or `RESEARCH_ULTRARAG_MODEL_CACHE_ROOT` | Move the shared model cache elsewhere. |
| `--dense-backend auto\|exact\|qdrant` or `RESEARCH_ULTRARAG_DENSE_BACKEND` | Choose how dense search is stored. `auto` (default) scans the portable vectors directly for normal-sized corpora and switches to an embedded index for very large ones. |
| `--embedding-threads` or `RESEARCH_ULTRARAG_EMBEDDING_THREADS` | Set the CPU thread count for embedding. Left unset by default because the best value depends on your machine. |
| `--runtime-root` or `RESEARCH_ULTRARAG_RUNTIME_ROOT` | Keep the working files (indexes, staging, logs) on a different disk. See "Where project data is stored". |
| `--offline` | Fail instead of downloading anything. Use it once the runtime and models are cached. |

## Create an isolated research project

A project is just a directory. Put your originals in a `sources/` subdirectory:

```text
my-research-project/
└── sources/
    ├── article-one.pdf
    └── book-two.epub
```

Everything the server creates lives in a `.research-rag/` folder next to it, so one directory holds one research project and nothing is shared between projects except the downloaded model files.

## Connect an AI agent

Add the server to your MCP client configuration. The `"mcpServers"` entry below is a complete example; replace the absolute paths with your own:

```json
{
  "mcpServers": {
    "research-ultra-rag": {
      "command": "/ABSOLUTE/PATH/research-ultra-rag-mcp-server/.venv/bin/research-ultra-rag-mcp",
      "args": [
        "--project-root",
        "/ABSOLUTE/PATH/my-research-project"
      ],
      "env": {},
      "disabled": false,
      "autoApprove": [],
      "timeout": 1800
    }
  }
}
```

The same installation can serve two isolated projects, which is why `--project-root` is part of the server entry rather than a per-call argument:

```json
{
  "mcpServers": {
    "research-history": {
      "command": "/ABSOLUTE/PATH/research-ultra-rag-mcp-server/.venv/bin/research-ultra-rag-mcp",
      "args": ["--project-root", "/data/projects/history"]
    },
    "research-philosophy": {
      "command": "/ABSOLUTE/PATH/research-ultra-rag-mcp-server/.venv/bin/research-ultra-rag-mcp",
      "args": ["--project-root", "/data/projects/philosophy"]
    }
  }
}
```

A ready-to-copy template is in [`mcp_settings.example.json`](mcp_settings.example.json).

Two practical notes:

- Running the executable by hand looks like nothing happens. That is correct — it is a stdio server waiting for an MCP client to send protocol messages.
- Keep the write tools out of automatic approval at first. Ingestion, metadata decisions, exclusions, exports, and imports either persist state or create large files, so they are worth confirming once.

## First use

The normal sequence an agent should follow:

1. Call `status`.
2. Report whether no generation exists yet, the selected generation is stale, or a schema/policy upgrade is required.
3. Ask permission before a persistent ingestion when that is appropriate.
4. Call `ingest`. While it returns `status="in_progress"`, call it again with the same settings.
5. Search only once the completed generation is active.

A prompt you can copy:

> Check the research knowledge-base status. Tell me whether a generation exists, is stale, or needs an upgrade. If ingestion is needed, explain what will be written and ask before doing it. After it succeeds, search the corpus for evidence about [YOUR QUESTION], including material that qualifies or contradicts the claim, and give me the original source paths and locators.

What to expect from a first build:

- It may download the embedding model and can take minutes on a CPU. Time depends mainly on corpus size, because embedding every chunk is the slow part.
- Work happens in project-local staging. Each call has a soft budget of `work_budget_seconds=45`; when it expires the call returns a checkpointed `in_progress` result and you repeat it. One expensive page, the first model download, or BM25 finalization can exceed that soft budget.
- `current.json` is switched only after the complete BM25 and dense indexes verify. A cancellation or timeout keeps the last checkpoint. A failure that cannot be resumed leaves the previous generation selected, removes the partial staging data, and writes a small failure record.

### Finding and referring to sources

`list_sources` works before any ingestion. Its `discovered_sources` array lists every live PDF/EPUB with a project-scoped `source_id`, whether it is included, and whether it is indexed in the selected generation.

Two identifiers matter, and they mean different things:

- A `source_id` is derived from the project ID and the source-relative path. It survives changes to the file's contents, but renaming or moving the file creates a new `source_id`. Use it for `set_source_metadata` and `set_source_inclusion`.
- A `document_id` identifies one content/path version inside a generation and changes when the bytes or the path change.

Calling `list_sources` also registers discovered IDs in the portable project catalog. That is what lets the lean `known_sources` array keep an ID-to-path handle when an original is renamed, moved, or temporarily absent — without guessing that metadata from an old path belongs to a new one. `reviewed_metadata_sources` lists every saved override by the same stable ID, so you can inspect, replace, or remove each persisted decision even if the original file is not present right now.

Both mutation tools require exactly one selector: `source_id` is preferred, and `source_path` remains available for compatibility.

## Research workflow

Prompts that work well:

- "Find the strongest source passages supporting the claim that …"
- "Search broadly across references; return at most two passages from any one source within the total passage budget."
- "Search for evidence that contradicts or qualifies this claim: …"
- "Compare how these sources explain …; distinguish agreement from conflict."
- "Get the neighboring passages around chunk `chk_…` before interpreting it."
- "Open the original at the returned path and page before quoting it."
- "List sources with missing metadata or extraction warnings. Treat automatic metadata as provisional; show me its provenance, then use the source ID with `set_source_metadata` after I review the original. Apply the correction now without re-ingesting."
- "List the source IDs, then exclude the duplicate source ID as a reviewed duplicate of the preferred source ID; do not delete either file."
- "Restore that source ID, then re-ingest if it is absent from the current generation."

### Choosing a search mode

Hybrid is the normal choice. Use the others to look at one signal at a time:

- **BM25** is lexical: it matches the words you typed. Best for exact names, terms, and phrases, and useful when a semantic result surprises you.
- **Dense** is semantic: it matches meaning. Useful for concepts phrased differently from the sources.
- **Hybrid** combines both rankings.
- **Reranking** is optional and slower. It reorders a candidate set that already looks plausible, so it is not needed for routine lookups.

Search can legitimately return fewer results than `top_k`, including none, when candidates fail the relevance gates — abstaining is a feature, not an error. A rank or similarity score is an ordering signal, never a truth or confidence probability.

### Passage view versus reference view

`result_view="passages"` (the default) is the plain global ranking.

`result_view="references"` is for when one prolific source would otherwise dominate. It walks the same relevance-gated candidates but admits at most `passages_per_reference` passages from each stable `source_id`, keeping `top_k` as the total number of passages. The response reports how many references it returned and how many were available. It does not merge editions by title, DOI, or filename. Two flags tell you why a result set may look short: `relevance_limited` means the candidate pool ran short, and `grouping_limited` means the per-reference cap stopped the view from filling its passage budget.

### Checking freshness per query

By default a search also reports whether the selected generation is stale, which means comparing every source file with what the generation recorded. That comparison costs about 10 ms for a 55-source project and grows with the collection, so it is the one part of a search whose cost depends on how many files you have rather than on the question asked. At this size it is a small share of a query, so the flag removes a cost that scales rather than a latency you feel today.

Pass `include_staleness=false` when a session has already checked `status` and only needs evidence. The response then reports `stale=null` and `staleness_checked=false`, which means "not checked", not "fresh". Everything else about the search is unchanged, including which hits are returned.

### What makes a generation stale

Adding, removing, or changing source content — or changing which sources are included — can make the selected generation stale, and `status` says so. When that happens you have two options:

- **Re-ingest changes** verifies every source hash and reuses compatible documents, chunks, and vectors, then rebuilds both indexes. This is the normal path and it is much cheaper than starting over.
- **Regenerate** uses `--force-recompute` and deliberately ignores all reuse.

Reviewed metadata is different from staleness. For a source already in the selected generation, `set_source_metadata` updates listings, filters, results, citations, and neighboring passages immediately (`effective_immediately`). It does not rewrite the immutable generation and does not change chunk IDs. A metadata-only difference is reported through the metadata overlay and snapshot fields (`metadata_overlay_active`) rather than as stale retrieval state. If you set metadata for a source that is missing from the selected generation, the decision is saved and takes effect after the next ingestion.

### What gets excluded, and what does not

Extraction rejects a whole page or section only when its text carries strong evidence of a broken character map: replacement characters, private-use or unassigned code points, or a known damaged encoding sequence. The diagnostics keep only the locator and reason codes, never the garbage text. If nothing readable remains in a source, the build fails so you can repair or OCR the file.

Two normalisation rules exist so that typed queries match printed text: formula-font letters (`𝑀` becomes `M`) and the presentation ligatures `ﬁ`, `ﬂ`, and `ﬀ` are folded to their plain spellings. Accented letters, superscripts, subscripts, and symbols are left untouched, because they carry meaning in citations and notation.

Script mixing and non-Latin dominance are never rejection reasons. A quotation in another language stays retrievable and is reported with advisory `text_notes` instead. Reviewed metadata is never overridden by this classifier.

Finally, ingestion drops nonempty chunks that contain no alphanumeric content at all. Ordinary prose, numbers, and formulas containing at least one letter or digit are not affected.

## Use the UI

From the installed server repository:

```bash
uv run research-ultra-rag-ui \
  --project-root /absolute/path/to/my-research-project
```

Open [http://127.0.0.1:5051](http://127.0.0.1:5051) if a browser does not open by itself. The UI binds to loopback only and calls the same nine public MCP tools against the same project state as an agent.

What you can do in it:

- inspect status, staleness, upgrade state, and build metrics;
- browse the indexed source list and resolved bibliography;
- run hybrid, BM25, or dense search with filters and optional reranking;
- read neighboring passages around a hit;
- open a PDF in the browser, or download an EPUB original;
- edit reviewed metadata and see it apply immediately;
- exclude or restore a source without deleting anything;
- **Create generation** for the first normal build;
- **Re-ingest changes** for verified reuse after the project changes;
- **Regenerate** to force extraction, chunking, and embedding again;
- export a bundle; and
- import a bundle already placed in `.research-rag/bundles/`.

One UI process serves one project; to keep two open at once, start a second process with another project root and port, for example `--port 5052`. During ingestion the UI shows the operation as busy. A project lock serialises agent and UI operations, so a second request waits instead of reading a half-built index. The UI follows checkpointed `in_progress` responses automatically until the generation is ready or nothing changed.

## Use the terminal verifier

Read-only status and search check:

```bash
uv run research-ultra-rag-verify \
  /absolute/path/to/my-research-project \
  --query "commodity fetishism and artificial intelligence"
```

First ingestion or re-ingestion:

```bash
uv run research-ultra-rag-verify \
  /absolute/path/to/my-research-project \
  --ingest \
  --query "commodity fetishism and artificial intelligence"
```

Forced recomputation:

```bash
uv run research-ultra-rag-verify \
  /absolute/path/to/my-research-project \
  --ingest --force-recompute
```

Useful flags: `--retrieval-method bm25|dense|hybrid`, `--rerank`, `--top-k`, and `--result-view references --passages-per-reference 2` for source-diverse results. Add `--offline` when every required cache already exists.

On success it prints a JSON object with `"status": "passed"`, the before/after status, optional ingestion metrics, and the search result. With `--ingest` it repeats checkpointed calls until ingestion finishes. The usual first-run failures are a missing network or model download, an unsupported Python version, a damaged PDF/EPUB, or `--offline` before the runtime and models are cached.

## Where project data is stored

```text
my-research-project/
├── sources/                              untouched PDF/EPUB originals
└── .research-rag/                        all research-RAG project state
    ├── project.json                      stable ID, name, source setting
    ├── source-catalog.json               durable source ID-to-path registry
    ├── source-metadata.json              authoritative reviewed metadata overlay
    ├── source-exclusions.json            reviewed decisions, when present
    ├── bundles/                          exported/import-ready archives
    └── runtime/                          disposable derived state
        ├── current.json                  selected generation pointer
        ├── project.lock
        ├── logs/
        ├── failures/                     small failed-build records
        ├── staging/<build-id>/           resumable incomplete build + checkpoint
        ├── ultrarag-runtime/
        └── generations/<generation-id>/
            ├── manifest.json
            ├── corpus/extracted-units.jsonl
            ├── chunks/chunks.jsonl
            ├── portable/embeddings.npy
            └── indexes/
                ├── artifact-lookup.sqlite3
                ├── bm25/
                └── qdrant/

~/.cache/research-ultra-rag-mcp/models/  shared model binaries only
```

How to read that tree:

- `sources/` is the authority for exact quotation. Nothing in this server edits it.
- Inside `.research-rag/`, the top-level JSON files and `bundles/` are **portable review state**: your decisions about the project. They are the part worth backing up.
- `runtime/` is **derived state**. It can be rebuilt from `sources/` plus the portable state, so it is safe to delete if you are willing to rebuild.
- Document text, embeddings, indexes, logs, and query state never cross project roots. Only immutable model binaries are shared.

On first use after upgrading, an existing `.ultrarag/research/` directory is moved automatically to `.research-rag/runtime/`. If both locations already hold runtime data, startup stops rather than guessing. Other `.ultrarag/` content belonging to different tools is left alone. Stop every research MCP and UI process before the first launch of the upgraded package.

`current.json` names the one generation search uses. Earlier successful generations stay on disk but are not searched, and automatic pruning is not implemented.

### Put derived state on fast local storage

If your project lives on a slow disk, point the working files at a fast one. Embedding work lives on the CPU, but staging, vectors, and index writes are disk-bound, so this is often the single biggest speed-up available.

```bash
research-ultra-rag-mcp \
  --project-root /mnt/data/projects/ai-and-fetishism \
  --runtime-root /ssd/research-runtime/ai-and-fetishism
```

Set `RESEARCH_ULTRARAG_RUNTIME_ROOT` instead if you prefer a variable. The MCP server, UI, and verifier all honour it.

Rules that keep this safe:

- The path must be absolute.
- The first run claims an empty directory by writing a small marker file naming this project. Later runs check that marker, so a root belonging to another project, a non-empty directory without a marker, and a path that is actually a file are all refused with an explicit message instead of silently mixing two projects together.
- Only derived state moves. Your portable review state stays in `<project>/.research-rag`.
- `status` reports the effective location as `runtime_root` (`null` when the default is in use). To move back, drop the option and relocate the directory.

A bind mount still works if you prefer the path to stay literally inside the project: stop every research MCP, UI, and verifier process, copy `runtime/` to the fast device, then `mount --bind` it at `<project>/.research-rag/runtime` and add that to `/etc/fstab`. The trade-off is that the server cannot detect a bind mount and cannot warn you when a moved project loses it. The two approaches are mutually exclusive; `--runtime-root` is the portable one.

## Export, import, and move a project

An agent can call `export_bundle` and `import_bundle`, and the UI has matching buttons. From a terminal:

```bash
uv run research-ultra-rag-bundle export \
  --project-root /absolute/path/to/my-research-project
```

```bash
uv run research-ultra-rag-bundle import \
  --project-root /absolute/path/to/my-research-project \
  bundle-name.research-rag.zip
```

Export writes into `.research-rag/bundles/`. The terminal command accepts an archive from anywhere, copies it safely into that directory, and then invokes the same import an agent or the UI would use.

A bundle contains every original PDF/EPUB (including excluded ones), the project descriptor, your reviewed metadata and exclusions, the generation manifest, cleaned semantic units and chunks, and the float32 embeddings in stable chunk order. It deliberately excludes live BM25 and dense index directories, locks, logs, temporary and runtime files, and model caches — those are rebuilt.

Import validates archive paths and entry types, checksums, project ID, schemas, embedding compatibility, and every original. It refuses to overwrite an existing path with different bytes. It rebuilds both indexes without re-extracting or re-embedding, and it moves `current.json` last, only when you asked it to activate the imported generation.

**You are responsible for having the right to redistribute** every PDF and EPUB you put in a bundle.

## Complete MCP tool reference

Nine tools are exposed. All are project-scoped and none of them deletes a source file.

| Tool | What it does |
|---|---|
| `status` | Reports readiness, staleness, upgrade requirements, counts, review-state revisions, and build metrics. Read-only. |
| `ingest` | Creates or refreshes a generation. Resumable, with a soft per-call work budget. |
| `search` | Retrieves evidence candidates. Supports BM25, dense, and hybrid retrieval, metadata filters, optional reranking, and the passage or reference view. Checks whether the generation is stale unless `include_staleness=false`. |
| `list_sources` | Lists discovered and indexed sources with stable IDs, inclusion state, and saved metadata overrides. Registers discovered IDs in the project catalog. |
| `get_passage` | Returns one passage with its neighbors and provenance. |
| `set_source_metadata` | Saves a reviewed metadata correction for one source. Applies immediately to an indexed source. |
| `set_source_inclusion` | Excludes or restores one source. Reversible; never deletes the file. |
| `export_bundle` | Writes a portable project archive. |
| `import_bundle` | Validates and reconstructs a project from an archive placed in `.research-rag/bundles/`. |

## How it works under the hood

You do not need this section to use the server, but it explains what the settings in "Install once" actually control.

```text
project PDF/EPUB files
        │
        ▼
research extraction ── bibliography + layout + original locators
        │
        ▼
UltraRAG GPT-2 token chunking
        ├──────────────► UltraRAG BM25
        └──► FastEmbed CPU vectors ──► project-local dense index
                                       │
                          weighted rank fusion
                                       │
                                       ▼
                          structured MCP evidence
```

**Extraction and cleaning.** The research layer chooses the allowed files, extracts layout-aware semantic units, resolves bibliography, removes layout artifacts, and keeps a locator for every unit.

**Chunking.** UltraRAG splits units into GPT-2 token chunks of the configured size. Several units share one chunker call, which is a throughput optimisation only: each unit still gets its own durable output file and its own place to resume from.

**Embedding.** FastEmbed produces revision-pinned CPU embeddings. Two details matter for speed and honesty:

- Ingestion audits every chunk against the model's token limit and records the token count plus a truncation flag, so a chunk whose vector covers only a prefix is counted and reported rather than silently degraded.
- Sequences are padded to the longest member of their inference batch, so the server embeds one sequence per inference. A large batch would make every short chunk as expensive as the longest one in its batch.

**Dense storage.** By default the generation's portable float32 vectors *are* the dense index, searched exactly. That removes index maintenance entirely and makes scores reproducible. Above a documented corpus size the server switches to an embedded index instead, and every manifest records which backend built it, so a generation always searches with the backend it was built with.

**BM25 and fusion.** UltraRAG supplies lexical retrieval. The two rankings are combined with weighted reciprocal-rank fusion. The embedded index backend stores vectors with lean lookup payloads (`chunk_id`, `document_id`, `source_id`), so canonical passage content and locators stay in `chunks.jsonl` while document metadata and provenance stay in the manifest.

**The sidecar.** A compact SQLite file inside each generation stores IDs, content hashes, vector ordinals, byte offsets, and one precomputed retrieval verdict per chunk. The verdict records whether a chunk is structurally unusable (`corrupt-text`, no readable content, extraction artifact) so a query can reject it without re-reading its text. It is derived metadata, never a copy of your text. Public results also carry `"direct_quote_safe": false`, because cleaned semantic text is not a transcript.

**Durability.** Generations are immutable. Reviewed metadata lives outside them and is layered on at read time, which is why a metadata edit appears immediately without rewriting chunk, BM25, vector, or dense index files. For dense filtering, current metadata is resolved to document IDs inside the selected generation rather than trusting stale copied metadata.

Before ingestion decides what to do, it hashes every discovered PDF/EPUB and fingerprints exclusions, chunk settings, processing policies, and the embedding model. Reviewed metadata is deliberately excluded from that identity, because it does not change source text or rankings. An exact match is a true no-op.

Unchanged documents keep their extracted units, chunks, and — for an identical canonical `contents` value and model fingerprint — their vectors. Final chunk records keep one semantic-text field, `contents`, plus identity, locator, and structural fields.

Changed material is recomputed. A changed generation always gets freshly reconstructed BM25 and dense indexes, and the pointer moves only after they verify. Restartable units are: source hashes, eight-page PDF scan and extraction batches, EPUB spine sections, chunking batches of up to 16 extraction units, 64-passage embedding batches, and 64-point dense uploads. Every source is hashed again before activation. `force_recompute=true` bypasses reuse while still resuming its own checkpoint.

Upstream extraction and dense output are not reused: this server needs layout-aware PDF/EPUB handling, bibliographic provenance, original locators, scored results, payload filters, portable vectors, and exact reuse accounting. Those responsibilities stay thin layers around UltraRAG rather than changes to its source.

## Limitations and troubleshooting

Things to know before you rely on a result:

- Scanned PDFs need OCR first. Password-protected PDFs are rejected.
- Locators are navigation aids, not quote offsets. PDF locators use the physical page plus a page label when available; reflowable EPUBs use a spine section plus an existing fragment or deterministic block, because EPUBs have no stable page numbers.
- The embedding and reranking models are English-oriented, and English is the supported workload. Non-English-primary corpora, OCR or scanned sources, handwriting, and formula-heavy corpora are outside the designed scope.
- The text-health policy withholds a passage only for corruption evidence, never for mixing scripts, so a quotation in another language stays retrievable and comes back with `text_notes`.
- A chunk longer than the embedding model's token limit is embedded from its beginning only. BM25 still matches its full text, while dense search covers the start. Ingestion counts and reports these chunks (`dense_truncated`) but does not split them.
- Duplicate sources are a judgement call. The server never deletes an original; you review and exclude.
- Large CPU ingestions and optional reranking are slow. Ingestion is resumable, but one expensive page, the first model download, or the BM25 step can exceed the soft per-call budget.
- Cleaned text is not a quote-verification surface — open the original.
- Earlier successful generations are kept. Automatic pruning is not implemented, so old generations accumulate until you remove them yourself.

If something looks wrong:

- The stdio executable appears to hang when run directly. It is waiting for an MCP client.
- `--offline` reports missing runtime or models. Run once online, or point `--model-cache-root` at a populated cache.
- `status` says the generation is stale. Search still uses the previous generation until a re-ingestion succeeds. That is intentional.
- Search returns nothing. This can be correct: the relevance gates prefer abstaining over returning weak matches. Try a precise BM25 query, or inspect dense mode, before lowering any quality expectation.
- Import reports a project-ID conflict. The wrong `.research-rag/project.json` is in use. A source conflict means an existing file has different bytes. Neither safeguard should be bypassed.

## UltraRAG credit and licensing

This independent server directly uses UltraRAG and gratefully credits the UltraRAG team and contributor community. UltraRAG identifies itself as a joint project of [THUNLP](https://nlp.csai.tsinghua.edu.cn/), [NEUIR](https://neuir.github.io/), [OpenBMB](https://www.openbmb.cn/home), [AI9stars](https://github.com/AI9Stars), and [contributors](https://github.com/OpenBMB/UltraRAG/graphs/contributors).

The dependency is pinned to UltraRAG `0.3.0.2`, commit [`3a709a2aea3fbe46acca59c422621c94b6e86857`](https://github.com/OpenBMB/UltraRAG/tree/3a709a2aea3fbe46acca59c422621c94b6e86857). UltraRAG is distributed under the [Apache License 2.0](https://github.com/OpenBMB/UltraRAG/blob/3a709a2aea3fbe46acca59c422621c94b6e86857/LICENSE.txt). The installed upstream snapshot retains its license and notices.

This repository is not an official UltraRAG release and is not affiliated with or endorsed by OpenBMB, THUNLP, NEUIR, AI9stars, or UltraRAG contributors. See [`NOTICE`](NOTICE) for full upstream, model, retrieval-component, and UI attribution.

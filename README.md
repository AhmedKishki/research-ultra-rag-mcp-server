## Purpose

This server lets an AI agent search a personal collection of PDF and EPUB files and hand back the passages that matter, with enough provenance to check them.

It is built for research work where you need to find relationships between sources, compare authors, or locate passages that support, qualify, or contradict a claim. It returns **evidence candidates**, not conclusions. You and your agent stay responsible for reading, judging, and writing.

> **Retrieved text is cleaned semantic text, not a quote-safe transcript.** Open the original PDF or EPUB at the returned locator before using a direct quotation.

The server is built on [UltraRAG](https://github.com/OpenBMB/UltraRAG), whose corpus chunking and BM25 retrieval it uses. Full credit and licensing details appear at the end of this document.

## What the server provides

Plain-language summary of every capability, and why it is there:

- **Lean tool answers, with a developer debugging mode.** Every tool returns the fields an agent acts on; a field that is empty, null, or at its default is omitted. `--tool-detail full` returns the complete payload instead.
- **An MCP server for AI agents, plus a local browser UI.** Agents call tools over stdio; you can also click through the same features in a browser. Both operate on the same project state.
- **A per-project UI launcher, created when the project is initialised.** The first run writes `open-ui.sh` in the project root as a symlink to a generated script under `.research-rag/bin/`, so starting or stopping that project's browser UI is one command and the private server it starts cannot be left behind. It starts on the first free port at or above the one it was generated with and remembers that port, because two projects must never serve their UI from the same port: opening the wrong project's UI is worse than a second URL. An existing file or symlink is never overwritten.
- **Ingests regular `.pdf` and `.epub` files only.** It walks the source directory recursively. Markdown and every other format are ignored, so a stray `notes.md` in the folder never enters the knowledge base.
- **Resolves bibliographic metadata and locators.** Titles, authors, years, DOIs, and a locator that points back into the original file (PDF page, or EPUB section), so you can always find the page behind a passage.
- **One way to search, with the measured best settings.** Every search is hybrid retrieval with CPU reranking, which is the largest measured quality gain available. A query can be narrowed by source, by the project a source was gathered for, by the branch it belongs to, and by the terms that identify it, all from reviewed metadata. Filters are applied before ranking, so `top_k` is a budget inside the selection.
- **Reviewed metadata that applies immediately.** Fix a wrong author or year in the project's review-state file and the change shows up in listings, citations, filters, and results at once — without re-ingesting or rewriting the generation.
- **Reversible source inclusion.** Exclude a duplicate source, later restore it. The original file is never deleted or modified.
- **Resumable ingestion.** Every call has a soft time budget. If it runs out, the server returns a checkpointed `in_progress` result and you simply call it again. Client timeouts, cancellations, and restarts lose at most one small batch of work.
- **Text-health policy with disclosure.** Corrupt text (broken character maps) and chunks with no readable letters or digits are excluded, and `status` reports the corpus-level counts. A quotation in another language or script is *not* excluded — it is returned as it is, and the advisory script note about it belongs to the full-detail payload.
- **An embedding-fidelity audit.** Each chunk records how many embedding tokens it contains and whether its vector covers only the beginning of its text, so silent truncation is counted in the ingestion and status output.
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

The pinned UltraRAG runtime is installed automatically. The embedding model is downloaded on first ingestion, and the reranker model on its first search. Model binaries are shared at `~/.cache/research-ultra-rag-mcp/models`, while all document data stays inside each project.

## Settings

Nothing that decides an outcome is compiled in. `default.toml` ships inside the package with the values this server uses when nobody says otherwise, and every layer above it names only what it changes — a handful of keys, or a whole section, and the rest is inherited. Later layers win, per key:

| Layer | Where |
|---|---|
| built-in defaults | `default.toml` inside the installed package |
| per-user config | `~/.config/research-ultra-rag-mcp/config.toml` (platform config directory) |
| per-project config | `<project>/.research-rag/config.toml` |
| an explicit file | `--config PATH` or `RESEARCH_ULTRARAG_CONFIG` |
| environment | `RESEARCH_ULTRARAG_*`, one per setting |
| command line | `--set key=value`, repeatable |

Inspect the result instead of guessing: `research-ultra-rag-mcp --project-root <project> --print-config` prints every key, its effective value, and the layer that supplied it. A key that is not declared is refused in every layer, so a mistyped name is an error rather than a silent default, and a value outside its bounds is refused with the key and the bound named.

Settings come in three classes, marked in `default.toml`:

- **identity** — the value decides what a generation contains. The ranking values (fusion weights, the RRF constant, the dense cosine gate, candidate counts, the rerank cap, the withheld-example limit) enter the retrieval-policy fingerprint, and the chunking values are recorded with the generation. Change one and the next ingestion builds a new generation instead of quietly mixing two; a generation that recorded a different policy is reported as needing an upgrade, never reused as if it matched.
- **engine choice** — the dense backend, the reranker model, and the exact-scan threshold. Each is recorded in the generation it built and in the answers that used it.
- **runtime** — thread counts, batch sizes, the work budget, log level, tool detail, and where the shared model cache lives. These shape this process only and cannot change an artifact.

The operational values stay command-line only, because they name an invocation rather than a preference: `--project-root`, `--source-directory`, `--runtime-root`, `--vanilla-executable`, `--runtime-cache-root`, `--port`, and `--ui-port`.

The settings that are commonly set, and what they do:

| Setting | What it does |
|---|---|
| `runtime.model_cache_root` or `--model-cache-root` | Move the shared model cache elsewhere. |
| `dense.backend auto\|exact\|qdrant` or `--dense-backend` | Choose how dense search is stored. `auto` (default) scans the portable vectors directly for normal-sized corpora and switches to an embedded index for very large ones. |
| `dense.reranker_model NAME` or `--reranker-model` | Choose the CPU cross-encoder that reranks every search. Six are supported, each pinned to a revision, and an unknown name is refused rather than resolved to whatever the model hub serves that day. See "How a search works". |
| `language.corpus` | The language of the corpus as an ISO 639-1 code (`en`, `de`, ...). It selects the BM25 stopword list and it decides whether the embedding model covers the text; a mismatch is reported by `status` rather than embedded silently. An identity setting: it is part of the ranking policy recorded with a generation. |
| `dense.embedding_model NAME` | The embedding model for the dense half of retrieval, from the pinned table in `embeddings.py`. Each name carries its revision, vector dimension, token limit, licence, the languages it covers, and any query/passage prefix it requires. Changing it is a re-ingestion with a new index. |
| `runtime.embedding_threads` or `--embedding-threads` | Set the CPU thread count for embedding. Left to the runtime by default because the best value depends on your machine. |
| `retrieval.*` | Tune fusion and gating: `rrf_k`, `bm25_weight`, `dense_weight`, `minimum_candidates`, `maximum_candidates`, `dense_minimum_cosine_similarity`, `rerank_max_candidates`, `maximum_withheld_examples`. These are identity settings: the next ingestion is a new generation. |
| `chunking.size`, `chunking.overlap` | Chunk length and overlap in tokens. Identity settings, recorded with the generation. |
| `ingestion.work_budget_seconds` | Soft time budget for one `ingest` call before it returns a checkpointed `in_progress` result. |
| `runtime.tool_detail` or `--tool-detail` | `lean` (what an agent gets) or `full` (the developer debugging payload). |
| `--runtime-root` or `RESEARCH_ULTRARAG_RUNTIME_ROOT` | Keep the working files (indexes, staging, logs) on a different disk. See "Where project data is stored". |
| `--offline` | Fail instead of downloading anything. Use it once the runtime and models are cached. |

Two examples:

```bash
# One run, without editing any file.
research-ultra-rag-mcp --project-root ~/research \
    --set retrieval.rrf_k=30 --set chunking.size=300 --print-config

# A project that always uses a different reranker and a tighter dense gate.
cat > ~/research/.research-rag/config.toml <<'TOML'
[dense]
reranker_model = "jinaai/jina-reranker-v1-turbo-en"

[retrieval]
dense_minimum_cosine_similarity = 0.65
TOML
```

### Working in another language

Extraction is language-neutral: it repairs layout (for instance re-joining a hyphen a PDF broke across a line) and it reasons about *scripts*, not languages, so text in any Latin-script language is extracted as it stands. The two stages that do depend on language are settings:

- `language.corpus` selects the BM25 stopwords, and the BM25 relevance gate uses them, so German function words stop counting as evidence. Only a language BM25 can tokenize is accepted — English, German, Dutch, French, Spanish, Portuguese, Italian, Russian, Swedish, Norwegian, Chinese, Turkish, Korean — because that list is where the stopwords come from. A language outside that set is refused while settings are read, instead of after a build has already extracted and embedded the corpus.
- `dense.embedding_model` selects the embedding model. The default is English-only; the table offers `jinaai/jina-embeddings-v2-base-de` (768-d, 0.32 GB, Apache-2.0) for German and `intfloat/multilingual-e5-large` (1024-d, 2.24 GB, MIT) for one model across languages.

A German project, entirely in its own `.research-rag/config.toml`:

```toml
[language]
corpus = "de"

[dense]
embedding_model = "jinaai/jina-embeddings-v2-base-de"
```

`status` reports the mismatch when a corpus language and an embedding model disagree, which is the case before any of this is set: the default model covers `en` only, so a German corpus starts with a warning instead of a silent quality loss. The reranker is query-time only, so a multilingual one can be chosen separately; note that the registry's multilingual reranker (`jinaai/jina-reranker-v2-base-multilingual`) is CC-BY-NC-4.0, which suits non-commercial research but is a licence decision rather than a default.

### What is worth setting on this machine

Two things have a measured payoff; everything else is worth leaving at its shipped default until a measurement says otherwise.

- `runtime.embedding_threads = 8` shortens every rebuild. On the reference machine it measured 31.66 chunks/s against the runtime's default 23.65, so the embedding phase of a 17-minute build drops by about a third. The optimum is machine-specific, so it is the one setting worth re-measuring on your own hardware.
- A larger `top_k` is free accuracy. The reranker reorders about twice as many candidates as the caller asks for, so the depth of the request sets both how many passages come back and how deep the ranking goes. Measured on the judged set: 8 passages answers 80.0% of questions at rank 1 with MRR 0.825, while 10 and above answers 83.3% with MRR 0.858. The default is 10 for that reason, and asking for 15 is the first move when an answer looks thin.

Everything under `retrieval.*` and `chunking.*` is a measurement waiting to happen rather than a knob to turn: `TODO.md` lists which values are worth trying and what would have to be true before a change to the shipped default is justified.

## Update an existing installation

An editable install reads this checkout when a process starts, so updating is a pull plus a sync:

```bash
git -C /path/to/research-ultra-rag-mcp-server pull --ff-only
uv sync
```

`scripts/update.sh` does both, reports what is left, and can restart a project's browser UI:

```bash
scripts/update.sh --check                 # report only: commits behind, versions, pending dependencies
scripts/update.sh                         # pull and sync
scripts/update.sh /path/to/my-project     # … and restart that project's UI first
```

A **running** server keeps the code it started with. Its process belongs to your MCP client, so nothing here can reload it: restart the server in the client (MCP panel → restart or toggle it, or reload the window). `status.version` reports `server` (the version that process started with), `installed` (the version installed now), `ui` (the pinned browser-UI package), and `restart_required`, which reads `false` once the two agree. The browser UI shows the server and UI versions under the project name in its header.

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

### What a tool answer contains

A search returns `query`, `generation_id`, `stale`, `reranked`, and the selected passages. A passage holds `chunk_id`, `source_relative_path` (the filename), `authors`, `locator`, and `text`. The locator is the position alone: a page, with the printed page label only where that label differs from the physical page, or a section for an EPUB. A passage is deliberately not citation-ready and repeats nothing: it carries no title and no citation, because the text is cleaned for retrieval rather than quotation, and `--tool-detail full` is what returns the citation, the resolved title, the year and DOI, the per-field provenance, the quote-safety flag, the advisory script note, and the ranking accounting.

`status` returns readiness, `stale`, the selected generation's `generation_id`, `created_at` and `chunk_count`, the source counts, `hybrid_ready` when the generation cannot serve the hybrid search the tool always runs, `generation_upgrade_required` with `upgrade_reasons`, `metadata_overlay_active`, `metadata_pending_source_count`, `ingestion_progress`, what a prune would consider as `retained_generation_count` and `retained_generation_bytes`, and, when the generation is stale, `changes` — counts of added and modified sources, `removed_sources` naming the files that disappeared, and the review and exclusion flags. It inventories nothing: the retained-generation list, the category inventory, the project inventory, and the generation's own method list come back from `--tool-detail full`. `ingest` returns its `status`, `generation_changed`, and the document, chunk, vector, reuse, and discard counts. `set_source_inclusion` returns the decision, its reason, `effective_immediately`, and whether the next ingestion should rebuild without the source.

A field that is empty, null, or false is omitted, so an absent field means there is nothing to report. `stale` (where `null` means the freshness check was skipped) and `reranked` are always present.

`--tool-detail full` (`RESEARCH_ULTRARAG_TOOL_DETAIL=full`) returns the complete payload instead: ranking and reranking scores, candidate and gate counts, withheld candidates with their reason codes, per-field metadata provenance, embedding audits, filesystem paths, model identifiers, revision fingerprints, and phase timings. It is a developer debugging mode: the browser UI, `research-ultra-rag-verify`, and the evaluation harness select it for their own server, and no part of this project requires it.

### Let the server host the browser UI (optional)

Add `--ui-port` and the same server also serves the local UI on that loopback port for as long as it runs, reusing exactly the project, runtime root, model cache, dense backend, and offline setting it was launched with:

```json
{
  "mcpServers": {
    "research-ultra-rag": {
      "command": "/ABSOLUTE/PATH/research-ultra-rag-mcp-server/.venv/bin/research-ultra-rag-mcp",
      "args": [
        "--project-root", "/ABSOLUTE/PATH/my-research-project",
        "--ui-port", "5051"
      ],
      "timeout": 3600
    }
  }
}
```

Then open `http://127.0.0.1:5051`. What this does and does not do:

- Each project's UI is its own: `open-ui.sh --open` opens the URL of *that* project, on a port no other project's UI holds, and the page names the project it serves. Verified with two projects running side by side.
- The port is bound on `127.0.0.1` only, and no browser window is opened for you.
- The UI stops when the server stops. Measured on the reference project: the UI answers `/api/health` about six seconds after the server starts, and the port is released within half a second of the server being terminated, so there is no orphan process and nothing to clean up by hand.
- `status` reports `ui_url`, `ui_ready`, and `ui_error`. The port is claimed by binding it before the UI starts, so two servers starting together cannot both believe they hold it: the MCP server keeps serving tools, `ui_ready` stays false, and `ui_error` names the port and the reason. A released port is claimable again rather than remembered as taken.
- Serving the UI is an explicit option with no environment default, and a server this project starts for itself refuses `--ui-port` as a managed child, so no exported variable and no inherited argument can turn one UI into a chain of servers.
- This is the option that keeps the UI and the agent on the same state. A separately launched `research-ultra-rag-ui` has to be given the same `--runtime-root` by hand, because it never reads your MCP client's configuration.
- An MCP client that keeps one settings file for every window gives each window's server the same `--ui-port`. Only the first server claims it; the others keep serving tools with `ui_ready: false` and `ui_error` naming the taken port, and every window still carries its own server, gateway and UI stack. With more than one window or project open, a per-project launcher that starts `research-ultra-rag-ui` with that project's `--project-root`, `--runtime-root`, and `--port` stays predictable in a way a shared `--ui-port` cannot.
- Reranking is not a switch anywhere. The search tool always reranks, so an agent and the browser UI return the same reranked order; which model does the reranking is a server setting, not a per-search choice.

Two practical notes:

- Running the executable by hand looks like nothing happens. That is correct — it is a stdio server waiting for an MCP client to send protocol messages.
- Keep `set_source_inclusion` out of automatic approval at first. It persists a reviewed decision, so it is worth confirming once.

## First use

The normal sequence an agent should follow:

1. Call `status`.
2. Report whether no generation exists yet, the selected generation is stale, or a schema/policy upgrade is required.
3. Ask permission before a persistent ingestion when that is appropriate.
4. Call `ingest`. While it returns `status="in_progress"`, call it again unchanged.
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

- A `source_id` is derived from the project ID and the source-relative path. It survives changes to the file's contents, but renaming or moving the file creates a new `source_id`. Use it, or the filename, with `set_source_inclusion`.
- A `document_id` identifies one content/path version inside a generation and changes when the bytes or the path change. It is an internal identity for the generation's own accounting, and it is not reported in answers.

Calling `list_sources` also registers discovered IDs in the portable project catalog. That is what keeps a registered ID addressable when an original is renamed, moved, or temporarily absent — the full-detail payload lists those records as `known_sources` — without guessing that metadata from an old path belongs to a new one. `reviewed_metadata_sources` lists every saved override by the same stable ID, so you can inspect, replace, or remove each persisted decision even if the original file is not present right now.

The one mutation tool names its source by the filename that `list_sources` reports, so an exclusion reads the way a person would describe it.

## Research workflow

Prompts that work well:

- "Find the strongest source passages supporting the claim that …"
- "Search broadly across references; return at most two passages from any one source within the total passage budget."
- "Search for evidence that contradicts or qualifies this claim: …"
- "Compare how these sources explain …; distinguish agreement from conflict."
- "Get the neighboring passages around chunk `chk_…` before interpreting it."
- "Open the original at the returned path and page before quoting it."
- "List sources with missing metadata or extraction warnings. Treat automatic metadata as provisional; tell me which entry in `.research-rag/source-metadata.json` looks wrong, and confirm the correction shows up without re-ingesting."
- "List the source IDs, then exclude the duplicate source ID as a reviewed duplicate of the preferred source ID; do not delete either file."
- "Restore that source ID, then re-ingest if it is absent from the current generation."

### How a search works

Hybrid is the engine's default and the tool's only setting. BM25 matches the words you typed; a dense vector search matches meaning; the two rankings are fused; and a CPU cross-encoder then reranks the fused candidates. The engine can also serve either signal alone, which is how `MEASUREMENTS.md` reports each mode, but the tool does not ask which one you want — the measurements say hybrid with reranking is the best of them. BM25 is strongest single-signal retrieval when a question names a person, place, or project; dense alone is the weakest, finding the judged passage within ten results for 63% of questions and worse on names, because its similarity gate returns fewer passages; hybrid beat BM25 alone at the top; and reranking is the largest measured gain of all, putting the judged passage first for 81% of questions against 66% without it, with the right document in the results for 94% against 91%.

Reranking costs time — about 2.3 s per warm query against 0.17 s for unranked hybrid, plus a second model downloaded on first use — and if the pinned model cannot be loaded the search returns the unranked order and says so in `rerank_fallback` rather than failing.

The reranker model is an engine setting rather than a search option, because a per-search switch would make two searches incomparable. Six FastEmbed CPU cross-encoders are supported, each pinned to a revision; the default is `Xenova/ms-marco-MiniLM-L-6-v2`, the model every number above was measured with and the faster of the two measured so far. To measure another, name it for the harness — `scripts/evaluate_retrieval.py --reranker-model Xenova/ms-marco-MiniLM-L-6-v2 --reranker-model jinaai/jina-reranker-v1-turbo-en` scores both over the same judged queries in one run, and `MEASUREMENTS.md` records what that comparison found: on the reference corpus the default stays, because `jinaai/jina-reranker-v1-turbo-en` reaches the same depth but puts the judged passage first less often (73% against 83%) at about half again the cost per query (3.47 s against 2.34 s). Naming a model is an engine decision, so it takes a launch setting, a second harness row, or a code-level caller — never an agent's search.

Search can legitimately return fewer results than `top_k`, including none, when candidates fail the relevance gates — abstaining is a feature, not an error. A rank or similarity score is an ordering signal, never a truth or confidence probability. The mode comparisons above are measurements, not impressions: the judged question set and the harness that runs it are in `evaluation/` and `scripts/evaluate_retrieval.py`, and the full tables, including the fact that questions phrased in your own words are much harder for every mode than remembered phrasing, are in `MEASUREMENTS.md`.

### Selecting and excluding sources per query

`source_ids` restricts a search to the sources you name and `exclude_source_ids` removes sources from the result, both by the stable `source_id` that `list_sources` reports. Omitting both searches the whole corpus, which is the default: include everything, exclude nothing. Filters are applied before ranking, so `top_k` is the budget inside the selection.

A `source_id` is derived from a source's normalized relative path: it survives edits to the file's bytes and changes when the file is renamed or moved. An ID that resolves to nothing in the selected generation is reported as `unresolved_source_ids` (`unresolved_exclude_source_ids` for exclusions), and an include list that resolves to nothing at all is an error rather than a silently unfiltered search. A reviewed exclusion always wins: naming an excluded source in `source_ids` cannot bring it back, so the response returns no hits. The full-detail payload reports the same facts under `filters`, where `active_document_count` says how many sources the search actually covered.

### Dividing a corpus with categories and projects

Reviewed source metadata carries three filter layers, and each is independent:

| Layer | What it records | Example |
|---|---|---|
| `project` | which project a source was gathered for | `ai-and-fetishism` |
| `categories` | the branch or branches the source belongs to | `marxism`, `critical realism`, `political ecology` |
| `keywords` | the terms that identify the source, or that it leans on | `fetishism`, `use value` |

`categories`, `categories_any`, `projects`, `projects_any`, and `keywords` all come from reviewed source metadata, so you can define them yourself in the project's `source-metadata.json`; an edit applies immediately and needs no re-ingestion. The strings are free, so they work as corpus partitions — a theoretical branch, a research strand, a sub-project — and one search can cover several parts at once:

```json
{
  "query": "labour in the supply chain",
  "projects": ["ai-and-fetishism"],
  "categories_any": ["marxism", "political ecology"],
  "keywords": ["fetishism"],
  "top_k": 8
}
```

Each filter is all-of at the plural name (`categories`, `projects`, `keywords` require every listed value) and any-of at the `_any` variant (`categories_any`, `projects_any`). A one-project server normally tags every source with its own project name, so the project layer is a passthrough there; it becomes useful when a corpus is copied into another project or shared. `status` does not inventory them: the vocabulary is reviewed metadata, so read it from the project's `source-metadata.json`, from the UI's partition chips, or from `--tool-detail full`, where `categories` and `projects` are reported with each one's `searchable_source_count` and reviewed exclusions are not counted. The browser UI lists those partitions as chips beside the status — select one or several to search their union — and can include or exclude named sources from the search panel or straight from a source card.

### Freshness with every search

Every search reports whether the selected generation is stale, which means comparing every source file with what the generation recorded. That comparison costs about 10 ms for a 55-source project and grows with the collection, so it is the one part of a search whose cost depends on how many files you have rather than on the question asked; at this size it is a small share of a query, so it is always done rather than left to the caller to remember.

### What makes a generation stale

Adding, removing, or changing source content — or changing which sources are included — can make the selected generation stale, and `status` says so: it counts the sources it gained and modified, names the ones the directory no longer has, and flags a changed review or exclusion. It never lists the sources that are still available, so read `changes` as a verdict on the corpus rather than as an inventory. When the generation is stale you have two options:

- **Re-ingest changes** verifies every source hash and reuses compatible documents, chunks, and vectors, then rebuilds both indexes. This is the normal path and it is much cheaper than starting over.
- **Regenerate** uses `--force-recompute` and deliberately ignores all reuse.

Reviewed metadata is different from staleness. For a source already in the selected generation, an edit to `source-metadata.json` updates listings, filters, results, citations, and neighboring passages at the next read. It does not rewrite the immutable generation and does not change chunk IDs. A metadata-only difference is reported through the metadata overlay field (`metadata_overlay_active`) rather than as stale retrieval state. Metadata for a source that is missing from the selected generation is saved in the file and takes effect after the next ingestion.

### What gets excluded, and what does not

Extraction rejects a whole page or section only when its text carries strong evidence of a broken character map: replacement characters, private-use or unassigned code points, or a known damaged encoding sequence. The diagnostics keep only the locator and reason codes, never the garbage text. If nothing readable remains in a source, the build fails so you can repair or OCR the file.

Two normalisation rules exist so that typed queries match printed text: formula-font letters (`𝑀` becomes `M`) and the presentation ligatures `ﬁ`, `ﬂ`, and `ﬀ` are folded to their plain spellings. Accented letters, superscripts, subscripts, and symbols are left untouched, because they carry meaning in citations and notation.

Script mixing and non-Latin dominance are never rejection reasons. A quotation in another language stays retrievable and its text is returned unchanged. Reviewed metadata is never overridden by this classifier.

Finally, ingestion drops nonempty chunks that contain no alphanumeric content at all. Ordinary prose, numbers, and formulas containing at least one letter or digit are not affected.

## Use the UI

From the installed server repository:

```bash
uv run research-ultra-rag-ui \
  --project-root /absolute/path/to/my-research-project
```

Open [http://127.0.0.1:5051](http://127.0.0.1:5051) if a browser does not open by itself. The UI binds to loopback only and calls the same six public MCP tools against the same project state as an agent. Its header shows the project name and, underneath it, the versions of this server and of the pinned browser-UI package, so it is visible which software the page is running.

What you can do in it:

- inspect status, staleness, upgrade state, and build metrics;
- browse the indexed source list and resolved bibliography;
- search the corpus, limiting a search to or away from named sources and to one or several category or project partitions at once;
- read neighboring passages around a hit;
- open a PDF in the browser, or download an EPUB original;
- exclude or restore a source without deleting anything;
- **Create generation** for the first normal build;
- **Re-ingest changes** for verified reuse after the project changes; and
- **Regenerate** to force extraction, chunking, and embedding again.

One UI process serves one project; to keep two open at once, start a second process with another project root and port, for example `--port 5052`. During ingestion the UI shows the operation as busy. A project lock serialises agent and UI operations, so a second request waits instead of reading a half-built index. The UI follows checkpointed `in_progress` responses automatically until the generation is ready or nothing changed.

**Give the UI the same runtime settings you gave your agent.** The UI starts its own MCP server and never reads your MCP client's configuration, so a project whose derived state lives outside it needs the setting repeated:

```bash
RESEARCH_ULTRARAG_RUNTIME_ROOT=/ssd/research-runtime/my-project \
  uv run research-ultra-rag-ui --project-root /absolute/path/to/my-research-project
```

Without it, the UI and your agent read different runtime roots and can show different generations for the same project — the agent on the fast device and the UI on the old in-project copy. The simpler option is to let the MCP server host the UI with `--ui-port`, described under [Connect an AI agent](#connect-an-ai-agent): it reuses the server's settings, so the two can never disagree.

To make the separate process repeatable the server writes that launcher for you when it initialises the project: `<project>/open-ui.sh` is a symlink to `.research-rag/bin/open-ui.sh`, which starts the standalone UI with this project's `--project-root`, `--runtime-root`, and `--port`. `./open-ui.sh` starts it and prints the URL, `./open-ui.sh --open` also opens the browser, `./open-ui.sh --stop` stops it together with the private server it started, and `./open-ui.sh --port 5052` moves it to another port. An existing `open-ui.sh` is never overwritten, and `status.ui_launcher` reports whether the script and the link are present. The generated script passes this project's own flags explicitly rather than exporting settings, and serving a UI is an explicit option with no environment default, so the servers it starts cannot be handed a UI choice they never asked for. It calls the UI command by absolute path — the console script installed beside the server's interpreter — so it works from a shell where that command is not on `PATH`. Keep it out of version control: it is machine-local.

One difference between a UI session and an agent session is worth knowing before you compare results: the retained-generation inventory — `generations`, with each generation's counts, size, file count and schema — is returned only by `--tool-detail full`, and is not shown in the current UI views, so use that mode or `research-ultra-rag-verify` to see what retained generations occupy. A lean `status` reports only `retained_generation_count` and `retained_generation_bytes`.

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

Useful flags: `--top-k`, `--query`, and `--ingest --force-recompute` to rebuild first. Add `--offline` when every required cache already exists.

On success it prints a JSON object with `"status": "passed"`, the before/after status, optional ingestion metrics, and the search result. With `--ingest` it repeats checkpointed calls until ingestion finishes. The usual first-run failures are a missing network or model download, an unsupported Python version, a damaged PDF/EPUB, or `--offline` before the runtime and models are cached.

## Where project data is stored

```text
my-research-project/
├── sources/                              untouched PDF/EPUB originals
├── open-ui.sh                            symlink to .research-rag/bin/open-ui.sh
└── .research-rag/                        all research-RAG project state
    ├── bin/open-ui.sh                    generated machine-local UI launcher
    ├── project.json                      stable ID, name, source setting
    ├── source-catalog.json               durable source ID-to-path registry
    ├── source-metadata.json              authoritative reviewed metadata overlay
    ├── source-exclusions.json            reviewed decisions, when present
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
- The only research-RAG path outside `.research-rag/` is the single `open-ui.sh` symlink in the project root. The server creates it on first use, never overwrites an existing file or symlink, and `status.ui_launcher` reports it; delete the symlink to opt out.
- Inside `.research-rag/`, the top-level JSON files are **portable review state**: your decisions about the project. They are the part worth backing up.
- `runtime/` is **derived state**. It can be rebuilt from `sources/` plus the portable state, so it is safe to delete if you are willing to rebuild.
- Document text, embeddings, indexes, logs, and query state never cross project roots. Only immutable model binaries are shared.

On first use after upgrading, an existing `.ultrarag/research/` directory is moved automatically to `.research-rag/runtime/`. If both locations already hold runtime data, startup stops rather than guessing. Other `.ultrarag/` content belonging to different tools is left alone. Stop every research MCP and UI process before the first launch of the upgraded package.

`current.json` names the one generation search uses. Earlier successful generations stay on disk but are not searched, and automatic pruning is not implemented. `status` lists them (`generations`) with their creation time, chunk and document counts, file count, and size, marks the current one, and reports `retained_generation_count` and `retained_generation_bytes`, so you can see what they occupy. A directory whose manifest is missing or unreadable is reported with `manifest_error` rather than failing the call. Nothing is ever deleted by `status`; remove an old generation directory yourself, and only when you are sure no MCP or UI process is using it. On the reference project two retained generations occupy 208 MB in total.

### Edit review state by hand

The portable files under `.research-rag/` are plain JSON and stay authoritative at read time, so you can edit them in a text editor instead of going through a tool:

| File | What it holds | Keyed by |
|---|---|---|
| `source-metadata.json` | the reviewed metadata overlay | normalized source-relative path |
| `source-exclusions.json` | each exclusion decision and its reason | normalized source-relative path |
| `project.json` | project name, stable id, and the sources directory | — |

`source-catalog.json` is the durable source-id registry this server maintains; leave it alone.

A metadata entry is the **complete** override for that source and takes the same seven fields as the tool:

```json
{
  "schema_version": 1,
  "sources": {
    "Harvey, The Fetish of Technology - Causes and Consequences.pdf": {
      "title": "The Fetish of Technology: Causes and Consequences",
      "authors": ["David Harvey"],
      "year": 2003,
      "doi": "",
      "categories": ["Commodity fetishism", "marxism", "media theory"],
      "keywords": ["fetishism of technology", "technology", "ideology", "marxism"],
      "project": ["ai-and-fetishism"]
    }
  }
}
```

How a hand edit behaves:

- It applies at the next read — listings, filters, search results, citations, and neighbouring passages — with no ingestion. `status.metadata_revision` changes, and `generation_metadata_snapshot_outdated` may become true, which only says the immutable generation predates this review.
- Omitting a field stops overriding it, so the extracted or automatic value returns; an explicitly empty list or string clears it.
- Deleting a source's whole entry removes every reviewed field for that source. An entry of `{}` does the same.

Mistakes fail loudly rather than doing nothing quietly, so you can trust a hand edit:

- an unknown field name is rejected with `Unsupported metadata fields: …`;
- a wrong type, such as `"year": "2003"`, is rejected with that field's rule;
- a source path that is absolute, contains `..` or a backslash, or is otherwise not normalized is rejected;
- any `schema_version` other than 1 is rejected.

Editing that file is the only route for metadata: the tool surface has no metadata writer, and the browser UI has no metadata dialog. `--tool-detail full` reports `metadata_provenance` per field in `list_sources`, so you can see which values are reviewed and which are still automatic.

### Put derived state on fast local storage

If your project lives on a slow disk, you can point the working files at a fast one. The measured benefit is narrow, and worth stating precisely: the dense index is an exact scan of the portable vectors the generation already stores (0.03 s of index work on the reference corpus, against 3,040.73 s to build an embedded index for the same vectors), so what a rebuild still does on disk is staging writes and reading the sources — grouping durability writes per unit is worth 17–37 s of chunking on the reference HDD, and source reads cost whatever the device costs. Use it when the project's disk is genuinely the bottleneck, not as a routine default.

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

## Move or back up a project

A project is self-contained: copy the project directory — `sources/` plus `.research-rag/` — and the copy is a complete project on another disk or machine. `runtime/` is disposable and can be left behind; it is rebuilt from `sources/` and the portable review state on the next ingestion.

What to keep:

- `sources/` — the originals, and the only authority for exact quotation;
- `.research-rag/project.json`, `source-metadata.json`, `source-exclusions.json`, `source-catalog.json` — the project identity and your reviewed decisions;
- `.research-rag/runtime/` — optional; it holds the generations, so copying it keeps the project searchable without a rebuild.

Continue in one place at a time. Two servers pointed at the same project root, or at a copy that shares a relocated runtime root, are refused by the project lock rather than silently interleaved.

## Complete MCP tool reference

Six tools are exposed. All are project-scoped and none of them deletes a source file.

| Tool | What it does |
|---|---|
| `status` | Reports readiness, staleness, the source and generation counts, the category and project inventories, the available retrieval methods, any required upgrade with its reasons, resumable-ingestion progress, and every retained generation with its creation time, counts, and size. It also reports `restart_required` when the running process is older than the installed version. The full-detail payload adds the paths, the version block, revision fingerprints, build metrics, UI-launcher state, and per-source exclusion records. Read-only. |
| `ingest` | Creates or refreshes a generation with the server's own chunking settings. Resumable, with a soft per-call work budget; `force_recompute` bypasses reuse. Reports what changed and how much was reused; discarded, withheld, and densely truncated material is reported only when there is any. |
| `search` | Retrieves evidence candidates with hybrid retrieval and reranking. Optional narrowing by source (`source_ids`, `exclude_source_ids`) and by reviewed metadata (`projects_any` and `categories_any` keep a result carrying at least one listed value; `keywords` requires every listed term). Always reports whether the generation is stale. Returns 10 passages by default; when an answer is thin, ask the question again in different words and raise `top_k`, which also widens the window the reranker reorders. Answers with the passages, `stale`, `reranked`, and any unresolved ID. |
| `list_sources` | Lists discovered and indexed sources with stable IDs, inclusion state, and saved metadata overrides. Takes no parameters: it is the corpus inventory. Registers discovered IDs in the project catalog. |
| `get_passage` | Returns one passage with its immediate neighbors and provenance. |
| `set_source_inclusion` | Excludes or restores one source, named by its filename. Reversible; never deletes the file. |

Every one of them answers as described under [What a tool answer contains](#what-a-tool-answer-contains).

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

**The sidecar.** A compact SQLite file inside each generation stores IDs, content hashes, vector ordinals, byte offsets, and one precomputed retrieval verdict per chunk. The verdict records whether a chunk is structurally unusable (`corrupt-text`, no readable content, extraction artifact) so a query can reject it without re-reading its text. It is derived metadata, never a copy of your text. Returned text is cleaned semantic text rather than a transcript, and the tool description says so once instead of every passage repeating a quote-safety flag.

**Durability.** Generations are immutable. Reviewed metadata lives outside them and is layered on at read time, which is why a metadata edit appears immediately without rewriting chunk, BM25, vector, or dense index files. For dense filtering, current metadata is resolved to document IDs inside the selected generation rather than trusting stale copied metadata.

Before ingestion decides what to do, it hashes every discovered PDF/EPUB and fingerprints exclusions, chunk settings, processing policies, and the embedding model. Reviewed metadata is deliberately excluded from that identity, because it does not change source text or rankings. An exact match is a true no-op.

Unchanged documents keep their extracted units, chunks, and — for an identical canonical `contents` value and model fingerprint — their vectors. Final chunk records keep one semantic-text field, `contents`, plus identity, locator, and structural fields.

Changed material is recomputed. A changed generation always gets freshly reconstructed BM25 and dense indexes, and the pointer moves only after they verify. Restartable units are: source hashes, eight-page PDF scan and extraction batches, EPUB spine sections, chunking batches of up to 16 extraction units, 64-passage embedding batches, and 64-point dense uploads. Every source is hashed again before activation. `force_recompute=true` bypasses reuse while still resuming its own checkpoint.

Upstream extraction and dense output are not reused: this server needs layout-aware PDF/EPUB handling, bibliographic provenance, original locators, scored results, payload filters, portable vectors, and exact reuse accounting. Those responsibilities stay thin layers around UltraRAG rather than changes to its source.

## Limitations and troubleshooting

Things to know before you rely on a result:

- Scanned PDFs need OCR first. Password-protected PDFs are rejected.
- Locators are navigation aids, not quote offsets. PDF locators use the physical page plus a page label when available; reflowable EPUBs use a spine section plus an existing fragment or deterministic block, because EPUBs have no stable page numbers.
- The default embedding and reranking models are English-oriented, which is the supported default workload; both are now settings, and the pinned tables offer a German-native embedding model and a multilingual reranker. Non-English-primary corpora, OCR or scanned sources, handwriting, and formula-heavy corpora are outside the designed scope.
- The text-health policy withholds a passage only for corruption evidence, never for mixing scripts, so a quotation in another language stays retrievable and comes back with its text unchanged.
- A chunk longer than the embedding model's token limit is embedded from its beginning only. BM25 still matches its full text, while dense search covers the start. Ingestion counts and reports these chunks (`dense_truncated`) but does not split them.
- Duplicate sources are a judgement call. The server never deletes an original; you review and exclude.
- Large CPU ingestions and reranking are slow. Ingestion is resumable, but one expensive page, the first model download, or the BM25 step can exceed the soft per-call budget. Reranking always runs for `search`, so the slow path is the only path; there is no faster switch to remember.
- Cleaned text is not a quote-verification surface — open the original.
- A running server keeps the code it started with, because a stdio server's process belongs to your MCP client. `status` reports `restart_required` when the process is older than what is installed, and the full-detail payload names the running and installed versions; restarting the server in the client clears it. The browser UI can be restarted from this side, and `scripts/update.sh <project>` does that for a named project.
- Earlier successful generations are kept. Automatic pruning is not implemented, so old generations accumulate until you remove them yourself.

If something looks wrong:

- The stdio executable appears to hang when run directly. It is waiting for an MCP client.
- `--offline` reports missing runtime or models. Run once online, or point `--model-cache-root` at a populated cache.
- `status` says the generation is stale. Search still uses the previous generation until a re-ingestion succeeds. That is intentional.
- Search returns nothing. This can be correct: the relevance gates prefer abstaining over returning weak matches. Try a precise BM25 query, or inspect dense mode, before lowering any quality expectation.
- You want the ranking or candidate detail behind a query, or the ingestion timings. Start the server with `--tool-detail full` (`RESEARCH_ULTRARAG_TOOL_DETAIL=full`) and inspect the complete payload.
- Import reports a project-ID conflict. The wrong `.research-rag/project.json` is in use. A source conflict means an existing file has different bytes. Neither safeguard should be bypassed.

## UltraRAG credit and licensing

This independent server directly uses UltraRAG and gratefully credits the UltraRAG team and contributor community. UltraRAG identifies itself as a joint project of [THUNLP](https://nlp.csai.tsinghua.edu.cn/), [NEUIR](https://neuir.github.io/), [OpenBMB](https://www.openbmb.cn/home), [AI9stars](https://github.com/AI9Stars), and [contributors](https://github.com/OpenBMB/UltraRAG/graphs/contributors).

The dependency is pinned to UltraRAG `0.3.0.2`, commit [`3a709a2aea3fbe46acca59c422621c94b6e86857`](https://github.com/OpenBMB/UltraRAG/tree/3a709a2aea3fbe46acca59c422621c94b6e86857). UltraRAG is distributed under the [Apache License 2.0](https://github.com/OpenBMB/UltraRAG/blob/3a709a2aea3fbe46acca59c422621c94b6e86857/LICENSE.txt). The installed upstream snapshot retains its license and notices.

This repository is not an official UltraRAG release and is not affiliated with or endorsed by OpenBMB, THUNLP, NEUIR, AI9stars, or UltraRAG contributors. See [`NOTICE`](NOTICE) for full upstream, model, retrieval-component, and UI attribution.

This repository's own code is licensed under the [Apache License 2.0](LICENSE), matching the pinned upstream's terms. That grant covers this repository's code only. Two extraction dependencies are licensed differently and matter to anyone redistributing or operating this server: PyMuPDF (AGPL-3.0, or an Artifex commercial licence) and EbookLib (AGPL-3.0-or-later). Their obligations apply separately from this repository's licence, and [`NOTICE`](NOTICE) records them. Nothing here is legal advice; confirm the combination that applies to your use.

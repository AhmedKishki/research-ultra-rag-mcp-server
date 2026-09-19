# Research UltraRAG MCP Server

## Purpose

This server helps human researchers work across large collections of PDF and
EPUB sources through an AI agent. It retrieves material for discovering
relationships, comparing authors, and finding passages that support, qualify,
or contradict a claim. Retrieval supplies evidence candidates; the researcher
and agent remain responsible for interpretation, source criticism, and writing.

> **Retrieved text is cleaned semantic text, not a quote-safe transcript.**
> Open the original PDF or EPUB at the returned locator before using a direct
> quotation.

The server is built on [UltraRAG](https://github.com/OpenBMB/UltraRAG), whose
corpus chunking and BM25 retrieval it uses. Full credit and licensing details
appear at the end of this document.

## What the server provides

- A CPU-first stdio MCP server for AI agents and a local browser UI.
- Recursive ingestion of regular `.pdf` and `.epub` files only. Markdown and
  all other formats are ignored.
- Resolved titles, authors, years, DOIs, per-field metadata provenance and
  warnings, and original-file PDF page or EPUB internal locators.
- BM25 lexical search, dense semantic search, hybrid search, metadata filters,
  optional CPU reranking, and an opt-in reference-grouped result view.
- Reviewed metadata corrections that apply immediately to indexed sources,
  reversible source inclusion/exclusion, and portable project bundles.
- Durable CPU ingestion checkpoints: bounded calls can be repeated after a
  client timeout, cancellation, service restart, or ordinary time-budget return.
- English-oriented corrupt-text and symbol-only rejection with source locators
  and reason codes; unreadable units are excluded without guessing replacement
  text, while foreign-script quotations are flagged with `text_notes` instead of
  being withheld.
- Immutable, project-local generations that become active only after both
  indexes pass.

It does not determine truth, autonomously delete or exclude sources, generate
answers on the server, or promise exact quotation transcripts. Its job is to
return structured evidence for an agent and researcher to assess.

## Install once

Requirements are Python 3.11 or 3.12, Linux for the currently tested setup,
and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/AhmedKishki/research-ultra-rag-mcp-server.git
cd research-ultra-rag-mcp-server
uv sync --frozen
```

No separate UltraRAG checkout, external Qdrant service, or virtual environment
inside each research project is required. Use the one `.venv` created in this
server repository for every project.

The pinned UltraRAG runtime is installed automatically. The embedding model is
downloaded on first ingestion; the optional reranker is downloaded on its first
use. Model binaries are shared by default at
`~/.cache/research-ultra-rag-mcp/models`, while document data remains isolated
inside each project. Override the model location with
`--model-cache-root /absolute/cache/path` or
`RESEARCH_ULTRARAG_MODEL_CACHE_ROOT`.

Use `--offline` only after the runtime and required model files have been
downloaded. In offline mode, an existing legacy project-local model cache can
still be read, but it is never moved or deleted automatically.

## Create an isolated research project

Put originals beneath a project root:

```text
my-research-project/
└── sources/
    ├── articles/
    │   └── article.pdf
    ├── book.epub
    └── notes.md                 # ignored
```

Only regular PDF and EPUB files are indexed. Discovery is recursive; source
symlinks are rejected.

One server process is bound to one absolute `--project-root`. It cannot search
another project. Multiple projects use the same installation and model cache,
but need separate MCP entries/processes and retain separate documents,
metadata, chunks, vectors, indexes, bundles, and logs.

On first use, the server creates `.research-rag/project.json` with a stable
project ID, project name, and project-relative source directory. The default is
`sources`. To choose another directory, pass, for example,
`--source-directory library` on first use. Later commands reuse the stored
setting; a conflicting setting is rejected.

## Connect an AI agent

Add an entry like this to the MCP configuration used by your client, such as
`mcp_settings.json`. Use absolute paths:

```json
{
  "mcpServers": {
    "research-corpus": {
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

The same installation can serve two isolated projects:

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

A complete template is in
[`mcp_settings.example.json`](mcp_settings.example.json). Running the stdio
executable directly appears idle because it is waiting for an MCP client to
send protocol messages. Keep write tools out of automatic approval initially:
ingestion, metadata decisions, exclusions, exports, and imports persist state
or create large files.

## First use

The normal agent sequence is:

1. Call `status`.
2. Report whether no generation exists, the selected generation is stale, or a
   schema/policy upgrade is required.
3. Obtain permission before a persistent ingestion when appropriate.
4. Call `ingest`; while it returns `status="in_progress"`, call it again with
   the same settings.
5. Search only after the completed generation is active.

Ready-to-copy prompt:

> Check the research knowledge-base status. Tell me whether a generation
> exists, is stale, or needs an upgrade. If ingestion is needed, explain what
> will be written and ask before doing it. After it succeeds, search the corpus
> for evidence about [YOUR QUESTION], including material that qualifies or
> contradicts the claim, and give me the original source paths and locators.

First ingestion may download the embedding model and can take minutes on a CPU;
duration depends mainly on corpus size and embedding work. The server builds in
project-local staging. Each call has a soft 45-second work budget and returns a
checkpointed `in_progress` result when more work remains. Repeat the same call
to continue. A single expensive page, first model download, or BM25 finalization
can exceed that soft budget. The server switches `current.json` only after the
complete BM25 and Qdrant indexes verify. Cancellation and timeout retain the
last atomic checkpoint; a non-resumable failure leaves the previous generation
selected, removes its partial staging data, and records a small failure report.

`list_sources` can be called before ingestion. Its `discovered_sources` array
lists every live PDF/EPUB with a project-scoped `source_id`, inclusion state,
and whether it is indexed in the selected generation. A `source_id` is derived
from the stable project ID and source-relative path: it survives changes to the
file's bytes, but a rename or move creates a new source ID. Prefer this ID when
an agent calls `set_source_metadata` or `set_source_inclusion`. A
`document_id`, by contrast, identifies one content/path version used by a
generation and can change after the source bytes or path changes.
Calling `list_sources` idempotently registers discovered IDs in the portable
project catalog. Its lean `known_sources` array therefore retains an ID-to-path
handle after an original is renamed, moved, or temporarily absent; it does not
silently transfer metadata from an old path to a new one.
`reviewed_metadata_sources` enumerates every saved override with the same
stable ID, including a source whose original is temporarily absent, so an agent
can inspect, replace, or remove every persisted metadata decision.

Both mutation tools require exactly one selector: `source_id` is the preferred
form, while `source_path` remains available for compatibility.

## Research workflow

Useful prompts include:

- “Find the strongest source passages supporting the claim that …”
- “Search broadly across references; return at most two passages from any one
  source within the total passage budget.”
- “Search for evidence that contradicts or qualifies this claim: …”
- “Compare how these sources explain …; distinguish agreement from conflict.”
- “Get the neighboring passages around chunk `chk_…` before interpreting it.”
- “Open the original at the returned path and page before quoting it.”
- “List sources with missing metadata or extraction warnings. Treat automatic
  metadata as provisional; show me its provenance, then use the source ID with
  `set_source_metadata` after I review the original. Apply the correction now
  without re-ingesting.”
- “List the source IDs, then exclude the duplicate source ID as a reviewed
  duplicate of the preferred source ID; do not delete either file.”
- “Restore that source ID, then re-ingest if it is absent from the current
  generation.”

Hybrid is the normal search mode. Use BM25 diagnostically for exact names,
terms, and phrases; use dense search to inspect conceptual similarity. Optional
reranking is slower and is best reserved for a candidate set that needs another
ordering pass.

Search can correctly return fewer than `top_k`, including zero, when candidates
fail relevance gates. A rank or similarity score is an ordering signal, not a
truth or confidence probability.

The default `result_view="passages"` is the unchanged global passage ranking.
Use `result_view="references"` when one prolific source would otherwise occupy
the result budget. It scans the relevance-gated candidate ordering, admits at
most `passages_per_reference` passages from each stable `source_id`, and keeps
`top_k` as the total number of passages returned. The response reports explicit
returned and candidate-pool reference counts; it does not merge editions by
title, DOI, or filename. `relevance_limited` reports a candidate-pool shortfall;
`grouping_limited` separately reports when the per-reference cap prevents the
view from filling that passage budget.

Adding, removing, or changing source content—or changing source inclusion—can
make the selected generation stale. Reviewed metadata is different: for a
source already in the selected generation, `set_source_metadata` immediately
updates source listings, filters, search results and citations, and neighboring
passages. It does not rewrite the immutable generation or change chunk IDs, and
a metadata-only difference is reported separately through the metadata overlay
and snapshot fields rather than as stale retrieval state. Metadata for a source
absent from the selected generation is saved but requires ingestion before that
source can appear. **Re-ingest
changes** verifies every source hash, reuses compatible documents/chunks/vectors,
and reconstructs both complete indexes. **Regenerate** sets
`force_recompute=true` and deliberately bypasses all reuse.

Extraction rejects whole corrupt pages/sections only when their text carries
strong evidence of a broken character map: replacement characters, private-use
or unassigned code points, or a known damaged encoding sequence. Diagnostics
retain only the source locator and reason codes, not the rejected garbage. If no
readable unit remains in a source, the build fails so the original can be
repaired or OCRed. Formula-font letters (`𝑀` → `M`) and the presentation
ligatures `ﬁ`, `ﬂ`, and `ﬀ` are folded to their plain spellings so a typed query
matches the printed text, while accented letters, superscripts, subscripts, and
symbols are left unchanged. Script mixing and non-Latin dominance are never
rejection reasons; they are reported as advisory notes so a quotation in another
language stays retrievable. Reviewed metadata is not overridden by this automatic
classifier. Ingestion also excludes nonempty chunks
that contain no Unicode alphanumeric content. Ordinary text, numeric content,
and formulas containing at least one letter or digit are not classified as
symbol-only; the existing extraction-artifact checks still apply.

## Use the UI

From the installed server repository:

```bash
uv run research-ultra-rag-ui \
  --project-root /absolute/path/to/my-research-project
```

Open [http://127.0.0.1:5051](http://127.0.0.1:5051) manually if a browser does
not open. The UI binds only to loopback and calls the same nine public MCP tools
against the same project state as an agent.

The UI provides:

- status, staleness, upgrade, and build-metric inspection;
- the indexed source list and resolved bibliography;
- hybrid, BM25, or dense search with filters and optional reranking;
- neighboring passage context;
- opening a PDF in the browser or downloading an EPUB original;
- reviewed metadata editing that applies immediately to indexed sources;
- reviewed source exclusion and restoration without deleting originals;
- **Create generation** for the first normal build;
- **Re-ingest changes** for verified reuse after project changes;
- **Regenerate** for forced extraction, chunking, and embedding recomputation;
- bundle export; and
- import of a bundle already placed in `.research-rag/bundles/`.

One UI process serves one project. To keep two open simultaneously, launch a
second process with another project root and port, for example `--port 5052`.
During ingestion, the UI shows the operation as busy. A project lock serializes
agent and UI operations, so another operation waits rather than observing a
partially built index. The adapter automatically follows checkpointed
`in_progress` responses until the generation is ready or unchanged.

## Use the terminal verifier

Read-only status and search verification:

```bash
uv run research-ultra-rag-verify \
  /absolute/path/to/my-research-project \
  --query "commodity fetishism and artificial intelligence"
```

Normal first ingestion or re-ingestion:

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

Choose `--retrieval-method bm25|dense|hybrid`, add `--rerank`, change
`--top-k`, or use `--result-view references --passages-per-reference 2` for
source-diverse results. Add `--offline` when all required caches exist. Success
prints a JSON object with `"status": "passed"`, before/after status, optional
ingestion metrics, and the search result. Common first-run failures are an
unavailable network/model download, an unsupported Python version, a missing or
corrupt PDF/EPUB, or `--offline` before the runtime/model cache exists. With
`--ingest`, the verifier automatically repeats checkpointed calls until
ingestion finishes.

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

`sources/` remains the authority for exact quotation. `.research-rag/` is the
single root for this server's project state. Its top-level JSON files and
`bundles/` are portable; `runtime/` can be regenerated from the sources and
portable state. Document text, embeddings, indexes, query/runtime state, and
logs never cross project roots; only immutable model binaries are shared.

On first use after upgrading, an existing `.ultrarag/research/` directory is
moved automatically to `.research-rag/runtime/`. If both locations already
contain runtime data, startup stops rather than choosing one. Other
`.ultrarag/` contents belonging to different tools are not changed. Stop all
running research MCP and UI processes before the first launch with the upgraded
package so no process continues writing to the legacy location.

`current.json` points to the one generation used by search. Earlier successful
generations remain on disk, but are not searched. Automatic generation pruning
is not implemented.

## Export, import, and move a project

An agent can call `export_bundle` and `import_bundle`; the UI exposes matching
controls. Terminal commands are:

```bash
uv run research-ultra-rag-bundle export \
  --project-root /absolute/path/to/my-research-project

uv run research-ultra-rag-bundle import \
  --project-root /absolute/path/to/my-research-project \
  --bundle /path/to/project-generation.research-rag.zip
```

MCP and UI import accept a filename already placed directly beneath
`.research-rag/bundles/`. The terminal command accepts an external archive,
copies it safely into that directory, then invokes the same MCP import.

A bundle contains every original PDF/EPUB—including excluded sources—the
project descriptor, reviewed metadata/exclusions, generation manifest, cleaned
semantic units/chunks, and float32 embeddings in stable chunk order. It excludes
live BM25/Qdrant databases, locks, logs, temporary files, runtime files, and
model caches. Import validates archive paths and entry types, checksums,
project ID, schemas, embedding compatibility, and every original. It refuses to
overwrite an existing path with different bytes. It reconstructs complete BM25
and Qdrant indexes without re-extraction or re-embedding, then changes
`current.json` last when activation is requested.

To move to another device:

1. Preserve the original `.research-rag/project.json` so the stable project ID
   remains the same.
2. Transfer the bundle and its `.sha256` sidecar.
3. Install this package once on the destination.
4. Put the project descriptor in the new project, then run the terminal import
   command. Future searches still need the pinned embedding model to encode
   queries, so allow its download or prepare the shared cache before offline use.

> **Bundles contain complete original works and derived text. You are
> responsible for having the right to redistribute every bundled source.**

For Git, commit `.research-rag/project.json`, reviewed metadata/exclusions, and
only legally shareable sources. Ignore `.research-rag/runtime/`. Track PDFs,
EPUBs, and bundle archives with Git LFS. Usually share either standalone
originals for regeneration or a source-containing bundle, not both, unless
duplication is intentional.

```gitattributes
*.pdf filter=lfs diff=lfs merge=lfs -text
*.epub filter=lfs diff=lfs merge=lfs -text
*.research-rag.zip filter=lfs diff=lfs merge=lfs -text
```

## Complete MCP tool reference

| Tool | Parameters and defaults | Access | When to call it |
|---|---|---|---|
| `status` | none | Read | Before research or ingestion; reports project identity, current generation, source changes, whether `metadata_overlay_active`, any `metadata_pending_source_paths`, upgrade reasons, model-cache path, last build metrics, and any `ingestion_progress`. |
| `ingest` | `chunk_size=384` (50–384 GPT-2 tokens); `chunk_overlap=64` (0 to `chunk_size-1`); `force_recompute=false`; `work_budget_seconds=45` (10–300) | Write | First build, stale collection refresh, schema upgrade, or deliberate forced regeneration. Repeat matching calls while the result is `in_progress`; `ready` and `unchanged` are terminal. |
| `search` | required `query`; `top_k=8` (1–50 total passages); `categories=null`; `keywords=null`; `document_ids=null`; `retrieval_method="hybrid"`; `rerank=false`; `result_view="passages"`; `passages_per_reference=2` (1–5) | Read | Retrieve relevance-limited evidence. The optional reference view caps passages per stable `source_id` and returns `reference_groups`; filters retain their existing semantics. |
| `list_sources` | `categories=null`; `keywords=null` | Idempotent project-state write | Inspect indexed bibliography and stable source IDs, filter by reviewed metadata, and register live IDs in the portable catalog so `known_sources` remains addressable if a file later disappears. It never changes an original or a generation. |
| `get_passage` | required `chunk_id`; `context_chunks=1` (0–5 on each side) | Read | Inspect nearby cleaned passages from the same source and generation. |
| `set_source_metadata` | required `metadata` object; exactly one of stable `source_id` or source-relative `source_path` | Write | Replace the reviewed `title`, `authors`, `year`, `doi`, `categories`, and/or `keywords` override. It applies immediately when the source is indexed—even if the original is temporarily absent—and reports `effective_metadata`, `effective_immediately`, and `requires_ingest`; omitted fields remove previous overrides. |
| `set_source_inclusion` | required `included`; exactly one of stable `source_id` or source-relative `source_path`; `reason=null` | Write | Exclude or restore a reviewed source without changing the original. A non-empty reason is required for exclusion. |
| `export_bundle` | none | Write | Export the selected fresh, upgrade-compatible generation and all originals beneath `.research-rag/bundles/`. |
| `import_bundle` | required `bundle_name`; `activate=true` | Write | Validate a project-local archive, install non-conflicting originals/state, reconstruct indexes, and optionally select it. Portable state becomes authoritative immediately; when a generation is already selected, matching imported metadata and exclusions govern its retrieval even with `activate=false`, while its pointer stays unchanged. |

For example, a post-ingestion metadata correction can be sent directly through
the MCP tool:

```json
{
  "source_id": "src_0123456789abcdef01234567",
  "metadata": {
    "title": "Corrected Title",
    "authors": ["Reviewed Author"],
    "categories": ["political economy"]
  }
}
```

When that source is in the selected generation, the response includes
`"effective_immediately": true`, `"requires_ingest": false`, and the resolved
`effective_metadata` so an agent can verify the change in the same call. The
metadata object is the complete reviewed override, not a patch: omitting a
field removes its previous reviewed value. An empty object restores every
automatic value; an explicit empty value (`""`, `[]`, or `null` where accepted)
deliberately clears that automatic field.

A representative abbreviated search response is:

```json
{
  "query": "supply chains and labour precarity",
  "result_view": "passages",
  "requested_top_k": 8,
  "result_count": 1,
  "distinct_reference_count": 1,
  "relevance_limited": true,
  "grouping_limited": false,
  "hits": [
    {
      "rank": 1,
      "source_id": "src_0123456789abcdef01234567",
      "document_id": "doc_0123456789abcdef01234567",
      "title": "Supply Chains and the Human Condition",
      "authors": ["Anna Tsing"],
      "source_path": "sources/tsing.pdf",
      "locator": {"type": "pdf_page", "page": 4, "page_label": "4"},
      "text": "Cleaned semantic text from the relevant passage...",
      "direct_quote_safe": false,
      "match_kind": "hybrid",
      "component_ranks": {"bm25": 2, "dense": 1},
      "component_scores": {"dense_cosine_similarity": 0.79, "bm25": null},
      "fusion_score": 0.0363,
      "rerank_score": null
    }
  ]
}
```

Researchers normally use `source_id`, `title`, `authors`, `source_path`, and
`locator` to identify the original; `text` to assess semantic relevance;
`match_kind` to see which route found it; and component ranks/scores to
understand ordering. `document_id` identifies the indexed source version, not
the durable handle for metadata edits. BM25 does not expose a comparable raw
score in this integration, so it reports rank only. Reference view retains the
selected passages in `hits` and additionally groups them beneath
`reference_groups` in first-best-passage order.

## How it works under the hood

```text
project PDF/EPUB files
        │
        ▼
research extraction ── bibliography + layout + original locators
        │
        ▼
UltraRAG GPT-2 token chunking
        ├──────────────► UltraRAG BM25
        └──► FastEmbed CPU vectors ──► embedded project-local Qdrant
                                      │
                         weighted rank fusion
                                      │
                                      ▼
                         structured MCP evidence
```

The research layer selects allowed files, extracts layout-aware semantic units,
resolves bibliography, cleans layout artifacts, and preserves locators.
UltraRAG performs GPT-2 token chunking and BM25 indexing/search. FastEmbed
produces revision-pinned CPU embeddings. Embedded Qdrant stores vectors with
only lean lookup payloads (`chunk_id`, `document_id`, and `source_id`) inside
the generation; canonical passage content and locators stay in `chunks.jsonl`,
while document metadata and provenance stay in the manifest. A compact SQLite
sidecar stores only IDs, content hashes, vector ordinals, and JSONL byte offsets
so query and reuse paths load selected records without copying corpus text into
another artifact. Weighted
reciprocal-rank fusion combines BM25 weight `1.25` and dense weight `1.0`. The
MCP returns structured passages; the calling agent performs interpretation and
answer generation.

Generations are immutable. Reviewed metadata lives outside them as portable
project state and is overlaid at read time. Search, source listing, neighboring
passages, citations, and category/keyword filters therefore use the current
reviewed values immediately without rewriting chunk, BM25, vector, or Qdrant
files. For dense filtering, current metadata is resolved to document IDs within
the selected generation instead of trusting stale copied metadata.

Before ingestion decides what to do, it hashes every discovered PDF/EPUB and
fingerprints exclusions, chunk settings, processing policies, and the embedding
model. Reviewed metadata is intentionally excluded from generation and
checkpoint identity because it does not change source text or ranking indexes.
An exact match is a true no-op.
Otherwise, compatible unchanged documents retain their extracted units and
chunks; an exact canonical `contents` value plus model fingerprint can retain
its vector. Final chunk records keep one semantic-text field, `contents`, plus
identity, locator, and structural fields. Reading legacy `text` or
`embedding_text` fields is compatibility behavior, not the current artifact
format.
Changed material is recomputed. The server always reconstructs complete new
BM25 and Qdrant indexes for a changed generation and atomically switches the
pointer only after verification. Source hashes, fixed eight-page PDF
scan/extraction batches, EPUB spine sections, extraction-unit chunking,
64-passage embedding batches, and 64-point Qdrant uploads commit restartable
atomic units; UltraRAG BM25 is a restartable finalization step. Every source is
hashed again before activation.
`force_recompute=true` bypasses reuse while still resuming its own matching
checkpoint.

Upstream extraction is not used because this research contract needs
layout-aware PDF/EPUB handling, bibliographic provenance, and original locators.
Upstream dense output is not used because this server needs scored results,
payload filters, portable vectors, and exact reuse accounting. Those
research-specific responsibilities remain thin layers around UltraRAG rather
than changes to its source.

Bibliographic fields resolve independently. Automatic PDF metadata uses valid
visible front matter before valid embedded metadata, with the filename stem as
a title-only fallback. Automatic EPUB metadata uses valid OPF values before
visible title/byline values, again with the filename stem as a title-only
fallback. Reviewed values are an authoritative per-field overlay on either
result. Authors are never inferred from filenames, and DOI-like titles are
moved to the DOI field. Provenance names the source selected for each field;
warnings identify concrete missing, conflicting, fallback, or corrupt values.

Automatic bibliography is deliberately best-effort and uses only general
signals. It contains no document-, author-, or publisher-specific exceptions.
Correct an uncertain or wrong field with `set_source_metadata`; the reviewed
value has highest precedence and applies immediately if the source is indexed.
Omitted fields remove previous reviewed overrides. When an older generation
cannot recover an automatic bibliographic value that was hidden by the removed
override, it uses a safe missing value—or the filename for title—and reports
`automatic_metadata_unavailable_after_override_removal`. A later ingestion can
recover automatic metadata from the original, but is not required for supplied
reviewed values to work.

Hybrid search uses weighted reciprocal-rank fusion with `k=60`. BM25 candidates
must contain a non-stopword query token; dense candidates require cosine
similarity of at least `0.72`; known extraction artifacts are rejected before
optional reranking. Corrupt-text checks also run during retrieval, so older
generations stop returning rejected text before re-ingestion removes it from
their successors. Every result discloses what was withheld in
`withheld_candidates`, including the reason codes and example chunk IDs, and every
returned passage carries `text_notes` when it mixes scripts. The same retrieval
guard excludes symbol-only chunks from older generations immediately. The pinned
embedding model is
`BAAI/bge-small-en-v1.5` (384 dimensions), and the optional reranker is
`Xenova/ms-marco-MiniLM-L-6-v2`.

Reference grouping is applied only after retrieval, relevance gates, fusion,
and optional reranking. It groups strictly by stable `source_id`, preserves
the ranked order of admitted passages, and reports how many candidates were
skipped by the per-reference cap. A checked-in graded source-level judgment
fixture guards the default cap against both single-source crowding and blind
maximal diversification; it is a regression evaluation, not a claim of
project-specific retrieval quality.

## Limitations and troubleshooting

- Scanned PDFs require OCR before ingestion. Password-protected PDFs are
  rejected.
- PDF locators use physical pages plus a page label when available. Reflowable
  EPUBs have spine-section plus exact existing fragment or deterministic XHTML
  block locators, not stable page numbers. These improve navigation but are not
  quote offsets or synthesized CFIs.
- The embedding and reranking models are English-oriented.
- The text-health policy is intentionally English-oriented. It withholds a
  passage only for corruption evidence and never for mixing scripts, so a
  quotation in another language stays retrievable and is returned with
  `text_notes`. It does not perform OCR or attempt encoding repair.
- English is the supported language and workload. Non-English-primary corpora,
  OCR or scanned sources, handwriting, and formula-heavy corpora are outside the
  designed scope.
- Duplicate decisions require agent/user review; the server never deletes the
  original.
- Large CPU ingestions and optional reranking can be slow. Ingestion is
  resumable, but one expensive page, initial model download, or BM25 step can
  exceed the soft per-call budget.
- Cleaned text is not an exact-quote verification surface.
- Earlier successful generations are retained; automatic pruning is absent.

If the server appears stuck when run directly, it is waiting for an MCP client.
If `--offline` reports missing runtime or models, run once online or point
`--model-cache-root` to a populated cache. If `status` is stale, search still
uses the prior generation until re-ingestion succeeds. Empty search results can
be correct; try a precise BM25 query or inspect dense mode before lowering any
quality expectation. Import project-ID conflicts mean the wrong
`.research-rag/project.json` is in use; source conflicts mean an existing path
has different bytes. Neither safeguard should be bypassed.

## UltraRAG credit and licensing

This independent server directly uses UltraRAG and gratefully credits the
UltraRAG team and contributor community. UltraRAG identifies itself as a joint
project of [THUNLP](https://nlp.csai.tsinghua.edu.cn/),
[NEUIR](https://neuir.github.io/), [OpenBMB](https://www.openbmb.cn/home),
[AI9stars](https://github.com/AI9Stars), and
[contributors](https://github.com/OpenBMB/UltraRAG/graphs/contributors).

The dependency is pinned to UltraRAG `0.3.0.2`, commit
[`3a709a2aea3fbe46acca59c422621c94b6e86857`](https://github.com/OpenBMB/UltraRAG/tree/3a709a2aea3fbe46acca59c422621c94b6e86857).
UltraRAG is distributed under the
[Apache License 2.0](https://github.com/OpenBMB/UltraRAG/blob/3a709a2aea3fbe46acca59c422621c94b6e86857/LICENSE.txt).
The installed upstream snapshot retains its license and notices.

This repository is not an official UltraRAG release and is not affiliated with
or endorsed by OpenBMB, THUNLP, NEUIR, AI9stars, or UltraRAG contributors. See
[`NOTICE`](NOTICE) for full upstream, model, retrieval-component, and UI
attribution.

"""Layered, file-backed settings: one registry, five precedence layers.

No tunable is hard-coded here. `default.toml`, which ships inside this package,
holds the values the server uses when nobody says otherwise, and every layer
above it names only what it changes. Later layers win **per key**:

    default.toml  <  user config  <  project config  <  environment  <  command line

A layer may therefore define a handful of keys, and the rest are still inherited
from the layer below. A layer may equally define a whole section and replace it.

`SETTINGS` is the registry. Each entry declares the key, the type and bounds the
value must satisfy, which layer class it belongs to, and the name it takes in the
environment and on the command line. Two rules follow from it:

* a key that is not in the registry is an error, in every layer, so a typo is
  refused instead of being ignored;
* a setting whose class is ``identity`` changes what a generation *is*, so its
  value enters the retrieval-policy fingerprint and changing it means the next
  ingestion is a new generation rather than a silent mix of two.

`AGENTS.md` records the constants that deliberately stay in code, because a
generation's identity and the security boundary must not be configurable.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import platformdirs

from .embeddings import (
    EMBEDDING_MODEL_CHOICES,
    EmbeddingModel,
    models_covering,
    resolve_embedding_model,
)
from .rerankers import RERANKER_MODELS

Layer = Literal["identity", "engine", "runtime"]

# The layer names, lowest first. `provenance` reports these strings, so
# `--print-config` says where each effective value came from.
LAYER_DEFAULT = "default"
LAYER_USER = "user config"
LAYER_PROJECT = "project config"
LAYER_FILE = "--config"
LAYER_ENVIRONMENT = "environment"
LAYER_COMMAND_LINE = "command line"

DEFAULT_CONFIG_FILENAME = "default.toml"
USER_CONFIG_DIRECTORY = "research-ultra-rag-mcp"
PROJECT_CONFIG_RELATIVE = Path(".research-rag") / "config.toml"
SETTINGS_ENVIRONMENT_PREFIX = "RESEARCH_ULTRARAG_"


class SettingsError(ValueError):
    """Raised when a settings layer is unreadable, unknown, or out of bounds."""


@dataclass(frozen=True, slots=True)
class Setting:
    """One tunable: where it lives, what it accepts, and how it is named.

    ``layer`` is the contract that keeps configurability honest:

    * ``identity`` — the value decides what a generation contains, so it enters
      the retrieval-policy fingerprint and a change invalidates reuse.
    * ``engine`` — the value chooses an engine component (a backend, a model)
      whose identity is recorded in the generation manifest and in answers.
    * ``runtime`` — the value shapes this process and its answers only (threads,
      budgets, logging, tool detail, where the shared model cache lives, and the
      source-diversity penalty, which reorders an answer after ranking) and
      cannot affect an artifact.
    """

    key: str
    field: str
    kind: type
    layer: Layer
    doc: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    env: str = ""
    flag: str = ""
    section: str = ""
    # A setting that names one of a fixed set of modes also accepts any spelling
    # of them. A setting that names a model, a path, or a revision does not: those
    # are case-sensitive identifiers, not mode words.
    normalize_case: bool = False

    def coerce(self, raw: Any, *, source: str) -> Any:
        """Return ``raw`` as this setting's type, or refuse it by name."""

        where = f"{self.key} ({source})"
        if isinstance(raw, bool) and self.kind is not bool:
            raise SettingsError(f"{where} must be {self.kind.__name__}, not a boolean")
        if self.kind is bool:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str) and raw.strip().casefold() in {"true", "false"}:
                return raw.strip().casefold() == "true"
            raise SettingsError(f"{where} must be true or false")
        if self.kind is int and isinstance(raw, bool):
            raise SettingsError(f"{where} must be an integer, not a boolean")
        try:
            value = self.kind(raw)
        except (TypeError, ValueError) as exc:
            if isinstance(raw, str) and not raw.strip() and self.kind is not str:
                # An empty string in a file means "leave this to the runtime",
                # which only the settings that accept it can express.
                raise SettingsError(f"{where} must be {self.kind.__name__}") from exc
            raise SettingsError(f"{where} must be {self.kind.__name__}") from exc
        if self.kind is str:
            # Surrounding whitespace is never meaningful in a settings value.
            value = value.strip()
            if self.normalize_case:
                value = value.casefold()
        if self.choices and value not in self.choices:
            raise SettingsError(f"{where} must be one of: " + ", ".join(self.choices))
        if self.minimum is not None and value < self.minimum:
            raise SettingsError(f"{where} must be at least {self.minimum:g}")
        if self.maximum is not None and value > self.maximum:
            raise SettingsError(f"{where} must be at most {self.maximum:g}")
        return value


# Tool answer detail. `lean` is what the MCP tools return; `full` is the
# developer debugging mode and returns the service payload unchanged.
LEAN_TOOL_DETAIL = "lean"
FULL_TOOL_DETAIL = "full"
TOOL_DETAIL_MODES = (LEAN_TOOL_DETAIL, FULL_TOOL_DETAIL)
LOG_LEVELS = ("debug", "info", "warn", "error")
DENSE_BACKENDS = ("auto", "exact", "qdrant")
DEFAULT_LANGUAGE = "en"
LANGUAGE_PATTERN = r"^[a-z]{2,3}$"
# The longest one ingest call may be told to run. A caller driving a build in one
# call sets the budget below its own client's request timeout; a caller that wraps
# this server in another process has to allow longer than this, or the wrapper's
# timeout fires first and the work continues unseen.
MAXIMUM_WORK_BUDGET_SECONDS = 3600

# bm25s ships a stopword list for exactly these languages and rejects every other
# name, so a corpus language outside this set has to fail here: the BM25 index is
# built after extraction and embedding, and a build that dies there has already
# spent an hour on work it cannot keep.
BM25_STOPWORD_LANGUAGES = frozenset(
    {"en", "de", "nl", "fr", "es", "pt", "it", "ru", "sv", "no", "zh", "tr", "ko"}
)


def bm25_stopwords(language: str) -> frozenset[str] | None:
    """Return one language's BM25 stopword list, or None when there is none.

    The list is the one bm25s itself filters with, so a detector built on it
    scores the same function words the lexical half will ignore. None means
    bm25s is absent or does not know the language: a caller treats that as
    "not detectable here", never as a wrong answer.
    """

    code = str(language).strip().casefold()
    if code not in BM25_STOPWORD_LANGUAGES:
        return None
    try:
        from bm25s.tokenization import _infer_stopwords
    except ImportError:
        return None
    try:
        return frozenset(str(item).casefold() for item in _infer_stopwords(code))
    except ValueError:
        return None


def normalize_corpus_languages(value: Any) -> tuple[str, ...]:
    """Parse `language.corpus` into the languages a corpus is written in.

    One code is the common case and stays exactly what it was, so a corpus in a
    single language spells its setting the way it always did. Several codes
    describe a corpus in more than one language. The order is kept as written
    rather than sorted, because the value is part of the ranking policy and
    `de,en` and `en,de` are the same corpus only if nothing reads the order.
    """

    if isinstance(value, (list, tuple)):
        raw = [str(item) for item in value]
    else:
        raw = str(value).split(",")
    languages: list[str] = []
    for item in raw:
        code = item.strip().casefold()
        if not code:
            raise SettingsError(f"language.corpus has an empty language: {value!r}")
        if not re.fullmatch(LANGUAGE_PATTERN, code):
            raise SettingsError(
                "language.corpus must be one or more two- or three-letter "
                f"ISO 639-1 codes separated by commas: {value!r}"
            )
        if code not in BM25_STOPWORD_LANGUAGES:
            raise SettingsError(
                f"language.corpus has no BM25 stopword list: {code!r}. "
                "Supported: "
                + ", ".join(sorted(BM25_STOPWORD_LANGUAGES))
                + ". BM25 needs one of those, or it fails after the corpus has "
                "already been extracted and embedded."
            )
        if code not in languages:
            languages.append(code)
    return tuple(languages)


def _bm25_stopword_roundtrip_error(language: str) -> str | None:
    """Return why a stopword list cannot survive bm25s's own save/load, if any.

    bm25s 0.3.10 wrote its stopwords file with a non-JSON escape for non-ASCII
    characters, so a German corpus failed in the BM25 index phase after
    extraction and embedding had already finished. This check exercises the
    *installed* bm25s's serializer, so it turns that failure into a settings-time
    error and is self-healing once a fixed bm25s is pinned.
    """

    try:
        from bm25s.tokenization import _infer_stopwords
        from bm25s.utils import json_functions
    except ImportError:
        return None  # bm25s is not installed here; the build would fail anyway.

    try:
        stopwords = _infer_stopwords(language)
    except ValueError:
        return None  # already rejected as an unknown language earlier.

    try:
        json_functions.loads(json_functions.dumps(list(stopwords)))
    except Exception as exc:  # noqa: BLE001 - any parse error is the same failure.
        return (
            f"language.bm25_stopwords {language!r} has a stopword list the "
            f"installed bm25s cannot re-read after writing it ({exc}). This "
            "would fail the BM25 index after extraction and embedding. Pin a "
            "fixed bm25s, or choose a language whose list round-trips (English "
            "does)."
        )
    return None


# The registry, in the order `--print-config` prints it. Every value in
# `default.toml` is validated against this table, and a `--set` or environment
# name that is not here is refused.
SETTINGS: tuple[Setting, ...] = (
    Setting(
        key="language.corpus",
        field="language_corpus",
        kind=str,
        layer="identity",
        doc=(
            "The languages the corpus is written in, as ISO 639-1 codes: one "
            "code, or several separated by commas for a corpus in more than one "
            "language. A language BM25 cannot tokenize is refused, because the "
            "stopword list comes from it, and every language named has to be "
            "covered by the embedding model."
        ),
        env="RESEARCH_ULTRARAG_LANGUAGE_CORPUS",
    ),
    Setting(
        key="language.bm25_stopwords",
        field="bm25_stopwords",
        kind=str,
        layer="identity",
        doc=(
            "Which language's stopword list BM25 filters with. Empty means the "
            "first language in language.corpus, because BM25 takes a single list "
            "and a corpus in several languages has to point it at one of them."
        ),
        env="RESEARCH_ULTRARAG_LANGUAGE_BM25_STOPWORDS",
    ),
    Setting(
        key="runtime.offline",
        field="offline",
        kind=bool,
        layer="runtime",
        doc=(
            "Require an installed vanilla runtime and already-cached models "
            "instead of downloading anything."
        ),
        env="RESEARCH_ULTRARAG_OFFLINE",
    ),
    Setting(
        key="runtime.log_level",
        field="log_level",
        kind=str,
        layer="runtime",
        normalize_case=True,
        doc="Verbosity of this process's own logging.",
        choices=LOG_LEVELS,
        env="RESEARCH_ULTRARAG_LOG_LEVEL",
    ),
    Setting(
        key="runtime.tool_detail",
        field="tool_detail",
        kind=str,
        layer="runtime",
        normalize_case=True,
        doc=(
            "Tool answer detail: 'lean' is what an agent gets, 'full' is the "
            "developer debugging payload."
        ),
        choices=TOOL_DETAIL_MODES,
        env="RESEARCH_ULTRARAG_TOOL_DETAIL",
    ),
    Setting(
        key="runtime.embedding_threads",
        field="embedding_threads",
        kind=int,
        layer="runtime",
        doc=(
            "ONNX Runtime threads for the embedding model; 0 leaves the choice "
            "to the runtime."
        ),
        minimum=0,
        maximum=1024,
        env="RESEARCH_ULTRARAG_EMBEDDING_THREADS",
    ),
    Setting(
        key="runtime.nice",
        field="nice",
        kind=int,
        layer="runtime",
        doc=(
            "CPU niceness for this process and every child it starts; 0 leaves "
            "priority unchanged, and a higher value keeps the machine responsive "
            "during a long build by yielding to whatever else is running."
        ),
        minimum=0,
        maximum=19,
        env="RESEARCH_ULTRARAG_NICE",
    ),
    Setting(
        key="runtime.model_cache_root",
        field="model_cache_root",
        kind=str,
        layer="runtime",
        doc=(
            "Shared FastEmbed model cache; empty means the per-user cache "
            "directory for this application."
        ),
        env="RESEARCH_ULTRARAG_MODEL_CACHE_ROOT",
    ),
    # --- Retrieval: what the fused ranking is, and what it will not accept. ---
    Setting(
        key="retrieval.rrf_k",
        field="rrf_k",
        kind=int,
        layer="identity",
        doc="Reciprocal-rank-fusion constant: higher flattens the rank curve.",
        minimum=1,
        maximum=1000,
        env="RESEARCH_ULTRARAG_RETRIEVAL_RRF_K",
    ),
    Setting(
        key="retrieval.bm25_weight",
        field="bm25_weight",
        kind=float,
        layer="identity",
        doc="Weight of the BM25 rank in the fusion.",
        minimum=0.0,
        maximum=10.0,
        env="RESEARCH_ULTRARAG_RETRIEVAL_BM25_WEIGHT",
    ),
    Setting(
        key="retrieval.dense_weight",
        field="dense_weight",
        kind=float,
        layer="identity",
        doc="Weight of the dense rank in the fusion.",
        minimum=0.0,
        maximum=10.0,
        env="RESEARCH_ULTRARAG_RETRIEVAL_DENSE_WEIGHT",
    ),
    Setting(
        key="retrieval.minimum_candidates",
        field="minimum_candidates",
        kind=int,
        layer="identity",
        doc=(
            "Fewest fused candidates a search considers, before relevance gates "
            "and reranking."
        ),
        minimum=1,
        maximum=1000,
        env="RESEARCH_ULTRARAG_RETRIEVAL_MINIMUM_CANDIDATES",
    ),
    Setting(
        key="retrieval.maximum_candidates",
        field="maximum_candidates",
        kind=int,
        layer="identity",
        doc="Most fused candidates a search considers.",
        minimum=1,
        maximum=5000,
        env="RESEARCH_ULTRARAG_RETRIEVAL_MAXIMUM_CANDIDATES",
    ),
    Setting(
        key="retrieval.dense_minimum_cosine_similarity",
        field="dense_minimum_cosine_similarity",
        kind=float,
        layer="identity",
        doc=(
            "Dense relevance gate: a candidate below this cosine similarity is "
            "withheld rather than ranked. Measured: a fused ranking is "
            "insensitive to the values below this default and loses answers "
            "above it, while a dense-only ranking wants a much lower value."
        ),
        minimum=-1.0,
        maximum=1.0,
        env="RESEARCH_ULTRARAG_RETRIEVAL_DENSE_MINIMUM_COSINE_SIMILARITY",
    ),
    Setting(
        key="retrieval.rerank_max_candidates",
        field="rerank_max_candidates",
        kind=int,
        layer="identity",
        doc=(
            "Most candidates the cross-encoder reorders by score; the unranked "
            "tail is appended after them so a reference group is still reachable."
        ),
        minimum=1,
        maximum=5000,
        env="RESEARCH_ULTRARAG_RETRIEVAL_RERANK_MAX_CANDIDATES",
    ),
    Setting(
        key="retrieval.rerank_window_multiple",
        field="rerank_window_multiple",
        kind=int,
        layer="identity",
        doc=(
            "Depth of the reranked window as a multiple of the requested top_k. "
            "The window is max(top_k * this, rerank_window_floor), capped by "
            "rerank_max_candidates and the fused candidate count. Measured: 20 "
            "is the shallowest window that reaches the plateau, and each ten "
            "more candidates cost about 0.7 s per query."
        ),
        minimum=1,
        maximum=1000,
        env="RESEARCH_ULTRARAG_RETRIEVAL_RERANK_WINDOW_MULTIPLE",
    ),
    Setting(
        key="retrieval.rerank_window_floor",
        field="rerank_window_floor",
        kind=int,
        layer="identity",
        doc=(
            "Fewest candidates the cross-encoder reorders, whatever top_k asks "
            "for, so a shallow request still ranks a useful group."
        ),
        minimum=1,
        maximum=5000,
        env="RESEARCH_ULTRARAG_RETRIEVAL_RERANK_WINDOW_FLOOR",
    ),
    Setting(
        key="retrieval.prf",
        field="prf",
        kind=bool,
        layer="identity",
        doc=(
            "Pseudo-relevance feedback: mine terms from the lexical leaders and "
            "search again with them, so a question that does not use the "
            "author's words still reaches the passages that do. Off by default "
            "until it is measured."
        ),
        env="RESEARCH_ULTRARAG_RETRIEVAL_PRF",
    ),
    Setting(
        key="retrieval.prf_documents",
        field="prf_documents",
        kind=int,
        layer="identity",
        doc="How many of the lexical leaders the feedback terms are mined from.",
        minimum=1,
        maximum=100,
        env="RESEARCH_ULTRARAG_RETRIEVAL_PRF_DOCUMENTS",
    ),
    Setting(
        key="retrieval.prf_terms",
        field="prf_terms",
        kind=int,
        layer="identity",
        doc="Most feedback terms added to one query.",
        minimum=1,
        maximum=100,
        env="RESEARCH_ULTRARAG_RETRIEVAL_PRF_TERMS",
    ),
    Setting(
        key="retrieval.maximum_withheld_examples",
        field="maximum_withheld_examples",
        kind=int,
        layer="identity",
        doc="Withheld candidates quoted per gate reason in a search answer.",
        minimum=0,
        maximum=100,
        env="RESEARCH_ULTRARAG_RETRIEVAL_MAXIMUM_WITHHELD_EXAMPLES",
    ),
    Setting(
        key="retrieval.source_diversity_penalty",
        field="source_diversity_penalty",
        kind=float,
        layer="runtime",
        doc=(
            "Share of a candidate's normalized relevance charged for each "
            "candidate already selected from the same source, so one prolific "
            "source cannot fill the answer. Applied to the final top_k pick "
            "over candidates that were already ranked: it reorders what the "
            "fusion and the reranker returned, and can neither add nor remove "
            "a candidate. Zero returns the ranked order unchanged, and so does "
            "an unreranked BM25 or dense ranking, which has no score to charge "
            "a repeat against."
        ),
        minimum=0.0,
        maximum=1.0,
        env="RESEARCH_ULTRARAG_RETRIEVAL_SOURCE_DIVERSITY_PENALTY",
    ),
    Setting(
        key="retrieval.dense_relative_similarity_margin",
        field="dense_relative_similarity_margin",
        kind=float,
        layer="runtime",
        doc=(
            "How far below the query's own best dense similarity a candidate may "
            "score and still be admitted when it misses the cosine floor, so a "
            "short or abstract query whose whole candidate list sits in a narrow "
            "band is not left with a handful of passages. The rescue applies only "
            "when at least one candidate cleared the floor, so a query the corpus "
            "cannot support still abstains. 0 applies the floor to every "
            "candidate. Runtime: it re-ranks a query and changes no artifact."
        ),
        minimum=0.0,
        maximum=0.5,
        env="RESEARCH_ULTRARAG_RETRIEVAL_DENSE_RELATIVE_SIMILARITY_MARGIN",
    ),
    Setting(
        key="retrieval.minimum_passage_words",
        field="minimum_passage_words",
        kind=int,
        layer="runtime",
        doc=(
            "Words a candidate's cleaned text must have before it can be "
            "evidence. Chunks never span extraction units, so a short unit — an "
            "index line, a heading, a copyright line, a caption — becomes a short "
            "chunk that matches a query about its own words while carrying no "
            "prose to cite. 0 admits every candidate. Runtime: it filters a query "
            "and changes no artifact."
        ),
        minimum=0,
        maximum=400,
        env="RESEARCH_ULTRARAG_RETRIEVAL_MINIMUM_PASSAGE_WORDS",
    ),
    Setting(
        key="retrieval.minimum_passage_token_fraction",
        field="minimum_passage_token_fraction",
        kind=float,
        layer="runtime",
        doc=(
            "Smallest candidate a query may return, as a fraction of the "
            "generation's recorded `chunking.size`. A chunk is built to hold "
            "`chunking.size` tokens, so a candidate holding a small fraction of "
            "that is a fragment — an index line, a heading, a caption — whatever "
            "its word count happens to be. Counted in the tokens the generation "
            "was chunked with, over the returned text, so a contextual header "
            "cannot make a fragment look long. 0 admits every candidate. Runtime: "
            "it filters a query and changes no artifact."
        ),
        minimum=0.0,
        maximum=1.0,
        env="RESEARCH_ULTRARAG_RETRIEVAL_MINIMUM_PASSAGE_TOKEN_FRACTION",
    ),
    # --- Chunking: what a chunk is. Recorded per generation. ---
    Setting(
        key="chunking.size",
        field="chunk_size",
        kind=int,
        layer="identity",
        doc="Target chunk length in GPT-2 tokens.",
        minimum=50,
        maximum=384,
        env="RESEARCH_ULTRARAG_CHUNKING_SIZE",
    ),
    Setting(
        key="chunking.overlap",
        field="chunk_overlap",
        kind=int,
        layer="identity",
        doc="Tokens consecutive chunks share; must be below the chunk size.",
        minimum=0,
        maximum=383,
        env="RESEARCH_ULTRARAG_CHUNKING_OVERLAP",
    ),
    Setting(
        key="chunking.headers",
        field="chunk_headers",
        kind=bool,
        layer="identity",
        doc=(
            "Prepend the source title and the section to the text a chunk is "
            "embedded from, never to the text a search returns, so returned "
            "text stays quote-clean. A re-ingest with it on recomputes every "
            "vector, because vector reuse is keyed on the passage. Measured "
            "neutral on the reference corpus, so the default is off and a "
            "project opts in."
        ),
        env="RESEARCH_ULTRARAG_CHUNKING_HEADERS",
    ),
    Setting(
        key="chunking.batch_units",
        field="chunk_batch_units",
        kind=int,
        layer="runtime",
        doc=(
            "Extraction units sent to one chunker call. Throughput only: each "
            "unit keeps its own durable output and redo boundary."
        ),
        minimum=1,
        maximum=256,
        env="RESEARCH_ULTRARAG_CHUNKING_BATCH_UNITS",
    ),
    # --- Ingestion: how much work one call does, and in what batches. ---
    Setting(
        key="ingestion.work_budget_seconds",
        field="work_budget_seconds",
        kind=int,
        layer="runtime",
        doc=(
            "Soft time budget for one ingest call; exhausting it returns a "
            "checkpointed in_progress result instead of losing work. A build larger "
            "than one budget needs one call per slice, which an agent whose client "
            "stops repeating identical calls cannot finish: set this below that "
            "client's own request timeout so a single call can carry the build."
        ),
        minimum=10,
        maximum=MAXIMUM_WORK_BUDGET_SECONDS,
        env="RESEARCH_ULTRARAG_INGESTION_WORK_BUDGET_SECONDS",
    ),
    Setting(
        key="ingestion.embedding_batch_size",
        field="embedding_batch_size",
        kind=int,
        layer="runtime",
        doc="Chunks embedded per gateway call during ingestion.",
        minimum=1,
        maximum=1024,
        env="RESEARCH_ULTRARAG_INGESTION_EMBEDDING_BATCH_SIZE",
    ),
    Setting(
        key="ingestion.pdf_page_batch_size",
        field="pdf_page_batch_size",
        kind=int,
        layer="runtime",
        doc="PDF pages extracted per gateway call.",
        minimum=1,
        maximum=64,
        env="RESEARCH_ULTRARAG_INGESTION_PDF_PAGE_BATCH_SIZE",
    ),
    # --- Dense engine and models. ---
    Setting(
        key="dense.backend",
        field="dense_backend",
        kind=str,
        layer="engine",
        normalize_case=True,
        doc=(
            "Dense index backend for new generations: 'auto' scans the portable "
            "vectors below the documented corpus threshold."
        ),
        choices=DENSE_BACKENDS,
        env="RESEARCH_ULTRARAG_DENSE_BACKEND",
    ),
    Setting(
        key="dense.embedding_model",
        field="embedding_model",
        kind=str,
        layer="engine",
        doc=(
            "Embedding model for the dense half of retrieval. Every supported "
            "name is pinned to a revision in embeddings.py, and each declares "
            "the languages it covers."
        ),
        choices=EMBEDDING_MODEL_CHOICES,
        env="RESEARCH_ULTRARAG_EMBEDDING_MODEL",
    ),
    Setting(
        key="dense.reranker_model",
        field="reranker_model",
        kind=str,
        layer="engine",
        doc=(
            "CPU cross-encoder that reranks every search. Every supported name "
            "is pinned to a revision in rerankers.py."
        ),
        choices=tuple(RERANKER_MODELS),
        env="RESEARCH_ULTRARAG_RERANKER_MODEL",
    ),
    Setting(
        key="dense.embedding_inference_batch_size",
        field="embedding_inference_batch_size",
        kind=int,
        layer="runtime",
        doc=(
            "Sequences per embedding inference. Throughput only: a batch of 1 "
            "returns exactly the same floats as a batch of 64 (MEASUREMENTS.md)."
        ),
        minimum=1,
        maximum=1024,
        env="RESEARCH_ULTRARAG_EMBEDDING_INFERENCE_BATCH_SIZE",
    ),
    Setting(
        key="dense.exact_backend_chunk_limit",
        field="exact_backend_chunk_limit",
        kind=int,
        layer="engine",
        doc=(
            "Corpus size above which 'auto' selects the embedded ANN index "
            "instead of the exact scan; recorded in each generation."
        ),
        minimum=1,
        maximum=100_000_000,
        env="RESEARCH_ULTRARAG_EXACT_BACKEND_CHUNK_LIMIT",
    ),
)


SETTINGS_BY_KEY: dict[str, Setting] = {}
for _setting in SETTINGS:
    if _setting.key in SETTINGS_BY_KEY:
        raise SettingsError(f"Duplicate setting key: {_setting.key}")
    SETTINGS_BY_KEY[_setting.key] = _setting
SETTINGS_BY_FIELD = {setting.field: setting for setting in SETTINGS}
SETTINGS_SECTIONS = tuple(
    dict.fromkeys(setting.key.split(".")[0] for setting in SETTINGS)
)


def default_config_path() -> Path:
    """Return the packaged default config file."""

    return Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)


def user_config_path() -> Path:
    """Return the per-user overlay path for this platform."""

    return platformdirs.user_config_path(USER_CONFIG_DIRECTORY) / "config.toml"


def project_config_path(project_root: str | Path) -> Path:
    """Return the per-project overlay path, inside the project's own state."""

    return Path(project_root) / PROJECT_CONFIG_RELATIVE


def read_config_document(path: Path, *, source: str) -> dict[str, Any]:
    """Read one TOML layer, refusing anything that is not a plain document."""

    if path.is_symlink():
        raise SettingsError(f"{source} must not be a symlink: {path}")
    if not path.is_file():
        raise SettingsError(f"{source} is not a readable file: {path}")
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise SettingsError(f"{source} is not valid TOML ({path}): {exc}") from exc
    except OSError as exc:
        raise SettingsError(f"{source} cannot be read ({path}): {exc}") from exc
    if not isinstance(document, dict):
        raise SettingsError(f"{source} must contain a TOML table: {path}")
    return document


def merge_settings(
    base: dict[str, Any],
    overlay: Mapping[str, Any],
    *,
    source: str,
    prefix: str = "",
) -> set[str]:
    """Merge one layer into ``base`` per key and return the keys it set.

    Tables merge recursively, so a layer names only what it changes. A scalar, an
    array, or a table the overlay supplies in full replaces whatever the layer
    below had. A key the registry does not declare is an error, which is what
    makes a typo loud instead of silent.
    """

    written: set[str] = set()
    for key, value in overlay.items():
        if not isinstance(key, str):
            raise SettingsError(f"{source} has a non-string key: {key!r}")
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            if dotted in SETTINGS_BY_KEY:
                raise SettingsError(
                    f"{source} gives a table for the single setting {dotted}"
                )
            written |= merge_settings(
                base,
                value,
                source=source,
                prefix=f"{dotted}.",
            )
            continue
        if dotted not in SETTINGS_BY_KEY:
            raise SettingsError(f"{source} sets an unknown setting: {dotted}")
        base[dotted] = value
        written.add(dotted)
    return written


def environment_settings(
    environ: Mapping[str, str],
) -> dict[str, tuple[str, str]]:
    """Return the environment layer as ``key -> (raw value, variable name)``."""

    values: dict[str, tuple[str, str]] = {}
    for setting in SETTINGS:
        if not setting.env:
            continue
        raw = environ.get(setting.env)
        if raw is None or not raw.strip():
            continue
        values[setting.key] = (raw, setting.env)
    return values


def override_settings(overrides: Sequence[str]) -> dict[str, str]:
    """Return the command-line layer from repeated ``--set key=value`` pairs."""

    values: dict[str, str] = {}
    for item in overrides:
        key, separator, raw = item.partition("=")
        key = key.strip()
        if not separator or not key:
            raise SettingsError(f"--set expects key=value, got: {item!r}")
        if key not in SETTINGS_BY_KEY:
            raise SettingsError(
                f"--set names an unknown setting: {key}; run --print-config "
                "to see every key"
            )
        values[key] = raw.strip()
    return values


@dataclass(frozen=True, slots=True)
class EffectiveSettings:
    """Every tunable after the layers have been merged, typed and checked."""

    offline: bool
    log_level: str
    tool_detail: str
    embedding_threads: int | None
    nice: int
    model_cache_root: Path | None
    rrf_k: int
    bm25_weight: float
    dense_weight: float
    minimum_candidates: int
    maximum_candidates: int
    dense_minimum_cosine_similarity: float
    rerank_max_candidates: int
    rerank_window_multiple: int
    rerank_window_floor: int
    prf: bool
    prf_documents: int
    prf_terms: int
    maximum_withheld_examples: int
    source_diversity_penalty: float
    dense_relative_similarity_margin: float
    minimum_passage_words: int
    minimum_passage_token_fraction: float
    chunk_size: int
    chunk_overlap: int
    chunk_headers: bool
    chunk_batch_units: int
    work_budget_seconds: int
    embedding_batch_size: int
    pdf_page_batch_size: int
    dense_backend: str
    embedding_model: str
    reranker_model: str
    language_corpus: str
    bm25_stopwords: str
    embedding_inference_batch_size: int
    exact_backend_chunk_limit: int

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> EffectiveSettings:
        """Build the settings from a field-keyed mapping, checking every rule."""

        known = {item.name for item in fields(cls)}
        missing = sorted(known - set(values))
        if missing:
            raise SettingsError(
                "The default config is incomplete; missing: " + ", ".join(missing)
            )
        unknown = sorted(set(values) - known)
        if unknown:
            raise SettingsError("Unknown settings: " + ", ".join(unknown))

        languages = normalize_corpus_languages(values["language_corpus"])
        language = ",".join(languages)
        stopwords = str(values["bm25_stopwords"]).strip().casefold()
        if stopwords and stopwords not in BM25_STOPWORD_LANGUAGES:
            raise SettingsError(
                "language.bm25_stopwords has no BM25 stopword list: "
                f"{stopwords!r}. Supported: "
                + ", ".join(sorted(BM25_STOPWORD_LANGUAGES))
            )

        roundtrip_error = _bm25_stopword_roundtrip_error(stopwords or languages[0])
        if roundtrip_error:
            raise SettingsError(roundtrip_error)

        threads = values["embedding_threads"]
        cache_root = values["model_cache_root"]
        settings = cls(
            **{
                **values,
                "language_corpus": language,
                "bm25_stopwords": stopwords,
                "embedding_threads": None if not threads else int(threads),
                "model_cache_root": (
                    None if not cache_root else Path(str(cache_root)).expanduser()
                ),
            }
        )
        if settings.chunk_overlap >= settings.chunk_size:
            raise SettingsError(
                "chunking.overlap must be below chunking.size: "
                f"{settings.chunk_overlap} >= {settings.chunk_size}"
            )
        if settings.minimum_candidates > settings.maximum_candidates:
            raise SettingsError(
                "retrieval.minimum_candidates must not exceed "
                "retrieval.maximum_candidates: "
                f"{settings.minimum_candidates} > {settings.maximum_candidates}"
            )
        return settings

    # The embedding model carries facts that must not be configured twice: its
    # vector dimension, its token limit, its revision, and the languages it covers.

    @property
    def embedding_facts(self) -> EmbeddingModel:
        """Return the pinned facts of the configured embedding model."""

        return resolve_embedding_model(self.embedding_model)

    @property
    def embedding_dimension(self) -> int:
        return self.embedding_facts.dimension

    @property
    def embedding_model_revision(self) -> str:
        return self.embedding_facts.revision

    @property
    def embedding_maximum_tokens(self) -> int:
        return self.embedding_facts.maximum_tokens

    @property
    def corpus_languages(self) -> tuple[str, ...]:
        """The languages this corpus is written in, as named."""

        return tuple(self.language_corpus.split(","))

    @property
    def bm25_stopwords_language(self) -> str:
        """The one language whose stopword list BM25 filters with.

        BM25 takes a single list, so a corpus in several languages filters the
        function words of the first language it names unless another is chosen.
        """

        return self.bm25_stopwords or self.corpus_languages[0]

    @property
    def embedding_language_warning(self) -> str | None:
        """Explain corpus languages the embedding model cannot serve."""

        facts = self.embedding_facts
        missing = [code for code in self.corpus_languages if not facts.covers(code)]
        if not missing:
            return None
        covered = ", ".join(facts.languages) if facts.languages else "any language"
        message = (
            f"The embedding model {facts.name} covers {covered}, not "
            + " or ".join(f"'{code}'" for code in missing)
            + ": the dense half of retrieval will be weak for this corpus."
        )
        alternatives = models_covering(self.corpus_languages)
        if alternatives:
            return (
                message
                + " Set dense.embedding_model to one that covers it: "
                + ", ".join(
                    f"{model.name} ({model.size_gb:g} GB)" for model in alternatives
                )
                + "."
            )
        return (
            message
            + " No model in the pinned table covers "
            + " and ".join(f"'{code}'" for code in missing)
            + ", so this corpus needs a model added to embeddings.py."
        )

    def value(self, key: str) -> Any:
        """Return one setting by its dotted key."""

        setting = SETTINGS_BY_KEY.get(key)
        if setting is None:
            raise SettingsError(f"Unknown setting: {key}")
        return getattr(self, setting.field)


def resolve_settings(
    project_root: str | Path,
    *,
    config_path: str | Path | None = None,
    overrides: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    override_source: str = LAYER_COMMAND_LINE,
) -> tuple[EffectiveSettings, dict[str, str]]:
    """Merge every layer, lowest precedence first, and report where each value came from.

    The layers are the packaged default, the per-user config, the project's own
    config, an explicitly named file, the environment, and finally the command
    line. Each later layer names only the keys it changes. Nothing is read
    relative to the working directory: a client may start this server anywhere.
    """

    environment = os.environ if environ is None else environ
    merged: dict[str, Any] = {}
    provenance: dict[str, str] = {}

    layers: list[tuple[str, Path]] = [(LAYER_DEFAULT, default_config_path())]
    user_config = user_config_path()
    if user_config.exists():
        layers.append((LAYER_USER, user_config))
    project_config = project_config_path(project_root)
    if project_config.exists():
        layers.append((LAYER_PROJECT, project_config))
    if config_path is not None:
        layers.append((LAYER_FILE, Path(config_path).expanduser()))

    for name, layer_path in layers:
        document = read_config_document(layer_path, source=name)
        written = merge_settings(merged, document, source=f"{name} ({layer_path})")
        for key in written:
            provenance[key] = f"{name} ({layer_path})"

    for key, (raw, variable) in environment_settings(environment).items():
        merged[key] = raw
        provenance[key] = f"{LAYER_ENVIRONMENT} ({variable})"

    for key, raw in override_settings(overrides).items():
        merged[key] = raw
        provenance[key] = override_source

    coerced: dict[str, Any] = {}
    for key, setting in SETTINGS_BY_KEY.items():
        if key in merged:
            coerced[setting.field] = setting.coerce(
                merged[key],
                source=provenance.get(key, LAYER_DEFAULT),
            )
        else:
            raise SettingsError(f"No value for setting: {key}")
    for key in SETTINGS_BY_KEY:
        provenance.setdefault(key, LAYER_DEFAULT)

    return EffectiveSettings.from_values(coerced), provenance


def describe_settings(
    settings: EffectiveSettings,
    provenance: Mapping[str, str],
) -> str:
    """Render the effective settings, one line per key, with its source."""

    lines: list[str] = [
        "Effective settings, later layers overriding earlier ones:",
        "  default.toml < user config < project config < --config < environment < --set",
    ]
    width = max(len(setting.key) for setting in SETTINGS)
    for section in SETTINGS_SECTIONS:
        lines.append("")
        lines.append(f"[{section}]")
        for setting in SETTINGS:
            if not setting.key.startswith(f"{section}."):
                continue
            value = getattr(settings, setting.field)
            rendered = '""' if value is None else repr(value)
            if isinstance(value, str):
                rendered = f'"{value}"'
            lines.append(
                f"  {setting.key:<{width}} = {rendered:<28} "
                f"# {provenance.get(setting.key, LAYER_DEFAULT)}"
            )
    return "\n".join(lines)

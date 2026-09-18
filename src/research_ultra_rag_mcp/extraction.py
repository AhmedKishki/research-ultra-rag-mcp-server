"""Layout-aware extraction and bibliographic identity for PDF/EPUB sources."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any

import ebooklib
import pymupdf
from bs4 import BeautifulSoup, Tag
from ebooklib import epub

from .sources import SourceFile, sha256_file

pymupdf.no_recommend_layout()


class ExtractionError(RuntimeError):
    """Raised when a source cannot be represented safely in the knowledge base."""


_HORIZONTAL_SPACE = re.compile(r"[\t\f\v \u00a0]+")
_LIST_ITEM = re.compile(r"^(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_URL = re.compile(r"^(?:https?://|www\.|(?:dx\.)?doi\.org/)", re.IGNORECASE)
_PAGE_NUMBER = re.compile(
    r"^(?:page\s+)?(?:[ivxlcdm]+|\d+)(?:\s+(?:of|/|-|–)\s*\d+)?$",
    re.IGNORECASE,
)
_TITLE_REJECTIONS = (
    re.compile(r"^untitled$", re.IGNORECASE),
    re.compile(
        r"^(?:executive\s+summary|summary|abstract|foreword|preface|introduction|"
        r"contents|table\s+of\s+contents)$",
        re.IGNORECASE,
    ),
    re.compile(r"^microsoft\s+word\s*[-–:]", re.IGNORECASE),
    re.compile(r"^adobe\s+(?:acrobat|indesign)", re.IGNORECASE),
    re.compile(r"^(?:document|export|file)(?:[-_ ]?\d+)?$", re.IGNORECASE),
)
_FRONT_MATTER_EXCLUSIONS = re.compile(
    r"^(?:abstract|keywords?|key\s+words?|contents|table\s+of\s+contents|"
    r"volume\s+\d+|number\s+\d+|issn\b|doi\b|copyright\b|©|arxiv:|"
    r"check\s+for\s+updates$)",
    re.IGNORECASE,
)
_AUTHOR_EXCLUSIONS = re.compile(
    r"^(?:abstract|keywords?|volume|number|doi|issn|copyright|published|received|"
    r"accepted|introduction)\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b")
_STANDALONE_YEAR = re.compile(r"^\(?\s*(18\d{2}|19\d{2}|20\d{2}|21\d{2})\s*\)?$")
_PDF_DATE_YEAR = re.compile(r"^(?:D:)?(18\d{2}|19\d{2}|20\d{2}|21\d{2})")
_DATED_LINE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b.*\b(?:18\d{2}|19\d{2}|20\d{2}|21\d{2})\b",
    re.IGNORECASE,
)
_LEGEND_ASTERISK = re.compile(r"(?<!\w)\*|\*(?!\w)")
_EXPLICIT_MARKER_LEGEND = re.compile(
    r"^\s*\*\s*(?:(?:indicates?|denotes?|means?)\b|[:=])",
    re.IGNORECASE,
)
_BYLINE = re.compile(r"^(?:by|written\s+by|author(?:s)?\s*:)\s+", re.IGNORECASE)
_AUTHOR_JUNK = re.compile(
    r"^(?:unknown|anonymous|untitled|microsoft\s+word|adobe\s+(?:acrobat|indesign)|"
    r"administrator|admin|user)$",
    re.IGNORECASE,
)
_TITLE_LABEL = re.compile(
    r"^(?:review\s+of\s+the\s+month|research\s+article|original\s+article|article)$",
    re.IGNORECASE,
)
_AUTHOR_AFFILIATION = re.compile(
    r"\s*\([^)]*(?:university|institute|college)[^)]*\)\s*$", re.IGNORECASE
)
_AUTHOR_MARKER = re.compile(r"(?:\d+(?:\s*,\s*\d+)*|[*∗†‡§])+$")
_FIGURE_CAPTION = re.compile(r"^(?:fig(?:ure)?\.?)\s*(?:\d|[ivxlcdm])", re.IGNORECASE)
_TABLE_CAPTION = re.compile(r"^table\s*(?:\d|[ivxlcdm])", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _TextBlock:
    number: int
    bbox: tuple[float, float, float, float]
    lines: tuple[str, ...]
    text: str
    font_size: float

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])


@dataclass(frozen=True, slots=True)
class _Region:
    kind: str
    bbox: tuple[float, float, float, float]


def _starts_with_alpha(value: str) -> bool:
    for character in value:
        if character.isalpha():
            return True
        if character.isdigit():
            return False
    return False


def _ends_sentence(value: str) -> bool:
    return value.rstrip("\"'”’)]}").endswith((".", "!", "?", "…", ":"))


def normalize_reading_text(value: str) -> str:
    """Remove extraction layout wrapping without rewriting source prose."""

    text = unicodedata.normalize("NFC", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = _CONTROL_CHARACTERS.sub("", text)
    paragraphs: list[str] = []
    current = ""
    separated = False
    for raw_line in text.splitlines():
        line = _HORIZONTAL_SPACE.sub(" ", raw_line).strip()
        # PDF text spans sometimes leave a layout-only space after a hyphen
        # even though the printed form is a normal hyphenated word.
        line = re.sub(r"(?<=[^\W\d_])-\s+(?=[^\W\d_])", "-", line)
        if not line:
            separated = bool(current)
            continue
        if not current:
            current = line
        elif _LIST_ITEM.match(line) or (separated and _ends_sentence(current)):
            paragraphs.append(current)
            current = line
        elif (
            current.endswith("-")
            and current[-2:-1].isalpha()
            and _starts_with_alpha(line)
        ):
            current = current[:-1] + line
        else:
            current = f"{current} {line}"
        separated = False
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs)


def normalize_inline_text(value: str) -> str:
    """Normalize extracted metadata to one display-safe line."""

    return _HORIZONTAL_SPACE.sub(
        " ", normalize_reading_text(value).replace("\n", " ")
    ).strip()


def _normalize_text_list(values: list[Any]) -> list[str]:
    return [
        normalized
        for value in values
        if (normalized := normalize_inline_text(str(value)))
    ]


def _document_id(source: SourceFile, digest: str) -> str:
    identity = f"{source.source_relative_path}\0{digest}".encode()
    return f"doc_{hashlib.sha256(identity).hexdigest()[:24]}"


def _doi(value: str) -> str:
    match = _DOI.search(value)
    if match is None:
        return ""
    return match.group(0).rstrip(".,;:)]}")


def _valid_title(
    value: str,
    filename_stem: str,
    *,
    reject_filename_match: bool = True,
) -> bool:
    normalized = normalize_inline_text(value)
    if not normalized or len(normalized) > 500:
        return False
    doi_candidate = re.sub(r"^doi\s*:\s*", "", normalized, flags=re.IGNORECASE)
    if _DOI.fullmatch(doi_candidate) or _URL.match(normalized):
        return False
    if any(pattern.search(normalized) for pattern in _TITLE_REJECTIONS):
        return False
    comparable_title = re.sub(r"\.(?:pdf|epub)$", "", normalized, flags=re.IGNORECASE)
    comparable = re.sub(r"[^a-z0-9]+", "", comparable_title.casefold())
    filename = re.sub(r"[^a-z0-9]+", "", filename_stem.casefold())
    if reject_filename_match and comparable and comparable == filename:
        return False
    return sum(character.isalpha() for character in normalized) >= 3


def _valid_author(value: str) -> bool:
    normalized = normalize_inline_text(value)
    doi_candidate = re.sub(r"^doi\s*:\s*", "", normalized, flags=re.IGNORECASE)
    return bool(
        normalized
        and len(normalized) <= 240
        and not _URL.match(normalized)
        and not _DOI.fullmatch(doi_candidate)
        and not _AUTHOR_JUNK.fullmatch(normalized)
        and sum(character.isalpha() for character in normalized) >= 2
    )


def _clean_author_name(value: str) -> str:
    normalized = _BYLINE.sub("", normalize_inline_text(value)).strip()
    normalized = re.sub(r"^(?:and|&)\s+", "", normalized, flags=re.IGNORECASE)
    normalized = _AUTHOR_AFFILIATION.sub("", normalized)
    normalized = _AUTHOR_MARKER.sub("", normalized).strip(" ,;|")
    return normalized


def _author_names(value: str) -> list[str]:
    normalized = normalize_inline_text(value)
    explicitly_by = bool(_BYLINE.match(normalized))
    cleaned = _BYLINE.sub("", normalized).strip()
    if (
        not cleaned
        or _YEAR.fullmatch(cleaned)
        or _URL.match(cleaned)
        or ":" in cleaned
        or re.search(
            r"\b(?:journal|papers?|volume|issue|no\.?|issn)\b",
            cleaned,
            re.IGNORECASE,
        )
    ):
        return []
    if _ends_sentence(cleaned) or len(cleaned.split()) > 32:
        return []

    has_list_separator = bool(re.search(r"\s(?:&|and)\s|;", cleaned, re.IGNORECASE))
    if cleaned.count(",") >= 2 or has_list_separator:
        parts = re.split(r"\s*(?:,|;|\s&\s|\sand\s)\s*", cleaned, flags=re.IGNORECASE)
    else:
        parts = [cleaned]
    names = [name for part in parts if (name := _clean_author_name(part))]
    if not names or any(not _valid_author(name) for name in names):
        return []

    organization_words = {
        "agency",
        "association",
        "commission",
        "committee",
        "council",
        "department",
        "foundation",
        "institute",
        "international",
        "laboratory",
        "organisation",
        "organization",
        "university",
    }
    if explicitly_by or len(names) > 1:
        return names
    words = names[0].split()
    capitalized = sum(word[:1].isupper() for word in words if word)
    is_organization = any(
        word.casefold().strip(".,") in organization_words for word in words
    )
    if is_organization or (2 <= len(words) <= 8 and capitalized >= 2):
        return names
    return []


def _title_and_embedded_byline(
    blocks: list[_TextBlock],
) -> tuple[str, list[str]]:
    lines = [line for block in blocks for line in block.lines]
    authors: list[str] = []
    if len(lines) >= 3 and lines[1].rstrip().endswith(":"):
        possible_author = _author_names(lines[0])
        if possible_author:
            authors = possible_author
            lines = lines[1:]

    cleaned_lines: list[str] = []
    for line in lines:
        normalized = normalize_inline_text(line)
        if (
            not normalized
            or _TITLE_LABEL.fullmatch(normalized)
            or _FRONT_MATTER_EXCLUSIONS.match(normalized)
            or _URL.match(normalized)
        ):
            continue
        cleaned_lines.append(normalized)
    # Preserve the original line boundaries long enough for the common
    # ``inter-\nword`` PDF wrapping case to be joined as ``interword``.
    title = normalize_inline_text(normalize_reading_text("\n".join(cleaned_lines)))
    title = re.sub(
        r"\s+arxiv:\d{4}\.\d+(?:v\d+)?(?:\s+\[[^]]+\])?(?:\s+\d{1,2}\s+\w+\s+\d{4})?$",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()
    return title, authors


def _metadata_confidence(provenance: str) -> float:
    return {
        "reviewed_override": 1.0,
        "epub_opf": 0.98,
        "pdf_front_matter": 0.92,
        "epub_visible": 0.86,
        "pdf_metadata": 0.72,
        "filename": 0.30,
        "missing": 0.0,
    }.get(provenance, 0.0)


def _metadata_value(
    field: str,
    override: dict[str, Any],
    automatic: Any,
    automatic_source: str,
) -> tuple[Any, str]:
    if field in override:
        value = override[field]
        if value not in (None, "", []):
            return value, "reviewed_override"
    return automatic, automatic_source


def _base_metadata(
    *,
    source: SourceFile,
    document_id: str,
    digest: str,
    automatic: dict[str, Any],
    provenance: dict[str, str],
    warnings: list[str],
    override: dict[str, Any],
) -> dict[str, Any]:
    title, title_source = _metadata_value(
        "title",
        override,
        automatic.get("title") or source.path.stem,
        provenance.get("title", "filename"),
    )
    authors, author_source = _metadata_value(
        "authors",
        override,
        automatic.get("authors", []),
        provenance.get("authors", "missing"),
    )
    year, year_source = _metadata_value(
        "year",
        override,
        automatic.get("year"),
        provenance.get("year", "missing"),
    )
    doi, doi_source = _metadata_value(
        "doi",
        override,
        automatic.get("doi", ""),
        provenance.get("doi", "missing"),
    )
    categories = _normalize_text_list(override.get("categories", []))
    keywords = _normalize_text_list(override.get("keywords", []))
    resolved_authors = _normalize_text_list(list(authors or []))
    resolved_title = normalize_inline_text(str(title))
    if title_source == "reviewed_override" and not _valid_title(
        resolved_title,
        source.path.stem,
        reject_filename_match=False,
    ):
        invalid_title_source = title_source
        warnings.append("invalid_title_candidate")
        candidate_doi = _doi(resolved_title)
        automatic_title = normalize_inline_text(str(automatic.get("title") or ""))
        if _valid_title(automatic_title, source.path.stem):
            resolved_title = automatic_title
            title_source = provenance.get("title", "filename")
        else:
            resolved_title = source.path.stem
            title_source = "filename"
        if candidate_doi and not doi:
            doi = candidate_doi
            doi_source = invalid_title_source
    resolved_title = resolved_title or source.path.stem
    resolved_doi = _doi(str(doi)) or normalize_inline_text(str(doi or ""))
    metadata_warnings = list(dict.fromkeys(warnings))
    if title_source == "filename":
        metadata_warnings.append("title_from_filename")
    if not resolved_authors:
        metadata_warnings.append("authors_missing")
    metadata_provenance = {
        "title": title_source,
        "authors": author_source,
        "year": year_source,
        "doi": doi_source,
        "categories": "reviewed_override" if "categories" in override else "missing",
        "keywords": "reviewed_override" if "keywords" in override else "missing",
    }
    return {
        "document_id": document_id,
        "source_path": source.project_relative_path,
        "source_relative_path": source.source_relative_path,
        "format": source.extension.removeprefix("."),
        "sha256": digest,
        "size": source.size,
        "mtime_ns": source.mtime_ns,
        "title": resolved_title,
        "authors": resolved_authors,
        "year": year,
        "doi": resolved_doi,
        "categories": categories,
        "keywords": keywords,
        "metadata_provenance": metadata_provenance,
        "metadata_confidence": {
            field: _metadata_confidence(source_name)
            for field, source_name in metadata_provenance.items()
        },
        "metadata_warnings": list(dict.fromkeys(metadata_warnings)),
    }


def _pdf_author_list(metadata: dict[str, Any]) -> list[str]:
    author = normalize_inline_text(str(metadata.get("author") or ""))
    if not author:
        return []
    parts = [item.strip() for item in re.split(r"\s*[;|]\s*", author)]
    return [item for item in parts if _valid_author(item)]


def _pdf_line_text(spans: list[dict[str, Any]]) -> str:
    """Repair split dash glyphs only when span geometry proves an overlay."""

    result: list[str] = []
    index = 0
    while index < len(spans):
        span = spans[index]
        text = str(span.get("text") or "")
        if text in {"*", "\x01"} and index + 1 < len(spans):
            following = spans[index + 1]
            following_text = str(following.get("text") or "")
            left_bbox = tuple(float(value) for value in span.get("bbox", ()))
            right_bbox = tuple(float(value) for value in following.get("bbox", ()))
            split_overlay = bool(
                following_text == "/"
                and span.get("font") != following.get("font")
                and len(left_bbox) == 4
                and len(right_bbox) == 4
                and left_bbox[0] <= right_bbox[0] <= left_bbox[2] + 1.0
                and max(left_bbox[1], right_bbox[1]) < min(left_bbox[3], right_bbox[3])
            )
            if split_overlay:
                result.append("—" if text == "*" else "–")
                index += 2
                continue
        result.append(text)
        index += 1
    return "".join(result)


def _page_blocks(page: pymupdf.Page) -> list[_TextBlock]:
    result: list[_TextBlock] = []
    payload = page.get_text("dict", sort=False, flags=pymupdf.TEXTFLAGS_TEXT)
    for number, block in enumerate(payload.get("blocks", [])):
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        sizes: list[float] = []
        for line in block.get("lines", []):
            value = _pdf_line_text(line.get("spans", []))
            value = _HORIZONTAL_SPACE.sub(" ", value).strip()
            if value:
                lines.append(value)
            sizes.extend(
                float(span.get("size") or 0.0)
                for span in line.get("spans", [])
                if span.get("text")
            )
        if not lines:
            continue
        bbox = tuple(float(item) for item in block.get("bbox", (0, 0, 0, 0)))
        result.append(
            _TextBlock(
                number=number,
                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                lines=tuple(lines),
                text="\n".join(lines),
                font_size=max(sizes, default=0.0),
            )
        )
    return result


def _front_matter_identity(
    pages: list[tuple[list[_TextBlock], float, float]],
    metadata: dict[str, Any],
    filename_stem: str,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    embedded_title = normalize_inline_text(str(metadata.get("title") or ""))
    embedded_authors = _pdf_author_list(metadata)
    visible_title = ""
    visible_authors: list[str] = []
    visible_year: int | None = None
    text_bearing_pages = [page for page in pages if page[0]][:5]
    first_page_text = (
        "\n".join(block.text for block in text_bearing_pages[0][0])
        if text_bearing_pages
        else ""
    )
    # A DOI in an arbitrary prose sentence is often a cited work, not this
    # document. Visible DOI candidates therefore need an explicit DOI cue.
    visible_doi_text = "\n".join(
        line
        for line in first_page_text.splitlines()
        if re.search(r"(?:\bdoi\b|https?://doi\.org/)", line, re.IGNORECASE)
    )
    publication_text = "\n".join(
        block.text
        for blocks, _width, _height in text_bearing_pages[:2]
        for block in blocks
    )

    for blocks, _width, height in text_bearing_pages:
        if not blocks:
            continue
        vertical_limit = height * (0.95 if len(blocks) <= 5 else 0.62)
        candidates = [
            block
            for block in blocks
            if block.bbox[1] <= vertical_limit
            and not _FRONT_MATTER_EXCLUSIONS.match(normalize_inline_text(block.text))
            and _valid_title(
                _title_and_embedded_byline([block])[0],
                filename_stem,
                reject_filename_match=False,
            )
        ]
        body_sizes = [
            block.font_size
            for block in blocks
            if len(normalize_inline_text(block.text)) >= 200 and block.font_size > 0
        ]
        all_sizes = [block.font_size for block in blocks if block.font_size > 0]
        body_size = median(body_sizes or [min(all_sizes, default=10.0)])
        if not candidates:
            continue
        maximum = max(block.font_size for block in candidates)
        minimum_title_size = 12.0 if len(blocks) <= 5 else max(12.0, body_size * 1.22)
        if maximum < minimum_title_size:
            continue
        anchor = max(
            (block for block in candidates if block.font_size >= maximum * 0.98),
            key=lambda block: (
                len(_title_and_embedded_byline([block])[0]),
                -block.bbox[1],
            ),
        )
        top = anchor.bbox[1]
        title_blocks = sorted(
            [
                block
                for block in candidates
                if block.font_size >= maximum * 0.79
                and top - 20 <= block.bbox[1] <= top + 110
            ],
            key=lambda item: (item.bbox[1], item.bbox[0]),
        )
        if anchor not in title_blocks:
            title_blocks.append(anchor)
            title_blocks.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
        visible_title, embedded_byline = _title_and_embedded_byline(title_blocks)
        title_bottom = max(block.bbox[3] for block in title_blocks)
        visible_authors = embedded_byline
        if not visible_authors:
            possible_authors = sorted(
                [
                    block
                    for block in blocks
                    if title_bottom < block.bbox[1] <= title_bottom + 360
                    and len(normalize_inline_text(block.text)) <= 240
                    and not _AUTHOR_EXCLUSIONS.match(normalize_inline_text(block.text))
                ],
                key=lambda item: (item.bbox[1], item.bbox[0]),
            )
            explicit_bylines = [
                block
                for block in possible_authors
                if _BYLINE.match(normalize_inline_text(block.text))
            ]
            for author_block in explicit_bylines or possible_authors:
                if author_names := _author_names(author_block.text):
                    visible_authors = author_names
                    break
        for block in blocks:
            normalized = normalize_inline_text(block.text)
            if title_bottom < block.bbox[1] <= title_bottom + 360 and (
                match := _STANDALONE_YEAR.fullmatch(normalized)
            ):
                visible_year = int(match.group(1))
                break
        break

    warnings: list[str] = []
    if visible_title:
        title = visible_title
        title_source = "pdf_front_matter"
        if _valid_title(embedded_title, filename_stem) and (
            normalize_inline_text(embedded_title).casefold() != visible_title.casefold()
        ):
            warnings.append("conflicting_candidates")
    elif _valid_title(embedded_title, filename_stem):
        title = embedded_title
        title_source = "pdf_metadata"
    else:
        title = filename_stem
        title_source = "filename"

    if visible_authors:
        authors = visible_authors
        authors_source = "pdf_front_matter"
    elif embedded_authors:
        authors = embedded_authors
        authors_source = "pdf_metadata"
    else:
        authors = []
        authors_source = "missing"

    doi = ""
    doi_source = "missing"
    for candidate, source_name in (
        (embedded_title, "pdf_metadata"),
        (" ".join(str(value) for value in metadata.values()), "pdf_metadata"),
        (visible_doi_text, "pdf_front_matter"),
    ):
        if extracted := _doi(candidate):
            doi = extracted
            doi_source = source_name
            break

    year: int | None = None
    year_source = "missing"
    if visible_year is not None:
        year = visible_year
        year_source = "pdf_front_matter"
    else:
        for line in publication_text.splitlines():
            normalized = normalize_inline_text(line)
            has_publication_cue = bool(
                re.search(
                    r"(?:©|copyright|published|volume|number|issue)",
                    normalized,
                    re.IGNORECASE,
                )
                or (len(normalized) <= 100 and _DATED_LINE.search(normalized))
            )
            if has_publication_cue and (match := _YEAR.search(normalized)):
                year = int(match.group(1))
                year_source = "pdf_front_matter"
                break
    if year is None:
        for field in ("creationDate", "modDate"):
            if match := _PDF_DATE_YEAR.search(
                normalize_inline_text(str(metadata.get(field) or ""))
            ):
                year = int(match.group(1))
                year_source = "pdf_metadata"
                break

    return (
        {"title": title, "authors": authors, "year": year, "doi": doi},
        {
            "title": title_source,
            "authors": authors_source,
            "year": year_source,
            "doi": doi_source,
        },
        warnings,
    )


def _block_signature(block: _TextBlock) -> str:
    value = normalize_inline_text(block.text).casefold()
    return re.sub(r"\d+", "#", value)


def _repeated_margin_signatures(
    pages: list[tuple[list[_TextBlock], float, float]],
) -> set[str]:
    if len(pages) < 3:
        return set()
    page_sets: list[set[str]] = []
    for blocks, _width, height in pages:
        page_sets.append(
            {
                signature
                for block in blocks
                if (block.bbox[3] <= height * 0.12 or block.bbox[1] >= height * 0.88)
                and (signature := _block_signature(block))
            }
        )
    counts = Counter(signature for values in page_sets for signature in values)
    threshold = max(3, math.ceil(len(pages) * 0.30))
    return {signature for signature, count in counts.items() if count >= threshold}


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _rect_intersection_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection = (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )
    return _rect_area(intersection) / max(_rect_area(left), 1.0)


def _merge_rectangles(
    rectangles: list[pymupdf.Rect], padding: float = 8.0
) -> list[pymupdf.Rect]:
    merged: list[pymupdf.Rect] = []
    for original in rectangles:
        candidate = pymupdf.Rect(original)
        changed = True
        while changed:
            changed = False
            remaining: list[pymupdf.Rect] = []
            padded = candidate + (-padding, -padding, padding, padding)
            for existing in merged:
                if padded.intersects(existing):
                    candidate |= existing
                    changed = True
                else:
                    remaining.append(existing)
            merged = remaining
        merged.append(candidate)
    return merged


def _non_prose_regions(page: pymupdf.Page) -> list[_Region]:
    regions: list[_Region] = []
    try:
        finder = page.find_tables()
    except (RuntimeError, ValueError):
        finder = None
    if finder is not None:
        for table in finder.tables:
            bbox = tuple(float(item) for item in table.bbox)
            regions.append(_Region("table", bbox))

    payload = page.get_text("dict", sort=False, flags=pymupdf.TEXTFLAGS_TEXT)
    for block in payload.get("blocks", []):
        if block.get("type") == 1:
            bbox = tuple(float(item) for item in block.get("bbox", (0, 0, 0, 0)))
            if _rect_area(bbox) >= page.rect.get_area() * 0.02:
                regions.append(_Region("figure", bbox))

    try:
        drawing_rectangles = [
            pymupdf.Rect(item["rect"])
            for item in page.get_drawings()
            if item.get("rect") is not None
            and pymupdf.Rect(item["rect"]).get_area() > 4
        ]
    except (RuntimeError, ValueError):
        drawing_rectangles = []
    for rect in _merge_rectangles(drawing_rectangles):
        if rect.get_area() >= page.rect.get_area() * 0.025:
            expanded = rect + (-12, -24, 12, 12)
            expanded &= page.rect
            regions.append(_Region("figure", tuple(float(item) for item in expanded)))

    figures = [region for region in regions if region.kind == "figure"]
    regions = [
        region
        for region in regions
        if region.kind != "table"
        or not any(
            _rect_intersection_ratio(region.bbox, figure.bbox) >= 0.55
            for figure in figures
        )
    ]
    deduplicated: list[_Region] = []
    for region in sorted(
        regions, key=lambda item: (item.kind != "table", -_rect_area(item.bbox))
    ):
        if any(
            _rect_intersection_ratio(region.bbox, item.bbox) > 0.8
            for item in deduplicated
        ):
            continue
        deduplicated.append(region)
    return sorted(deduplicated, key=lambda item: (item.bbox[1], item.bbox[0]))


def _sort_band(blocks: list[_TextBlock], page_width: float) -> list[_TextBlock]:
    center = page_width / 2
    left = [block for block in blocks if block.bbox[2] <= center * 1.03]
    right = [block for block in blocks if block.bbox[0] >= center * 0.97]
    other = [block for block in blocks if block not in left and block not in right]
    if len(left) >= 2 and len(right) >= 2 and not other:
        return sorted(left, key=lambda item: (item.bbox[1], item.bbox[0])) + sorted(
            right, key=lambda item: (item.bbox[1], item.bbox[0])
        )
    return sorted(blocks, key=lambda item: (item.bbox[1], item.bbox[0]))


def _reading_order(blocks: list[_TextBlock], page_width: float) -> list[_TextBlock]:
    full_width = [block for block in blocks if block.width >= page_width * 0.62]
    narrow = [block for block in blocks if block not in full_width]
    ordered: list[_TextBlock] = []
    previous_bottom = -1.0
    for anchor in sorted(full_width, key=lambda item: (item.bbox[1], item.bbox[0])):
        band = [
            block
            for block in narrow
            if previous_bottom <= block.bbox[1] < anchor.bbox[1]
        ]
        ordered.extend(_sort_band(band, page_width))
        ordered.append(anchor)
        previous_bottom = anchor.bbox[3]
    ordered.extend(
        _sort_band([block for block in narrow if block not in ordered], page_width)
    )
    return list(dict.fromkeys(ordered))


def _marker_annotations(blocks: list[_TextBlock]) -> tuple[list[dict[str, Any]], bool]:
    legend = ""
    for block in blocks:
        normalized = normalize_inline_text(block.text)
        if _EXPLICIT_MARKER_LEGEND.search(normalized):
            legend = normalized
        if legend:
            break
    if not legend:
        return [], False
    affected: list[str] = []
    for block in blocks:
        for line in block.lines:
            if "*" not in line or _EXPLICIT_MARKER_LEGEND.search(line):
                continue
            cleaned = normalize_inline_text(_LEGEND_ASTERISK.sub("", line))
            if cleaned and cleaned not in affected:
                affected.append(cleaned)
    return (
        [
            {
                "type": "legend_marker",
                "marker": "*",
                "legend_text": legend,
                "meaning": normalize_inline_text(_LEGEND_ASTERISK.sub("", legend)),
                "applies_to": affected,
            }
        ],
        True,
    )


def _vertical_gap(
    block: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> float:
    if block[3] < region[1]:
        return region[1] - block[3]
    if block[1] > region[3]:
        return block[1] - region[3]
    return 0.0


def _horizontal_overlap_ratio(
    block: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(block[2], region[2]) - max(block[0], region[0]))
    return overlap / max(1.0, min(block[2] - block[0], region[2] - region[0]))


def _nearby_region_labels(
    region: _Region,
    blocks: list[_TextBlock],
) -> list[_TextBlock]:
    """Attach only explicit, nearby captions and marker legends to a region."""

    caption_pattern = _TABLE_CAPTION if region.kind == "table" else _FIGURE_CAPTION
    result: list[_TextBlock] = []
    for block in blocks:
        normalized = normalize_inline_text(block.text)
        explicit_caption = bool(caption_pattern.match(normalized))
        explicit_legend = bool(_EXPLICIT_MARKER_LEGEND.search(normalized))
        if not (explicit_caption or explicit_legend):
            continue
        if _vertical_gap(block.bbox, region.bbox) > 96:
            continue
        if _horizontal_overlap_ratio(block.bbox, region.bbox) < 0.20:
            continue
        result.append(block)
    return sorted(result, key=lambda item: (item.bbox[1], item.bbox[0]))


def _label_annotations(
    kind: str,
    blocks: list[_TextBlock],
) -> list[dict[str, Any]]:
    caption_pattern = _TABLE_CAPTION if kind == "table" else _FIGURE_CAPTION
    return [
        {
            "type": "caption",
            "content_kind": kind,
            "text": normalize_reading_text(block.text),
        }
        for block in blocks
        if caption_pattern.match(normalize_inline_text(block.text))
    ]


def _quality_flags(text: str, kind: str) -> list[str]:
    """Mark only high-confidence extraction debris for retrieval rejection."""

    normalized = normalize_inline_text(text)
    alphanumeric = sum(character.isalnum() for character in normalized)
    alphabetic = sum(character.isalpha() for character in normalized)
    flags: list[str] = []
    if not normalized or alphanumeric == 0 or (kind == "prose" and alphabetic < 3):
        flags.append("extraction_artifact")
    return flags


def _split_prose_and_lists(
    blocks: list[_TextBlock],
) -> list[tuple[str, list[_TextBlock]]]:
    """Preserve reading order while separating list blocks from prose blocks."""

    groups: list[tuple[str, list[_TextBlock]]] = []
    for block in blocks:
        populated_lines = [line for line in block.lines if line.strip()]
        list_lines = sum(
            bool(_LIST_ITEM.match(line.strip())) for line in populated_lines
        )
        kind = (
            "list"
            if populated_lines and list_lines * 2 >= len(populated_lines)
            else "prose"
        )
        if groups and groups[-1][0] == kind:
            groups[-1][1].append(block)
        else:
            groups.append((kind, [block]))
    return groups


def _pdf_locator(page: pymupdf.Page, page_number: int) -> dict[str, Any]:
    try:
        page_label = str(page.get_label() or page_number)
    except (RuntimeError, ValueError):
        page_label = str(page_number)
    return {"type": "pdf_page", "page": page_number, "page_label": page_label}


def _extract_pdf(
    source: SourceFile,
    override: dict[str, Any],
    digest: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    digest = digest or sha256_file(source.path)
    document_id = _document_id(source, digest)
    try:
        document = pymupdf.open(source.path)
    except Exception as exc:
        raise ExtractionError(f"Cannot open PDF source: {source.path}") from exc

    try:
        if document.needs_pass:
            raise ExtractionError(
                f"Password-protected PDF is unsupported: {source.path}"
            )
        metadata = document.metadata or {}
        pages: list[tuple[list[_TextBlock], float, float]] = [
            (_page_blocks(page), float(page.rect.width), float(page.rect.height))
            for page in document
        ]
        automatic, provenance, warnings = _front_matter_identity(
            pages, metadata, source.path.stem
        )
        record = _base_metadata(
            source=source,
            document_id=document_id,
            digest=digest,
            automatic=automatic,
            provenance=provenance,
            warnings=warnings,
            override=override,
        )
        repeated_margins = _repeated_margin_signatures(pages)
        units: list[dict[str, Any]] = []
        empty_pages = 0
        removed_margin_blocks = 0
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            locator = _pdf_locator(page, page_number)
            blocks, page_width, page_height = pages[page_index]
            filtered: list[_TextBlock] = []
            for block in blocks:
                in_margin = (
                    block.bbox[3] <= page_height * 0.12
                    or block.bbox[1] >= page_height * 0.88
                )
                if in_margin and (
                    _block_signature(block) in repeated_margins
                    or _PAGE_NUMBER.fullmatch(normalize_inline_text(block.text))
                ):
                    removed_margin_blocks += 1
                    continue
                filtered.append(block)
            if not filtered:
                empty_pages += 1
                continue

            regions = _non_prose_regions(page)
            assigned: set[int] = set()
            page_units: list[tuple[str, list[_TextBlock]]] = []
            for region in regions:
                members = [
                    block
                    for block in filtered
                    if _rect_intersection_ratio(block.bbox, region.bbox) >= 0.35
                ]
                nearby_labels = _nearby_region_labels(
                    region,
                    [
                        block
                        for block in filtered
                        if block.number not in assigned and block not in members
                    ],
                )
                members.extend(nearby_labels)
                if not members:
                    continue
                assigned.update(block.number for block in members)
                page_units.append((region.kind, members))
            prose = [block for block in filtered if block.number not in assigned]
            if prose:
                page_units[0:0] = _split_prose_and_lists(
                    _reading_order(prose, page_width)
                )

            for region_index, (kind, members) in enumerate(page_units, 1):
                ordered = (
                    members
                    if kind == "prose"
                    else sorted(members, key=lambda item: (item.bbox[1], item.bbox[0]))
                )
                annotations = _label_annotations(kind, ordered)
                marker_annotations, strip_marker = _marker_annotations(ordered)
                annotations.extend(marker_annotations)
                paragraphs = []
                for block in ordered:
                    value = (
                        _LEGEND_ASTERISK.sub("", block.text)
                        if strip_marker
                        else block.text
                    )
                    if normalized := normalize_reading_text(value):
                        paragraphs.append(normalized)
                text = "\n\n".join(paragraphs)
                if not text:
                    continue
                units.append(
                    {
                        "id": f"{document_id}:pdf-page:{page_number:06d}:region:{region_index:03d}",
                        "document_id": document_id,
                        "title": record["title"],
                        "contents": text,
                        "content_kind": kind,
                        "annotations": annotations,
                        "quality_flags": _quality_flags(text, kind),
                        "locator": locator,
                    }
                )

        record.update(
            {
                "physical_pages": document.page_count,
                "extracted_units": len(units),
                "empty_units": empty_pages,
                "removed_repeated_margin_blocks": removed_margin_blocks,
            }
        )
    finally:
        document.close()

    if not units:
        raise ExtractionError(
            f"PDF produced no text; OCR may be required: {source.path}"
        )
    return record, units


def _epub_metadata_values(book: epub.EpubBook, name: str) -> list[str]:
    values: list[str] = []
    for value, _attributes in book.get_metadata("DC", name):
        normalized = normalize_inline_text(str(value))
        if normalized:
            values.append(normalized)
    return values


def _epub_visible_identity(
    book: epub.EpubBook,
    filename_stem: str,
) -> tuple[str, list[str]]:
    """Read a conservative title/byline fallback from the first visible sections."""

    for spine_entry in book.spine[:5]:
        item_id = spine_entry[0] if isinstance(spine_entry, tuple) else spine_entry
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        heading = soup.find(re.compile(r"^h1$")) or soup.find("title")
        visible_title = (
            normalize_inline_text(heading.get_text(" ", strip=True))
            if isinstance(heading, Tag)
            else ""
        )
        if not _valid_title(
            visible_title,
            filename_stem,
            reject_filename_match=False,
        ):
            visible_title = ""

        visible_authors: list[str] = []
        author_element = soup.find(
            attrs={"class": re.compile(r"(?:author|byline)", re.IGNORECASE)}
        )
        if not isinstance(author_element, Tag) and isinstance(heading, Tag):
            sibling = heading.find_next_sibling(["p", "div"])
            if isinstance(sibling, Tag) and _BYLINE.match(
                normalize_inline_text(sibling.get_text(" ", strip=True))
            ):
                author_element = sibling
        if isinstance(author_element, Tag):
            byline = _BYLINE.sub(
                "", normalize_inline_text(author_element.get_text(" ", strip=True))
            )
            visible_authors = [
                author
                for value in re.split(r"\s*(?:;|\band\b)\s*", byline)
                if (author := normalize_inline_text(value)) and _valid_author(author)
            ]
        if visible_title or visible_authors:
            return visible_title, visible_authors
    return "", []


def _extract_epub(
    source: SourceFile,
    override: dict[str, Any],
    digest: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    digest = digest or sha256_file(source.path)
    document_id = _document_id(source, digest)
    try:
        book = epub.read_epub(str(source.path), options={"ignore_ncx": True})
    except Exception as exc:
        raise ExtractionError(f"Cannot open EPUB source: {source.path}") from exc

    titles = _epub_metadata_values(book, "title")
    authors = _epub_metadata_values(book, "creator")
    identifiers = _epub_metadata_values(book, "identifier")
    dates = _epub_metadata_values(book, "date")
    visible_title, visible_authors = _epub_visible_identity(book, source.path.stem)
    opf_title = next(
        (item for item in titles if _valid_title(item, source.path.stem)),
        "",
    )
    title = opf_title or visible_title or source.path.stem
    title_source = (
        "epub_opf" if opf_title else "epub_visible" if visible_title else "filename"
    )
    valid_opf_authors = [author for author in authors if _valid_author(author)]
    resolved_authors = valid_opf_authors or visible_authors
    year_match = next(
        (_YEAR.search(item) for item in dates if _YEAR.search(item)), None
    )
    automatic = {
        "title": title,
        "authors": resolved_authors,
        "year": int(year_match.group(1)) if year_match else None,
        "doi": next((_doi(item) for item in identifiers if _doi(item)), ""),
    }
    provenance = {
        "title": title_source,
        "authors": (
            "epub_opf"
            if valid_opf_authors
            else "epub_visible"
            if visible_authors
            else "missing"
        ),
        "year": "epub_opf" if year_match else "missing",
        "doi": "epub_opf" if automatic["doi"] else "missing",
    }
    record = _base_metadata(
        source=source,
        document_id=document_id,
        digest=digest,
        automatic=automatic,
        provenance=provenance,
        warnings=(
            ["conflicting_candidates"]
            if opf_title
            and visible_title
            and opf_title.casefold() != visible_title.casefold()
            else []
        ),
        override=override,
    )

    units: list[dict[str, Any]] = []
    empty_sections = 0
    for spine_position, spine_entry in enumerate(book.spine, 1):
        item_id = spine_entry[0] if isinstance(spine_entry, tuple) else spine_entry
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for unwanted in soup(["script", "style", "nav"]):
            unwanted.decompose()
        heading = soup.find(re.compile(r"^h[1-6]$"))
        section_title = (
            normalize_inline_text(heading.get_text(" ", strip=True))
            if heading is not None
            else ""
        )
        href = str(item.get_name() or "")
        locator = {
            "type": "epub_section",
            "section_index": spine_position,
            "section_title": section_title,
            "href": href,
        }
        prose_parts: list[str] = []
        tables: list[str] = []
        for element in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "table"]
        ):
            if not isinstance(element, Tag):
                continue
            if element.name == "table":
                rows = []
                for row in element.find_all("tr"):
                    cells = [
                        normalize_inline_text(cell.get_text(" ", strip=True))
                        for cell in row.find_all(["th", "td"])
                    ]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    tables.append("\n".join(rows))
            elif element.find_parent("table") is None:
                value = normalize_reading_text(element.get_text("\n", strip=True))
                if value:
                    prose_parts.append(value)
        if not prose_parts and not tables:
            empty_sections += 1
            continue
        region_index = 0
        if prose_parts:
            region_index += 1
            units.append(
                {
                    "id": f"{document_id}:epub-section:{spine_position:06d}:region:{region_index:03d}",
                    "document_id": document_id,
                    "title": record["title"],
                    "contents": "\n\n".join(prose_parts),
                    "content_kind": "prose",
                    "annotations": [],
                    "quality_flags": _quality_flags("\n\n".join(prose_parts), "prose"),
                    "locator": locator,
                }
            )
        for table in tables:
            region_index += 1
            units.append(
                {
                    "id": f"{document_id}:epub-section:{spine_position:06d}:region:{region_index:03d}",
                    "document_id": document_id,
                    "title": record["title"],
                    "contents": table,
                    "content_kind": "table",
                    "annotations": [],
                    "quality_flags": _quality_flags(table, "table"),
                    "locator": locator,
                }
            )

    record.update(
        {
            "spine_items": len(book.spine),
            "extracted_units": len(units),
            "empty_units": empty_sections,
        }
    )
    if not units:
        raise ExtractionError(f"EPUB produced no readable sections: {source.path}")
    return record, units


def extract_sources(
    sources: tuple[SourceFile, ...],
    metadata_overrides: dict[str, dict[str, Any]],
    source_digests: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return indexed documents and cleaned semantic units."""

    documents: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for source in sources:
        override = metadata_overrides.get(source.source_relative_path, {})
        digest = (source_digests or {}).get(source.source_relative_path)
        if source.extension == ".pdf":
            document, extracted = _extract_pdf(source, override, digest)
        elif source.extension == ".epub":
            document, extracted = _extract_epub(source, override, digest)
        else:  # pragma: no cover
            raise ExtractionError(f"Unsupported source format: {source.path}")
        documents.append(document)
        units.extend(extracted)
    return documents, units

"""Page- and section-aware extraction for original PDF and EPUB sources."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import ebooklib
import pymupdf
from bs4 import BeautifulSoup
from ebooklib import epub

from .sources import SourceFile, sha256_file


class ExtractionError(RuntimeError):
    """Raised when a source cannot be represented safely in the knowledge base."""


def _clean_text(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _document_id(source: SourceFile, digest: str) -> str:
    identity = f"{source.source_relative_path}\0{digest}".encode()
    return f"doc_{hashlib.sha256(identity).hexdigest()[:24]}"


def _base_metadata(
    *,
    source: SourceFile,
    document_id: str,
    digest: str,
    title: str,
    authors: list[str],
    override: dict[str, Any],
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "source_path": source.project_relative_path,
        "source_relative_path": source.source_relative_path,
        "format": source.extension.removeprefix("."),
        "sha256": digest,
        "size": source.size,
        "mtime_ns": source.mtime_ns,
        "title": override.get("title") or title or source.path.stem,
        "authors": override.get("authors", authors),
        "year": override.get("year"),
        "doi": override.get("doi", ""),
        "categories": override.get("categories", []),
        "keywords": override.get("keywords", []),
    }


def _pdf_author_list(metadata: dict[str, Any]) -> list[str]:
    author = str(metadata.get("author") or "").strip()
    if not author:
        return []
    parts = [item.strip() for item in re.split(r"\s*[;|]\s*", author)]
    return [item for item in parts if item]


def _extract_pdf(
    source: SourceFile,
    override: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    digest = sha256_file(source.path)
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
        title = str(metadata.get("title") or source.path.stem).strip()
        record = _base_metadata(
            source=source,
            document_id=document_id,
            digest=digest,
            title=title,
            authors=_pdf_author_list(metadata),
            override=override,
        )
        units: list[dict[str, Any]] = []
        empty_pages = 0
        for page_index, page in enumerate(document):
            text = _clean_text(page.get_text("text", sort=True))
            if not text:
                empty_pages += 1
                continue
            page_number = page_index + 1
            try:
                page_label = str(page.get_label() or page_number)
            except (RuntimeError, ValueError):
                page_label = str(page_number)
            units.append(
                {
                    "id": f"{document_id}:pdf-page:{page_number:06d}",
                    "document_id": document_id,
                    "title": record["title"],
                    "contents": text,
                    "locator": {
                        "type": "pdf_page",
                        "page": page_number,
                        "page_label": page_label,
                    },
                }
            )
        record.update(
            {
                "physical_pages": document.page_count,
                "extracted_units": len(units),
                "empty_units": empty_pages,
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
        normalized = str(value).strip()
        if normalized:
            values.append(normalized)
    return values


def _extract_epub(
    source: SourceFile,
    override: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    digest = sha256_file(source.path)
    document_id = _document_id(source, digest)
    try:
        book = epub.read_epub(str(source.path), options={"ignore_ncx": True})
    except Exception as exc:
        raise ExtractionError(f"Cannot open EPUB source: {source.path}") from exc

    titles = _epub_metadata_values(book, "title")
    authors = _epub_metadata_values(book, "creator")
    record = _base_metadata(
        source=source,
        document_id=document_id,
        digest=digest,
        title=titles[0] if titles else source.path.stem,
        authors=authors,
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
        for unwanted in soup(["script", "style"]):
            unwanted.decompose()
        heading = soup.find(re.compile(r"^h[1-6]$"))
        section_title = (
            _clean_text(heading.get_text(" ", strip=True))
            if heading is not None
            else ""
        )
        text = _clean_text(soup.get_text("\n"))
        if not text:
            empty_sections += 1
            continue
        href = str(item.get_name() or "")
        units.append(
            {
                "id": f"{document_id}:epub-section:{spine_position:06d}",
                "document_id": document_id,
                "title": record["title"],
                "contents": text,
                "locator": {
                    "type": "epub_section",
                    "section_index": spine_position,
                    "section_title": section_title,
                    "href": href,
                },
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for source in sources:
        override = metadata_overrides.get(source.source_relative_path, {})
        if source.extension == ".pdf":
            document, extracted = _extract_pdf(source, override)
        elif source.extension == ".epub":
            document, extracted = _extract_epub(source, override)
        else:  # pragma: no cover - scan_sources enforces this invariant.
            raise ExtractionError(f"Unsupported source format: {source.path}")
        documents.append(document)
        units.extend(extracted)
    return documents, units

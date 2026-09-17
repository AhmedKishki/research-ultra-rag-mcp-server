from __future__ import annotations

import sys
from pathlib import Path

import pytest
from conftest import write_epub, write_pdf

from research_ultra_rag_mcp.config import ConfigurationError, resolve_config
from research_ultra_rag_mcp.extraction import (
    extract_sources,
    normalize_inline_text,
    normalize_reading_text,
)
from research_ultra_rag_mcp.sources import SourcePolicyError, scan_sources


def test_only_pdf_and_epub_are_selected(project: Path) -> None:
    write_pdf(project / "sources" / "article.pdf", ["A cobalt research passage."])
    write_epub(project / "sources" / "book.epub", "An amber research passage.")
    (project / "sources" / "notes.md").write_text("derived notes", encoding="utf-8")

    config = resolve_config(project, vanilla_executable=sys.executable)
    scan = scan_sources(config)

    assert [item.extension for item in scan.selected] == [".pdf", ".epub"]
    assert scan.ignored_extensions == {".md": 1}


def test_pdf_pages_and_epub_sections_preserve_locators(project: Path) -> None:
    write_pdf(
        project / "sources" / "article.pdf",
        ["First page cobalt evidence.", "Second page amber evidence."],
        title="Located PDF",
    )
    write_epub(
        project / "sources" / "book.epub",
        "Section evidence about quartz.",
        title="Located EPUB",
    )
    config = resolve_config(project, vanilla_executable=sys.executable)

    documents, units = extract_sources(scan_sources(config).selected, {})

    assert {item["format"] for item in documents} == {"pdf", "epub"}
    pdf_units = [item for item in units if item["locator"]["type"] == "pdf_page"]
    epub_units = [item for item in units if item["locator"]["type"] == "epub_section"]
    assert [item["locator"]["page"] for item in pdf_units] == [1, 2]
    assert any("quartz" in item["contents"] for item in epub_units)
    assert all("section_index" in item["locator"] for item in epub_units)


def test_extracted_text_removes_layout_wrapping() -> None:
    raw = (
        "A sentence wraps in the\nmiddle of a line.\n\n"
        "A com-\nmodity remains readable.\r\n\r\nFinal paragraph."
    )

    assert normalize_reading_text(raw) == (
        "A sentence wraps in the middle of a line.\n\n"
        "A commodity remains readable.\n\nFinal paragraph."
    )
    assert normalize_inline_text(raw) == (
        "A sentence wraps in the middle of a line. "
        "A commodity remains readable. Final paragraph."
    )


def test_source_directory_cannot_escape_project(project: Path) -> None:
    with pytest.raises(ConfigurationError, match="escapes"):
        resolve_config(
            project,
            source_directory="../outside",
            vanilla_executable=sys.executable,
        )


def test_pdf_symlink_is_rejected(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.pdf"
    write_pdf(outside, ["outside"])
    (project / "sources" / "linked.pdf").symlink_to(outside)
    config = resolve_config(project, vanilla_executable=sys.executable)

    with pytest.raises(SourcePolicyError, match="Symbolic-link"):
        scan_sources(config)

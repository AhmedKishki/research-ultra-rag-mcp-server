from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from ebooklib import epub


def write_pdf(path: Path, pages: list[str], *, title: str = "Test PDF") -> None:
    document = pymupdf.open()
    document.set_metadata({"title": title, "author": "Test Author"})
    for text in pages:
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(72, 72, 540, 760),
            text,
            fontsize=11,
        )
    document.save(path)
    document.close()


def write_epub(path: Path, text: str, *, title: str = "Test EPUB") -> None:
    book = epub.EpubBook()
    book.set_identifier("test-identifier")
    book.set_title(title)
    book.set_language("en")
    book.add_author("EPUB Author")
    chapter = epub.EpubHtml(
        title="Opening Chapter",
        file_name="chapter-1.xhtml",
        lang="en",
    )
    chapter.content = f"<h1>Opening Chapter</h1><p>{text}</p>"
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.toc = (epub.Link("chapter-1.xhtml", "Opening Chapter", "chapter-1"),)
    book.spine = ["nav", chapter]
    epub.write_epub(str(path), book)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "research-project"
    (root / "sources").mkdir(parents=True)
    return root

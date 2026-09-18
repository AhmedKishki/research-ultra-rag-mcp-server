from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf
import pytest
from conftest import write_epub, write_pdf
from ebooklib import epub

from research_ultra_rag_mcp.config import (
    ConfigurationError,
    configured_source_directory,
    resolve_config,
)
from research_ultra_rag_mcp.extraction import (
    _marker_annotations,
    _pdf_line_text,
    _TextBlock,
    extract_sources,
    normalize_inline_text,
    normalize_reading_text,
)
from research_ultra_rag_mcp.sources import SourcePolicyError, scan_sources
from research_ultra_rag_mcp.ultrarag import create_vanilla_transport


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
        "A com-\nmodity remains readable.\r\n\r\n"
        "A PDF span says waste- value.\n\nFinal paragraph."
    )

    assert normalize_reading_text(raw) == (
        "A sentence wraps in the middle of a line.\n\n"
        "A commodity remains readable.\n\n"
        "A PDF span says waste-value.\n\nFinal paragraph."
    )
    assert normalize_inline_text(raw) == (
        "A sentence wraps in the middle of a line. "
        "A commodity remains readable. "
        "A PDF span says waste-value. Final paragraph."
    )


def test_pdf_split_symbol_font_dashes_are_restored_from_geometry() -> None:
    spans = [
        {"text": "before", "font": "Body", "bbox": (0, 0, 30, 10)},
        {"text": "*", "font": "Symbol", "bbox": (30, 0, 39, 10)},
        {"text": "/", "font": "Body", "bbox": (38.5, 4, 41, 9)},
        {"text": "after", "font": "Body", "bbox": (40, 0, 65, 10)},
        {"text": " ", "font": "Body", "bbox": (65, 0, 68, 10)},
        {"text": "post", "font": "Body", "bbox": (68, 0, 88, 10)},
        {"text": "\x01", "font": "Symbol", "bbox": (88, 0, 93, 10)},
        {"text": "/", "font": "Body", "bbox": (92.5, 4, 95, 9)},
        {"text": "cold", "font": "Body", "bbox": (94, 0, 114, 10)},
    ]

    assert _pdf_line_text(spans) == "before—after post–cold"


def test_marker_legend_detection_requires_explicit_legend_syntax() -> None:
    prose = _TextBlock(
        number=1,
        bbox=(0, 0, 100, 20),
        lines=("histories*/rather than this means something else",),
        text="histories*/rather than this means something else",
        font_size=10,
    )

    assert _marker_annotations([prose]) == ([], False)


def test_pdf_front_matter_resolves_title_authors_year_and_doi(project: Path) -> None:
    path = project / "sources" / "tsing.pdf"
    document = pymupdf.open()
    document.set_metadata(
        {
            "title": "doi:10.1080/08935690902743088",
            "author": "Unknown",
        }
    )
    page = document.new_page()
    page.insert_text((72, 90), "Supply Chains and the Human Condition", fontsize=20)
    page.insert_text((72, 132), "Anna Tsing", fontsize=14)
    page.insert_text((72, 165), "Copyright 2009", fontsize=10)
    page.insert_textbox(
        pymupdf.Rect(72, 210, 530, 500),
        "This article examines supply chains and the conditions of human labour. "
        "The discussion provides enough prose to establish the normal body font.",
        fontsize=10,
    )
    document.save(path)
    document.close()

    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, _units = extract_sources(scan_sources(config).selected, {})
    source = documents[0]

    assert source["title"] == "Supply Chains and the Human Condition"
    assert source["authors"] == ["Anna Tsing"]
    assert source["year"] == 2009
    assert source["doi"] == "10.1080/08935690902743088"
    assert source["metadata_provenance"] == {
        "title": "pdf_front_matter",
        "authors": "pdf_front_matter",
        "year": "pdf_front_matter",
        "doi": "pdf_metadata",
        "categories": "missing",
        "keywords": "missing",
    }
    assert source["metadata_confidence"]["title"] > 0.9
    assert "conflicting_candidates" not in source["metadata_warnings"]


def test_explicit_pdf_byline_outranks_name_like_subtitle(project: Path) -> None:
    path = project / "sources" / "report.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 90), "Systems and Society", fontsize=24)
    page.insert_text(
        (72, 125),
        "A study of labor, infrastructure and material resources",
        fontsize=9,
    )
    page.insert_text((72, 150), "By Ada Example and Ben Example", fontsize=10)
    page.insert_text((72, 172), "Copyright 2024", fontsize=10)
    page.insert_textbox(
        pymupdf.Rect(72, 220, 530, 500),
        "Body prose about the material resources and labor behind a system.",
        fontsize=10,
    )
    document.save(path)
    document.close()

    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, _units = extract_sources(scan_sources(config).selected, {})
    source = documents[0]

    assert source["title"] == "Systems and Society"
    assert source["authors"] == ["Ada Example", "Ben Example"]
    assert source["year"] == 2024
    assert source["metadata_provenance"]["authors"] == "pdf_front_matter"
    assert source["metadata_provenance"]["year"] == "pdf_front_matter"


def test_doi_title_becomes_doi_and_filename_is_only_title_fallback(
    project: Path,
) -> None:
    path = project / "sources" / "fallback-title.pdf"
    document = pymupdf.open()
    document.set_metadata(
        {"title": "DOI: 10.5555/example.42", "author": "Microsoft Word"}
    )
    page = document.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 100, 530, 700),
        "Ordinary body prose without a visible title or byline.",
        fontsize=10,
    )
    document.save(path)
    document.close()

    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, _units = extract_sources(scan_sources(config).selected, {})
    source = documents[0]

    assert source["title"] == "fallback-title"
    assert source["authors"] == []
    assert source["doi"] == "10.5555/example.42"
    assert source["metadata_provenance"]["title"] == "filename"
    assert {"title_from_filename", "authors_missing"}.issubset(
        source["metadata_warnings"]
    )


def test_valid_pdf_embedded_metadata_is_used_independently(project: Path) -> None:
    path = project / "sources" / "well-named-article.pdf"
    document = pymupdf.open()
    document.set_metadata(
        {
            "title": "Embedded Article Title",
            "author": "Ada Lovelace; Grace Hopper",
            "creationDate": "D:20240310000000Z",
        }
    )
    page = document.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 100, 530, 700),
        "Ordinary body prose without a high-confidence visible title or byline.",
        fontsize=10,
    )
    document.save(path)
    document.close()

    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, _units = extract_sources(scan_sources(config).selected, {})
    source = documents[0]

    assert source["title"] == "Embedded Article Title"
    assert source["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert source["year"] == 2024
    assert source["metadata_provenance"]["title"] == "pdf_metadata"
    assert source["metadata_provenance"]["authors"] == "pdf_metadata"
    assert source["metadata_provenance"]["year"] == "pdf_metadata"


def test_visible_cover_title_may_legitimately_match_filename(project: Path) -> None:
    path = project / "sources" / "Visible Cover Title.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 180), "Visible Cover Title", fontsize=28)
    page.insert_text((72, 230), "By Visible Author", fontsize=14)
    document.save(path)
    document.close()

    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, _units = extract_sources(scan_sources(config).selected, {})

    assert documents[0]["title"] == "Visible Cover Title"
    assert documents[0]["metadata_provenance"]["title"] == "pdf_front_matter"
    assert "title_from_filename" not in documents[0]["metadata_warnings"]


def test_epub_uses_opf_then_visible_metadata_and_reviewed_overrides(
    project: Path,
) -> None:
    opf = project / "sources" / "opf.epub"
    write_epub(opf, "EPUB evidence.", title="Validated OPF Title")

    visible = project / "sources" / "visible.epub"
    book = epub.EpubBook()
    book.set_identifier("10.1000/visible")
    book.set_title("doi:10.1000/visible")
    book.set_language("en")
    book.add_author("Unknown")
    chapter = epub.EpubHtml(title="Visible", file_name="visible.xhtml", lang="en")
    chapter.content = (
        "<h1>The Visible Book Title</h1>"
        '<p class="byline">By Ada Lovelace and Grace Hopper</p>'
        "<p>Semantic evidence in the opening chapter.</p>"
    )
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    epub.write_epub(str(visible), book)

    config = resolve_config(project, vanilla_executable=sys.executable)
    overrides = {
        "opf.epub": {
            "title": "Reviewed Book Title",
            "authors": ["Reviewed Author"],
            "year": 2025,
            "doi": "10.2000/reviewed",
            "categories": ["History"],
            "keywords": ["Archives"],
        }
    }
    documents, _units = extract_sources(scan_sources(config).selected, overrides)
    by_path = {item["source_relative_path"]: item for item in documents}

    assert by_path["opf.epub"]["title"] == "Reviewed Book Title"
    assert by_path["opf.epub"]["authors"] == ["Reviewed Author"]
    assert set(by_path["opf.epub"]["metadata_provenance"].values()) == {
        "reviewed_override"
    }
    assert by_path["visible.epub"]["title"] == "The Visible Book Title"
    assert by_path["visible.epub"]["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert by_path["visible.epub"]["doi"] == "10.1000/visible"
    assert by_path["visible.epub"]["metadata_provenance"]["title"] == ("epub_visible")


def test_pdf_layout_restores_columns_removes_margins_and_separates_lists(
    project: Path,
) -> None:
    path = project / "sources" / "layout.pdf"
    document = pymupdf.open()
    document.set_metadata({"title": "Layout Study", "author": "Layout Author"})
    for page_number in range(1, 4):
        page = document.new_page()
        page.insert_text((72, 35), "REPEATED JOURNAL HEADER", fontsize=8)
        page.insert_text((290, 810), str(page_number), fontsize=8)
        page.insert_textbox(
            pymupdf.Rect(55, 120, 270, 210),
            f"Left column first {page_number}. Left column second sentence.",
            fontsize=10,
        )
        page.insert_textbox(
            pymupdf.Rect(55, 240, 270, 330),
            "- First listed item\n- Second listed item",
            fontsize=10,
        )
        page.insert_textbox(
            pymupdf.Rect(325, 120, 540, 210),
            f"Right column follows {page_number}. Right column second sentence.",
            fontsize=10,
        )
        page.insert_textbox(
            pymupdf.Rect(325, 240, 540, 330),
            "Right column final paragraph.",
            fontsize=10,
        )
    document.save(path)
    document.close()

    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, units = extract_sources(scan_sources(config).selected, {})

    combined = "\n".join(item["contents"] for item in units)
    assert "REPEATED JOURNAL HEADER" not in combined
    assert combined.index("Left column first 1") < combined.index(
        "Right column follows 1"
    )
    assert any(item["content_kind"] == "list" for item in units)
    assert documents[0]["removed_repeated_margin_blocks"] >= 6


def test_figure_markers_are_cleaned_but_preserved_as_annotations(project: Path) -> None:
    path = project / "sources" / "figure.pdf"
    document = pymupdf.open()
    document.set_metadata({"title": "Figure Study", "author": "Figure Author"})
    page = document.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 80, 520, 150),
        "Prose introducing the evidence shown in the figure below.",
        fontsize=10,
    )
    page.draw_rect(pymupdf.Rect(65, 200, 520, 430), width=1)
    page.insert_text((90, 245), "Lenovo*", fontsize=11)
    page.insert_text((90, 285), "Samsung*", fontsize=11)
    page.insert_text(
        (90, 390), "* indicates companies included in the sample.", fontsize=10
    )
    page.insert_text(
        (90, 455), "Figure 1. Companies included in the sample.", fontsize=10
    )
    document.save(path)
    document.close()

    config = resolve_config(project, vanilla_executable=sys.executable)
    _documents, units = extract_sources(scan_sources(config).selected, {})
    figure = next(item for item in units if item["content_kind"] == "figure")

    assert "*" not in figure["contents"]
    assert "Figure 1. Companies included in the sample." in figure["contents"]
    assert any(
        annotation["type"] == "caption"
        and annotation["text"] == "Figure 1. Companies included in the sample."
        for annotation in figure["annotations"]
    )
    annotation = next(
        item for item in figure["annotations"] if item["type"] == "legend_marker"
    )
    assert annotation["marker"] == "*"
    assert annotation["legend_text"].startswith("* indicates")
    assert annotation["applies_to"] == ["Lenovo", "Samsung"]


def test_source_directory_cannot_escape_project(project: Path) -> None:
    with pytest.raises(ConfigurationError, match="escapes"):
        resolve_config(
            project,
            source_directory="../outside",
            vanilla_executable=sys.executable,
        )


def test_initialized_source_directory_can_be_reused(project: Path) -> None:
    (project / "library").mkdir()
    resolve_config(
        project,
        source_directory="library",
        vanilla_executable=sys.executable,
    )

    assert configured_source_directory(project) == "library"


def test_new_project_keeps_all_research_state_under_one_root(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)

    assert config.portable_root == project / ".research-rag"
    assert config.state_root == project / ".research-rag" / "runtime"
    assert not (project / ".ultrarag").exists()


def test_portable_project_identity_is_stable_and_legacy_review_state_migrates(
    project: Path,
) -> None:
    legacy = project / ".ultrarag" / "research"
    legacy.mkdir(parents=True)
    (legacy / "source-metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {"article.pdf": {"title": "Reviewed title"}},
            }
        ),
        encoding="utf-8",
    )
    (legacy / "source-exclusions.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    "duplicate.pdf": {
                        "reason": "Reviewed duplicate",
                        "excluded_at": "2026-09-18T00:00:00Z",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (legacy / "current.json").write_text(
        json.dumps({"schema_version": 1, "generation_id": "legacy-generation"}),
        encoding="utf-8",
    )

    first = resolve_config(project, vanilla_executable=sys.executable)
    second = resolve_config(project, vanilla_executable=sys.executable)

    assert first.project_id == second.project_id
    descriptor = json.loads(first.project_config_path.read_text(encoding="utf-8"))
    assert descriptor == {
        "schema_version": 1,
        "project_id": first.project_id,
        "name": project.name,
        "source_directory": "sources",
    }
    assert json.loads(first.metadata_path.read_text(encoding="utf-8"))["sources"] == {
        "article.pdf": {"title": "Reviewed title"}
    }
    assert (
        "duplicate.pdf"
        in json.loads(first.source_exclusions_path.read_text(encoding="utf-8"))[
            "sources"
        ]
    )
    assert first.state_root == project / ".research-rag" / "runtime"
    assert (
        json.loads(first.current_path.read_text(encoding="utf-8"))["generation_id"]
        == "legacy-generation"
    )
    assert not legacy.exists()
    assert not (project / ".ultrarag").exists()


def test_conflicting_legacy_and_consolidated_runtime_is_refused(
    project: Path,
) -> None:
    legacy = project / ".ultrarag" / "research"
    consolidated = project / ".research-rag" / "runtime"
    legacy.mkdir(parents=True)
    consolidated.mkdir(parents=True)
    (legacy / "current.json").write_text("legacy", encoding="utf-8")
    (consolidated / "current.json").write_text("consolidated", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="both"):
        resolve_config(project, vanilla_executable=sys.executable)


def test_legacy_migration_preserves_other_ultrarag_state(project: Path) -> None:
    legacy = project / ".ultrarag" / "research"
    unrelated = project / ".ultrarag" / "vanilla-ui" / "settings.json"
    legacy.mkdir(parents=True)
    unrelated.parent.mkdir(parents=True)
    (legacy / "current.json").write_text("legacy", encoding="utf-8")
    unrelated.write_text("unrelated", encoding="utf-8")

    config = resolve_config(project, vanilla_executable=sys.executable)

    assert (config.state_root / "current.json").read_text(encoding="utf-8") == (
        "legacy"
    )
    assert unrelated.read_text(encoding="utf-8") == "unrelated"


def test_projects_share_only_the_configured_model_cache(tmp_path: Path) -> None:
    first_project = tmp_path / "first-project"
    second_project = tmp_path / "second-project"
    (first_project / "sources").mkdir(parents=True)
    (second_project / "sources").mkdir(parents=True)
    shared_models = tmp_path / "shared-models"

    first = resolve_config(
        first_project,
        vanilla_executable=sys.executable,
        model_cache_root=shared_models,
    )
    second = resolve_config(
        second_project,
        vanilla_executable=sys.executable,
        model_cache_root=shared_models,
    )

    assert first.model_cache_root == second.model_cache_root == shared_models
    assert first.state_root != second.state_root
    assert first.generations_root != second.generations_root
    assert not first.legacy_models_root.exists()
    assert not second.legacy_models_root.exists()


def test_research_starts_only_required_ultrarag_namespaces(project: Path) -> None:
    config = resolve_config(project, vanilla_executable=sys.executable)
    transport = create_vanilla_transport(config)

    selected = [
        transport.args[index + 1]
        for index, argument in enumerate(transport.args)
        if argument == "--namespace"
    ]
    assert selected == ["corpus", "retriever"]


def test_offline_mode_can_read_an_existing_legacy_model_cache(
    project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_cache = project / ".ultrarag" / "research" / "models"
    old_cache.mkdir(parents=True)
    (old_cache / "cached-model.bin").write_bytes(b"cached")
    empty_shared = tmp_path / "empty-global-cache"
    monkeypatch.setattr(
        "research_ultra_rag_mcp.config.user_cache_path",
        lambda *_args, **_kwargs: empty_shared,
    )

    config = resolve_config(
        project,
        vanilla_executable=sys.executable,
        offline=True,
    )

    migrated_cache = project / ".research-rag" / "runtime" / "models"
    assert config.model_cache_root == migrated_cache
    assert (migrated_cache / "cached-model.bin").read_bytes() == b"cached"
    assert not old_cache.exists()


def test_pdf_symlink_is_rejected(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.pdf"
    write_pdf(outside, ["outside"])
    (project / "sources" / "linked.pdf").symlink_to(outside)
    config = resolve_config(project, vanilla_executable=sys.executable)

    with pytest.raises(SourcePolicyError, match="Symbolic-link"):
        scan_sources(config)

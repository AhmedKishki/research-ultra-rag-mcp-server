from pathlib import Path


def test_readme_is_a_chronological_standalone_user_manual() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    headings = [
        "## Purpose",
        "## What the server provides",
        "## Install once",
        "## Create an isolated research project",
        "## Connect an AI agent",
        "## First use",
        "## Research workflow",
        "## Use the UI",
        "## Use the terminal verifier",
        "## Where project data is stored",
        "## Export, import, and move a project",
        "## Complete MCP tool reference",
        "## How it works under the hood",
        "## Limitations and troubleshooting",
        "## UltraRAG credit and licensing",
    ]
    positions = [readme.index(heading) for heading in headings]
    assert positions == sorted(positions)

    required = (
        "uv sync --frozen",
        '"mcpServers"',
        "research-ultra-rag-ui",
        "http://127.0.0.1:5051",
        "research-ultra-rag-verify",
        "--force-recompute",
        "work_budget_seconds=45",
        'status="in_progress"',
        "metadata_overlay_active",
        "effective_immediately",
        "discovered_sources",
        "exactly one selector",
        "canonical `contents`",
        "lean lookup payloads",
        "corrupt-text",
        "research-ultra-rag-bundle export",
        "research-ultra-rag-bundle import",
        ".research-rag/project.json",
        "~/.cache/research-ultra-rag-mcp/models",
        '"direct_quote_safe": false',
        "responsible for having the right to redistribute",
        "THUNLP",
        "NEUIR",
        "OpenBMB",
        "AI9stars",
        "NOTICE",
    )
    for value in required:
        assert value in readme

    for tool in (
        "status",
        "ingest",
        "search",
        "list_sources",
        "get_passage",
        "set_source_metadata",
        "set_source_inclusion",
        "export_bundle",
        "import_bundle",
    ):
        assert f"`{tool}`" in readme

    lowered = readme.casefold()
    assert "vanilla-ultra-rag-mcp" not in lowered
    assert "every `ingest` is a complete rebuild" not in lowered
    assert ".ultrarag/research/models" not in readme
    assert ".research-rag/runtime/" in readme
    assert "raw-extraction.jsonl" not in readme
    assert "ultrarag-chunks.jsonl" not in readme
    assert "metadata_confidence" not in readme
    assert "high-confidence" not in lowered


def test_features_document_separates_upstream_from_added_work() -> None:
    features = (Path(__file__).parents[1] / "FEATURES.md").read_text(encoding="utf-8")

    headings = [
        "# Features",
        "## 1. What UltraRAG provides, and what this server actually uses",
        "## 2. Features added on top of UltraRAG",
        "## 3. What this server deliberately does not do",
        "## 4. Planned additions",
        "## 5. Comparison with another MCP RAG server",
    ]
    positions = [features.index(heading) for heading in headings]
    assert positions == sorted(positions)

    # The document must keep naming what comes from upstream, what is added here,
    # and which work is only planned.
    for value in (
        "**UltraRAG**",
        "**Added here**",
        "**Planned**",
        "corpus_chunk_documents",
        "retriever_bm25_search",
        "mcp-rag-server",
        "Not yet measured",
    ):
        assert value in features

    # It must not claim unbuilt work as shipped.
    lowered = features.casefold()
    assert "is implemented" not in lowered
    assert "we have measured retrieval quality" not in lowered


def test_markdown_has_no_hard_wrapped_prose() -> None:
    """Paragraphs and list items must each be one line, not wrapped at a column."""

    import re

    block_start = re.compile(
        r"^(#{1,6}\s|\s*[-*+]\s|\s*\d+[.)]\s|\s*\||\s*>|\s*```|\s*~~~|\s*[-*_]{3,}\s*$)"
    )
    for path in sorted((Path(__file__).parents[1]).glob("*.md")):
        in_code = False
        previous_blank = True
        previous_block = True
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith(("```", "~~~")):
                in_code = not in_code
                previous_blank = previous_block = True
                continue
            if in_code:
                previous_blank = previous_block = True
                continue
            if not line.strip():
                previous_blank = previous_block = True
                continue
            is_block = bool(block_start.match(line))
            assert previous_blank or previous_block or is_block, (
                f"{path.name}:{number} is a wrapped continuation line: "
                f"{line.strip()[:60]!r}"
            )
            previous_blank = False
            previous_block = is_block


def test_agent_docs_define_source_identity_and_current_storage_contract() -> None:
    root = Path(__file__).parents[1]
    agent_guide = (root / "AGENT_GUIDE.md").read_text(encoding="utf-8")
    engineering_guide = (root / "AGENTS.md").read_text(encoding="utf-8")
    combined = f"{agent_guide}\n{engineering_guide}"

    for value in (
        "discovered_sources",
        "source_id",
        "document_id",
        "exactly one selector",
        "contents",
        "chunk_id",
    ):
        assert value in combined

    assert "metadata_confidence" not in combined
    assert "high-confidence" not in combined.casefold()

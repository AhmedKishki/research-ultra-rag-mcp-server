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

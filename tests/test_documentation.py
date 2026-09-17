from pathlib import Path


def test_readme_is_a_standalone_user_manual() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")

    assert "vanilla-ultra-rag-mcp" not in readme.casefold()
    for heading in (
        "## What the server offers",
        "## How it works",
        "## Install",
        "## Configure your MCP client",
        "## Use it",
        "## Expected search result",
    ):
        assert heading in readme

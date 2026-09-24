"""The import boundary: the domain core must not reach for the MCP surface.

The package is one distribution with two layers. The domain core — the service,
the generation store, extraction, the dense stack, the settings — is reusable on
its own: the evaluation harness imports it directly and scores in process, while
the browser UI and the verifier reach the same state through the MCP tool
surface. Nothing but this test keeps that arrow pointing one way, so it asserts
both halves of it: only the surface modules may import MCP machinery, and no core
module may import the surface or an entry point.

Imports are read with ``ast`` rather than executed, so the test cannot be fooled
by an import that only succeeds in one environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "research_ultra_rag_mcp"

# The modules allowed to import MCP machinery: the tool surface and its
# projector, the stdio transports, the two entry points that talk to a server,
# and the typed boundary over the vanilla runtime, which reaches it over MCP.
MCP_MODULES = frozenset(
    {"server.py", "transport.py", "ui.py", "ultrarag.py", "verify.py"}
)

# The modules no core module may import: the tool surface, its projector, the
# agent-facing instructions, and the two entry points. `launcher` and
# `transport` are deliberately absent — the core reads launcher state for
# `status`, and both are shared plumbing rather than the tool surface.
SURFACE_MODULES = frozenset({"server", "tool_views", "instructions", "ui", "verify"})

# `python -m research_ultra_rag_mcp` exists to launch the server, so this module
# is an entry point rather than core and is expected to import the surface.
ENTRY_MODULES = frozenset({"__main__.py"})


def _modules() -> list[Path]:
    return sorted(path for path in PACKAGE.glob("*.py") if path.name != "__init__.py")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _absolute_imports(path: Path) -> set[str]:
    """Return every absolute top-level module name this file imports."""

    names: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _package_imports(path: Path) -> set[str]:
    """Return the sibling modules this file imports, `from .x import y` alike."""

    names: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_only_the_surface_imports_mcp_machinery() -> None:
    """A core module that reaches for MCP machinery has changed the architecture."""

    importers = {
        path.name for path in _modules() if {"fastmcp", "mcp"} & _absolute_imports(path)
    }
    assert importers == set(MCP_MODULES), (
        "modules importing MCP machinery changed; the surface set is "
        f"{sorted(MCP_MODULES)}"
    )


def test_no_core_module_imports_the_surface() -> None:
    """The arrow points one way: the surface and the entry points use the core."""

    offenders = {
        path.name: sorted(_package_imports(path) & SURFACE_MODULES)
        for path in _modules()
        if path.name not in MCP_MODULES
        and path.name not in ENTRY_MODULES
        and _package_imports(path) & SURFACE_MODULES
    }
    assert offenders == {}, f"core modules importing the surface: {offenders}"

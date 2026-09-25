"""The core command line: initialising a project, and routing to the service.

These tests never start the vanilla gateway. That is part of what is under test:
the routing tests hand `_operate` a recording stand-in, the laziness tests make
the transport refuse to be built, and the tests that resolve a config touch only
settings, files, the launcher, and local state.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Self

import pytest

import research_ultra_rag_mcp.cli as cli_module
from research_ultra_rag_mcp.cli import (
    _init,
    _LazyGateway,
    _metadata_body,
    _operate,
    _parser,
    _resolve,
    _run,
    _ui,
)
from research_ultra_rag_mcp.config import ConfigurationError
from research_ultra_rag_mcp.support import ResearchError


def _args(*arguments: str) -> Any:
    return _parser().parse_args(list(arguments))


def _descriptor(project: Path) -> dict[str, Any]:
    path = project / ".research-rag" / "project.json"
    return json.loads(path.read_text(encoding="utf-8"))


class RecordingService:
    """A stand-in that records the one call a command makes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"operation": name, "arguments": arguments}

    async def status(self) -> dict[str, Any]:
        return self._record("status", {})

    async def ingest(self, *, force_recompute: bool = False) -> dict[str, Any]:
        return self._record("ingest", {"force_recompute": force_recompute})

    async def search(self, query: str, **options: Any) -> dict[str, Any]:
        return self._record("search", {"query": query, **options})

    async def list_sources(self) -> dict[str, Any]:
        return self._record("list_sources", {})

    async def get_passage(
        self, chunk_id: str, *, context_chunks: int = 1
    ) -> dict[str, Any]:
        return self._record(
            "get_passage",
            {"chunk_id": chunk_id, "context_chunks": context_chunks},
        )

    async def set_source_inclusion(
        self,
        source_path: str | None = None,
        *,
        source_id: str | None = None,
        included: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self._record(
            "set_source_inclusion",
            {
                "source_path": source_path,
                "source_id": source_id,
                "included": included,
                "reason": reason,
            },
        )

    async def set_source_metadata(
        self,
        metadata: dict[str, Any],
        *,
        source_path: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        return self._record(
            "set_source_metadata",
            {"metadata": metadata, "source_path": source_path, "source_id": source_id},
        )


def test_init_creates_a_project_and_records_the_name_it_was_given(
    tmp_path: Path,
) -> None:
    project = tmp_path / "fresh"

    report = _init(_args("--project-root", str(project), "init", "--name", "My Thesis"))

    assert report["project_name"] == "My Thesis"
    assert report["created"] == ["project_root", "source_root"]
    assert Path(report["source_root"]).is_dir()
    assert _descriptor(project) == {
        "name": "My Thesis",
        "project_id": report["project_id"],
        "schema_version": 1,
        "source_directory": "sources",
    }
    assert (project / ".research-rag" / "bin" / "open-ui.sh").is_file()
    assert Path(report["launcher"]["link_path"]).is_symlink()


def test_init_defaults_the_name_to_the_directory(tmp_path: Path) -> None:
    project = tmp_path / "plain-name"

    report = _init(_args("--project-root", str(project), "init"))

    assert report["project_name"] == "plain-name"
    assert _descriptor(project)["name"] == "plain-name"


def test_init_attaches_to_an_existing_directory_without_touching_its_files(
    tmp_path: Path,
) -> None:
    project = tmp_path / "existing"
    (project / "papers").mkdir(parents=True)
    (project / "papers" / "already-here.pdf").write_bytes(b"%PDF-1.4\n")
    (project / "README.md").write_text("my own work\n", encoding="utf-8")

    report = _init(_args("--project-root", str(project), "init", "--sources", "papers"))

    assert (project / "README.md").read_text(encoding="utf-8") == "my own work\n"
    assert (project / "papers" / "already-here.pdf").read_bytes() == b"%PDF-1.4\n"
    assert report["created"] == []
    assert report["source_root"] == str(project / "papers")
    assert _descriptor(project)["source_directory"] == "papers"


def test_init_is_idempotent_and_only_a_named_init_renames(tmp_path: Path) -> None:
    project = tmp_path / "stable"
    first = _init(_args("--project-root", str(project), "init", "--name", "First"))
    again = _init(_args("--project-root", str(project), "init"))
    renamed = _init(_args("--project-root", str(project), "init", "--name", "Second"))

    assert again["created"] == []
    assert again["project_name"] == "First"
    assert renamed["project_name"] == "Second"
    # A name is a label, so changing it must not move the identity every
    # generation is recorded against.
    assert first["project_id"] == again["project_id"] == renamed["project_id"]
    assert _descriptor(project)["name"] == "Second"


def test_init_refuses_to_move_an_initialized_project_to_another_source_directory(
    tmp_path: Path,
) -> None:
    project = tmp_path / "fixed-sources"
    _init(_args("--project-root", str(project), "init"))

    with pytest.raises(ConfigurationError, match="source directory differs"):
        _init(_args("--project-root", str(project), "init", "--sources", "docs"))


def test_init_refuses_a_name_that_cannot_be_recorded(tmp_path: Path) -> None:
    project = tmp_path / "bad-name"

    with pytest.raises(ConfigurationError, match="cannot be empty"):
        _init(_args("--project-root", str(project), "init", "--name", "   "))


def test_the_command_line_routes_each_command_to_its_service_operation() -> None:
    service = RecordingService()

    status = asyncio.run(_operate(_args("status"), service))
    assert status == ("status", {"operation": "status", "arguments": {}})

    refresh = asyncio.run(_operate(_args("ingest", "--force-recompute"), service))
    assert refresh[0] == "ingest"
    assert refresh[1]["arguments"] == {"force_recompute": True}

    sources = asyncio.run(_operate(_args("sources"), service))
    assert sources[0] == "list_sources"

    passage = asyncio.run(
        _operate(_args("passage", "abc123", "--context-chunks", "2"), service)
    )
    assert passage[1]["arguments"] == {"chunk_id": "abc123", "context_chunks": 2}


def test_search_carries_its_filters_and_reranks_by_default() -> None:
    service = RecordingService()

    tool, payload = asyncio.run(
        _operate(
            _args(
                "search",
                "articulation",
                "--top-k",
                "12",
                "--category",
                "theory",
                "--exclude-source-id",
                "sid-1",
            ),
            service,
        )
    )

    assert tool == "search"
    assert payload["arguments"] == {
        "query": "articulation",
        "top_k": 12,
        "categories_any": ["theory"],
        "projects_any": None,
        "keywords": None,
        "source_ids": None,
        "exclude_source_ids": ["sid-1"],
        "retrieval_method": "hybrid",
        "rerank": True,
    }


def test_excluding_a_source_records_the_reason_it_was_given() -> None:
    service = RecordingService()

    tool, payload = asyncio.run(
        _operate(
            _args("exclude", "a.pdf", "--reason", "superseded by the reprint"),
            service,
        )
    )

    assert tool == "set_source_inclusion"
    assert payload["arguments"] == {
        "source_path": "a.pdf",
        "source_id": None,
        "included": False,
        "reason": "superseded by the reprint",
    }


def test_metadata_clear_cannot_be_combined_with_a_field() -> None:
    with pytest.raises(ResearchError, match="cannot be combined"):
        _metadata_body(_args("metadata", "--clear", "--title", "X"))

    assert _metadata_body(_args("metadata", "--clear")) == {}
    assert _metadata_body(
        _args("metadata", "--title", "T", "--author", "A", "--author", "B")
    ) == {"title": "T", "authors": ["A", "B"]}


class _FakeGatewayClient:
    """A stand-in for the vanilla stdio client, counting how often it is opened."""

    opened: ClassVar[int] = 0
    calls: ClassVar[list[str]] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        type(self).opened += 1

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def call_tool(self, name: str, *_args: Any, **_kwargs: Any) -> dict[str, str]:
        type(self).calls.append(name)
        return {"tool": name}


def test_the_gateway_is_opened_by_the_first_call_and_then_reused(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeGatewayClient.opened = 0
    _FakeGatewayClient.calls = []
    monkeypatch.setattr(cli_module, "Client", _FakeGatewayClient)
    monkeypatch.setattr(
        cli_module, "create_vanilla_transport", lambda _config: object()
    )
    gateway = _LazyGateway(_resolve(_args("--project-root", str(project), "status")))

    async def scenario() -> None:
        first = await gateway.call_tool("retriever_retriever_init", {})
        second = await gateway.call_tool("retriever_retriever_search", {})
        await gateway.aclose()

        assert first == {"tool": "retriever_retriever_init"}
        assert second == {"tool": "retriever_retriever_search"}

    asyncio.run(scenario())

    assert _FakeGatewayClient.opened == 1
    assert _FakeGatewayClient.calls == [
        "retriever_retriever_init",
        "retriever_retriever_search",
    ]


def test_a_reading_command_never_opens_the_gateway(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_config: Any) -> Any:
        raise AssertionError("a command that only reads opened the vanilla gateway")

    monkeypatch.setattr(cli_module, "create_vanilla_transport", refuse)

    result = asyncio.run(_run(_args("--project-root", str(project), "sources")))

    assert result is not None
    assert result["source_count"] == 0


def test_set_overrides_reach_the_settings_by_resolving_in_this_process(
    project: Path,
) -> None:
    args = _args(
        "--project-root", str(project), "--set", "retrieval.rrf_k=30", "config"
    )

    config = _resolve(args)

    assert config.settings.rrf_k == 30
    assert config.settings_provenance["retrieval.rrf_k"] == "command line"


def test_ui_hands_over_to_the_project_launcher(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[list[str]] = []

    def record(command: list[str]) -> int:
        recorded.append(command)
        return 0

    monkeypatch.setattr(cli_module, "subprocess", SimpleNamespace(call=record))
    args = _args("--project-root", str(project), "ui", "--open", "--port", "5099")
    config = _resolve(args)

    _ui(args, config)

    assert len(recorded) == 1
    assert recorded[0][0].endswith(".research-rag/bin/open-ui.sh")
    assert recorded[0][1:] == ["--open", "--port", "5099"]


def test_ui_reports_a_launcher_that_fails(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "subprocess", SimpleNamespace(call=lambda _: 3))
    args = _args("--project-root", str(project), "ui")
    config = _resolve(args)

    with pytest.raises(ResearchError, match="exited with status 3"):
        _ui(args, config)

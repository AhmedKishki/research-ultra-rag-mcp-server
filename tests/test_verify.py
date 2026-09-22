from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Self

import pytest

import research_ultra_rag_mcp.verify as verify_module
from research_ultra_rag_mcp.verify import _ingest_until_complete, _parser, _verify


class BatchedClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], int]] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: int,
    ) -> SimpleNamespace:
        self.calls.append((name, arguments, timeout))
        status = "in_progress" if len(self.calls) < 3 else "ready"
        return SimpleNamespace(data={"status": status, "call": len(self.calls)})


def test_verifier_repeats_checkpointed_ingestion_until_terminal() -> None:
    client = BatchedClient()
    arguments = {"chunk_size": 50, "chunk_overlap": 10, "force_recompute": False}

    result = asyncio.run(_ingest_until_complete(client, arguments))

    assert result == {"status": "ready", "call": 3}
    assert client.calls == [("ingest", arguments, 1800)] * 3


class VerificationClient:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        type(self).calls = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def list_tools(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name) for name in ("status", "ingest", "search")]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        del timeout
        type(self).calls.append((name, arguments))
        if name == "status":
            return SimpleNamespace(data={"ready": True})
        return SimpleNamespace(data=dict(arguments))


def test_verifier_forwards_the_agent_search_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verify_module, "Client", VerificationClient)
    monkeypatch.setattr(
        verify_module,
        "resolve_config",
        lambda *_args, **_kwargs: SimpleNamespace(logs_root=tmp_path),
    )
    monkeypatch.setattr(
        verify_module,
        "create_research_transport",
        lambda *_args, **_kwargs: object(),
    )
    args = _parser().parse_args(
        [
            str(tmp_path),
            "--query",
            "grouped evidence",
            "--top-k",
            "7",
        ]
    )

    result = asyncio.run(_verify(args))

    assert result["search"] == {"query": "grouped evidence", "top_k": 7}
    assert VerificationClient.calls[-1] == ("search", result["search"])

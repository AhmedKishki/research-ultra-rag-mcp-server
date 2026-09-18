from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from research_ultra_rag_mcp.verify import _ingest_until_complete


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

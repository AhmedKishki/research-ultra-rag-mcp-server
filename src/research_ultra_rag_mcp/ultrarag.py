"""Persistent MCP connection to the pinned vanilla UltraRAG gateway."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from .config import ResearchConfig, child_process_environment


def create_vanilla_transport(config: ResearchConfig) -> StdioTransport:
    """Start the pinned gateway as a managed child of this server.

    The gateway inherits no top-level UI setting, so a variable exported for this
    server's own UI cannot travel down the process tree.
    """

    arguments = [
        "--workspace-root",
        str(config.ultrarag_workspace),
        "--log-level",
        config.log_level,
        "--namespace",
        "corpus",
        "--namespace",
        "retriever",
    ]
    if config.runtime_cache_root is not None:
        arguments.extend(["--runtime-cache-root", str(config.runtime_cache_root)])
    if config.offline:
        arguments.append("--offline")

    return StdioTransport(
        command=str(config.vanilla_executable),
        args=arguments,
        env=child_process_environment(),
        cwd=str(config.project_root),
        keep_alive=True,
        log_file=config.logs_root / "vanilla-gateway-stderr.log",
    )


class VanillaUltraRAG:
    """Small typed boundary around vanilla MCP tool calls."""

    def __init__(self, client: Client[Any]) -> None:
        self.client = client

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        result = await self.client.call_tool(
            name,
            arguments,
            timeout=1800,
            raise_on_error=True,
        )
        return result.data

    async def chunk(
        self,
        input_path: Path,
        output_path: Path,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        await self.call(
            "corpus_chunk_documents",
            {
                "raw_chunk_path": str(input_path),
                "chunk_backend_configs": {"token": {"chunk_overlap": chunk_overlap}},
                "chunk_backend": "token",
                "tokenizer_or_token_counter": "gpt2",
                "chunk_size": chunk_size,
                "chunk_path": str(output_path),
                "use_title": False,
            },
        )

    async def initialize_bm25(
        self,
        chunks_path: Path,
        index_path: Path,
        *,
        language: str = "en",
    ) -> None:
        await self.call(
            "retriever_retriever_init",
            {
                "model_name_or_path": "",
                "backend_configs": {
                    "bm25": {
                        "lang": language,
                        "tokenizer": "default",
                        "save_path": str(index_path),
                    }
                },
                "batch_size": 32,
                "corpus_path": str(chunks_path),
                "gpu_ids": None,
                "is_multimodal": False,
                "backend": "bm25",
                # Present in the unified upstream schema but unused by BM25.
                "index_backend": "faiss",
                "index_backend_configs": {},
                "is_demo": False,
                "collection_name": "",
            },
        )

    async def build_bm25(
        self,
        chunks_path: Path,
        index_path: Path,
        *,
        language: str = "en",
    ) -> None:
        await self.initialize_bm25(chunks_path, index_path, language=language)
        await self.call("retriever_bm25_index", {"overwrite": False})
        # UltraRAG 0.3.0.2 must reload a newly built index to attach passages.
        # The reload names the same stopword language as the build: load_stopwords
        # corrects the list from the saved file, but the language is what the
        # retriever records, so omitting it records a corpus as English when it
        # was built as German.
        await self.initialize_bm25(chunks_path, index_path, language=language)

    async def search_bm25(self, query: str, top_k: int) -> list[str]:
        payload = await self.call(
            "retriever_bm25_search",
            {"query_list": [query], "top_k": top_k},
        )
        if not isinstance(payload, dict):
            raise TypeError("UltraRAG BM25 returned a non-object result")
        batches = payload.get("ret_psg")
        if (
            not isinstance(batches, list)
            or not batches
            or not isinstance(batches[0], list)
        ):
            raise RuntimeError("UltraRAG BM25 returned an invalid passage result")
        return [str(item) for item in batches[0]]

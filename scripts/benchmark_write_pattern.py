"""Measure the per-unit write pattern on a real project build.

This is the harness behind the numbers in ``PLAN.md`` §10.5. It builds the same
corpus twice on the device you point ``--root`` at, using the real vanilla
gateway, real tokenizer chunking, and real embeddings, and changes nothing but
the write pattern:

* ``paired`` formats every atomic write durably, one directory fsync per file,
  and makes the handoff file durable too — the pattern before Step 2a.
* ``grouped`` is the current code: a unit's artifacts defer their directory
  fsync and the group is committed once, before the checkpoint that claims the
  unit is complete.

The second run of any pair benefits from a warm page cache, so the default
``--order both`` runs both orders and prints both tables. Read the difference,
not the absolute numbers.

Point ``--root`` at the device under test:

    uv run python scripts/benchmark_write_pattern.py --root /mnt/data/rr-write-bench

The first run downloads the pinned models if they are not cached yet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from fastmcp import Client

import research_ultra_rag_mcp.service as service_module
from research_ultra_rag_mcp.config import resolve_config
from research_ultra_rag_mcp.service import ResearchService
from research_ultra_rag_mcp.ultrarag import VanillaUltraRAG, create_vanilla_transport

WORDS = (
    "cobalt",
    "amber",
    "copper",
    "lithium",
    "silicon",
    "nickel",
    "graphite",
    "quartz",
    "manganese",
    "tungsten",
)


@dataclass(frozen=True, slots=True)
class RunResult:
    wall_seconds: float
    chunk_count: int
    unit_count: int
    phase_seconds: dict[str, float]


def write_pdf(path: Path, pages: list[str], *, title: str) -> None:
    document = pymupdf.open()
    document.set_metadata({"title": title, "author": "Benchmark"})
    for text in pages:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 760), text, fontsize=11)
    document.save(path)
    document.close()


def build_corpus(project: Path, *, sources: int, pages: int) -> None:
    root = project / "sources"
    root.mkdir(parents=True, exist_ok=True)
    for index in range(sources):
        word = WORDS[index % len(WORDS)]
        write_pdf(
            root / f"record-{index:02d}-{word}.pdf",
            [
                f"The {word} record {page} discusses labour, ecology, and value "
                f"across supply chains and the amber marsh."
                for page in range(pages)
            ],
            title=f"Record {index}",
        )


class durable_writes:
    """Restore the pre-Step-2a write pattern for the duration of a run."""

    def __enter__(self) -> None:
        names = (
            "atomic_write_json",
            "atomic_write_jsonl",
            "write_handoff_jsonl",
            "fsync_directories",
        )
        self._originals: dict[str, Any] = {
            name: getattr(service_module, name) for name in names
        }
        real_json = self._originals["atomic_write_json"]
        real_jsonl = self._originals["atomic_write_jsonl"]

        def durable_json(path: Path, value: Any, **kwargs: Any) -> None:
            real_json(path, value, fsync_parent=True)

        def durable_jsonl(path: Path, records: Any, **kwargs: Any) -> None:
            real_jsonl(path, records, fsync_parent=True)

        service_module.atomic_write_json = durable_json
        service_module.atomic_write_jsonl = durable_jsonl
        service_module.write_handoff_jsonl = durable_jsonl
        service_module.fsync_directories = lambda paths: None

    def __exit__(self, *exc_info: object) -> None:
        for name, original in self._originals.items():
            setattr(service_module, name, original)


async def run_once(
    project: Path,
    *,
    paired: bool,
    sources: int,
    pages: int,
    chunk_size: int,
    chunk_overlap: int,
) -> RunResult:
    shutil.rmtree(project, ignore_errors=True)
    project.mkdir(parents=True)
    build_corpus(project, sources=sources, pages=pages)
    config = resolve_config(project)

    started = time.perf_counter()
    with durable_writes() if paired else nullcontext():
        async with Client(
            create_vanilla_transport(config),
            name="write-pattern-benchmark",
            timeout=1800,
            init_timeout=1800,
        ) as client:
            service = ResearchService(config, VanillaUltraRAG(client))
            result = await service.ingest(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            while result.get("status") == "in_progress":
                result = await service.ingest(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
    wall = time.perf_counter() - started

    manifest = json.loads(
        (Path(result["generation_root"]) / "manifest.json").read_text(encoding="utf-8")
    )
    return RunResult(
        wall_seconds=wall,
        chunk_count=int(manifest["chunk_count"]),
        unit_count=int(manifest["extraction_unit_count"]),
        phase_seconds={
            str(key): float(value)
            for key, value in manifest["build_metrics"]["phase_timings_seconds"].items()
        },
    )


def report(before: RunResult, after: RunResult) -> None:
    assert before.chunk_count == after.chunk_count, (before, after)
    print(f"corpus: {before.unit_count} units, {before.chunk_count} chunks")
    print(f"{'phase':<22}{'before':>10}{'after':>10}{'saved':>10}")
    for phase in sorted(set(before.phase_seconds) | set(after.phase_seconds)):
        left = before.phase_seconds.get(phase, 0.0)
        right = after.phase_seconds.get(phase, 0.0)
        print(f"{phase:<22}{left:>10.2f}{right:>10.2f}{left - right:>10.2f}")
    left_sum = sum(before.phase_seconds.values())
    right_sum = sum(after.phase_seconds.values())
    print(
        f"{'phase sum':<22}{left_sum:>10.2f}{right_sum:>10.2f}"
        f"{left_sum - right_sum:>10.2f}"
    )
    print(
        f"{'wall clock':<22}{before.wall_seconds:>10.2f}"
        f"{after.wall_seconds:>10.2f}"
        f"{before.wall_seconds - after.wall_seconds:>10.2f}"
    )


async def compare(
    root: Path,
    *,
    label: str,
    reverse: bool,
    sources: int,
    pages: int,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[RunResult, RunResult]:
    async def run(name: str, paired: bool) -> RunResult:
        return await run_once(
            root / name,
            paired=paired,
            sources=sources,
            pages=pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    order = (
        (("grouped", False), ("paired", True))
        if reverse
        else (("paired", True), ("grouped", False))
    )
    first = await run(f"{label}-{order[0][0]}", order[0][1])
    second = await run(f"{label}-{order[1][0]}", order[1][1])
    if reverse:
        return second, first
    return first, second


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B the per-unit write pattern on a real build."
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Scratch directory on the device under test; it is replaced.",
    )
    parser.add_argument(
        "--order",
        choices=("both", "pair-then-group", "group-then-pair"),
        default="both",
        help="The second run of a pair sees a warm cache, so both orders matter.",
    )
    parser.add_argument("--sources", type=int, default=8)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    arguments = parser.parse_args()

    root = arguments.root.expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    settings = {
        "sources": arguments.sources,
        "pages": arguments.pages,
        "chunk_size": arguments.chunk_size,
        "chunk_overlap": arguments.chunk_overlap,
    }
    if arguments.order in {"both", "pair-then-group"}:
        print("=== paired then grouped ===")
        report(*await compare(root, label="forward", reverse=False, **settings))
    if arguments.order in {"both", "group-then-pair"}:
        print("=== grouped then paired ===")
        report(*await compare(root, label="reverse", reverse=True, **settings))

    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())

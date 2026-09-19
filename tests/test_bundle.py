from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import stat
import sys
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import write_pdf
from test_service import FakeDenseBackend, FakeUltraRAG

from research_ultra_rag_mcp.config import resolve_config
from research_ultra_rag_mcp.service import ResearchError, ResearchService
from research_ultra_rag_mcp.sources import stable_source_id
from research_ultra_rag_mcp.storage import read_jsonl


def _service(project: Path) -> ResearchService:
    config = resolve_config(project, vanilla_executable=sys.executable)
    return ResearchService(  # type: ignore[arg-type]
        config,
        FakeUltraRAG(),
        dense=FakeDenseBackend(),
    )


def _clone_project_identity(source: Path, target: Path) -> None:
    target.mkdir()
    portable = target / ".research-rag"
    portable.mkdir()
    shutil.copy2(source / ".research-rag" / "project.json", portable / "project.json")


def _copy_bundle(bundle: Path, target: Path) -> Path:
    bundles = target / ".research-rag" / "bundles"
    bundles.mkdir(exist_ok=True)
    copied = bundles / bundle.name
    shutil.copy2(bundle, copied)
    sidecar = bundle.with_suffix(bundle.suffix + ".sha256")
    if sidecar.exists():
        shutil.copy2(sidecar, copied.with_suffix(copied.suffix + ".sha256"))
    return copied


async def _create_export(project: Path) -> tuple[ResearchService, dict[str, object]]:
    write_pdf(
        project / "sources" / "article.pdf",
        ["Cobalt labour evidence for a portable research collection."],
        title="Portable Evidence",
    )
    write_pdf(
        project / "sources" / "excluded.pdf",
        ["An intentionally excluded duplicate representation."],
        title="Excluded Original",
    )
    service = _service(project)
    await service.set_source_inclusion(
        "excluded.pdf",
        included=False,
        reason="Reviewed duplicate retained in the portable source collection.",
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)
    first = await service.export_bundle()
    second = await service.export_bundle()
    assert first["sha256"] == second["sha256"]
    assert first["source_count"] == 2
    return service, first


async def _assert_bundle_round_trip(project: Path, tmp_path: Path) -> None:
    source_service, exported = await _create_export(project)
    bundle = Path(str(exported["bundle_path"]))
    with zipfile.ZipFile(bundle) as archive:
        assert "sources/article.pdf" in archive.namelist()
        assert "sources/excluded.pdf" in archive.namelist()
        assert "generation/portable/embeddings.npy" in archive.namelist()
        assert "generation/corpus/extracted-units.jsonl" in archive.namelist()
        assert "generation/chunks/chunks.jsonl" in archive.namelist()
        assert not any("raw-extraction" in name for name in archive.namelist())
        assert not any("ultrarag-chunks" in name for name in archive.namelist())
        assert not any(
            "qdrant" in name or "bm25" in name for name in archive.namelist()
        )
        assert not any("artifact-lookup" in name for name in archive.namelist())
        descriptor = json.loads(archive.read("bundle.json"))
        manifest = json.loads(archive.read("generation/manifest.json"))
        project_id = manifest["project_id"]
        expected_source_ids = {
            item["path"]: stable_source_id(project_id, item["path"])
            for item in descriptor["sources"]
        }
        assert {
            item["path"]: item["source_id"] for item in descriptor["sources"]
        } == expected_source_ids
        assert {
            item["source_relative_path"]: item["source_id"]
            for item in manifest["source_files"]
        } == expected_source_ids
        assert all("contents" in item for item in _read_jsonl_from_archive(archive))

    before = await source_service.search("cobalt labour", top_k=8)
    target = tmp_path / "imported-project"
    _clone_project_identity(project, target)
    copied = _copy_bundle(bundle, target)
    target_service = _service(target)

    imported = await target_service.import_bundle(copied.name)
    assert imported["status"] == "imported"
    assert imported["activated"] is True
    assert (
        Path(str(imported["generation_root"])) / "indexes" / "artifact-lookup.sqlite3"
    ).is_file()
    assert (target / "sources" / "article.pdf").read_bytes() == (
        project / "sources" / "article.pdf"
    ).read_bytes()
    assert (target / "sources" / "excluded.pdf").is_file()

    after = await target_service.search("cobalt labour", top_k=8)
    assert [hit["chunk_id"] for hit in after["hits"]] == [
        hit["chunk_id"] for hit in before["hits"]
    ]
    assert [hit["rank"] for hit in after["hits"]] == [
        hit["rank"] for hit in before["hits"]
    ]
    source_status = await source_service.status()
    assert read_jsonl(
        Path(str(imported["generation_root"])) / "chunks" / "chunks.jsonl"
    ) == read_jsonl(
        Path(str(source_status["generation_root"])) / "chunks" / "chunks.jsonl"
    )
    assert (await target_service.status())["stale"] is False

    manifest_path = Path(str(imported["generation_root"])) / "manifest.json"
    altered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    altered_manifest["retrieval"]["fusion"]["bm25_weight"] = 99
    manifest_path.write_text(
        json.dumps(altered_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ResearchError, match="conflicting retrieval metadata"):
        await target_service.import_bundle(copied.name)


def test_bundle_round_trip_preserves_chunks_and_rankings(
    project: Path,
    tmp_path: Path,
) -> None:
    asyncio.run(_assert_bundle_round_trip(project, tmp_path))


async def _assert_existing_corrupt_indexes_are_not_activated(
    project: Path,
    tmp_path: Path,
) -> None:
    _source_service, exported = await _create_export(project)
    target = tmp_path / "corrupt-existing-index"
    _clone_project_identity(project, target)
    copied = _copy_bundle(Path(str(exported["bundle_path"])), target)
    service = _service(target)
    imported = await service.import_bundle(copied.name, activate=False)
    generation_root = Path(str(imported["generation_root"]))
    manifest = json.loads(
        (generation_root / "manifest.json").read_text(encoding="utf-8")
    )
    dense_metadata = generation_root / manifest["files"]["dense_index"]
    (dense_metadata / "fake-qdrant.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(ResearchError, match="indexes failed validation"):
        await service.import_bundle(copied.name, activate=True)
    assert not (target / ".research-rag" / "runtime" / "current.json").exists()


def test_existing_corrupt_indexes_are_not_activated(
    project: Path,
    tmp_path: Path,
) -> None:
    asyncio.run(_assert_existing_corrupt_indexes_are_not_activated(project, tmp_path))


def _read_jsonl_from_archive(archive: zipfile.ZipFile) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in archive.read("generation/chunks/chunks.jsonl")
        .decode("utf-8")
        .splitlines()
        if line.strip()
    ]


async def _assert_export_rehashes_excluded_sources(project: Path) -> None:
    included = project / "sources" / "article.pdf"
    excluded = project / "sources" / "excluded.pdf"
    write_pdf(included, ["Stable included evidence."])
    write_pdf(excluded, ["Excluded bytes still belong to the bundle snapshot."])
    service = _service(project)
    await service.set_source_inclusion(
        "excluded.pdf",
        included=False,
        reason="Reviewed exclusion retained as an original.",
    )
    await service.ingest(chunk_size=50, chunk_overlap=10)

    original_stat = excluded.stat()
    mutated = bytearray(excluded.read_bytes())
    mutated[-1] ^= 1
    excluded.write_bytes(mutated)
    os.utime(excluded, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert excluded.stat().st_size == original_stat.st_size
    assert excluded.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert (await service.status())["stale"] is False

    with pytest.raises(ResearchError, match="Sources changed while"):
        await service.export_bundle()


def test_bundle_export_rehashes_excluded_sources_with_unchanged_stat(
    project: Path,
) -> None:
    asyncio.run(_assert_export_rehashes_excluded_sources(project))


async def _assert_bundle_export_rejects_invalid_document_source_id(
    project: Path,
) -> None:
    write_pdf(project / "sources" / "article.pdf", ["Stable source identity."])
    service = _service(project)
    ingested = await service.ingest(chunk_size=50, chunk_overlap=10)
    manifest_path = Path(str(ingested["generation_root"])) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"][0]["source_id"] = "src_not_the_project_source"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ResearchError, match="document source ID does not match"):
        await service.export_bundle()


def test_bundle_export_rejects_invalid_document_source_id(project: Path) -> None:
    asyncio.run(_assert_bundle_export_rejects_invalid_document_source_id(project))


async def _assert_bundle_preserves_post_ingestion_metadata_overlay(
    project: Path,
    tmp_path: Path,
) -> None:
    write_pdf(
        project / "sources" / "article.pdf",
        ["Cobalt evidence survives a reviewed bibliography correction."],
        title="Unreviewed Title",
    )
    source_service = _service(project)
    ingested = await source_service.ingest(chunk_size=50, chunk_overlap=10)
    corrected = await source_service.set_source_metadata(
        "article.pdf",
        {
            "title": "Reviewed Title",
            "authors": ["Reviewed Author"],
            "year": 2026,
            "doi": "10.1000/reviewed",
            "categories": ["corrected category"],
            "keywords": ["reviewed keyword"],
        },
    )
    assert corrected["effective_immediately"] is True
    assert corrected["requires_ingest"] is False
    status = await source_service.status()
    assert status["generation_id"] == ingested["generation_id"]
    assert status["stale"] is False

    exported = await source_service.export_bundle()
    bundle = Path(str(exported["bundle_path"]))
    with zipfile.ZipFile(bundle) as archive:
        portable_metadata = json.loads(
            archive.read("project/source-metadata.json").decode("utf-8")
        )
        archived_chunks = _read_jsonl_from_archive(archive)
    assert portable_metadata["sources"]["article.pdf"]["title"] == "Reviewed Title"
    assert "title" not in archived_chunks[0]

    target = tmp_path / "metadata-overlay-import"
    _clone_project_identity(project, target)
    copied = _copy_bundle(bundle, target)
    target_service = _service(target)
    await target_service.import_bundle(copied.name)

    listed = await target_service.list_sources(categories=["corrected category"])
    assert listed["source_count"] == 1
    assert listed["sources"][0]["title"] == "Reviewed Title"
    assert listed["sources"][0]["authors"] == ["Reviewed Author"]

    for retrieval_method in ("bm25", "dense", "hybrid"):
        searched = await target_service.search(
            "cobalt evidence",
            categories=["corrected category"],
            keywords=["reviewed keyword"],
            retrieval_method=retrieval_method,
        )
        assert searched["result_count"] == 1
        assert searched["hits"][0]["title"] == "Reviewed Title"
        assert searched["hits"][0]["authors"] == ["Reviewed Author"]
        assert "Reviewed Title" in searched["hits"][0]["citation"]


def test_bundle_preserves_post_ingestion_metadata_overlay(
    project: Path,
    tmp_path: Path,
) -> None:
    asyncio.run(
        _assert_bundle_preserves_post_ingestion_metadata_overlay(project, tmp_path)
    )


async def _assert_inactive_bundle_import_replaces_live_metadata_overlay(
    project: Path,
    tmp_path: Path,
) -> None:
    source = project / "sources" / "article.pdf"
    write_pdf(
        source,
        ["Cobalt evidence shared by two local generations."],
        title="Automatic Title",
    )
    source_service = _service(project)
    await source_service.ingest(chunk_size=50, chunk_overlap=10)
    await source_service.set_source_metadata(
        "article.pdf",
        {
            "title": "Bundled Reviewed Title",
            "categories": ["bundled category"],
        },
    )
    exported = await source_service.export_bundle()

    target = tmp_path / "inactive-import-target"
    _clone_project_identity(project, target)
    target_sources = target / "sources"
    target_sources.mkdir()
    shutil.copy2(source, target_sources / source.name)
    target_service = _service(target)
    await target_service.set_source_metadata(
        "article.pdf",
        {
            "title": "Local Reviewed Title",
            "categories": ["local category"],
        },
    )
    local_generation = await target_service.ingest(chunk_size=50, chunk_overlap=10)

    copied = _copy_bundle(Path(str(exported["bundle_path"])), target)
    imported = await target_service.import_bundle(copied.name, activate=False)

    assert imported["activated"] is False
    assert imported["portable_state_replaced"] is True
    assert imported["portable_state_authoritative"] is True
    assert imported["portable_metadata_effective_immediately"] is True
    assert "metadata and exclusions now govern" in imported["message"]
    status = await target_service.status()
    assert status["generation_id"] == local_generation["generation_id"]
    local_filter = await target_service.list_sources(categories=["local category"])
    assert local_filter["source_count"] == 0
    bundled_filter = await target_service.list_sources(categories=["bundled category"])
    assert bundled_filter["source_count"] == 1
    assert bundled_filter["sources"][0]["title"] == "Bundled Reviewed Title"


def test_inactive_bundle_import_replaces_live_metadata_overlay(
    project: Path,
    tmp_path: Path,
) -> None:
    asyncio.run(
        _assert_inactive_bundle_import_replaces_live_metadata_overlay(
            project,
            tmp_path,
        )
    )


def _rewrite_archive(
    source: Path,
    destination: Path,
    *,
    replacement: tuple[str, bytes] | None = None,
    extra: tuple[str, bytes] | None = None,
) -> None:
    with (
        zipfile.ZipFile(source) as old,
        zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as new,
    ):
        for info in old.infolist():
            data = old.read(info.filename)
            if replacement and info.filename == replacement[0]:
                data = replacement[1]
            new.writestr(info, data)
        if extra:
            new.writestr(extra[0], extra[1])


def _rewrite_manifest_with_valid_checksums(
    source: Path,
    destination: Path,
    *,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    with zipfile.ZipFile(source) as old:
        infos = {info.filename: info for info in old.infolist()}
        entries = {name: old.read(name) for name in infos}
    manifest_name = "generation/manifest.json"
    manifest = json.loads(entries[manifest_name])
    if mutate is None:
        manifest["source_file_count"] = 999
    else:
        mutate(manifest)
    entries[manifest_name] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = json.loads(entries["bundle.json"])
    descriptor["files"][manifest_name] = {
        "sha256": hashlib.sha256(entries[manifest_name]).hexdigest(),
        "size": len(entries[manifest_name]),
    }
    entries["bundle.json"] = (
        json.dumps(descriptor, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as new:
        for name, data in entries.items():
            new.writestr(infos[name], data)


async def _assert_bundle_rejects_invalid_stable_source_ids(
    project: Path,
    tmp_path: Path,
    record_name: str,
) -> None:
    _service_instance, exported = await _create_export(project)
    bundle = Path(str(exported["bundle_path"]))
    target = tmp_path / f"invalid-{record_name}"
    _clone_project_identity(project, target)
    invalid = target / ".research-rag" / "bundles" / "invalid-source-id.zip"
    invalid.parent.mkdir(exist_ok=True)

    def replace_source_id(manifest: dict[str, Any]) -> None:
        manifest[record_name][0]["source_id"] = "src_not_the_project_source"

    _rewrite_manifest_with_valid_checksums(
        bundle,
        invalid,
        mutate=replace_source_id,
    )
    with pytest.raises(ResearchError, match="identity differs"):
        await _service(target).import_bundle(invalid.name)


@pytest.mark.parametrize("record_name", ["source_files", "documents"])
def test_bundle_rejects_invalid_stable_source_ids(
    project: Path,
    tmp_path: Path,
    record_name: str,
) -> None:
    asyncio.run(
        _assert_bundle_rejects_invalid_stable_source_ids(
            project,
            tmp_path,
            record_name,
        )
    )


async def _assert_bundle_validation(project: Path, tmp_path: Path) -> None:
    _service_instance, exported = await _create_export(project)
    bundle = Path(str(exported["bundle_path"]))

    checksum_target = tmp_path / "checksum-target"
    _clone_project_identity(project, checksum_target)
    corrupt = checksum_target / ".research-rag" / "bundles" / "corrupt.zip"
    corrupt.parent.mkdir(exist_ok=True)
    _rewrite_archive(
        bundle,
        corrupt,
        replacement=("generation/chunks/chunks.jsonl", b"{}\n"),
    )
    with pytest.raises(ResearchError, match="checksum verification failed"):
        await _service(checksum_target).import_bundle(corrupt.name)
    assert not (checksum_target / ".research-rag" / "runtime" / "current.json").exists()

    malformed_target = tmp_path / "malformed-target"
    _clone_project_identity(project, malformed_target)
    malformed = malformed_target / ".research-rag" / "bundles" / "malformed.zip"
    malformed.parent.mkdir(exist_ok=True)
    _rewrite_manifest_with_valid_checksums(bundle, malformed)
    with pytest.raises(ResearchError, match="source files do not match"):
        await _service(malformed_target).import_bundle(malformed.name)

    traversal_target = tmp_path / "traversal-target"
    _clone_project_identity(project, traversal_target)
    traversal = traversal_target / ".research-rag" / "bundles" / "traversal.zip"
    traversal.parent.mkdir(exist_ok=True)
    _rewrite_archive(bundle, traversal, extra=("../escape", b"unsafe"))
    with pytest.raises(ResearchError, match="escapes its archive root"):
        await _service(traversal_target).import_bundle(traversal.name)
    assert not (traversal_target / "escape").exists()

    duplicate_target = tmp_path / "duplicate-target"
    _clone_project_identity(project, duplicate_target)
    duplicate = duplicate_target / ".research-rag" / "bundles" / "duplicate.zip"
    duplicate.parent.mkdir(exist_ok=True)
    shutil.copy2(bundle, duplicate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate, "a") as archive:
            archive.writestr("bundle.json", b"{}")
    with pytest.raises(ResearchError, match="duplicate entry names"):
        await _service(duplicate_target).import_bundle(duplicate.name)

    symlink_target = tmp_path / "symlink-target"
    _clone_project_identity(project, symlink_target)
    symlink = symlink_target / ".research-rag" / "bundles" / "symlink.zip"
    symlink.parent.mkdir(exist_ok=True)
    shutil.copy2(bundle, symlink)
    link_info = zipfile.ZipInfo("unsafe-link")
    link_info.create_system = 3
    link_info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink, "a") as archive:
        archive.writestr(link_info, "../escape")
    with pytest.raises(ResearchError, match="symbolic link"):
        await _service(symlink_target).import_bundle(symlink.name)

    conflict_target = tmp_path / "conflict-target"
    _clone_project_identity(project, conflict_target)
    (conflict_target / "sources").mkdir()
    (conflict_target / "sources" / "article.pdf").write_bytes(b"different")
    copied = _copy_bundle(bundle, conflict_target)
    with pytest.raises(ResearchError, match="conflicts with bundled bytes"):
        await _service(conflict_target).import_bundle(copied.name)
    assert (conflict_target / "sources" / "article.pdf").read_bytes() == b"different"

    isolated = tmp_path / "other-project"
    (isolated / "sources").mkdir(parents=True)
    other_service = _service(isolated)
    other_bundle = _copy_bundle(bundle, isolated)
    with pytest.raises(ResearchError, match="project_id does not match"):
        await other_service.import_bundle(other_bundle.name)


def test_bundle_rejects_corruption_traversal_conflicts_and_other_projects(
    project: Path,
    tmp_path: Path,
) -> None:
    asyncio.run(_assert_bundle_validation(project, tmp_path))

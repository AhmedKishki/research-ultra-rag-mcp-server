from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import stat
import sys
import warnings
import zipfile
from pathlib import Path

import pytest
from conftest import write_pdf
from test_service import FakeDenseBackend, FakeUltraRAG

from research_ultra_rag_mcp.config import resolve_config
from research_ultra_rag_mcp.service import ResearchError, ResearchService
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

    before = await source_service.search("cobalt labour", top_k=8)
    target = tmp_path / "imported-project"
    _clone_project_identity(project, target)
    copied = _copy_bundle(bundle, target)
    target_service = _service(target)

    imported = await target_service.import_bundle(copied.name)
    assert imported["status"] == "imported"
    assert imported["activated"] is True
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
) -> None:
    with zipfile.ZipFile(source) as old:
        infos = {info.filename: info for info in old.infolist()}
        entries = {name: old.read(name) for name in infos}
    manifest_name = "generation/manifest.json"
    manifest = json.loads(entries[manifest_name])
    manifest["source_file_count"] = 999
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

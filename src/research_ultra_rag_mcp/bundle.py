"""Validated, platform-neutral Research RAG generation bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from .config import ResearchConfig
from .sources import (
    ALLOWED_SOURCE_EXTENSIONS,
    SourceScan,
    normalize_metadata,
    sha256_file,
    stable_source_id,
)
from .storage import (
    iter_jsonl,
    load_metadata_overrides,
    load_source_catalog,
    load_source_exclusions,
    write_source_catalog,
)

BUNDLE_SCHEMA_VERSION = 2
BUNDLE_SUFFIX = ".research-rag.zip"
MAX_BUNDLE_ENTRIES = 100_000
MAX_EXPANDED_BYTES = 20 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1_000
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_GENERATION_EXPORT_FILES = (
    "corpus/extracted-units.jsonl",
    "chunks/chunks.jsonl",
    "portable/embeddings.npy",
    "manifest.json",
)
_EXPECTED_GENERATION_PATHS = {
    "extracted_units": "corpus/extracted-units.jsonl",
    "chunks": "chunks/chunks.jsonl",
    "bm25_index": "indexes/bm25",
    "dense_index": "indexes/qdrant",
    "portable_embeddings": "portable/embeddings.npy",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BundleError(RuntimeError):
    """Raised when a bundle cannot be exported or safely imported."""


@dataclass(frozen=True, slots=True)
class StagedBundle:
    root: Path
    descriptor: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def generation_root(self) -> Path:
        return self.root / "generation"

    @property
    def sources_root(self) -> Path:
        return self.root / "sources"


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _safe_name(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in value.strip()
    ).strip("-")
    return normalized or "research-project"


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.flag_bits |= 0x800
    return info


def _copy_and_hash(source: BinaryIO, target: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while block := source.read(1024 * 1024):
        target.write(block)
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def _write_path(
    archive: zipfile.ZipFile,
    name: str,
    source: Path,
) -> tuple[str, int]:
    with (
        source.open("rb") as input_handle,
        archive.open(_zip_info(name), "w") as output,
    ):
        return _copy_and_hash(input_handle, output)


def _write_bytes(
    archive: zipfile.ZipFile,
    name: str,
    value: bytes,
) -> tuple[str, int]:
    digest = hashlib.sha256(value).hexdigest()
    archive.writestr(_zip_info(name), value)
    return digest, len(value)


def _portable_json(path: Path, empty_key: str) -> bytes:
    if path.is_file():
        return path.read_bytes()
    return _json_bytes({"schema_version": 1, empty_key: {}})


def export_generation_bundle(
    config: ResearchConfig,
    generation_root: Path,
    manifest: dict[str, Any],
    scan: SourceScan,
) -> dict[str, Any]:
    """Write the selected generation and all source originals to a ZIP bundle."""

    generation_id = str(manifest.get("generation_id") or "")
    if not generation_id:
        raise BundleError("Current generation has no generation_id")
    bundle_name = (
        f"{_safe_name(config.project_name)}-{_safe_name(generation_id)}{BUNDLE_SUFFIX}"
    )
    destination = config.bundles_root / bundle_name
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    checksums: dict[str, dict[str, Any]] = {}
    source_records: list[dict[str, Any]] = []

    manifest_source_files = manifest.get("source_files")
    if not isinstance(manifest_source_files, list) or any(
        not isinstance(item, dict) for item in manifest_source_files
    ):
        raise BundleError("Generation manifest has invalid source-file records")
    manifest_sources_by_path: dict[str, dict[str, Any]] = {}
    for item in manifest_source_files:
        relative = item.get("source_relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in manifest_sources_by_path
        ):
            raise BundleError("Generation manifest has invalid source-file paths")
        expected_source_id = stable_source_id(config.project_id, relative)
        if item.get("source_id") != expected_source_id:
            raise BundleError(
                f"Generation source ID does not match its project path: {relative}"
            )
        manifest_sources_by_path[relative] = item

    manifest_documents = manifest.get("documents")
    if not isinstance(manifest_documents, list) or any(
        not isinstance(item, dict) for item in manifest_documents
    ):
        raise BundleError("Generation manifest has invalid document records")
    for item in manifest_documents:
        relative = item.get("source_relative_path")
        if not isinstance(relative, str) or not relative:
            raise BundleError("Generation manifest has invalid document paths")
        if item.get("source_id") != stable_source_id(config.project_id, relative):
            raise BundleError(
                "Generation document source ID does not match its project path: "
                f"{relative}"
            )

    scanned_paths = {source.source_relative_path for source in scan.selected}
    if scanned_paths != set(manifest_sources_by_path):
        raise BundleError(
            "Sources changed while the bundle was being exported: the current "
            "source path set differs from the selected generation"
        )

    entries: list[tuple[str, Path]] = []
    for relative in _GENERATION_EXPORT_FILES:
        source = generation_root / relative
        if not source.is_file():
            raise BundleError(f"Generation is missing portable artifact: {relative}")
        entries.append((f"generation/{relative}", source))
    for source in scan.selected:
        expected_source_id = stable_source_id(
            config.project_id,
            source.source_relative_path,
        )
        if source.source_id != expected_source_id:
            raise BundleError(
                "Scanned source ID does not match its project path: "
                f"{source.source_relative_path}"
            )
        name = f"sources/{source.source_relative_path}"
        entries.append((name, source.path))
        source_records.append(
            {
                "path": source.source_relative_path,
                "source_id": source.source_id,
                "size": source.size,
                "sha256": "",
            }
        )
    portable_values = {
        "project/project.json": config.project_config_path.read_bytes(),
        "project/source-metadata.json": _portable_json(config.metadata_path, "sources"),
        "project/source-catalog.json": (
            config.source_catalog_path.read_bytes()
            if config.source_catalog_path.is_file()
            else _json_bytes(
                {
                    "schema_version": 1,
                    "project_id": config.project_id,
                    "sources": {},
                }
            )
        ),
        "project/source-exclusions.json": _portable_json(
            config.source_exclusions_path, "sources"
        ),
    }

    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for name, source in sorted(entries):
                checksum, size = _write_path(archive, name, source)
                checksums[name] = {"sha256": checksum, "size": size}
            for name, value in sorted(portable_values.items()):
                checksum, size = _write_bytes(archive, name, value)
                checksums[name] = {"sha256": checksum, "size": size}

            for source_record in source_records:
                entry = checksums[f"sources/{source_record['path']}"]
                source_record["sha256"] = entry["sha256"]
                source_record["size"] = entry["size"]
            changed = [
                item["path"]
                for item in source_records
                if item["sha256"]
                != str(manifest_sources_by_path[item["path"]].get("sha256") or "")
                or item["size"] != manifest_sources_by_path[item["path"]].get("size")
            ]
            if changed:
                raise BundleError(
                    "Sources changed while the bundle was being exported: "
                    + ", ".join(changed)
                )
            descriptor = {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "project_id": config.project_id,
                "project_name": config.project_name,
                "generation_id": generation_id,
                "generation_schema_version": manifest.get("schema_version"),
                "embedding_model": manifest.get("retrieval", {})
                .get("dense", {})
                .get("embedding_model"),
                "embedding_model_revision": manifest.get("retrieval", {})
                .get("dense", {})
                .get("embedding_model_revision"),
                "embedding_dimension": manifest.get("retrieval", {})
                .get("dense", {})
                .get("embedding_dimension"),
                "source_directory": manifest.get("source_directory"),
                "sources": sorted(source_records, key=lambda item: item["path"]),
                "files": checksums,
                "notice": (
                    "This bundle contains complete original source works. The "
                    "exporter is responsible for redistribution rights."
                ),
            }
            archive.writestr(_zip_info("bundle.json"), _json_bytes(descriptor))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    bundle_sha256 = digest.hexdigest()
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    sidecar.write_text(f"{bundle_sha256}  {destination.name}\n", encoding="ascii")
    return {
        "status": "exported",
        "bundle_name": destination.name,
        "bundle_path": str(destination),
        "sha256_path": str(sidecar),
        "sha256": bundle_sha256,
        "size_bytes": destination.stat().st_size,
        "generation_id": generation_id,
        "source_count": len(source_records),
        "contains_original_sources": True,
        "redistribution_notice": descriptor["notice"],
    }


def _validate_member_name(name: str) -> None:
    if "\\" in name:
        raise BundleError(f"Bundle entry uses a non-portable path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise BundleError(f"Bundle entry escapes its archive root: {name!r}")


def _load_descriptor(archive: zipfile.ZipFile) -> dict[str, Any]:
    try:
        info = archive.getinfo("bundle.json")
    except KeyError as exc:
        raise BundleError("Bundle has no bundle.json descriptor") from exc
    if info.file_size > 2 * 1024 * 1024:
        raise BundleError("Bundle descriptor is unexpectedly large")
    try:
        value = json.loads(archive.read(info).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError("Bundle descriptor is invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != BUNDLE_SCHEMA_VERSION
    ):
        raise BundleError("Unsupported bundle schema")
    return value


def _verify_optional_sidecar(bundle_path: Path) -> None:
    sidecar = bundle_path.with_suffix(bundle_path.suffix + ".sha256")
    if not sidecar.exists():
        return
    if not sidecar.is_file() or sidecar.is_symlink():
        raise BundleError("Bundle SHA-256 sidecar is not a regular file")
    try:
        fields = sidecar.read_text(encoding="ascii").strip().split()
    except (OSError, UnicodeDecodeError) as exc:
        raise BundleError("Bundle SHA-256 sidecar cannot be read") from exc
    if len(fields) != 2 or not _SHA256.fullmatch(fields[0]):
        raise BundleError("Bundle SHA-256 sidecar is malformed")
    if fields[1].lstrip("*") != bundle_path.name:
        raise BundleError("Bundle SHA-256 sidecar names a different archive")
    if sha256_file(bundle_path) != fields[0]:
        raise BundleError("Bundle SHA-256 sidecar verification failed")


def _load_project_descriptor(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError("Bundled project descriptor is invalid") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise BundleError("Bundled project descriptor has an unsupported schema")
    return value


def _validate_generation_manifest(
    manifest: dict[str, Any],
    descriptor: dict[str, Any],
    config: ResearchConfig,
) -> None:
    generation_id = descriptor.get("generation_id")
    if (
        not isinstance(generation_id, str)
        or not generation_id
        or "/" in generation_id
        or "\\" in generation_id
        or generation_id in {".", ".."}
    ):
        raise BundleError("Bundle generation_id is unsafe")
    if manifest.get("generation_id") != descriptor.get("generation_id"):
        raise BundleError("Bundle generation IDs do not match")
    if manifest.get("project_id") != config.project_id:
        raise BundleError("Bundled generation belongs to another project")
    if manifest.get("schema_version") != descriptor.get("generation_schema_version"):
        raise BundleError("Bundled generation schema does not match its descriptor")
    if manifest.get("source_directory") != descriptor.get("source_directory"):
        raise BundleError("Bundled source-directory settings do not match")
    if (
        manifest.get("source_directory")
        != config.source_root.relative_to(config.project_root).as_posix()
    ):
        raise BundleError("Bundle uses a different source-directory configuration")
    files = manifest.get("files")
    if (
        not isinstance(files, dict)
        or set(files) != set(_EXPECTED_GENERATION_PATHS)
        or any(
            files.get(field) != expected
            for field, expected in _EXPECTED_GENERATION_PATHS.items()
        )
    ):
        raise BundleError("Bundled generation contains unsafe or unsupported paths")
    if not isinstance(manifest.get("documents"), list) or not isinstance(
        manifest.get("source_files"), list
    ):
        raise BundleError("Bundled generation manifest has invalid source records")
    dense = manifest.get("retrieval", {}).get("dense", {})
    for field in (
        "embedding_model",
        "embedding_model_revision",
        "embedding_dimension",
    ):
        if dense.get(field) != descriptor.get(field):
            raise BundleError(
                f"Bundled generation and descriptor disagree about {field}"
            )


def stage_bundle(
    config: ResearchConfig,
    bundle_path: Path,
    *,
    generation_schema_version: int,
    extraction_policy_version: int,
    cleaning_policy_version: int,
    artifact_policy_version: int,
    embedding_model: str,
    embedding_model_revision: str,
    embedding_dimension: int,
) -> StagedBundle:
    """Validate and extract a bundle beneath a disposable project-local root."""

    if not bundle_path.is_file() or bundle_path.is_symlink():
        raise BundleError(f"Bundle is not a regular file: {bundle_path}")
    _verify_optional_sidecar(bundle_path)
    staging = config.state_root / "imports" / uuid.uuid4().hex
    staging.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            members = archive.infolist()
            if len(members) > MAX_BUNDLE_ENTRIES:
                raise BundleError("Bundle contains too many entries")
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise BundleError("Bundle contains duplicate entry names")
            expanded_size = sum(member.file_size for member in members)
            if expanded_size > MAX_EXPANDED_BYTES:
                raise BundleError("Bundle expands beyond the supported size limit")
            for member in members:
                _validate_member_name(member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise BundleError(
                        f"Bundle contains a symbolic link: {member.filename}"
                    )
                if member.is_dir() or (mode and not stat.S_ISREG(mode)):
                    raise BundleError(
                        f"Bundle contains a non-regular entry: {member.filename}"
                    )
                if member.file_size and (
                    member.compress_size == 0
                    or member.file_size / member.compress_size > MAX_COMPRESSION_RATIO
                ):
                    raise BundleError(
                        f"Bundle entry has an unsafe compression ratio: {member.filename}"
                    )

            descriptor = _load_descriptor(archive)
            if descriptor.get("project_id") != config.project_id:
                raise BundleError(
                    "Bundle project_id does not match this initialized project"
                )
            compatibility = {
                "generation_schema_version": generation_schema_version,
                "embedding_model": embedding_model,
                "embedding_model_revision": embedding_model_revision,
                "embedding_dimension": embedding_dimension,
            }
            for field, expected in compatibility.items():
                if descriptor.get(field) != expected:
                    raise BundleError(
                        f"Bundle {field} is incompatible: "
                        f"{descriptor.get(field)!r} != {expected!r}"
                    )
            declared_files = descriptor.get("files")
            if not isinstance(declared_files, dict):
                raise BundleError("Bundle descriptor has no file checksum mapping")
            required = {
                *(f"generation/{item}" for item in _GENERATION_EXPORT_FILES),
                "project/project.json",
                "project/source-metadata.json",
                "project/source-catalog.json",
                "project/source-exclusions.json",
            }
            if missing := required - set(declared_files):
                raise BundleError(
                    f"Bundle is missing required files: {sorted(missing)}"
                )
            generation_entries = {
                name for name in declared_files if name.startswith("generation/")
            }
            expected_generation_entries = {
                f"generation/{item}" for item in _GENERATION_EXPORT_FILES
            }
            if generation_entries != expected_generation_entries:
                raise BundleError("Bundle contains unsupported generation artifacts")
            if set(names) != set(declared_files) | {"bundle.json"}:
                raise BundleError("Bundle contents differ from its declared file list")

            for name, expected in declared_files.items():
                _validate_member_name(name)
                if not isinstance(expected, dict):
                    raise BundleError(f"Invalid checksum record for {name}")
                if (
                    not isinstance(expected.get("sha256"), str)
                    or not _SHA256.fullmatch(expected["sha256"])
                    or isinstance(expected.get("size"), bool)
                    or not isinstance(expected.get("size"), int)
                    or expected["size"] < 0
                ):
                    raise BundleError(f"Invalid checksum record for {name}")
                target = staging / PurePosixPath(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("wb") as output:
                    checksum, size = _copy_and_hash(source, output)
                if checksum != expected.get("sha256") or size != expected.get("size"):
                    raise BundleError(f"Bundle checksum verification failed for {name}")

        try:
            manifest = json.loads(
                (staging / "generation" / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise BundleError("Bundled generation manifest is invalid") from exc
        if not isinstance(manifest, dict):
            raise BundleError("Bundled generation manifest must be an object")
        _validate_generation_manifest(manifest, descriptor, config)
        if (
            manifest.get("extraction_policy_version") != extraction_policy_version
            or manifest.get("cleaning_policy_version") != cleaning_policy_version
            or manifest.get("artifact_policy_version") != artifact_policy_version
        ):
            raise BundleError(
                "Bundled extraction, cleaning, or artifact policy is incompatible"
            )

        project = _load_project_descriptor(staging / "project" / "project.json")
        if (
            project.get("project_id") != config.project_id
            or project.get("source_directory") != descriptor.get("source_directory")
            or project.get("name") != descriptor.get("project_name")
        ):
            raise BundleError("Bundled project descriptor does not match the bundle")
        try:
            reviewed_metadata = load_metadata_overrides(
                staging / "project" / "source-metadata.json"
            )
            for value in reviewed_metadata.values():
                normalize_metadata(value)
            source_catalog = load_source_catalog(
                staging / "project" / "source-catalog.json",
                project_id=config.project_id,
            )
            source_exclusions = load_source_exclusions(
                staging / "project" / "source-exclusions.json"
            )
        except Exception as exc:
            raise BundleError("Bundled reviewed project state is invalid") from exc
        for relative in set(reviewed_metadata) | set(source_exclusions):
            expected_source_id = stable_source_id(config.project_id, relative)
            if source_catalog.get(expected_source_id) != relative:
                raise BundleError(
                    f"Bundled source catalog omits reviewed source identity: {relative}"
                )

        chunk_count = 0
        chunk_ids: set[str] = set()
        chunk_document_sources: set[tuple[str, str]] = set()
        try:
            for item in iter_jsonl(staging / "generation" / "chunks" / "chunks.jsonl"):
                chunk_id = item.get("chunk_id")
                document_id = item.get("document_id")
                source_id = item.get("source_id")
                contents = item.get("contents")
                if not isinstance(contents, str):
                    # Earlier artifacts used ``text`` as the canonical public
                    # field. Current artifacts write ``contents`` for UltraRAG.
                    contents = item.get("text")
                if (
                    not isinstance(chunk_id, str)
                    or not chunk_id
                    or not isinstance(document_id, str)
                    or not document_id
                    or not isinstance(source_id, str)
                    or not source_id
                    or not isinstance(contents, str)
                ):
                    raise BundleError(
                        "Bundled chunks do not match the generation manifest"
                    )
                if chunk_id in chunk_ids:
                    raise BundleError("Bundled chunks contain duplicate chunk IDs")
                chunk_ids.add(chunk_id)
                chunk_document_sources.add((document_id, source_id))
                chunk_count += 1
        except BundleError:
            raise
        except Exception as exc:
            raise BundleError("Bundled chunk collection is invalid") from exc
        if chunk_count != manifest.get("chunk_count"):
            raise BundleError("Bundled chunks do not match the generation manifest")

        sources = descriptor.get("sources")
        if not isinstance(sources, list):
            raise BundleError("Bundle source manifest is invalid")
        source_paths: set[str] = set()
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("path"), str):
                raise BundleError("Bundle contains an invalid source record")
            relative = PurePosixPath(source["path"])
            _validate_member_name(relative.as_posix())
            if relative.as_posix() in source_paths:
                raise BundleError("Bundle contains duplicate source records")
            source_paths.add(relative.as_posix())
            expected_source_id = stable_source_id(
                config.project_id,
                relative.as_posix(),
            )
            if source.get("source_id") != expected_source_id:
                raise BundleError(
                    f"Bundle source ID differs from its project path: {relative}"
                )
            if source_catalog.get(expected_source_id) != relative.as_posix():
                raise BundleError(
                    f"Bundle source is missing from its source catalog: {relative}"
                )
            if relative.suffix.casefold() not in ALLOWED_SOURCE_EXTENSIONS:
                raise BundleError(f"Bundle contains an unsupported source: {relative}")
            declared_source = descriptor["files"].get(f"sources/{relative.as_posix()}")
            if (
                not isinstance(declared_source, dict)
                or source.get("sha256") != declared_source.get("sha256")
                or source.get("size") != declared_source.get("size")
            ):
                raise BundleError(f"Bundle source checksum record differs: {relative}")
            staged_source = staging / "sources" / relative
            if not staged_source.is_file():
                raise BundleError(f"Bundled source is missing: {relative}")
            destination = (config.source_root / relative).resolve()
            try:
                destination.relative_to(config.source_root)
            except ValueError as exc:
                raise BundleError(
                    f"Bundled source escapes source root: {relative}"
                ) from exc
            if destination.exists():
                if not destination.is_file() or destination.is_symlink():
                    raise BundleError(f"Source destination is unsafe: {relative}")
                digest = sha256_file(destination)
                if digest != source.get("sha256"):
                    raise BundleError(
                        f"Existing source conflicts with bundled bytes: {relative}"
                    )
        archived_sources = {
            name.removeprefix("sources/")
            for name in descriptor["files"]
            if name.startswith("sources/")
        }
        if archived_sources != source_paths:
            raise BundleError("Bundle source list differs from its archived originals")

        source_records = {str(item["path"]): item for item in sources}
        manifest_source_files = manifest["source_files"]
        manifest_source_paths = [
            str(item.get("source_relative_path") or "")
            for item in manifest_source_files
            if isinstance(item, dict)
        ]
        if (
            len(manifest_source_paths) != len(manifest_source_files)
            or len(manifest_source_paths) != len(set(manifest_source_paths))
            or set(manifest_source_paths) != source_paths
            or manifest.get("source_file_count") != len(source_paths)
            or any(
                not isinstance(item.get("included"), bool)
                for item in manifest_source_files
            )
        ):
            raise BundleError(
                "Bundled source files do not match the generation manifest"
            )
        for item in manifest_source_files:
            relative = str(item["source_relative_path"])
            source_record = source_records[relative]
            expected_source_id = stable_source_id(config.project_id, relative)
            if (
                item.get("source_id") != expected_source_id
                or source_record.get("source_id") != expected_source_id
                or item.get("sha256") != source_record.get("sha256")
                or item.get("format")
                != PurePosixPath(relative).suffix.casefold().lstrip(".")
            ):
                raise BundleError(
                    "Bundled source-file identity differs from its original"
                )

        documents = manifest["documents"]
        document_ids = [
            str(item.get("document_id") or "")
            for item in documents
            if isinstance(item, dict)
        ]
        document_paths = [
            str(item.get("source_relative_path") or "")
            for item in documents
            if isinstance(item, dict)
        ]
        included_paths = {
            str(item["source_relative_path"])
            for item in manifest_source_files
            if item.get("included") is True
        }
        if (
            len(document_ids) != len(documents)
            or not all(document_ids)
            or len(document_ids) != len(set(document_ids))
            or len(document_paths) != len(set(document_paths))
            or set(document_paths) != included_paths
            or manifest.get("document_count") != len(documents)
        ):
            raise BundleError("Bundled documents do not match included sources")
        for document in documents:
            relative = str(document["source_relative_path"])
            source_record = source_records.get(relative)
            expected_source_id = stable_source_id(config.project_id, relative)
            if (
                source_record is None
                or document.get("source_id") != expected_source_id
                or source_record.get("source_id") != expected_source_id
                or document.get("sha256") != source_record.get("sha256")
                or document.get("format")
                != PurePosixPath(relative).suffix.casefold().lstrip(".")
            ):
                raise BundleError(
                    f"Bundled document identity differs from its source: {relative}"
                )
        known_document_ids = set(document_ids)
        if any(
            document_id not in known_document_ids
            for document_id, _source_id in chunk_document_sources
        ):
            raise BundleError("Bundled chunks refer to unknown documents")
        document_source_ids = {
            (str(document["document_id"]), str(document["source_id"]))
            for document in documents
        }
        if not chunk_document_sources.issubset(document_source_ids):
            raise BundleError("Bundled chunks refer to mismatched source identities")
        return StagedBundle(staging, descriptor, manifest)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise BundleError("Bundle is not a valid ZIP archive") from exc
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def install_staged_sources(config: ResearchConfig, staged: StagedBundle) -> None:
    """Install missing, already-validated source originals without overwriting."""

    for source in staged.descriptor["sources"]:
        relative = PurePosixPath(source["path"])
        destination = config.source_root.joinpath(*relative.parts)
        if destination.exists():
            if (
                not destination.is_file()
                or destination.is_symlink()
                or sha256_file(destination) != source["sha256"]
            ):
                raise BundleError(
                    f"Existing source conflicts with bundled bytes: {relative}"
                )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.resolve().relative_to(config.source_root)
        except ValueError as exc:
            raise BundleError(f"Source destination is unsafe: {relative}") from exc
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copy2(staged.sources_root.joinpath(*relative.parts), temporary)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if (
                    not destination.is_file()
                    or destination.is_symlink()
                    or sha256_file(destination) != source["sha256"]
                ):
                    raise BundleError(
                        f"Existing source conflicts with bundled bytes: {relative}"
                    )
        finally:
            temporary.unlink(missing_ok=True)


def install_portable_state(config: ResearchConfig, staged: StagedBundle) -> None:
    """Atomically install the reviewed metadata and source decisions."""

    local_catalog = load_source_catalog(
        config.source_catalog_path,
        project_id=config.project_id,
    )
    bundled_catalog = load_source_catalog(
        staged.root / "project" / "source-catalog.json",
        project_id=config.project_id,
    )
    merged_catalog = dict(local_catalog)
    paths_to_ids = {
        relative: source_id for source_id, relative in local_catalog.items()
    }
    for source_id, relative in bundled_catalog.items():
        if merged_catalog.get(source_id) not in {None, relative} or paths_to_ids.get(
            relative
        ) not in {None, source_id}:
            raise BundleError(
                f"Bundled source catalog conflicts with local identity: {relative}"
            )
        merged_catalog[source_id] = relative
        paths_to_ids[relative] = source_id

    mappings = {
        staged.root / "project" / "source-metadata.json": config.metadata_path,
        staged.root
        / "project"
        / "source-exclusions.json": config.source_exclusions_path,
    }
    for source, destination in mappings.items():
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    write_source_catalog(
        config.source_catalog_path,
        merged_catalog,
        project_id=config.project_id,
    )

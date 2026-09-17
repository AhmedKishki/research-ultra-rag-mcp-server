"""Durable, inspectable storage helpers for research generations."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any


class StorageError(RuntimeError):
    """Raised for missing or malformed research state."""


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StorageError(f"Required state file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StorageError(f"Invalid JSON state file: {path}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise StorageError(
                        f"Invalid JSON in {path} at line {line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise StorageError(
                        f"Expected an object in {path} at line {line_number}"
                    )
                records.append(value)
    except FileNotFoundError as exc:
        raise StorageError(f"Required state file does not exist: {path}") from exc
    return records


def load_metadata_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise StorageError(f"Unsupported metadata file: {path}")
    sources = value.get("sources", {})
    if not isinstance(sources, dict) or any(
        not isinstance(key, str) or not isinstance(item, dict)
        for key, item in sources.items()
    ):
        raise StorageError(f"Invalid source metadata mapping: {path}")
    return sources


def write_metadata_overrides(
    path: Path,
    overrides: dict[str, dict[str, Any]],
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "sources": overrides,
        },
    )


def load_source_exclusions(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise StorageError(f"Unsupported source-exclusion file: {path}")
    sources = value.get("sources", {})
    if not isinstance(sources, dict):
        raise StorageError(f"Invalid source-exclusion mapping: {path}")

    normalized: dict[str, dict[str, str]] = {}
    for source_path, record in sources.items():
        if not isinstance(source_path, str) or not source_path.strip():
            raise StorageError(f"Invalid excluded source path: {path}")
        relative = PurePosixPath(source_path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != source_path
        ):
            raise StorageError(
                f"Excluded source path must be normalized and relative: {source_path!r}"
            )
        if not isinstance(record, dict):
            raise StorageError(f"Invalid exclusion record for {source_path!r}: {path}")
        unknown = set(record) - {"reason", "excluded_at"}
        if unknown:
            raise StorageError(
                f"Unsupported exclusion fields for {source_path!r}: "
                f"{', '.join(sorted(unknown))}"
            )
        reason = record.get("reason")
        excluded_at = record.get("excluded_at")
        if not isinstance(reason, str) or not reason.strip():
            raise StorageError(
                f"Exclusion for {source_path!r} requires a non-empty reason: {path}"
            )
        if not isinstance(excluded_at, str) or not excluded_at.strip():
            raise StorageError(
                f"Exclusion for {source_path!r} requires excluded_at: {path}"
            )
        normalized[source_path] = {
            "reason": reason.strip(),
            "excluded_at": excluded_at.strip(),
        }
    return dict(sorted(normalized.items()))


def write_source_exclusions(
    path: Path,
    exclusions: dict[str, dict[str, str]],
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "sources": dict(sorted(exclusions.items())),
        },
    )


def load_current_generation(state_root: Path) -> tuple[Path, dict[str, Any]]:
    pointer = read_json(state_root / "current.json")
    if not isinstance(pointer, dict) or pointer.get("schema_version") != 1:
        raise StorageError("Invalid current-generation pointer")
    generation_id = pointer.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise StorageError("Current-generation pointer has no generation_id")
    generation_root = (state_root / "generations" / generation_id).resolve()
    generations_root = (state_root / "generations").resolve()
    try:
        generation_root.relative_to(generations_root)
    except ValueError as exc:
        raise StorageError(
            "Current-generation pointer escapes generation storage"
        ) from exc
    manifest = read_json(generation_root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("generation_id") != generation_id:
        raise StorageError("Current generation manifest does not match its pointer")
    return generation_root, manifest

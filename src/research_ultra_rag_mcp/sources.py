"""PDF/EPUB source policy, discovery, hashing, and metadata validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ResearchConfig

ALLOWED_SOURCE_EXTENSIONS = frozenset({".epub", ".pdf"})
METADATA_FIELDS = frozenset(
    {"title", "authors", "year", "doi", "categories", "keywords"}
)


class SourcePolicyError(ValueError):
    """Raised when a source violates the project research policy."""


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: Path
    source_relative_path: str
    project_relative_path: str
    extension: str
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class SourceScan:
    selected: tuple[SourceFile, ...]
    ignored_extensions: dict[str, int]


def scan_sources(config: ResearchConfig) -> SourceScan:
    if not config.source_root.exists():
        return SourceScan(selected=(), ignored_extensions={})
    if not config.source_root.is_dir():
        raise SourcePolicyError(
            f"Configured sources path is not a directory: {config.source_root}"
        )

    selected: list[SourceFile] = []
    ignored: Counter[str] = Counter()
    for path in sorted(config.source_root.rglob("*")):
        if not path.is_file():
            continue
        extension = path.suffix.lower()
        if extension not in ALLOWED_SOURCE_EXTENSIONS:
            ignored[extension or "<no extension>"] += 1
            continue
        if path.is_symlink():
            raise SourcePolicyError(f"Symbolic-link sources are not allowed: {path}")
        resolved = path.resolve()
        try:
            resolved.relative_to(config.source_root)
        except ValueError as exc:
            raise SourcePolicyError(
                f"Source escapes the configured source directory: {path}"
            ) from exc
        stat = resolved.stat()
        selected.append(
            SourceFile(
                path=resolved,
                source_relative_path=resolved.relative_to(
                    config.source_root
                ).as_posix(),
                project_relative_path=resolved.relative_to(
                    config.project_root
                ).as_posix(),
                extension=extension,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    return SourceScan(
        selected=tuple(selected),
        ignored_extensions=dict(sorted(ignored.items())),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def metadata_revision(overrides: dict[str, dict[str, Any]]) -> str:
    encoded = json.dumps(
        overrides,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SourcePolicyError(f"Metadata field {field!r} must be a list of strings")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = item.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            result.append(normalized)
            seen.add(key)
    return result


def normalize_metadata(value: dict[str, Any]) -> dict[str, Any]:
    unknown = set(value) - METADATA_FIELDS
    if unknown:
        raise SourcePolicyError(
            f"Unsupported metadata fields: {', '.join(sorted(unknown))}"
        )

    normalized: dict[str, Any] = {}
    for field in ("title", "doi"):
        if field in value:
            item = value[field]
            if not isinstance(item, str):
                raise SourcePolicyError(f"Metadata field {field!r} must be a string")
            normalized[field] = item.strip()

    for field in ("authors", "categories", "keywords"):
        if field in value:
            normalized[field] = _string_list(value[field], field)

    if "year" in value:
        year = value["year"]
        if year is not None and (
            isinstance(year, bool) or not isinstance(year, int) or not 1 <= year <= 9999
        ):
            raise SourcePolicyError("Metadata field 'year' must be null or an integer")
        normalized["year"] = year
    return normalized

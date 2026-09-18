"""Configuration and project-boundary validation."""

from __future__ import annotations

import filecmp
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_path


class ConfigurationError(ValueError):
    """Raised when the server cannot establish a safe project boundary."""


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    project_root: Path
    source_root: Path
    state_root: Path
    portable_root: Path
    project_id: str
    project_name: str
    vanilla_executable: Path
    runtime_cache_root: Path | None
    model_cache_root: Path
    offline: bool
    log_level: str

    @property
    def generations_root(self) -> Path:
        return self.state_root / "generations"

    @property
    def logs_root(self) -> Path:
        return self.state_root / "logs"

    @property
    def ultrarag_workspace(self) -> Path:
        return self.state_root / "ultrarag-runtime"

    @property
    def models_root(self) -> Path:
        """Compatibility name for the shared model-binary cache."""

        return self.model_cache_root

    @property
    def legacy_models_root(self) -> Path:
        return self.state_root / "models"

    @property
    def staging_root(self) -> Path:
        return self.state_root / "staging"

    @property
    def failures_root(self) -> Path:
        return self.state_root / "failures"

    @property
    def metadata_path(self) -> Path:
        return self.portable_root / "source-metadata.json"

    @property
    def source_exclusions_path(self) -> Path:
        return self.portable_root / "source-exclusions.json"

    @property
    def project_config_path(self) -> Path:
        return self.portable_root / "project.json"

    @property
    def bundles_root(self) -> Path:
        return self.portable_root / "bundles"

    @property
    def current_path(self) -> Path:
        return self.state_root / "current.json"


def configured_source_directory(project_root: str | Path) -> str:
    """Reuse an initialized project's source setting, or return the default."""

    descriptor_path = (
        Path(project_root).expanduser().resolve() / ".research-rag" / "project.json"
    )
    if not descriptor_path.exists():
        return "sources"
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"Invalid portable project descriptor: {descriptor_path}"
        ) from exc
    source_directory = (
        descriptor.get("source_directory")
        if isinstance(descriptor, dict) and descriptor.get("schema_version") == 1
        else None
    )
    if not isinstance(source_directory, str) or not source_directory:
        raise ConfigurationError(
            f"Portable project descriptor has no source_directory: {descriptor_path}"
        )
    return source_directory


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _contains_files(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _has_payload(path: Path) -> bool:
    return path.is_dir() and any(
        item.is_file() or item.is_symlink() for item in path.rglob("*")
    )


def _remove_empty_tree(path: Path) -> None:
    for child in sorted(
        path.rglob("*"), key=lambda item: len(item.parts), reverse=True
    ):
        child.rmdir()
    path.rmdir()


def _review_state_files(
    runtime_root: Path,
    project_state_root: Path,
) -> list[tuple[Path, Path]]:
    result: list[tuple[Path, Path]] = []
    for name in ("source-metadata.json", "source-exclusions.json"):
        legacy = runtime_root / name
        destination = project_state_root / name
        if not legacy.exists():
            continue
        if not legacy.is_file() or legacy.is_symlink():
            raise ConfigurationError(f"Legacy review state is not a file: {legacy}")
        if destination.is_symlink() or (
            destination.exists()
            and (
                not destination.is_file()
                or not filecmp.cmp(legacy, destination, shallow=False)
            )
        ):
            raise ConfigurationError(
                "Conflicting reviewed state exists in both the legacy and "
                f"consolidated locations: {name}"
            )
        result.append((legacy, destination))
    return result


def _move_review_state(runtime_root: Path, project_state_root: Path) -> None:
    files = _review_state_files(runtime_root, project_state_root)
    for legacy, destination in files:
        if destination.exists():
            legacy.unlink()
        else:
            os.replace(legacy, destination)


def _migrate_legacy_runtime(
    project: Path,
    project_state_root: Path,
    runtime_root: Path,
) -> None:
    """Move the former .ultrarag/research tree beneath .research-rag once."""

    legacy_root = project / ".ultrarag" / "research"
    if legacy_root.is_symlink():
        raise ConfigurationError(
            f"Legacy research state cannot be a symlink: {legacy_root}"
        )
    if legacy_root.exists() and not legacy_root.is_dir():
        raise ConfigurationError(
            f"Legacy research state is not a directory: {legacy_root}"
        )

    if legacy_root.is_dir():
        _review_state_files(legacy_root, project_state_root)
        if runtime_root.exists() and _has_payload(runtime_root):
            if _has_payload(legacy_root):
                raise ConfigurationError(
                    "Research runtime state exists in both .ultrarag/research and "
                    ".research-rag/runtime; resolve the duplicate state before "
                    "starting the server."
                )
            _remove_empty_tree(legacy_root)
        else:
            if runtime_root.exists():
                _remove_empty_tree(runtime_root)
            os.replace(legacy_root, runtime_root)

    if runtime_root.is_dir():
        _move_review_state(runtime_root, project_state_root)

    legacy_parent = project / ".ultrarag"
    if legacy_parent.is_dir():
        try:
            legacy_parent.rmdir()
        except OSError:
            pass


def _initialize_portable_project(
    project: Path,
    portable_root: Path,
    source_directory: str,
) -> tuple[str, str]:
    """Create or validate the small, Git-friendly project descriptor."""

    portable_root.mkdir(parents=True, exist_ok=True)
    (portable_root / "bundles").mkdir(exist_ok=True)
    descriptor_path = portable_root / "project.json"
    if descriptor_path.exists():
        try:
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"Invalid portable project descriptor: {descriptor_path}"
            ) from exc
        if not isinstance(descriptor, dict) or descriptor.get("schema_version") != 1:
            raise ConfigurationError(
                f"Unsupported portable project descriptor: {descriptor_path}"
            )
        project_id = descriptor.get("project_id")
        project_name = descriptor.get("name")
        configured_sources = descriptor.get("source_directory")
        if not isinstance(project_id, str) or not project_id.strip():
            raise ConfigurationError("Portable project descriptor has no project_id")
        if not isinstance(project_name, str) or not project_name.strip():
            raise ConfigurationError("Portable project descriptor has no project name")
        if configured_sources != source_directory:
            raise ConfigurationError(
                "Configured source directory differs from .research-rag/project.json: "
                f"{source_directory!r} != {configured_sources!r}"
            )
    else:
        project_id = str(uuid.uuid4())
        project_name = project.name
        temporary = descriptor_path.with_name(
            f".{descriptor_path.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "project_id": project_id,
                    "name": project_name,
                    "source_directory": source_directory,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, descriptor_path)

    return project_id, project_name


def resolve_config(
    project_root: str | Path,
    *,
    source_directory: str = "sources",
    vanilla_executable: str | Path | None = None,
    runtime_cache_root: str | Path | None = None,
    model_cache_root: str | Path | None = None,
    offline: bool = False,
    log_level: str = "warn",
) -> ResearchConfig:
    project = Path(project_root).expanduser().resolve()
    if not project.is_dir():
        raise ConfigurationError(f"Project root is not a directory: {project}")

    source_argument = Path(source_directory)
    if source_argument.is_absolute():
        raise ConfigurationError("--source-directory must be project-relative")
    sources = (project / source_argument).resolve()
    if not _within(sources, project):
        raise ConfigurationError(
            f"Source directory escapes the project root: {source_directory}"
        )

    portable = (project / ".research-rag").resolve()
    if not _within(portable, project):
        raise ConfigurationError(f"Portable state escapes the project root: {portable}")
    state = (portable / "runtime").resolve()
    if not _within(state, portable):
        raise ConfigurationError(f"Research runtime escapes its project state: {state}")

    executable = (
        Path(
            vanilla_executable or Path(sys.executable).parent / "vanilla-ultra-rag-mcp"
        )
        .expanduser()
        .absolute()
    )
    if not executable.is_file():
        raise ConfigurationError(
            f"vanilla-ultra-rag-mcp executable was not found: {executable}"
        )

    if log_level not in {"debug", "info", "warn", "error"}:
        raise ConfigurationError(f"Unsupported log level: {log_level}")

    portable.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_runtime(project, portable, state)

    cache = (
        Path(runtime_cache_root).expanduser().resolve()
        if runtime_cache_root is not None
        else None
    )
    configured_model_cache = (
        Path(model_cache_root).expanduser().resolve()
        if model_cache_root is not None
        else user_cache_path("research-ultra-rag-mcp", appauthor=False) / "models"
    )
    legacy_model_cache = state / "models"
    if (
        offline
        and model_cache_root is None
        and not _contains_files(configured_model_cache)
        and _contains_files(legacy_model_cache)
    ):
        configured_model_cache = legacy_model_cache

    state.mkdir(parents=True, exist_ok=True)
    (state / "generations").mkdir(exist_ok=True)
    (state / "logs").mkdir(exist_ok=True)
    (state / "staging").mkdir(exist_ok=True)
    (state / "failures").mkdir(exist_ok=True)
    (state / "ultrarag-runtime").mkdir(exist_ok=True)
    configured_model_cache.mkdir(parents=True, exist_ok=True)
    normalized_source_directory = source_argument.as_posix()
    project_id, project_name = _initialize_portable_project(
        project,
        portable,
        normalized_source_directory,
    )

    return ResearchConfig(
        project_root=project,
        source_root=sources,
        state_root=state,
        portable_root=portable,
        project_id=project_id,
        project_name=project_name,
        vanilla_executable=executable,
        runtime_cache_root=cache,
        model_cache_root=configured_model_cache,
        offline=offline,
        log_level=log_level,
    )


def resolve_source_reference(config: ResearchConfig, relative_path: str) -> Path:
    reference = Path(relative_path)
    if reference.is_absolute():
        raise ConfigurationError(
            "Source paths must be relative to the sources directory"
        )
    candidate = (config.source_root / reference).resolve()
    if not _within(candidate, config.source_root):
        raise ConfigurationError(
            f"Source path escapes the sources directory: {relative_path}"
        )
    return candidate

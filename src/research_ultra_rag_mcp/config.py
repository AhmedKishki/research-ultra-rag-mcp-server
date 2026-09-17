"""Configuration and project-boundary validation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(ValueError):
    """Raised when the server cannot establish a safe project boundary."""


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    project_root: Path
    source_root: Path
    state_root: Path
    vanilla_executable: Path
    runtime_cache_root: Path | None
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
        return self.state_root / "models"

    @property
    def metadata_path(self) -> Path:
        return self.state_root / "source-metadata.json"

    @property
    def source_exclusions_path(self) -> Path:
        return self.state_root / "source-exclusions.json"

    @property
    def current_path(self) -> Path:
        return self.state_root / "current.json"


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_config(
    project_root: str | Path,
    *,
    source_directory: str = "sources",
    vanilla_executable: str | Path | None = None,
    runtime_cache_root: str | Path | None = None,
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

    state = (project / ".ultrarag" / "research").resolve()
    if not _within(state, project):
        raise ConfigurationError(f"Research state escapes the project root: {state}")

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

    cache = (
        Path(runtime_cache_root).expanduser().resolve()
        if runtime_cache_root is not None
        else None
    )

    state.mkdir(parents=True, exist_ok=True)
    (state / "generations").mkdir(exist_ok=True)
    (state / "logs").mkdir(exist_ok=True)
    (state / "models").mkdir(exist_ok=True)
    (state / "ultrarag-runtime").mkdir(exist_ok=True)

    return ResearchConfig(
        project_root=project,
        source_root=sources,
        state_root=state,
        vanilla_executable=executable,
        runtime_cache_root=cache,
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

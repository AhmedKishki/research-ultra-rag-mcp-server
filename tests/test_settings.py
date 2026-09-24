"""Unit coverage for layered settings: one registry, five precedence layers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from research_ultra_rag_mcp.config import ConfigurationError, resolve_config
from research_ultra_rag_mcp.service import retrieval_policy_fingerprint
from research_ultra_rag_mcp.settings import (
    SETTINGS,
    SettingsError,
    default_config_path,
    describe_settings,
    resolve_settings,
)

# The fingerprint every published measurement was taken with. Layering has to
# reproduce it byte for byte: a different value would tell every existing
# generation that its ranking policy changed and force a rebuild for nothing.
PUBLISHED_FINGERPRINT = (
    "b7285824c79d662c9abdba84238077abc1271d11a861c931ee0e590c3d4f380f"
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_the_packaged_default_file_supplies_every_setting(tmp_path: Path) -> None:
    settings, provenance = resolve_settings(tmp_path, environ={})

    for setting in SETTINGS:
        assert setting.key in provenance
    default_source = f"default ({default_config_path()})"
    assert provenance["chunking.size"] == default_source
    assert provenance["retrieval.rrf_k"] == default_source
    assert settings.chunk_size == 384
    assert settings.chunk_overlap == 64


def test_defaults_reproduce_the_published_fingerprint(tmp_path: Path) -> None:
    settings, _ = resolve_settings(tmp_path, environ={})

    assert retrieval_policy_fingerprint(settings) == PUBLISHED_FINGERPRINT


def test_a_layer_names_only_what_it_changes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write(
        project / ".research-rag" / "config.toml",
        "[retrieval]\nrrf_k = 25\n\n[chunking]\noverlap = 32\n",
    )

    settings, provenance = resolve_settings(project, environ={})

    assert settings.rrf_k == 25
    assert settings.chunk_overlap == 32
    # Everything else is still inherited from the packaged default.
    assert settings.chunk_size == 384
    assert provenance["retrieval.rrf_k"] != provenance["chunking.size"]


def test_precedence_runs_file_then_environment_then_command_line(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write(project / ".research-rag" / "config.toml", "[retrieval]\nrrf_k = 25\n")
    named = _write(tmp_path / "named.toml", "[retrieval]\nrrf_k = 40\n")

    from_environment, provenance = resolve_settings(
        project,
        config_path=named,
        environ={"RESEARCH_ULTRARAG_RETRIEVAL_RRF_K": "15"},
    )
    assert from_environment.rrf_k == 15
    assert provenance["retrieval.rrf_k"].startswith("environment")

    from_command_line, provenance = resolve_settings(
        project,
        config_path=named,
        overrides=["retrieval.rrf_k=5"],
        environ={"RESEARCH_ULTRARAG_RETRIEVAL_RRF_K": "15"},
    )
    assert from_command_line.rrf_k == 5
    assert provenance["retrieval.rrf_k"] == "command line"


def test_an_unknown_key_is_refused_in_a_file_and_on_the_command_line(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write(project / ".research-rag" / "config.toml", "[retrieval]\nrrf_kx = 25\n")

    with pytest.raises(SettingsError, match="unknown setting"):
        resolve_settings(project, environ={})

    with pytest.raises(SettingsError, match="unknown setting"):
        resolve_settings(tmp_path, overrides=["retrieval.rrf_kx=1"], environ={})


def test_out_of_bounds_and_cross_field_values_are_refused(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="at most 384"):
        resolve_settings(tmp_path, overrides=["chunking.size=900"], environ={})

    with pytest.raises(SettingsError, match="must be below"):
        resolve_settings(
            tmp_path,
            overrides=["chunking.size=60", "chunking.overlap=60"],
            environ={},
        )

    with pytest.raises(SettingsError, match="must be one of"):
        resolve_settings(tmp_path, overrides=["dense.backend=lancedb"], environ={})

    with pytest.raises(SettingsError, match="must be true or false"):
        resolve_settings(tmp_path, overrides=["runtime.offline=maybe"], environ={})


def test_a_symlinked_config_layer_is_refused(tmp_path: Path) -> None:
    real = _write(tmp_path / "real.toml", "[retrieval]\nrrf_k = 30\n")
    link = tmp_path / "link.toml"
    link.symlink_to(real)

    with pytest.raises(SettingsError, match="must not be a symlink"):
        resolve_settings(tmp_path, config_path=link, environ={})


def test_the_report_names_every_setting_and_its_layer(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write(
        project / ".research-rag" / "config.toml",
        '[runtime]\nlog_level = "info"\n',
    )

    settings, provenance = resolve_settings(project, environ={})
    rendered = describe_settings(settings, provenance)

    assert "project config" in rendered
    for setting in SETTINGS:
        assert setting.key in rendered


def test_resolve_config_exposes_the_merged_settings(project: Path) -> None:
    config = resolve_config(
        project,
        vanilla_executable=sys.executable,
        settings_overrides=["retrieval.rrf_k=42", "chunking.size=200"],
    )

    assert config.settings.rrf_k == 42
    assert config.settings.chunk_size == 200
    # The convenience names read through to the settings rather than copying them.
    assert config.offline is False
    assert config.reranker_model == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert config.settings_provenance["retrieval.rrf_k"] == "command line"

    # A bad value from any layer is still reported as a configuration problem.
    with pytest.raises(ConfigurationError, match="must be one of"):
        resolve_config(
            project,
            vanilla_executable=sys.executable,
            dense_backend="lancedb",
        )

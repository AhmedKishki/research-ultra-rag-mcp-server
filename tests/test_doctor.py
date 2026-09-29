"""The doctor command: one line per dependency, and no repair without a flag.

The default run is the behaviour that matters most: it must read the installation
without changing it, name the condition it found, and name the command that fixes
it. The two operations that reach the network are tested against stand-ins, so
these tests never download anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from vanilla_ultra_rag_mcp import runtime as vanilla_runtime

import research_ultra_rag_mcp.doctor as doctor_module
from research_ultra_rag_mcp.config import ResearchConfig, resolve_config
from research_ultra_rag_mcp.doctor import (
    DoctorError,
    check_entry,
    mcp_entry_block,
    run_doctor,
)

READY_STATUS: dict[str, Any] = {
    "ready": True,
    "stale": False,
    "generation_id": "20260101T000000Z-abcdef",
    "upgrade_reasons": [],
    "retained_generation_bytes": 1024,
    "message": "Ready.",
}


@pytest.fixture
def config(project: Path, tmp_path: Path) -> ResearchConfig:
    return resolve_config(
        project_root=project,
        model_cache_root=tmp_path / "models",
        runtime_cache_root=tmp_path / "runtime-cache",
    )


@pytest.fixture
def healthy(config: ResearchConfig, monkeypatch: pytest.MonkeyPatch) -> ResearchConfig:
    """A project whose dependencies are all in place."""

    from research_ultra_rag_mcp.embeddings import resolve_embedding_model
    from research_ultra_rag_mcp.rerankers import resolve_reranker_model

    for name, revision in (
        (
            "qdrant/bge-small-en-v1.5-onnx-q",
            resolve_embedding_model(config.settings.embedding_model).revision,
        ),
        resolve_reranker_model(config.reranker_model),
    ):
        snapshot = (
            config.model_cache_root
            / f"models--{name.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        snapshot.mkdir(parents=True, exist_ok=True)
    root = Path(config.runtime_cache_root) / "runtime" / "UltraRAG-test"
    root.mkdir(parents=True, exist_ok=True)
    (root / vanilla_runtime.MARKER_FILENAME).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(vanilla_runtime, "managed_runtime_path", lambda _c: root)
    monkeypatch.setattr(vanilla_runtime, "validate_managed_runtime", lambda p: p)
    return config


def _run(config: ResearchConfig, **kwargs: Any) -> Any:
    return run_doctor(
        config,
        dict(READY_STATUS),
        running=kwargs.pop("running", []),
        **kwargs,
    )


def test_the_default_run_reports_every_check_and_changes_nothing(
    config: ResearchConfig,
) -> None:
    before = sorted(path.name for path in config.state_root.rglob("*"))

    result = _run(config)

    assert sorted(path.name for path in config.state_root.rglob("*")) == before
    for name in ("project_identity", "runtime_root", "vanilla_runtime", "lock"):
        assert any(name in line for line in result.lines)
    # Nothing is installed and nothing blocks: an online project downloads what
    # it needs on first use, which is a warning, not a broken installation.
    assert result.exit_code == 0
    assert any(line.startswith("warn") for line in result.lines)


def test_every_check_is_one_line_naming_its_state_and_its_remedy(
    config: ResearchConfig,
) -> None:
    offline = resolve_config(
        project_root=config.project_root,
        model_cache_root=config.model_cache_root,
        runtime_cache_root=config.runtime_cache_root,
        offline=True,
    )

    result = _run(offline)

    checks = [
        line for line in result.lines if line[:8].strip() in {"ok", "warn", "blocked"}
    ]
    assert len(checks) == 9
    for line in checks:
        state, name, _rest = line.split(maxsplit=2)
        assert state in {"ok", "warn", "blocked", "unknown"}
        assert name
    # Anything a caller must act on names the command that acts on it.
    for line in checks:
        if line.startswith("blocked"):
            assert "  ->  " in line


def test_a_healthy_installation_exits_zero(healthy: ResearchConfig) -> None:
    result = _run(healthy)

    assert result.exit_code == 0
    assert not [line for line in result.lines if line.startswith(("warn", "blocked"))]


def test_offline_without_an_installed_runtime_is_blocked(
    config: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline is the mode where a missing dependency stops the server."""

    offline = resolve_config(
        project_root=config.project_root,
        model_cache_root=config.model_cache_root,
        runtime_cache_root=config.runtime_cache_root,
        offline=True,
    )

    result = _run(offline)

    assert result.exit_code == 1
    assert any(
        line.startswith("blocked") and "vanilla_runtime" in line
        for line in result.lines
    )


def test_servers_are_reported_with_the_stop_command(config: ResearchConfig) -> None:
    running = [
        (4321, f"research-ultra-rag-mcp --project-root {config.project_root}"),
    ]

    result = _run(config, running=running)

    text = result.text()
    assert "4321" in text
    assert "stop --servers" in text
    # The doctor is not a server, so it does not report itself.
    doctor_pid = run_doctor(
        config,
        dict(READY_STATUS),
        running=[
            (9999, f"research-ultra-rag --project-root {config.project_root} doctor")
        ],
    )
    assert "9999" not in doctor_pid.text()


def test_a_relocated_root_in_use_is_named(
    config: ResearchConfig, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    running = [
        (
            4321,
            (
                "research-ultra-rag-mcp --project-root "
                f"{config.project_root} --runtime-root {elsewhere}"
            ),
        ),
    ]

    text = _run(config, running=running).text()

    assert str(elsewhere) in text
    assert "--runtime-root" in text


def test_the_mcp_entry_matches_the_resolved_configuration(
    config: ResearchConfig, tmp_path: Path
) -> None:
    relocated = resolve_config(
        project_root=config.project_root,
        runtime_root=tmp_path / "relocated",
        model_cache_root=config.model_cache_root,
    )

    entry = json.loads(mcp_entry_block(relocated))

    command = entry["mcp"]["research-ultra-rag-mcp"]["command"]
    assert command[0] == f"{Path(sys.executable).parent}/research-ultra-rag-mcp"
    assert Path(command[0]).is_file()
    flags = dict(zip(command[1::2], command[2::2], strict=True))
    assert flags == {
        "--project-root": str(config.project_root),
        "--runtime-root": str(tmp_path / "relocated"),
    }
    assert entry["mcp"]["research-ultra-rag-mcp"]["timeout"] == 3600000
    # The example entry this repository ships is the same shape, comments and all:
    # the reader that has to cope with a hand-edited file is the doctor itself.
    from research_ultra_rag_mcp.doctor import _strip_jsonc

    example = json.loads(
        _strip_jsonc(
            (Path(__file__).resolve().parents[1] / "kilo-mcp.example.jsonc").read_text(
                encoding="utf-8"
            )
        )
    )
    assert set(example["mcp"]["research-ultra-rag-mcp"]) == set(
        entry["mcp"]["research-ultra-rag-mcp"]
    )


def test_checking_an_entry_never_writes_it(
    config: ResearchConfig, tmp_path: Path
) -> None:
    entry_path = tmp_path / "mcp.jsonc"
    entry_path.write_text(
        json.dumps(
            {
                "mcp": {
                    "research-ultra-rag-mcp": {
                        "type": "local",
                        "command": [
                            f"{Path(sys.executable).parent}/research-ultra-rag-mcp",
                            "--project-root",
                            str(config.project_root),
                        ],
                        "timeout": 3600000,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    before = entry_path.read_bytes()

    findings = check_entry(config, entry_path)

    assert entry_path.read_bytes() == before
    assert findings[-1].state == "ok"


def test_an_entry_without_a_usable_executable_is_blocked(
    config: ResearchConfig, tmp_path: Path
) -> None:
    entry_path = tmp_path / "mcp.json"
    entry_path.write_text(
        json.dumps(
            {
                "mcp": {
                    "research-ultra-rag-mcp": {
                        "command": ["/nowhere/research-ultra-rag-mcp"],
                        "args": ["--project-root", str(config.project_root)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    findings = check_entry(config, entry_path)

    assert any(finding.state == "blocked" for finding in findings)
    assert any("timeout" in finding.name for finding in findings)


def test_a_second_entry_for_one_project_is_reported(
    config: ResearchConfig, tmp_path: Path
) -> None:
    entry_path = tmp_path / "mcp.json"
    entry_path.write_text(
        json.dumps(
            {
                "mcp": {
                    "research-ultra-rag-mcp": _command(config),
                    "research-ultra-rag-mcp-copy": _command(config),
                }
            }
        ),
        encoding="utf-8",
    )

    findings = check_entry(config, entry_path)

    duplicate = [f for f in findings if f.name == "entry.duplicate"]
    assert len(duplicate) == 1
    assert duplicate[0].state == "warn"
    assert "research-ultra-rag-mcp-copy" in duplicate[0].reason


def _command(config: ResearchConfig) -> dict[str, Any]:
    return {
        "command": ["/nowhere/research-ultra-rag-mcp"],
        "args": ["--project-root", str(config.project_root)],
        "timeout": 3600000,
    }


def test_an_entry_file_without_the_server_is_a_usage_error(
    config: ResearchConfig, tmp_path: Path
) -> None:
    entry_path = tmp_path / "mcp.json"
    entry_path.write_text(json.dumps({"mcp": {"other": {}}}), encoding="utf-8")

    with pytest.raises(DoctorError, match="No research-ultra-rag-mcp entry"):
        check_entry(config, entry_path)


def test_a_missing_entry_file_is_a_usage_error(config: ResearchConfig) -> None:
    with pytest.raises(DoctorError, match="No client entry file"):
        check_entry(config, Path("/nowhere/mcp.json"))


def test_a_comment_in_an_entry_file_is_not_a_parse_error(
    config: ResearchConfig, tmp_path: Path
) -> None:
    entry_path = tmp_path / "mcp.jsonc"
    entry_path.write_text(
        '{\n  // the project\n  "mcp": {"research-ultra-rag-mcp": '
        + json.dumps(_command(config))
        + "}\n}\n",
        encoding="utf-8",
    )

    findings = check_entry(config, entry_path)

    assert findings


def test_the_two_operations_cannot_run_together(config: ResearchConfig) -> None:
    with pytest.raises(DoctorError, match="separate operations"):
        _run(config, prefetch=True, repair=True)

    with pytest.raises(DoctorError, match="cannot run with an operation"):
        _run(config, entry=Path("/nowhere/mcp.json"), repair=True)


def test_repair_moves_a_mismatched_tree_aside_and_installs(
    config: ResearchConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A tree that failed validation is evidence, so it is kept, never deleted."""

    root = Path(config.runtime_cache_root) / "runtime" / "UltraRAG-deadbeef"
    (root / "servers").mkdir(parents=True)
    (root / "servers" / "stray.pyc").write_bytes(b"stale")
    installed: list[bool] = []

    def install(cache_root: Any) -> Path:
        installed.append(True)
        root.mkdir(parents=True, exist_ok=True)
        (root / "servers").mkdir(parents=True, exist_ok=True)
        (root / "servers" / "stray.pyc").unlink(missing_ok=True)
        return root

    monkeypatch.setattr(vanilla_runtime, "managed_runtime_path", lambda _c: root)
    monkeypatch.setattr(
        vanilla_runtime,
        "validate_managed_runtime",
        lambda path: _raise_unless_fresh(path, root),
    )
    monkeypatch.setattr(vanilla_runtime, "install_managed_runtime", install)

    lines = doctor_module.repair_runtime(config)

    assert installed == [True]
    preserved = next(line for line in lines if "preserved" in line)
    assert "quarantine" in preserved
    assert "deadbeef" in preserved
    # The evidence stays on disk, and the new tree is beside it, not in its place.
    assert Path(preserved.split()[-1].rstrip(".")).is_dir()
    assert (root / "servers" / "stray.pyc").exists() is False


def _raise_unless_fresh(path: Path, root: Path) -> Path:
    if (root / "servers" / "stray.pyc").exists():
        raise vanilla_runtime.RuntimeValidationError(
            "Managed runtime content hash mismatch: got 0123456789abcdef, "
            "expected fedcba9876543210. The tree differs at "
            "servers/stray.pyc: unexpected in the installed tree, mode -rw-r--r--."
        )
    return path


def test_repair_leaves_a_valid_tree_alone(
    config: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(config.runtime_cache_root) / "runtime" / "UltraRAG-good"
    root.mkdir(parents=True)
    monkeypatch.setattr(vanilla_runtime, "managed_runtime_path", lambda _c: root)
    monkeypatch.setattr(vanilla_runtime, "validate_managed_runtime", lambda p: p)
    monkeypatch.setattr(
        vanilla_runtime,
        "install_managed_runtime",
        lambda _c: pytest.fail("a valid tree must not be reinstalled"),
    )

    lines = doctor_module.repair_runtime(config)

    assert lines == (
        f"The managed runtime at {root} already matches the pinned snapshot.",
    )


def test_prefetch_reports_what_it_cached(
    config: ResearchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[str] = []

    import research_ultra_rag_mcp.dense as dense_module

    monkeypatch.setattr(
        dense_module,
        "_load_embedder",
        lambda _root, offline, model: loaded.append(model),
    )
    monkeypatch.setattr(
        dense_module,
        "_load_cross_encoder",
        lambda _root, offline, model: loaded.append(model),
    )

    lines = doctor_module.prefetch_models(config)

    assert len(loaded) == 2
    assert any("bge-small-en-v1.5" in line for line in lines)
    assert any("MiniLM" in line for line in lines)

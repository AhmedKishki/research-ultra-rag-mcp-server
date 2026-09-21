"""Project-root UI launcher generation and its link."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from research_ultra_rag_mcp.config import resolve_config
from research_ultra_rag_mcp.launcher import (
    LINK_TARGET,
    ensure_ui_launcher,
    launcher_path,
    launcher_script,
    ui_launcher_state,
)


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "research-project"
    (root / "sources").mkdir(parents=True)
    return root


def _alive(pid: int) -> bool:
    """True while the process exists and is not a zombie."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat.rsplit(") ", 1)[1].split()[0] != "Z"


def test_initialisation_creates_the_launcher_and_the_root_link(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    resolve_config(project, vanilla_executable=sys.executable)

    script = launcher_path(project / ".research-rag")
    link = project / "open-ui.sh"
    assert script.is_file()
    assert os.access(script, os.X_OK)
    assert link.is_symlink()
    assert os.readlink(link) == LINK_TARGET

    body = script.read_text(encoding="utf-8")
    assert str(project.resolve()) in body
    assert 'setsid "$UI_COMMAND"' in body
    assert "unset RESEARCH_ULTRARAG_UI_PORT" in body
    assert "RESEARCH_ULTRARAG_UI_PORT=" not in body

    state = ui_launcher_state(project, project / ".research-rag")
    assert state["script_present"] is True
    assert state["link_state"] == "linked"
    assert state["link_target"] == LINK_TARGET


def test_existing_launcher_and_foreign_root_entry_are_never_overwritten(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    portable = project / ".research-rag"
    script = launcher_path(portable)
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n# hand-written\n", encoding="utf-8")
    link = project / "open-ui.sh"
    link.write_text("# a real file the user owns\n", encoding="utf-8")

    result = ensure_ui_launcher(
        project_root=project,
        portable_root=portable,
        state_root=portable / "runtime",
        project_name="research-project",
    )

    assert script.read_text(encoding="utf-8") == "#!/bin/sh\n# hand-written\n"
    assert link.is_file() and not link.is_symlink()
    assert result["script_state"] == "present"
    assert result["link_state"] == "foreign_file"
    assert result["note"]

    file_state = ui_launcher_state(project, portable)
    assert file_state["link_state"] == "foreign_file"


def test_relocated_runtime_root_is_embedded_in_the_launcher(tmp_path: Path) -> None:
    project = _project(tmp_path)
    runtime_root = tmp_path / "relocated-runtime"
    resolve_config(
        project,
        vanilla_executable=sys.executable,
        runtime_root=runtime_root,
    )

    body = launcher_path(project / ".research-rag").read_text(encoding="utf-8")
    assert f'RUNTIME_ROOT="{runtime_root.resolve()}"' in body
    assert "--runtime-root" in body


def test_ui_launcher_state_reports_absent_and_broken_links(tmp_path: Path) -> None:
    project = _project(tmp_path)
    portable = project / ".research-rag"
    portable.mkdir(parents=True)

    assert ui_launcher_state(project, portable)["link_state"] == "absent"

    link = project / "open-ui.sh"
    link.symlink_to(LINK_TARGET)
    assert ui_launcher_state(project, portable)["link_state"] == "broken_link"

    (portable / "bin").mkdir()
    launcher_path(portable).write_text("#!/bin/sh\n", encoding="utf-8")
    assert ui_launcher_state(project, portable)["link_state"] == "linked"


def test_launcher_script_renders_project_specific_values() -> None:
    body = launcher_script(
        project_root=Path("/tmp/example-project"),
        state_root=Path("/tmp/example-project/.research-rag/runtime"),
        project_name="example-project",
        port=5099,
    )

    assert body.startswith("#!/bin/sh\n")
    assert "example-project" in body
    assert "PORT=5099" in body
    assert 'RUNTIME_ROOT=""' in body
    assert 'PROJECT_ROOT="/tmp/example-project"' in body


@pytest.mark.skipif(
    shutil.which("setsid") is None or not Path("/proc").is_dir(),
    reason="the launcher process-group test needs Linux, /proc, and setsid",
)
def test_launcher_starts_and_stops_the_whole_process_group(tmp_path: Path) -> None:
    project = _project(tmp_path)
    portable = project / ".research-rag"
    state = portable / "runtime"
    state.mkdir(parents=True)
    stub_directory = tmp_path / "stub-bin"
    stub_directory.mkdir()
    child_pid_file = tmp_path / "stub-child.pid"
    argument_file = tmp_path / "stub-arguments.txt"
    stub = stub_directory / "research-ultra-rag-ui"
    stub.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" > "$STUB_ARGUMENT_FILE"\n'
        "sleep 30 &\n"
        'printf \'%s\\n\' "$!" > "$STUB_CHILD_FILE"\n'
        "sleep 30\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)

    ensure_ui_launcher(
        project_root=project,
        portable_root=portable,
        state_root=state,
        project_name="research-project",
    )
    script = project / "open-ui.sh"
    environment = {
        **os.environ,
        "PATH": f"{stub_directory}{os.pathsep}{os.environ['PATH']}",
        "STUB_CHILD_FILE": str(child_pid_file),
        "STUB_ARGUMENT_FILE": str(argument_file),
    }

    try:
        started = subprocess.run(
            [str(script), "--port", "5099"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert started.returncode == 0, started.stderr
        assert "127.0.0.1:5099" in started.stdout
        arguments = argument_file.read_text(encoding="utf-8")
        assert "--project-root" in arguments
        assert "--port 5099" in arguments

        pid = int((state / "open-ui.pid").read_text(encoding="utf-8").strip())
        child = int(child_pid_file.read_text(encoding="utf-8").strip())
        assert _alive(pid)
        assert _alive(child)

        stopped = subprocess.run(
            [str(script), "--stop"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert stopped.returncode == 0, stopped.stderr
        for _ in range(40):
            if not _alive(pid) and not _alive(child):
                break
            time.sleep(0.25)
        assert not _alive(pid)
        assert not _alive(child)
        assert not (state / "open-ui.pid").exists()

        idle = subprocess.run(
            [str(script), "--stop"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert idle.returncode == 0
        assert "No UI is running" in idle.stdout
    finally:
        subprocess.run(
            [str(script), "--stop"],
            env=environment,
            capture_output=True,
            timeout=60,
            check=False,
        )

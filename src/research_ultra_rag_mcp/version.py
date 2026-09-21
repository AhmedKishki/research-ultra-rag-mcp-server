"""Version reporting for the running server, its environment, and the shared UI."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

DISTRIBUTION_NAME = "research-ultra-rag-mcp"
UI_DISTRIBUTION_NAME = "ui-ultra-rag-mcp"
UNKNOWN_VERSION = "0.0.0+unknown"


def distribution_version(name: str) -> str | None:
    """Return an installed distribution's version, or None when it is not installed."""

    try:
        return _distribution_version(name)
    except PackageNotFoundError:
        return None


# Read once, when this module is first imported: a running server keeps naming the
# version it started with, so an update that only changed the environment is visible
# as a difference instead of being reported as already current.
SERVER_VERSION = distribution_version(DISTRIBUTION_NAME) or UNKNOWN_VERSION


def installed_version() -> str:
    """Return the version installed in the environment right now."""

    return distribution_version(DISTRIBUTION_NAME) or UNKNOWN_VERSION


def ui_version() -> str | None:
    """Return the installed shared-UI version, or None when it is not installed."""

    return distribution_version(UI_DISTRIBUTION_NAME)


def restart_required() -> bool:
    """Whether this process started with a different version than is installed now."""

    return installed_version() != SERVER_VERSION


def version_block() -> dict[str, object]:
    """Describe the running, installed, and UI versions for `status`."""

    return {
        "server": SERVER_VERSION,
        "installed": installed_version(),
        "ui": ui_version(),
        "restart_required": restart_required(),
    }


def version_label() -> str:
    """Return the short header label the browser UI shows under the project name."""

    parts = [f"{DISTRIBUTION_NAME} {SERVER_VERSION}"]
    ui = ui_version()
    if ui is not None:
        parts.append(f"UI {ui}")
    return " · ".join(parts)

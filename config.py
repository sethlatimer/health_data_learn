"""Environment-backed settings for the local health data application."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
_LEGACY_DOCUMENT_CACHE = PROJECT_ROOT / r"C:\Temp\public_documents"


@dataclass(frozen=True)
class RuntimeSettings:
    project_root: Path
    database_path: Path
    corpus_dir: Path
    document_cache_dir: Path
    allowed_file_root: Path
    host: str
    port: int
    debug: bool


def _path_setting(name, default, environ):
    raw_value = environ.get(name)
    path = Path(raw_value).expanduser() if raw_value else Path(default)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _boolean_setting(name, default, environ):
    raw_value = environ.get(name)
    if raw_value is None:
        return default
    value = raw_value.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def get_settings(environ: Mapping[str, str] | None = None) -> RuntimeSettings:
    """Return validated settings; relative paths are rooted at the project."""
    source = os.environ if environ is None else environ
    corpus_dir = _path_setting(
        "HEALTH_DATA_CORPUS_DIR",
        PROJECT_ROOT / "corpus",
        source,
    )
    if _LEGACY_DOCUMENT_CACHE.is_dir():
        default_cache = _LEGACY_DOCUMENT_CACHE
    else:
        default_cache = PROJECT_ROOT / "data" / "public_documents"

    try:
        port = int(source.get("HEALTH_DATA_PORT", "5000"))
    except (TypeError, ValueError) as error:
        raise ValueError("HEALTH_DATA_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError("HEALTH_DATA_PORT must be between 1 and 65535")

    host = source.get("HEALTH_DATA_HOST", "127.0.0.1").strip()
    if not host:
        raise ValueError("HEALTH_DATA_HOST must not be empty")

    return RuntimeSettings(
        project_root=PROJECT_ROOT,
        database_path=_path_setting(
            "HEALTH_DATA_DB_PATH",
            PROJECT_ROOT / "agent.db",
            source,
        ),
        corpus_dir=corpus_dir,
        document_cache_dir=_path_setting(
            "HEALTH_DATA_DOCUMENT_CACHE_DIR",
            default_cache,
            source,
        ),
        allowed_file_root=_path_setting(
            "HEALTH_DATA_ALLOWED_FILE_ROOT",
            corpus_dir,
            source,
        ),
        host=host,
        port=port,
        debug=_boolean_setting("HEALTH_DATA_DEBUG", False, source),
    )

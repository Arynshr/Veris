"""Persist LLM connection settings to the user-level env file (~/.veris/.env, owner-only)."""

import os
from pathlib import Path

from veris.core.config import USER_ENV_PATH, Settings

_FIELDS = {
    "llm_provider": "LLM_PROVIDER",
    "llm_model": "LLM_MODEL",
    "llm_base_url": "LLM_BASE_URL",
    "llm_api_key": "LLM_API_KEY",
}


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def save_llm_settings(settings: Settings, path: Path = USER_ENV_PATH) -> Path:
    """Merge the LLM_* keys into `path`, keeping every other line, with 0600 permissions."""
    values = {
        env: (v.get_secret_value() if hasattr(v, "get_secret_value") else v)
        for field, env in _FIELDS.items()
        if (v := getattr(settings, field))
    }
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [ln for ln in existing if ln.split("=", 1)[0].strip() not in _FIELDS.values()]
    lines = kept + [f"{k}={_quote(v)}" for k, v in values.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    tmp.replace(path)
    return path

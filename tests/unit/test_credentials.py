import os

import pytest

from veris.testing import make_settings
from veris.core.credentials import save_llm_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "GROQ_API_KEY", "GROQ_MODEL"):
        monkeypatch.delenv(key, raising=False)


def test_legacy_groq_settings_migrate():
    s = make_settings(groq_api_key="k")
    assert (s.llm_provider, s.llm_api_key.get_secret_value(), s.llm_model) == ("groq", "k", "openai/gpt-oss-20b")


def test_explicit_llm_settings_beat_legacy():
    s = make_settings(groq_api_key="k", llm_provider="openai", llm_model="m", llm_api_key="x")
    assert (s.llm_provider, s.llm_api_key.get_secret_value()) == ("openai", "x")


def test_env_file_is_read(tmp_path):
    env = tmp_path / ".env"
    env.write_text('LLM_PROVIDER=openai\nLLM_MODEL=m\nLLM_API_KEY="k#1"\n')
    s = make_settings(env_file=env)
    assert s.llm_model == "m" and s.llm_api_key.get_secret_value() == "k#1"


def test_process_env_beats_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=from-file\n")
    monkeypatch.setenv("LLM_MODEL", "from-env")
    assert make_settings(env_file=env).llm_model == "from-env"


def test_save_round_trip_merges_and_restricts(tmp_path):
    path = tmp_path / "home" / ".env"
    path.parent.mkdir()
    path.write_text("TAVILY_API_KEY=t\nLLM_MODEL=old\n")
    s = make_settings(llm_provider="openai", llm_model="new", llm_api_key='k"#\\x')
    save_llm_settings(s, path)
    back = make_settings(env_file=path)
    assert back.llm_model == "new" and back.llm_api_key.get_secret_value() == 'k"#\\x'
    assert back.tavily_api_key == "t" and "old" not in path.read_text()
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600

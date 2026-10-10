import socket
from datetime import date
from pathlib import Path

import pytest

from tests.reporting import Recorder, results_dir
from veris.core.config import Settings
from veris.core.documents import build_document
from veris.core.models import Entity

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
_ENV_KEYS = (
    "TAVILY_API_KEY",
    "FIRECRAWL_API_KEY",
    "SEARXNG_BASE_URL",
    "GROQ_API_KEY",
)
_NO_KEYS = {
    "tavily_api_key": None,
    "firecrawl_api_key": None,
    "searxng_base_url": None,
    "groq_api_key": None,
}


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(Recorder(results_dir()), "veris-recorder")


def _spacy_available() -> bool:
    try:
        import spacy.util

        return spacy.util.is_package("en_core_web_sm")
    except ImportError:
        return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    skip_spacy = pytest.mark.skip(reason="spaCy model en_core_web_sm not installed")
    has_spacy = _spacy_available()
    for item in items:
        if not has_spacy and item.get_closest_marker("requires_spacy"):
            item.add_marker(skip_spacy)
        parts = Path(str(item.fspath)).parts
        if "integration" in parts:
            item.add_marker(pytest.mark.integration)
        elif "unit" in parts:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(self, *args, **kwargs):
        raise RuntimeError("network access is blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def settings_factory(tmp_path: Path):
    def _make(**overrides) -> Settings:
        base = {
            **_NO_KEYS,
            "as_of_date": date(2026, 1, 1),
            "runs_dir": tmp_path / "runs",
            "source_tiers_path": FIXTURES / "source_tiers.yaml",
            "taxonomy_path": ROOT / "config" / "taxonomy.yaml",
            "safety_path": ROOT / "config" / "safety.yaml",
        }
        return Settings(**{**base, **overrides}, _env_file=None)

    return _make


@pytest.fixture
def prod_settings(settings_factory) -> Settings:
    """Settings pointing at the real config/ files (for config-sanity tests only)."""
    return settings_factory(source_tiers_path=ROOT / "config" / "source_tiers.yaml")


@pytest.fixture
def settings(settings_factory) -> Settings:
    return settings_factory()


@pytest.fixture
def taxonomy(prod_settings):
    from veris.analysis.taxonomy import load_taxonomy

    return load_taxonomy(prod_settings.taxonomy_path)


@pytest.fixture
def golden_dir() -> Path:
    path = ROOT / "eval" / "golden"
    if not path.exists():
        pytest.skip(f"golden set not found at {path}")
    return path


@pytest.fixture
def make_doc():
    def _make(url: str, content: str, title: str = "t"):
        return build_document(url, title, content)

    return _make


@pytest.fixture
def entities() -> list[Entity]:
    return [
        Entity(text="Jane Doe", label="PERSON", start_char=0, end_char=8),
        Entity(text="Acme Corp", label="ORG", start_char=9, end_char=18),
    ]

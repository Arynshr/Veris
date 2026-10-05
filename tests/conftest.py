from pathlib import Path

import pytest

from veris.core.config import Settings
from veris.core.documents import build_document
from veris.core.models import Entity

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        source_tiers_path=ROOT / "config" / "source_tiers.yaml",
        _env_file=None,
    )


@pytest.fixture
def golden_dir() -> Path:
    return ROOT / "eval" / "golden"


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

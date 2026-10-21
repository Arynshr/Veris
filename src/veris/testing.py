"""Test doubles for the provider interfaces (LLM, Extractor, crawl guard)."""

from pathlib import Path

from pydantic_settings import SettingsConfigDict

from veris.core.config import Settings
from veris.core.documents import build_document
from veris.core.ports import FetchOutcome, FetchResult


class FakeLLM:
    def __init__(self, payload: dict | None = None, text: str = "", error: Exception | None = None):
        self._payload, self._text, self._error = payload or {}, text, error
        self.calls: list[tuple[str, str]] = []
        self.failures: list[str] = []

    def complete(self, system, user, *, schema=None, json_mode=False) -> str:
        self.calls.append((system, user))
        if self._error:
            raise self._error
        return self._text

    def complete_json(self, system, user, *, schema=None) -> dict:
        self.calls.append((system, user))
        if self._error:
            raise self._error
        return dict(self._payload)


def page(url: str, content: str = "body text", title: str = "t") -> FetchResult:
    doc = build_document(url=url, title=title, content=content)
    return FetchResult(outcome=FetchOutcome.OK, document=doc)


class FakeExtractor:
    """`pages` maps URL -> FetchResult; unknown URLs return an ERROR outcome."""

    def __init__(self, pages: dict[str, FetchResult]):
        self.pages, self.calls = pages, []

    async def check(self) -> None:
        return None

    async def fetch(self, url: str, title_hint: str = "") -> FetchResult:
        self.calls.append(url)
        return self.pages.get(url, FetchResult(outcome=FetchOutcome.ERROR, detail="not in fixtures"))

    async def close(self) -> None:
        return None


class AllowAllGuard:
    """Crawl guard that never blocks and never touches the network."""

    def __init__(self):
        self.recorded: list[tuple[str, FetchOutcome]] = []

    async def check(self, url: str):
        return None

    def record(self, url: str, outcome: FetchOutcome) -> None:
        self.recorded.append((url, outcome))


def make_settings(env_file: Path | None = None, **overrides) -> Settings:
    """Settings that ignore the user's real .env files (only `env_file`, if given, is read)."""

    class _Isolated(Settings):
        model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="utf-8", extra="ignore")

    return _Isolated(**overrides)

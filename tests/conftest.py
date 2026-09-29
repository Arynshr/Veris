import json
from collections.abc import Callable

import pytest

from veris.core.config import Settings
from veris.core.models import SearchProviderName, SearchQuery, SearchResult
from veris.retrieval.search import SearchProvider


@pytest.fixture
def settings_factory(tmp_path) -> Callable[..., Settings]:
    """Settings with no external services configured and an isolated runs_dir,
    so tests never touch real API keys or the real filesystem outside tmp_path.
    Override any field via kwargs."""

    def _make(**overrides) -> Settings:
        defaults = dict(
            tavily_api_key=None,
            firecrawl_api_key=None,
            searxng_base_url=None,
            groq_api_key=None,
            runs_dir=tmp_path / "runs",
        )
        defaults.update(overrides)
        return Settings(**defaults)

    return _make


@pytest.fixture
def settings(settings_factory) -> Settings:
    return settings_factory()


class FakeSearchProvider(SearchProvider):
    """Deterministic in-memory SearchProvider. `responses` maps normalized
    query text -> list of (title, snippet, url) tuples; anything not in the
    map returns []. Records every call for assertions."""

    name = SearchProviderName.TAVILY

    def __init__(
        self,
        responses: dict[str, list[tuple[str, str, str]]] | None = None,
        fail_on: set[str] | None = None,
    ):
        self.responses = responses or {}
        self.fail_on = fail_on or set()
        self.calls: list[str] = []

    async def search(
        self, query: SearchQuery, max_results: int, news_bias: bool = False
    ) -> list[SearchResult]:
        self.calls.append(query.text)
        if query.text in self.fail_on:
            raise RuntimeError(f"simulated failure for '{query.text}'")
        items = self.responses.get(query.text, [])
        return [
            SearchResult(
                query_id=query.query_id,
                provider=self.name,
                url=url,
                title=title,
                snippet=snippet,
                rank=i,
            )
            for i, (title, snippet, url) in enumerate(items[:max_results], start=1)
        ]


def make_search_query(
    text: str, query_id: str = "q1", priority: int = 1
) -> SearchQuery:
    return SearchQuery(
        query_id=query_id, text=text, priority=priority, rationale="test"
    )


class FakeGroqResponse:
    def __init__(self, content: str):
        self.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": content})()})
        ]


class FakeGroqClient:
    """Stand-in for groq.Groq(). `content_fn` receives the kwargs passed to
    chat.completions.create(...) and returns the JSON string the "model"
    should reply with - lets tests assert on what prompt was actually sent."""

    def __init__(self, content_fn: Callable[..., str], api_key: str | None = None):
        self._content_fn = content_fn
        self.calls: list[dict] = []
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, outer: "FakeGroqClient"):
            self.completions = outer._Completions(outer)

    class _Completions:
        def __init__(self, outer: "FakeGroqClient"):
            self._outer = outer

        def create(self, **kwargs):
            self._outer.calls.append(kwargs)
            content = self._outer._content_fn(**kwargs)
            return FakeGroqResponse(content)


def fake_groq_factory(content_fn: Callable[..., str]):
    """Returns a class usable as a drop-in replacement for groq.Groq via monkeypatch."""

    def _factory(api_key: str | None = None):
        return FakeGroqClient(content_fn, api_key=api_key)

    return _factory


def static_json_groq(payload: dict):
    """Fake Groq client that always replies with the same JSON payload,
    regardless of what was asked."""
    return fake_groq_factory(lambda **_: json.dumps(payload))

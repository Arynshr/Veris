import pytest

from veris.core.models import SearchProviderName, SearchResult
from veris.retrieval.extraction import _canonicalize_url, retrieve_documents


def _sr(url: str, rank: int, query_id: str = "q1") -> SearchResult:
    return SearchResult(
        query_id=query_id,
        provider=SearchProviderName.TAVILY,
        url=url,
        title=url,
        snippet="",
        rank=rank,
    )


def test_canonicalize_url_strips_query_and_fragment_and_trailing_slash():
    assert (
        _canonicalize_url("https://Example.com/Path/?utm_source=x#frag")
        == "https://example.com/Path"
    )
    assert _canonicalize_url("https://example.com/path/") == _canonicalize_url(
        "https://example.com/path"
    )


@pytest.mark.asyncio
async def test_retrieve_documents_dedupes_same_url_across_queries(monkeypatch):
    """Regression: the same URL returned by two different queries in a plan
    used to be crawled twice."""
    calls = []

    async def fake_extract(url, title_hint=""):
        calls.append(url)
        return _fake_document(url)

    monkeypatch.setattr("veris.retrieval.extraction._extract_document", fake_extract)

    results = [
        _sr("https://a.example/x", rank=1, query_id="q1"),
        _sr("https://a.example/x?utm=1", rank=2, query_id="q2"),
    ]
    docs = await retrieve_documents(results)
    assert len(docs) == 1
    assert calls == ["https://a.example/x"]  # best-ranked occurrence fetched


@pytest.mark.asyncio
async def test_retrieve_documents_caps_at_max_documents_best_ranked_first(monkeypatch):
    """Regression for the 413 (tokens-per-minute) bug: retrieval must not
    silently crawl every raw search hit across every query - it should keep
    only the best-ranked N."""

    async def fake_extract(url, title_hint=""):
        return _fake_document(url)

    monkeypatch.setattr("veris.retrieval.extraction._extract_document", fake_extract)

    results = [
        _sr(f"https://a.example/{i}", rank=i) for i in range(1, 31)
    ]  # 30 raw hits
    docs = await retrieve_documents(results, max_documents=10)
    assert len(docs) == 10
    fetched_urls = {d.url for d in docs}
    assert fetched_urls == {
        f"https://a.example/{i}" for i in range(1, 11)
    }  # best-ranked 10


@pytest.mark.asyncio
async def test_retrieve_documents_uses_shared_cache_across_calls(monkeypatch):
    calls = []

    async def fake_extract(url, title_hint=""):
        calls.append(url)
        return _fake_document(url)

    monkeypatch.setattr("veris.retrieval.extraction._extract_document", fake_extract)

    cache: dict = {}
    await retrieve_documents([_sr("https://a.example/x", rank=1)], cache=cache)
    await retrieve_documents([_sr("https://a.example/x", rank=1)], cache=cache)
    assert calls == ["https://a.example/x"]  # second call was a cache hit


@pytest.mark.asyncio
async def test_retrieve_documents_skips_failed_extractions(monkeypatch):
    async def fake_extract(url, title_hint=""):
        return None  # simulates a failed crawl

    monkeypatch.setattr("veris.retrieval.extraction._extract_document", fake_extract)
    docs = await retrieve_documents([_sr("https://a.example/x", rank=1)])
    assert docs == []


def _fake_document(url: str):
    import hashlib

    from veris.core.models import Document

    canonical = _canonicalize_url(url)
    return Document(
        document_id=hashlib.sha1(canonical.encode()).hexdigest()[:16],
        url=url,
        canonical_url=canonical,
        title=url,
        content="some content here",
        source="a.example",
        content_hash=hashlib.sha256(url.encode()).hexdigest(),
    )

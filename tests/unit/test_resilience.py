import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from veris.testing import make_settings
from veris.core.documents import build_document
from veris.core.models import RetrievalStats, SearchProviderName, SearchQuery, SearchResult
from veris.core.ports import FetchOutcome, FetchResult
from veris.retrieval.crawl_policy import BlockReason, CrawlGuard
from veris.retrieval.extraction import retrieve_documents
from veris.retrieval.search import (
    ProviderResponseError,
    SearchProvider,
    _normalize,
    search_with_fallback,
)

ROBOTS = "User-agent: *\nDisallow: /private/*\n"


def settings(tmp_path, **kw):
    return make_settings(crawl_memory_path=tmp_path / "mem.json", **kw)


def guard(tmp_path, robots=(200, ROBOTS), clock=None, calls=None):
    def handler(request):
        if calls is not None:
            calls.append(str(request.url))
        return httpx.Response(robots[0], text=robots[1])

    return CrawlGuard(settings(tmp_path), transport=httpx.MockTransport(handler), clock=clock)


def test_robots_disallow_is_remembered_across_runs(tmp_path):
    calls = []
    g = guard(tmp_path, calls=calls)
    assert asyncio.run(g.check("https://a.test/private/x")) == (BlockReason.ROBOTS, False)
    assert asyncio.run(g.check("https://a.test/public")) is None
    g2 = guard(tmp_path, calls=calls)
    calls.clear()
    assert asyncio.run(g2.check("https://a.test/private/x")) == (BlockReason.ROBOTS, True)
    assert calls == []


def test_unreachable_robots_blocks_but_is_not_remembered(tmp_path):
    g = guard(tmp_path, robots=(503, ""))
    assert asyncio.run(g.check("https://a.test/x")) == (BlockReason.ROBOTS, False)
    assert not (tmp_path / "mem.json").exists()


def test_missing_robots_allows(tmp_path):
    assert asyncio.run(guard(tmp_path, robots=(404, "")).check("https://a.test/x")) is None


def test_repeated_denials_block_host_and_transients_are_ignored(tmp_path):
    g = guard(tmp_path, robots=(404, ""))
    g.record("https://a.test/1", FetchOutcome.TIMEOUT)
    assert not (tmp_path / "mem.json").exists()
    for i in range(3):
        g.record(f"https://a.test/{i}", FetchOutcome.FORBIDDEN)
    assert asyncio.run(g.check("https://a.test/new")) == (BlockReason.FORBIDDEN, True)
    assert asyncio.run(g.check("https://b.test/new")) is None


def test_entries_expire(tmp_path):
    now = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    g = guard(tmp_path, robots=(404, ""), clock=lambda: now[0])
    g.record("https://a.test/p", FetchOutcome.PAYWALLED)
    assert asyncio.run(g.check("https://a.test/p")) == (BlockReason.PAYWALLED, True)
    now[0] += timedelta(days=8)
    assert asyncio.run(g.check("https://a.test/p")) is None


def test_corrupt_memory_is_ignored(tmp_path):
    (tmp_path / "mem.json").write_text("{not json")
    assert asyncio.run(guard(tmp_path, robots=(404, "")).check("https://a.test/x")) is None


class FakeExtractor:
    def __init__(self, script):
        self.script, self.calls = script, []

    async def fetch(self, url, title_hint=""):
        self.calls.append(url)
        outcome = self.script[url].pop(0)
        doc = build_document(url=url, title="t", content="body text") if outcome is FetchOutcome.OK else None
        return FetchResult(outcome=outcome, document=doc)

    async def close(self):
        pass


def result(url, rank, snippet=""):
    return SearchResult(query_id="q", provider=SearchProviderName.TAVILY, url=url, title="t", snippet=snippet, rank=rank)


def test_retrieval_outcomes_retry_and_stats(tmp_path, monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("veris.retrieval.extraction.asyncio.sleep", no_sleep)
    g = guard(tmp_path, robots=(200, "User-agent: *\nDisallow: /blocked\n"))
    ex = FakeExtractor(
        {
            "https://a.test/ok": [FetchOutcome.OK],
            "https://a.test/flaky": [FetchOutcome.TIMEOUT, FetchOutcome.OK],
            "https://a.test/denied": [FetchOutcome.FORBIDDEN],
            "https://a.test/dead": [FetchOutcome.ERROR, FetchOutcome.ERROR],
        }
    )
    stats = RetrievalStats()
    docs = asyncio.run(
        retrieve_documents(
            [
                result("https://a.test/ok", 1),
                result("https://a.test/flaky", 2),
                result("https://a.test/denied", 3, snippet="s"),
                result("https://a.test/dead", 4),
                result("https://a.test/blocked", 5),
            ],
            ex,
            g,
            stats=stats,
            retries=1,
        )
    )
    assert len(docs) == 2
    assert "https://a.test/blocked" not in ex.calls
    assert (stats.attempted, stats.fetched, stats.access_denied, stats.failed, stats.robots_blocked) == (5, 2, 1, 1, 1)
    assert stats.snippet_only == 1
    assert asyncio.run(g.check("https://a.test/denied")) == (BlockReason.FORBIDDEN, True)


def test_normalize_skips_malformed_and_rejects_bad_shape():
    q = SearchQuery(query_id="q", text="x", priority=1)
    payload = {"results": [{"url": "https://a.test"}, {"title": "no url"}, "junk", {"url": "ftp://x"}]}
    out = _normalize(q, SearchProviderName.TAVILY, payload, "results", "content")
    assert [r.url for r in out] == ["https://a.test"] and out[0].rank == 1
    with pytest.raises(ProviderResponseError):
        _normalize(q, SearchProviderName.TAVILY, ["x"], "results", "content")


def test_circuit_breaker_and_failure_report(tmp_path):
    class Bad(SearchProvider):
        name = SearchProviderName.TAVILY
        calls = 0

        async def search(self, query, max_results, news_bias=False):
            Bad.calls += 1
            raise RuntimeError("down")

    class Good(SearchProvider):
        name = SearchProviderName.FIRECRAWL

        async def search(self, query, max_results, news_bias=False):
            return [result("https://a.test/x", 1)]

    s, failures, chain = settings(tmp_path, provider_failure_threshold=2), [], [Bad(), Good()]
    for i in range(4):
        q = SearchQuery(query_id=f"q{i}", text=f"q{i}", priority=1)
        assert asyncio.run(search_with_fallback(q, s, chain=chain, failures=failures))
    assert Bad.calls == 2 and len(failures) == 2 and failures[0].startswith("tavily")

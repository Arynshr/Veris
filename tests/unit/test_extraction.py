import asyncio

from veris.core.models import SearchProviderName, SearchResult
from veris.core.ports import FetchOutcome, FetchResult
from veris.retrieval.extraction import retrieve_documents
from veris.testing import AllowAllGuard, FakeExtractor, page


def result(url, rank):
    return SearchResult(query_id="q", provider=SearchProviderName.TAVILY, url=url, title="t", snippet="", rank=rank)


def run(results, extractor, **kw):
    return asyncio.run(retrieve_documents(results, extractor, AllowAllGuard(), retries=0, **kw))


def test_duplicate_urls_fetched_once_best_rank_first():
    ex = FakeExtractor({"https://a.test/1": page("https://a.test/1"), "https://a.test/2": page("https://a.test/2")})
    docs = run([result("https://a.test/2", 2), result("https://a.test/1", 1), result("https://a.test/1", 3)], ex)
    assert sorted(ex.calls) == ["https://a.test/1", "https://a.test/2"] and len(docs) == 2


def test_max_documents_keeps_best_ranked():
    urls = [f"https://a.test/{i}" for i in range(4)]
    ex = FakeExtractor({u: page(u) for u in urls})
    run([result(u, i) for i, u in enumerate(urls)], ex, max_documents=2)
    assert sorted(ex.calls) == urls[:2]


def test_shared_cache_avoids_refetch_including_failures():
    ok, bad = "https://a.test/ok", "https://a.test/bad"
    ex, cache = FakeExtractor({ok: page(ok)}), {}
    for _ in range(2):
        docs = run([result(ok, 1), result(bad, 2)], ex, cache=cache)
    assert len(docs) == 1 and sorted(ex.calls) == [bad, ok]


def test_one_failure_does_not_stop_the_rest():
    ok = "https://a.test/ok"
    ex = FakeExtractor({ok: page(ok), "https://a.test/x": FetchResult(outcome=FetchOutcome.TIMEOUT)})
    assert [d.url for d in run([result("https://a.test/x", 1), result(ok, 2)], ex)] == [ok]


def test_unavailable_extractor_is_not_retried_and_error_is_surfaced():
    from veris.core.models import RetrievalStats

    url, detail = "https://a.test/x", "The browser used for page extraction is not installed."
    ex = FakeExtractor({url: FetchResult(outcome=FetchOutcome.UNAVAILABLE, detail=detail)})
    stats = RetrievalStats()
    asyncio.run(retrieve_documents([result(url, 1)], ex, AllowAllGuard(), retries=3, stats=stats))
    assert ex.calls == [url] and stats.failed == 1 and stats.sample_errors == [detail]


def test_launch_hint_gives_actionable_install_command():
    from veris.retrieval.extraction import _launch_hint

    assert "crawl4ai-setup" in _launch_hint(Exception("BrowserType.launch: Executable doesn't exist at /x"))
    assert "install-deps" in _launch_hint(Exception("Host system is missing dependencies to run browsers"))


def test_degraded_report_includes_retrieval_errors():
    from veris.core.models import RetrievalStats
    from veris.pipeline import RunCounters, _degraded

    counters = RunCounters(retrieval=RetrievalStats(attempted=3, failed=3, sample_errors=["boom"]))
    assert {"retrieval_no_documents", "retrieval_error:boom"} <= set(_degraded(counters, []))

import pytest
from tests.helpers import FakeSearchProvider, make_search_query

from veris.retrieval.search import (
    FirecrawlProvider,
    SearXNGProvider,
    TavilyProvider,
    build_provider_chain,
    search_with_fallback,
)


def test_build_provider_chain_raises_when_nothing_configured(settings_factory):
    with pytest.raises(RuntimeError, match="No search providers configured"):
        build_provider_chain(settings_factory())


def test_build_provider_chain_skips_unconfigured_searxng(settings_factory):
    """Regression: previously searxng_base_url defaulted to localhost, so
    SearXNGProvider was always added to the chain even with nothing running
    there. With only Tavily configured, the chain must contain exactly
    Tavily - not Tavily + a doomed SearXNG entry."""
    chain = build_provider_chain(settings_factory(tavily_api_key="fake-key"))
    assert len(chain) == 1
    assert isinstance(chain[0], TavilyProvider)


def test_build_provider_chain_includes_searxng_only_when_explicitly_set(
    settings_factory,
):
    chain = build_provider_chain(
        settings_factory(searxng_base_url="http://localhost:8080")
    )
    assert len(chain) == 1
    assert isinstance(chain[0], SearXNGProvider)


def test_build_provider_chain_order_and_multi(settings_factory):
    chain = build_provider_chain(
        settings_factory(
            tavily_api_key="a", firecrawl_api_key="b", searxng_base_url="http://x"
        )
    )
    assert [type(p) for p in chain] == [
        TavilyProvider,
        FirecrawlProvider,
        SearXNGProvider,
    ]


@pytest.mark.asyncio
async def test_search_with_fallback_returns_first_nonempty_provider(settings_factory):
    empty = FakeSearchProvider(responses={})
    hit = FakeSearchProvider(
        responses={"solid-state batteries": [("T", "S", "https://a.example/x")]}
    )
    settings = settings_factory()
    results = await search_with_fallback(
        make_search_query("solid-state batteries"), settings, chain=[empty, hit]
    )
    assert len(results) == 1
    assert empty.calls == ["solid-state batteries"]
    assert hit.calls == ["solid-state batteries"]


@pytest.mark.asyncio
async def test_search_with_fallback_skips_provider_that_raises(settings_factory):
    broken = FakeSearchProvider(fail_on={"q"})
    ok = FakeSearchProvider(responses={"q": [("T", "S", "https://a.example/x")]})
    results = await search_with_fallback(
        make_search_query("q"), settings_factory(), chain=[broken, ok]
    )
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_with_fallback_caches_identical_query_text(settings_factory):
    """Regression for 'searching the same task multiple times': an identical
    query, however it re-enters the pipeline (duplicate plan query, role
    resolution re-asking the same thing), must only hit providers once."""
    provider = FakeSearchProvider(
        responses={"who is the ceo of acme": [("T", "S", "https://a.example/x")]}
    )
    settings = settings_factory()
    cache: dict = {}
    q1 = make_search_query("who is the CEO of Acme")
    q2 = make_search_query(
        "  Who is the CEO of Acme  "
    )  # different case/whitespace, same meaning

    r1 = await search_with_fallback(q1, settings, chain=[provider], cache=cache)
    r2 = await search_with_fallback(q2, settings, chain=[provider], cache=cache)

    assert provider.calls == ["who is the CEO of Acme"]  # only called once
    assert r1 == r2


@pytest.mark.asyncio
async def test_search_with_fallback_no_cache_means_no_dedup(settings_factory):
    provider = FakeSearchProvider(responses={"q": [("T", "S", "https://a.example/x")]})
    settings = settings_factory()
    await search_with_fallback(make_search_query("q"), settings, chain=[provider])
    await search_with_fallback(make_search_query("q"), settings, chain=[provider])
    assert provider.calls == ["q", "q"]

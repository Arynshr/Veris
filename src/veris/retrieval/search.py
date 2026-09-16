"""
Search layer: a common SearchProvider interface with three
backends, orchestrated with fallback Tavily -> Firecrawl -> SearXNG. A single
provider failure never crashes the run
"""
import asyncio
from abc import ABC, abstractmethod

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from tavily import TavilyClient

from veris.core.config import Settings, get_logger
from veris.core.models import SearchProviderName, SearchQuery, SearchResult

logger = get_logger(__name__)

_FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v1/search"


class ProviderUnavailableError(RuntimeError):
    """Raised when a provider is misconfigured (e.g. missing API key)."""


class SearchProvider(ABC):
    name: SearchProviderName

    @abstractmethod
    async def search(self, query: SearchQuery, max_results: int, news_bias: bool = False) -> list[SearchResult]:
        """Return ranked results. Must raise on hard failure so the caller can fall back."""


class TavilyProvider(SearchProvider):
    name = SearchProviderName.TAVILY

    def __init__(self, api_key: str | None, max_retries: int = 3, lookback_days: int | None = None) -> None:
        if not api_key:
            raise ProviderUnavailableError("TAVILY_API_KEY not configured")
        self._client = TavilyClient(api_key=api_key)
        self._max_retries = max_retries
        self._lookback_days = lookback_days

    async def search(self, query: SearchQuery, max_results: int, news_bias: bool = False) -> list[SearchResult]:
        @retry(stop=stop_after_attempt(self._max_retries), wait=wait_exponential(multiplier=1, min=1, max=8))
        def _call():
            kwargs = {"query": query.text, "max_results": max_results}
            if news_bias:
                kwargs["topic"] = "news"
                if self._lookback_days:
                    kwargs["days"] = self._lookback_days
            return self._client.search(**kwargs)

        response = await asyncio.to_thread(_call)
        return [
            SearchResult(
                query_id=query.query_id, provider=self.name, url=item["url"],
                title=item.get("title", ""), snippet=item.get("content", ""), rank=rank,
            )
            for rank, item in enumerate(response.get("results", []), start=1)
        ]


class FirecrawlProvider(SearchProvider):
    """Fallback search provider used when Tavily fails or is unavailable."""

    name = SearchProviderName.FIRECRAWL

    def __init__(self, api_key: str | None, timeout: int = 20, max_retries: int = 3) -> None:
        if not api_key:
            raise ProviderUnavailableError("FIRECRAWL_API_KEY not configured")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries

    async def search(self, query: SearchQuery, max_results: int, news_bias: bool = False) -> list[SearchResult]:
        @retry(stop=stop_after_attempt(self._max_retries), wait=wait_exponential(multiplier=1, min=1, max=8))
        async def _call():
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    _FIRECRAWL_SEARCH_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"query": query.text, "limit": max_results},
                )
                resp.raise_for_status()
                return resp.json()

        data = await _call()
        return [
            SearchResult(
                query_id=query.query_id, provider=self.name, url=item["url"],
                title=item.get("title", ""), snippet=item.get("description", ""), rank=rank,
            )
            for rank, item in enumerate(data.get("data", [])[:max_results], start=1)
        ]


class SearXNGProvider(SearchProvider):
    """Metadata/search support provider. Self-hosted, opt-in - no default URL."""

    name = SearchProviderName.SEARXNG

    def __init__(self, base_url: str | None, timeout: int = 20, max_retries: int = 3) -> None:
        if not base_url:
            raise ProviderUnavailableError("SEARXNG_BASE_URL not configured")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries

    async def search(self, query: SearchQuery, max_results: int, news_bias: bool = False) -> list[SearchResult]:
        @retry(stop=stop_after_attempt(self._max_retries), wait=wait_exponential(multiplier=1, min=1, max=8))
        async def _call():
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self._base_url}/search", params={"q": query.text, "format": "json"})
                resp.raise_for_status()
                return resp.json()

        data = await _call()
        return [
            SearchResult(
                query_id=query.query_id, provider=self.name, url=item["url"],
                title=item.get("title", ""), snippet=item.get("content", ""), rank=rank,
            )
            for rank, item in enumerate(data.get("results", [])[:max_results], start=1)
        ]


def build_provider_chain(settings: Settings) -> list[SearchProvider]:
    """Build the fallback chain once per run (not once per query - previously
    rebuilt on every single search call, up to ~48 times per run, each
    re-triggering SearXNG's always-configured bug and re-logging warnings)."""
    chain: list[SearchProvider] = []
    for provider_cls, kwargs in (
        (TavilyProvider, {"api_key": settings.tavily_api_key, "max_retries": settings.max_retries, "lookback_days": settings.adverse_lookback_days}),
        (FirecrawlProvider, {"api_key": settings.firecrawl_api_key, "timeout": settings.request_timeout_seconds, "max_retries": settings.max_retries}),
        (SearXNGProvider, {"base_url": settings.searxng_base_url, "timeout": settings.request_timeout_seconds, "max_retries": settings.max_retries}),
    ):
        try:
            chain.append(provider_cls(**kwargs))
        except ProviderUnavailableError as exc:
            logger.debug("Provider not configured, skipping: %s", exc)
    if not chain:
        raise RuntimeError(
            "No search providers configured. Set at least one of TAVILY_API_KEY, "
            "FIRECRAWL_API_KEY, or SEARXNG_BASE_URL."
        )
    return chain


async def search_with_fallback(
    query: SearchQuery, settings: Settings, chain: list[SearchProvider], news_bias: bool = False,
) -> list[SearchResult]:
    last_error: Exception | None = None
    for provider in chain:
        try:
            results = await provider.search(query, settings.max_search_results_per_query, news_bias=news_bias)
            if results:
                return results
        except Exception as exc: 
            logger.warning("Search provider %s failed for query '%s': %s", provider.name, query.text, exc)
            last_error = exc
    logger.error("All search providers returned nothing for query '%s': %s", query.text, last_error or "no results")
    return []

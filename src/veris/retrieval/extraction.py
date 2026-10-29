"""Extraction: Crawl4AI behind the Extractor interface, plus compliant, resilient retrieval."""

import asyncio
from collections.abc import Collection

from veris.core.config import Settings, get_logger
from veris.core.documents import build_document, canonicalize_url
from veris.core.models import Document, RetrievalStats, SearchResult
from veris.core.ports import Extractor, ExtractorUnavailable, FetchOutcome, FetchResult
from veris.retrieval.crawl_policy import CrawlGuard

logger = get_logger(__name__)

_RETRYABLE = {FetchOutcome.TIMEOUT, FetchOutcome.ERROR}
_MAX_SAMPLE_ERRORS = 3
_STAT_FIELD = {
    "error": "failed",
    "empty": "failed",
    "rate_limited": "failed",
    "unavailable": "failed",
    "timeout": "timed_out",
    "forbidden": "access_denied",
    "paywalled": "paywalled",
    "robots_blocked": "robots_blocked",
    "memory_skipped": "memory_skipped",
}


def _launch_hint(exc: Exception) -> str:
    text = str(exc)
    if "Executable doesn't exist" in text or "playwright install" in text:
        return (
            "The browser used for page extraction is not installed. "
            "Run `crawl4ai-setup` (or `python -m playwright install chromium`) in this environment, then retry."
        )
    if "missing dependencies" in text.lower():
        return "The browser is missing system libraries. Run `python -m playwright install-deps chromium`, then retry."
    return f"Could not start the extraction browser: {text.splitlines()[0][:200] if text else type(exc).__name__}"


class Crawl4AIExtractor:
    def __init__(
        self,
        user_agent: str,
        timeout: int,
        concurrency: int,
        paywall_codes: Collection[int],
        denied_codes: Collection[int],
    ) -> None:
        self._ua = user_agent
        self._timeout = timeout
        self._paywall = set(paywall_codes)
        self._denied = set(denied_codes)
        self._sem = asyncio.Semaphore(concurrency)
        self._start_lock = asyncio.Lock()
        self._crawler = None
        self._checked = False

    async def check(self) -> None:
        """Launch and close a headless browser once, so a broken install fails before any spend."""
        if self._checked:
            return
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise ExtractorUnavailable("crawl4ai is not installed. Run `pip install crawl4ai`, then `crawl4ai-setup`.") from exc
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                await browser.close()
        except Exception as exc:  # noqa: BLE001
            raise ExtractorUnavailable(_launch_hint(exc)) from exc
        self._checked = True

    async def _ensure(self):
        async with self._start_lock:
            if self._crawler is None:
                try:
                    from crawl4ai import AsyncWebCrawler, BrowserConfig

                    crawler = AsyncWebCrawler(config=BrowserConfig(user_agent=self._ua, verbose=False))
                    await crawler.start()
                except ImportError as exc:
                    raise ExtractorUnavailable("crawl4ai is not installed. Run `pip install crawl4ai`.") from exc
                except Exception as exc:  # noqa: BLE001
                    raise ExtractorUnavailable(_launch_hint(exc)) from exc
                self._crawler = crawler
        return self._crawler

    async def close(self) -> None:
        crawler, self._crawler = self._crawler, None
        if crawler is not None:
            try:
                await crawler.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Crawler shutdown failed: %s", exc)

    async def fetch(self, url: str, title_hint: str = "") -> FetchResult:
        async with self._sem:
            try:
                crawler = await self._ensure()
                result = await asyncio.wait_for(crawler.arun(url=url), self._timeout)
            except ExtractorUnavailable as exc:
                return FetchResult(outcome=FetchOutcome.UNAVAILABLE, detail=str(exc)[:200])
            except asyncio.TimeoutError:
                return FetchResult(outcome=FetchOutcome.TIMEOUT, detail=f"timed out after {self._timeout}s")
            except Exception as exc:  # noqa: BLE001
                return FetchResult(outcome=FetchOutcome.ERROR, detail=f"{type(exc).__name__}: {exc}"[:200])
        return self._to_result(url, title_hint, result)

    def _to_result(self, url: str, title_hint: str, result) -> FetchResult:
        code = getattr(result, "status_code", None)
        content = str(getattr(result, "markdown", "") or "").strip()
        if code in self._paywall:
            outcome = FetchOutcome.PAYWALLED
        elif code in self._denied:
            outcome = FetchOutcome.FORBIDDEN
        elif code == 429:
            outcome = FetchOutcome.RATE_LIMITED
        elif getattr(result, "success", False) and content:
            title = (getattr(result, "metadata", None) or {}).get("title") or title_hint
            doc = build_document(url=url, title=title, content=content)
            return FetchResult(outcome=FetchOutcome.OK, document=doc, status_code=code)
        else:
            outcome = FetchOutcome.EMPTY if getattr(result, "success", False) else FetchOutcome.ERROR
        detail = str(getattr(result, "error_message", "") or "")[:200]
        return FetchResult(outcome=outcome, status_code=code, detail=detail)


def build_extractor(settings: Settings) -> Extractor:
    return Crawl4AIExtractor(
        settings.crawler_user_agent,
        settings.extraction_timeout_seconds,
        settings.extraction_concurrency,
        settings.paywall_status_codes,
        settings.denied_status_codes,
    )


async def _fetch_with_retry(extractor: Extractor, r: SearchResult, retries: int) -> FetchResult:
    for attempt in range(retries + 1):
        result = await extractor.fetch(r.url, r.title)
        if result.outcome not in _RETRYABLE or attempt == retries:
            break
        await asyncio.sleep(min(2**attempt, 8))
    return result


async def _retrieve_one(
    r: SearchResult, extractor: Extractor, guard: CrawlGuard, retries: int
) -> tuple[Document | None, str, str]:
    try:
        if blocked := await guard.check(r.url):
            reason, remembered = blocked
            logger.info("Skipping %s (%s)", r.url, reason.value)
            return None, "memory_skipped" if remembered else "robots_blocked", reason.value
        result = await _fetch_with_retry(extractor, r, retries)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Retrieval failed for %s: %s", r.url, exc)
        return None, "error", f"{type(exc).__name__}: {exc}"[:200]
    guard.record(r.url, result.outcome)
    if result.outcome is not FetchOutcome.OK:
        log = logger.warning if result.outcome in (FetchOutcome.ERROR, FetchOutcome.UNAVAILABLE) else logger.info
        log("Not retrieved %s: %s %s", r.url, result.outcome.value, result.detail)
    return result.document, result.outcome.value, result.detail


async def retrieve_documents(
    search_results: list[SearchResult],
    extractor: Extractor,
    guard: CrawlGuard,
    *,
    cache: dict[str, Document | None] | None = None,
    max_documents: int | None = None,
    stats: RetrievalStats | None = None,
    retries: int = 1,
) -> list[Document]:
    """One Document per unique URL, best-ranked first, capped at `max_documents`.
    `cache` is shared across the run; `stats` accumulates every outcome."""
    by_canonical: dict[str, SearchResult] = {}
    for r in search_results:
        canonical = canonicalize_url(r.url)
        existing = by_canonical.get(canonical)
        if existing is None or r.rank < existing.rank:
            by_canonical[canonical] = r

    ranked = sorted(by_canonical.values(), key=lambda r: r.rank)
    if max_documents is not None:
        ranked = ranked[:max_documents]

    to_fetch: list[SearchResult] = []
    cached_docs: list[Document] = []
    for r in ranked:
        canonical = canonicalize_url(r.url)
        if cache is not None and canonical in cache:
            if (hit := cache[canonical]) is not None:
                cached_docs.append(hit)
            continue
        to_fetch.append(r)

    outcomes = await asyncio.gather(*(_retrieve_one(r, extractor, guard, retries) for r in to_fetch))

    fresh_docs: list[Document] = []
    for r, (doc, outcome, detail) in zip(to_fetch, outcomes):
        if cache is not None:
            cache[canonicalize_url(r.url)] = doc
        if stats is not None:
            stats.attempted += 1
            if doc is not None:
                stats.fetched += 1
            else:
                field = _STAT_FIELD[outcome]
                setattr(stats, field, getattr(stats, field) + 1)
                stats.snippet_only += bool(r.snippet.strip())
                sample = detail.strip()[:120]
                if sample and sample not in stats.sample_errors and len(stats.sample_errors) < _MAX_SAMPLE_ERRORS:
                    stats.sample_errors.append(sample)
        if doc is not None:
            fresh_docs.append(doc)

    return cached_docs + fresh_docs

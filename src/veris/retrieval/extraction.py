"""Content extraction via Crawl4AI. Document failures are logged and counted, never fatal."""

import asyncio
from collections.abc import Collection

from crawl4ai import AsyncWebCrawler

from veris.core.config import get_logger
from veris.core.documents import build_document, canonicalize_url
from veris.core.models import Document, RetrievalStats, SearchResult

logger = get_logger(__name__)


async def _extract_document(
    url: str, title_hint: str = "", paywall_codes: Collection[int] = ()
) -> tuple[Document | None, str]:
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
    except Exception as exc:
        logger.warning("Extraction failed for %s: %s", url, exc)
        return None, "failed"

    if result and result.success and (result.markdown or "").strip():
        title = (getattr(result, "metadata", {}) or {}).get("title") or title_hint
        return build_document(url=url, title=title, content=result.markdown), "ok"
    paywalled = getattr(result, "status_code", None) in paywall_codes
    return None, "paywalled" if paywalled else "failed"


async def retrieve_documents(
    search_results: list[SearchResult],
    cache: dict[str, Document | None] | None = None,
    max_documents: int | None = None,
    stats: RetrievalStats | None = None,
    paywall_codes: Collection[int] = (),
) -> list[Document]:
    """One Document per unique URL, best-ranked first, capped at `max_documents`.
    `cache` is shared across the run; `stats` accumulates fetch outcomes."""
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

    extracted = await asyncio.gather(
        *(_extract_document(r.url, r.title, paywall_codes) for r in to_fetch)
    )

    fresh_docs: list[Document] = []
    for r, (doc, outcome) in zip(to_fetch, extracted):
        if cache is not None:
            cache[canonicalize_url(r.url)] = doc
        if stats is not None:
            stats.attempted += 1
            if doc is not None:
                stats.fetched += 1
            else:
                stats.paywalled += outcome == "paywalled"
                stats.failed += outcome == "failed"
                stats.snippet_only += bool(r.snippet.strip())
        if doc is not None:
            fresh_docs.append(doc)

    return cached_docs + fresh_docs

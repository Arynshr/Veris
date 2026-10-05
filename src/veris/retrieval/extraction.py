"""
Content extraction, renamed from the old retrieval.py to avoid
a package/module name clash. Fetches
search results, extracts normalized Documents via Crawl4AI. Individual document
failures are logged and skipped, never crash the run.
"""

import asyncio

from crawl4ai import AsyncWebCrawler

from veris.core.config import get_logger
from veris.core.documents import build_document, canonicalize_url
from veris.core.models import Document, SearchResult

logger = get_logger(__name__)


async def _extract_document(url: str, title_hint: str = "") -> Document | None:
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
    except Exception as exc:
        logger.warning("Extraction failed for %s: %s", url, exc)
        return None

    if not result or not result.success or not (result.markdown or "").strip():
        return None

    return build_document(
        url=url,
        title=(getattr(result, "metadata", {}) or {}).get("title") or title_hint,
        content=result.markdown,
    )


async def retrieve_documents(
    search_results: list[SearchResult],
    cache: dict[str, Document | None] | None = None,
    max_documents: int | None = None,
) -> list[Document]:
    """Extract one Document per unique URL, best-ranked first, capped at
    `max_documents` (settings.max_documents_per_subject).

    Three things fixed here that previously made a single subject balloon
    into far more crawling/tokens downstream than intended:
    1. Within this call: different queries in the same plan frequently
       return the same top result(s) - each occurrence used to be crawled
       separately.
    2. Across calls, via `cache` (share one dict across an entire run in
       pipeline.py): the same URL can surface again for a different role/
       subject search.
    3. `max_documents`: max_search_results_per_query is a *per query* cap,
       so N queries could still add up to N * max_search_results_per_query
       raw hits (e.g. 8 queries x 10 results = up to 80) - all of which
       used to get crawled and every one of which feeds evidence into the
       synthesis prompt. Selecting only the best-ranked `max_documents`
       unique URLs, across all queries combined, is what actually keeps
       retrieval (and the eventual Groq request) concise.
    """
    by_canonical: dict[str, SearchResult] = {}
    for r in search_results:
        canonical = canonicalize_url(r.url)
        existing = by_canonical.get(canonical)
        if existing is None or r.rank < existing.rank:
            by_canonical[canonical] = r  # keep the best rank seen for this URL

    ranked = sorted(by_canonical.values(), key=lambda r: r.rank)
    if max_documents is not None:
        ranked = ranked[:max_documents]

    to_fetch: list[SearchResult] = []
    cached_docs: list[Document] = []
    for r in ranked:
        canonical = canonicalize_url(r.url)
        if cache is not None and canonical in cache:
            hit = cache[canonical]
            if hit is not None:
                cached_docs.append(hit)
            continue
        to_fetch.append(r)

    extracted = await asyncio.gather(
        *(_extract_document(r.url, title_hint=r.title) for r in to_fetch)
    )

    fresh_docs: list[Document] = []
    for r, doc in zip(to_fetch, extracted):
        if cache is not None:
            cache[canonicalize_url(r.url)] = doc
        if doc is not None:
            fresh_docs.append(doc)

    return cached_docs + fresh_docs

"""
Content extraction, renamed from the old retrieval.py to avoid
a package/module name clash. Fetches
search results, extracts normalized Documents via Crawl4AI. Individual document
failures are logged and skipped, never crash the run.
"""
import asyncio
import hashlib
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit

from crawl4ai import AsyncWebCrawler

from veris.core.config import get_logger
from veris.core.models import Document, SearchResult

logger = get_logger(__name__)


def _canonicalize_url(url: str) -> str:
    """Strip fragments/tracking params and normalize scheme/host for dedup."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", parts.netloc.lower(), path, "", ""))


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def _document_id(canonical_url: str) -> str:
    return hashlib.sha1(canonical_url.encode("utf-8")).hexdigest()[:16]


async def _extract_document(url: str, title_hint: str = "") -> Document | None:
    canonical = _canonicalize_url(url)
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
    except Exception as exc: 
        logger.warning("Extraction failed for %s: %s", url, exc)
        return None

    if not result or not result.success or not (result.markdown or "").strip():
        return None

    content = result.markdown.strip()
    return Document(
        document_id=_document_id(canonical),
        url=url,
        canonical_url=canonical,
        title=(getattr(result, "metadata", {}) or {}).get("title") or title_hint or canonical,
        content=content,
        source=canonical.split("/")[2] if "//" in canonical else canonical,
        retrieved_at=datetime.now(UTC),
        published_at=None,
        content_hash=_content_hash(content),
    )


async def retrieve_documents(search_results: list[SearchResult]) -> list[Document]:
    extracted = await asyncio.gather(
        *(_extract_document(r.url, title_hint=r.title) for r in search_results)
    )
    return [doc for doc in extracted if doc is not None]

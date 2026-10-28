"""Document construction shared by live extraction and offline eval fixtures."""

import hashlib
import re
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit

from veris.core.ids import stable_id
from veris.core.models import Document


_AMP_SEGMENT_RE = re.compile(r"/amp(?=/|$)", re.I)


def canonicalize_url(url: str) -> str:
    """Strip fragments/query/tracking, `www.`/`amp.` hosts and /amp segments so
    URL variants of one article dedupe to a single canonical URL."""
    parts = urlsplit(url)
    host = parts.netloc.lower().removeprefix("www.").removeprefix("amp.")
    path = _AMP_SEGMENT_RE.sub("", parts.path).rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", host, path, "", ""))


def content_hash(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def build_document(
    url: str,
    title: str,
    content: str,
    published_at: datetime | None = None,
) -> Document:
    canonical = canonicalize_url(url)
    content = content.strip()
    return Document(
        document_id=stable_id(canonical, length=16),
        url=url,
        canonical_url=canonical,
        title=title or canonical,
        content=content,
        source=canonical.split("/")[2] if "//" in canonical else canonical,
        retrieved_at=datetime.now(UTC),
        published_at=published_at,
        content_hash=content_hash(content),
    )

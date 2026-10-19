"""Crawl compliance: robots.txt (RFC 9309) is always honoured, and refusals are remembered.

A refusal (robots disallow, 401/403, paywall) is never bypassed and is not requested again
until its entry expires. This is a technical-compliance policy, not a legal determination.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from urllib.parse import urlsplit

import httpx
from protego import Protego
from pydantic import BaseModel

from veris.core.config import Settings, get_logger
from veris.core.documents import canonicalize_url
from veris.core.ports import FetchOutcome

logger = get_logger(__name__)

SUPPORTED_SCHEMA_VERSION = 1
_TRANSIENT_ROBOTS_TTL = timedelta(minutes=5)
_ROBOTS_MAX_CHARS = 500_000
_DISALLOW_ALL = "User-agent: *\nDisallow: /"


class BlockReason(str, Enum):
    ROBOTS = "robots_disallow"
    FORBIDDEN = "access_denied"
    PAYWALLED = "paywalled"


_OUTCOME_REASON = {
    FetchOutcome.FORBIDDEN: BlockReason.FORBIDDEN,
    FetchOutcome.PAYWALLED: BlockReason.PAYWALLED,
}


class MemoryEntry(BaseModel):
    reason: BlockReason
    recorded_at: datetime
    expires_at: datetime


class CrawlMemory(BaseModel):
    schema_version: int = SUPPORTED_SCHEMA_VERSION
    urls: dict[str, MemoryEntry] = {}
    hosts: dict[str, MemoryEntry] = {}


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


class CrawlGuard:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._ua = settings.crawler_user_agent
        self._token = self._ua.split("/")[0]
        self._path = settings.crawl_memory_path
        self._robots_ttl = timedelta(hours=settings.robots_ttl_hours)
        self._denied_ttl = timedelta(days=settings.denied_ttl_days)
        self._threshold = settings.host_denied_threshold
        self._timeout = settings.request_timeout_seconds
        self._transport = transport
        self._now = clock or (lambda: datetime.now(timezone.utc))
        self._robots: dict[str, tuple[datetime, Protego, bool]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._memory = self._load()

    def _load(self) -> CrawlMemory:
        try:
            memory = CrawlMemory.model_validate_json(self._path.read_text(encoding="utf-8"))
            if memory.schema_version != SUPPORTED_SCHEMA_VERSION:
                raise ValueError(f"unsupported schema {memory.schema_version}")
        except FileNotFoundError:
            return CrawlMemory()
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable crawl memory %s: %s", self._path, exc)
            return CrawlMemory()
        now = self._now()
        memory.urls = {k: v for k, v in memory.urls.items() if v.expires_at > now}
        memory.hosts = {k: v for k, v in memory.hosts.items() if v.expires_at > now}
        return memory

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(self._memory.model_dump_json(indent=2), encoding="utf-8")
            tmp.replace(self._path)
        except OSError as exc:
            logger.warning("Could not persist crawl memory: %s", exc)

    def _lookup(self, key: str, host: str) -> MemoryEntry | None:
        now = self._now()
        for entry in (self._memory.hosts.get(host), self._memory.urls.get(key)):
            if entry and entry.expires_at > now:
                return entry
        return None

    def _remember(
        self, key: str, reason: BlockReason, ttl: timedelta, host_wide: bool = False
    ) -> None:
        now = self._now()
        entry = MemoryEntry(reason=reason, recorded_at=now, expires_at=now + ttl)
        host = _host(key)
        self._memory.urls[key] = entry
        repeated = reason is not BlockReason.ROBOTS and (
            sum(1 for k, e in self._memory.urls.items() if e.reason is reason and _host(k) == host)
            >= self._threshold
        )
        if host_wide or repeated:
            self._memory.hosts[host] = entry
        self._save()

    async def check(self, url: str) -> tuple[BlockReason, bool] | None:
        """(reason, from_memory) when the URL must not be fetched, else None."""
        key = canonicalize_url(url)
        if entry := self._lookup(key, _host(url)):
            return entry.reason, True
        parser, transient = await self._robots_for(url)
        if parser.can_fetch(url, self._token):
            return None
        if not transient:
            parts = urlsplit(url)
            root = f"{parts.scheme}://{parts.netloc}/"
            self._remember(
                key,
                BlockReason.ROBOTS,
                self._robots_ttl,
                host_wide=not parser.can_fetch(root, self._token),
            )
        return BlockReason.ROBOTS, False

    def record(self, url: str, outcome: FetchOutcome) -> None:
        if reason := _OUTCOME_REASON.get(outcome):
            self._remember(canonicalize_url(url), reason, self._denied_ttl)

    async def _robots_for(self, url: str) -> tuple[Protego, bool]:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        async with self._locks.setdefault(origin, asyncio.Lock()):
            cached = self._robots.get(origin)
            if cached and cached[0] > self._now():
                return cached[1], cached[2]
            parser, transient = await self._fetch_robots(origin)
            ttl = _TRANSIENT_ROBOTS_TTL if transient else self._robots_ttl
            self._robots[origin] = (self._now() + ttl, parser, transient)
            return parser, transient

    async def _fetch_robots(self, origin: str) -> tuple[Protego, bool]:
        """RFC 9309: 2xx parse, 4xx no restrictions, 5xx/unreachable treated as disallow-all (transient)."""
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": self._ua},
                transport=self._transport,
            ) as client:
                resp = await client.get(f"{origin}/robots.txt")
        except httpx.HTTPError as exc:
            logger.warning("robots.txt unreachable for %s (%s); not crawling for now", origin, exc)
            return Protego.parse(_DISALLOW_ALL), True
        if resp.status_code >= 500:
            logger.warning("robots.txt %s for %s; not crawling for now", resp.status_code, origin)
            return Protego.parse(_DISALLOW_ALL), True
        if resp.is_success:
            return Protego.parse(resp.text[:_ROBOTS_MAX_CHARS]), False
        return Protego.parse(""), False

"""Shared fakes and builders. No network, no real LLM."""

import json
from collections.abc import Callable
from datetime import date
from types import SimpleNamespace
from typing import Any

from veris.core.models import (
    Evidence,
    SearchQuery,
    SearchResult,
    SourceMetadata,
)
from veris.retrieval.search import SearchProvider, SearchProviderName

Row = tuple[str, str, str]  # (title, snippet, url)


def _norm(text: str) -> str:
    return text.strip().lower()


def make_search_query(
    text: str, query_id: str = "q1", priority: int = 1
) -> SearchQuery:
    return SearchQuery(query_id=query_id, text=text, priority=priority)


class FakeSearchProvider(SearchProvider):
    """Responses are keyed by normalized query text; `calls` records raw text."""

    name = SearchProviderName.TAVILY

    def __init__(
        self,
        responses: dict[str, list[Row]] | None = None,
        fail_on: set[str] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.fail_on = set(fail_on or ())
        self.calls: list[str] = []

    async def search(self, query, max_results, news_bias=False):
        self.calls.append(query.text)
        if _norm(query.text) in {_norm(t) for t in self.fail_on}:
            raise RuntimeError(f"provider down for {query.text!r}")
        table = {_norm(k): v for k, v in self.responses.items()}
        rows = table.get(_norm(query.text), [])[:max_results]
        return [
            SearchResult(
                query_id=query.query_id,
                provider=self.name,
                url=url,
                title=title,
                snippet=snippet,
                rank=i,
            )
            for i, (title, snippet, url) in enumerate(rows, start=1)
        ]


def fake_groq_factory(content_fn: Callable[..., str]) -> type:
    """Stand-in for `groq.Groq`; `content_fn(**create_kwargs)` returns message text."""

    class _Completions:
        @staticmethod
        def create(**kwargs: Any):
            message = SimpleNamespace(content=content_fn(**kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class _Groq:
        def __init__(self, api_key: str | None = None, **_: Any) -> None:
            self.chat = SimpleNamespace(completions=_Completions())

    return _Groq


def static_json_groq(payload: dict) -> type:
    return fake_groq_factory(lambda **_: json.dumps(payload))


def make_evidence(
    eid: str,
    claim: str,
    url: str = "https://reuters.com/a",
    published: str | None = "2025-03-01",
    flags: list[str] | None = None,
    passage: str | None = None,
) -> Evidence:
    passage = passage or claim
    return Evidence(
        evidence_id=eid,
        document_id=f"d_{eid}",
        claim=claim,
        supporting_passage=passage,
        span_start=0,
        span_end=len(passage),
        flags=flags or [],
        source_metadata=SourceMetadata(
            url=url, title="t", source=url.split("/")[2], published_at=published
        ),
    )


AS_OF = date(2026, 1, 1)


class FakeLLM:
    """Slot LLM stub: replays `replies` (last one repeats) and records prompts."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies, self.calls, self.prompts = list(replies), 0, []

    def __call__(self, system: str, user: str) -> str:
        self.prompts.append(user)
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return reply


def slot_json(text: str, ids: list[str], slot_id: str = "summary") -> str:
    return json.dumps({"slots": [{"slot_id": slot_id, "text": text, "claim_ids": ids}]})


def verify_all(evidence: list[Evidence], settings, subject: str = "John Smith"):
    """Claims, verdicts and a ValidationContext for the given evidence."""
    from veris.analysis.taxonomy import load_taxonomy
    from veris.analysis.verification import run_verification
    from veris.core.models import ResearchMode
    from veris.synthesis.validator import ValidationContext

    claims, results = run_verification(
        evidence, [], subject, settings, ResearchMode.BACKGROUND_CHECK
    )
    ctx = ValidationContext(
        claims={c.claim_id: c for c in claims},
        evidence={e.evidence_id: e for e in evidence},
        verdicts={r.claim_id: r for r in results},
        taxonomy=load_taxonomy(settings.taxonomy_path),
        min_chars=settings.slot_min_chars,
        max_chars=settings.slot_max_chars,
    )
    return claims, results, ctx


def check_slot(ctx, text: str, ids: list[str]) -> list[str]:
    from veris.core.models import SynthesisSlot
    from veris.synthesis.validator import validate_slot

    return validate_slot(SynthesisSlot(slot_id="s", text=text, claim_ids=ids), ctx)

"""
Subject resolution: when query understanding says the request names
a role/group ("the board of directors of XYZ") rather than concrete names, the
actual names aren't known yet and must be looked up before research can start.

Fixed bug: previously every role term (singular or collective) was resolved
by concatenating ALL search-result snippets together and pulling out every
PERSON entity found anywhere in that blob. For a query like "CEO and CFO of
Acme", each role search returned news articles that routinely also mention
other board members, former executives, or journalists in the same snippet -
all of which got added as resolved subjects. The result: asking for two
named seats (CEO, CFO) silently expanded into a background check on the
entire board. Singular, named seats ("CEO", "CFO", "president", "founder", ...)
must resolve to exactly one person each; only genuinely collective phrases
("board of directors", "executives", "leadership team", ...) may resolve to
more than one, and even then only up to a small, dedicated cap
(settings.max_group_role_size), scored by how often a name actually
co-occurs with the role/org context rather than by mere presence anywhere
in the result set.
"""

import asyncio
import json
import re
from collections import Counter

from veris.core.config import Settings, get_logger
from veris.core.models import (
    QueryUnderstanding,
    ResolvedSubject,
    SearchQuery,
    SearchResult,
    SubjectMode,
    SubjectType,
)
from veris.intake.ingestion import extract_entities
from veris.retrieval.search import SearchProvider, search_with_fallback

logger = get_logger(__name__)

# Roles that name exactly one seat. A request naming N of these must resolve
# to exactly N people - never more.
_SINGULAR_ROLE_PATTERN = re.compile(
    r"^(CEO|CFO|COO|CTO|CMO|CIO|CISO|chief executive officer|chief financial officer|"
    r"chief operating officer|chief technology officer|chief marketing officer|"
    r"chairman|chairwoman|chairperson|president|founder|co-founder|owner)$",
    re.IGNORECASE,
)

_ROLE_EXTRACTION_PROMPT = """You are identifying who currently holds a specific named role \
at a specific organization, from search-result snippets. Only name someone if the snippets \
clearly identify them as holding THIS role at THIS organization - do not guess, and do not \
name people who hold a different role or work at a different organization. If no snippet \
clearly supports an answer, say so. Respond with JSON only (no prose, no markdown fences):
{{"name": "<full name>", "confidence": <0-1>}} or {{"name": null, "confidence": 0}}"""


def _is_singular_role(role: str) -> bool:
    return bool(_SINGULAR_ROLE_PATTERN.match(role.strip()))


def _resolution_query(role: str, organization: str | None) -> SearchQuery:
    org_part = f" of {organization}" if organization else ""
    # "{role}{org} names" (old code) read like keyword soup to a search
    # engine and returned low-precision hits. A natural-language query
    # anchored to the org returns more on-topic, single-person results.
    text = f"who is the {role}{org_part}" if organization else f"{role} name"
    return SearchQuery(
        query_id=f"resolve_{re.sub(r'[^a-z0-9]+', '_', role.lower())}",
        text=text,
        priority=1,
        rationale="subject resolution",
    )


def _candidate_counts(
    role: str, organization: str | None, results: list[SearchResult]
) -> Counter:
    """Count PERSON entities per individual result (title+snippet), not one
    entity extraction over the whole concatenated blob - this keeps a
    candidate's count tied to how often *that specific result* (already
    topically scoped by the query) mentions them, instead of every name
    that appears anywhere across all results."""
    counts: Counter = Counter()
    role_lower = role.lower()
    org_lower = (organization or "").lower()
    for r in results:
        text = f"{r.title}. {r.snippet}"
        lowered = text.lower()
        # Require the snippet to actually mention the role or the org, not
        # just contain a stray name - filters out bylines, unrelated people
        # quoted in the same article, etc.
        if role_lower not in lowered and (not org_lower or org_lower not in lowered):
            continue
        for entity in extract_entities(text):
            if entity.label != "PERSON" or len(entity.text.split()) < 2:
                continue
            counts[entity.text.strip()] += 1
    return counts


def _llm_pick_singular(
    role: str, organization: str | None, results: list[SearchResult], settings: Settings
) -> str | None:
    """Best-effort precise pick for a single-seat role, used when a Groq key
    is configured. Falls back silently to the frequency-based candidate on
    any failure - never raises."""
    from groq import Groq

    if not results:
        return None
    snippets = "\n".join(f"- {r.title}: {r.snippet}" for r in results[:8])
    org_part = f" at {organization}" if organization else ""
    try:
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": _ROLE_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": f"Role: {role}{org_part}\n\nSnippets:\n{snippets}",
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        name = (data.get("name") or "").strip()
        if name and data.get("confidence", 0) >= 0.5:
            return name
    except Exception as exc:
        logger.debug(
            "LLM role extraction failed for '%s', using frequency fallback: %s",
            role,
            exc,
        )
    return None


async def resolve_subjects(
    understanding: QueryUnderstanding,
    settings: Settings,
    provider_chain: list[SearchProvider],
    search_cache: dict[str, list[SearchResult]] | None = None,
) -> list[ResolvedSubject]:
    explicit = [
        ResolvedSubject(
            name=name.strip(),
            organization=understanding.organization,
            resolution_confidence=1.0,
        )
        for name in understanding.explicit_subjects
        if name.strip()
    ]

    if understanding.mode != SubjectMode.ROLE_GROUP or not understanding.role_terms:
        return explicit or [
            ResolvedSubject(
                name=understanding.raw_query.strip(), resolution_confidence=1.0
            )
        ]

    resolved: list[ResolvedSubject] = list(explicit)
    seen_names = {r.name.lower() for r in resolved}

    for role in understanding.role_terms:
        if len(resolved) >= settings.max_resolved_subjects:
            logger.info(
                "Reached max_resolved_subjects (%d); skipping remaining role terms.",
                settings.max_resolved_subjects,
            )
            break

        query = _resolution_query(role, understanding.organization)
        try:
            results = await search_with_fallback(
                query,
                settings,
                chain=provider_chain,
                news_bias=False,
                cache=search_cache,
            )
        except Exception as exc:
            logger.warning("Resolution search failed for role '%s': %s", role, exc)
            continue

        singular = _is_singular_role(role)
        # A singular seat resolves to at most 1 new person; a collective
        # role resolves to at most max_group_role_size, and never past the
        # overall max_resolved_subjects ceiling either way.
        slot_budget = (
            1
            if singular
            else max(
                0,
                min(
                    settings.max_group_role_size,
                    settings.max_resolved_subjects - len(resolved),
                ),
            )
        )
        if slot_budget <= 0:
            continue

        chosen: list[str] = []
        if singular and settings.groq_api_key:
            picked = await asyncio.to_thread(
                _llm_pick_singular, role, understanding.organization, results, settings
            )
            if picked:
                chosen = [picked]
        if not chosen:
            counts = _candidate_counts(role, understanding.organization, results)
            # Most-frequently-co-occurring name(s) first - for a singular
            # role this picks the one person the results agree on, instead
            # of appending every name the old code happened to encounter.
            chosen = [name for name, _ in counts.most_common(slot_budget)]

        for name in chosen:
            key = name.strip().lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            resolved.append(
                ResolvedSubject(
                    name=name.strip(),
                    subject_type=SubjectType.PERSON,
                    role=role,
                    organization=understanding.organization,
                    resolution_confidence=0.85 if singular else 0.6,
                )
            )
            if len(resolved) >= settings.max_resolved_subjects:
                break

    return resolved or explicit

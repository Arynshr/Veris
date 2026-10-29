"""
Query understanding: classifies a free-text request into (a) how
many subjects it implies and whether they need to be looked up, and (b) which
research mode applies
"""

import asyncio
import re


from veris.core.config import get_logger
from veris.core.ports import LLM
from veris.core.models import Entity, QueryUnderstanding, ResearchMode, SubjectMode

logger = get_logger(__name__)

_ROLE_PATTERN = re.compile(
    r"\b(board of directors|board members?|executives?|leadership team|management team|"
    r"c-suite|officers?|founders?|co-founders?|directors?|CEO|CFO|COO|CTO|chairman|president)\b",
    re.IGNORECASE,
)

_BACKGROUND_CHECK_PATTERN = re.compile(
    r"\b(background check|due diligence|adverse media|vet(ted|ting)?|screen(ing)?|"
    r"risk (check|assessment)|kyc|any (red flags|controversies|lawsuits|scandals))\b",
    re.IGNORECASE,
)

_UNDERSTANDING_SYSTEM_PROMPT = """Classify a research request. Respond with JSON only \
(no prose, no markdown fences):
{{
  "mode": "single_subject" | "named_list" | "role_group",
  "research_mode": "general" | "background_check",
  "organization": "<company/org name, or null>",
  "role_terms": ["<role or group phrase, e.g. 'CEO', 'CFO', 'board of directors'>"],
  "explicit_subjects": ["<any full names or topics given verbatim in the request>"]
}}
Rules for "mode":
- "single_subject": exactly one named individual/topic is given, nothing to look up.
- "named_list": two or more named individuals/topics are given verbatim, nothing to look up.
- "role_group": the request names a role or group at an organization (board, executives,
  "CEO and CFO", leadership team, etc.) and the actual person names are NOT given - they
  must be looked up. Use this even if some names ARE given alongside unresolved roles;
  put the given names in explicit_subjects and the unresolved roles in role_terms.
Rules for "role_terms":
- List each role as its own separate entry, one person-holding title per entry - e.g.
  "CEO and CFO" -> ["CEO", "CFO"] (two entries, two people), NOT ["CEO and CFO"] (one
  entry). Each of these is a single named seat and must resolve to exactly one person.
- Only use a collective/group phrase ("board of directors", "executives", "leadership
  team", "management team", "c-suite", "officers") as its own single entry when the
  request is actually asking about the whole group - never use a group phrase to
  represent what is really a list of distinct individual titles.
Rules for "research_mode":
- "background_check": the request explicitly asks for a background check, due diligence,
  adverse-media screening, risk check, vetting, or red flags on a person/company/group.
- "general": everything else - explanations, comparisons, current events, how-to, analysis,
  or any other research question, even if it names people or companies."""


def _regex_fallback(query: str, entities: list[Entity]) -> QueryUnderstanding:
    """Deterministic fallback used when the LLM is unavailable."""
    persons = [e.text for e in entities if e.label == "PERSON"]
    orgs = [
        e.text
        for e in entities
        if e.label == "ORG" and not _ROLE_PATTERN.fullmatch(e.text.strip())
    ]
    role_matches = list(
        dict.fromkeys(m.group(0) for m in _ROLE_PATTERN.finditer(query))
    )
    research_mode = (
        ResearchMode.BACKGROUND_CHECK
        if _BACKGROUND_CHECK_PATTERN.search(query)
        else ResearchMode.GENERAL
    )

    if role_matches and not persons:
        return QueryUnderstanding(
            mode=SubjectMode.ROLE_GROUP,
            research_mode=research_mode,
            organization=orgs[0] if orgs else None,
            role_terms=role_matches,
            explicit_subjects=[],
            raw_query=query,
        )
    if len(persons) >= 2:
        return QueryUnderstanding(
            mode=SubjectMode.NAMED_LIST,
            research_mode=research_mode,
            organization=orgs[0] if orgs else None,
            role_terms=[],
            explicit_subjects=persons,
            raw_query=query,
        )
    if len(persons) == 1:
        return QueryUnderstanding(
            mode=SubjectMode.SINGLE,
            research_mode=research_mode,
            organization=orgs[0] if orgs else None,
            role_terms=[],
            explicit_subjects=persons,
            raw_query=query,
        )
    return QueryUnderstanding(
        mode=SubjectMode.SINGLE,
        research_mode=research_mode,
        organization=orgs[0] if orgs else None,
        role_terms=[],
        explicit_subjects=[query.strip()],
        raw_query=query,
    )


async def understand_query(
    query: str, entities: list[Entity], llm: LLM | None
) -> QueryUnderstanding:
    if llm is None:
        return _regex_fallback(query, entities)
    # Blocking network call - offloaded to a thread so it doesn't stall the
    # rest of the async pipeline (see planning.run_planning for the same fix).
    try:
        return await asyncio.to_thread(_llm_classify, query, llm)
    except Exception as exc:
        logger.warning(
            "LLM query understanding failed, falling back to regex classification: %s",
            exc,
        )
        return _regex_fallback(query, entities)


def _llm_classify(query: str, llm: LLM) -> QueryUnderstanding:
    data = llm.complete_json(_UNDERSTANDING_SYSTEM_PROMPT, query)
    return QueryUnderstanding(
        mode=SubjectMode(data["mode"]),
        research_mode=ResearchMode(data.get("research_mode", "general")),
        organization=data.get("organization"),
        role_terms=data.get("role_terms", []),
        explicit_subjects=data.get("explicit_subjects", []),
        raw_query=query,
    )

"""
Research planning (spec section 5), mode-aware: GENERAL mode generates balanced
research-angle queries (overview, recent developments, analysis, counterpoints);
BACKGROUND_CHECK mode generates adverse-media risk-category queries.
"""

import asyncio
import json

from groq import Groq

from veris.core.config import Settings, get_logger
from veris.core.models import (
    AdverseCategory,
    IngestionResult,
    ResearchMode,
    ResearchPlan,
    SearchQuery,
    SourceRequirement,
)

logger = get_logger(__name__)

_GENERAL_TEMPLATES: list[str] = [
    "{subject}",
    "{subject} overview",
    "{subject} latest developments",
    "{subject} analysis",
    "{subject} criticism OR controversy",
]

_ADVERSE_TEMPLATES: dict[AdverseCategory, list[str]] = {
    AdverseCategory.LEGAL: ["{subject} lawsuit", "{subject} litigation OR sued"],
    AdverseCategory.REGULATORY: [
        "{subject} sanctions OR fine OR regulatory action",
        "{subject} SEC OR FTC OR investigation",
    ],
    AdverseCategory.CRIMINAL: [
        "{subject} fraud OR indicted OR charged",
        "{subject} arrest OR criminal investigation",
    ],
    AdverseCategory.FINANCIAL: [
        "{subject} bankruptcy OR insolvency",
        "{subject} embezzlement OR financial misconduct",
    ],
    AdverseCategory.REPUTATIONAL: [
        "{subject} scandal OR controversy",
        "{subject} allegations",
    ],
}

_GENERAL_PLAN_PROMPT = """You are planning research on a topic or named subject. Propose \
additional targeted search queries that are NOT already covered by the "already planned" \
list below - angles specific to this subject (sub-topics, related events, opposing \
viewpoints, recent developments). Each query must:
- mention the subject (or an unambiguous synonym) explicitly, so it can't drift onto an \
unrelated entity
- be a short, literal search-engine query (3-8 words), not a question or full sentence
- not duplicate or trivially reword any query already in the "already planned" list

Respond with JSON only (no prose, no markdown fences):
{{
  "queries": [{{"text": "<search query>", "priority": <1-5, 1=highest>, "rationale": "<why>"}}]
}}
Generate at most {max_queries} additional queries. If you cannot think of that many \
genuinely new angles, return fewer - never pad with near-duplicates."""

_ADVERSE_PLAN_PROMPT = """You are planning an adverse-media background check on a named \
subject. Given the subject and any known aliases/affiliations, propose additional targeted \
search queries that are NOT already covered by the "already planned" list below and that \
would surface adverse news specific to this subject - e.g. their industry, role, known \
associates, or past employers. Each query must:
- mention the subject by name explicitly (never a role, title, or group term alone), so \
results can't be misattributed to a different person or company holding the same role
- be a short, literal search-engine query (3-8 words), not a question or full sentence
- not duplicate or trivially reword any query already in the "already planned" list

Respond with JSON only (no prose, no markdown fences):
{{
  "queries": [{{"text": "<search query>", "priority": <1-5, 1=highest>, "rationale": "<why>", "category": "<legal_litigation|regulatory_sanctions|criminal|financial|reputational>"}}]
}}
Generate at most {max_queries} additional queries. If you cannot think of that many \
genuinely new angles, return fewer - never pad with near-duplicates."""


def _template_plan(
    subject: str, research_mode: ResearchMode, max_queries: int
) -> ResearchPlan:
    queries: list[SearchQuery] = []
    idx = 1

    if research_mode == ResearchMode.BACKGROUND_CHECK:
        objective = f"Adverse media background check: {subject}"
        source_requirement = SourceRequirement(category="news", min_sources=3)
        for category, templates in _ADVERSE_TEMPLATES.items():
            for template in templates:
                if len(queries) >= max_queries:
                    break
                queries.append(
                    SearchQuery(
                        query_id=f"q{idx}",
                        text=template.format(subject=subject),
                        priority=1,
                        rationale=f"Baseline {category.value} coverage",
                    )
                )
                idx += 1
    else:
        objective = f"Research: {subject}"
        source_requirement = SourceRequirement(category="general", min_sources=2)
        for template in _GENERAL_TEMPLATES:
            if len(queries) >= max_queries:
                break
            queries.append(
                SearchQuery(
                    query_id=f"q{idx}",
                    text=template.format(subject=subject),
                    priority=1,
                    rationale="Baseline research-angle coverage",
                )
            )
            idx += 1

    return ResearchPlan(
        objective=objective, queries=queries, source_requirements=[source_requirement]
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _dedupe_queries(queries: list[SearchQuery]) -> list[SearchQuery]:
    """Drop exact/near-duplicate query text, keeping the first (i.e.
    highest-priority / template-before-LLM) occurrence."""
    seen: set[str] = set()
    deduped: list[SearchQuery] = []
    for q in queries:
        key = _normalize(q.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
    return deduped


async def run_planning(
    ingestion: IngestionResult,
    settings: Settings,
    subject: str,
    research_mode: ResearchMode = ResearchMode.GENERAL,
) -> ResearchPlan:
    """Planning entrypoint: baseline mode-appropriate templates + optional LLM-augmented queries."""
    plan = _template_plan(subject, research_mode, settings.max_queries_per_plan)

    if settings.groq_api_key:
        already_planned = [q.text for q in plan.queries]
        llm_queries = await asyncio.to_thread(
            _llm_augment, ingestion, settings, subject, research_mode, already_planned
        )
        plan.queries.extend(llm_queries)

    plan.queries = _dedupe_queries(plan.queries)
    plan.queries.sort(key=lambda q: q.priority)
    plan.queries = plan.queries[: settings.max_queries_per_plan]
    return plan


def _llm_augment(
    ingestion: IngestionResult,
    settings: Settings,
    subject: str,
    research_mode: ResearchMode,
    already_planned: list[str],
) -> list[SearchQuery]:
    client = Groq(api_key=settings.groq_api_key)
    aliases = (
        ", ".join(
            e.text for e in ingestion.entities if e.text.lower() != subject.lower()
        )
        or "none known"
    )
    planned_block = "\n".join(f"- {q}" for q in already_planned) or "(none yet)"
    user_msg = (
        f"Subject: {subject}\n"
        f"Known related entities/aliases: {aliases}\n"
        f"Context: {ingestion.original_prompt}\n\n"
        f"Already planned (do not duplicate):\n{planned_block}"
    )
    prompt_template = (
        _ADVERSE_PLAN_PROMPT
        if research_mode == ResearchMode.BACKGROUND_CHECK
        else _GENERAL_PLAN_PROMPT
    )

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": prompt_template.format(
                        max_queries=max(2, settings.max_queries_per_plan // 2)
                    ),
                },
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return [
            SearchQuery(
                query_id=f"llm_q{i}",
                text=q["text"],
                priority=q.get("priority", 2),
                rationale=q.get("rationale", ""),
            )
            for i, q in enumerate(data.get("queries", []), start=1)
            if q.get("text", "").strip()
        ]
    except Exception as exc:
        logger.warning(
            "LLM query augmentation failed for '%s', continuing with template queries only: %s",
            subject,
            exc,
        )
        return []

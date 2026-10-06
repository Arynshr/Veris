"""
Research planning (spec section 5), mode-aware: GENERAL mode generates balanced
research-angle queries (overview, recent developments, analysis, counterpoints);
BACKGROUND_CHECK mode generates adverse-media risk-category queries.
"""

from veris.core.config import Settings, get_logger
from veris.core.models import (
    AdverseCategory,
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
    """Drop exact/near-duplicate query text, keeping the first occurrence."""
    seen: set[str] = set()
    deduped: list[SearchQuery] = []
    for q in queries:
        key = _normalize(q.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
    return deduped


def run_planning(
    settings: Settings,
    subject: str,
    research_mode: ResearchMode = ResearchMode.GENERAL,
) -> ResearchPlan:
    plan = _template_plan(subject, research_mode, settings.max_queries_per_plan)
    plan.queries = _dedupe_queries(plan.queries)
    plan.queries.sort(key=lambda q: q.priority)
    plan.queries = plan.queries[: settings.max_queries_per_plan]
    return plan

"""
Subject resolution: when query understanding says the request names
a role/group ("the board of directors of XYZ") rather than concrete names, the
actual names aren't known yet and must be looked up before research can start.
"""
from veris.core.config import Settings, get_logger
from veris.core.models import QueryUnderstanding, ResolvedSubject, SearchQuery, SubjectMode, SubjectType
from veris.intake.ingestion import extract_entities
from veris.retrieval.search import SearchProvider, search_with_fallback

logger = get_logger(__name__)


async def resolve_subjects(
    understanding: QueryUnderstanding, settings: Settings, provider_chain: list[SearchProvider],
) -> list[ResolvedSubject]:
    explicit = [
        ResolvedSubject(name=name.strip(), organization=understanding.organization, resolution_confidence=1.0)
        for name in understanding.explicit_subjects
        if name.strip()
    ]

    if understanding.mode != SubjectMode.ROLE_GROUP or not understanding.role_terms:
        return explicit or [ResolvedSubject(name=understanding.raw_query.strip(), resolution_confidence=1.0)]

    resolved: list[ResolvedSubject] = list(explicit)
    seen_names = {r.name.lower() for r in resolved}

    for role in understanding.role_terms:
        if len(resolved) >= settings.max_resolved_subjects:
            break
        org_part = f" {understanding.organization}" if understanding.organization else ""
        query = SearchQuery(
            query_id=f"resolve_{role.replace(' ', '_')}", text=f"{role}{org_part} names", priority=1,
            rationale="subject resolution",
        )
        try:
            results = await search_with_fallback(query, settings, chain=provider_chain, news_bias=False)
        except Exception as exc: 
            logger.warning("Resolution search failed for role '%s': %s", role, exc)
            continue

        snippet_text = " ".join(f"{r.title}. {r.snippet}" for r in results)
        for entity in extract_entities(snippet_text):
            if entity.label != "PERSON" or len(entity.text.split()) < 2:
                continue  
            key = entity.text.strip().lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            resolved.append(
                ResolvedSubject(
                    name=entity.text.strip(), subject_type=SubjectType.PERSON, role=role,
                    organization=understanding.organization, resolution_confidence=0.6,
                )
            )
            if len(resolved) >= settings.max_resolved_subjects:
                break

    return resolved or explicit

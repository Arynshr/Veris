"""
Claim verification (spec section 10): entity matching, validity, source quality.
Near-duplicate evidence is clustered into one claim so corroboration counts
independent domains, not articles. Produces VERIFIED / SINGLE_SOURCE / CONFLICTED -
never a probability-of-truth score.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from veris.analysis.classification import Classification, classify
from veris.analysis.matching import subject_match_score
from veris.analysis.sources import (
    SourceClassification,
    load_source_policy,
    registrable_domain,
)
from veris.analysis.taxonomy import UNCATEGORIZED, Taxonomy, load_taxonomy
from veris.analysis.temporal import DateInfo, resolve_dates, window_start
from veris.core.config import Settings
from veris.analysis.safety import INSTRUCTION_LIKE
from veris.core.ids import stable_id
from veris.core.models import (
    Claim,
    Entity,
    Evidence,
    LegalStatus,
    ResearchMode,
    Severity,
    VerificationResult,
    VerificationStatus,
)


@dataclass(frozen=True)
class _EvidenceCheck:
    evidence: Evidence
    subject_score: float
    entity_score: float
    source: SourceClassification
    identity_ok: bool


def entity_match_score(text: str, entities: list[Entity]) -> float:
    if not entities:
        return 0.5
    lowered = text.lower()
    matches = sum(1 for e in entities if e.text.lower() in lowered)
    return min(1.0, matches / max(1, len(entities)))


def _cluster_evidence(
    evidence_list: list[Evidence],
    threshold: float,
    text_of: Callable[[Evidence], str] = lambda e: e.claim,
) -> list[list[Evidence]]:
    """Greedy single-pass clustering of near-duplicate claim sentences by TF-IDF
    cosine similarity, strongest evidence first so the representative is stable."""
    ordered = sorted(evidence_list, key=lambda e: (-e.relevance_score, e.evidence_id))
    if len(ordered) < 2:
        return [[e] for e in ordered]
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(
            [text_of(e) for e in ordered]
        )
    except ValueError:
        return [[e] for e in ordered]
    similarity = cosine_similarity(matrix)

    clusters: list[list[int]] = []
    for idx in range(len(ordered)):
        for members in clusters:
            if similarity[members[0], idx] >= threshold:
                members.append(idx)
                break
        else:
            clusters.append([idx])
    return [[ordered[i] for i in members] for members in clusters]


@dataclass(frozen=True)
class _Item:
    evidence: Evidence
    cls: Classification
    dates: DateInfo


_TERMINAL = (LegalStatus.CONVICTED, LegalStatus.DISMISSED_ACQUITTED)
_NEUTRAL = Classification(
    categories=[UNCATEGORIZED], legal_status=LegalStatus.UNKNOWN, severity=Severity.LOW
)


def _published_date(raw: str | None) -> date | None:
    try:
        return datetime.fromisoformat(raw).date() if raw else None
    except ValueError:
        return None


def _gap(item: _Item, group: list[_Item]) -> int:
    day = item.dates.event_date
    days = [g.dates.event_date for g in group if g.dates.event_date]
    return min((abs((day - d).days) for d in days), default=0) if day else 0


def _split_terminal(items: list[_Item]) -> list[list[_Item]]:
    """Opposing terminal statuses become separate sides; other items join the
    side nearest in time (first side on ties/undated)."""
    sides = [[i for i in items if i.cls.legal_status == s] for s in _TERMINAL]
    if not all(sides):
        return [items]
    for item in items:
        if item.cls.legal_status not in _TERMINAL:
            min(sides, key=lambda g: _gap(item, g)).append(item)
    return sides


def _make_claim(
    items: list[_Item],
    subject: str,
    taxonomy: Taxonomy,
    settings: Settings,
    as_of: date,
    adverse: bool,
) -> Claim:
    latest = max(
        items,
        key=lambda i: (
            i.dates.event_date or date.min,
            taxonomy.status_rank(i.cls.legal_status),
            i.evidence.evidence_id,
        ),
    )
    categories = [
        c for c in taxonomy.category_order if any(c in i.cls.categories for i in items)
    ] or [UNCATEGORIZED]
    status = latest.cls.legal_status
    event = latest.dates.event_date
    return Claim(
        claim_id=f"claim_{stable_id(*sorted(i.evidence.evidence_id for i in items), length=10)}",
        text=next((i for i in items if not i.evidence.flags), items[0]).evidence.claim,
        subject=subject,
        evidence_ids=[i.evidence.evidence_id for i in items],
        categories=categories,
        legal_status=status,
        severity=taxonomy.severity_for(status, categories),
        event_date=event,
        published_date=latest.dates.published_date,
        date_source=latest.dates.date_source,
        undated=event is None,
        flags=[INSTRUCTION_LIKE] if all(i.evidence.flags for i in items) else [],
        out_of_window=adverse
        and event is not None
        and event < window_start(as_of, settings.lookback_years),
    )


def _build_claims(
    evidence_list: list[Evidence],
    subject: str,
    taxonomy: Taxonomy,
    settings: Settings,
    as_of: date,
    adverse: bool,
) -> tuple[list[Claim], dict[str, list[str]]]:
    """One claim per dedup cluster (traceability per spec section 9). Status
    resolves to the latest-dated evidence; a cluster whose terminal statuses
    disagree splits into one claim per side, linked via the conflicts map."""
    claims: list[Claim] = []
    conflicts: dict[str, list[str]] = {}
    text_of = (lambda e: taxonomy.strip_cues(e.claim)) if adverse else (lambda e: e.claim)
    for cluster in _cluster_evidence(
        evidence_list, settings.claim_dedup_similarity, text_of
    ):
        items = [
            _Item(
                e,
                classify(e.supporting_passage, taxonomy, claim=e.claim)
                if adverse
                else _NEUTRAL,
                resolve_dates(
                    e.supporting_passage,
                    e.source_metadata.url,
                    _published_date(e.source_metadata.published_at),
                    as_of,
                ),
            )
            for e in cluster
        ]
        built = [
            _make_claim(g, subject, taxonomy, settings, as_of, adverse)
            for g in (_split_terminal(items) if adverse else [items])
        ]
        if len(built) > 1:
            for claim in built:
                conflicts[claim.claim_id] = [
                    o.claim_id for o in built if o.claim_id != claim.claim_id
                ]
        claims.extend(built)
    return claims, conflicts


def run_verification(
    evidence_list: list[Evidence],
    entities: list[Entity],
    subject: str,
    settings: Settings,
    research_mode: ResearchMode = ResearchMode.GENERAL,
) -> tuple[list[Claim], list[VerificationResult]]:
    policy = load_source_policy(settings.source_tiers_path)
    taxonomy = load_taxonomy(settings.taxonomy_path)
    is_background_check = research_mode == ResearchMode.BACKGROUND_CHECK
    claims, conflicts = _build_claims(
        evidence_list,
        subject,
        taxonomy,
        settings,
        settings.as_of_date or date.today(),
        is_background_check,
    )
    evidence_by_id = {e.evidence_id: e for e in evidence_list}
    results: list[VerificationResult] = []

    off_topic_reason = (
        "subject not clearly named in this claim - possible misattribution/namesake risk"
        if is_background_check
        else "claim doesn't clearly reference the research subject/topic"
    )

    for claim in claims:
        checks = []
        for eid in claim.evidence_ids:
            passage = evidence_by_id[eid].supporting_passage
            subject_score = subject_match_score(passage, subject)
            entity_score = entity_match_score(passage, entities)
            checks.append(
                _EvidenceCheck(
                    evidence=evidence_by_id[eid],
                    subject_score=subject_score,
                    entity_score=entity_score,
                    source=policy.classify(evidence_by_id[eid].source_metadata.source),
                    identity_ok=subject_score >= settings.min_subject_name_overlap
                    and (
                        not is_background_check
                        or not entities
                        or entity_score >= settings.min_entity_match_score
                    ),
                )
            )
        matching = [c for c in checks if c.identity_ok]
        corroborating_domains = {
            registrable_domain(c.evidence.source_metadata.source)
            for c in matching
            if c.source.quality_score >= settings.min_corroboration_quality
        }
        has_top_tier_source = any(
            c.source.tier <= settings.verified_alone_max_tier for c in matching
        )

        reasons: list[str] = []
        if not matching:
            status = VerificationStatus.CONFLICTED
            reasons.append(off_topic_reason)
        elif (
            len(corroborating_domains) >= settings.min_independent_domains
            or has_top_tier_source
        ):
            status = VerificationStatus.VERIFIED
            reasons.append(
                "top-tier source"
                if has_top_tier_source
                else f"corroborated by {len(corroborating_domains)} independent domains"
            )
        else:
            status = VerificationStatus.SINGLE_SOURCE
            reasons.append(
                "not corroborated by an independent domain or top-tier source"
            )
        if claim.claim_id in conflicts:
            status = VerificationStatus.CONFLICTED
            reasons.append("terminal legal statuses disagree (convicted vs dismissed_acquitted)")
        scored = matching or checks
        results.append(
            VerificationResult(
                claim_id=claim.claim_id,
                status=status,
                subject_match_score=round(max(c.subject_score for c in scored), 2),
                entity_match_score=round(max(c.entity_score for c in scored), 2),
                source_quality_score=round(
                    max(c.source.quality_score for c in scored), 2
                ),
                independent_domains=len(corroborating_domains),
                reasons=reasons,
                conflicts_with=conflicts.get(claim.claim_id, []),
            )
        )
    return claims, results

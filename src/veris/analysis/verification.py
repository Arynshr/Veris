"""
Claim verification (spec section 10): entity matching, validity, source quality.
Near-duplicate evidence is clustered into one claim so corroboration counts
independent domains, not articles. Produces VERIFIED / SINGLE_SOURCE / CONFLICTED -
never a probability-of-truth score.
"""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from veris.analysis.matching import subject_match_score
from veris.analysis.sources import (
    SourceClassification,
    load_source_policy,
    registrable_domain,
)
from veris.core.config import Settings
from veris.core.ids import stable_id
from veris.core.models import (
    Claim,
    Entity,
    Evidence,
    ResearchMode,
    VerificationResult,
    VerificationStatus,
)


@dataclass(frozen=True)
class _EvidenceCheck:
    evidence: Evidence
    subject_score: float
    entity_score: float
    source: SourceClassification
    indentity_ok: bool


def entity_match_score(text: str, entities: list[Entity]) -> float:
    if not entities:
        return 0.5
    lowered = text.lower()
    matches = sum(1 for e in entities if e.text.lower() in lowered)
    return min(1.0, matches / max(1, len(entities)))


def _cluster_evidence(
    evidence_list: list[Evidence], threshold: float
) -> list[list[Evidence]]:
    """Greedy single-pass clustering of near-duplicate claim sentences by TF-IDF
    cosine similarity, strongest evidence first so the representative is stable."""
    ordered = sorted(evidence_list, key=lambda e: (-e.relevance_score, e.evidence_id))
    if len(ordered) < 2:
        return [[e] for e in ordered]
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(
            [e.claim for e in ordered]
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


def _build_claims(
    evidence_list: list[Evidence], subject: str, dedup_similarity: float
) -> list[Claim]:
    """One claim per cluster of near-duplicate evidence (traceability per spec
    section 9); the cluster representative supplies the claim text."""
    return [
        Claim(
            claim_id=f"claim_{stable_id(*sorted(e.evidence_id for e in cluster), length=10)}",
            text=cluster[0].claim,
            subject=subject,
            evidence_ids=[e.evidence_id for e in cluster],
        )
        for cluster in _cluster_evidence(evidence_list, dedup_similarity)
    ]


def run_verification(
    evidence_list: list[Evidence],
    entities: list[Entity],
    subject: str,
    settings: Settings,
    research_mode: ResearchMode = ResearchMode.GENERAL,
) -> tuple[list[Claim], list[VerificationResult]]:
    policy = load_source_policy(settings.source_tiers_path)
    claims = _build_claims(evidence_list, subject, settings.claim_dedup_similarity)
    evidence_by_id = {e.evidence_id: e for e in evidence_list}
    is_background_check = research_mode == ResearchMode.BACKGROUND_CHECK
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
                    indentity_ok=subject_score >= settings.min_subject_name_overlap
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
        scored = matching or checks
        results.append(
            VerificationResult(
                claim_id=claim.claim_id,
                status=status,
                subject_match_score=round(max(c.subject_score for c in scored), 2),
                entity_match_score=round(max(c.entity_score for c in scored), 2),
                source_quality_score=round(
                    max(c.source_quality_score for c in scored), 2
                ),
                independent_domains=len(corroborating_domains),
                reasons=reasons,
            )
        )
    return claims, results

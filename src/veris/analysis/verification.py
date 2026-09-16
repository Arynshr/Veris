"""
Claim verification (spec section 10): entity matching, validity, source quality.
Produces VERIFIED / CONFLICTED only - never a probability-of-truth score.
"""
import uuid

from veris.core.config import Settings
from veris.core.models import Claim, Entity, Evidence, ResearchMode, VerificationResult, VerificationStatus

_SOURCE_QUALITY_THRESHOLD = 0.5
_MIN_CLAIM_LENGTH = 30

_HIGH_QUALITY_TLDS = (".gov", ".edu", ".int")
_HIGH_QUALITY_DOMAINS = ("reuters.com", "apnews.com", "bbc.com", "bloomberg.com", "wsj.com", "ft.com", "nature.com")
_LOW_QUALITY_MARKERS = ("blogspot.", "medium.com/@", "wordpress.com")


def _score_source(domain: str) -> float:
    domain = domain.lower()
    if any(domain.endswith(tld) for tld in _HIGH_QUALITY_TLDS):
        return 1.0
    if any(hq in domain for hq in _HIGH_QUALITY_DOMAINS):
        return 0.9
    if any(marker in domain for marker in _LOW_QUALITY_MARKERS):
        return 0.3
    return 0.6


def _subject_match_score(claim_text: str, subject: str) -> float:
    """Fraction of subject/topic name tokens present in the claim."""
    tokens = [t for t in subject.lower().split() if len(t) > 1]
    if not tokens:
        return 0.0
    lowered = claim_text.lower()
    return sum(1 for t in tokens if t in lowered) / len(tokens)


def _entity_match_score(claim_text: str, entities: list[Entity]) -> float:
    if not entities:
        return 0.5
    lowered = claim_text.lower()
    matches = sum(1 for e in entities if e.text.lower() in lowered)
    return min(1.0, matches / max(1, len(entities)))


def _build_claims(evidence_list: list[Evidence], subject: str) -> list[Claim]:
    """v0.1: one candidate claim per evidence passage (traceability per spec section 9)."""
    return [
        Claim(claim_id=f"claim_{uuid.uuid4().hex[:10]}", text=ev.claim, subject=subject, evidence_ids=[ev.evidence_id])
        for ev in evidence_list
    ]


def run_verification(
    evidence_list: list[Evidence],
    entities: list[Entity],
    subject: str,
    settings: Settings,
    research_mode: ResearchMode = ResearchMode.GENERAL,
) -> tuple[list[Claim], list[VerificationResult]]:
    claims = _build_claims(evidence_list, subject)
    evidence_by_id = {e.evidence_id: e for e in evidence_list}
    results: list[VerificationResult] = []

    off_topic_reason = (
        "subject not clearly named in this claim - possible misattribution/namesake risk"
        if research_mode == ResearchMode.BACKGROUND_CHECK
        else "claim doesn't clearly reference the research subject/topic"
    )

    for claim in claims:
        related_evidence = [evidence_by_id[eid] for eid in claim.evidence_ids if eid in evidence_by_id]

        subject_score = _subject_match_score(claim.text, subject)
        entity_score = _entity_match_score(claim.text, entities)
        quality_scores = [_score_source(e.source_metadata.source) for e in related_evidence] or [0.0]
        quality_score = sum(quality_scores) / len(quality_scores)
        valid = len(claim.text.strip()) >= _MIN_CLAIM_LENGTH

        reasons: list[str] = []
        if not valid:
            reasons.append("claim text too short / low information content")
        if subject_score < settings.min_subject_name_overlap:
            reasons.append(off_topic_reason)
        if quality_score < _SOURCE_QUALITY_THRESHOLD:
            reasons.append("source quality below threshold")
        if not related_evidence:
            reasons.append("no supporting evidence found")

        status = (
            VerificationStatus.VERIFIED
            if valid and subject_score >= settings.min_subject_name_overlap and quality_score >= _SOURCE_QUALITY_THRESHOLD and related_evidence
            else VerificationStatus.CONFLICTED
        )
        if status == VerificationStatus.VERIFIED:
            reasons = ["subject clearly referenced, source quality and validity checks passed"]

        results.append(
            VerificationResult(
                claim_id=claim.claim_id, status=status,
                subject_match_score=round(subject_score, 2), entity_match_score=round(entity_score, 2),
                source_quality_score=round(quality_score, 2), reasons=reasons,
            )
        )
    return claims, results

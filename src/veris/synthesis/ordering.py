"""Fixed claim ordering: severity desc, event date desc (undated last), relevance desc, claim_id."""

from veris.core.models import Claim, Evidence, Severity

_SEVERITY_RANK = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}


def claim_relevance(claim: Claim, evidence: dict[str, Evidence]) -> float:
    return max((evidence[i].relevance_score for i in claim.evidence_ids if i in evidence), default=0.0)


def claim_sort_key(claim: Claim, relevance: float) -> tuple:
    day = claim.event_date
    return (_SEVERITY_RANK[claim.severity], -day.toordinal() if day else 1, -relevance, claim.claim_id)

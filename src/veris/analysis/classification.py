"""Deterministic claim classification: (passage, taxonomy) -> categories, status, severity."""

import re

from pydantic import BaseModel

from veris.analysis.taxonomy import UNCATEGORIZED, Taxonomy
from veris.core.models import LegalStatus, Severity

_CLAUSE_BREAK_RE = re.compile(r"[.;:,!?]")
_TOKEN_RE = re.compile(r"[\w']+")

class Classification(BaseModel):
    categories: list[str]
    legal_status: LegalStatus
    severity: Severity
    
def _negated(text:str, start: int, taxonomy: Taxonomy) -> bool:
    prefix = _CLAUSE_BREAK_RE.split(text[max(0, start - 80): start])[-1]
    tokens = _TOKEN_RE.findall(prefix.lower().replace("\u2019", "'"))
    return any(t in taxonomy.negators for t in tokens[-taxonomy.negation_window_tokens :])

def resolve_status(passage: str, taxonomy: Taxonomy, adverse: bool) -> LegalStatus:
    if not adverse:
        return LegalStatus.UNKNOWN
    text = passage.replace("\u2019", "'")
    for status in taxonomy.status_precedence:
        if any(not _negated(text, m.start(), taxonomy) for m in taxonomy.cue_matches(status, text)):
            return status
    return LegalStatus.ALLEGED

def classify(passage: str, taxonomy: Taxonomy, claim: str | None = None) -> Classification:
    """Categories come from the claim sentance when given (precision); status cues
    are read from the whole passage window"""
    categories = taxonomy.match_categories(claim or passage)
    status = resolve_status(passage, taxonomy, adverse=bool(categories))
    return Classification(
        categories=categories or [UNCATEGORIZED],
        legal_status=status,
        severity=taxonomy.severity_for(status, categories),
    )

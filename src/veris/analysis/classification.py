"""Deterministic claim classification: (passage, taxonomy) -> categories, status, severity."""

import re

from pydantic import BaseModel

from veris.analysis.taxonomy import UNCATEGORIZED, Taxonomy
from veris.core.models import LegalStatus, Severity

_CLAUSE_BREAK_RE = re.compile(r"[.;:,!?]")
_TOKEN_RE = re.compile(r"[\w']+")

Span = tuple[int, int]


class Classification(BaseModel):
    categories: list[str]
    legal_status: LegalStatus
    severity: Severity


def _negated(text: str, start: int, taxonomy: Taxonomy) -> bool:
    prefix = _CLAUSE_BREAK_RE.split(text[max(0, start - 80) : start])[-1]
    tokens = _TOKEN_RE.findall(prefix.lower().replace("\u2019", "'"))
    return any(t in taxonomy.negators for t in tokens[-taxonomy.negation_window_tokens :])


def _near_action(cue: re.Match[str], text: str, spans: list[Span], window: int) -> bool:
    for start, end in spans:
        if cue.end() <= start:
            gap = text[cue.end() : start]
        elif cue.start() >= end:
            gap = text[end : cue.start()]
        else:
            return True
        if len(gap) <= window and ";" not in gap:
            return True
    return False


def resolve_status(
    passage: str, taxonomy: Taxonomy, adverse: bool, spans: list[Span] | None = None
) -> LegalStatus:
    """`spans` are the matched action offsets; when given, cues must sit near one."""
    if not adverse:
        return LegalStatus.UNKNOWN
    text = passage.replace("\u2019", "'")
    for status in taxonomy.status_precedence:
        for cue in taxonomy.cue_matches(status, text):
            if _negated(text, cue.start(), taxonomy):
                continue
            if spans is None or _near_action(cue, text, spans, taxonomy.status_window_chars):
                return status
    return LegalStatus.ALLEGED


def classify(passage: str, taxonomy: Taxonomy, claim: str | None = None) -> Classification:
    """Categories come from the claim sentence when given; status cues are read
    from the passage, near the matched action."""
    base = claim or passage
    categories = taxonomy.match_categories(base)
    pos = passage.find(base)
    spans = taxonomy.category_spans(base if pos >= 0 else passage)
    if pos > 0:
        spans = [(s + pos, e + pos) for s, e in spans]
    status = resolve_status(passage, taxonomy, adverse=bool(categories), spans=spans or None)
    return Classification(
        categories=categories or [UNCATEGORIZED],
        legal_status=status,
        severity=taxonomy.severity_for(status, categories),
    )

"""Text-matching helpers shared by evidence mapping and verification."""

import re

_HONORIFICS = frozenset(
    "mr mrs ms miss mx dr prof sir shri sri smt the a an and or of by for with from to in on at as".split()
)
_WORD_RE = re.compile(r"\w+")


def subject_match_score(text: str, subject: str) -> float:
    """Fraction of subject/topic name tokens present in the text."""
    tokens = [t for t in subject.lower().split() if len(t) > 1]
    if not tokens:
        return 0.0
    lowered = text.lower()
    return sum(1 for t in tokens if t in lowered) / len(tokens)


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.casefold()))


def name_mentioned(text: str, subject: str) -> bool:
    """Exact normalized full-name match on word boundaries (no fuzzy/partial credit)."""
    name = _normalize(subject)
    return bool(name) and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", _normalize(text)) is not None


def namesake_mentioned(text: str, subject: str) -> bool:
    """True when the text names someone else sharing the subject's surname
    (e.g. "Nikhil Kamath" for subject "Nithin Kamath"). Multi-token subjects only."""
    tokens = _normalize(subject).split()
    if len(tokens) < 2:
        return False
    known = set(tokens) | _HONORIFICS
    for match in re.finditer(rf"\b([^\W\d_]+)\s+{re.escape(tokens[-1])}\b", text, re.I):
        first = match.group(1)
        if first[0].isupper() and first.casefold() not in known:
            return True
    return False

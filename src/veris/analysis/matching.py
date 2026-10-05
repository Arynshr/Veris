"""Text-matching helpers shared by evidence mapping and verification."""


def subject_match_score(text: str, subject: str) -> float:
    """Fraction of subject/topic name tokens present in the text."""
    tokens = [t for t in subject.lower().split() if len(t) > 1]
    if not tokens:
        return 0.0
    lowered = text.lower()
    return sum(1 for t in tokens if t in lowered) / len(tokens)

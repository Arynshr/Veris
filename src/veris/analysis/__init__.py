"""Turns retrieved documents into verified, traceable claims: evidence mapping
(TF-IDF relevance ranking) -> claim verification (subject-match + source quality)."""
from veris.analysis.evidence import run_evidence_mapping
from veris.analysis.verification import run_verification

__all__ = ["run_evidence_mapping", "run_verification"]

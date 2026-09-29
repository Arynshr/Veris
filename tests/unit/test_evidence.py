import hashlib

from veris.analysis.evidence import run_evidence_mapping
from veris.core.models import AdverseCategory, Document, ResearchMode


def _doc(content: str, url: str = "https://a.example/x") -> Document:
    canonical = url
    return Document(
        document_id=hashlib.sha1(url.encode()).hexdigest()[:16],
        url=url,
        canonical_url=canonical,
        title="Test doc",
        content=content,
        source="a.example",
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )


def test_general_mode_caps_evidence_per_document():
    sentences = [
        f"Solid-state batteries improve energy density in point number {i} of this overview."
        for i in range(20)
    ]
    doc = _doc(" ".join(sentences))
    evidence = run_evidence_mapping([doc], subject="solid-state batteries", entities=[])
    assert 0 < len(evidence) <= 5  # _MAX_EVIDENCE_PER_DOCUMENT


def test_general_mode_filters_low_relevance_sentences():
    doc = _doc(
        "Solid-state batteries are a major focus of current battery research and development efforts today. "
        "The weather in Paris was mild and sunny for most of the entire week according to reports."
    )
    evidence = run_evidence_mapping([doc], subject="solid-state batteries", entities=[])
    assert all(
        "solid-state" in e.claim.lower()
        or "battery" in e.claim.lower()
        or "batteries" in e.claim.lower()
        for e in evidence
    )


def test_background_check_categorizes_adverse_sentences():
    doc = _doc(
        "Jane Doe was named in a lawsuit filed by a former business partner last year. "
        "Jane Doe also attended a technology conference in the spring of this year."
    )
    evidence = run_evidence_mapping(
        [doc],
        subject="Jane Doe",
        entities=[],
        research_mode=ResearchMode.BACKGROUND_CHECK,
    )
    categories = {e.category for e in evidence}
    assert AdverseCategory.LEGAL in categories


def test_no_documents_returns_no_evidence():
    assert run_evidence_mapping([], subject="anything", entities=[]) == []

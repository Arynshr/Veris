from veris.analysis.verification import run_verification
from veris.core.models import (
    AdverseCategory,
    Evidence,
    SourceMetadata,
    VerificationStatus,
)


def _ev(
    claim: str, url: str = "https://reuters.com/a", source: str = "reuters.com"
) -> Evidence:
    return Evidence(
        evidence_id="e1",
        document_id="d1",
        claim=claim,
        supporting_passage=claim,
        category=AdverseCategory.GENERAL,
        relevance_score=0.5,
        source_metadata=SourceMetadata(url=url, title="t", source=source),
    )


def test_claim_verified_when_subject_named_and_source_high_quality(settings_factory):
    settings = settings_factory()
    evidence = [
        _ev(
            "Acme reported record revenue growth in its latest quarterly earnings statement."
        )
    ]
    claims, results = run_verification(
        evidence, entities=[], subject="Acme", settings=settings
    )
    assert results[0].status == VerificationStatus.VERIFIED


def test_claim_conflicted_when_subject_not_named(settings_factory):
    settings = settings_factory()
    evidence = [
        _ev("A completely unrelated company reported record revenue this quarter.")
    ]
    claims, results = run_verification(
        evidence, entities=[], subject="Acme", settings=settings
    )
    assert results[0].status == VerificationStatus.CONFLICTED
    assert any(
        "subject" in r or "doesn't clearly reference" in r for r in results[0].reasons
    )


def test_claim_conflicted_when_source_quality_low(settings_factory):
    settings = settings_factory()
    evidence = [
        _ev(
            "Acme was mentioned in this blog post about corporate news today.",
            source="someuser.blogspot.com",
        )
    ]
    claims, results = run_verification(
        evidence, entities=[], subject="Acme", settings=settings
    )
    assert results[0].status == VerificationStatus.CONFLICTED


def test_claim_conflicted_when_too_short(settings_factory):
    settings = settings_factory()
    evidence = [_ev("Acme news.")]
    claims, results = run_verification(
        evidence, entities=[], subject="Acme", settings=settings
    )
    assert results[0].status == VerificationStatus.CONFLICTED


def test_no_evidence_yields_no_claims(settings_factory):
    claims, results = run_verification(
        [], entities=[], subject="Acme", settings=settings_factory()
    )
    assert claims == [] and results == []

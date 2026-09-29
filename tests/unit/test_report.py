import pytest
from conftest import fake_groq_factory

from veris.core.models import (
    AdverseCategory,
    Claim,
    Evidence,
    ResolvedSubject,
    SourceMetadata,
    VerificationResult,
    VerificationStatus,
)
from veris.synthesis.report import _MAX_CLAIM_CHARS_IN_PROMPT, _truncate, run_synthesis


def _verified_pair(
    i: int, subject: str, text: str | None = None
) -> tuple[Claim, Evidence, VerificationResult]:
    ev = Evidence(
        evidence_id=f"e{i}",
        document_id=f"d{i}",
        claim=text
        or f"Finding number {i} about {subject} reported in coverage from a news outlet.",
        supporting_passage="...",
        category=AdverseCategory.GENERAL,
        relevance_score=1.0 - i * 0.001,
        source_metadata=SourceMetadata(
            url=f"https://news.example/{i}", title=f"Article {i}", source="news.example"
        ),
    )
    claim = Claim(
        claim_id=f"c{i}", text=ev.claim, subject=subject, evidence_ids=[ev.evidence_id]
    )
    verification = VerificationResult(
        claim_id=claim.claim_id,
        status=VerificationStatus.VERIFIED,
        subject_match_score=1.0,
        entity_match_score=1.0,
        source_quality_score=1.0,
    )
    return claim, ev, verification


def test_truncate_leaves_short_text_unchanged():
    assert _truncate("short claim") == "short claim"


def test_truncate_shortens_long_text_with_ellipsis():
    long_text = "x" * 500
    result = _truncate(long_text)
    assert len(result) <= _MAX_CLAIM_CHARS_IN_PROMPT
    assert result.endswith("…")


@pytest.mark.asyncio
async def test_synthesis_caps_evidence_per_subject_in_prompt(
    settings_factory, monkeypatch
):
    """Direct regression test for the 413 'request too large' error: even
    with far more verified claims than fit comfortably in one Groq request
    (e.g. from 30 documents x 5 evidence each), the prompt sent to the model
    must stay bounded by max_synthesis_evidence_per_subject."""
    subject = ResolvedSubject(name="solid-state batteries", resolution_confidence=1.0)
    claims, evidence, verifications = [], [], []
    for i in range(150):  # far more than any reasonable per-subject cap
        c, e, v = _verified_pair(i, subject.name)
        claims.append(c)
        evidence.append(e)
        verifications.append(v)

    captured = {}

    def content_fn(**kwargs):
        captured["user_msg"] = kwargs["messages"][1]["content"]
        return "# Research Report\nok"

    monkeypatch.setattr("veris.synthesis.report.Groq", fake_groq_factory(content_fn))
    settings = settings_factory(
        groq_api_key="fake", max_synthesis_evidence_per_subject=15
    )

    await run_synthesis([subject], claims, verifications, evidence, settings)

    finding_lines = [l for l in captured["user_msg"].splitlines() if l.startswith("- ")]
    assert len(finding_lines) <= 15


@pytest.mark.asyncio
async def test_synthesis_truncates_long_claim_text_in_prompt(
    settings_factory, monkeypatch
):
    subject = ResolvedSubject(name="Acme", resolution_confidence=1.0)
    long_text = "Acme did something notable. " * 40  # well over the truncation limit
    c, e, v = _verified_pair(1, "Acme", text=long_text)

    captured = {}

    def content_fn(**kwargs):
        captured["user_msg"] = kwargs["messages"][1]["content"]
        return "# Research Report\nok"

    monkeypatch.setattr("veris.synthesis.report.Groq", fake_groq_factory(content_fn))
    settings = settings_factory(groq_api_key="fake")

    await run_synthesis([subject], [c], [v], [e], settings)
    assert long_text not in captured["user_msg"]


@pytest.mark.asyncio
async def test_synthesis_without_groq_key_uses_deterministic_fallback(settings_factory):
    subject = ResolvedSubject(name="Acme", resolution_confidence=1.0)
    c, e, v = _verified_pair(1, "Acme")
    report = await run_synthesis([subject], [c], [v], [e], settings_factory())
    assert "Acme" in report
    assert "Sources" in report


@pytest.mark.asyncio
async def test_synthesis_no_verified_claims_reports_no_findings(settings_factory):
    """With a Groq key configured but zero verified claims, synthesis
    short-circuits before ever calling the model (nothing to summarize)."""
    subject = ResolvedSubject(name="Acme", resolution_confidence=1.0)
    report = await run_synthesis(
        [subject], [], [], [], settings_factory(groq_api_key="fake")
    )
    assert "No claims passed verification" in report


@pytest.mark.asyncio
async def test_synthesis_no_verified_claims_without_groq_key_uses_fallback_text(
    settings_factory,
):
    subject = ResolvedSubject(name="Acme", resolution_confidence=1.0)
    report = await run_synthesis([subject], [], [], [], settings_factory())
    assert "No findings passed verification" in report

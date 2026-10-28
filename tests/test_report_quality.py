"""Report quality: passage cleaning, namesake gate, URL dedup, reference registry, bounded findings."""

import re
from datetime import date

from veris.analysis.cleaning import NOISE, clean_claim, clean_text, load_cleaning, truncate
from veris.analysis.evidence import IDENTITY_MISMATCH, run_evidence_mapping
from veris.analysis.matching import name_mentioned, namesake_mentioned
from veris.core.config import PROJECT_ROOT
from veris.core.documents import build_document, canonicalize_url
from veris.core.models import (
    Claim,
    Evidence,
    ResearchMode,
    ResearchRun,
    ResolvedSubject,
    Severity,
    SourceMetadata,
    SubjectMode,
    VerificationResult,
    VerificationStatus,
)
from veris.synthesis import build_report, render_markdown
from veris.testing import make_settings

CLEANING = load_cleaning(PROJECT_ROOT / "config" / "cleaning.yaml")
SUBJECT = "Nithin Kamath"
NOISY = (
    "![logo](https://cdn.example.com/a.png) [Home](https://x.com/) | [About](https://x.com/a) | "
    "[Contact](https://x.com/c) https://x.com/ads"
)


# --- cleaning -----------------------------------------------------------------


def test_clean_text_strips_markup_urls_and_boilerplate():
    raw = (
        "![img](https://a.com/i.png) Zerodha founder [Nithin Kamath](https://en.wikipedia.org/wiki/N) "
        "spoke at www.example.com/event on Monday.\nAccept all cookies\n"
    )
    out = clean_text(raw, CLEANING)
    assert out == "Zerodha founder Nithin Kamath spoke at on Monday."
    assert "http" not in out and "![" not in out and "cookies" not in out


def test_clean_claim_flags_navigation_noise_but_keeps_real_sentences():
    assert clean_claim(NOISY, CLEANING).noisy
    real = "Zerodha founder [Nithin Kamath](https://en.wikipedia.org/wiki/Nithin_Kamath) said the firm turned profitable in 2024."
    cleaned = clean_claim(real, CLEANING)
    assert not cleaned.noisy and "http" not in cleaned.snippet


def test_truncate_is_bounded_and_sentence_aligned():
    text = "First sentence is here. " + "word " * 100
    out = truncate(text, 60)
    assert len(out) <= 60 and out.startswith("First sentence is here.")


# --- identity -----------------------------------------------------------------


def test_exact_name_and_namesake_detection():
    assert name_mentioned("Nithin Kamath's firm grew.", SUBJECT)
    assert not name_mentioned("Nikhil Kamath's firm grew.", SUBJECT)
    assert namesake_mentioned("Investor Nikhil Kamath said so.", SUBJECT)
    assert not namesake_mentioned("Mr Kamath said so. Nithin Kamath agreed.", SUBJECT)
    assert not namesake_mentioned("Nikhil Kamath", "Zerodha")  # single-token subjects are skipped


def _map(content: str, url: str = "https://news.example.com/a"):
    doc = build_document(url, "t", content)
    return run_evidence_mapping([doc], SUBJECT, [], make_settings(), ResearchMode.GENERAL)


def test_namesake_only_document_is_rejected():
    text = "Nikhil Kamath is an investor who spoke about markets and startups at a large conference in Mumbai."
    assert _map(text) == []


def test_namesake_passage_in_mixed_document_is_flagged_identity_mismatch():
    text = (
        "Nithin Kamath is the founder of Zerodha, a large discount broker in India. "
        + "Filler sentence about unrelated market conditions and weather patterns today. " * 4
        + "Nikhil Kamath is an investor who backs startups across consumer and fintech sectors."
    )
    flagged = [e for e in _map(text) if IDENTITY_MISMATCH in e.flags]
    assert flagged and all("Nikhil" in e.claim for e in flagged)


def test_noise_sentences_are_flagged_and_never_displace_clean_ones():
    text = f"{NOISY} Nithin Kamath. Zerodha founder Nithin Kamath said the broker turned profitable this year."
    by_flag = {bool(e.flags): e for e in _map(text)}
    assert NOISE in by_flag.get(True, Evidence.model_construct(flags=[NOISE])).flags
    assert by_flag[False].snippet.startswith("Zerodha founder")


# --- URL dedup ----------------------------------------------------------------


def test_url_variants_share_one_canonical_url():
    variants = [
        "https://www.site.com/news/story/",
        "https://site.com/news/story?utm_source=x#top",
        "https://amp.site.com/news/story",
        "https://site.com/amp/news/story",
    ]
    assert len({canonicalize_url(u) for u in variants}) == 1


# --- report -------------------------------------------------------------------

AS_OF = date(2026, 10, 28)


def _fixture(n_claims: int = 4, urls: list[str] | None = None, categories=None):
    urls = urls or [f"https://site{i}.com/story" for i in range(n_claims)]
    evidence, claims, verdicts = [], [], []
    for i in range(n_claims):
        url = urls[i % len(urls)]
        passage = f"Nithin Kamath faced matter number {i}. ![i](https://cdn.x.com/i.png) See [more](https://x.com/m)."
        ev = Evidence(
            evidence_id=f"e{i}", document_id=f"d{i}", claim=f"Nithin Kamath faced matter number {i}.",
            snippet=f"Nithin Kamath faced matter number {i}.", supporting_passage=passage, span_start=0, span_end=10,
            relevance_score=0.9 - i * 0.1,
            source_metadata=SourceMetadata(url=url, title=f"Story {i}", source=url.split("/")[2]),
        )
        claims.append(
            Claim(claim_id=f"c{i}", text=ev.snippet, subject=SUBJECT, evidence_ids=[ev.evidence_id],
                  categories=categories or ["criminal"], severity=Severity.HIGH, event_date=date(2026, 1, i + 1))
        )
        verdicts.append(VerificationResult(claim_id=f"c{i}", status=VerificationStatus.SINGLE_SOURCE,
                                           subject_match_score=1, entity_match_score=1, source_quality_score=0.5))
        evidence.append(ev)
    return claims, verdicts, evidence


def _report(claims, verdicts, evidence, **overrides):
    settings = make_settings(as_of_date=AS_OF, **overrides)
    run = ResearchRun(run_id="r1", query="q", mode=SubjectMode.SINGLE, research_mode=ResearchMode.BACKGROUND_CHECK,
                      subjects=[SUBJECT], output_path="", config_hash="abc")
    report = build_report(run, [ResolvedSubject(name=SUBJECT)], claims, verdicts, evidence,
                          {e.document_id: "2026-10-28T00:00:00" for e in evidence}, settings, None, ["criminal"])
    return report, render_markdown(report, settings)


def test_same_article_via_url_variants_is_one_reference():
    urls = ["https://www.site.com/a/", "https://site.com/a?utm_source=x", "https://amp.site.com/a"]
    report, md = _report(*_fixture(3, urls))
    assert len(report.references) == 1 and report.references[0].ref_id == "R1"
    assert all(f.ref_ids == ["R1"] for f in report.subjects[0].findings)


def test_identical_claim_text_is_merged_into_one_finding():
    claims, verdicts, evidence = _fixture(2)
    for c in claims:
        c.text = "Same sentence repeated across pages."
    report, _ = _report(claims, verdicts, evidence)
    findings = report.subjects[0].findings
    assert len(findings) == 1 and findings[0].merged_claim_ids and len(findings[0].ref_ids) == 2
    primary = findings[0].claim_id
    assert {primary, *findings[0].merged_claim_ids} == {"c0", "c1"}
    assert [r.claim_id for r in report.appendix] == [primary, primary]  # merged evidence stays traceable


def test_main_findings_are_capped_and_overflow_goes_to_appendix_a():
    report, md = _report(*_fixture(4), max_findings_per_subject=2)
    section = report.subjects[0]
    assert len(section.findings) == 2 and len(section.secondary) == 2
    assert {f.secondary_reason for f in section.secondary} == {"beyond top findings"}
    assert "Appendix A" in md and report.coverage.claims_secondary == 2


def test_low_relevance_findings_are_secondary():
    claims, verdicts, evidence = _fixture(2)
    evidence[1].relevance_score = 0.0
    report, _ = _report(claims, verdicts, evidence)
    assert [f.claim_id for f in report.subjects[0].secondary] == ["c1"]
    assert report.subjects[0].secondary[0].secondary_reason == "low relevance"


def test_urls_only_appear_in_references_and_appendix():
    _, md = _report(*_fixture(3))
    body = md.split("## References")[0]
    assert "http" not in body and "![" not in body
    assert "http" in md.split("## References")[1]


def test_every_ref_marker_resolves_and_every_reference_has_a_document():
    report, md = _report(*_fixture(3))
    known = {r.ref_id for r in report.references}
    cited = set(re.findall(r"\[(R\d+)\]", md.split("## References")[0]))
    assert cited and cited <= known
    assert all(r.document_id for r in report.references)


def test_ref_ids_follow_sorted_canonical_url():
    report, _ = _report(*_fixture(3, ["https://b.com/x", "https://a.com/x", "https://c.com/x"]))
    assert [r.domain for r in report.references] == ["a.com", "b.com", "c.com"]


def test_uncategorized_findings_show_no_severity_badge():
    claims, verdicts, evidence = _fixture(1, categories=["uncategorized"])
    claims[0].severity = Severity.LOW
    settings = make_settings(as_of_date=AS_OF)
    run = ResearchRun(run_id="r1", query="q", mode=SubjectMode.SINGLE, research_mode=ResearchMode.GENERAL,
                      subjects=[SUBJECT], output_path="", config_hash="abc")
    report = build_report(run, [ResolvedSubject(name=SUBJECT)], claims, verdicts, evidence, {}, settings, None, [])
    md = render_markdown(report, settings)
    assert not report.subjects[0].findings[0].adverse
    assert "`LOW`" not in md and "`SINGLE_SOURCE`" in md


def test_noise_and_identity_flagged_claims_are_excluded_and_counted():
    claims, verdicts, evidence = _fixture(3)
    claims[0].flags = [NOISE]
    claims[1].flags = [IDENTITY_MISMATCH]
    report, _ = _report(claims, verdicts, evidence)
    assert (report.coverage.excluded_noise, report.coverage.excluded_identity) == (1, 1)
    assert [f.claim_id for f in report.subjects[0].findings] == ["c2"]


def test_report_rendering_is_byte_identical_across_runs():
    fixture = _fixture(4)
    first = _report(*fixture)
    second = _report(*fixture)
    assert first[0].model_dump_json() == second[0].model_dump_json() and first[1] == second[1]

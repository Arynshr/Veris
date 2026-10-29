import re
from datetime import date

from veris.testing import make_settings
from veris.core.models import (
    Claim,
    Evidence,
    LegalStatus,
    ResearchMode,
    ResearchRun,
    ResolvedSubject,
    RetrievalStats,
    Severity,
    SourceMetadata,
    SubjectMode,
    SynthesisSlot,
    VerificationResult,
    VerificationStatus,
)
from veris.synthesis import build_report, render_markdown

ORDER = ["criminal", "litigation", "reputational"]


def settings():
    return make_settings(as_of_date=date(2026, 6, 30), lookback_years=2)


def evidence(n, passage="The CEO was convicted of fraud in court.", flags=()):
    return Evidence(
        evidence_id=f"e{n}", document_id=f"d{n}", claim=passage, supporting_passage=passage,
        span_start=0, span_end=len(passage), relevance_score=0.5,
        source_metadata=SourceMetadata(url=f"https://news{n}.test/a b", title="t", source=f"news{n}.test"),
        flags=list(flags),
    )


def claim(n, **kw):
    base = dict(claim_id=f"c{n}", text=f"Claim {n}", subject="Jane Doe", evidence_ids=[f"e{n}"],
                categories=["criminal"], legal_status=LegalStatus.CONVICTED, severity=Severity.HIGH,
                event_date=date(2026, 1, n))
    return Claim(**{**base, **kw})


def verdict(n, status=VerificationStatus.VERIFIED, conflicts=()):
    return VerificationResult(claim_id=f"c{n}", status=status, subject_match_score=1, entity_match_score=1,
                              source_quality_score=1, conflicts_with=list(conflicts))


def run(**kw):
    base = dict(run_id="r1", query="Jane Doe", mode=SubjectMode.SINGLE, research_mode=ResearchMode.BACKGROUND_CHECK,
                subjects=["Jane Doe"], output_path="", config_hash="abc", queries_run=3, providers_used=["tavily"],
                retrieval=RetrievalStats(attempted=4, fetched=4), sources_retrieved=4)
    return ResearchRun(**{**base, **kw})


def build(claims, verdicts, evs, run_=None, summary=None):
    r = build_report(run_ or run(), [ResolvedSubject(name="Jane Doe", resolution_confidence=1.0)], claims, verdicts,
                     evs, {f"d{i}": "2026-06-01T00:00:00Z" for i in range(1, 9)}, settings(), summary, ORDER)
    return r, render_markdown(r, settings())


def test_deterministic_and_round_trips():
    args = ([claim(1), claim(2, severity=Severity.LOW)], [verdict(1), verdict(2)], [evidence(1), evidence(2)])
    summary = SynthesisSlot(slot_id="summary", text="Two findings.", claim_ids=["c1", "c2"])
    (r1, m1), (r2, m2) = build(*args, summary=summary), build(*args, summary=summary)
    assert m1 == m2 and r1.model_dump_json() == r2.model_dump_json()
    assert type(r1).model_validate_json(r1.model_dump_json()) == r1


def test_fixed_ordering():
    claims = [claim(1, severity=Severity.LOW), claim(2), claim(3, event_date=None, undated=True), claim(4, event_date=date(2026, 3, 1))]
    r, _ = build(claims, [verdict(i) for i in range(1, 5)], [evidence(i) for i in range(1, 5)])
    assert [f.claim_id for f in r.subjects[0].findings] == ["c4", "c2", "c3", "c1"]


def test_every_finding_line_traces_to_claim_and_appendix():
    r, md = build([claim(1), claim(2)], [verdict(1), verdict(2)], [evidence(1), evidence(2)])
    body = md.split("## Findings by subject")[1].split("## Findings by category")[0]
    assert all(re.search(r"`c\d`", line) for line in body.splitlines() if line.startswith("- "))
    assert {a.claim_id for a in r.appendix} == {"c1", "c2"}


def test_clean_wording_depends_on_coverage():
    failed = run(retrieval=RetrievalStats(attempted=5, fetched=3, failed=2))
    _, md = build([], [], [], run_=failed)
    assert "No adverse findings in retrieved sources; 2 sources could not be retrieved." in md
    _, md = build([], [], [], run_=run(retrieval=RetrievalStats()))
    assert "no sources could be retrieved" in md
    _, md = build([], [], [])
    assert "No adverse findings in the 4 sources retrieved." in md and "clean" not in md.lower()


def test_conflicts_identity_window_and_flags():
    claims = [claim(1, legal_status=LegalStatus.CONVICTED), claim(2, legal_status=LegalStatus.DISMISSED_ACQUITTED),
              claim(3), claim(4, out_of_window=True), claim(5, flags=["instruction_like"])]
    verdicts = [verdict(1, VerificationStatus.CONFLICTED, ["c2"]), verdict(2, VerificationStatus.CONFLICTED, ["c1"]),
                verdict(3, VerificationStatus.CONFLICTED), verdict(4), verdict(5)]
    r, md = build(claims, verdicts, [evidence(i) for i in range(1, 6)])
    section = r.subjects[0]
    assert section.findings == [] and {f.claim_id for f in section.disputed} == {"c1", "c2"}
    c = r.coverage
    assert (c.claims_total, c.claims_reported, c.out_of_window, c.excluded_identity, c.excluded_unsupported) == (5, 2, 1, 1, 1)
    assert "Disputed" in md and "c3" not in md


def test_untrusted_text_is_escaped_and_undated_flagged():
    c = claim(1, text="Bad [click](http://evil.test) | <b>x</b>\n## Fake", event_date=None, undated=True)
    r, md = build([c], [verdict(1)], [evidence(1, passage="a | b [x](y)")])
    assert "[click](http" not in md and "<b>" not in md and "\n## Fake" not in md
    assert "_(undated)_" in md and "<https://news1.test/a%20b>" in md
    assert len(r.appendix) == 1


def test_category_table_and_single_source_tag():
    r, md = build([claim(1), claim(2, severity=Severity.LOW, categories=["litigation"])],
                  [verdict(1), verdict(2, VerificationStatus.SINGLE_SOURCE)], [evidence(1), evidence(2)])
    assert [(x.category, x.high, x.low) for x in r.category_rows] == [("criminal", 1, 0), ("litigation", 0, 1)]
    assert "_(single source, uncorroborated)_" in md and "| Jane Doe | Criminal | 1 | 0 | 0 | `c1` |" in md

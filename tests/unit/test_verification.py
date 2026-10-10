from veris.analysis import run_evidence_mapping, run_verification
from veris.core.models import ResearchMode, VerificationStatus

BG = ResearchMode.BACKGROUND_CHECK
S = VerificationStatus
SENTENCE = "Jane Doe of Acme Corp was charged with fraud by prosecutors on Tuesday."


def _verify(docs, entities, settings, mode=BG):
    evidence = run_evidence_mapping(
        docs, "Jane Doe", entities, settings, research_mode=mode
    )
    return run_verification(
        evidence, entities, "Jane Doe", settings, research_mode=mode
    )


def _statuses(results):
    return [r.status for r in results]


def test_single_unknown_domain_is_single_source(make_doc, entities, settings):
    docs = [make_doc("https://one.example/a", SENTENCE)]
    _, results = _verify(docs, entities, settings)
    assert _statuses(results) == [S.SINGLE_SOURCE]


def test_two_independent_domains_verify(make_doc, entities, settings):
    docs = [
        make_doc("https://one.example/a", SENTENCE),
        make_doc("https://two.example/b", SENTENCE + " Details followed."),
    ]
    claims, results = _verify(docs, entities, settings)
    assert len(claims) == 1 and len(claims[0].evidence_ids) == 2
    assert _statuses(results) == [S.VERIFIED]
    assert results[0].independent_domains == 2


def test_same_domain_articles_count_once(make_doc, entities, settings):
    docs = [
        make_doc("https://one.example/a", SENTENCE),
        make_doc("https://www.one.example/b", SENTENCE + " Details followed."),
    ]
    _, results = _verify(docs, entities, settings)
    assert _statuses(results) == [S.SINGLE_SOURCE]
    assert results[0].independent_domains == 1


def test_primary_source_verifies_alone(make_doc, entities, settings):
    docs = [make_doc("https://courts.example.gov/a", SENTENCE)]
    _, results = _verify(docs, entities, settings)
    assert _statuses(results) == [S.VERIFIED]


def test_low_quality_domains_do_not_corroborate(make_doc, entities, settings):
    docs = [
        make_doc("https://a.blogspot.com/p", SENTENCE),
        make_doc("https://b.wordpress.com/p", SENTENCE + " Details followed."),
    ]
    _, results = _verify(docs, entities, settings)
    assert _statuses(results) == [S.SINGLE_SOURCE]
    assert results[0].independent_domains == 0


def test_identity_mismatch_is_conflicted(make_doc, entities, settings):
    doc = make_doc(
        "https://one.example/a",
        "Jane Doe of Acme Corp attended the annual charity gala this spring season. "
        "The regulator imposed a penalty on an unrelated firm for late filings. "
        "Another unrelated sentence about the weather and local traffic conditions.",
    )
    evidence = run_evidence_mapping(
        [doc], "Jane Doe", entities, settings, research_mode=BG
    )
    far = [e for e in evidence if "penalty" in e.claim]
    far[0].supporting_passage = "The regulator imposed a penalty on an unrelated firm."
    _, results = run_verification(far, entities, "Jane Doe", settings, research_mode=BG)
    assert _statuses(results) == [S.CONFLICTED]


def test_ids_are_deterministic(make_doc, entities, settings):
    docs = [
        make_doc("https://one.example/a", SENTENCE),
        make_doc("https://two.example/b", SENTENCE + " Details followed."),
    ]
    first, _ = _verify(docs, entities, settings)
    second, _ = _verify(docs, entities, settings)
    assert [c.claim_id for c in first] == [c.claim_id for c in second]

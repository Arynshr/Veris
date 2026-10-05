from veris.analysis import run_evidence_mapping
from veris.core.models import ResearchMode

BG = ResearchMode.BACKGROUND_CHECK


def _run(docs, entities, settings, subject="Jane Doe", mode=BG):
    return run_evidence_mapping(docs, subject, entities, settings, research_mode=mode)


def test_ids_are_deterministic(make_doc, entities, settings):
    text = "Jane Doe of Acme Corp was sued in court over a contract dispute last year."
    first = _run([make_doc("https://a.example/x", text)], entities, settings)
    second = _run([make_doc("https://a.example/x", text)], entities, settings)
    assert first and [e.evidence_id for e in first] == [e.evidence_id for e in second]


def test_span_locates_claim_in_document(make_doc, entities, settings):
    doc = make_doc(
        "https://a.example/x",
        "Intro sentence that is fairly long and unrelated to anything here. "
        "Jane Doe of Acme Corp was sued in court over a contract dispute.",
    )
    (ev,) = _run([doc], entities, settings)
    assert doc.content[ev.span_start : ev.span_end] == ev.claim


def test_passage_includes_neighbouring_sentences(make_doc, entities, settings):
    doc = make_doc(
        "https://a.example/x",
        "Jane Doe joined Acme Corp as a director in 2019. "
        "The regulator later imposed a penalty for reporting failures. "
        "Acme Corp said it has since tightened its controls.",
    )
    evidence = _run([doc], entities, settings)
    penalty = next(e for e in evidence if "penalty" in e.claim)
    assert "Jane Doe joined" in penalty.supporting_passage
    assert "tightened its controls" in penalty.supporting_passage
    assert penalty.claim in penalty.supporting_passage


def test_distant_subject_mention_recorded_as_context_only(make_doc, entities, settings):
    filler = "Filler sentence number {} that says nothing at all about anyone. "
    content = (
        "Jane Doe was mentioned near the top of this Acme Corp article. "
        + filler.format(1)
        + filler.format(2)
        + "The regulator imposed a penalty on the firm for late filings."
    )
    evidence = _run([make_doc("https://a.example/x", content)], entities, settings)
    penalty = next(e for e in evidence if "penalty" in e.claim)
    assert "Jane Doe" not in penalty.supporting_passage
    assert penalty.subject_context and "Jane Doe" in penalty.subject_context


def test_identity_gate_rejects_document_without_subject(make_doc, entities, settings):
    doc = make_doc(
        "https://a.example/x",
        "A local fraud ring was charged after a long police investigation this year.",
    )
    assert _run([doc], entities, settings) == []


def test_identity_gate_requires_context_anchor_in_background_check(
    make_doc, entities, settings
):
    doc = make_doc(
        "https://a.example/x",
        "Jane Doe, a surgeon at another clinic, was sued in a malpractice lawsuit.",
    )
    assert _run([doc], entities, settings) == []
    assert _run([doc], entities, settings, mode=ResearchMode.GENERAL) != []

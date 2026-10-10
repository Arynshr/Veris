from datetime import date

import pytest

from tests.helpers import make_evidence, verify_all
from veris.core.models import DateSource, LegalStatus, VerificationStatus

S = VerificationStatus


def _by_id(claims, results):
    verdicts = {r.claim_id: r for r in results}
    return [(c, verdicts[c.claim_id]) for c in claims]


def test_progression_resolves_to_latest_status(settings):
    ev = [
        make_evidence(
            "e2",
            "John Smith was charged with fraud in the Acme case.",
            "https://apnews.com/b",
            "2024-01-01",
        ),
        make_evidence(
            "e3",
            "John Smith was convicted of fraud in the Acme case.",
            "https://reuters.com/c",
            "2025-01-01",
        ),
    ]
    claims, results, _ = verify_all(ev, settings)
    assert len(claims) == 1
    claim, verdict = _by_id(claims, results)[0]
    assert claim.legal_status == LegalStatus.CONVICTED
    assert (
        claim.event_date == date(2025, 1, 1)
        and claim.date_source == DateSource.METADATA
    )
    assert verdict.status != S.CONFLICTED and verdict.conflicts_with == []


def test_terminal_disagreement_splits_and_links_both_sides(settings):
    ev = [
        make_evidence(
            "e1",
            "John Smith was convicted of fraud in the Acme case.",
            "https://reuters.com/a",
            "2025-03-01",
        ),
        make_evidence(
            "e2",
            "John Smith was acquitted of fraud in the Acme case.",
            "https://apnews.com/b",
            "2025-09-01",
        ),
    ]
    claims, results, _ = verify_all(ev, settings)
    pairs = _by_id(claims, results)
    assert len(pairs) == 2
    assert all(v.status == S.CONFLICTED for _, v in pairs)
    assert {c.legal_status for c, _ in pairs} == {
        LegalStatus.CONVICTED,
        LegalStatus.DISMISSED_ACQUITTED,
    }
    ids = {c.claim_id for c, _ in pairs}
    for claim, verdict in pairs:
        assert verdict.conflicts_with == sorted(ids - {claim.claim_id})
    assert {eid for c, _ in pairs for eid in c.evidence_ids} == {"e1", "e2"}


def test_non_terminal_disagreement_is_not_a_conflict(settings):
    ev = [
        make_evidence(
            "e1",
            "John Smith was charged with fraud in the Acme case.",
            "https://reuters.com/a",
            "2025-01-01",
        ),
        make_evidence(
            "e2",
            "John Smith was settled with fraud in the Acme case.",
            "https://apnews.com/b",
            "2025-02-01",
        ),
    ]
    claims, results, _ = verify_all(ev, settings)
    assert all(v.status != S.CONFLICTED for _, v in _by_id(claims, results))


def test_conflict_ids_are_deterministic(settings):
    ev = [
        make_evidence(
            "e1",
            "John Smith was convicted of fraud in the Acme case.",
            "https://reuters.com/a",
            "2025-03-01",
        ),
        make_evidence(
            "e2",
            "John Smith was acquitted of fraud in the Acme case.",
            "https://apnews.com/b",
            "2025-09-01",
        ),
    ]
    first = [c.claim_id for c in verify_all(ev, settings)[0]]
    second = [c.claim_id for c in verify_all(list(reversed(ev)), settings)[0]]
    assert sorted(first) == sorted(second)


@pytest.mark.parametrize(
    ("published", "out_of_window", "undated"),
    [
        ("2025-06-01", False, False),
        ("2023-12-31", True, False),
        ("2024-01-02", False, False),
        (None, False, True),
    ],
)
def test_lookback_window_flags(settings, published, out_of_window, undated):
    ev = [
        make_evidence(
            "e1",
            "John Smith was sued over a lease dispute.",
            "https://blog.example.org/c",
            published,
        )
    ]
    claims, _, _ = verify_all(ev, settings)
    assert (claims[0].out_of_window, claims[0].undated) == (out_of_window, undated)


def test_longer_lookback_keeps_older_claims(settings_factory):
    ev = [
        make_evidence(
            "e1",
            "John Smith was sued over a lease dispute.",
            "https://blog.example.org/c",
            "2020-06-01",
        )
    ]
    assert verify_all(ev, settings_factory(lookback_years=2))[0][0].out_of_window
    assert not verify_all(ev, settings_factory(lookback_years=10))[0][0].out_of_window


def test_text_date_used_only_when_published_at_missing(settings):
    ev = [
        make_evidence(
            "e1",
            "John Smith was sued in 2015 over a lease dispute.",
            "https://blog.example.org/c",
            None,
        )
    ]
    claim = verify_all(ev, settings)[0][0]
    assert claim.date_source == DateSource.TEXT and claim.out_of_window

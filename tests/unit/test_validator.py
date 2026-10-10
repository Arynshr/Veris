import pytest

from tests.helpers import check_slot, make_evidence, verify_all
from veris.core.models import SynthesisSlot
from veris.synthesis.validator import is_eligible, numbers, validate_slot

GOOD = "John Smith was convicted of fraud in 2024 and fined $2 million by regulators."


@pytest.fixture
def base(settings):
    claims, results, ctx = verify_all([make_evidence("e1", GOOD)], settings)
    return claims[0], ctx


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$2m", {2e6}),
        ("$2 million", {2e6}),
        ("1,200 people", {1200.0}),
        ("3.5 billion", {3.5e9}),
        ("the 3rd of May", set()),
        ("Q3 results", set()),
        ("in 2019 and 2020", {2019.0, 2020.0}),
        ("no numbers", set()),
    ],
)
def test_number_extraction(text, expected):
    assert numbers(text) == expected


def test_accepts_faithful_text(base):
    claim, ctx = base
    assert check_slot(ctx, GOOD, [claim.claim_id]) == []


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (GOOD.replace("$2 million", "$5 million"), "values"),
        (GOOD.replace("2024", "2019"), "values"),
        (GOOD.replace("convicted", "acquitted"), "legal status"),
        (GOOD.replace("convicted", "charged"), "legal status"),
        ("   ", "empty"),
        ("Too short.", "length"),
        (GOOD + " x" * 600, "length"),
    ],
)
def test_rejections(base, text, fragment):
    claim, ctx = base
    assert any(fragment in e for e in check_slot(ctx, text, [claim.claim_id]))


def test_rejects_unknown_and_missing_citations(base):
    claim, ctx = base
    assert any("unknown" in e for e in check_slot(ctx, GOOD, ["claim_fake"]))
    assert any(
        "unknown" in e for e in check_slot(ctx, GOOD, [claim.claim_id, "claim_fake"])
    )
    assert check_slot(ctx, GOOD, []) != []


def test_hedged_wording_is_always_allowed(base):
    claim, ctx = base
    assert (
        check_slot(
            ctx,
            "John Smith is accused of fraud, reported in 2024 by regulators.",
            [claim.claim_id],
        )
        == []
    )


def test_eligibility_matrix(base):
    claim, ctx = base
    assert is_eligible(claim, ctx)
    assert not is_eligible(
        claim.model_copy(update={"flags": ["instruction_like"]}), ctx
    )
    assert not is_eligible(claim.model_copy(update={"out_of_window": True}), ctx)
    assert not is_eligible(claim.model_copy(update={"claim_id": "claim_nope"}), ctx)


def test_slot_with_flagged_claim_is_rejected(base):
    claim, ctx = base
    flagged = claim.model_copy(update={"flags": ["instruction_like"]})
    ctx.claims[claim.claim_id] = flagged
    errors = validate_slot(
        SynthesisSlot(slot_id="s", text=GOOD, claim_ids=[claim.claim_id]), ctx
    )
    assert any("flagged" in e for e in errors)


def test_conflicted_requires_both_sides(settings):
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
    claims, _, ctx = verify_all(ev, settings)
    a, b = (c.claim_id for c in claims)
    text = "The fraud case against John Smith is disputed: one report says convicted, another acquitted."
    assert any("both sides" in e for e in check_slot(ctx, text, [a]))
    assert check_slot(ctx, text, [a, b]) == []

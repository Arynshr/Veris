import pytest

from veris.analysis.classification import classify
from veris.analysis.taxonomy import UNCATEGORIZED
from veris.core.models import LegalStatus as L
from veris.core.models import Severity


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("Acme was fined by the regulator over reporting.", L.ALLEGED),
        ("He was convicted of fraud in 2019.", L.CONVICTED),
        ("He was found guilty of fraud.", L.CONVICTED),
        ("He was not convicted of fraud.", L.ALLEGED),
        ("He was never charged with fraud.", L.ALLEGED),
        ("He wasn't convicted of fraud.", L.ALLEGED),
        ("He wasn\u2019t convicted of fraud.", L.ALLEGED),
        ("He was neither charged nor convicted of fraud.", L.ALLEGED),
        ("No charges were filed, and later he was convicted of fraud.", L.CONVICTED),
        ("He denies the fraud allegations.", L.ALLEGED),
        ("The fraud charges were dismissed last year.", L.DISMISSED_ACQUITTED),
        ("The judge dismissed the fraud case.", L.DISMISSED_ACQUITTED),
        ("He was convicted of fraud but later acquitted on appeal.", L.DISMISSED_ACQUITTED),
        ("The fraud conviction was overturned.", L.DISMISSED_ACQUITTED),
        ("He was cleared of fraud by the court.", L.DISMISSED_ACQUITTED),
        ("Acme settled the fraud lawsuit for $2m.", L.SETTLED),
        ("He was charged with fraud in March.", L.CHARGED),
        ("He was indicted for fraud.", L.CHARGED),
        ("She pleaded not guilty to the fraud charges.", L.ALLEGED),
        ("He was convicted, then the fraud conviction was quashed.", L.DISMISSED_ACQUITTED),
    ],
)
def test_legal_status(taxonomy, text, status):
    assert classify(text, taxonomy).legal_status == status


@pytest.mark.parametrize(
    "text",
    [
        "She sells fine art in Paris.",
        "Wait 30 sec for the charged battery.",
        "A tennis court opened downtown.",
        "The criminality of the plot kept readers hooked.".replace("criminality", "mystery"),
        "",
        "   ",
    ],
)
def test_no_false_positives(taxonomy, text):
    out = classify(text, taxonomy)
    assert out.categories == [UNCATEGORIZED]
    assert out.legal_status == L.UNKNOWN and out.severity == Severity.LOW


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("The SEC probe found compliance violations.", "regulatory_sanctions"),
        ("He faces a class action lawsuit.", "litigation"),
        ("Executives paid bribes to win contracts.", "corruption_bribery"),
        ("The plant faced child labor allegations.", "esg_labor_safety"),
        ("The firm collapsed in a Ponzi scheme.", "financial_crime_fraud"),
        ("A scandal engulfed the board.", "reputational"),
    ],
)
def test_category_detection(taxonomy, text, category):
    assert category in classify(text, taxonomy).categories


@pytest.mark.parametrize(
    ("text", "severity"),
    [
        ("He was convicted of fraud.", Severity.HIGH),
        ("He was charged with bribery.", Severity.HIGH),
        ("Acme settled the fraud lawsuit.", Severity.MEDIUM),
        ("The fraud charges were dismissed.", Severity.LOW),
        ("Critics said a scandal engulfed the board.", Severity.LOW),
    ],
)
def test_severity_matrix(taxonomy, text, severity):
    assert classify(text, taxonomy).severity == severity


def test_multilabel_keeps_taxonomy_order(taxonomy):
    out = classify("He was convicted of bribery and fraud.", taxonomy)
    assert out.categories == [c for c in taxonomy.category_order if c in out.categories]
    assert {"financial_crime_fraud", "corruption_bribery"} <= set(out.categories)
    assert out.severity == Severity.HIGH


def test_categories_come_from_claim_status_from_passage(taxonomy):
    claim = "John Smith faces a fraud lawsuit."
    passage = "Weather was mild. John Smith faces a fraud lawsuit. The case was dismissed in May."
    out = classify(passage, taxonomy, claim=claim)
    assert "criminal" not in out.categories
    assert out.legal_status == L.DISMISSED_ACQUITTED


def test_classification_is_deterministic(taxonomy):
    text = "He was convicted of fraud and bribery in 2019, then fined $2 million."
    results = {classify(text, taxonomy).model_dump_json() for _ in range(25)}
    assert len(results) == 1

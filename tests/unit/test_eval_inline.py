import json

import pytest

from veris.evaluation import compare_reports, load_golden_cases, run_eval
from veris.evaluation.golden import ExpectedFinding

URL = "https://reuters.com/2025/03/01/a"


def _case(tmp_path, expected: list[dict], content: str = "John Smith was convicted of fraud in the Acme case last year. Acme later said it would appeal."):
    (tmp_path / "c1.json").write_text(json.dumps({
        "case_id": "c1", "subject": "John Smith", "query": "background check John Smith", "entities": [],
        "documents": [{"url": URL, "title": "t", "content": content, "published_at": "2025-03-01T00:00:00Z"}],
        "expected_findings": expected,
    }))
    return load_golden_cases(tmp_path)


FINDING = {"url": URL, "match_text": "convicted of fraud", "verification": "VERIFIED"}


@pytest.mark.parametrize(
    ("legacy", "migrated"),
    [("legal_litigation", "litigation"), ("financial", "financial_crime_fraud"), ("general", "uncategorized"), ("criminal", "criminal")],
)
def test_legacy_category_names_are_migrated(legacy, migrated):
    f = ExpectedFinding.model_validate({**FINDING, "category": legacy})
    assert f.categories == [migrated]


def test_multilabel_expectation_scores_perfectly(tmp_path, settings):
    cases = _case(tmp_path, [{**FINDING, "categories": ["criminal", "financial_crime_fraud"], "legal_status": "convicted", "severity": "HIGH"}])
    report = run_eval(cases, settings)
    assert report.headline()["category.f1"] == 1.0
    assert report.legal_status_accuracy == 1.0 and report.severity_accuracy == 1.0


def test_wrong_labels_reduce_scores(tmp_path, settings):
    cases = _case(tmp_path, [{**FINDING, "categories": ["litigation"], "legal_status": "settled", "severity": "LOW"}])
    report = run_eval(cases, settings)
    assert report.headline()["category.f1"] < 1.0
    assert report.legal_status_accuracy == 0.0 and report.severity_accuracy == 0.0


def test_eval_is_deterministic(tmp_path, settings):
    cases = _case(tmp_path, [{**FINDING, "categories": ["criminal", "financial_crime_fraud"]}])
    assert run_eval(cases, settings).model_dump_json() == run_eval(cases, settings).model_dump_json()


def test_compare_reports_flags_regressions_and_ignores_missing_baseline_metrics(tmp_path, settings):
    good = run_eval(_case(tmp_path, [{**FINDING, "categories": ["criminal", "financial_crime_fraud"], "legal_status": "convicted"}]), settings)
    bad = good.model_copy(update={"legal_status_accuracy": 0.5})
    assert compare_reports(good, good) == []
    assert any("legal_status_accuracy" in r for r in compare_reports(bad, good))
    legacy_baseline = good.model_copy(update={"legal_status_accuracy": None, "severity_accuracy": None})
    assert compare_reports(bad, legacy_baseline) == []
    assert compare_reports(bad, good, tolerance=0.6) == []

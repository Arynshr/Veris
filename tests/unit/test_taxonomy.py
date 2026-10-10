import pytest
import yaml
from pydantic import ValidationError

from veris.analysis.taxonomy import UNCATEGORIZED, Taxonomy, load_taxonomy
from veris.core.models import LegalStatus, Severity


@pytest.fixture
def raw(prod_settings) -> dict:
    return yaml.safe_load(prod_settings.taxonomy_path.read_text(encoding="utf-8"))


def test_production_taxonomy_loads_with_expected_categories(taxonomy):
    expected = {
        "criminal",
        "regulatory_sanctions",
        "litigation",
        "financial_crime_fraud",
        "corruption_bribery",
        "esg_labor_safety",
        "reputational",
    }
    assert set(taxonomy.categories) == expected
    assert UNCATEGORIZED not in taxonomy.categories


def test_severity_matrix_is_total(taxonomy):
    for status in LegalStatus:
        for category in taxonomy.categories:
            assert isinstance(taxonomy.severity_for(status, [category]), Severity)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(schema_version=99),
        lambda r: r["categories"].update({UNCATEGORIZED: ["x"]}),
        lambda r: r["severity"]["convicted"].pop("default"),
        lambda r: r["severity"]["convicted"].update({"bogus": "HIGH"}),
        lambda r: r["severity"]["convicted"].update({"default": "CRITICAL"}),
        lambda r: r.update(status_precedence=r["status_precedence"][:-1]),
    ],
    ids=[
        "schema",
        "uncategorized",
        "no-default",
        "unknown-category",
        "bad-level",
        "precedence",
    ],
)
def test_invalid_taxonomy_is_rejected(raw, mutate):
    mutate(raw)
    with pytest.raises(ValidationError):
        Taxonomy.model_validate(raw)


def test_strip_cues_removes_status_words_only(taxonomy):
    out = taxonomy.strip_cues("John Smith was convicted of fraud in the Acme case.")
    assert "convicted" not in out and "fraud" in out and "Acme" in out


def test_status_rank_follows_precedence(taxonomy):
    ranks = [taxonomy.status_rank(s) for s in taxonomy.status_precedence]
    assert ranks == sorted(ranks, reverse=True)
    assert taxonomy.status_rank(LegalStatus.ALLEGED) == 0


def test_loader_is_cached(prod_settings):
    assert load_taxonomy(prod_settings.taxonomy_path) is load_taxonomy(
        prod_settings.taxonomy_path
    )


def test_loader_error_names_the_file(tmp_path, raw):
    raw["schema_version"] = 99
    bad = tmp_path / "taxonomy.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Invalid taxonomy file"):
        load_taxonomy(bad)

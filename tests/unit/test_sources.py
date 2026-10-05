import pytest
from pydantic import ValidationError

from veris.analysis.sources import SourcePolicy, load_source_policy, registrable_domain


@pytest.mark.parametrize(
    ("host", "tier", "source_type"),
    [
        ("www.sec.gov", 1, "primary"),
        ("cs.stanford.edu", 1, "primary"),
        ("reuters.com", 2, "wire"),
        ("someone.blogspot.com", 4, "blog"),
        ("old.reddit.com", 4, "forum"),
        ("unknown-site.example", 3, "news"),
    ],
)
def test_classify(settings, host, tier, source_type):
    result = load_source_policy(settings.source_tiers_path).classify(host)
    assert (result.tier, result.source_type.value) == (tier, source_type)


def test_unknown_domain_is_never_top_tier(settings):
    policy = load_source_policy(settings.source_tiers_path)
    assert policy.classify("nobody.example").tier > settings.verified_alone_max_tier


def test_policy_rejects_tier_without_score():
    with pytest.raises(ValidationError):
        SourcePolicy.model_validate(
            {
                "schema_version": 1,
                "tiers": {1: 1.0},
                "default": {"tier": 2, "type": "news"},
            }
        )


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("www.example.com", "example.com"),
        ("news.example.com", "example.com"),
        ("www.bbc.co.uk", "bbc.co.uk"),
    ],
)
def test_registrable_domain(host, expected):
    assert registrable_domain(host) == expected

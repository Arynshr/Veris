import pytest
from pydantic import ValidationError

from veris.core.config import MAX_SEARCH_RESULTS_CEILING


def test_max_search_results_default_is_within_bounds(settings_factory):
    assert 1 <= settings_factory().max_search_results_per_query <= MAX_SEARCH_RESULTS_CEILING


def test_max_search_results_hard_ceiling(settings_factory):
    """Regression: the ceiling must be enforced by validation, not just by
    the CLI help text - config/env overrides can't sneak past it either."""
    assert MAX_SEARCH_RESULTS_CEILING == 10
    with pytest.raises(ValidationError):
        settings_factory(max_search_results_per_query=11)
    # exactly at the ceiling is fine
    assert (
        settings_factory(max_search_results_per_query=10).max_search_results_per_query
        == 10
    )


def test_searxng_not_configured_by_default(settings_factory):
    """Regression for the 'always-configured' bug: unset must mean None,
    never a hardcoded localhost URL."""
    assert settings_factory().searxng_base_url is None


def test_max_queries_per_plan_default_is_usable(settings_factory):
    """A default of 1 (the original bug) makes 'research planning' a no-op."""
    assert settings_factory().max_queries_per_plan > 1


def test_max_documents_per_subject_default(settings_factory):
    assert settings_factory().max_documents_per_subject == 10


def test_max_synthesis_evidence_per_subject_default(settings_factory):
    assert settings_factory().max_synthesis_evidence_per_subject >= 1

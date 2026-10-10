import pytest

from tests.helpers import static_json_groq
from veris.core.models import ResearchMode, SubjectMode
from veris.intake.ingestion import extract_entities
from veris.intake.understanding import _regex_fallback, understand_query

pytestmark = pytest.mark.requires_spacy


def test_regex_fallback_splits_ceo_and_cfo_into_two_role_terms():
    """Regression: 'CEO and CFO' must become two role entries, not one merged
    group - each is a distinct seat that must resolve to one person."""
    query = "background check on the CEO and CFO of Acme Corp"
    entities = extract_entities(query)
    result = _regex_fallback(query, entities)
    assert result.mode == SubjectMode.ROLE_GROUP
    assert set(result.role_terms) == {"CEO", "CFO"}
    assert result.research_mode == ResearchMode.BACKGROUND_CHECK


def test_regex_fallback_single_named_person():
    query = "Tell me about Tim Cook's career"
    entities = extract_entities(query)
    result = _regex_fallback(query, entities)
    assert result.mode in (SubjectMode.SINGLE, SubjectMode.NAMED_LIST)
    assert any("Cook" in s for s in result.explicit_subjects)


def test_regex_fallback_general_mode_by_default():
    query = "compare the EU and US approaches to AI regulation"
    entities = extract_entities(query)
    result = _regex_fallback(query, entities)
    assert result.research_mode == ResearchMode.GENERAL


@pytest.mark.asyncio
async def test_understand_query_without_groq_uses_regex_fallback(settings_factory):
    settings = settings_factory()
    result = await understand_query("the board of directors of Acme", [], settings)
    assert result.mode == SubjectMode.ROLE_GROUP


@pytest.mark.asyncio
async def test_understand_query_with_groq_parses_response(
    settings_factory, monkeypatch
):

    fake = static_json_groq(
        {
            "mode": "role_group",
            "research_mode": "general",
            "organization": "Acme",
            "role_terms": ["CEO", "CFO"],
            "explicit_subjects": [],
        }
    )
    monkeypatch.setattr("veris.intake.understanding.Groq", fake)
    settings = settings_factory(groq_api_key="fake")
    result = await understand_query("CEO and CFO of Acme", [], settings)
    assert result.mode == SubjectMode.ROLE_GROUP
    assert result.role_terms == ["CEO", "CFO"]


@pytest.mark.asyncio
async def test_understand_query_falls_back_when_groq_errors(
    settings_factory, monkeypatch
):
    class Boom:
        def __init__(self, api_key=None):
            raise RuntimeError("down")

    monkeypatch.setattr("veris.intake.understanding.Groq", Boom)
    settings = settings_factory(groq_api_key="fake")
    result = await understand_query("the board of directors of Acme", [], settings)
    assert result.mode == SubjectMode.ROLE_GROUP  # regex fallback still worked

import json

import pytest

from tests.conftest import make_search_query, static_json_groq
from veris.core.models import IngestionResult, ResearchMode
from veris.retrieval.planning import _dedupe_queries, run_planning


def _ingestion(prompt="solid-state batteries") -> IngestionResult:
    return IngestionResult(
        original_prompt=prompt, entities=[], system_instructions="be rigorous"
    )


def test_dedupe_queries_keeps_first_occurrence_case_insensitive():
    queries = [
        make_search_query("Solid-state batteries overview", "q1"),
        make_search_query("solid-state batteries OVERVIEW", "q2"),  # dup
        make_search_query("solid-state batteries safety", "q3"),
    ]
    deduped = _dedupe_queries(queries)
    assert [q.query_id for q in deduped] == ["q1", "q3"]


@pytest.mark.asyncio
async def test_run_planning_without_groq_uses_templates_only(settings_factory):
    settings = settings_factory(max_queries_per_plan=5)
    plan = await run_planning(_ingestion(), settings, subject="solid-state batteries")
    assert 1 <= len(plan.queries) <= 5
    assert all(q.text for q in plan.queries)


@pytest.mark.asyncio
async def test_run_planning_deduplicates_against_llm_augmented_queries(
    settings_factory, monkeypatch
):
    """Regression: the LLM wasn't shown what was already planned and would
    regenerate a template query verbatim - that duplicate used to get
    searched a second time for no new information."""
    settings = settings_factory(groq_api_key="fake", max_queries_per_plan=10)
    template_plan = await run_planning(
        _ingestion(),
        settings_factory(max_queries_per_plan=10),
        subject="solid-state batteries",
    )
    a_template_query = template_plan.queries[0].text

    fake = static_json_groq(
        {
            "queries": [
                {
                    "text": a_template_query,
                    "priority": 1,
                    "rationale": "dup",
                },  # duplicate of a template
                {
                    "text": "solid-state battery manufacturing scale-up 2026",
                    "priority": 2,
                    "rationale": "new angle",
                },
            ]
        }
    )
    monkeypatch.setattr("veris.retrieval.planning.Groq", fake)

    plan = await run_planning(_ingestion(), settings, subject="solid-state batteries")
    texts_normalized = [q.text.strip().lower() for q in plan.queries]
    assert len(texts_normalized) == len(set(texts_normalized)), (
        "duplicate query text leaked into the plan"
    )
    assert any("manufacturing scale-up" in t for t in texts_normalized)


@pytest.mark.asyncio
async def test_run_planning_survives_llm_failure(settings_factory, monkeypatch):
    settings = settings_factory(groq_api_key="fake")

    class Boom:
        def __init__(self, api_key=None):
            raise RuntimeError("network down")

    monkeypatch.setattr("veris.retrieval.planning.Groq", Boom)
    plan = await run_planning(_ingestion(), settings, subject="solid-state batteries")
    assert len(plan.queries) >= 1  # falls back to templates, never raises


@pytest.mark.asyncio
async def test_run_planning_background_check_mode_uses_adverse_prompt(
    settings_factory, monkeypatch
):
    captured = {}

    def content_fn(**kwargs):
        captured["system"] = kwargs["messages"][0]["content"]
        return json.dumps({"queries": []})

    from tests.conftest import fake_groq_factory

    monkeypatch.setattr("veris.retrieval.planning.Groq", fake_groq_factory(content_fn))

    settings = settings_factory(groq_api_key="fake")
    await run_planning(
        _ingestion(),
        settings,
        subject="Jane Doe",
        research_mode=ResearchMode.BACKGROUND_CHECK,
    )
    assert "adverse" in captured["system"].lower()

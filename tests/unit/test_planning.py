import pytest

from tests.helpers import make_search_query
from veris.core.models import ResearchMode
from veris.retrieval import planning
from veris.retrieval.planning import _dedupe_queries, run_planning


def test_dedupe_queries_keeps_first_occurrence_case_insensitive():
    queries = [
        make_search_query("Solid-state batteries overview", "q1"),
        make_search_query("solid-state batteries OVERVIEW", "q2"),
        make_search_query("solid-state batteries safety", "q3"),
    ]
    assert [q.query_id for q in _dedupe_queries(queries)] == ["q1", "q3"]


def test_planner_has_no_llm_dependency():
    assert not hasattr(planning, "Groq")


@pytest.mark.parametrize("mode", list(ResearchMode))
def test_plan_is_deterministic_bounded_and_unique(settings_factory, mode):
    settings = settings_factory(max_queries_per_plan=5, groq_api_key="ignored")
    a = run_planning(settings, "Acme Corp", mode)
    b = run_planning(settings, "Acme Corp", mode)
    assert a.model_dump() == b.model_dump()
    assert 1 <= len(a.queries) <= 5
    texts = [q.text.strip().lower() for q in a.queries]
    assert len(texts) == len(set(texts))
    assert len({q.query_id for q in a.queries}) == len(a.queries)
    assert [q.priority for q in a.queries] == sorted(q.priority for q in a.queries)
    assert all("acme corp" in t for t in texts)


def test_plan_cap_is_respected(settings_factory):
    plan = run_planning(settings_factory(max_queries_per_plan=1), "Acme Corp")
    assert len(plan.queries) == 1


def test_background_check_plan_targets_adverse_topics(settings_factory):
    plan = run_planning(
        settings_factory(max_queries_per_plan=25),
        "Jane Doe",
        ResearchMode.BACKGROUND_CHECK,
    )
    joined = " ".join(q.text.lower() for q in plan.queries)
    assert any(term in joined for term in ("lawsuit", "fraud", "sanction", "indict"))

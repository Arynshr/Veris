import asyncio

from veris.core.llm import LLMError
from veris.core.models import Entity, ResearchMode, SubjectMode
from veris.intake.understanding import understand_query
from veris.testing import FakeLLM

ENTITIES = [Entity(text="Jane Doe", label="PERSON", start_char=0, end_char=8)]
QUERY = "background check on Jane Doe"


def run(llm):
    return asyncio.run(understand_query(QUERY, ENTITIES, llm))


def test_llm_response_is_used():
    llm = FakeLLM({"mode": SubjectMode.NAMED_LIST.value, "research_mode": "general", "organization": None,
                   "role_terms": [], "explicit_subjects": ["A", "B"]})
    u = run(llm)
    assert (u.mode, u.research_mode, u.explicit_subjects) == (SubjectMode.NAMED_LIST, ResearchMode.GENERAL, ["A", "B"])


def test_llm_error_and_malformed_response_fall_back_to_regex():
    for llm in (FakeLLM(error=LLMError("down")), FakeLLM({"unexpected": 1}), None):
        u = run(llm)
        assert u.mode == SubjectMode.SINGLE and u.explicit_subjects == ["Jane Doe"]
        assert u.research_mode == ResearchMode.BACKGROUND_CHECK

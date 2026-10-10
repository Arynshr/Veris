import hashlib

import pytest
from tests.helpers import FakeSearchProvider

from veris.core.models import (
    Document,
    PipelineStage,
    ResearchMode,
    ResolvedSubject,
    SubjectType,
)
from veris.pipeline import run_research



pytestmark = pytest.mark.requires_spacy


def _fake_document(url: str) -> Document:
    content = f"This is coverage about the subject, published at {url}, discussing recent developments in detail here."
    return Document(
        document_id=hashlib.sha1(url.encode()).hexdigest()[:16],
        url=url,
        canonical_url=url,
        title=f"Article at {url}",
        content=content,
        source="news.example",
        content_hash=hashlib.sha256((url + content).encode()).hexdigest(),
    )


@pytest.fixture
def patched_infra(monkeypatch):
    """Fake the two things that touch the real outside world - search
    providers and page crawling - while exercising every other real
    pipeline stage (understanding, resolution, planning, evidence,
    verification, synthesis fallback, storage)."""
    provider = FakeSearchProvider()
    monkeypatch.setattr(
        "veris.pipeline.build_provider_chain", lambda settings: [provider]
    )

    extract_calls: list[str] = []

    async def fake_extract(url, title_hint=""):
        extract_calls.append(url)
        return _fake_document(url)

    monkeypatch.setattr("veris.retrieval.extraction._extract_document", fake_extract)
    return provider, extract_calls


@pytest.mark.asyncio
async def test_single_subject_run_end_to_end(patched_infra, settings_factory):
    provider, extract_calls = patched_infra
    provider.responses = {
        "solid-state batteries": [
            (
                f"Article {i}",
                "Solid-state batteries research snippet.",
                f"https://news.example/{i}",
            )
            for i in range(1, 6)
        ],
    }
    settings = settings_factory(max_queries_per_plan=1, max_documents_per_subject=5)

    summary = await run_research(
        query="solid-state batteries",
        output_path="report.md",
        settings=settings,
        resolved_subjects_override=[
            ResolvedSubject(
                name="solid-state batteries",
                subject_type=SubjectType.TOPIC,
                resolution_confidence=1.0,
            )
        ],
        research_mode_override=ResearchMode.GENERAL,
    )

    assert summary.subjects == ["solid-state batteries"]
    assert summary.sources_retrieved >= 1
    from pathlib import Path

    assert Path(summary.report_path).exists()


@pytest.mark.asyncio
async def test_ceo_and_cfo_role_group_resolves_to_two_subjects_end_to_end(
    patched_infra, settings_factory, monkeypatch
):
    """Full-pipeline regression test for the over-resolution bug: driving
    the request through resolution -> per-subject research must still land
    on exactly 2 subjects, not a whole board.

    Understanding itself (free text -> role_terms) is real-spaCy-dependent
    and already covered by dedicated unit tests in test_understanding.py;
    here we pin it to a fixed, known-good QueryUnderstanding so this test's
    pass/fail is about resolution + orchestration, not spaCy's NER call on
    a particular sentence.
    """
    from veris.core.models import QueryUnderstanding, SubjectMode

    async def fake_understand(query, entities, settings):
        return QueryUnderstanding(
            mode=SubjectMode.ROLE_GROUP,
            research_mode=ResearchMode.BACKGROUND_CHECK,
            organization="Acme Corp",
            role_terms=["CEO", "CFO"],
            explicit_subjects=[],
            raw_query=query,
        )

    monkeypatch.setattr("veris.pipeline.understand_query", fake_understand)

    provider, extract_calls = patched_infra
    provider.responses = {
        "who is the CEO of Acme Corp": [
            (
                "Acme CEO",
                "Jane Doe is CEO of Acme Corp, joined by board member Alice Nguyen at the event.",
                "https://news.example/ceo",
            )
        ],
        "who is the CFO of Acme Corp": [
            ("Acme CFO", "Mark Reyes is CFO of Acme Corp.", "https://news.example/cfo")
        ],
    }
    settings = settings_factory(max_queries_per_plan=1, max_documents_per_subject=5)

    summary = await run_research(
        query="background check on the CEO and CFO of Acme Corp",
        settings=settings,
    )

    assert set(summary.subjects) == {"Jane Doe", "Mark Reyes"}


@pytest.mark.asyncio
async def test_duplicate_url_across_subjects_only_crawled_once(
    patched_infra, settings_factory
):
    provider, extract_calls = patched_infra
    shared_url = "https://news.example/shared"
    provider.responses = {
        "Jane Doe": [
            ("Shared article", "Mentions Jane Doe and Mark Reyes together.", shared_url)
        ],
        "Mark Reyes": [
            ("Shared article", "Mentions Jane Doe and Mark Reyes together.", shared_url)
        ],
    }
    settings = settings_factory(max_queries_per_plan=1, max_documents_per_subject=5)

    await run_research(
        query="Jane Doe and Mark Reyes",
        settings=settings,
        resolved_subjects_override=[
            ResolvedSubject(
                name="Jane Doe",
                subject_type=SubjectType.PERSON,
                resolution_confidence=1.0,
            ),
            ResolvedSubject(
                name="Mark Reyes",
                subject_type=SubjectType.PERSON,
                resolution_confidence=1.0,
            ),
        ],
    )

    assert extract_calls.count(shared_url) == 1  # shared cache prevented a second crawl


@pytest.mark.asyncio
async def test_run_fails_cleanly_when_no_providers_configured(settings_factory):
    with pytest.raises(RuntimeError, match="No search providers configured"):
        await run_research(query="anything", settings=settings_factory())


@pytest.mark.asyncio
async def test_events_persisted_in_stage_order(patched_infra, settings_factory):
    provider, _ = patched_infra
    provider.responses = {
        "Acme": [("Acme news", "Acme reported results.", "https://news.example/acme")]
    }
    settings = settings_factory(max_queries_per_plan=1)

    events_seen = []
    from veris.pipeline import stream_research

    async for event in stream_research(
        query="Acme",
        settings=settings,
        resolved_subjects_override=[
            ResolvedSubject(
                name="Acme", subject_type=SubjectType.TOPIC, resolution_confidence=1.0
            )
        ],
    ):
        events_seen.append(event.stage)

    # sanity: pipeline progresses through the documented spine in order
    first_occurrence = []
    for stage in events_seen:
        if stage not in first_occurrence:
            first_occurrence.append(stage)
    assert first_occurrence == [
        PipelineStage.UNDERSTANDING,
        PipelineStage.RESOLUTION,
        PipelineStage.PLANNING,
        PipelineStage.SEARCH,
        PipelineStage.RETRIEVAL,
        PipelineStage.REGISTRY,
        PipelineStage.EVIDENCE,
        PipelineStage.VERIFICATION,
        PipelineStage.SYNTHESIS,
        PipelineStage.OUTPUT,
    ]

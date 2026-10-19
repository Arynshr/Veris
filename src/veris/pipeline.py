"""
Deterministic asynchronous pipeline orchestrator:
UNDERSTAND -> RESOLVE -> [ for each resolved subject: PLAN -> SEARCH -> RETRIEVE ->
REGISTER -> MAP EVIDENCE -> VERIFY ] -> SYNTHESIZE (all subjects) -> OUTPUT
"""

import asyncio
import shutil
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from veris.analysis import run_evidence_mapping, run_verification
from veris.analysis.safety import load_safety
from veris.analysis.sources import load_source_policy
from veris.analysis.taxonomy import UNCATEGORIZED, load_taxonomy
from veris.core.config import Settings, config_hash, get_logger, get_settings
from veris.core.llm import LLMError, get_llm
from veris.core.ports import Extractor
from veris.core.models import (
    Claim,
    Document,
    EventStatus,
    Evidence,
    PipelineEvent,
    PipelineStage,
    QueryUnderstanding,
    ResearchMode,
    ResearchRun,
    ResolvedSubject,
    RetrievalStats,
    SearchResult,
    Severity,
    SubjectMode,
    VerificationResult,
    VerificationStatus,
)
from veris.core.storage import RunPersistence
from veris.intake import resolve_subjects, run_ingestion, understand_query
from veris.retrieval import (
    CrawlGuard,
    build_extractor,
    build_provider_chain,
    retrieve_documents,
    run_planning,
    search_with_fallback,
)
from veris.retrieval.search import SearchProvider
from veris.synthesis import build_report, generate_summary, render_markdown
from veris.synthesis.slots import make_slot_llm

logger = get_logger(__name__)
_STATE_LOG_LEVEL = {
    EventStatus.STARTED: "debug",
    EventStatus.COMPLETED: "info",
    EventStatus.FAILED: "error",
}


@dataclass
class RunSummary:
    run_id: str
    subjects: list[str]
    research_mode: str
    sources_retrieved: int
    evidence_collected: int
    claims_verified: int
    claims_single_source: int
    claims_conflicted: int
    report_path: str

    def render(self) -> str:
        return (
            "Research Completed\n"
            f"Run id: {self.run_id}\n"
            f"Mode: {self.research_mode}\n"
            f"Subjects Covered: {', '.join(self.subjects)}\n"
            f"Sources retrieved: {self.sources_retrieved}\n"
            f"Evidence collected: {self.evidence_collected}\n"
            f"Claims verified: {self.claims_verified}\n"
            f"Claims single-source: {self.claims_single_source}\n"
            f"Claims conflicted: {self.claims_conflicted}\n"
            f"Report: {self.report_path}\n"
        )


@dataclass
class RunCounters:
    queries: int = 0
    providers: set[str] = field(default_factory=set)
    retrieval: RetrievalStats = field(default_factory=RetrievalStats)
    search_failures: list[str] = field(default_factory=list)
    empty_searches: list[str] = field(default_factory=list)


def _degraded(counters: RunCounters, llm_failures: list[str]) -> list[str]:
    items = {f"search_provider_failed:{f.split(':', 1)[0]}" for f in counters.search_failures}
    items |= {f"search_no_results:{name}" for name in counters.empty_searches}
    if counters.retrieval.attempted and not counters.retrieval.fetched:
        items.add("retrieval_no_documents")
    items |= {f"llm_failed:{f[:100]}" for f in llm_failures}
    return sorted(items)


def _adverse_by_severity(claims: list[Claim]) -> dict[str, int]:
    counts = Counter(
        c.severity.value
        for c in claims
        if c.categories != [UNCATEGORIZED] and not c.out_of_window and not c.flags
    )
    return {s.value: counts[s.value] for s in Severity}


def _count_statuses(
    verifications: list[VerificationResult],
) -> tuple[int, int, int]:
    counts = Counter(v.status for v in verifications)
    return (
        counts[VerificationStatus.VERIFIED],
        counts[VerificationStatus.SINGLE_SOURCE],
        counts[VerificationStatus.CONFLICTED],
    )


def _emit(
    store: RunPersistence,
    stage: PipelineStage,
    status: EventStatus,
    message: str,
    **data,
) -> PipelineEvent:
    event = PipelineEvent(stage=stage, status=status, message=message, data=data)
    subject_note = f"[{data['subject']}]" if "subject" in data else ""
    log_fn = getattr(logger, _STATE_LOG_LEVEL[status])
    log_fn("%-13s %-9s%s %s", stage.value, status.value, subject_note, message)
    store.append_event(event)
    return event


async def _run_subject_pipeline(
    subject: ResolvedSubject,
    ingestion,
    settings: Settings,
    store: RunPersistence,
    research_mode: ResearchMode,
    provider_chain: list[SearchProvider],
    collected_claims: list[Claim],
    collected_verifications: list[VerificationResult],
    collected_evidence: list[Evidence],
    search_cache: dict[str, list[SearchResult]],
    document_cache: dict[str, Document | None],
    extractor: Extractor,
    guard: CrawlGuard,
    counters: RunCounters,
) -> AsyncIterator[PipelineEvent]:
    """PLAN -> SEARCH -> RETRIEVE -> REGISTER -> EVIDENCE -> VERIFY

    `search_cache` and `document_cache` are shared across every subject in
    the run (see stream_research), so a query or URL already handled for one
    resolved subject is never re-searched/re-crawled for another.
    """
    name = subject.name
    news_bias = research_mode == ResearchMode.BACKGROUND_CHECK

    yield _emit(
        store,
        PipelineStage.PLANNING,
        EventStatus.STARTED,
        f"Planning queries for {name}",
        subject=name,
    )
    plan = run_planning(settings, subject=name, research_mode=research_mode)
    store.save_plan(plan, subject=name)
    counters.queries += len(plan.queries)
    yield _emit(
        store,
        PipelineStage.PLANNING,
        EventStatus.COMPLETED,
        f"{len(plan.queries)} queries planned",
        subject=name,
        queries=len(plan.queries),
    )

    yield _emit(
        store,
        PipelineStage.SEARCH,
        EventStatus.STARTED,
        f"Searching {len(plan.queries)} queries for {name}",
        subject=name,
    )
    batches = await asyncio.gather(
        *(
            search_with_fallback(
                q,
                settings,
                chain=provider_chain,
                news_bias=news_bias,
                cache=search_cache,
                failures=counters.search_failures,
            )
            for q in plan.queries
        )
    )
    search_results = [r for batch in batches for r in batch]
    store.save_search_results(search_results, subject=name)
    counters.providers.update(r.provider.value for r in search_results)
    if search_results:
        yield _emit(
            store,
            PipelineStage.SEARCH,
            EventStatus.COMPLETED,
            f"{len(search_results)} results found",
            subject=name,
            results=len(search_results),
        )
    else:
        counters.empty_searches.append(name)
        yield _emit(
            store,
            PipelineStage.SEARCH,
            EventStatus.FAILED,
            f"No results from any provider for {name}; continuing with no sources",
            subject=name,
        )

    yield _emit(
        store,
        PipelineStage.RETRIEVAL,
        EventStatus.STARTED,
        f"Extracting up to {settings.max_documents_per_subject} pages for {name}",
        subject=name,
    )
    documents = await retrieve_documents(
        search_results,
        extractor,
        guard,
        cache=document_cache,
        max_documents=settings.max_documents_per_subject,
        stats=counters.retrieval,
        retries=settings.extraction_retries,
    )
    yield _emit(
        store,
        PipelineStage.RETRIEVAL,
        EventStatus.COMPLETED,
        f"{len(documents)} documents extracted",
        subject=name,
        documents=len(documents),
    )

    yield _emit(
        store,
        PipelineStage.REGISTRY,
        EventStatus.STARTED,
        f"Registering documents for {name}",
        subject=name,
    )
    registered = store.register_documents(documents)
    yield _emit(
        store,
        PipelineStage.REGISTRY,
        EventStatus.COMPLETED,
        f"{len(registered)} new unique documents ({len(documents) - len(registered)} already seen this run)",
        subject=name,
        unique_documents=len(registered),
    )

    yield _emit(
        store,
        PipelineStage.EVIDENCE,
        EventStatus.STARTED,
        f"Mapping evidence for {name}",
        subject=name,
    )
    evidence_list = run_evidence_mapping(
        registered,
        subject=name,
        entities=ingestion.entities,
        settings=settings,
        research_mode=research_mode,
    )
    store.save_evidence(evidence_list, subject=name)
    yield _emit(
        store,
        PipelineStage.EVIDENCE,
        EventStatus.COMPLETED,
        f"{len(evidence_list)} evidence items mapped",
        subject=name,
        evidence=len(evidence_list),
    )

    yield _emit(
        store,
        PipelineStage.VERIFICATION,
        EventStatus.STARTED,
        f"Verifying claims for {name}",
        subject=name,
    )
    claims, verifications = run_verification(
        evidence_list,
        ingestion.entities,
        subject=name,
        settings=settings,
        research_mode=research_mode,
    )
    store.save_claims(claims, subject=name)
    store.save_verification(verifications, subject=name)
    collected_claims.extend(claims)
    collected_verifications.extend(verifications)
    collected_evidence.extend(evidence_list)
    verified, single_source, conflicted = _count_statuses(verifications)
    yield _emit(
        store,
        PipelineStage.VERIFICATION,
        EventStatus.COMPLETED,
        f"{verified} verified, {single_source} single-source, {conflicted} conflicted",
        subject=name,
        verified=verified,
        single_source=single_source,
        conflicted=conflicted,
    )


def _validate_policy_files(settings: Settings) -> None:
    """Fail before any search/crawl spend if a policy YAML is invalid."""
    load_source_policy(settings.source_tiers_path)
    load_taxonomy(settings.taxonomy_path)
    load_safety(settings.safety_path)


async def stream_research(
    query: str,
    output_path: str = "report.md",
    settings: Settings | None = None,
    run_id: str | None = None,
    resolved_subjects_override: list[ResolvedSubject] | None = None,
    research_mode_override: ResearchMode | None = None,
) -> AsyncIterator[PipelineEvent]:
    settings = settings or get_settings()
    store = RunPersistence(settings, run_id=run_id)
    store.init_dirs()

    yield _emit(
        store, PipelineStage.UNDERSTANDING, EventStatus.STARTED, "Parsing request"
    )

    try:
        _validate_policy_files(settings)
    except ValueError as exc:
        yield _emit(store, PipelineStage.UNDERSTANDING, EventStatus.FAILED, str(exc))
        raise

    try:
        provider_chain = build_provider_chain(settings)
    except RuntimeError as exc:
        yield _emit(store, PipelineStage.UNDERSTANDING, EventStatus.FAILED, str(exc))
        raise

    try:
        llm = get_llm(settings)
        await asyncio.to_thread(llm.validate)
    except LLMError as exc:
        yield _emit(store, PipelineStage.UNDERSTANDING, EventStatus.FAILED, str(exc))
        raise

    llm_failures_start = len(llm.failures)
    ingestion = run_ingestion(user_prompt=query, system_prompt=settings.system_prompt)
    store.save_entities(ingestion.entities)

    # Shared across the whole run (resolution + every resolved subject's
    # plan) so an identical query is never re-sent to a provider twice, and
    # an identical URL is never re-crawled twice.
    search_cache: dict[str, list[SearchResult]] = {}
    document_cache: dict[str, Document | None] = {}
    counters = RunCounters()
    guard = CrawlGuard(settings)
    extractor = build_extractor(settings)

    if resolved_subjects_override:
        understanding = QueryUnderstanding(
            mode=SubjectMode.SINGLE,
            research_mode=research_mode_override or ResearchMode.GENERAL,
            explicit_subjects=[s.name for s in resolved_subjects_override],
            raw_query=query,
        )
    else:
        understanding = await understand_query(query, ingestion.entities, llm)
        if research_mode_override:
            understanding.research_mode = research_mode_override
    store.save_understanding(understanding)
    yield _emit(
        store,
        PipelineStage.UNDERSTANDING,
        EventStatus.COMPLETED,
        f"Mode={understanding.mode.value}, research_mode={understanding.research_mode.value}"
        + (f", org={understanding.organization}" if understanding.organization else ""),
        mode=understanding.mode.value,
        research_mode=understanding.research_mode.value,
    )

    yield _emit(
        store,
        PipelineStage.RESOLUTION,
        EventStatus.STARTED,
        "Resolving research subjects",
    )
    if resolved_subjects_override:
        resolved = resolved_subjects_override
    else:
        try:
            resolved = await resolve_subjects(
                understanding, settings, provider_chain, search_cache=search_cache, llm=llm
            )
        except Exception as exc:
            yield _emit(store, PipelineStage.RESOLUTION, EventStatus.FAILED, str(exc))
            raise
    if not resolved:
        message = "Could not resolve any concrete subjects from this request - try naming individuals/topics directly."
        yield _emit(store, PipelineStage.RESOLUTION, EventStatus.FAILED, message)
        raise RuntimeError(message)
    store.save_resolution(resolved)
    yield _emit(
        store,
        PipelineStage.RESOLUTION,
        EventStatus.COMPLETED,
        f"{len(resolved)} subject(s): {', '.join(s.name for s in resolved)}",
        subjects=[s.name for s in resolved],
    )

    all_claims: list[Claim] = []
    all_verifications: list[VerificationResult] = []
    all_evidence: list[Evidence] = []

    for subject in resolved:
        try:
            async for event in _run_subject_pipeline(
                subject,
                ingestion,
                settings,
                store,
                understanding.research_mode,
                provider_chain,
                all_claims,
                all_verifications,
                all_evidence,
                search_cache,
                document_cache,
                extractor,
                guard,
                counters,
            ):
                yield event
        except Exception as exc:
            yield _emit(
                store,
                PipelineStage.VERIFICATION,
                EventStatus.FAILED,
                f"{subject.name}: {exc}",
                subject=subject.name,
            )
            await extractor.close()
            raise

    await extractor.close()

    total_verified, total_single_source, total_conflicted = _count_statuses(
        all_verifications
    )
    total_documents = store.total_documents_registered

    yield _emit(
        store, PipelineStage.SYNTHESIS, EventStatus.STARTED, "Synthesizing report"
    )
    try:
        summary = await generate_summary(
            all_claims, all_verifications, all_evidence, settings, make_slot_llm(llm)
        )
    except Exception as exc:
        yield _emit(store, PipelineStage.SYNTHESIS, EventStatus.FAILED, str(exc))
        raise
    yield _emit(
        store, PipelineStage.SYNTHESIS, EventStatus.COMPLETED, "Executive summary ready"
    )

    yield _emit(
        store, PipelineStage.OUTPUT, EventStatus.STARTED, "Writing report to disk"
    )
    degraded = _degraded(counters, llm.failures[llm_failures_start:])
    run_record = ResearchRun(
        run_id=store.run_id,
        query=query,
        mode=understanding.mode,
        research_mode=understanding.research_mode,
        subjects=[s.name for s in resolved],
        output_path="",
        sources_retrieved=total_documents,
        evidence_collected=len(all_evidence),
        claims_verified=total_verified,
        claims_single_source=total_single_source,
        claims_conflicted=total_conflicted,
        queries_run=counters.queries,
        providers_used=sorted(counters.providers),
        retrieval=counters.retrieval,
        adverse_by_severity=_adverse_by_severity(all_claims),
        degraded=degraded,
        config_hash=config_hash(settings),
    )
    try:
        report = build_report(
            run_record,
            resolved,
            all_claims,
            all_verifications,
            all_evidence,
            {d["document_id"]: str(d["retrieved_at"]) for d in store.load_documents()},
            settings,
            summary,
            load_taxonomy(settings.taxonomy_path).category_order,
        )
        store.save_report_json(report)
        report_path = store.save_report(render_markdown(report, settings))
        final_path: Path = report_path
        if output_path and output_path != "report.md":
            final_path = Path(output_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(report_path, final_path)
        store.save_run_record(run_record.model_copy(update={"output_path": str(final_path)}))
    except Exception as exc:
        yield _emit(store, PipelineStage.OUTPUT, EventStatus.FAILED, str(exc))
        raise
    yield _emit(
        store,
        PipelineStage.OUTPUT,
        EventStatus.COMPLETED,
        f"Report written to {final_path}",
        run_id=store.run_id,
        report_path=str(final_path),
        subjects=[s.name for s in resolved],
        research_mode=understanding.research_mode.value,
        sources_retrieved=total_documents,
        evidence_collected=len(all_evidence),
        claims_verified=total_verified,
        claims_single_source=total_single_source,
        claims_conflicted=total_conflicted,
        degraded=degraded,
    )


async def run_research(
    query: str,
    output_path: str = "report.md",
    settings: Settings | None = None,
    run_id: str | None = None,
    resolved_subjects_override: list[ResolvedSubject] | None = None,
    research_mode_override: ResearchMode | None = None,
) -> RunSummary:
    """Non-streaming wrapper: drains stream_research and returns the final summary."""
    final_data: dict = {}
    async for event in stream_research(
        query=query,
        output_path=output_path,
        settings=settings,
        run_id=run_id,
        resolved_subjects_override=resolved_subjects_override,
        research_mode_override=research_mode_override,
    ):
        if (
            event.stage == PipelineStage.OUTPUT
            and event.status == EventStatus.COMPLETED
        ):
            final_data = event.data

    return RunSummary(
        run_id=final_data["run_id"],
        subjects=final_data["subjects"],
        research_mode=final_data["research_mode"],
        sources_retrieved=final_data["sources_retrieved"],
        evidence_collected=final_data["evidence_collected"],
        claims_verified=final_data["claims_verified"],
        claims_single_source=final_data["claims_single_source"],
        claims_conflicted=final_data["claims_conflicted"],
        report_path=final_data["report_path"],
    )

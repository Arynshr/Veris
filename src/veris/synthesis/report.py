"""Canonical report: build `Report` from run data; Markdown is rendered only from it."""

import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from veris.analysis.taxonomy import UNCATEGORIZED
from veris.analysis.temporal import window_start
from veris.core.config import Settings
from veris.core.models import (
    AppendixRow,
    CategoryRow,
    Claim,
    Coverage,
    Evidence,
    Report,
    ReportFinding,
    ReportMeta,
    ResearchMode,
    ResearchRun,
    ResolvedSubject,
    Severity,
    SubjectSection,
    SynthesisSlot,
    VerificationResult,
    VerificationStatus,
)
from veris.synthesis.ordering import claim_relevance, claim_sort_key

_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>|])")
_URL_SAFE = ":/?#[]@!$&'()*+,;=%~-._"
_Item = tuple[Claim, VerificationResult, list[Evidence]]


def _esc(value: object) -> str:
    return _MD_SPECIAL.sub(r"\\\1", " ".join(str(value).split()))


def _url(value: str) -> str:
    return f"<{quote(value.strip(), safe=_URL_SAFE)}>"


@lru_cache
def _env(root: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(root),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters.update(esc=_esc, url=_url)
    return env


def render_markdown(report: Report, settings: Settings) -> str:
    root = settings.templates_dir / report.template_version
    if not root.is_dir():
        raise FileNotFoundError(f"Report templates not found: {root}")
    return _env(root).get_template("report.md.j2").render(r=report.model_dump(mode="json"))


def _no_findings_note(adverse: bool, attempted: int, fetched: int, degraded: list[str]) -> str:
    what = "adverse findings" if adverse else "findings"
    if not attempted:
        return f"No {what} in retrieved sources; no sources could be retrieved."
    if attempted > fetched:
        return f"No {what} in retrieved sources; {attempted - fetched} sources could not be retrieved."
    if degraded:
        return f"No {what} in retrieved sources; coverage was incomplete (see Coverage)."
    return f"No {what} in the {fetched} sources retrieved."


def _finding(claim: Claim, verdict: VerificationResult, evidence: Evidence) -> ReportFinding:
    return ReportFinding(
        claim_id=claim.claim_id,
        text=claim.text,
        categories=claim.categories,
        legal_status=claim.legal_status,
        severity=claim.severity,
        verification=verdict.status,
        event_date=claim.event_date,
        undated=claim.undated,
        source=evidence.source_metadata.source,
        url=evidence.source_metadata.url,
        conflicts_with=sorted(verdict.conflicts_with),
    )


def _passage(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_report(
    run: ResearchRun,
    subjects: list[ResolvedSubject],
    claims: list[Claim],
    verifications: list[VerificationResult],
    evidence_list: list[Evidence],
    retrieved_at: dict[str, str],
    settings: Settings,
    summary: SynthesisSlot | None,
    category_order: list[str],
) -> Report:
    as_of = settings.as_of_date or date.today()
    evidence_by_id = {e.evidence_id: e for e in evidence_list}
    verdicts = {v.claim_id: v for v in verifications}
    adverse_mode = run.research_mode == ResearchMode.BACKGROUND_CHECK

    items: list[_Item] = []
    out_of_window = excluded_identity = excluded_unsupported = 0
    for claim in claims:
        verdict = verdicts.get(claim.claim_id)
        clean = [evidence_by_id[i] for i in claim.evidence_ids if i in evidence_by_id and not evidence_by_id[i].flags]
        if claim.out_of_window:
            out_of_window += 1
        elif claim.flags or not clean or verdict is None:
            excluded_unsupported += 1
        elif verdict.status == VerificationStatus.CONFLICTED and not verdict.conflicts_with:
            excluded_identity += 1
        elif not (adverse_mode and claim.categories == [UNCATEGORIZED]):
            items.append((claim, verdict, clean))
    items.sort(key=lambda t: claim_sort_key(t[0], claim_relevance(t[0], evidence_by_id)))

    only_direct = len(subjects) == 1 and subjects[0].resolution_confidence >= 0.99
    sections: list[SubjectSection] = []
    category_rows: list[CategoryRow] = []
    appendix: list[AppendixRow] = []
    for subject in subjects:
        mine = [t for t in items if t[0].subject == subject.name]
        established = [t for t in mine if t[1].status != VerificationStatus.CONFLICTED]
        disputed = [t for t in mine if t[1].status == VerificationStatus.CONFLICTED]
        if subject.resolution_confidence >= 0.99:
            note = "" if only_direct else "given directly"
        else:
            note = f"resolved via search, confidence {subject.resolution_confidence:.1f}"
        sections.append(
            SubjectSection(
                name=subject.name,
                role=subject.role,
                resolution_note=note,
                findings=[_finding(c, v, e[0]) for c, v, e in established],
                disputed=[_finding(c, v, e[0]) for c, v, e in disputed],
            )
        )
        for category in category_order:
            rows = [t for t in established if category in t[0].categories]
            if rows:
                count = lambda sev: sum(1 for t in rows if t[0].severity == sev)  # noqa: E731
                category_rows.append(
                    CategoryRow(
                        subject=subject.name,
                        category=category,
                        high=count(Severity.HIGH),
                        medium=count(Severity.MEDIUM),
                        low=count(Severity.LOW),
                        claim_ids=[t[0].claim_id for t in rows],
                    )
                )
        for claim, verdict, clean in established + disputed:
            appendix += [
                AppendixRow(
                    claim_id=claim.claim_id,
                    passage=_passage(e.supporting_passage, settings.report_passage_max_chars),
                    url=e.source_metadata.url,
                    retrieved_at=retrieved_at.get(e.document_id, ""),
                    status=verdict.status,
                )
                for e in clean
            ]

    dates = [c.event_date for c, _, _ in items if c.event_date]
    retrieval = run.retrieval
    coverage = Coverage(
        queries_run=run.queries_run,
        providers_used=run.providers_used,
        retrieval=retrieval,
        sources_registered=run.sources_retrieved,
        unretrieved=max(retrieval.attempted - retrieval.fetched, 0),
        earliest_event=min(dates, default=None),
        latest_event=max(dates, default=None),
        lookback_years=settings.lookback_years,
        window_start=window_start(as_of, settings.lookback_years),
        as_of=as_of,
        claims_total=len(claims),
        claims_reported=len(items),
        out_of_window=out_of_window,
        undated=sum(1 for c, _, _ in items if c.undated),
        excluded_identity=excluded_identity,
        excluded_unsupported=excluded_unsupported,
        degraded=run.degraded,
        no_findings_note=_no_findings_note(adverse_mode, retrieval.attempted, retrieval.fetched, run.degraded),
    )
    return Report(
        template_version=settings.template_version,
        run_id=run.run_id,
        config_hash=run.config_hash,
        meta=ReportMeta(
            query=run.query,
            research_mode=run.research_mode,
            min_independent_domains=settings.min_independent_domains,
        ),
        summary=summary,
        subjects=sections,
        category_rows=category_rows,
        appendix=appendix,
        coverage=coverage,
    )

"""Canonical report: build `Report` from run data; Markdown is rendered only from it."""

import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from veris.analysis.cleaning import IDENTITY_MISMATCH, NOISE, CleaningConfig, clean_text, load_cleaning, truncate
from veris.analysis.sources import load_source_policy
from veris.analysis.taxonomy import UNCATEGORIZED
from veris.analysis.temporal import window_start
from veris.core.config import Settings
from veris.core.documents import canonicalize_url
from veris.core.models import (
    AppendixRow,
    CategoryRow,
    Claim,
    Coverage,
    Evidence,
    Reference,
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
_NORMALIZE_RE = re.compile(r"[^\w]+")
_SECONDARY_ORDER = ("beyond top findings", "low relevance")


@dataclass
class _Entry:
    claim: Claim
    verdict: VerificationResult
    evidence: list[Evidence]
    relevance: float
    merged: list[str] = field(default_factory=list)
    reason: str = ""


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


def _dedup_key(claim: Claim) -> tuple[str, str]:
    return _NORMALIZE_RE.sub(" ", claim.text.casefold()).strip(), claim.legal_status.value


def _merge_duplicates(entries: list[_Entry]) -> list[_Entry]:
    """Fold same-text claims (e.g. a Wikipedia sentence repeated across pages) into
    the best-ranked one; its evidence and claim IDs are kept so provenance holds."""
    kept: dict[tuple[str, str], _Entry] = {}
    for entry in entries:
        first = kept.get(_dedup_key(entry.claim))
        if first is None:
            kept[_dedup_key(entry.claim)] = entry
            continue
        first.merged.append(entry.claim.claim_id)
        seen = {e.evidence_id for e in first.evidence}
        first.evidence += [e for e in entry.evidence if e.evidence_id not in seen]
    return list(kept.values())


def _label(value: str) -> str:
    return value.replace("_", " ").title()


def _title(claim: Claim, adverse: bool) -> str:
    year = str(claim.event_date.year) if claim.event_date else "undated"
    if not adverse:
        return f"Finding · {year}"
    parts = [_label(claim.categories[0])]
    if claim.legal_status.value != "unknown":
        parts.append(_label(claim.legal_status.value))
    return " · ".join([*parts, year])


def _assign_refs(evidence: list[Evidence], retrieved_at: dict[str, str], settings: Settings) -> tuple[list[Reference], dict[str, str]]:
    """Registry keyed by canonical URL; ref_ids follow sorted canonical URL (stable across runs)."""
    policy = load_source_policy(settings.source_tiers_path)
    by_url: dict[str, Evidence] = {}
    for e in sorted(evidence, key=lambda e: e.evidence_id):
        by_url.setdefault(canonicalize_url(e.source_metadata.url), e)
    refs, ref_of_url = [], {}
    for n, url in enumerate(sorted(by_url), start=1):
        e = by_url[url]
        domain = e.source_metadata.source.lower().removeprefix("www.")
        info = policy.classify(domain)
        ref_of_url[url] = f"R{n}"
        refs.append(
            Reference(
                ref_id=f"R{n}",
                document_id=e.document_id,
                url=e.source_metadata.url,
                title=" ".join(e.source_metadata.title.split()) or domain,
                domain=domain,
                tier=info.tier,
                source_type=info.source_type.value,
                published_at=(e.source_metadata.published_at or "")[:10] or None,
                retrieved_at=retrieved_at.get(e.document_id, ""),
            )
        )
    return refs, ref_of_url


def _ref_ids(evidence: list[Evidence], ref_of_url: dict[str, str]) -> list[str]:
    ids = {ref_of_url[canonicalize_url(e.source_metadata.url)] for e in evidence}
    return sorted(ids, key=lambda r: int(r[1:]))


def _finding(entry: _Entry, ref_of_url: dict[str, str], conflicts: list[str]) -> ReportFinding:
    claim = entry.claim
    adverse = claim.categories != [UNCATEGORIZED]
    return ReportFinding(
        claim_id=claim.claim_id,
        merged_claim_ids=sorted(entry.merged),
        title=_title(claim, adverse),
        snippet=claim.text,
        categories=claim.categories,
        adverse=adverse,
        legal_status=claim.legal_status,
        severity=claim.severity,
        verification=entry.verdict.status,
        event_date=claim.event_date,
        undated=claim.undated,
        relevance=round(entry.relevance, 3),
        ref_ids=_ref_ids(entry.evidence, ref_of_url),
        conflicts_with=conflicts,
        secondary_reason=entry.reason,
    )


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
    cleaning: CleaningConfig = load_cleaning(settings.cleaning_path)
    evidence_by_id = {e.evidence_id: e for e in evidence_list}
    verdicts = {v.claim_id: v for v in verifications}
    adverse_mode = run.research_mode == ResearchMode.BACKGROUND_CHECK

    entries: list[_Entry] = []
    out_of_window = excluded_identity = excluded_noise = excluded_unsupported = 0
    for claim in claims:
        verdict = verdicts.get(claim.claim_id)
        clean = [evidence_by_id[i] for i in claim.evidence_ids if i in evidence_by_id and not evidence_by_id[i].flags]
        if claim.out_of_window:
            out_of_window += 1
        elif IDENTITY_MISMATCH in claim.flags:
            excluded_identity += 1
        elif NOISE in claim.flags:
            excluded_noise += 1
        elif claim.flags or not clean or verdict is None:
            excluded_unsupported += 1
        elif verdict.status == VerificationStatus.CONFLICTED and not verdict.conflicts_with:
            excluded_identity += 1
        elif not (adverse_mode and claim.categories == [UNCATEGORIZED]):
            entries.append(_Entry(claim, verdict, clean, claim_relevance(claim, evidence_by_id)))
    entries.sort(key=lambda t: claim_sort_key(t.claim, t.relevance))

    only_direct = len(subjects) == 1 and subjects[0].resolution_confidence >= 0.99
    prepared: list[tuple[ResolvedSubject, str, list[_Entry], list[_Entry], list[_Entry]]] = []
    for subject in subjects:
        mine = [t for t in entries if t.claim.subject == subject.name]
        disputed = [t for t in mine if t.verdict.status == VerificationStatus.CONFLICTED]
        established = _merge_duplicates([t for t in mine if t.verdict.status != VerificationStatus.CONFLICTED])
        main: list[_Entry] = []
        secondary: list[_Entry] = []
        for entry in established:
            if entry.relevance < settings.min_finding_relevance:
                entry.reason = "low relevance"
            elif len(main) >= settings.max_findings_per_subject:
                entry.reason = "beyond top findings"
            (secondary if entry.reason else main).append(entry)
        secondary.sort(key=lambda t: _SECONDARY_ORDER.index(t.reason))
        if subject.resolution_confidence >= 0.99:
            note = "" if only_direct else "given directly"
        else:
            note = f"resolved via search, confidence {subject.resolution_confidence:.1f}"
        prepared.append((subject, note, main, secondary, disputed))

    summary_claims = [c for c in claims if summary and c.claim_id in summary.claim_ids]
    summary_evidence = [
        evidence_by_id[i] for c in summary_claims for i in c.evidence_ids if i in evidence_by_id and not evidence_by_id[i].flags
    ]
    reported = [t for _, _, m, s, d in prepared for t in (*m, *s, *d)]
    references, ref_by_url = _assign_refs(
        [e for t in reported for e in t.evidence] + summary_evidence, retrieved_at, settings
    )

    sections: list[SubjectSection] = []
    category_rows: list[CategoryRow] = []
    appendix: list[AppendixRow] = []
    for subject, note, main, secondary, disputed in prepared:
        sections.append(
            SubjectSection(
                name=subject.name,
                role=subject.role,
                resolution_note=note,
                findings=[_finding(t, ref_by_url, []) for t in main],
                secondary=[_finding(t, ref_by_url, []) for t in secondary],
                disputed=[_finding(t, ref_by_url, sorted(t.verdict.conflicts_with)) for t in disputed],
            )
        )
        for category in category_order:
            rows = [t for t in (*main, *secondary) if category in t.claim.categories]
            if rows:
                count = lambda sev: sum(1 for t in rows if t.claim.severity == sev)  # noqa: E731
                category_rows.append(
                    CategoryRow(
                        subject=subject.name,
                        category=category,
                        high=count(Severity.HIGH),
                        medium=count(Severity.MEDIUM),
                        low=count(Severity.LOW),
                        claim_ids=[t.claim.claim_id for t in rows],
                    )
                )
        for t in (*main, *secondary, *disputed):
            appendix += [
                AppendixRow(
                    claim_id=t.claim.claim_id,
                    passage=truncate(clean_text(e.supporting_passage, cleaning), settings.report_passage_max_chars),
                    ref_id=ref_by_url[canonicalize_url(e.source_metadata.url)],
                    url=e.source_metadata.url,
                    retrieved_at=retrieved_at.get(e.document_id, ""),
                    status=t.verdict.status,
                )
                for e in sorted(t.evidence, key=lambda e: e.evidence_id)
            ]

    dates = [t.claim.event_date for t in entries if t.claim.event_date]
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
        claims_reported=len(entries),
        claims_secondary=sum(len(s.secondary) for s in sections),
        out_of_window=out_of_window,
        undated=sum(1 for t in entries if t.claim.undated),
        excluded_identity=excluded_identity,
        excluded_noise=excluded_noise,
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
        summary_ref_ids=_ref_ids(summary_evidence, ref_by_url),
        subjects=sections,
        category_rows=category_rows,
        references=references,
        appendix=appendix,
        coverage=coverage,
    )

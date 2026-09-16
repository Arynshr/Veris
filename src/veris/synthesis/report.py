""" 
Only VERIFIED claims are handed to the model; 
CONFLICTED claims are never presented as fact. 
GENERAL mode produces a plain cited research report; 
BACKGROUND_CHECK mode produces an adverse-media
report grouped subject -> risk category, with an identity/screening disclaimer.
Deterministic non-LLM fallback in both modes so the pipeline never hard-fails.
"""
from collections import defaultdict

from groq import Groq

from veris.core.config import Settings
from veris.core.models import (
    AdverseCategory,
    Claim,
    Evidence,
    ResearchMode,
    ResolvedSubject,
    VerificationResult,
    VerificationStatus,
)

_GENERAL_SYSTEM_PROMPT = """You are writing a cited research report on one or more subjects. \
You MUST only use the verified findings and evidence provided below - never invent facts, \
dates, or sources. Every substantive sentence must be followed by a citation marker like \
[S1], [S2] referencing the numbered sources list. Structure the report with one heading per \
subject (skip subject headings if there's only one); write a coherent narrative synthesis \
under each, not just a bullet dump of the claims - group related points, note agreement or \
disagreement between sources where evidence shows it. End with a "Sources" section listing \
every numbered source and its URL. Output clean Markdown."""

_BACKGROUND_CHECK_SYSTEM_PROMPT = """You are writing a multi-subject adverse-media \
background-check report. You MUST only use the verified findings and evidence provided \
below - never invent facts, dates, or sources. Describe every finding as "reported by \
<source>" or "according to <source>", never as an established fact, since this is a \
screening aid and not a legal or compliance determination. Structure the report with one \
heading per subject; within each subject, group findings under: Legal & Litigation, \
Regulatory & Sanctions, Criminal, Financial, Reputational & Other - omitting any heading \
with no findings for that subject, and omitting any subject with no findings entirely (just \
note "no adverse findings" for them). Every substantive sentence must be followed by a \
citation marker like [S1], [S2] referencing the numbered sources list. End with a "Sources" \
section listing every numbered source and its URL. Output clean Markdown."""

_CATEGORY_HEADINGS: dict[AdverseCategory, str] = {
    AdverseCategory.LEGAL: "Legal & Litigation",
    AdverseCategory.REGULATORY: "Regulatory & Sanctions",
    AdverseCategory.CRIMINAL: "Criminal",
    AdverseCategory.FINANCIAL: "Financial",
    AdverseCategory.REPUTATIONAL: "Reputational & Other",
}

_SCREENING_DISCLAIMER = (
    "> **Screening aid only.** Findings below are drawn from public web sources and are "
    "not verified beyond automated subject-name and source-quality checks. Common names "
    "can produce false matches - confirm identity details (role, location, employer) "
    "before relying on any finding. This is not a legal, credit, or compliance determination."
)


def _resolution_summary(subjects: list[ResolvedSubject]) -> str:
    if len(subjects) == 1 and subjects[0].resolution_confidence >= 0.99:
        return ""  # single directly-named subject - resolution transparency adds no value
    lines = ["## Subjects Covered", ""]
    for s in subjects:
        confidence_note = "given directly" if s.resolution_confidence >= 0.99 else f"resolved via search, confidence {s.resolution_confidence:.1f}"
        role_note = f" — {s.role}" if s.role else ""
        lines.append(f"- **{s.name}**{role_note} ({confidence_note})")
    lines.append("")
    return "\n".join(lines)


def _grouped_by_subject_category(
    claims: list[Claim], verifications: list[VerificationResult], evidence_list: list[Evidence]
) -> dict[str, list[tuple[Claim, Evidence]]]:
    """subject -> [(claim, evidence)], sorted by evidence relevance_score descending."""
    evidence_by_id = {e.evidence_id: e for e in evidence_list}
    claim_by_id = {c.claim_id: c for c in claims}
    verified_ids = {v.claim_id for v in verifications if v.status == VerificationStatus.VERIFIED}

    grouped: dict[str, list[tuple[Claim, Evidence]]] = defaultdict(list)
    for claim_id in verified_ids:
        claim = claim_by_id.get(claim_id)
        if not claim:
            continue
        for eid in claim.evidence_ids:
            ev = evidence_by_id.get(eid)
            if ev:
                grouped[claim.subject].append((claim, ev))
    for subject in grouped:
        grouped[subject].sort(key=lambda pair: pair[1].relevance_score, reverse=True)
    return grouped


def _build_sources_block(evidence_items: list[Evidence]) -> str:
    seen: dict[str, int] = {}
    lines = []
    for ev in evidence_items:
        key = ev.source_metadata.url
        if key not in seen:
            seen[key] = len(seen) + 1
            lines.append(f"[S{seen[key]}] {ev.source_metadata.title} - {key}")
    return "\n".join(lines)


def _fallback_general_report(subjects: list[ResolvedSubject], grouped: dict[str, list[tuple[Claim, Evidence]]]) -> str:
    lines = ["# Research Report", ""]
    resolution = _resolution_summary(subjects)
    if resolution:
        lines.append(resolution)
    all_evidence: list[Evidence] = []
    for subject in subjects:
        items = grouped.get(subject.name, [])
        if len(subjects) > 1:
            lines.append(f"## {subject.name}")
        if not items:
            lines.append("_No findings passed verification for this subject._\n")
            continue
        for claim, ev in items:
            all_evidence.append(ev)
            lines.append(f"- {claim.text} ({ev.source_metadata.source}, {ev.source_metadata.url})")
        lines.append("")
    lines.append("## Sources")
    lines.append(_build_sources_block(all_evidence))
    return "\n".join(lines)


def _fallback_background_check_report(subjects: list[ResolvedSubject], grouped: dict[str, list[tuple[Claim, Evidence]]]) -> str:
    lines = ["# Background Check Report", "", _SCREENING_DISCLAIMER, "", _resolution_summary(subjects)]
    all_evidence: list[Evidence] = []
    for subject in subjects:
        items = grouped.get(subject.name, [])
        lines.append(f"## {subject.name}")
        if not items:
            lines.append("_No adverse findings passed verification for this subject._\n")
            continue
        by_category: dict[AdverseCategory, list[tuple[Claim, Evidence]]] = defaultdict(list)
        for claim, ev in items:
            by_category[ev.category].append((claim, ev))
        for category, heading in _CATEGORY_HEADINGS.items():
            cat_items = by_category.get(category, [])
            if not cat_items:
                continue
            lines.append(f"### {heading}")
            for claim, ev in cat_items:
                all_evidence.append(ev)
                lines.append(f"- {claim.text} (reported by {ev.source_metadata.source}, {ev.source_metadata.url})")
            lines.append("")
    lines.append("## Sources")
    lines.append(_build_sources_block(all_evidence))
    return "\n".join(lines)


def run_synthesis(
    subjects: list[ResolvedSubject],
    claims: list[Claim],
    verifications: list[VerificationResult],
    evidence_list: list[Evidence],
    settings: Settings,
    research_mode: ResearchMode = ResearchMode.GENERAL,
) -> str:
    grouped = _grouped_by_subject_category(claims, verifications, evidence_list)
    is_background_check = research_mode == ResearchMode.BACKGROUND_CHECK

    if not settings.groq_api_key:
        return _fallback_background_check_report(subjects, grouped) if is_background_check else _fallback_general_report(subjects, grouped)

    all_evidence = [ev for items in grouped.values() for _, ev in items]
    if not all_evidence:
        title = "# Background Check Report" if is_background_check else "# Research Report"
        no_findings = (
            "No adverse findings passed verification for any subject. Either no adverse "
            "coverage was found, or candidate matches could not be confidently attributed."
            if is_background_check else
            "No claims passed verification for any subject. Consider broadening the query "
            "or adding search providers."
        )
        disclaimer = f"\n\n{_SCREENING_DISCLAIMER}" if is_background_check else ""
        return f"{title}{disclaimer}\n\n{_resolution_summary(subjects)}{no_findings}\n"

    sources_block = _build_sources_block(all_evidence)
    findings_lines = []
    for subject in subjects:
        items = grouped.get(subject.name, [])
        findings_lines.append(f"## {subject.name}")
        if not items:
            findings_lines.append("(no verified findings)")
            continue
        for claim, ev in items:
            findings_lines.append(f"- {claim.text} (source: {ev.source_metadata.url})")
    findings_block = "\n".join(findings_lines)

    client = Groq(api_key=settings.groq_api_key)
    system_prompt = _BACKGROUND_CHECK_SYSTEM_PROMPT if is_background_check else _GENERAL_SYSTEM_PROMPT
    user_msg = (
        f"Subjects: {', '.join(s.name for s in subjects)}\n\n"
        f"Verified findings by subject:\n{findings_block}\n\n"
        f"Numbered sources:\n{sources_block}\n\n"
        "Write the Markdown report now."
    )
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
        temperature=0.2,
    )
    body = response.choices[0].message.content
    resolution = _resolution_summary(subjects)
    return f"{resolution}\n{body}" if resolution else body

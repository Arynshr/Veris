"""
Only VERIFIED and SINGLE_SOURCE claims are handed to the model, the latter tagged
as uncorroborated; CONFLICTED claims are never presented as fact.
GENERAL mode produces a plain cited research report;
BACKGROUND_CHECK mode produces an adverse-media
report grouped subject -> risk category, with an identity/screening disclaimer.
Body text is rendered from data; the single LLM output is the validated summary slot.
"""

import asyncio
from collections import defaultdict

from veris.core.config import Settings
from veris.synthesis.slots import SlotLLM, slot_llm
from veris.synthesis.summary import SUMMARY_SLOT_ID, build_summary_slots
from veris.core.models import (
    SynthesisSlot,
    Claim,
    Evidence,
    ResearchMode,
    ResolvedSubject,
    VerificationResult,
    VerificationStatus,
)

_CATEGORY_HEADINGS: dict[str, str] = {
    "criminal": "Criminal",
    "regulatory_sanctions": "Regulatory & Sanctions",
    "litigation": "Legal & Litigation",
    "financial_crime_fraud": "Financial Crime & Fraud",
    "corruption_bribery": "Corruption & Bribery",
    "esg_labor_safety": "ESG, Labor & Safety",
    "reputational": "Reputational & Other",
}
_UNDATED_TAG = "(undated)"

_SINGLE_SOURCE_TAG = "(single source, uncorroborated)"

_SCREENING_DISCLAIMER = (
    "> **Screening aid only.** Findings below are drawn from public web sources and are "
    "not verified beyond automated subject-name and source-quality checks. Common names "
    "can produce false matches - confirm identity details (role, location, employer) "
    "before relying on any finding. This is not a legal, credit, or compliance determination."
)

def _resolution_summary(subjects: list[ResolvedSubject]) -> str:
    if len(subjects) == 1 and subjects[0].resolution_confidence >= 0.99:
        return (
            ""  # single directly-named subject - resolution transparency adds no value
        )
    lines = ["## Subjects Covered", ""]
    for s in subjects:
        confidence_note = (
            "given directly"
            if s.resolution_confidence >= 0.99
            else f"resolved via search, confidence {s.resolution_confidence:.1f}"
        )
        role_note = f" — {s.role}" if s.role else ""
        lines.append(f"- **{s.name}**{role_note} ({confidence_note})")
    lines.append("")
    return "\n".join(lines)


def _grouped_by_subject_category(
    claims: list[Claim],
    verifications: list[VerificationResult],
    evidence_list: list[Evidence],
) -> dict[str, list[tuple[Claim, Evidence]]]:
    """subject -> [(claim, representative evidence)], sorted by evidence
    relevance_score descending. Uncorroborated claims carry a visible tag."""
    evidence_by_id = {e.evidence_id: e for e in evidence_list}
    claim_by_id = {c.claim_id: c for c in claims}
    reportable = {
        v.claim_id: v.status
        for v in verifications
        if v.status != VerificationStatus.CONFLICTED
    }

    grouped: dict[str, list[tuple[Claim, Evidence]]] = defaultdict(list)
    for claim_id, status in reportable.items():
        claim = claim_by_id.get(claim_id)
        ev = next(
            (
                evidence_by_id[i]
                for i in (claim.evidence_ids if claim else [])
                if i in evidence_by_id and not evidence_by_id[i].flags
            ),
            None,
        )
        if not claim or not ev or claim.out_of_window or claim.flags:
            continue
        if claim.undated:
            claim = claim.model_copy(update={"text": f"{claim.text} {_UNDATED_TAG}"})
        if status == VerificationStatus.SINGLE_SOURCE:
            claim = claim.model_copy(
                update={"text": f"{claim.text} {_SINGLE_SOURCE_TAG}"}
            )
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


def _render_general_report(
    subjects: list[ResolvedSubject],
    grouped: dict[str, list[tuple[Claim, Evidence]]],
    summary_md: str,
) -> str:
    lines = ["# Research Report", ""]
    resolution = _resolution_summary(subjects)
    if resolution:
        lines.append(resolution)
    if summary_md:
        lines.append(summary_md)
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
            lines.append(
                f"- {claim.text} ({ev.source_metadata.source}, {ev.source_metadata.url})"
            )
        lines.append("")
    lines.append("## Sources")
    lines.append(_build_sources_block(all_evidence))
    return "\n".join(lines)


def _render_background_check_report(
    subjects: list[ResolvedSubject],
    grouped: dict[str, list[tuple[Claim, Evidence]]],
    summary_md: str,
) -> str:
    lines = [
        "# Background Check Report",
        "",
        _SCREENING_DISCLAIMER,
        "",
        _resolution_summary(subjects),
        summary_md,
    ]
    all_evidence: list[Evidence] = []
    for subject in subjects:
        items = grouped.get(subject.name, [])
        lines.append(f"## {subject.name}")
        if not items:
            lines.append(
                "_No adverse findings passed verification for this subject._\n"
            )
            continue
        by_category: dict[str, list[tuple[Claim, Evidence]]] = defaultdict(list)
        for claim, ev in items:
            for cat in claim.categories:
                by_category[cat].append((claim, ev))
        for category, heading in _CATEGORY_HEADINGS.items():
            cat_items = by_category.get(category, [])
            if not cat_items:
                continue
            lines.append(f"### {heading}")
            for claim, ev in cat_items:
                all_evidence.append(ev)
                lines.append(
                    f"- {claim.text} (reported by {ev.source_metadata.source}, {ev.source_metadata.url})"
                )
            lines.append("")
    lines.append("## Sources")
    lines.append(_build_sources_block(all_evidence))
    return "\n".join(lines)


def _render_summary(slots: list[SynthesisSlot]) -> str:
    if not slots:
        return ""
    lines = ["## Executive Summary", ""]
    for slot in slots:
        cite = f"(claims: {', '.join(slot.claim_ids)})"
        bullet = "" if slot.slot_id == SUMMARY_SLOT_ID else "- "
        lines.append(f"{bullet}{slot.text} {cite}")
    lines.append("")
    return "\n".join(lines)


async def run_synthesis(
    subjects: list[ResolvedSubject],
    claims: list[Claim],
    verifications: list[VerificationResult],
    evidence_list: list[Evidence],
    settings: Settings,
    research_mode: ResearchMode = ResearchMode.GENERAL,
    llm: SlotLLM | None = None,
) -> str:
    """Body is rendered from data; the only model output is the validated summary slot."""
    llm = llm or slot_llm(settings)

    def build() -> str:
        grouped = _grouped_by_subject_category(claims, verifications, evidence_list)
        summary_md = _render_summary(
            build_summary_slots(claims, verifications, evidence_list, settings, llm)
        )
        render = (
            _render_background_check_report
            if research_mode == ResearchMode.BACKGROUND_CHECK
            else _render_general_report
        )
        return render(subjects, grouped, summary_md)

    return await asyncio.to_thread(build)

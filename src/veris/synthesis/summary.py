"""Executive-summary slot: eligibility, fixed ordering, data block, validated synthesis."""

from collections import defaultdict

from veris.analysis.taxonomy import load_taxonomy
from veris.core.config import Settings
from veris.core.models import (
    Claim,
    Evidence,
    Severity,
    SynthesisSlot,
    VerificationResult,
)
from veris.synthesis.slots import SlotLLM, sanitize, synthesize_slots
from veris.synthesis.validator import ValidationContext, is_eligible

SUMMARY_SLOT_ID = "summary"
_SEVERITY_ORDER = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
_INSTRUCTION = (
    "Write the executive summary: 2-4 sentences covering the key findings below."
)


def _sort_key(claim: Claim, relevance: float) -> tuple:
    day = claim.event_date
    return (
        _SEVERITY_ORDER[claim.severity],
        -(day.toordinal()) if day else 1,
        -relevance,
        claim.claim_id,
    )


def build_summary_slots(
    claims: list[Claim],
    verifications: list[VerificationResult],
    evidence_list: list[Evidence],
    settings: Settings,
    llm: SlotLLM | None,
) -> list[SynthesisSlot]:
    ctx = ValidationContext(
        claims={c.claim_id: c for c in claims},
        evidence={e.evidence_id: e for e in evidence_list},
        verdicts={v.claim_id: v for v in verifications},
        taxonomy=load_taxonomy(settings.taxonomy_path),
        min_chars=settings.slot_min_chars,
        max_chars=settings.slot_max_chars,
    )
    relevance = {
        c.claim_id: max(
            (
                ctx.evidence[i].relevance_score
                for i in c.evidence_ids
                if i in ctx.evidence
            ),
            default=0.0,
        )
        for c in claims
    }
    by_subject: dict[str, list[Claim]] = defaultdict(list)
    for claim in sorted(
        (c for c in claims if is_eligible(c, ctx)),
        key=lambda c: _sort_key(c, relevance[c.claim_id]),
    ):
        by_subject[claim.subject].append(claim)
    capped = [
        c
        for items in by_subject.values()
        for c in items[: settings.max_synthesis_evidence_per_subject]
    ]
    ids = {c.claim_id for c in capped}
    selected = sorted(
        (c for c in capped if set(ctx.verdicts[c.claim_id].conflicts_with) <= ids),
        key=lambda c: _sort_key(c, relevance[c.claim_id]),
    )
    if not selected:
        return []
    data_block = "\n".join(
        f"[{c.claim_id}] subject={sanitize(c.subject, 80)} | categories={','.join(c.categories)} "
        f"| legal_status={c.legal_status.value} | severity={c.severity.value} "
        f"| verification={ctx.verdicts[c.claim_id].status.value} "
        f"| date={c.event_date or 'undated'} | text={sanitize(c.text)}"
        for c in selected
    )
    return synthesize_slots(
        expected={SUMMARY_SLOT_ID: selected},
        instruction=_INSTRUCTION,
        data_block=data_block,
        ctx=ctx,
        llm=llm,
        settings=settings,
    )

"""Executive-summary slot: eligibility, fixed ordering, data block, validated synthesis."""

import asyncio
from collections import defaultdict

from veris.analysis.taxonomy import load_taxonomy
from veris.core.config import Settings
from veris.core.models import Claim, Evidence, SynthesisSlot, VerificationResult
from veris.synthesis.ordering import claim_relevance, claim_sort_key
from veris.synthesis.slots import SlotLLM, sanitize, synthesize_slots
from veris.synthesis.validator import ValidationContext, is_eligible

SUMMARY_SLOT_ID = "summary"
_INSTRUCTION = "Write the executive summary: 2-4 sentences covering the key findings below."


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
    relevance = {c.claim_id: claim_relevance(c, ctx.evidence) for c in claims}
    by_subject: dict[str, list[Claim]] = defaultdict(list)
    for claim in sorted(
        (c for c in claims if is_eligible(c, ctx)), key=lambda c: claim_sort_key(c, relevance[c.claim_id])
    ):
        by_subject[claim.subject].append(claim)
    capped = [c for items in by_subject.values() for c in items[: settings.max_synthesis_evidence_per_subject]]
    ids = {c.claim_id for c in capped}
    selected = sorted(
        (c for c in capped if set(ctx.verdicts[c.claim_id].conflicts_with) <= ids),
        key=lambda c: claim_sort_key(c, relevance[c.claim_id]),
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


async def generate_summary(
    claims: list[Claim],
    verifications: list[VerificationResult],
    evidence_list: list[Evidence],
    settings: Settings,
    llm: SlotLLM | None,
) -> SynthesisSlot | None:
    """The run's single model-written slot; None when no claim is eligible."""
    slots = await asyncio.to_thread(build_summary_slots, claims, verifications, evidence_list, settings, llm)
    return slots[0] if slots else None

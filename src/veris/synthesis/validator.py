"""Deterministic slot validation (no model call)."""

import re
from dataclasses import dataclass

from veris.analysis.classification import resolve_status
from veris.analysis.taxonomy import Taxonomy
from veris.core.models import (
    Claim,
    Evidence,
    LegalStatus,
    SynthesisSlot,
    VerificationResult,
    VerificationStatus,
)

_SCALE = {
    "thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12,
}
_NUM_RE = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:\s*(thousand|million|billion|trillion|bn|k|m|b)\b)?(?![A-Za-z])",
    re.I,
)


@dataclass(frozen=True)
class ValidationContext:
    claims: dict[str, Claim]
    evidence: dict[str, Evidence]
    verdicts: dict[str, VerificationResult]
    taxonomy: Taxonomy
    min_chars: int
    max_chars: int


def numbers(text: str) -> set[float]:
    out: set[float] = set()
    for raw, scale in _NUM_RE.findall(text):
        try:
            out.add(float(raw.replace(",", "")) * _SCALE.get(scale.lower(), 1.0))
        except ValueError:
            continue
    return out


_MON_FULL = "January|February|March|April|June|July|August|September|October|November|December"
_MON_ABBR = "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec"
_MONTH_RE = re.compile(
    rf"\b(?:{_MON_FULL}|(?:{_MON_ABBR})\.?(?=\s+\d)|May(?=\s+\d)|(?<=\d\s)May\b)"
)


def months(text: str) -> set[str]:
    return {m.group()[:3].lower() for m in _MONTH_RE.finditer(text)}


def is_eligible(claim: Claim, ctx: ValidationContext) -> bool:
    verdict = ctx.verdicts.get(claim.claim_id)
    if claim.flags or claim.out_of_window or verdict is None:
        return False
    if verdict.status == VerificationStatus.CONFLICTED:
        return bool(verdict.conflicts_with)
    return True


def cited_passages(claim: Claim, ctx: ValidationContext) -> list[str]:
    texts = [claim.text]
    for eid in claim.evidence_ids:
        ev = ctx.evidence.get(eid)
        if ev and not ev.flags:
            texts.append(ev.supporting_passage)
    return texts


def validate_slot(
    slot: SynthesisSlot, ctx: ValidationContext, *, check_min: bool = True
) -> list[str]:
    errors: list[str] = []
    text = slot.text.strip()
    if not text:
        return ["empty text"]
    if len(text) > ctx.max_chars or (check_min and len(text) < ctx.min_chars):
        errors.append(f"length {len(text)} outside {ctx.min_chars}-{ctx.max_chars}")
    if not slot.claim_ids:
        return [*errors, "no claim_ids"]

    cited: list[Claim] = []
    for cid in slot.claim_ids:
        claim = ctx.claims.get(cid)
        if claim is None:
            errors.append(f"unknown claim_id {cid}")
        elif not is_eligible(claim, ctx):
            errors.append(f"claim {cid} is flagged/excluded")
        else:
            cited.append(claim)
    for claim in cited:
        verdict = ctx.verdicts[claim.claim_id]
        if verdict.status == VerificationStatus.CONFLICTED and not set(
            verdict.conflicts_with
        ) <= set(slot.claim_ids):
            errors.append(f"conflicted claim {claim.claim_id} cited without both sides")
    if errors:
        return errors

    passages = [p for c in cited for p in cited_passages(c, ctx)]
    if extra := sorted(numbers(text) - set().union(*map(numbers, passages))):
        errors.append(f"values not in cited passages: {extra}")
    if extra := sorted(months(text) - set().union(*map(months, passages))):
        errors.append(f"months not in cited passages: {extra}")
    stated = resolve_status(text, ctx.taxonomy, adverse=True)
    if stated != LegalStatus.ALLEGED and stated not in {c.legal_status for c in cited}:
        errors.append(f"legal status '{stated.value}' not supported by cited claims")
    return errors

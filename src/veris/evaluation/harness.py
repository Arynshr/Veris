"""Runs the deterministic analysis stages (evidence mapping -> verification) over
golden cases and scores findings, categories, and verification states."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel

from veris.analysis import run_evidence_mapping, run_verification
from veris.core.config import Settings
from veris.core.documents import build_document
from veris.core.models import (
    AdverseCategory,
    Evidence,
    ResearchMode,
    VerificationStatus,
)
from veris.evaluation.golden import ExpectedFinding, GoldenCase

SCHEMA_VERSION = 1


class PRF(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 1.0
    recall: float = 1.0
    f1: float = 1.0


class CaseResult(BaseModel):
    case_id: str
    findings: PRF
    state_accuracy: float


class EvalReport(BaseModel):
    schema_version: int = SCHEMA_VERSION
    cases: list[CaseResult]
    findings: PRF
    categories: dict[str, PRF]
    verification_states: dict[str, PRF]
    state_accuracy: float

    def headline(self) -> dict[str, float]:
        """Metrics gated by compare_reports. F1 (not raw precision/recall) so a
        deliberate precision/recall trade-off is judged on balance; precision and
        recall stay in the report for inspection."""
        return {
            "findings.f1": self.findings.f1,
            "category.f1": _micro(self.categories.values()).f1,
            "state_accuracy": self.state_accuracy,
        }


def _prf(tp: int, fp: int, fn: int) -> PRF:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )


def _micro(scores) -> PRF:
    scores = list(scores)
    return _prf(
        sum(s.tp for s in scores), sum(s.fp for s in scores), sum(s.fn for s in scores)
    )


def _matches(evidence: Evidence, expected: ExpectedFinding) -> bool:
    return (
        evidence.source_metadata.url == expected.url
        and expected.match_text.lower() in evidence.claim.lower()
    )


def _is_finding(evidence: Evidence, research_mode: ResearchMode) -> bool:
    return (
        research_mode != ResearchMode.BACKGROUND_CHECK
        or evidence.category != AdverseCategory.GENERAL
    )


def _evidence_states(claims, verifications) -> dict[str, VerificationStatus]:
    status_by_claim = {v.claim_id: v.status for v in verifications}
    return {
        eid: status_by_claim[c.claim_id]
        for c in claims
        for eid in c.evidence_ids
        if c.claim_id in status_by_claim
    }


@dataclass
class _CaseScores:
    result: CaseResult
    cat_tp: Counter = field(default_factory=Counter)
    cat_fp: Counter = field(default_factory=Counter)
    cat_fn: Counter = field(default_factory=Counter)
    state_pairs: list[tuple[str, str | None]] = field(default_factory=list)


def _score_case(case: GoldenCase, settings: Settings) -> _CaseScores:
    documents = [
        build_document(d.url, d.title, d.content, d.published_at)
        for d in case.documents
    ]
    entities = case.build_entities()
    evidence = run_evidence_mapping(
        documents,
        case.subject,
        entities,
        settings,
        research_mode=case.research_mode,
    )
    claims, verifications = run_verification(
        evidence,
        entities,
        case.subject,
        settings,
        research_mode=case.research_mode,
    )
    state_of = _evidence_states(claims, verifications)

    reportable_expected = [
        f
        for f in case.expected_findings
        if f.verification != VerificationStatus.CONFLICTED
    ]
    predicted = [
        e
        for e in evidence
        if _is_finding(e, case.research_mode)
        and state_of.get(e.evidence_id) not in (None, VerificationStatus.CONFLICTED)
    ]

    matched_expected = [
        f for f in reportable_expected if any(_matches(e, f) for e in predicted)
    ]
    matched_predicted = [
        e for e in predicted if any(_matches(e, f) for f in reportable_expected)
    ]
    findings = _prf(
        tp=len(matched_expected),
        fp=len(predicted) - len(matched_predicted),
        fn=len(reportable_expected) - len(matched_expected),
    )

    scores = _CaseScores(
        result=CaseResult(case_id=case.case_id, findings=findings, state_accuracy=1.0)
    )
    for f in reportable_expected:
        hit = any(_matches(e, f) and e.category == f.category for e in predicted)
        (scores.cat_tp if hit else scores.cat_fn)[f.category.value] += 1
    for e in predicted:
        if not any(
            _matches(e, f) and e.category == f.category for f in reportable_expected
        ):
            scores.cat_fp[e.category.value] += 1

    state_pairs = scores.state_pairs
    for f in case.expected_findings:
        hit = next((e for e in evidence if _matches(e, f)), None)
        predicted_state = state_of.get(hit.evidence_id) if hit else None
        state_pairs.append(
            (f.verification.value, predicted_state.value if predicted_state else None)
        )
    scores.result.state_accuracy = _accuracy(state_pairs)
    return scores


def _accuracy(pairs: list[tuple[str, str | None]]) -> float:
    if not pairs:
        return 1.0
    return round(sum(1 for exp, got in pairs if exp == got) / len(pairs), 4)


def run_eval(cases: list[GoldenCase], settings: Settings) -> EvalReport:
    results: list[CaseResult] = []
    cat_tp: Counter = Counter()
    cat_fp: Counter = Counter()
    cat_fn: Counter = Counter()
    state_pairs: list[tuple[str, str | None]] = []
    for case in cases:
        scores = _score_case(case, settings)
        results.append(scores.result)
        cat_tp.update(scores.cat_tp)
        cat_fp.update(scores.cat_fp)
        cat_fn.update(scores.cat_fn)
        state_pairs.extend(scores.state_pairs)

    categories = {
        c: _prf(cat_tp[c], cat_fp[c], cat_fn[c])
        for c in sorted(set(cat_tp) | set(cat_fp) | set(cat_fn))
    }
    per_state: dict[str, Counter] = defaultdict(Counter)
    for expected, got in state_pairs:
        if expected == got:
            per_state[expected]["tp"] += 1
        else:
            per_state[expected]["fn"] += 1
            if got is not None:
                per_state[got]["fp"] += 1
    states = {s: _prf(c["tp"], c["fp"], c["fn"]) for s, c in sorted(per_state.items())}
    return EvalReport(
        cases=results,
        findings=_micro(r.findings for r in results),
        categories=categories,
        verification_states=states,
        state_accuracy=_accuracy(state_pairs),
    )


def compare_reports(
    current: EvalReport, baseline: EvalReport, tolerance: float = 0.0
) -> list[str]:
    base = baseline.headline()
    return [
        f"{name}: {base[name]:.4f} -> {value:.4f}"
        for name, value in current.headline().items()
        if value < base[name] - tolerance
    ]

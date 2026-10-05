"""Offline evaluation of the analysis stages against a golden set of cached fixtures."""

from veris.evaluation.golden import GoldenCase, load_golden_cases
from veris.evaluation.harness import EvalReport, compare_reports, run_eval

__all__ = [
    "EvalReport",
    "GoldenCase",
    "compare_reports",
    "load_golden_cases",
    "run_eval",
]

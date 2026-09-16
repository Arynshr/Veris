"""Turns verified claims into a cited Markdown report, mode-aware (general
research narrative vs. adverse-media background-check report)."""
from veris.synthesis.report import run_synthesis

__all__ = ["run_synthesis"]

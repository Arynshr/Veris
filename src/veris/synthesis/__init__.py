"""Deterministic report: canonical Report model, Jinja2 rendering, and the single summary slot."""

from veris.synthesis.report import build_report, render_markdown
from veris.synthesis.summary import generate_summary

__all__ = ["build_report", "generate_summary", "render_markdown"]

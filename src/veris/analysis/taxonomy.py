"""Taxonomy config: category patterns, status cues and severity matrix (YAML-driven)."""

import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, PrivateAttr, ValidationError, model_validator

from veris.core.models import LegalStatus, Severity

SUPPORTED_SCHEMA_VERSION = 1
UNCATEGORIZED = "uncategorized"
_SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2}


def _compile(patterns: list[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)


class Taxonomy(BaseModel):
    schema_version: int
    categories: dict[str, list[str]]
    query_terms: list[str]
    status_precedence: list[LegalStatus]
    status_cues: dict[LegalStatus, list[str]]
    negators: list[str]
    negation_window_tokens: int = 3
    status_window_chars: int = 160
    severity: dict[LegalStatus, dict[str, Severity]]

    _category_res: dict[str, re.Pattern[str]] = PrivateAttr(default_factory=dict)
    _cue_res: dict[LegalStatus, re.Pattern[str]] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> "Taxonomy":
        if self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(f"Unsupported taxonomy schema {self.schema_version}")
        if UNCATEGORIZED in self.categories:
            raise ValueError(f"'{UNCATEGORIZED}' is implicit and takes no patterns")
        for status in LegalStatus:
            if "default" not in self.severity.get(status, {}):
                raise ValueError(f"severity.{status.value} needs a default")
        for status, row in self.severity.items():
            if unknown := set(row) - {"default", *self.categories}:
                raise ValueError(f"severity.{status.value}: unknown {sorted(unknown)}")
        if set(self.status_precedence) != set(self.status_cues):
            raise ValueError("status_precedence and status_cues must list the same statuses")
        self._category_res = {c: _compile(p) for c, p in self.categories.items()}
        self._cue_res = {s: _compile(self.status_cues[s]) for s in self.status_precedence}
        return self

    @property
    def category_order(self) -> list[str]:
        return list(self.categories)

    def match_categories(self, text: str) -> list[str]:
        return [c for c, rx in self._category_res.items() if rx.search(text)]

    def category_spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for rx in self._category_res.values() for m in rx.finditer(text)]

    def cue_matches(self, status: LegalStatus, text: str) -> list[re.Match[str]]:
        return list(self._cue_res[status].finditer(text))

    def strip_cues(self, text: str) -> str:
        """Remove status-cue phrases so claims differing only in legal status cluster together."""
        for rx in self._cue_res.values():
            text = rx.sub(" ", text)
        return " ".join(text.split())

    def severity_for(self, status: LegalStatus, categories: list[str]) -> Severity:
        row = self.severity[status]
        levels = [row.get(c, row["default"]) for c in categories if c != UNCATEGORIZED]
        return max(levels, key=_SEVERITY_RANK.__getitem__, default=Severity.LOW)

    def status_rank(self, status: LegalStatus) -> int:
        order = self.status_precedence
        return len(order) - order.index(status) if status in order else 0


@lru_cache
def load_taxonomy(path: Path) -> Taxonomy:
    try:
        return Taxonomy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as exc:
        raise ValueError(f"Invalid taxonomy file {path}:\n{exc}") from exc

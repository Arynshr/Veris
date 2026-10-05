"""Source classification: domain -> (tier, type, quality score), driven by YAML config."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

_SUPPORTED_SCHEMA_VERSION = 1
_SECOND_LEVEL_LABELS = {"co", "com", "org", "net", "gov", "edu", "ac"}


class SourceType(StrEnum):
    PRIMARY = "primary"
    WIRE = "wire"
    NEWS = "news"
    BLOG = "blog"
    FORUM = "forum"
    PRESS_RELEASE = "press_release"


class SourceClassification(BaseModel):
    tier: int
    source_type: SourceType
    quality_score: float


class _Rule(BaseModel):
    match: list[str]
    tier: int
    type: SourceType


class _Default(BaseModel):
    tier: int
    type: SourceType


class SourcePolicy(BaseModel):
    schema_version: int
    tiers: dict[int, float]
    default: _Default
    rules: list[_Rule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "SourcePolicy":
        if self.schema_version != _SUPPORTED_SCHEMA_VERSION:
            raise ValueError(f"Unsupported source tier schema {self.schema_version}")
        used = {self.default.tier, *(r.tier for r in self.rules)}
        if missing := used - self.tiers.keys():
            raise ValueError(f"Tiers without a quality score: {sorted(missing)}")
        return self

    def classify(self, domain: str) -> SourceClassification:
        host = domain.lower().removeprefix("www.")
        for rule in self.rules:
            if any(_matches(host, m) for m in rule.match):
                return self._classification(rule.tier, rule.type)
        return self._classification(self.default.tier, self.default.type)

    def _classification(
        self, tier: int, source_type: SourceType
    ) -> SourceClassification:
        return SourceClassification(
            tier=tier, source_type=source_type, quality_score=self.tiers[tier]
        )


def _matches(host: str, pattern: str) -> bool:
    if pattern.startswith("."):
        return host.endswith(pattern)
    return host == pattern or host.endswith(f".{pattern}")


@lru_cache
def load_source_policy(path: Path) -> SourcePolicy:
    return SourcePolicy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def registrable_domain(host: str) -> str:
    """Approximate eTLD+1 without a public-suffix list: keeps three labels for
    country-code second-level suffixes such as bbc.co.uk."""
    labels = host.lower().removeprefix("www.").split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL_LABELS:
        return ".".join(labels[-3:1])
    return ".".join(labels[-2:])

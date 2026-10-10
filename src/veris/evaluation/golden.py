"""Golden-set contracts and loader. Fixtures carry their own documents and pinned
entities, so evaluation needs no network, provider, or spaCy model."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from veris.core.models import (
    Entity,
    LegalStatus,
    ResearchMode,
    Severity,
    VerificationStatus,
)

_LEGACY_CATEGORIES = {
    "legal_litigation": "litigation",
    "financial": "financial_crime_fraud",
    "general": "uncategorized",
}


class GoldenEntity(BaseModel):
    text: str
    label: str


class GoldenDocument(BaseModel):
    url: str
    title: str = ""
    content: str
    published_at: datetime | None = None


class ExpectedFinding(BaseModel):
    url: str
    match_text: str = Field(
        description="Case-insensitive substring identifying the finding's claim sentence."
    )
    categories: list[str]
    verification: VerificationStatus
    legal_status: LegalStatus | None = None
    severity: Severity | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        """Accept legacy single `category` and legacy category names."""
        if isinstance(data, dict):
            data = dict(data)
            if "category" in data:
                data.setdefault("categories", [data.pop("category")])
            data["categories"] = [
                _LEGACY_CATEGORIES.get(c, c) for c in data.get("categories", [])
            ]
        return data


class GoldenCase(BaseModel):
    schema_version: int = 1
    case_id: str
    subject: str
    query: str
    research_mode: ResearchMode = ResearchMode.BACKGROUND_CHECK
    entities: list[GoldenEntity] = Field(default_factory=list)
    documents: list[GoldenDocument]
    expected_findings: list[ExpectedFinding] = Field(default_factory=list)

    def build_entities(self) -> list[Entity]:
        entities = []
        for e in self.entities:
            start = max(self.query.find(e.text), 0)
            entities.append(
                Entity(
                    text=e.text,
                    label=e.label,
                    start_char=start,
                    end_char=start + len(e.text),
                )
            )
        return entities


def load_golden_cases(golden_dir: Path) -> list[GoldenCase]:
    paths = sorted(golden_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No golden cases (*.json) found in {golden_dir}")
    return [
        GoldenCase.model_validate_json(p.read_text(encoding="utf-8")) for p in paths
    ]

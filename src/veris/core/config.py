"""Configuration and logging setup. Secrets always come from environment variables."""

import hashlib
import json
import logging
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_SEARCH_RESULTS_CEILING = 10
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Search providers
    tavily_api_key: str | None = None
    firecrawl_api_key: str | None = None
    searxng_base_url: str | None = None

    llm_provider: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_timeout_seconds: int = Field(default=60, ge=1)
    llm_max_tokens: int = Field(default=2048, ge=1)
    llm_providers_path: Path = PROJECT_ROOT / "config" / "llm_providers.yaml"

    max_search_results_per_query: int = Field(
        default=2, ge=1, le=MAX_SEARCH_RESULTS_CEILING
    )
    max_queries_per_plan: int = Field(default=8, ge=1, le=25)
    max_documents_per_subject: int = Field(default=10, ge=1, le=50)
    max_synthesis_evidence_per_subject: int = Field(default=15, ge=1, le=100)
    request_timeout_seconds: int = 20
    max_retries: int = 3

    system_prompt: str = (
        "You are Veris, a rigorous research assistant. Investigate the given topic "
        "or subject using multiple sources, only make claims supported by retrieved "
        "evidence, and always cite sources."
    )

    lookback_years: int = Field(default=2, ge=1)
    as_of_date: date | None = Field(
        default=None, description="Reference date for the lookback window (default: today)."
    )
    min_subject_name_overlap: float = 0.6
    max_resolved_subjects: int = 8
    max_group_role_size: int = Field(default=8, ge=1)

    source_tiers_path: Path = PROJECT_ROOT / "config" / "source_tiers.yaml"
    taxonomy_path: Path = PROJECT_ROOT / "config" / "taxonomy.yaml"
    safety_path: Path = PROJECT_ROOT / "config" / "safety.yaml"
    templates_dir: Path = PROJECT_ROOT / "templates"
    template_version: str = "v1"
    paywall_status_codes: list[int] = [401, 402, 403]
    slot_min_chars: int = Field(default=40, ge=1)
    slot_max_chars: int = Field(default=900, ge=1)
    summary_fallback_bullets: int = Field(default=3, ge=1)
    evidence_window_sentences: int = Field(default=1, ge=0, le=5)
    evidence_mention_max_distance: int = Field(default=3, ge=0, le=20)
    identity_anchor_labels: list[str] = ["ORG", "GPE", "FAC", "LOC", "NORP"]
    claim_dedup_similarity: float = Field(default=0.8, ge=0.0, le=1.0)
    min_independent_domains: int = Field(default=2, ge=1)
    min_corroboration_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    verified_alone_max_tier: int = Field(default=1, ge=1)
    min_entity_match_score: float = Field(default=0.25, ge=0.0, le=1.0)

    runs_dir: Path = Path("runs")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def config_hash(settings: Settings) -> str:
    """Stable hash of non-secret, non-path settings plus policy YAMLs and templates."""
    skip = {k for k in type(settings).model_fields if k.endswith(("_api_key", "_path", "_dir"))}
    payload = settings.model_dump(mode="json", exclude=skip)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    for path in (settings.source_tiers_path, settings.taxonomy_path, settings.safety_path):
        digest.update(Path(path).read_bytes())
    root = settings.templates_dir / settings.template_version
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]

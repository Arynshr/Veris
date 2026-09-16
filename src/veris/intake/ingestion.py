"""Ingestion: deterministic spaCy NER over the raw query"""
from functools import lru_cache

import spacy

from veris.core.models import Entity, IngestionResult

_SPACY_MODEL = "en_core_web_sm"


@lru_cache
def _load_spacy_model():
    try:
        return spacy.load(_SPACY_MODEL)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{_SPACY_MODEL}' not installed. Run: "
            f"uv run python -m spacy download {_SPACY_MODEL}"
        ) from exc


def extract_entities(text: str) -> list[Entity]:
    doc = _load_spacy_model()(text)
    return [
        Entity(text=ent.text, label=ent.label_, start_char=ent.start_char, end_char=ent.end_char)
        for ent in doc.ents
    ]


def run_ingestion(user_prompt: str, system_prompt: str) -> IngestionResult:
    """Ingestion entrypoint. Produces normalized entities for query understanding."""
    return IngestionResult(
        original_prompt=user_prompt,
        entities=extract_entities(user_prompt),
        system_instructions=system_prompt,
    )

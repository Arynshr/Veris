"""Turns a raw query into concrete research subjects: NER -> understanding
(mode + subject-structure classification) -> resolution (role/group -> names)."""
from veris.intake.ingestion import extract_entities, run_ingestion
from veris.intake.resolution import resolve_subjects
from veris.intake.understanding import understand_query

__all__ = ["run_ingestion", "extract_entities", "understand_query", "resolve_subjects"]

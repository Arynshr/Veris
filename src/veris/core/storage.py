"""
Storage layer persists every stage's output to
runs/{run_id}/... and deduplicates documents by canonical URL / content hash.
"""
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from veris.core.config import Settings
from veris.core.models import Document


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "subject"


class RunPersistence:
    def __init__(self, settings: Settings, run_id: str | None = None) -> None:
        self.settings = settings
        self.run_id = run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        self.run_dir: Path = settings.runs_dir / self.run_id
        self.documents_dir: Path = self.run_dir / "documents"
        self.subjects_dir: Path = self.run_dir / "subjects"
        self._registered_urls: set[str] = set()
        self._registered_hashes: set[str] = set()

    def init_dirs(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.subjects_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, payload: Any) -> Path:
        if isinstance(payload, BaseModel):
            data = payload.model_dump(mode="json")
        elif isinstance(payload, list):
            data = [p.model_dump(mode="json") if isinstance(p, BaseModel) else p for p in payload]
        else:
            data = payload
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def _read_json(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found for run {self.run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _subject_dir(self, subject: str) -> Path:
        return self.subjects_dir / _slug(subject)

    def save_input(self, user_prompt: str, system_prompt: str) -> None:
        self._write_json(self.run_dir / "input.json", {"user_prompt": user_prompt, "system_prompt": system_prompt})

    def save_entities(self, entities: list) -> None:
        self._write_json(self.run_dir / "entities.json", entities)

    def save_understanding(self, understanding) -> None:
        self._write_json(self.run_dir / "understanding.json", understanding)

    def save_resolution(self, resolved_subjects: list) -> None:
        self._write_json(self.run_dir / "resolution.json", resolved_subjects)

    def save_run_record(self, run_record) -> None:
        self._write_json(self.run_dir / "run.json", run_record)

    def append_event(self, event) -> None:
        """Durable checkpoint: every PipelineEvent is appended to events.jsonl as
        it happens, so a run's progress survives the CLI process (or an SSH
        session) dying, and can be inspected or replayed after the fact via
        `veris events <run_id>` instead of only existing in a live terminal."""
        path = self.run_dir / "events.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.model_dump(mode="json"), default=str) + "\n")

    def load_events(self) -> list[dict]:
        path = self.run_dir / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def save_report(self, markdown: str) -> Path:
        path = self.run_dir / "report.md"
        path.write_text(markdown, encoding="utf-8")
        return path

    def save_plan(self, plan, subject: str) -> None:
        self._write_json(self._subject_dir(subject) / "plan.json", plan)

    def save_search_results(self, results: list, subject: str) -> None:
        self._write_json(self._subject_dir(subject) / "search_results.json", results)

    def save_evidence(self, evidence: list, subject: str) -> None:
        self._write_json(self._subject_dir(subject) / "evidence.json", evidence)

    def save_verification(self, results: list, subject: str) -> None:
        self._write_json(self._subject_dir(subject) / "verification.json", results)

    def load_documents(self) -> list[dict]:
        if not self.documents_dir.exists():
            return []
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.documents_dir.glob("*.json"))]

    def load_verification(self, subject: str | None = None) -> list[dict]:
        if subject:
            return self._read_json(self._subject_dir(subject) / "verification.json")
        merged: list[dict] = []
        for subject_dir in sorted(self.subjects_dir.glob("*")) if self.subjects_dir.exists() else []:
            path = subject_dir / "verification.json"
            if path.exists():
                merged.extend(json.loads(path.read_text(encoding="utf-8")))
        return merged

    def load_search_results(self, subject: str) -> list[dict]:
        return self._read_json(self._subject_dir(subject) / "search_results.json")

    def register_document(self, document: Document) -> Document | None:
        if document.canonical_url in self._registered_urls or document.content_hash in self._registered_hashes:
            return None
        self._registered_urls.add(document.canonical_url)
        self._registered_hashes.add(document.content_hash)
        path = self.documents_dir / f"{document.document_id}.json"
        path.write_text(json.dumps(document.model_dump(mode="json"), indent=2, default=str), encoding="utf-8")
        return document

    def register_documents(self, documents: list[Document]) -> list[Document]:
        return [d for d in (self.register_document(doc) for doc in documents) if d is not None]

    @property
    def total_documents_registered(self) -> int:
        return len(self._registered_hashes)

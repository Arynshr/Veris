"""Narrow provider interfaces. Business logic depends on these, never on SDKs or credentials."""

from enum import Enum
from typing import Protocol

from pydantic import BaseModel

from veris.core.models import Document


class LLM(Protocol):
    def complete(
        self, system: str, user: str, *, schema: dict | None = None, json_mode: bool = False
    ) -> str: ...

    def complete_json(self, system: str, user: str, *, schema: dict | None = None) -> dict: ...


class FetchOutcome(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    ERROR = "error"
    FORBIDDEN = "forbidden"
    PAYWALLED = "paywalled"
    RATE_LIMITED = "rate_limited"


class FetchResult(BaseModel):
    outcome: FetchOutcome
    document: Document | None = None
    status_code: int | None = None
    detail: str = ""


class Extractor(Protocol):
    async def fetch(self, url: str, title_hint: str = "") -> FetchResult: ...

    async def close(self) -> None: ...

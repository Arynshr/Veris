"""Evidence-time passage cleaning: strip crawler markup/boilerplate and flag noise (YAML-driven, pure)."""

import html
import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, PrivateAttr, model_validator

NOISE = "low_quality"
IDENTITY_MISMATCH = "identity_mismatch"
SUPPORTED_SCHEMA_VERSION = 1

_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s)>\]]+", re.I)
_TAG_RE = re.compile(r"<[^>]{1,200}>")
_MARKER_RE = re.compile(r"[*`]+|^\s*(?:#{1,6}\s|>\s?|[-+]\s)", re.M)
_WORD_RE = re.compile(r"\w+")


class CleaningConfig(BaseModel):
    schema_version: int
    boilerplate_patterns: list[str] = Field(default_factory=list)
    boilerplate_max_chars: int = Field(default=200, ge=1)
    min_snippet_chars: int = Field(default=40, ge=1)
    snippet_max_chars: int = Field(default=280, ge=50)
    max_link_text_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    repeat_min_tokens: int = Field(default=8, ge=1)
    min_unique_token_ratio: float = Field(default=0.5, ge=0.0, le=1.0)

    _boilerplate: re.Pattern[str] = PrivateAttr()

    @model_validator(mode="after")
    def _compile(self) -> "CleaningConfig":
        if self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(f"Unsupported cleaning schema {self.schema_version}")
        joined = "|".join(f"(?:{p})" for p in self.boilerplate_patterns) or r"(?!)"
        self._boilerplate = re.compile(joined, re.I)
        return self

    def is_boilerplate(self, line: str) -> bool:
        return len(line) <= self.boilerplate_max_chars and bool(self._boilerplate.search(line))


class CleanedClaim(BaseModel):
    snippet: str
    noisy: bool


def _strip_markup(text: str) -> str:
    text = _IMAGE_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", _TAG_RE.sub(" ", html.unescape(text)))
    return _MARKER_RE.sub("", text)


def clean_text(text: str, cfg: CleaningConfig) -> str:
    """Markup-free text with boilerplate lines removed and whitespace collapsed."""
    lines = (" ".join(_strip_markup(line).split()) for line in text.splitlines())
    return " ".join(line for line in lines if line and not cfg.is_boilerplate(line))


def truncate(text: str, limit: int) -> str:
    """Cut at a sentence end past the midpoint when possible, else at a word boundary."""
    if len(text) <= limit:
        return text
    head = text[: limit - 1]
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", head)]
    if ends and ends[-1] >= limit // 2:
        return head[: ends[-1]]
    return head.rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def _link_text_ratio(raw: str, visible_chars: int) -> float:
    anchors = sum(len(m.group(1).strip()) for m in _LINK_RE.finditer(_IMAGE_RE.sub(" ", raw)))
    return anchors / max(visible_chars, 1)


def clean_claim(raw: str, cfg: CleaningConfig) -> CleanedClaim:
    """Cleaned, bounded snippet for a claim sentence plus a noise verdict."""
    snippet = clean_text(raw, cfg)
    tokens = [t.lower() for t in _WORD_RE.findall(snippet)]
    repetitive = (
        len(tokens) >= cfg.repeat_min_tokens
        and len(set(tokens)) / len(tokens) < cfg.min_unique_token_ratio
    )
    noisy = (
        len(snippet) < cfg.min_snippet_chars
        or repetitive
        or _link_text_ratio(raw, len(snippet)) > cfg.max_link_text_ratio
    )
    return CleanedClaim(snippet=truncate(snippet, cfg.snippet_max_chars), noisy=noisy)


@lru_cache
def load_cleaning(path: Path) -> CleaningConfig:
    return CleaningConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

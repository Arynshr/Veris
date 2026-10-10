"""instruction-like passage detection (prompt-injection defence at evidence time)"""

import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, PrivateAttr, model_validator

INSTRUCTION_LIKE = "instruction_like"
SUPPORTED_SCHEMA_VERSION = 1

class SafetyConfig(BaseModel):
    schema_version: int
    injection_patterns: list[str]
    
    _rx: re.Pattern[str] = PrivateAttr()
    
    @model_validator(mode="after")
    def _compile(self) -> "SafetyConfig":
        if self.schema_version != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(f"Unsupported safety schema {self.schema_version}")
        joined = "|".join(f"(?:{p})" for p in self.injection_patterns)
        self._rx = re.compile(rf"\b(?:{joined})\b", re.I)
        return self
    
    def matches(self, text: str) -> bool:
        return bool(self._rx.search(text))
    
@lru_cache
def load_safety(path: Path) -> SafetyConfig:
    return SafetyConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    
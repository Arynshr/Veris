"""Configuration and logging setup. Secrets always come from environment variables."""
import logging
import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file = ".env", env_file_encoding = "utf-8", extra = "ignore")
    
    # Search providers
    tavily_api_key: str | None = None
    firecrawl_api_key: str | None = None
    searxng_base_url: str = "http://localhost:8080"
    
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"
    
    max_search_results_per_query: int = 2
    max_queries_per_plan: int = 1
    request_timeout_seconds: int = 20
    max_retries: int = 3
    
    system_prompt: str = (
        "You are Veris, a rigorous research assistant. Investigate the given topic "
        "or subject using multiple sources, only make claims supported by retrieved "
        "evidence, and always cite sources."
    )
    
    adverse_lookback_days: int = 730
    min_subject_name_overlap: float = 0.6
    max_resolved_subjects: int = 8
    
    runs_dir: Path = Path("runs")
    
@lru_cache
def get_settings() -> Settings:
    return Settings()

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level = level,
        stream = sys.stderr,
        format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%H:%M:%S" ,
    )
    
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

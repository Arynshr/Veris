"""Turns a research plan into raw content: query planning -> web search ->
content extraction."""

from veris.retrieval.crawl_policy import CrawlGuard
from veris.retrieval.extraction import build_extractor, retrieve_documents
from veris.retrieval.planning import run_planning
from veris.retrieval.search import build_provider_chain, search_with_fallback

__all__ = [
    "CrawlGuard",
    "build_extractor",
    "build_provider_chain",
    "retrieve_documents",
    "run_planning",
    "search_with_fallback",
]

"""Turns a research plan into raw content: query planning -> web search ->
content extraction."""
from veris.retrieval.extraction import retrieve_documents
from veris.retrieval.planning import run_planning
from veris.retrieval.search import build_provider_chain, search_with_fallback

__all__ = ["run_planning", "build_provider_chain", "search_with_fallback", "retrieve_documents"]

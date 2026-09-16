"""
Evidence mapping (spec section 9), mode-aware and using TF-IDF cosine similarity
instead of raw keyword substring counting: Document -> relevant passages ->
Evidence -> candidate claim.
"""
import re
import uuid

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from veris.core.config import get_logger
from veris.core.models import AdverseCategory, Document, Entity, Evidence, ResearchMode, SourceMetadata

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_SENTENCE_LEN = 40
_MAX_EVIDENCE_PER_DOCUMENT = 5
_MIN_RELEVANCE_SCORE_GENERAL = 0.05 

_CATEGORY_KEYWORDS: dict[AdverseCategory, list[str]] = {
    AdverseCategory.LEGAL: ["lawsuit", "sued", "litigation", "court", "plaintiff", "defendant"],
    AdverseCategory.REGULATORY: ["sanction", "regulator", "sec ", "ftc", "compliance violation", "fine", "penalty"],
    AdverseCategory.CRIMINAL: ["fraud", "indicted", "charged", "arrest", "criminal", "convicted", "prosecut"],
    AdverseCategory.FINANCIAL: ["bankruptcy", "insolvency", "embezzle", "money laundering", "financial misconduct"],
    AdverseCategory.REPUTATIONAL: ["scandal", "controversy", "allegation", "accused", "backlash", "criticized"],
}
_ALL_ADVERSE_KEYWORDS = [kw for kws in _CATEGORY_KEYWORDS.values() for kw in kws]


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if len(s.strip()) >= _MIN_SENTENCE_LEN]


def _categorize(sentence: str) -> AdverseCategory:
    lowered = sentence.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return category
    return AdverseCategory.GENERAL


def _is_adverse_candidate(sentence: str, subject: str) -> bool:
    lowered = sentence.lower()
    return subject.lower() in lowered or any(kw in lowered for kw in _ALL_ADVERSE_KEYWORDS)


def _rank_by_tfidf(sentences: list[str], query: str) -> list[tuple[str, float]]:
    """Rank sentences by TF-IDF cosine similarity to the query. Falls back to a
    neutral 0.0 score (stable order preserved) if vectorization fails - e.g. an
    all-stopword query or a single-sentence document with no shared vocabulary."""
    if not sentences:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([query, *sentences])
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return sorted(zip(sentences, scores.tolist()), key=lambda pair: pair[1], reverse=True)
    except ValueError as exc:
        logger.debug("TF-IDF ranking skipped (%s); preserving document order", exc)
        return [(s, 0.0) for s in sentences]


def run_evidence_mapping(
    documents: list[Document],
    subject: str,
    entities: list[Entity],
    research_mode: ResearchMode = ResearchMode.GENERAL,
) -> list[Evidence]:
    context_terms = [e.text for e in entities if e.text]

    if research_mode == ResearchMode.BACKGROUND_CHECK:
        query = " ".join([subject, *context_terms, *_ALL_ADVERSE_KEYWORDS])
    else:
        query = " ".join([subject, *context_terms])

    evidence_list: list[Evidence] = []
    for document in documents:
        sentences = _split_sentences(document.content)
        if research_mode == ResearchMode.BACKGROUND_CHECK:
            candidates = [s for s in sentences if _is_adverse_candidate(s, subject)]
            ranked = _rank_by_tfidf(candidates, query)[:_MAX_EVIDENCE_PER_DOCUMENT]
        else:
            ranked = [
                (s, score) for s, score in _rank_by_tfidf(sentences, query)
                if score >= _MIN_RELEVANCE_SCORE_GENERAL
            ][:_MAX_EVIDENCE_PER_DOCUMENT]

        for sentence, score in ranked:
            category = _categorize(sentence) if research_mode == ResearchMode.BACKGROUND_CHECK else AdverseCategory.GENERAL
            evidence_list.append(
                Evidence(
                    evidence_id=uuid.uuid4().hex[:12],
                    document_id=document.document_id,
                    claim=sentence,
                    supporting_passage=sentence,
                    category=category,
                    relevance_score=round(float(score), 3),
                    source_metadata=SourceMetadata(
                        url=document.url,
                        title=document.title,
                        source=document.source,
                        published_at=str(document.published_at) if document.published_at else None,
                    ),
                )
            )
    return evidence_list

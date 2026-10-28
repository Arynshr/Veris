"""
Evidence mapping, mode-aware and using TF-IDF cosine similarity
instead of raw keyword substring counting: Document -> relevant passages ->
Evidence -> candidate claim.
"""

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from veris.analysis.cleaning import IDENTITY_MISMATCH, NOISE, clean_claim, load_cleaning
from veris.analysis.matching import name_mentioned, namesake_mentioned, subject_match_score
from veris.analysis.safety import INSTRUCTION_LIKE, load_safety
from veris.analysis.taxonomy import load_taxonomy
from veris.core.config import Settings, get_logger
from veris.core.ids import stable_id
from veris.core.models import (
    Document,
    Entity,
    Evidence,
    ResearchMode,
    SourceMetadata,
)

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_SENTENCE_LEN = 40
_MAX_EVIDENCE_PER_DOCUMENT = 5
_MIN_RELEVANCE_SCORE_GENERAL = 0.05

@dataclass(frozen=True)
class _Sentence:
    text: str
    start: int
    end: int


def _split_sentences(text: str) -> list[_Sentence]:
    """All sentences with their offsets in `text` (short ones kept: they are
    valid window context even though they can't anchor evidence)."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_SPLIT_RE.finditer(text):
        bounds.append((start, match.start()))
        start = match.end()
    bounds.append((start, len(text)))

    sentences = []
    for begin, end in bounds:
        chunk = text[begin:end]
        stripped = chunk.strip()
        if stripped:
            offset = begin + len(chunk) - len(chunk.lstrip())
            sentences.append(_Sentence(stripped, offset, offset + len(stripped)))
    return sentences


def _rank_by_tfidf(texts: list[str], query: str) -> list[tuple[int, float]]:
    """Rank texts (as indices) by TF-IDF cosine similarity to the query. Falls
    back to a neutral 0.0 score (stable order preserved) if vectorization fails -
    e.g. an all-stopword query or a single-sentence document with no shared
    vocabulary."""
    if not texts:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([query, *texts])
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return sorted(
            enumerate(scores.tolist()), key=lambda pair: pair[1], reverse=True
        )
    except ValueError as exc:
        logger.debug("TF-IDF ranking skipped (%s); preserving document order", exc)
        return [(i, 0.0) for i in range(len(texts))]


def _window_bounds(
    sentences: list[_Sentence], anchor: int, settings: Settings
) -> tuple[int, int]:
    window = settings.evidence_window_sentences
    return max(0, anchor - window), min(len(sentences) - 1, anchor + window)


def _nearest_subject_mention(
    sentences: list[_Sentence],
    anchor: int,
    bounds: tuple[int, int],
    subject: str,
    settings: Settings,
) -> str | None:
    low, high = bounds
    nearest = min(
        (
            i
            for i, s in enumerate(sentences)
            if (i < low or i > high)
            and abs(i - anchor) <= settings.evidence_mention_max_distance
            and subject_match_score(s.text, subject)
            >= settings.min_subject_name_overlap
        ),
        key=lambda i: (abs(i - anchor), i),
        default=None,
    )
    return sentences[nearest].text if nearest is not None else None


def _passes_identity_gate(
    document: Document,
    subject: str,
    entities: list[Entity],
    settings: Settings,
    research_mode: ResearchMode,
) -> bool:
    """Document-level gate run before any passage extraction: the subject must be
    named, and in BACKGROUND_CHECK mode at least one context anchor (org, place...)
    from the request must co-occur - the cheapest defence against namesakes."""
    if (
        subject_match_score(document.content, subject)
        < settings.min_subject_name_overlap
    ):
        return False
    if (
        settings.reject_namesake_documents
        and namesake_mentioned(document.content, subject)
        and not name_mentioned(document.content, subject)
    ):
        return False
    if research_mode != ResearchMode.BACKGROUND_CHECK:
        return True
    lowered_subject = subject.lower()
    anchors = [
        e.text.lower()
        for e in entities
        if e.label in settings.identity_anchor_labels
        and e.text.lower() not in lowered_subject
        and lowered_subject not in e.text.lower()
    ]
    lowered_content = document.content.lower()
    return not anchors or any(a in lowered_content for a in anchors)


def run_evidence_mapping(
    documents: list[Document],
    subject: str,
    entities: list[Entity],
    settings: Settings,
    research_mode: ResearchMode = ResearchMode.GENERAL,
) -> list[Evidence]:
    taxonomy = load_taxonomy(settings.taxonomy_path)
    safety = load_safety(settings.safety_path)
    cleaning = load_cleaning(settings.cleaning_path)
    context_terms = [e.text for e in entities if e.text]

    if research_mode == ResearchMode.BACKGROUND_CHECK:
        query = " ".join([subject, *context_terms, *taxonomy.query_terms])
    else:
        query = " ".join([subject, *context_terms])

    evidence_list: list[Evidence] = []
    for document in documents:
        if not _passes_identity_gate(
            document, subject, entities, settings, research_mode
        ):
            logger.info("Identity gate rejected %s", document.url)
            continue

        sentences = _split_sentences(document.content)
        anchors = [
            i for i, s in enumerate(sentences) if len(s.text) >= _MIN_SENTENCE_LEN
        ]
        cleaned = {i: clean_claim(sentences[i].text, cleaning) for i in anchors}
        if research_mode == ResearchMode.BACKGROUND_CHECK:
            anchors = [
                i for i in anchors if taxonomy.match_categories(cleaned[i].snippet)
            ]
            ranked = _rank_by_tfidf([cleaned[i].snippet for i in anchors], query)
        else:
            ranked = [
                (pos, score)
                for pos, score in _rank_by_tfidf(
                    [cleaned[i].snippet for i in anchors], query
                )
                if score >= _MIN_RELEVANCE_SCORE_GENERAL
            ]
        # Noisy sentences only fill slots clean ones leave free (kept flagged for audit).
        ranked = sorted(ranked, key=lambda r: cleaned[anchors[r[0]]].noisy)[
            :_MAX_EVIDENCE_PER_DOCUMENT
        ]

        for pos, score in ranked:
            anchor = anchors[pos]
            sentence = sentences[anchor]
            low, high = bounds = _window_bounds(sentences, anchor, settings)
            passage = document.content[sentences[low].start : sentences[high].end]
            flags = []
            if cleaned[anchor].noisy:
                flags.append(NOISE)
            if namesake_mentioned(passage, subject) and not name_mentioned(passage, subject):
                flags.append(IDENTITY_MISMATCH)
            if safety.matches(passage):
                flags.append(INSTRUCTION_LIKE)
            evidence_list.append(
                Evidence(
                    evidence_id=stable_id(
                        document.document_id, sentence.start, sentence.end
                    ),
                    document_id=document.document_id,
                    claim=sentence.text,
                    snippet=cleaned[anchor].snippet,
                    supporting_passage=passage,
                    flags=sorted(flags),
                    subject_context=None
                    if subject_match_score(passage, subject)
                    >= settings.min_subject_name_overlap
                    else _nearest_subject_mention(
                        sentences, anchor, bounds, subject, settings
                    ),
                    span_start=sentence.start,
                    span_end=sentence.end,
                    relevance_score=round(float(score), 3),
                    source_metadata=SourceMetadata(
                        url=document.url,
                        title=document.title,
                        source=document.source,
                        published_at=str(document.published_at)
                        if document.published_at
                        else None,
                    ),
                )
            )
    return evidence_list

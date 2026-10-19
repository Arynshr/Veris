"""
Pydantic contracts shared by every pipeline stage (spec section 15).
Kept in one module since these are small, tightly related data shapes — splitting
them across files added navigation overhead without real separation of concerns.
"""

from datetime import UTC, date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ResearchMode(StrEnum):
    """What kind of request this is - determines planning strategy, evidence
    categorization, and report tone downstream. Detected during understanding,
    or set explicitly via --mode."""

    GENERAL = "general"
    BACKGROUND_CHECK = "background_check"


class VerificationStatus(StrEnum):
    """VERIFIED: corroborated by independent domains or a top-tier source.
    SINGLE_SOURCE: identity holds but support is uncorroborated.
    CONFLICTED: identity mismatch or a contradiction; never presented as fact."""

    VERIFIED = "VERIFIED"
    SINGLE_SOURCE = "SINGLE_SOURCE"
    CONFLICTED = "CONFLICTED"


class SearchProviderName(StrEnum):
    TAVILY = "tavily"
    FIRECRAWL = "firecrawl"
    SEARXNG = "searxng"


class SubjectType(StrEnum):
    PERSON = "person"
    COMPANY = "company"
    GROUP = "group"
    TOPIC = "topic"


class SubjectMode(StrEnum):
    """How a free-text request maps to concrete research subjects."""

    SINGLE = "single_subject"
    NAMED_LIST = "named_list"
    ROLE_GROUP = "role_group"


class LegalStatus(StrEnum):
    ALLEGED = "alleged"
    CHARGED = "charged"
    CONVICTED = "convicted"
    SETTLED = "settled"
    DISMISSED_ACQUITTED = "dismissed_acquitted"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class DateSource(StrEnum):
    METADATA = "metadata"
    TEXT = "text"
    URL = "url"
    NONE = "none"


class PipelineStage(StrEnum):
    UNDERSTANDING = "understanding"
    RESOLUTION = "resolution"
    PLANNING = "planning"
    SEARCH = "search"
    RETRIEVAL = "retrieval"
    REGISTRY = "registry"
    EVIDENCE = "evidence"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    OUTPUT = "output"


class EventStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineEvent(BaseModel):
    """Emitted by the pipeline as each stage progresses, for live CLI streaming."""

    stage: PipelineStage
    status: EventStatus
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict = Field(
        default_factory=dict
    )  # may include "subject" for per-subject stages


class Entity(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int


class IngestionResult(BaseModel):
    original_prompt: str
    entities: list[Entity] = Field(default_factory=list)
    system_instructions: str


class QueryUnderstanding(BaseModel):
    """Structured decomposition of the user's free-text request."""

    mode: SubjectMode
    research_mode: ResearchMode = ResearchMode.GENERAL
    organization: str | None = None
    role_terms: list[str] = Field(
        default_factory=list
    )  # e.g. ["CEO", "CFO"], ["board of directors"]
    explicit_subjects: list[str] = Field(
        default_factory=list
    )  # names/topics given verbatim
    raw_query: str


class ResolvedSubject(BaseModel):
    """A single concrete research target, whether given directly or looked up."""

    name: str
    subject_type: SubjectType = SubjectType.TOPIC
    role: str | None = None
    organization: str | None = None
    resolution_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="1.0 = user named this directly; lower = inferred via a resolution search.",
    )


class SearchQuery(BaseModel):
    query_id: str
    text: str
    priority: int = Field(ge=1, le=5, description="1 = highest priority")
    rationale: str = ""


class SourceRequirement(BaseModel):
    category: str
    min_sources: int = 1


class ResearchPlan(BaseModel):
    schema_version: int = 1
    objective: str
    queries: list[SearchQuery]
    source_requirements: list[SourceRequirement] = Field(default_factory=list)


class SearchResult(BaseModel):
    query_id: str
    provider: SearchProviderName
    url: str
    title: str
    snippet: str = ""
    rank: int
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Document(BaseModel):
    document_id: str
    url: str
    canonical_url: str
    title: str
    content: str
    source: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    published_at: datetime | None = None
    content_hash: str


class SourceMetadata(BaseModel):
    url: str
    title: str
    source: str
    published_at: str | None = None


class Evidence(BaseModel):
    schema_version: int = 1
    evidence_id: str
    document_id: str
    claim: str
    supporting_passage: str
    span_start: int = Field(
        description="Claim sentence offsets within Document.content."
    )
    span_end: int
    subject_context: str | None = Field(
        default=None,
        description="Nearest sentence naming the subject when the passage doesn't.",
    )
    source_metadata: SourceMetadata
    relevance_score: float = Field(
        default=0.0,
        description="TF-IDF cosine similarity to the subject/context query.",
    )
    flags: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    schema_version: int = 2
    claim_id: str
    text: str
    subject: str
    evidence_ids: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=lambda: ["uncategorized"])
    legal_status: LegalStatus = LegalStatus.UNKNOWN
    severity: Severity = Severity.LOW
    event_date: date | None = None
    published_date: date | None = None
    date_source: DateSource = DateSource.NONE
    out_of_window: bool = False
    undated: bool = False
    flags: list[str] = Field(
        default_factory=list, description="Set when every supporting evidence item is flagged."
    )


class VerificationResult(BaseModel):
    schema_version: int = 1
    claim_id: str
    status: VerificationStatus
    subject_match_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Does the claim actually reference the subject/topic?",
    )
    entity_match_score: float = Field(ge=0.0, le=1.0)
    source_quality_score: float = Field(ge=0.0, le=1.0)
    independent_domains: int = Field(default=0, ge=0)
    reasons: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(
        default_factory=list, description="Claim IDs on the opposing terminal-status side."
    )


class RetrievalStats(BaseModel):
    attempted: int = 0
    fetched: int = 0
    failed: int = 0
    paywalled: int = 0
    snippet_only: int = 0


class ResearchRun(BaseModel):
    schema_version: int = 1
    run_id: str
    query: str
    mode: SubjectMode
    research_mode: ResearchMode
    subjects: list[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    output_path: str
    sources_retrieved: int = 0
    evidence_collected: int = 0
    claims_verified: int = 0
    claims_single_source: int = 0
    claims_conflicted: int = 0
    queries_run: int = 0
    providers_used: list[str] = Field(default_factory=list)
    retrieval: RetrievalStats = Field(default_factory=RetrievalStats)
    adverse_by_severity: dict[str, int] = Field(default_factory=dict)
    config_hash: str = ""


class SynthesisSlot(BaseModel):
    slot_id: str
    text: str
    claim_ids: list[str]

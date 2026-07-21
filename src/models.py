"""Pydantic models for the fashion-blogger selection pipeline."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class StrictModel(BaseModel):
    """Shared validation policy for every project model."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class Platform(StrEnum):
    """Platforms represented in the prepared mock datasets."""

    INSTAGRAM = "instagram"
    YOUTUBE_SHORTS = "youtube_shorts"
    TELEGRAM = "telegram"
    OTHER = "other"


class CandidateSource(StrEnum):
    """Documented provenance of a candidate profile."""

    PREPARED_PUBLIC_LIST = "prepared_public_list"
    MANUAL_RESEARCH = "manual_research"
    BRAND_SUBMISSION = "brand_submission"


class PriceSegment(StrEnum):
    """Audience purchasing segment used by the fashion scoring rules."""

    MASS_MARKET = "mass_market"
    MIDDLE = "middle"
    PREMIUM = "premium"
    MIXED = "mixed"


class BrandSafetyLevel(StrEnum):
    """Editorially reviewed suitability for brand collaboration."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AuditReason(StrEnum):
    """Final disposition of every link returned by a search provider."""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    UNSUPPORTED_DOMAIN = "unsupported_domain"
    BRAND_OR_STORE = "brand_or_store"
    INSUFFICIENT_DATA = "insufficient_data"
    LOW_CONFIDENCE = "low_confidence"
    BELOW_MIN_SCORE = "below_min_score"


class RawPlatform(StrEnum):
    """Platform detected solely from a raw public URL domain."""

    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    TELEGRAM = "telegram"
    UNKNOWN = "unknown"


class SourceValidationStatus(StrEnum):
    """Validation state of an unmodified source-table row."""

    VALID = "valid"
    PARTIAL = "partial"
    INVALID = "invalid"


class ProfileEnrichmentStatus(StrEnum):
    """Outcome of collecting one public Instagram profile."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class ProfileEnrichmentAuditStatus(StrEnum):
    """Disposition of one URL supplied to profile enrichment."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INVALID_URL = "invalid_url"
    DUPLICATE = "duplicate"
    SKIPPED_LIMIT = "skipped_limit"


class RawSourceBlogger(StrictModel):
    """Lossless normalized view of a row from an unknown source schema."""

    original_name: str | None = Field(default=None, max_length=1_000)
    profile_url: str | None = Field(default=None, max_length=2_000)
    platform: RawPlatform
    source_notes: str | None = Field(default=None, max_length=4_000)
    raw_fields: dict[str, str | int | float | bool | None]
    validation_status: SourceValidationStatus
    missing_data: list[str]


class SourceInspectionReport(StrictModel):
    """Aggregate diagnostic report persisted without source-row contents."""

    source_url: str
    sheet_gid: int | None
    row_count: int = Field(ge=0)
    original_columns: list[str]
    detected_mapping: dict[str, str]
    unmapped_columns: list[str]
    missing_required_fields: list[str]
    rows_with_valid_urls: int = Field(ge=0)
    rows_without_urls: int = Field(ge=0)
    platform_distribution: dict[str, int]
    inspection_timestamp: str


class PublicInstagramProfile(StrictModel):
    """Public profile fields required for later content analysis."""

    username: str = Field(pattern=r"^[A-Za-z0-9._]{1,30}$")
    profile_url: HttpUrl
    full_name: str | None = Field(default=None, max_length=500)
    biography: str | None = Field(default=None, max_length=5_000)
    followers_count: int | None = Field(default=None, ge=0)
    following_count: int | None = Field(default=None, ge=0)
    posts_count: int | None = Field(default=None, ge=0)
    is_verified: bool | None = None
    is_private: bool | None = None
    external_url: HttpUrl | None = None
    profile_image_url: HttpUrl | None = None
    raw_source: str = Field(min_length=1, max_length=100)
    fetched_at: datetime


class PublicInstagramPost(StrictModel):
    """Small public-data subset for one recent Instagram publication."""

    post_url: HttpUrl | None = None
    post_type: str | None = Field(default=None, max_length=100)
    caption: str | None = Field(default=None, max_length=20_000)
    hashtags: list[str] | None = Field(default=None, max_length=100)
    likes_count: int | None = Field(default=None, ge=0)
    comments_count: int | None = Field(default=None, ge=0)
    timestamp: datetime | None = None
    display_url: HttpUrl | None = None
    video_url: HttpUrl | None = None
    accessibility_caption: str | None = Field(default=None, max_length=5_000)


class EnrichedSourceBlogger(StrictModel):
    """A source blogger joined with recent public content and completeness data."""

    profile: PublicInstagramProfile
    recent_posts: list[PublicInstagramPost]
    calculated_engagement_rate: float | None = Field(default=None, ge=0)
    available_post_count: int = Field(ge=0)
    data_confidence: float = Field(ge=0, le=1)
    missing_fields: list[str]
    enrichment_status: ProfileEnrichmentStatus
    enrichment_error: str | None = Field(default=None, max_length=2_000)


class LLMRecentPostInput(StrictModel):
    """Minimal recent-post evidence that may be sent to an LLM provider."""

    caption: str | None = Field(default=None, max_length=1_500)
    hashtags: list[str] = Field(default_factory=list, max_length=30)
    post_type: str | None = Field(default=None, max_length=100)
    accessibility_caption: str | None = Field(default=None, max_length=800)


class LLMSourceProfileInput(StrictModel):
    """Allow-listed source-profile data for batch analysis."""

    username: str = Field(pattern=r"^[A-Za-z0-9._]{1,30}$")
    full_name: str | None = Field(default=None, max_length=300)
    biography: str | None = Field(default=None, max_length=1_000)
    followers_count: int | None = Field(default=None, ge=0)
    calculated_engagement_rate: float | None = Field(default=None, ge=0)
    is_private: bool | None = None
    recent_posts: list[LLMRecentPostInput] = Field(default_factory=list, max_length=10)


class BatchBloggerInsights(StrictModel):
    """Pydantic Structured Output returned for one source-profile batch."""

    analyzed_usernames: list[str] = Field(min_length=1, max_length=25)
    dominant_topics: list[str] = Field(max_length=20)
    secondary_topics: list[str] = Field(max_length=20)
    content_formats: list[str] = Field(max_length=20)
    tone_patterns: list[str] = Field(max_length=20)
    audience_signals: list[str] = Field(max_length=20)
    price_segment_signals: list[str] = Field(max_length=20)
    engagement_observations: list[str] = Field(max_length=20)
    advertising_load_signals: list[str] = Field(max_length=20)
    brand_safety_observations: list[str] = Field(max_length=20)
    positive_patterns: list[str] = Field(max_length=20)
    negative_patterns: list[str] = Field(max_length=20)
    uncertainty_notes: list[str] = Field(max_length=20)
    confidence_score: float = Field(ge=0, le=1)


class LLMIdealBloggerProfile(StrictModel):
    """Evidence-based ideal profile synthesized from all batch insights.

    The distinct class name preserves the deterministic scoring portrait used by
    the existing MVP. Its JSON schema title remains ``IdealBloggerProfile`` for
    the Responses API contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        title="IdealBloggerProfile",
    )

    dominant_topics: list[str] = Field(max_length=20)
    secondary_topics: list[str] = Field(max_length=20)
    content_formats: list[str] = Field(max_length=20)
    visual_style_signals: list[str] = Field(max_length=20)
    tone_of_voice: list[str] = Field(max_length=20)
    target_audience: list[str] = Field(max_length=20)
    audience_interests: list[str] = Field(max_length=20)
    price_segment: str = Field(min_length=1, max_length=500)
    typical_follower_range: str = Field(min_length=1, max_length=500)
    engagement_rate_range: str = Field(min_length=1, max_length=500)
    advertising_load_preferences: list[str] = Field(max_length=20)
    preferred_integration_formats: list[str] = Field(max_length=20)
    brand_safety_requirements: list[str] = Field(max_length=20)
    positive_signals: list[str] = Field(max_length=20)
    negative_signals: list[str] = Field(max_length=20)
    exclusion_criteria: list[str] = Field(max_length=20)
    search_keywords: list[str] = Field(max_length=30)
    search_queries: list[str] = Field(max_length=20)
    confidence_score: float = Field(ge=0, le=1)
    evidence_summary: str = Field(min_length=1, max_length=4_000)
    sample_profile_usernames: list[str] = Field(max_length=25)


class LLMAnalysisAudit(StrictModel):
    """Safe operational metadata for a mock, dry-run, or OpenAI analysis."""

    provider: str = Field(min_length=1, max_length=50)
    model: str = Field(min_length=1, max_length=200)
    profile_count: int = Field(ge=0)
    batch_count: int = Field(ge=0)
    duration: float = Field(ge=0)
    retries: int = Field(ge=0)
    usage: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list, max_length=100)
    completed_batches: int = Field(default=0, ge=0)
    dry_run: bool = False
    inspection_timestamp: str


class ProfileEnrichmentAuditRow(StrictModel):
    """Safe audit record for one source URL without API headers or raw payloads."""

    input_url: str | None = Field(default=None, max_length=2_000)
    username: str | None = Field(default=None, max_length=30)
    status: ProfileEnrichmentAuditStatus
    reason: str = Field(min_length=1, max_length=2_000)
    cache_used: bool
    fetched_at: datetime | None = None


class SourceBlogger(StrictModel):
    """A known suitable fashion blogger used as a positive example."""

    handle: str = Field(min_length=2, max_length=100)
    display_name: str = Field(min_length=1, max_length=200)
    platform: Platform
    profile_url: HttpUrl
    followers: int = Field(ge=0)
    engagement_rate_pct: float = Field(ge=0, le=100)
    average_views: int = Field(ge=0)
    content_topics: list[str] = Field(min_length=1, max_length=15)
    audience_description: str = Field(min_length=1, max_length=1_000)
    audience_interests: list[str] = Field(min_length=1, max_length=15)
    content_style: str = Field(min_length=1, max_length=500)
    content_formats: list[str] = Field(min_length=1, max_length=10)
    tone: list[str] = Field(min_length=1, max_length=10)
    visual_style: list[str] = Field(min_length=1, max_length=10)
    advertising_load_pct: float = Field(ge=0, le=100)
    price_segment: PriceSegment
    brand_safety: BrandSafetyLevel
    location: str = Field(min_length=1, max_length=200)
    collaboration_notes: str = Field(min_length=1, max_length=1_000)


class CandidateProfile(StrictModel):
    """A fashion candidate supplied through a reviewed public-profile list."""

    handle: str = Field(min_length=2, max_length=100)
    display_name: str = Field(min_length=1, max_length=200)
    platform: Platform
    profile_url: HttpUrl
    followers: int | None = Field(default=None, ge=0)
    engagement_rate_pct: float | None = Field(default=None, ge=0, le=100)
    average_views: int | None = Field(default=None, ge=0)
    content_topics: list[str] | None = Field(default=None, min_length=1, max_length=15)
    audience_description: str | None = Field(default=None, min_length=1, max_length=1_000)
    audience_interests: list[str] | None = Field(default=None, min_length=1, max_length=15)
    content_style: str | None = Field(default=None, min_length=1, max_length=500)
    content_formats: list[str] | None = Field(default=None, min_length=1, max_length=10)
    tone: list[str] | None = Field(default=None, min_length=1, max_length=10)
    visual_style: list[str] | None = Field(default=None, min_length=1, max_length=10)
    advertising_load_pct: float | None = Field(default=None, ge=0, le=100)
    price_segment: PriceSegment | None = None
    brand_safety: BrandSafetyLevel | None = None
    location: str | None = Field(default=None, min_length=1, max_length=200)
    source: CandidateSource
    source_url: HttpUrl
    notes: str = Field(min_length=1, max_length=1_000)
    source_query: str | None = Field(default=None, max_length=1_000)
    source_title: str | None = Field(default=None, max_length=1_000)
    source_snippet: str | None = Field(default=None, max_length=4_000)
    data_confidence: float = Field(default=1.0, ge=0, le=1)


class SearchHit(StrictModel):
    """One untrusted result returned by a replaceable search provider."""

    url: str = Field(min_length=1, max_length=2_000)
    title: str | None = Field(default=None, max_length=1_000)
    snippet: str | None = Field(default=None, max_length=4_000)
    source_query: str = Field(min_length=1, max_length=1_000)
    provider_score: float | None = Field(default=None, ge=0, le=1)
    prefilled_candidate: CandidateProfile | None = Field(default=None, exclude=True)


class ContentAuthorResolution(StrictModel):
    """Auditable conversion of a content URL into its public author profile."""

    content_url: str = Field(min_length=1, max_length=2_000)
    platform: Platform | None = None
    resolved_profile_url: HttpUrl | None = None
    resolved_username_or_channel: str | None = Field(default=None, max_length=300)
    resolution_method: str = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0, le=1)
    status: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=2_000)


class EnrichedCandidate(StrictModel):
    """Public search evidence with nullable inferred fields."""

    name: str | None = Field(default=None, max_length=200)
    platform: Platform | None = None
    profile_url: HttpUrl
    title: str | None = Field(default=None, max_length=1_000)
    snippet: str | None = Field(default=None, max_length=4_000)
    source_query: str = Field(min_length=1, max_length=1_000)
    content_topics: list[str] | None = Field(default=None, max_length=15)
    visual_style: list[str] | None = Field(default=None, max_length=10)
    tone: list[str] | None = Field(default=None, max_length=10)
    audience_description: str | None = Field(default=None, max_length=1_000)
    audience_interests: list[str] | None = Field(default=None, max_length=15)
    content_formats: list[str] | None = Field(default=None, max_length=10)
    content_style: str | None = Field(default=None, max_length=500)
    followers: int | None = Field(default=None, ge=0)
    engagement_rate_pct: float | None = Field(default=None, ge=0, le=100)
    average_views: int | None = Field(default=None, ge=0)
    advertising_load_pct: float | None = Field(default=None, ge=0, le=100)
    price_segment: PriceSegment | None = None
    brand_safety: BrandSafetyLevel | None = None
    is_brand_or_store: bool
    data_confidence: float = Field(ge=0, le=1)


class SearchAuditRow(StrictModel):
    """Explainable final decision for one discovered URL."""

    url: str = Field(min_length=1, max_length=2_000)
    normalized_url: str | None = Field(default=None, max_length=2_000)
    source_query: str = Field(min_length=1, max_length=1_000)
    source_title: str | None = Field(default=None, max_length=1_000)
    data_confidence: float | None = Field(default=None, ge=0, le=1)
    reason: AuditReason


class RealCandidateProfile(StrictModel):
    """Normalized real-search evidence, optionally enriched by Apify."""

    name: str = Field(min_length=1, max_length=300)
    username: str = Field(min_length=1, max_length=200)
    platform: Platform
    profile_url: HttpUrl
    title: str | None = Field(default=None, max_length=1_000)
    snippet: str | None = Field(default=None, max_length=4_000)
    source_query: str = Field(min_length=1, max_length=1_000)
    tavily_score: float | None = Field(default=None, ge=0, le=1)
    full_name: str | None = Field(default=None, max_length=500)
    biography: str | None = Field(default=None, max_length=2_000)
    followers_count: int | None = Field(default=None, ge=0)
    engagement_rate: float | None = Field(default=None, ge=0)
    is_private: bool | None = None
    recent_posts: list[LLMRecentPostInput] = Field(default_factory=list, max_length=3)
    content_formats: list[str] = Field(default_factory=list, max_length=10)
    evidence: list[str] = Field(default_factory=list, max_length=20)
    evidence_count: int = Field(default=1, ge=1)
    evidence_urls: list[str] = Field(default_factory=list, max_length=20)
    author_resolution_confidence: float = Field(default=1.0, ge=0, le=1)
    data_confidence: float = Field(ge=0, le=1)
    enrichment_status: str = Field(min_length=1, max_length=100)
    enrichment_error: str | None = Field(default=None, max_length=1_000)


class FinalScoreBreakdown(StrictModel):
    """Confidence-adjusted final scoring components totaling at most 100."""

    fashion_relevance_score: int = Field(ge=0, le=20)
    visual_text_score: int = Field(ge=0, le=15)
    audience_score: int = Field(ge=0, le=15)
    tone_score: int = Field(ge=0, le=10)
    engagement_score: int = Field(ge=0, le=10)
    advertising_load_score: int = Field(ge=0, le=10)
    price_segment_score: int = Field(ge=0, le=5)
    content_format_score: int = Field(ge=0, le=5)
    brand_safety_score: int = Field(ge=0, le=5)
    data_confidence_score: int = Field(ge=0, le=5)
    total_score: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_total(self) -> "FinalScoreBreakdown":
        component_total = (
            self.fashion_relevance_score
            + self.visual_text_score
            + self.audience_score
            + self.tone_score
            + self.engagement_score
            + self.advertising_load_score
            + self.price_segment_score
            + self.content_format_score
            + self.brand_safety_score
            + self.data_confidence_score
        )
        if self.total_score != component_total:
            raise ValueError("total_score must equal all final score components")
        return self


class ScoreCriterionDetail(StrictModel):
    """Human-readable evidence and confidence for one unchanged score component."""

    score: int = Field(ge=0, le=100)
    max_score: int = Field(ge=1, le=100)
    reason: str = Field(min_length=1, max_length=2_000)
    confidence: Literal["low", "medium", "high"]


class FinalScoredCandidate(StrictModel):
    """One real candidate with explainable confidence-adjusted scoring."""

    candidate: RealCandidateProfile
    score: FinalScoreBreakdown
    match_reason: str = Field(min_length=1, max_length=2_000)
    evidence: list[str] = Field(min_length=1, max_length=20)
    criterion_details: dict[str, ScoreCriterionDetail] = Field(default_factory=dict)


class FinalScoreBreakdownRow(StrictModel):
    """Flat explainable-scoring export shared by mock and real runs."""

    profile_url: HttpUrl
    platform: Platform
    total_score: int = Field(ge=0, le=100)
    max_score: int = Field(default=100, ge=100, le=100)
    evidence_count: int = Field(default=1, ge=1)
    topic_score: int = Field(ge=0, le=20)
    topic_reason: str = Field(min_length=1, max_length=2_000)
    visual_score: int = Field(ge=0, le=20)
    visual_reason: str = Field(min_length=1, max_length=2_000)
    audience_score: int = Field(ge=0, le=15)
    audience_reason: str = Field(min_length=1, max_length=2_000)
    tone_score: int = Field(ge=0, le=10)
    tone_reason: str = Field(min_length=1, max_length=2_000)
    engagement_score: int = Field(ge=0, le=10)
    engagement_reason: str = Field(min_length=1, max_length=2_000)
    ad_load_score: int = Field(ge=0, le=10)
    ad_load_reason: str = Field(min_length=1, max_length=2_000)
    price_segment_score: int = Field(ge=0, le=5)
    price_segment_reason: str = Field(min_length=1, max_length=2_000)
    format_score: int = Field(ge=0, le=5)
    format_reason: str = Field(min_length=1, max_length=2_000)
    brand_safety_score: int = Field(ge=0, le=5)
    brand_safety_reason: str = Field(min_length=1, max_length=2_000)
    data_confidence: float = Field(ge=0, le=1)
    confidence_reason: str = Field(min_length=1, max_length=2_000)


class FinalPersonalizedOffer(StrictModel):
    """Structured OpenAI draft that is never sent automatically."""

    candidate_username: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=3_000)
    evidence_used: list[str] = Field(min_length=1, max_length=8)
    human_review_required: bool

    @model_validator(mode="after")
    def require_human_review(self) -> "FinalPersonalizedOffer":
        if not self.human_review_required:
            raise ValueError("human_review_required must be true")
        return self


class FinalRealBloggerRow(StrictModel):
    """Exact public-data schema for the final manual-review CSV."""

    name: str = Field(min_length=1, max_length=300)
    username: str = Field(min_length=1, max_length=200)
    platform: Platform
    profile_url: HttpUrl
    total_score: int = Field(ge=0, le=100)
    data_confidence: float = Field(ge=0, le=1)
    followers_count: int | None = Field(default=None, ge=0)
    engagement_rate: float | None = Field(default=None, ge=0)
    match_reason: str = Field(min_length=1, max_length=2_000)
    evidence: str = Field(min_length=1, max_length=5_000)
    personalized_offer: str = Field(min_length=1, max_length=3_000)
    manual_review_status: Literal["needs_review"] = "needs_review"


class RealCandidateAuditRow(StrictModel):
    """One evolving decision for every Tavily URL in a final run."""

    raw_url: str = Field(min_length=1, max_length=2_000)
    normalized_url: str | None = Field(default=None, max_length=2_000)
    platform: Platform | None = None
    source_query: str = Field(min_length=1, max_length=1_000)
    title: str | None = Field(default=None, max_length=1_000)
    tavily_score: float | None = Field(default=None, ge=0, le=1)
    decision: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=2_000)
    data_confidence: float | None = Field(default=None, ge=0, le=1)
    total_score: int | None = Field(default=None, ge=0, le=100)
    evidence_count: int | None = Field(default=None, ge=1)
    evidence_urls: list[str] = Field(default_factory=list, max_length=20)
    author_resolution_confidence: float | None = Field(default=None, ge=0, le=1)


class FinalRunAudit(StrictModel):
    """Safe aggregate audit without credentials or request bodies."""

    status: str = Field(min_length=1, max_length=100)
    dry_run: bool
    query_count: int = Field(ge=0)
    tavily_result_limit: int = Field(ge=0)
    raw_result_count: int = Field(ge=0)
    candidates_before_enrichment: int = Field(ge=0)
    apify_candidate_count: int = Field(ge=0)
    enriched_candidate_count: int = Field(ge=0)
    scored_candidate_count: int = Field(ge=0)
    finalist_count: int = Field(ge=0, le=5)
    openai_offer_calls: int = Field(ge=0, le=5)
    final_min_score: int = Field(ge=0, le=100)
    limits: dict[str, int]
    openai_model: str = Field(min_length=1, max_length=200)
    openai_usage: dict[str, int] = Field(default_factory=dict)
    apify_content_resolution_runs: int = Field(default=0, ge=0)
    content_urls_found: int = Field(default=0, ge=0)
    content_resolution_attempted: int = Field(default=0, ge=0)
    resolved_author_count: int = Field(default=0, ge=0)
    unresolved_author_count: int = Field(default=0, ge=0)
    duration_seconds: float = Field(ge=0)
    errors: list[str] = Field(default_factory=list, max_length=100)
    finished_at: str


class IdealBloggerProfile(StrictModel):
    """Fashion-relevant aggregate portrait inferred from source bloggers."""

    summary: str = Field(min_length=1, max_length=2_000)
    content_topics: list[str] = Field(min_length=1, max_length=15)
    content_formats: list[str] = Field(min_length=1, max_length=10)
    visual_style: list[str] = Field(min_length=1, max_length=10)
    tone: list[str] = Field(min_length=1, max_length=10)
    target_audience: list[str] = Field(min_length=1, max_length=10)
    price_segment: PriceSegment
    engagement_rate_pct: float = Field(ge=0, le=100)
    advertising_load_pct: float = Field(ge=0, le=100)
    brand_safety: BrandSafetyLevel
    preferred_integration_formats: list[str] = Field(min_length=1, max_length=10)
    target_platforms: list[Platform] = Field(min_length=1)
    followers_min: int = Field(ge=0)
    followers_max: int = Field(ge=0)
    preferred_locations: list[str] = Field(min_length=1, max_length=20)
    must_have_traits: list[str] = Field(min_length=1, max_length=10)
    red_flags: list[str] = Field(max_length=10)

    @model_validator(mode="after")
    def validate_follower_range(self) -> "IdealBloggerProfile":
        """Reject an inverted audience-size range."""

        if self.followers_max < self.followers_min:
            raise ValueError("followers_max must be greater than or equal to followers_min")
        return self


class ScoreBreakdown(StrictModel):
    """Nine fashion criteria that add up to a 100-point score."""

    topic_score: int = Field(ge=0, le=20)
    visual_score: int = Field(ge=0, le=20)
    audience_score: int = Field(ge=0, le=15)
    tone_score: int = Field(ge=0, le=10)
    engagement_score: int = Field(ge=0, le=10)
    ad_load_score: int = Field(ge=0, le=10)
    price_segment_score: int = Field(ge=0, le=5)
    format_score: int = Field(ge=0, le=5)
    brand_safety_score: int = Field(ge=0, le=5)
    total_score: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_total(self) -> "ScoreBreakdown":
        """Keep the persisted total consistent with its components."""

        component_total = (
            self.topic_score
            + self.visual_score
            + self.audience_score
            + self.tone_score
            + self.engagement_score
            + self.ad_load_score
            + self.price_segment_score
            + self.format_score
            + self.brand_safety_score
        )
        if self.total_score != component_total:
            raise ValueError("total_score must equal the sum of all score components")
        return self


class CandidateEvaluation(StrictModel):
    """A fashion candidate score with its nearest source blogger."""

    candidate_handle: str = Field(min_length=2, max_length=100)
    score: ScoreBreakdown
    similar_to: str = Field(min_length=1, max_length=200)
    match_reason: str = Field(min_length=1, max_length=2_000)
    criterion_details: dict[str, ScoreCriterionDetail] = Field(default_factory=dict)


class RankedCandidate(StrictModel):
    """A candidate joined with its evaluation and final rank."""

    rank: int = Field(ge=1)
    candidate: CandidateProfile
    evaluation: CandidateEvaluation


class BarterOffer(StrictModel):
    """A deterministic personalized fashion barter proposal."""

    candidate_handle: str = Field(min_length=2, max_length=100)
    subject: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=3_000)
    proposed_barter: str = Field(min_length=1, max_length=1_000)
    personalization_facts: list[str] = Field(min_length=1, max_length=8)


class ResultRow(StrictModel):
    """Exact flat schema written to data/results.csv."""

    name: str = Field(min_length=1, max_length=200)
    platform: Platform
    profile_url: HttpUrl
    total_score: int = Field(ge=0, le=100)
    topic_score: int = Field(ge=0, le=20)
    visual_score: int = Field(ge=0, le=20)
    audience_score: int = Field(ge=0, le=15)
    tone_score: int = Field(ge=0, le=10)
    engagement_score: int = Field(ge=0, le=10)
    ad_load_score: int = Field(ge=0, le=10)
    price_segment_score: int = Field(ge=0, le=5)
    format_score: int = Field(ge=0, le=5)
    brand_safety_score: int = Field(ge=0, le=5)
    similar_to: str = Field(min_length=1, max_length=200)
    match_reason: str = Field(min_length=1, max_length=2_000)
    personalized_offer: str = Field(min_length=1, max_length=3_000)
    source_query: str | None = Field(default=None, max_length=1_000)
    source_title: str | None = Field(default=None, max_length=1_000)
    source_snippet: str | None = Field(default=None, max_length=4_000)
    data_confidence: float = Field(ge=0, le=1)


class PipelineResult(StrictModel):
    """Validated in-memory result of a complete mock run."""

    ideal_profile: IdealBloggerProfile
    selected_candidates: list[RankedCandidate] = Field(max_length=5)
    offers: list[BarterOffer] = Field(max_length=5)

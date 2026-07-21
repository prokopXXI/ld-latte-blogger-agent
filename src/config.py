"""Environment-based settings and logging configuration."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationError(ValueError):
    """Raised when an environment value cannot produce a safe configuration."""


def _as_bool(value: str, variable_name: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(
        f"{variable_name} must be true/false, yes/no, on/off, or 1/0; got {value!r}"
    )


def _as_top_k(value: str) -> int:
    try:
        top_k = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"TOP_K must be an integer; got {value!r}") from exc
    if not 3 <= top_k <= 5:
        raise ConfigurationError("TOP_K must be between 3 and 5 for this demo")
    return top_k


def _as_int_range(value: str, variable_name: str, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{variable_name} must be an integer; got {value!r}") from exc
    if not minimum <= result <= maximum:
        raise ConfigurationError(
            f"{variable_name} must be between {minimum} and {maximum}; got {result}"
        )
    return result


def _as_positive_float(value: str, variable_name: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{variable_name} must be a number; got {value!r}") from exc
    if result <= 0:
        raise ConfigurationError(f"{variable_name} must be greater than zero")
    return result


def _as_non_negative_float(value: str, variable_name: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{variable_name} must be a number; got {value!r}") from exc
    if result < 0:
        raise ConfigurationError(f"{variable_name} must be zero or greater")
    return result


def _as_probability(value: str, variable_name: str) -> float:
    result = _as_non_negative_float(value, variable_name)
    if result > 1:
        raise ConfigurationError(f"{variable_name} must be between 0 and 1")
    return result


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings resolved once at application startup."""

    mock_mode: bool
    source_provider: str
    source_csv_path: Path
    source_real_csv_path: Path
    source_inspection_path: Path
    google_sheet_url: str | None
    google_sheet_gid: int
    profile_enrichment_provider: str
    apify_api_token: str | None
    apify_actor_id: str | None
    profile_posts_limit: int
    profile_enrichment_concurrency: int
    profile_enrichment_delay_seconds: float
    profile_cache_enabled: bool
    profile_cache_dir: Path
    profile_enrichment_mock_path: Path
    enriched_source_json_path: Path
    enriched_source_summary_path: Path
    profile_enrichment_audit_path: Path
    apify_raw_response_path: Path
    candidates_csv_path: Path
    results_csv_path: Path
    search_queries_path: Path
    search_audit_path: Path
    top_k: int
    min_score: int
    search_provider: str
    search_max_results: int
    tavily_api_key: str | None
    youtube_api_key: str | None
    log_level: str
    request_timeout_seconds: float
    llm_provider: str
    openai_api_key: str | None
    openai_model: str
    openai_max_profiles_per_batch: int
    openai_max_posts_per_profile: int
    openai_max_total_profiles: int
    openai_request_timeout_seconds: float
    llm_batch_prompt_path: Path
    llm_synthesis_prompt_path: Path
    llm_batch_insights_path: Path
    ideal_blogger_profile_json_path: Path
    ideal_blogger_profile_markdown_path: Path
    llm_analysis_audit_path: Path
    tavily_max_queries: int
    tavily_results_per_query: int
    max_candidates_before_enrichment: int
    max_candidates_for_apify: int
    max_final_candidates: int
    final_min_score: int
    real_candidate_cache_dir: Path
    real_candidates_raw_path: Path
    real_candidates_audit_path: Path
    real_candidates_enriched_path: Path
    final_real_bloggers_csv_path: Path
    final_real_bloggers_markdown_path: Path
    final_run_audit_path: Path
    final_offer_prompt_path: Path
    final_apify_raw_response_path: Path
    max_content_urls_for_resolution: int
    min_author_resolution_confidence: float
    content_author_cache_path: Path
    content_author_resolution_audit_path: Path
    content_author_apify_raw_path: Path
    real_candidates_raw_v2_path: Path
    real_candidates_audit_v2_path: Path
    real_candidates_enriched_v2_path: Path
    final_real_bloggers_csv_v2_path: Path
    final_real_bloggers_markdown_v2_path: Path
    final_run_audit_v2_path: Path
    final_apify_raw_response_v2_path: Path
    final_score_breakdown_path: Path
    final_all_candidates_csv_path: Path
    final_all_candidates_markdown_path: Path
    final_recommended_bloggers_csv_path: Path


def load_settings(env_file: Path | None = None) -> Settings:
    """Load `.env` values and return validated absolute project paths."""

    dotenv_path = env_file or PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)

    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if log_level not in logging.getLevelNamesMapping():
        raise ConfigurationError(f"Unknown LOG_LEVEL: {log_level!r}")
    search_provider = os.getenv("SEARCH_PROVIDER", "mock").strip().casefold()
    if search_provider not in {"mock", "tavily"}:
        raise ConfigurationError(
            "SEARCH_PROVIDER must be 'mock' or 'tavily'; "
            f"got {search_provider!r}"
        )
    source_provider = os.getenv("SOURCE_PROVIDER", "csv").strip().casefold()
    if source_provider not in {"csv", "google_sheets"}:
        raise ConfigurationError(
            "SOURCE_PROVIDER must be 'csv' or 'google_sheets'; "
            f"got {source_provider!r}"
        )
    profile_enrichment_provider = os.getenv(
        "PROFILE_ENRICHMENT_PROVIDER", "mock"
    ).strip().casefold()
    if profile_enrichment_provider not in {"mock", "apify"}:
        raise ConfigurationError(
            "PROFILE_ENRICHMENT_PROVIDER must be 'mock' or 'apify'; "
            f"got {profile_enrichment_provider!r}"
        )
    llm_provider = os.getenv("LLM_PROVIDER", "mock").strip().casefold()
    if llm_provider not in {"mock", "openai"}:
        raise ConfigurationError(
            "LLM_PROVIDER must be 'mock' or 'openai'; "
            f"got {llm_provider!r}"
        )
    max_candidates_before_enrichment = _as_int_range(
        os.getenv("MAX_CANDIDATES_BEFORE_ENRICHMENT", "20"),
        "MAX_CANDIDATES_BEFORE_ENRICHMENT",
        1,
        100,
    )
    max_candidates_for_apify = _as_int_range(
        os.getenv("MAX_CANDIDATES_FOR_APIFY", "8"),
        "MAX_CANDIDATES_FOR_APIFY",
        1,
        20,
    )
    max_final_candidates = _as_int_range(
        os.getenv("MAX_FINAL_CANDIDATES", "5"),
        "MAX_FINAL_CANDIDATES",
        3,
        5,
    )
    if max_candidates_for_apify > max_candidates_before_enrichment:
        raise ConfigurationError(
            "MAX_CANDIDATES_FOR_APIFY cannot exceed "
            "MAX_CANDIDATES_BEFORE_ENRICHMENT"
        )
    if max_final_candidates > max_candidates_before_enrichment:
        raise ConfigurationError(
            "MAX_FINAL_CANDIDATES cannot exceed "
            "MAX_CANDIDATES_BEFORE_ENRICHMENT"
        )

    return Settings(
        mock_mode=_as_bool(os.getenv("MOCK_MODE", "true"), "MOCK_MODE"),
        source_provider=source_provider,
        source_csv_path=_project_path(
            os.getenv("SOURCE_CSV_PATH", "data/source_bloggers.example.csv")
        ),
        source_real_csv_path=_project_path(
            os.getenv("SOURCE_REAL_CSV_PATH", "data/source_bloggers.real.csv")
        ),
        source_inspection_path=_project_path(
            os.getenv("SOURCE_INSPECTION_PATH", "data/source_inspection.json")
        ),
        google_sheet_url=os.getenv("GOOGLE_SHEET_URL", "").strip() or None,
        google_sheet_gid=_as_int_range(
            os.getenv("GOOGLE_SHEET_GID", "0"),
            "GOOGLE_SHEET_GID",
            0,
            2_147_483_647,
        ),
        profile_enrichment_provider=profile_enrichment_provider,
        apify_api_token=os.getenv("APIFY_API_TOKEN", "").strip() or None,
        apify_actor_id=os.getenv("APIFY_ACTOR_ID", "").strip() or None,
        profile_posts_limit=_as_int_range(
            os.getenv("PROFILE_POSTS_LIMIT", "6"),
            "PROFILE_POSTS_LIMIT",
            1,
            50,
        ),
        profile_enrichment_concurrency=_as_int_range(
            os.getenv("PROFILE_ENRICHMENT_CONCURRENCY", "2"),
            "PROFILE_ENRICHMENT_CONCURRENCY",
            1,
            10,
        ),
        profile_enrichment_delay_seconds=_as_non_negative_float(
            os.getenv("PROFILE_ENRICHMENT_DELAY_SECONDS", "1"),
            "PROFILE_ENRICHMENT_DELAY_SECONDS",
        ),
        profile_cache_enabled=_as_bool(
            os.getenv("PROFILE_CACHE_ENABLED", "true"),
            "PROFILE_CACHE_ENABLED",
        ),
        profile_cache_dir=_project_path("data/profile_cache"),
        profile_enrichment_mock_path=_project_path(
            "data/profile_enrichment_mock.json"
        ),
        enriched_source_json_path=_project_path(
            "data/enriched_source_bloggers.json"
        ),
        enriched_source_summary_path=_project_path(
            "data/enriched_source_summary.csv"
        ),
        profile_enrichment_audit_path=_project_path(
            "data/profile_enrichment_audit.csv"
        ),
        apify_raw_response_path=_project_path(
            os.getenv("APIFY_RAW_RESPONSE_PATH", "data/apify_raw_response.json")
        ),
        candidates_csv_path=_project_path(
            os.getenv("CANDIDATES_CSV_PATH", "data/candidates.example.csv")
        ),
        results_csv_path=_project_path(
            os.getenv("RESULTS_CSV_PATH", "data/results.csv")
        ),
        search_queries_path=_project_path(
            os.getenv("SEARCH_QUERIES_PATH", "data/search_queries.json")
        ),
        search_audit_path=_project_path(
            os.getenv("SEARCH_AUDIT_PATH", "data/search_audit.csv")
        ),
        top_k=_as_top_k(os.getenv("TOP_K", "5")),
        min_score=_as_int_range(os.getenv("MIN_SCORE", "70"), "MIN_SCORE", 0, 100),
        search_provider=search_provider,
        search_max_results=_as_int_range(
            os.getenv("SEARCH_MAX_RESULTS", "30"),
            "SEARCH_MAX_RESULTS",
            1,
            30,
        ),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip() or None,
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", "").strip() or None,
        log_level=log_level,
        request_timeout_seconds=_as_positive_float(
            os.getenv("REQUEST_TIMEOUT_SECONDS", "30"),
            "REQUEST_TIMEOUT_SECONDS",
        ),
        llm_provider=llm_provider,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip() or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini").strip() or "gpt-5-mini",
        openai_max_profiles_per_batch=_as_int_range(
            os.getenv("OPENAI_MAX_PROFILES_PER_BATCH", "8"),
            "OPENAI_MAX_PROFILES_PER_BATCH",
            1,
            25,
        ),
        openai_max_posts_per_profile=_as_int_range(
            os.getenv("OPENAI_MAX_POSTS_PER_PROFILE", "3"),
            "OPENAI_MAX_POSTS_PER_PROFILE",
            1,
            10,
        ),
        openai_max_total_profiles=_as_int_range(
            os.getenv("OPENAI_MAX_TOTAL_PROFILES", "25"),
            "OPENAI_MAX_TOTAL_PROFILES",
            1,
            100,
        ),
        openai_request_timeout_seconds=_as_positive_float(
            os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "120"),
            "OPENAI_REQUEST_TIMEOUT_SECONDS",
        ),
        llm_batch_prompt_path=_project_path("prompts/source_batch_analysis.txt"),
        llm_synthesis_prompt_path=_project_path("prompts/ideal_profile_synthesis.txt"),
        llm_batch_insights_path=_project_path("data/llm_batch_insights.json"),
        ideal_blogger_profile_json_path=_project_path(
            "data/ideal_blogger_profile.json"
        ),
        ideal_blogger_profile_markdown_path=_project_path(
            "data/ideal_blogger_profile.md"
        ),
        llm_analysis_audit_path=_project_path("data/llm_analysis_audit.json"),
        tavily_max_queries=_as_int_range(
            os.getenv("TAVILY_MAX_QUERIES", "9"),
            "TAVILY_MAX_QUERIES",
            1,
            20,
        ),
        tavily_results_per_query=_as_int_range(
            os.getenv("TAVILY_RESULTS_PER_QUERY", "5"),
            "TAVILY_RESULTS_PER_QUERY",
            1,
            20,
        ),
        max_candidates_before_enrichment=max_candidates_before_enrichment,
        max_candidates_for_apify=max_candidates_for_apify,
        max_final_candidates=max_final_candidates,
        final_min_score=_as_int_range(
            os.getenv("FINAL_MIN_SCORE", "70"),
            "FINAL_MIN_SCORE",
            0,
            100,
        ),
        real_candidate_cache_dir=_project_path("data/real_candidate_cache"),
        real_candidates_raw_path=_project_path("data/real_candidates_raw.csv"),
        real_candidates_audit_path=_project_path("data/real_candidates_audit.csv"),
        real_candidates_enriched_path=_project_path(
            "data/real_candidates_enriched.json"
        ),
        final_real_bloggers_csv_path=_project_path(
            "data/final_real_bloggers.csv"
        ),
        final_real_bloggers_markdown_path=_project_path(
            "data/final_real_bloggers.md"
        ),
        final_run_audit_path=_project_path("data/final_run_audit.json"),
        final_offer_prompt_path=_project_path("prompts/final_offer_generation.txt"),
        final_apify_raw_response_path=_project_path(
            "data/real_candidate_apify_raw.json"
        ),
        max_content_urls_for_resolution=_as_int_range(
            os.getenv("MAX_CONTENT_URLS_FOR_RESOLUTION", "20"),
            "MAX_CONTENT_URLS_FOR_RESOLUTION",
            1,
            20,
        ),
        min_author_resolution_confidence=_as_probability(
            os.getenv("MIN_AUTHOR_RESOLUTION_CONFIDENCE", "0.65"),
            "MIN_AUTHOR_RESOLUTION_CONFIDENCE",
        ),
        content_author_cache_path=_project_path(
            "data/content_author_resolution_cache.json"
        ),
        content_author_resolution_audit_path=_project_path(
            "data/content_author_resolution_audit.csv"
        ),
        content_author_apify_raw_path=_project_path(
            "data/content_author_apify_raw.json"
        ),
        real_candidates_raw_v2_path=_project_path(
            "data/real_candidates_raw_v2.csv"
        ),
        real_candidates_audit_v2_path=_project_path(
            "data/real_candidates_audit_v2.csv"
        ),
        real_candidates_enriched_v2_path=_project_path(
            "data/real_candidates_enriched_v2.json"
        ),
        final_real_bloggers_csv_v2_path=_project_path(
            "data/final_real_bloggers_v2.csv"
        ),
        final_real_bloggers_markdown_v2_path=_project_path(
            "data/final_real_bloggers_v2.md"
        ),
        final_run_audit_v2_path=_project_path("data/final_run_audit_v2.json"),
        final_apify_raw_response_v2_path=_project_path(
            "data/real_candidate_apify_raw_v2.json"
        ),
        final_score_breakdown_path=_project_path(
            "data/final_score_breakdown.csv"
        ),
        final_all_candidates_csv_path=_project_path(
            "data/final_all_candidates.csv"
        ),
        final_all_candidates_markdown_path=_project_path(
            "data/final_all_candidates.md"
        ),
        final_recommended_bloggers_csv_path=_project_path(
            "data/final_recommended_bloggers.csv"
        ),
    )


def configure_logging(level: str) -> None:
    """Configure concise console logging for a one-command demo run."""

    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(name)s | %(message)s",
        force=True,
    )

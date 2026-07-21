"""Orchestration for the explicitly requested paid real-discovery pipeline."""

from __future__ import annotations

import json
import logging
import re
import time
import csv
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
from pydantic import ValidationError

from src.candidate_enricher import normalize_profile_url, platform_from_url
from src.config import Settings
from src.content_author_resolver import (
    ContentAuthorResolutionError,
    ContentAuthorResolver,
    classify_public_url,
    create_content_author_resolver,
)
from src.final_candidate_ranker import rank_real_candidates, select_finalists
from src.final_offer_generator import (
    FinalOfferGenerationError,
    OpenAIFinalOfferProvider,
    generate_final_offers,
)
from src.llm_profile_analyzer import prepare_llm_profile
from src.models import (
    ContentAuthorResolution,
    EnrichedSourceBlogger,
    FinalPersonalizedOffer,
    FinalRealBloggerRow,
    FinalRunAudit,
    FinalScoreBreakdownRow,
    FinalScoredCandidate,
    LLMIdealBloggerProfile,
    Platform,
    RealCandidateAuditRow,
    RealCandidateProfile,
    SearchHit,
)
from src.profile_enrichment_providers import (
    ProfileEnrichmentProvider,
    create_profile_enrichment_provider,
    enrich_profile_urls,
    load_profile_cache,
)
from src.search_providers import SearchProviderError, TavilySearchProvider


LOGGER = logging.getLogger(__name__)
RAW_COLUMNS = ["title", "url", "content", "source_query", "score"]
AUDIT_COLUMNS = [
    "raw_url",
    "normalized_url",
    "platform",
    "source_query",
    "title",
    "tavily_score",
    "decision",
    "reason",
    "data_confidence",
    "total_score",
]
AUDIT_V2_COLUMNS = AUDIT_COLUMNS + [
    "evidence_count",
    "evidence_urls",
    "author_resolution_confidence",
]
RESOLUTION_AUDIT_COLUMNS = [
    "content_url",
    "platform",
    "resolved_profile_url",
    "resolved_author",
    "resolution_method",
    "confidence",
    "status",
    "reason",
]
FINAL_COLUMNS = [
    "name",
    "username",
    "platform",
    "profile_url",
    "total_score",
    "data_confidence",
    "followers_count",
    "engagement_rate",
    "match_reason",
    "evidence",
    "personalized_offer",
    "manual_review_status",
]
SCORE_BREAKDOWN_COLUMNS = [
    "profile_url",
    "platform",
    "total_score",
    "max_score",
    "evidence_count",
    "topic_score",
    "topic_reason",
    "visual_score",
    "visual_reason",
    "audience_score",
    "audience_reason",
    "tone_score",
    "tone_reason",
    "engagement_score",
    "engagement_reason",
    "ad_load_score",
    "ad_load_reason",
    "price_segment_score",
    "price_segment_reason",
    "format_score",
    "format_reason",
    "brand_safety_score",
    "brand_safety_reason",
    "data_confidence",
    "confidence_reason",
]
FINAL_ALL_COLUMNS = [
    "name",
    "username",
    "platform",
    "profile_url",
    "total_score",
    "decision_status",
    "data_confidence",
    "evidence_count",
    "topic_score",
    "topic_reason",
    "visual_score",
    "visual_reason",
    "audience_score",
    "audience_reason",
    "tone_score",
    "tone_reason",
    "engagement_score",
    "engagement_reason",
    "ad_load_score",
    "ad_load_reason",
    "price_segment_score",
    "price_segment_reason",
    "format_score",
    "format_reason",
    "brand_safety_score",
    "brand_safety_reason",
    "confidence_reason",
    "scoring_breakdown",
    "match_reason",
    "personalized_offer",
    "manual_review_status",
]
FASHION_MARKERS = (
    "женская мода",
    "женская одежда",
    "fashion",
    "стиль",
    "образ",
    "outfit",
    "плать",
    "гардероб",
    "примерк",
    "try-on",
    "wildberries",
    "ozon",
    "маркетплейс",
)
AUTHOR_MARKERS = ("блог", "автор", "стилист", "образы", "примерк", "creator")
STORE_TITLE_MARKERS = (
    "интернет-магазин",
    "официальный магазин",
    "магазин одежды",
    "официальный бренд",
    "бренд женской одежды",
    "каталог товаров",
    "бутик одежды",
    "marketplace seller",
)
STORE_HANDLE_MARKERS = ("shop", "store", "official", "catalog", "market")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
FINAL_V2_SEARCH_QUERIES = (
    "Instagram fashion блогер примерки женской одежды Wildberries",
    "блогер капсульный гардероб женские образы Instagram",
    "YouTube канал примерки женской одежды Wildberries",
    "Telegram канал женская мода подборки образов",
    "примерка женской одежды Wildberries Reel",
    "капсульный гардероб женские образы Shorts",
    "офисные образы женская одежда Reel",
    "честный обзор одежды Wildberries блогер",
)


class FinalPipelineError(RuntimeError):
    """Raised for safe final-pipeline configuration or data failures."""


@dataclass(frozen=True, slots=True)
class FinalRunPlan:
    """Cost ceiling shown before any external client exists."""

    query_count: int
    tavily_result_limit: int
    max_candidates_before_enrichment: int
    max_apify_candidates: int
    max_openai_offers: int
    final_min_score: int
    openai_model: str
    missing_credentials: list[str]
    max_content_urls_for_resolution: int = 0
    max_apify_content_resolution_runs: int = 0
    min_author_resolution_confidence: float = 0.0
    pipeline_version: str = "v1"


@dataclass(frozen=True, slots=True)
class CleaningResult:
    """Accepted preliminary candidates and one audit row per raw result."""

    candidates: list[RealCandidateProfile]
    audit_rows: list[RealCandidateAuditRow]


@dataclass(frozen=True, slots=True)
class FinalPipelineResult:
    """Dry-run plan or completed real pipeline result."""

    plan: FinalRunPlan
    finalists: list[FinalScoredCandidate]
    audit: FinalRunAudit | None


@dataclass(frozen=True, slots=True)
class SavedPoolPlan:
    """Network-free preview for finalizing the already discovered v2 pool."""

    pool_size: int
    saved_enriched_count: int
    fresh_cache_count: int
    apify_required_count: int
    max_candidates_for_apify: int
    posts_per_profile: int
    max_openai_offers: int
    tavily_calls: int = 0


@dataclass(frozen=True, slots=True)
class SavedPoolResult:
    """Completed saved-pool scoring and human-review categorization."""

    plan: SavedPoolPlan
    ranked: list[FinalScoredCandidate]
    recommended: list[FinalScoredCandidate]
    manual_review: list[FinalScoredCandidate]
    rejected: list[FinalScoredCandidate]
    offer_targets: list[FinalScoredCandidate]


def _load_profile(path: Path) -> LLMIdealBloggerProfile:
    if not path.is_file():
        raise FinalPipelineError(f"IdealBloggerProfile not found: {path}")
    try:
        return LLMIdealBloggerProfile.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise FinalPipelineError(f"IdealBloggerProfile is invalid: {path}") from exc


def load_reference_identities(path: Path) -> tuple[set[str], set[str]]:
    """Load every source identity, including failed enrichment records."""

    if not path.is_file():
        raise FinalPipelineError(f"Enriched source base not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise FinalPipelineError("Enriched source base must be a JSON list")
        bloggers = [EnrichedSourceBlogger.model_validate(item) for item in payload]
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise FinalPipelineError(f"Cannot validate enriched source base: {path}") from exc
    urls = {str(item.profile.profile_url).rstrip("/").casefold() for item in bloggers}
    usernames = {item.profile.username.casefold() for item in bloggers}
    return urls, usernames


def _safe_public_text(value: str | None, limit: int) -> str | None:
    if not value:
        return None
    cleaned = EMAIL_PATTERN.sub("[redacted-email]", value)
    cleaned = PHONE_PATTERN.sub("[redacted-phone]", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def _strict_profile_url(raw_url: str) -> tuple[str | None, str]:
    """Normalize only profile/channel URLs and reject post/video/article pages."""

    candidate = raw_url.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = urlsplit(candidate)
    host = parsed.netloc.casefold().split(":", maxsplit=1)[0]
    parts = [part for part in parsed.path.split("/") if part]
    if host in {"instagram.com", "www.instagram.com"}:
        if len(parts) != 1:
            return None, "instagram_post_or_navigation_page"
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if not parts:
            return None, "youtube_root_or_article_page"
        first = parts[0].casefold()
        is_handle = first.startswith("@") and len(parts) == 1
        is_channel = first in {"channel", "user", "c"} and len(parts) == 2
        if not (is_handle or is_channel):
            return None, "youtube_video_short_or_non_channel_page"
    elif host == "youtu.be":
        return None, "youtube_short_link_is_a_video_not_a_profile"
    elif host in {"t.me", "telegram.me", "www.telegram.me"}:
        if parts and parts[0].casefold() == "s":
            if len(parts) != 2:
                return None, "telegram_post_or_navigation_page"
        elif len(parts) != 1:
            return None, "telegram_post_or_navigation_page"
    else:
        return None, "unsupported_domain_or_article"
    normalized = normalize_profile_url(raw_url)
    return (normalized, "profile_url") if normalized else (None, "invalid_profile_url")


def _username_from_url(profile_url: str, platform: Platform) -> str:
    parts = [part for part in urlsplit(profile_url).path.split("/") if part]
    if platform == Platform.YOUTUBE_SHORTS and len(parts) == 2:
        return parts[1].lstrip("@")
    return (parts[-1] if parts else "unknown").lstrip("@")


def _display_name(title: str | None, username: str) -> str:
    safe_title = _safe_public_text(title, 300)
    if not safe_title:
        return username
    cleaned = re.split(r"[|•]", safe_title, maxsplit=1)[0].strip()
    cleaned = re.sub(
        r"\s*[-—]\s*(Instagram|YouTube|Telegram).*$",
        "",
        cleaned,
        flags=re.I,
    ).strip()
    return cleaned or username


def _is_store_or_brand(title: str | None, snippet: str | None, profile_url: str) -> bool:
    title_text = (title or "").casefold()
    snippet_text = (snippet or "").casefold()
    username = _username_from_url(
        profile_url,
        platform_from_url(profile_url) or Platform.OTHER,
    ).casefold()
    if any(marker in title_text for marker in STORE_TITLE_MARKERS) and not any(
        marker in title_text for marker in AUTHOR_MARKERS
    ):
        return True
    if any(marker in username for marker in STORE_HANDLE_MARKERS):
        return True
    combined = f"{title_text} {snippet_text}"
    commercial_context = any(marker in combined for marker in STORE_TITLE_MARKERS)
    author_context = any(marker in combined for marker in AUTHOR_MARKERS)
    return commercial_context and not author_context


def _parse_public_count(text: str) -> int | None:
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(тыс\.?|k|млн\.?|m)?\s*"
        r"(?:подписчик(?:ов|а)?|followers)",
        text,
        re.I,
    )
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    suffix = (match.group(2) or "").casefold()
    if suffix.startswith("тыс") or suffix == "k":
        value *= 1_000
    elif suffix.startswith("млн") or suffix == "m":
        value *= 1_000_000
    return round(value)


def _parse_public_engagement(text: str) -> float | None:
    match = re.search(
        r"(?:\bER\b|вовлеч[её]нность)[^0-9]{0,20}(\d+(?:[.,]\d+)?)\s*%",
        text,
        re.I,
    )
    return float(match.group(1).replace(",", ".")) if match else None


def _search_confidence(hit: SearchHit, text: str, platform: Platform) -> float:
    confidence = 0.10
    confidence += 0.12 if hit.title else 0.0
    confidence += 0.18 if hit.snippet else 0.0
    confidence += 0.22 if any(marker in text for marker in FASHION_MARKERS) else 0.0
    confidence += 0.10 if any(marker in text for marker in AUTHOR_MARKERS) else 0.0
    confidence += 0.08 if platform in {
        Platform.INSTAGRAM,
        Platform.YOUTUBE_SHORTS,
        Platform.TELEGRAM,
    } else 0.0
    confidence += 0.08 if hit.provider_score is not None else 0.0
    confidence += 0.05 if (
        _parse_public_count(text) is not None
        or _parse_public_engagement(text) is not None
    ) else 0.0
    return round(min(0.78, confidence), 3)


def _content_formats(text: str, platform: Platform) -> list[str]:
    formats: list[str] = []
    if any(marker in text for marker in ("reel", "рилс", "video", "видео")):
        formats.append("reels/video")
    if any(marker in text for marker in ("carousel", "sidecar", "карусел")):
        formats.append("carousel")
    if platform == Platform.YOUTUBE_SHORTS:
        formats.append("youtube_shorts")
    elif platform == Platform.TELEGRAM:
        formats.append("telegram_publication")
    elif platform == Platform.INSTAGRAM and not formats:
        formats.append("instagram_publication")
    return list(dict.fromkeys(formats))


def _candidate_from_hit(
    hit: SearchHit,
    normalized_url: str,
    platform: Platform,
) -> RealCandidateProfile:
    title = _safe_public_text(hit.title, 1_000)
    snippet = _safe_public_text(hit.snippet, 4_000)
    text = f"{title or ''} {snippet or ''}".casefold().replace("ё", "е")
    username = _username_from_url(normalized_url, platform)
    matched = [marker for marker in FASHION_MARKERS if marker in text]
    followers_count = _parse_public_count(text)
    engagement_rate = _parse_public_engagement(text)
    evidence = []
    if title:
        evidence.append(f"Tavily title: {title}")
    if snippet:
        evidence.append(f"Tavily snippet: {snippet[:500]}")
    evidence.append(f"Source query: {hit.source_query}")
    if matched:
        evidence.append(f"Matched public text signals: {', '.join(matched[:8])}")
    if followers_count is not None:
        evidence.append(f"Tavily public text followers_count: {followers_count}")
    if engagement_rate is not None:
        evidence.append(f"Tavily public text engagement_rate: {engagement_rate}%")
    return RealCandidateProfile(
        name=_display_name(title, username),
        username=username,
        platform=platform,
        profile_url=normalized_url,
        title=title,
        snippet=snippet,
        source_query=hit.source_query,
        tavily_score=hit.provider_score,
        followers_count=followers_count,
        engagement_rate=engagement_rate,
        content_formats=_content_formats(text, platform),
        evidence=evidence,
        data_confidence=_search_confidence(hit, text, platform),
        enrichment_status="tavily_only",
    )


def clean_real_search_hits(
    hits: list[SearchHit],
    *,
    reference_urls: set[str],
    reference_usernames: set[str],
    maximum_candidates: int,
) -> CleaningResult:
    """Filter profiles without substituting mock data or missing metrics."""

    seen: set[str] = set()
    accepted: list[tuple[RealCandidateProfile, int, tuple[float, float, str]]] = []
    audit_rows: list[RealCandidateAuditRow] = []
    for hit in hits:
        normalized, url_reason = _strict_profile_url(hit.url)
        if normalized is None:
            audit_rows.append(
                RealCandidateAuditRow(
                    raw_url=hit.url,
                    source_query=hit.source_query,
                    title=hit.title,
                    tavily_score=hit.provider_score,
                    decision="rejected",
                    reason=url_reason,
                )
            )
            continue
        platform = platform_from_url(normalized)
        if platform is None:
            audit_rows.append(
                RealCandidateAuditRow(
                    raw_url=hit.url,
                    normalized_url=normalized,
                    source_query=hit.source_query,
                    title=hit.title,
                    tavily_score=hit.provider_score,
                    decision="rejected",
                    reason="unsupported_platform",
                )
            )
            continue
        canonical = normalized.rstrip("/").casefold()
        username = _username_from_url(normalized, platform).casefold()
        if canonical in seen:
            audit_rows.append(
                RealCandidateAuditRow(
                    raw_url=hit.url,
                    normalized_url=normalized,
                    platform=platform,
                    source_query=hit.source_query,
                    title=hit.title,
                    tavily_score=hit.provider_score,
                    decision="rejected",
                    reason="duplicate_profile",
                )
            )
            continue
        seen.add(canonical)
        if canonical in reference_urls or username in reference_usernames:
            audit_rows.append(
                RealCandidateAuditRow(
                    raw_url=hit.url,
                    normalized_url=normalized,
                    platform=platform,
                    source_query=hit.source_query,
                    title=hit.title,
                    tavily_score=hit.provider_score,
                    decision="rejected",
                    reason="source_reference_profile",
                )
            )
            continue
        if _is_store_or_brand(hit.title, hit.snippet, normalized):
            audit_rows.append(
                RealCandidateAuditRow(
                    raw_url=hit.url,
                    normalized_url=normalized,
                    platform=platform,
                    source_query=hit.source_query,
                    title=hit.title,
                    tavily_score=hit.provider_score,
                    decision="rejected",
                    reason="store_brand_or_catalog",
                )
            )
            continue
        text = f"{hit.title or ''} {hit.snippet or ''}".casefold().replace("ё", "е")
        if not any(marker in text for marker in FASHION_MARKERS):
            audit_rows.append(
                RealCandidateAuditRow(
                    raw_url=hit.url,
                    normalized_url=normalized,
                    platform=platform,
                    source_query=hit.source_query,
                    title=hit.title,
                    tavily_score=hit.provider_score,
                    decision="rejected",
                    reason="insufficient_fashion_evidence",
                )
            )
            continue
        candidate = _candidate_from_hit(hit, normalized, platform)
        audit_index = len(audit_rows)
        audit_rows.append(
            RealCandidateAuditRow(
                raw_url=hit.url,
                normalized_url=normalized,
                platform=platform,
                source_query=hit.source_query,
                title=hit.title,
                tavily_score=hit.provider_score,
                decision="accepted_pre_enrichment",
                reason="unique_public_author_profile_with_fashion_signals",
                data_confidence=candidate.data_confidence,
            )
        )
        accepted.append(
            (
                candidate,
                audit_index,
                (
                    hit.provider_score or 0.0,
                    candidate.data_confidence,
                    str(candidate.profile_url),
                ),
            )
        )

    accepted.sort(key=lambda item: (-item[2][0], -item[2][1], item[2][2]))
    selected = accepted[:maximum_candidates]
    for _, audit_index, _ in accepted[maximum_candidates:]:
        audit_rows[audit_index] = audit_rows[audit_index].model_copy(
            update={
                "decision": "rejected",
                "reason": "max_candidates_before_enrichment_limit",
            }
        )
    return CleaningResult(
        candidates=[candidate for candidate, _, _ in selected],
        audit_rows=audit_rows,
    )


def _author_evidence_confidence(
    items: list[tuple[int, SearchHit, ContentAuthorResolution]],
) -> float:
    """Increase identity confidence only for independent author evidence."""

    base = max(result.confidence for _, _, result in items)
    independent_queries = len({hit.source_query.casefold() for _, hit, _ in items})
    return round(min(1.0, base + min(0.05, 0.02 * (independent_queries - 1))), 3)


def clean_resolved_search_hits(
    hits: list[SearchHit],
    resolutions: list[ContentAuthorResolution],
    *,
    reference_urls: set[str],
    reference_usernames: set[str],
    maximum_candidates: int,
    minimum_resolution_confidence: float,
) -> CleaningResult:
    """Aggregate resolved authors, then apply the existing candidate filters."""

    if len(hits) != len(resolutions):
        raise FinalPipelineError("Every Tavily hit must have one author-resolution result")
    audit_slots: list[RealCandidateAuditRow | None] = [None] * len(hits)
    groups: dict[
        str,
        list[tuple[int, SearchHit, ContentAuthorResolution]],
    ] = {}
    for index, (hit, resolution) in enumerate(zip(hits, resolutions, strict=True)):
        profile_url = (
            str(resolution.resolved_profile_url)
            if resolution.resolved_profile_url is not None
            else None
        )
        if (
            resolution.status not in {"profile_url", "resolved_author"}
            or profile_url is None
        ):
            audit_slots[index] = RealCandidateAuditRow(
                raw_url=hit.url,
                normalized_url=profile_url,
                platform=resolution.platform,
                source_query=hit.source_query,
                title=hit.title,
                tavily_score=hit.provider_score,
                decision="rejected",
                reason=f"{resolution.status}: {resolution.reason}"[:2_000],
                author_resolution_confidence=resolution.confidence,
            )
            continue
        if resolution.confidence < minimum_resolution_confidence:
            audit_slots[index] = RealCandidateAuditRow(
                raw_url=hit.url,
                normalized_url=profile_url,
                platform=resolution.platform,
                source_query=hit.source_query,
                title=hit.title,
                tavily_score=hit.provider_score,
                decision="rejected",
                reason=(
                    f"author_resolution_confidence={resolution.confidence:.2f} below "
                    f"MIN_AUTHOR_RESOLUTION_CONFIDENCE={minimum_resolution_confidence:.2f}"
                ),
                author_resolution_confidence=resolution.confidence,
            )
            continue
        groups.setdefault(profile_url.rstrip("/").casefold(), []).append(
            (index, hit, resolution)
        )

    accepted: list[tuple[RealCandidateProfile, list[int], tuple[float, float, str]]] = []
    for canonical, items in groups.items():
        _, best_hit, best_resolution = max(
            items,
            key=lambda item: (item[1].provider_score or 0.0, -item[0]),
        )
        profile_url = str(best_resolution.resolved_profile_url)
        platform = best_resolution.platform or platform_from_url(profile_url)
        if platform is None:
            for index, hit, resolution in items:
                audit_slots[index] = RealCandidateAuditRow(
                    raw_url=hit.url,
                    normalized_url=profile_url,
                    source_query=hit.source_query,
                    title=hit.title,
                    tavily_score=hit.provider_score,
                    decision="rejected",
                    reason="unsupported_platform_after_author_resolution",
                    author_resolution_confidence=resolution.confidence,
                )
            continue
        username = _username_from_url(profile_url, platform).casefold()
        source_queries = list(dict.fromkeys(hit.source_query for _, hit, _ in items))
        snippets = list(
            dict.fromkeys(
                safe
                for _, hit, _ in items
                if (safe := _safe_public_text(hit.snippet, 1_500))
            )
        )
        titles = list(
            dict.fromkeys(
                safe
                for _, hit, _ in items
                if (safe := _safe_public_text(hit.title, 500))
            )
        )
        content_urls = list(
            dict.fromkeys(
                hit.url
                for _, hit, resolution in items
                if resolution.resolution_method != "direct_profile_url"
            )
        )
        evidence_urls = content_urls or [hit.url for _, hit, _ in items]
        evidence_count = len(items)
        resolution_confidence = _author_evidence_confidence(items)
        aggregate_hit = SearchHit(
            url=profile_url,
            title=titles[0] if titles else best_hit.title,
            snippet=" | ".join(snippets)[:4_000] or None,
            source_query=" | ".join(source_queries)[:1_000],
            provider_score=max(
                (hit.provider_score for _, hit, _ in items if hit.provider_score is not None),
                default=None,
            ),
        )
        candidate = _candidate_from_hit(aggregate_hit, profile_url, platform)
        resolved_only_from_content = all(
            resolution.resolution_method != "direct_profile_url"
            for _, _, resolution in items
        )
        identity_evidence = list(candidate.evidence)
        identity_evidence.append(
            f"Author identity resolved from {evidence_count} Tavily result(s); "
            f"confidence={resolution_confidence:.2f}"
        )
        identity_evidence.extend(f"Content evidence URL: {url}" for url in content_urls[:8])
        candidate = candidate.model_copy(
            update={
                "evidence": identity_evidence[:20],
                "evidence_count": evidence_count,
                "evidence_urls": evidence_urls[:20],
                "author_resolution_confidence": resolution_confidence,
                "data_confidence": min(candidate.data_confidence, resolution_confidence),
                # A Reel/Short URL proves identity evidence, not a preferred format.
                # Instagram enrichment may add verified recent-post formats later.
                "content_formats": (
                    [] if resolved_only_from_content else candidate.content_formats
                ),
            }
        )

        rejection_reason: str | None = None
        if canonical in reference_urls or username in reference_usernames:
            rejection_reason = "source_reference_profile_after_resolution"
        elif _is_store_or_brand(aggregate_hit.title, aggregate_hit.snippet, profile_url):
            rejection_reason = "store_brand_or_catalog"
        else:
            text = (
                f"{aggregate_hit.title or ''} {aggregate_hit.snippet or ''}"
                .casefold()
                .replace("ё", "е")
            )
            if not any(marker in text for marker in FASHION_MARKERS):
                rejection_reason = "insufficient_fashion_evidence"
        group_indices = [index for index, _, _ in items]
        for item_number, (index, hit, resolution) in enumerate(items):
            audit_slots[index] = RealCandidateAuditRow(
                raw_url=hit.url,
                normalized_url=profile_url,
                platform=platform,
                source_query=hit.source_query,
                title=hit.title,
                tavily_score=hit.provider_score,
                decision=(
                    "rejected"
                    if rejection_reason
                    else (
                        "accepted_pre_enrichment"
                        if item_number == 0
                        else "merged_author_evidence"
                    )
                ),
                reason=(
                    rejection_reason
                    or (
                        "unique_resolved_author_with_fashion_signals"
                        if item_number == 0
                        else "merged_with_same_resolved_author"
                    )
                ),
                data_confidence=candidate.data_confidence,
                evidence_count=evidence_count,
                evidence_urls=evidence_urls[:20],
                author_resolution_confidence=resolution_confidence,
            )
        if rejection_reason is None:
            accepted.append(
                (
                    RealCandidateProfile.model_validate(candidate.model_dump(mode="json")),
                    group_indices,
                    (
                        aggregate_hit.provider_score or 0.0,
                        candidate.data_confidence,
                        profile_url,
                    ),
                )
            )

    accepted.sort(key=lambda item: (-item[2][0], -item[2][1], item[2][2]))
    selected = accepted[:maximum_candidates]
    for _, indices, _ in accepted[maximum_candidates:]:
        for index in indices:
            row = audit_slots[index]
            if row is not None:
                audit_slots[index] = row.model_copy(
                    update={
                        "decision": "rejected",
                        "reason": "max_candidates_before_enrichment_limit",
                    }
                )
    if any(row is None for row in audit_slots):
        raise FinalPipelineError("Author-resolution audit is incomplete")
    return CleaningResult(
        candidates=[candidate for candidate, _, _ in selected],
        audit_rows=[row for row in audit_slots if row is not None],
    )


def save_raw_hits(hits: list[SearchHit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "title": hit.title,
            "url": hit.url,
            "content": hit.snippet,
            "source_query": hit.source_query,
            "score": hit.provider_score,
        }
        for hit in hits
    ]
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(path, index=False, encoding="utf-8")


def save_real_audit(rows: list[RealCandidateAuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [row.model_dump(mode="json") for row in rows],
        columns=AUDIT_COLUMNS,
    ).to_csv(path, index=False, encoding="utf-8")


def save_real_audit_v2(rows: list[RealCandidateAuditRow], path: Path) -> None:
    """Save expanded evidence fields without changing the v1 audit schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        record = row.model_dump(mode="json")
        record["evidence_urls"] = " | ".join(row.evidence_urls)
        records.append(record)
    pd.DataFrame(records, columns=AUDIT_V2_COLUMNS).to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


def save_content_author_resolution_audit(
    hits: list[SearchHit],
    resolutions: list[ContentAuthorResolution],
    path: Path,
) -> None:
    """Persist one token-free resolution row for every content URL."""

    rows = []
    for hit, result in zip(hits, resolutions, strict=True):
        kind, _, _ = classify_public_url(hit.url)
        if kind != "content":
            continue
        rows.append(
            {
                "content_url": hit.url,
                "platform": result.platform.value if result.platform else None,
                "resolved_profile_url": (
                    str(result.resolved_profile_url)
                    if result.resolved_profile_url is not None
                    else None
                ),
                "resolved_author": result.resolved_username_or_channel,
                "resolution_method": result.resolution_method,
                "confidence": result.confidence,
                "status": result.status,
                "reason": result.reason,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESOLUTION_AUDIT_COLUMNS).to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


def _replace_audit(
    rows: list[RealCandidateAuditRow],
    profile_url: str,
    **updates: Any,
) -> list[RealCandidateAuditRow]:
    canonical = profile_url.rstrip("/").casefold()
    replaced: list[RealCandidateAuditRow] = []
    updated_once = False
    for row in rows:
        row_url = (row.normalized_url or "").rstrip("/").casefold()
        if not updated_once and row_url == canonical and row.decision != "rejected":
            replaced.append(row.model_copy(update=updates))
            updated_once = True
        else:
            replaced.append(row)
    return replaced


def enrich_instagram_candidates(
    candidates: list[RealCandidateProfile],
    audit_rows: list[RealCandidateAuditRow],
    *,
    provider: ProfileEnrichmentProvider | None,
    settings: Settings,
) -> tuple[list[RealCandidateProfile], list[RealCandidateAuditRow], int]:
    """Send at most the configured Instagram subset to Apify and keep failures."""

    instagram = [c for c in candidates if c.platform == Platform.INSTAGRAM]
    selected_instagram = instagram[: settings.max_candidates_for_apify]
    selected_urls = {str(c.profile_url).rstrip("/").casefold() for c in selected_instagram}
    updated_audit = list(audit_rows)
    for candidate in instagram[settings.max_candidates_for_apify :]:
        updated_audit = _replace_audit(
            updated_audit,
            str(candidate.profile_url),
            decision="rejected",
            reason="max_candidates_for_apify_limit",
        )
    non_instagram = [c for c in candidates if c.platform != Platform.INSTAGRAM]
    if not selected_instagram:
        return non_instagram, updated_audit, 0
    if provider is None:
        raise FinalPipelineError("Apify provider was not created for Instagram candidates")

    run = enrich_profile_urls(
        input_urls=[str(candidate.profile_url) for candidate in selected_instagram],
        provider=provider,
        posts_limit=min(3, settings.profile_posts_limit),
        cache_dir=settings.real_candidate_cache_dir,
        cache_enabled=True,
        refresh_profiles=False,
        limit_profiles=len(selected_instagram),
        concurrency=settings.profile_enrichment_concurrency,
        delay_seconds=settings.profile_enrichment_delay_seconds,
        replace_failed_with_next=False,
    )
    enriched_by_username = {
        blogger.profile.username.casefold(): blogger for blogger in run.bloggers
    }
    merged: list[RealCandidateProfile] = list(non_instagram)
    for candidate in selected_instagram:
        blogger = enriched_by_username.get(candidate.username.casefold())
        if blogger is None:
            merged_candidate = candidate.model_copy(
                update={
                    "data_confidence": min(candidate.data_confidence, 0.25),
                    "enrichment_status": "failed",
                    "enrichment_error": "Apify returned no aligned profile record",
                }
            )
        else:
            safe_input = prepare_llm_profile(blogger, max_posts=3)
            status = blogger.enrichment_status.value
            confidence = round(
                min(1.0, 0.25 * candidate.data_confidence + 0.75 * blogger.data_confidence),
                3,
            )
            if status == "failed":
                confidence = min(confidence, 0.25)
            elif blogger.profile.is_private is True:
                status = "private"
                confidence = min(confidence, 0.25)
            elif status == "partial":
                confidence = min(confidence, 0.75)
            formats = list(candidate.content_formats)
            formats.extend(
                post.post_type
                for post in safe_input.recent_posts
                if post.post_type is not None
            )
            evidence = list(candidate.evidence)
            evidence.append(f"Apify enrichment status: {status}")
            if safe_input.followers_count is not None:
                evidence.append(f"Apify followers_count: {safe_input.followers_count}")
            if safe_input.calculated_engagement_rate is not None:
                evidence.append(
                    "Calculated engagement_rate: "
                    f"{safe_input.calculated_engagement_rate:.4f}%"
                )
            merged_candidate = candidate.model_copy(
                update={
                    "name": safe_input.full_name or candidate.name,
                    "full_name": safe_input.full_name,
                    "biography": safe_input.biography,
                    "followers_count": safe_input.followers_count,
                    "engagement_rate": safe_input.calculated_engagement_rate,
                    "is_private": safe_input.is_private,
                    "recent_posts": safe_input.recent_posts,
                    "content_formats": list(dict.fromkeys(formats))[:10],
                    "evidence": evidence[:20],
                    "data_confidence": confidence,
                    "enrichment_status": status,
                    "enrichment_error": (
                        blogger.enrichment_error[:1_000]
                        if blogger.enrichment_error
                        else None
                    ),
                }
            )
        merged.append(RealCandidateProfile.model_validate(merged_candidate.model_dump(mode="json")))
        updated_audit = _replace_audit(
            updated_audit,
            str(candidate.profile_url),
            decision=(
                "enrichment_failed"
                if merged_candidate.enrichment_status == "failed"
                else "enriched_or_partial"
            ),
            reason=(
                merged_candidate.enrichment_error
                or f"instagram_{merged_candidate.enrichment_status}; not automatically approved"
            )[:2_000],
            data_confidence=merged_candidate.data_confidence,
        )

    order = {str(candidate.profile_url): index for index, candidate in enumerate(candidates)}
    merged.sort(key=lambda item: order.get(str(item.profile_url), len(order)))
    return merged, updated_audit, len(selected_instagram)


def save_enriched_candidates(candidates: list[RealCandidateProfile], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [candidate.model_dump(mode="json") for candidate in candidates],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _detail_or_fallback(
    item: FinalScoredCandidate,
    key: str,
    score: int,
    maximum: int,
) -> tuple[int, int, str]:
    detail = item.criterion_details.get(key)
    if detail is None:
        return (
            score,
            maximum,
            "Недостаточно данных для уверенной оценки. confidence: low",
        )
    reason = detail.reason
    if "confidence:" not in reason.casefold():
        reason = f"{reason} confidence: {detail.confidence}"
    return detail.score, detail.max_score, reason


def final_score_breakdown_row(item: FinalScoredCandidate) -> FinalScoreBreakdownRow:
    """Flatten one explainable real score without recalculating any points."""

    score = item.score
    topic = _detail_or_fallback(item, "topic", score.fashion_relevance_score, 20)
    visual = _detail_or_fallback(item, "visual", score.visual_text_score, 15)
    audience = _detail_or_fallback(item, "audience", score.audience_score, 15)
    tone = _detail_or_fallback(item, "tone", score.tone_score, 10)
    engagement = _detail_or_fallback(item, "engagement", score.engagement_score, 10)
    ad_load = _detail_or_fallback(item, "ad_load", score.advertising_load_score, 10)
    price = _detail_or_fallback(item, "price_segment", score.price_segment_score, 5)
    content_format = _detail_or_fallback(item, "format", score.content_format_score, 5)
    safety = _detail_or_fallback(item, "brand_safety", score.brand_safety_score, 5)
    confidence = item.criterion_details.get("data_confidence")
    confidence_reason = (
        (
            confidence.reason
            if "confidence:" in confidence.reason.casefold()
            else f"{confidence.reason} confidence: {confidence.confidence}"
        )
        if confidence is not None
        else (
            f"data_confidence={item.candidate.data_confidence:.3f}; "
            f"вклад полноты данных {score.data_confidence_score}/5."
        )
    )
    return FinalScoreBreakdownRow(
        profile_url=item.candidate.profile_url,
        platform=item.candidate.platform,
        total_score=score.total_score,
        max_score=100,
        evidence_count=item.candidate.evidence_count,
        topic_score=topic[0],
        topic_reason=topic[2],
        visual_score=visual[0],
        visual_reason=visual[2],
        audience_score=audience[0],
        audience_reason=audience[2],
        tone_score=tone[0],
        tone_reason=tone[2],
        engagement_score=engagement[0],
        engagement_reason=engagement[2],
        ad_load_score=ad_load[0],
        ad_load_reason=ad_load[2],
        price_segment_score=price[0],
        price_segment_reason=price[2],
        format_score=content_format[0],
        format_reason=content_format[2],
        brand_safety_score=safety[0],
        brand_safety_reason=safety[2],
        data_confidence=item.candidate.data_confidence,
        confidence_reason=confidence_reason,
    )


def save_final_score_breakdown(
    ranked: list[FinalScoredCandidate],
    path: Path,
) -> None:
    """Persist explanations for every scored candidate, not only finalists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [final_score_breakdown_row(item) for item in ranked]
    pd.DataFrame(
        [row.model_dump(mode="json") for row in rows],
        columns=SCORE_BREAKDOWN_COLUMNS,
    ).to_csv(path, index=False, encoding="utf-8")


def _apply_score_audit(
    audit_rows: list[RealCandidateAuditRow],
    ranked: list[FinalScoredCandidate],
    finalists: list[FinalScoredCandidate],
    min_score: int,
) -> list[RealCandidateAuditRow]:
    finalist_urls = {str(item.candidate.profile_url) for item in finalists}
    updated = list(audit_rows)
    for item in ranked:
        candidate = item.candidate
        if str(candidate.profile_url) in finalist_urls:
            decision = "selected_needs_review"
            reason = "passed threshold and TOP_K; manual review required before contact"
        elif item.score.total_score < min_score:
            decision = "below_final_min_score"
            confidence_note = (
                " Low data_confidence reduced all criteria."
                if candidate.data_confidence < 0.6
                else ""
            )
            reason = (
                f"score {item.score.total_score} below FINAL_MIN_SCORE={min_score}."
                f"{confidence_note}"
            )
        else:
            decision = "above_threshold_not_selected"
            reason = "passed threshold but excluded by MAX_FINAL_CANDIDATES"
        updated = _replace_audit(
            updated,
            str(candidate.profile_url),
            decision=decision,
            reason=reason,
            data_confidence=candidate.data_confidence,
            total_score=item.score.total_score,
        )
    return updated


def save_final_outputs(
    finalists: list[FinalScoredCandidate],
    offers: list[FinalPersonalizedOffer],
    *,
    csv_path: Path,
    markdown_path: Path,
) -> None:
    offers_by_username = {offer.candidate_username.casefold(): offer for offer in offers}
    rows: list[FinalRealBloggerRow] = []
    for finalist in finalists:
        candidate = finalist.candidate
        offer = offers_by_username.get(candidate.username.casefold())
        if offer is None:
            raise FinalPipelineError(f"No offer record for finalist {candidate.username}")
        rows.append(
            FinalRealBloggerRow(
                name=candidate.name,
                username=candidate.username,
                platform=candidate.platform,
                profile_url=candidate.profile_url,
                total_score=finalist.score.total_score,
                data_confidence=candidate.data_confidence,
                followers_count=candidate.followers_count,
                engagement_rate=candidate.engagement_rate,
                match_reason=finalist.match_reason,
                evidence=" | ".join(finalist.evidence),
                personalized_offer=offer.message,
            )
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [row.model_dump(mode="json") for row in rows],
        columns=FINAL_COLUMNS,
    ).to_csv(csv_path, index=False, encoding="utf-8")

    lines = [
        "# Реальные кандидаты LD LATTE — ручная проверка",
        "",
        "> Ни один кандидат не одобрен автоматически. Офферы не отправлены.",
    ]
    if not rows:
        lines.extend(("", "Нет кандидатов, прошедших FINAL_MIN_SCORE."))
    for index, (row, finalist) in enumerate(zip(rows, finalists, strict=True), start=1):
        score_table: list[str] = []
        criterion_labels = (
            ("topic", "Тематика"),
            ("visual", "Визуальные/текстовые сигналы"),
            ("audience", "Аудитория"),
            ("tone", "Тон"),
            ("engagement", "Вовлечённость"),
            ("ad_load", "Рекламная нагрузка"),
            ("price_segment", "Ценовой сегмент"),
            ("format", "Форматы"),
            ("brand_safety", "Brand safety"),
            ("data_confidence", "Полнота данных"),
        )
        for key, label in criterion_labels:
            detail = finalist.criterion_details.get(key)
            if detail is None:
                continue
            safe_reason = detail.reason.replace("|", "\\|").replace("\n", " ")
            if "confidence:" not in safe_reason.casefold():
                safe_reason = f"{safe_reason} confidence: {detail.confidence}"
            score_table.append(
                f"| {label} | {detail.score}/{detail.max_score} | "
                f"{safe_reason} |"
            )
        lines.extend(
            (
                "",
                f"## {index}. {row.name} — {row.total_score}/100",
                "",
                f"- Площадка: {row.platform.value}",
                f"- Профиль: {row.profile_url}",
                f"- Data confidence: {row.data_confidence:.2f}",
                f"- Followers: {row.followers_count if row.followers_count is not None else 'не подтверждено'}",
                f"- ER: {row.engagement_rate if row.engagement_rate is not None else 'не подтверждено'}",
                f"- Статус: `{row.manual_review_status}`",
                f"- Причина: {row.match_reason}",
                "",
                "### Почему кандидат получил именно такой score",
                "",
                "| Критерий | Баллы | Причина |",
                "|---|---:|---|",
                *(score_table or [
                    "| Данные | — | Недостаточно данных для уверенной оценки. confidence: low |"
                ]),
                "",
                "### Evidence",
                "",
                *[f"- {item}" for item in finalist.evidence],
                "",
                "### Черновик оффера — не отправлен",
                "",
                row.personalized_offer,
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run_audit(audit: FinalRunAudit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(audit.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_pipeline_error(exc: Exception) -> str:
    if isinstance(
        exc,
        (
            FinalPipelineError,
            SearchProviderError,
            FinalOfferGenerationError,
            ContentAuthorResolutionError,
        ),
    ):
        return str(exc)[:2_000]
    return f"{type(exc).__name__}: final pipeline stage failed"


def _search_all_queries(
    provider: Any,
    queries: list[str],
    per_query: int,
) -> tuple[list[SearchHit], list[str]]:
    hits: list[SearchHit] = []
    errors: list[str] = []
    for index, query in enumerate(queries, start=1):
        try:
            query_hits = provider.search([query], per_query)
            hits.extend(query_hits[:per_query])
        except Exception as exc:
            safe = (
                str(exc)
                if isinstance(exc, SearchProviderError)
                else f"{type(exc).__name__}: Tavily query failed"
            )
            errors.append(f"query {index}: {safe}"[:2_000])
    return hits, errors


def build_final_plan(settings: Settings, profile: LLMIdealBloggerProfile) -> FinalRunPlan:
    queries = profile.search_queries[: settings.tavily_max_queries]
    missing = [
        name
        for name, value in (
            ("TAVILY_API_KEY", settings.tavily_api_key),
            ("APIFY_API_TOKEN", settings.apify_api_token),
            ("APIFY_ACTOR_ID", settings.apify_actor_id),
            ("OPENAI_API_KEY", settings.openai_api_key),
        )
        if not value
    ]
    return FinalRunPlan(
        query_count=len(queries),
        tavily_result_limit=len(queries) * settings.tavily_results_per_query,
        max_candidates_before_enrichment=settings.max_candidates_before_enrichment,
        max_apify_candidates=settings.max_candidates_for_apify,
        max_openai_offers=settings.max_final_candidates,
        final_min_score=settings.final_min_score,
        openai_model=settings.openai_model,
        missing_credentials=missing,
    )


def build_final_plan_v2(settings: Settings) -> FinalRunPlan:
    """Build v2 cost ceilings from fixed, interview-readable search queries."""

    missing = [
        name
        for name, value in (
            ("TAVILY_API_KEY", settings.tavily_api_key),
            ("APIFY_API_TOKEN", settings.apify_api_token),
            ("APIFY_ACTOR_ID", settings.apify_actor_id),
            ("OPENAI_API_KEY", settings.openai_api_key),
        )
        if not value
    ]
    query_count = min(8, len(FINAL_V2_SEARCH_QUERIES))
    return FinalRunPlan(
        query_count=query_count,
        tavily_result_limit=query_count * min(5, settings.tavily_results_per_query),
        max_candidates_before_enrichment=settings.max_candidates_before_enrichment,
        max_apify_candidates=min(8, settings.max_candidates_for_apify),
        max_openai_offers=settings.max_final_candidates,
        final_min_score=settings.final_min_score,
        openai_model=settings.openai_model,
        missing_credentials=missing,
        max_content_urls_for_resolution=settings.max_content_urls_for_resolution,
        max_apify_content_resolution_runs=1,
        min_author_resolution_confidence=settings.min_author_resolution_confidence,
        pipeline_version="v2",
    )


def _validate_credentials(
    settings: Settings,
    *,
    search_provider: Any | None,
    enrichment_provider: Any | None,
    offer_provider: Any | None,
) -> None:
    missing: list[str] = []
    if search_provider is None and not settings.tavily_api_key:
        missing.append("TAVILY_API_KEY")
    if enrichment_provider is None:
        if not settings.apify_api_token:
            missing.append("APIFY_API_TOKEN")
        if not settings.apify_actor_id:
            missing.append("APIFY_ACTOR_ID")
    if offer_provider is None and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise FinalPipelineError(
            "Missing credentials for real run: "
            f"{', '.join(missing)}. Use --dry-run-final before adding secrets."
        )


def _validate_credentials_v2(
    settings: Settings,
    *,
    search_provider: Any | None,
    content_resolver: ContentAuthorResolver | None,
    enrichment_provider: Any | None,
    offer_provider: Any | None,
) -> None:
    missing: list[str] = []
    if search_provider is None and not settings.tavily_api_key:
        missing.append("TAVILY_API_KEY")
    if content_resolver is None or enrichment_provider is None:
        if not settings.apify_api_token:
            missing.append("APIFY_API_TOKEN")
        if not settings.apify_actor_id:
            missing.append("APIFY_ACTOR_ID")
    if offer_provider is None and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    missing = list(dict.fromkeys(missing))
    if missing:
        raise FinalPipelineError(
            "Missing credentials for real v2 run: "
            f"{', '.join(missing)}. Use --dry-run-final-v2 before adding secrets."
        )


def run_final_pipeline(
    settings: Settings,
    *,
    dry_run: bool,
    search_provider: Any | None = None,
    enrichment_provider: ProfileEnrichmentProvider | None = None,
    offer_provider: Any | None = None,
) -> FinalPipelineResult:
    """Run the paid pipeline only after an explicit non-dry CLI command."""

    ideal = _load_profile(settings.ideal_blogger_profile_json_path)
    reference_urls, reference_usernames = load_reference_identities(
        settings.enriched_source_json_path
    )
    plan = build_final_plan(settings, ideal)
    if dry_run:
        return FinalPipelineResult(plan=plan, finalists=[], audit=None)

    _validate_credentials(
        settings,
        search_provider=search_provider,
        enrichment_provider=enrichment_provider,
        offer_provider=offer_provider,
    )
    started = time.monotonic()
    queries = ideal.search_queries[: settings.tavily_max_queries]
    errors: list[str] = []
    hits: list[SearchHit] = []
    cleaning = CleaningResult(candidates=[], audit_rows=[])
    enriched: list[RealCandidateProfile] = []
    ranked: list[FinalScoredCandidate] = []
    finalists: list[FinalScoredCandidate] = []
    apify_count = 0
    active_offer_provider = offer_provider

    def audit_snapshot(status: str) -> FinalRunAudit:
        return FinalRunAudit(
            status=status,
            dry_run=False,
            query_count=len(queries),
            tavily_result_limit=plan.tavily_result_limit,
            raw_result_count=len(hits),
            candidates_before_enrichment=len(cleaning.candidates),
            apify_candidate_count=apify_count,
            enriched_candidate_count=len(enriched),
            scored_candidate_count=len(ranked),
            finalist_count=len(finalists),
            openai_offer_calls=(
                getattr(active_offer_provider, "call_count", 0)
                if active_offer_provider is not None
                else 0
            ),
            final_min_score=settings.final_min_score,
            limits={
                "TAVILY_MAX_QUERIES": settings.tavily_max_queries,
                "TAVILY_RESULTS_PER_QUERY": settings.tavily_results_per_query,
                "MAX_CANDIDATES_BEFORE_ENRICHMENT": settings.max_candidates_before_enrichment,
                "MAX_CANDIDATES_FOR_APIFY": settings.max_candidates_for_apify,
                "MAX_FINAL_CANDIDATES": settings.max_final_candidates,
            },
            openai_model=settings.openai_model,
            openai_usage=(
                dict(getattr(active_offer_provider, "usage", {}))
                if active_offer_provider is not None
                else {}
            ),
            duration_seconds=round(time.monotonic() - started, 3),
            errors=errors,
            finished_at=datetime.now(UTC).isoformat(),
        )

    _write_run_audit(audit_snapshot("started"), settings.final_run_audit_path)
    try:
        active_search = search_provider or TavilySearchProvider(
            api_key=settings.tavily_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
        hits, search_errors = _search_all_queries(
            active_search,
            queries,
            settings.tavily_results_per_query,
        )
        errors.extend(search_errors)
        save_raw_hits(hits, settings.real_candidates_raw_path)
        if not hits and search_errors:
            raise FinalPipelineError("All Tavily queries failed; no candidates were produced")

        cleaning = clean_real_search_hits(
            hits,
            reference_urls=reference_urls,
            reference_usernames=reference_usernames,
            maximum_candidates=settings.max_candidates_before_enrichment,
        )
        save_real_audit(cleaning.audit_rows, settings.real_candidates_audit_path)
        _write_run_audit(audit_snapshot("search_cleaned"), settings.final_run_audit_path)

        has_instagram = any(
            candidate.platform == Platform.INSTAGRAM
            for candidate in cleaning.candidates
        )
        active_enrichment = enrichment_provider
        if has_instagram and active_enrichment is None:
            active_enrichment = create_profile_enrichment_provider(
                provider_name="apify",
                mock_fixture_path=settings.profile_enrichment_mock_path,
                apify_api_token=settings.apify_api_token,
                apify_actor_id=settings.apify_actor_id,
                timeout_seconds=max(settings.request_timeout_seconds, 300),
                apify_raw_response_path=settings.final_apify_raw_response_path,
            )
        enriched, updated_audit, apify_count = enrich_instagram_candidates(
            cleaning.candidates,
            cleaning.audit_rows,
            provider=active_enrichment,
            settings=settings,
        )
        cleaning = CleaningResult(cleaning.candidates, updated_audit)
        save_enriched_candidates(enriched, settings.real_candidates_enriched_path)

        ranked = rank_real_candidates(enriched, ideal)
        save_final_score_breakdown(ranked, settings.final_score_breakdown_path)
        finalists = select_finalists(
            ranked,
            min_score=settings.final_min_score,
            maximum=settings.max_final_candidates,
        )
        scored_audit = _apply_score_audit(
            cleaning.audit_rows,
            ranked,
            finalists,
            settings.final_min_score,
        )
        save_real_audit(scored_audit, settings.real_candidates_audit_path)
        _write_run_audit(audit_snapshot("scored"), settings.final_run_audit_path)

        offers: list[FinalPersonalizedOffer] = []
        if finalists:
            if active_offer_provider is None:
                try:
                    prompt = settings.final_offer_prompt_path.read_text(
                        encoding="utf-8"
                    )
                except OSError as exc:
                    raise FinalPipelineError(
                        f"Cannot read final offer prompt: {settings.final_offer_prompt_path}"
                    ) from exc
                active_offer_provider = OpenAIFinalOfferProvider(
                    api_key=settings.openai_api_key,
                    model=settings.openai_model,
                    timeout_seconds=settings.openai_request_timeout_seconds,
                    prompt=prompt,
                )
            offer_result = generate_final_offers(finalists, active_offer_provider)
            offers = offer_result.offers
            errors.extend(offer_result.errors)
        save_final_outputs(
            finalists,
            offers,
            csv_path=settings.final_real_bloggers_csv_path,
            markdown_path=settings.final_real_bloggers_markdown_path,
        )
        final_status = "completed_with_errors" if errors else "completed"
        final_audit = audit_snapshot(final_status)
        _write_run_audit(final_audit, settings.final_run_audit_path)
        return FinalPipelineResult(plan=plan, finalists=finalists, audit=final_audit)
    except Exception as exc:
        errors.append(_safe_pipeline_error(exc))
        failed_audit = audit_snapshot("failed")
        _write_run_audit(failed_audit, settings.final_run_audit_path)
        raise


def run_final_pipeline_v2(
    settings: Settings,
    *,
    dry_run: bool,
    search_provider: Any | None = None,
    content_resolver: ContentAuthorResolver | None = None,
    enrichment_provider: ProfileEnrichmentProvider | None = None,
    offer_provider: Any | None = None,
) -> FinalPipelineResult:
    """Run content-author resolution before v2 filtering and scoring."""

    ideal = _load_profile(settings.ideal_blogger_profile_json_path)
    reference_urls, reference_usernames = load_reference_identities(
        settings.enriched_source_json_path
    )
    plan = build_final_plan_v2(settings)
    if dry_run:
        return FinalPipelineResult(plan=plan, finalists=[], audit=None)

    _validate_credentials_v2(
        settings,
        search_provider=search_provider,
        content_resolver=content_resolver,
        enrichment_provider=enrichment_provider,
        offer_provider=offer_provider,
    )
    started = time.monotonic()
    queries = list(FINAL_V2_SEARCH_QUERIES[: plan.query_count])
    errors: list[str] = []
    hits: list[SearchHit] = []
    resolutions: list[ContentAuthorResolution] = []
    cleaning = CleaningResult(candidates=[], audit_rows=[])
    enriched: list[RealCandidateProfile] = []
    ranked: list[FinalScoredCandidate] = []
    finalists: list[FinalScoredCandidate] = []
    apify_count = 0
    active_resolver = content_resolver
    active_offer_provider = offer_provider

    def audit_snapshot(status: str) -> FinalRunAudit:
        content_pairs = [
            (hit, result)
            for hit, result in zip(hits, resolutions)
            if classify_public_url(hit.url)[0] == "content"
        ]
        resolved_profiles = {
            str(result.resolved_profile_url).rstrip("/").casefold()
            for _, result in content_pairs
            if result.status == "resolved_author"
            and result.resolved_profile_url is not None
        }
        resolution_runs = getattr(
            getattr(active_resolver, "instagram_provider", None),
            "run_count",
            0,
        )
        return FinalRunAudit(
            status=status,
            dry_run=False,
            query_count=len(queries),
            tavily_result_limit=plan.tavily_result_limit,
            raw_result_count=len(hits),
            candidates_before_enrichment=len(cleaning.candidates),
            apify_candidate_count=apify_count,
            enriched_candidate_count=len(enriched),
            scored_candidate_count=len(ranked),
            finalist_count=len(finalists),
            openai_offer_calls=(
                getattr(active_offer_provider, "call_count", 0)
                if active_offer_provider is not None
                else 0
            ),
            final_min_score=settings.final_min_score,
            limits={
                "TAVILY_MAX_QUERIES": 8,
                "TAVILY_RESULTS_PER_QUERY": min(5, settings.tavily_results_per_query),
                "MAX_CONTENT_URLS_FOR_RESOLUTION": settings.max_content_urls_for_resolution,
                "MAX_CANDIDATES_BEFORE_ENRICHMENT": settings.max_candidates_before_enrichment,
                "MAX_CANDIDATES_FOR_APIFY": min(8, settings.max_candidates_for_apify),
                "MAX_FINAL_CANDIDATES": settings.max_final_candidates,
            },
            openai_model=settings.openai_model,
            openai_usage=(
                dict(getattr(active_offer_provider, "usage", {}))
                if active_offer_provider is not None
                else {}
            ),
            apify_content_resolution_runs=resolution_runs,
            content_urls_found=len(content_pairs),
            content_resolution_attempted=sum(
                result.status != "skipped_resolution_limit"
                for _, result in content_pairs
            ),
            resolved_author_count=len(resolved_profiles),
            unresolved_author_count=sum(
                result.status != "resolved_author" for _, result in content_pairs
            ),
            duration_seconds=round(time.monotonic() - started, 3),
            errors=errors,
            finished_at=datetime.now(UTC).isoformat(),
        )

    _write_run_audit(audit_snapshot("started"), settings.final_run_audit_v2_path)
    try:
        active_search = search_provider or TavilySearchProvider(
            api_key=settings.tavily_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
        hits, search_errors = _search_all_queries(
            active_search,
            queries,
            min(5, settings.tavily_results_per_query),
        )
        errors.extend(search_errors)
        save_raw_hits(hits, settings.real_candidates_raw_v2_path)
        if not hits and search_errors:
            raise FinalPipelineError("All Tavily v2 queries failed; no candidates were produced")

        if active_resolver is None:
            active_resolver = create_content_author_resolver(
                apify_api_token=settings.apify_api_token,
                apify_actor_id=settings.apify_actor_id,
                youtube_api_key=settings.youtube_api_key,
                timeout_seconds=settings.request_timeout_seconds,
                maximum_content_urls=settings.max_content_urls_for_resolution,
                minimum_confidence=settings.min_author_resolution_confidence,
                cache_path=settings.content_author_cache_path,
                apify_raw_response_path=settings.content_author_apify_raw_path,
            )
        resolutions = active_resolver.resolve_many(hits)
        save_content_author_resolution_audit(
            hits,
            resolutions,
            settings.content_author_resolution_audit_path,
        )
        cleaning = clean_resolved_search_hits(
            hits,
            resolutions,
            reference_urls=reference_urls,
            reference_usernames=reference_usernames,
            maximum_candidates=settings.max_candidates_before_enrichment,
            minimum_resolution_confidence=settings.min_author_resolution_confidence,
        )
        save_real_audit_v2(cleaning.audit_rows, settings.real_candidates_audit_v2_path)
        _write_run_audit(
            audit_snapshot("authors_resolved_and_search_cleaned"),
            settings.final_run_audit_v2_path,
        )

        has_instagram = any(
            candidate.platform == Platform.INSTAGRAM
            for candidate in cleaning.candidates
        )
        active_enrichment = enrichment_provider
        if has_instagram and active_enrichment is None:
            active_enrichment = create_profile_enrichment_provider(
                provider_name="apify",
                mock_fixture_path=settings.profile_enrichment_mock_path,
                apify_api_token=settings.apify_api_token,
                apify_actor_id=settings.apify_actor_id,
                timeout_seconds=max(settings.request_timeout_seconds, 300),
                apify_raw_response_path=settings.final_apify_raw_response_v2_path,
            )
        enriched, updated_audit, apify_count = enrich_instagram_candidates(
            cleaning.candidates,
            cleaning.audit_rows,
            provider=active_enrichment,
            settings=settings,
        )
        cleaning = CleaningResult(cleaning.candidates, updated_audit)
        save_enriched_candidates(enriched, settings.real_candidates_enriched_v2_path)

        ranked = rank_real_candidates(enriched, ideal)
        save_final_score_breakdown(ranked, settings.final_score_breakdown_path)
        finalists = select_finalists(
            ranked,
            min_score=settings.final_min_score,
            maximum=settings.max_final_candidates,
        )
        scored_audit = _apply_score_audit(
            cleaning.audit_rows,
            ranked,
            finalists,
            settings.final_min_score,
        )
        save_real_audit_v2(scored_audit, settings.real_candidates_audit_v2_path)
        _write_run_audit(audit_snapshot("scored"), settings.final_run_audit_v2_path)

        offers: list[FinalPersonalizedOffer] = []
        if finalists:
            if active_offer_provider is None:
                try:
                    prompt = settings.final_offer_prompt_path.read_text(encoding="utf-8")
                except OSError as exc:
                    raise FinalPipelineError(
                        f"Cannot read final offer prompt: {settings.final_offer_prompt_path}"
                    ) from exc
                active_offer_provider = OpenAIFinalOfferProvider(
                    api_key=settings.openai_api_key,
                    model=settings.openai_model,
                    timeout_seconds=settings.openai_request_timeout_seconds,
                    prompt=prompt,
                )
            offer_result = generate_final_offers(finalists, active_offer_provider)
            offers = offer_result.offers
            errors.extend(offer_result.errors)
        save_final_outputs(
            finalists,
            offers,
            csv_path=settings.final_real_bloggers_csv_v2_path,
            markdown_path=settings.final_real_bloggers_markdown_v2_path,
        )
        final_status = "completed_with_errors" if errors else "completed"
        final_audit = audit_snapshot(final_status)
        _write_run_audit(final_audit, settings.final_run_audit_v2_path)
        return FinalPipelineResult(plan=plan, finalists=finalists, audit=final_audit)
    except Exception as exc:
        errors.append(_safe_pipeline_error(exc))
        failed_audit = audit_snapshot("failed")
        _write_run_audit(failed_audit, settings.final_run_audit_v2_path)
        raise


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_saved_v2_pool(settings: Settings) -> list[RealCandidateProfile]:
    """Rebuild the cleaned canonical pool from saved v2 files without search."""

    required = (
        settings.real_candidates_raw_v2_path,
        settings.content_author_resolution_audit_path,
        settings.real_candidates_audit_v2_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FinalPipelineError(
            "Saved v2 pool is incomplete; missing files: " + ", ".join(missing)
        )
    with settings.real_candidates_raw_v2_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        hits = [
            SearchHit(
                url=row["url"],
                title=row.get("title") or None,
                snippet=row.get("content") or None,
                source_query=row.get("source_query") or "saved v2 query",
                provider_score=_optional_float(row.get("score")),
            )
            for row in csv.DictReader(stream)
            if row.get("url")
        ]
    with settings.content_author_resolution_audit_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        saved_resolution_rows = list(csv.DictReader(stream))
    resolution_by_url: dict[str, ContentAuthorResolution] = {}
    for row in saved_resolution_rows:
        platform_value = row.get("platform") or None
        platform = Platform(platform_value) if platform_value else None
        resolution_by_url[row["content_url"]] = ContentAuthorResolution(
            content_url=row["content_url"],
            platform=platform,
            resolved_profile_url=row.get("resolved_profile_url") or None,
            resolved_username_or_channel=row.get("resolved_author") or None,
            resolution_method=row.get("resolution_method") or "saved_v2_audit",
            confidence=_optional_float(row.get("confidence")) or 0,
            status=row.get("status") or "unresolved_author",
            reason=row.get("reason") or "Saved resolution has no reason",
        )
    resolutions: list[ContentAuthorResolution] = []
    for hit in hits:
        saved = resolution_by_url.get(hit.url)
        if saved is not None:
            resolutions.append(saved)
            continue
        kind, platform, normalized = classify_public_url(hit.url)
        if kind == "profile" and platform is not None and normalized is not None:
            identity = [
                part for part in urlsplit(normalized).path.split("/") if part
            ][-1].lstrip("@")
            resolutions.append(
                ContentAuthorResolution(
                    content_url=hit.url,
                    platform=platform,
                    resolved_profile_url=normalized,
                    resolved_username_or_channel=identity,
                    resolution_method="saved_direct_profile_url",
                    confidence=1.0,
                    status="profile_url",
                    reason="Saved Tavily URL is already a canonical profile",
                )
            )
        else:
            resolutions.append(
                ContentAuthorResolution(
                    content_url=hit.url,
                    platform=platform,
                    resolved_profile_url=None,
                    resolved_username_or_channel=None,
                    resolution_method="saved_not_applicable",
                    confidence=0,
                    status="unsupported_url",
                    reason="Saved URL has no supported canonical profile",
                )
            )
    reference_urls, reference_usernames = load_reference_identities(
        settings.enriched_source_json_path
    )
    cleaned = clean_resolved_search_hits(
        hits,
        resolutions,
        reference_urls=reference_urls,
        reference_usernames=reference_usernames,
        maximum_candidates=20,
        minimum_resolution_confidence=settings.min_author_resolution_confidence,
    )
    if len(cleaned.candidates) != 20:
        raise FinalPipelineError(
            "Saved v2 pool must contain exactly 20 cleaned candidates; "
            f"reconstructed {len(cleaned.candidates)}"
        )
    return cleaned.candidates


def _load_saved_enriched_candidates(settings: Settings) -> list[RealCandidateProfile]:
    path = settings.real_candidates_enriched_v2_path
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("saved enrichment must be a list")
        return [RealCandidateProfile.model_validate(item) for item in payload]
    except (OSError, ValueError, ValidationError) as exc:
        raise FinalPipelineError(f"Saved v2 enrichment is invalid: {path}") from exc


def build_saved_pool_plan(
    settings: Settings,
    pool: list[RealCandidateProfile] | None = None,
) -> SavedPoolPlan:
    """Count saved/fresh cache hits without constructing any API provider."""

    candidates = pool if pool is not None else load_saved_v2_pool(settings)
    candidate_urls = {str(item.profile_url).rstrip("/").casefold() for item in candidates}
    saved = {
        str(item.profile_url).rstrip("/").casefold(): item
        for item in _load_saved_enriched_candidates(settings)
        if str(item.profile_url).rstrip("/").casefold() in candidate_urls
    }
    fresh_cache = 0
    for candidate in candidates:
        canonical = str(candidate.profile_url).rstrip("/").casefold()
        if canonical in saved:
            continue
        if load_profile_cache(candidate.username, settings.real_candidate_cache_dir) is not None:
            fresh_cache += 1
    cache_total = len(saved) + fresh_cache
    return SavedPoolPlan(
        pool_size=len(candidates),
        saved_enriched_count=len(saved),
        fresh_cache_count=fresh_cache,
        apify_required_count=max(0, len(candidates) - cache_total),
        max_candidates_for_apify=20,
        posts_per_profile=3,
        max_openai_offers=5,
    )


def categorize_saved_pool(
    ranked: list[FinalScoredCandidate],
) -> tuple[
    list[FinalScoredCandidate],
    list[FinalScoredCandidate],
    list[FinalScoredCandidate],
]:
    """Apply decision labels without changing score or FINAL_MIN_SCORE."""

    recommended = [item for item in ranked if item.score.total_score >= 70]
    manual_review = [item for item in ranked if 60 <= item.score.total_score <= 69]
    rejected = [item for item in ranked if item.score.total_score < 60]
    return recommended, manual_review, rejected


def select_saved_pool_offer_targets(
    recommended: list[FinalScoredCandidate],
    manual_review: list[FinalScoredCandidate],
) -> list[FinalScoredCandidate]:
    """Draft for recommended profiles, filling to three from manual review."""

    targets = list(recommended[:5])
    if len(targets) < 3:
        targets.extend(manual_review[: 3 - len(targets)])
    return targets[:5]


def _decision_status(score: int) -> str:
    if score >= 70:
        return "recommended"
    if score >= 60:
        return "manual_review"
    return "rejected"


def _all_candidate_record(
    item: FinalScoredCandidate,
    offer: FinalPersonalizedOffer | None,
) -> dict[str, Any]:
    candidate = item.candidate
    breakdown = final_score_breakdown_row(item)
    decision = _decision_status(item.score.total_score)
    details = {
        key: detail.model_dump(mode="json")
        for key, detail in item.criterion_details.items()
    }
    offer_text = offer.message if offer is not None else ""
    if offer_text and decision == "manual_review":
        offer_text = (
            "[MANUAL_REVIEW — не одобрено автоматически]\n" + offer_text
        )
    return {
        "name": candidate.name,
        "username": candidate.username,
        "platform": candidate.platform.value,
        "profile_url": str(candidate.profile_url),
        "total_score": item.score.total_score,
        "decision_status": decision,
        "data_confidence": candidate.data_confidence,
        "evidence_count": candidate.evidence_count,
        "topic_score": breakdown.topic_score,
        "topic_reason": breakdown.topic_reason,
        "visual_score": breakdown.visual_score,
        "visual_reason": breakdown.visual_reason,
        "audience_score": breakdown.audience_score,
        "audience_reason": breakdown.audience_reason,
        "tone_score": breakdown.tone_score,
        "tone_reason": breakdown.tone_reason,
        "engagement_score": breakdown.engagement_score,
        "engagement_reason": breakdown.engagement_reason,
        "ad_load_score": breakdown.ad_load_score,
        "ad_load_reason": breakdown.ad_load_reason,
        "price_segment_score": breakdown.price_segment_score,
        "price_segment_reason": breakdown.price_segment_reason,
        "format_score": breakdown.format_score,
        "format_reason": breakdown.format_reason,
        "brand_safety_score": breakdown.brand_safety_score,
        "brand_safety_reason": breakdown.brand_safety_reason,
        "confidence_reason": breakdown.confidence_reason,
        "scoring_breakdown": json.dumps(details, ensure_ascii=False),
        "match_reason": item.match_reason,
        "personalized_offer": offer_text,
        "manual_review_status": (
            "needs_review" if decision in {"recommended", "manual_review"} else "not_selected"
        ),
    }


def save_saved_pool_outputs(
    ranked: list[FinalScoredCandidate],
    offers: list[FinalPersonalizedOffer],
    *,
    all_csv_path: Path,
    markdown_path: Path,
    recommended_csv_path: Path,
) -> None:
    """Write all three human-in-the-loop saved-pool deliverables."""

    offers_by_username = {offer.candidate_username.casefold(): offer for offer in offers}
    records = [
        _all_candidate_record(
            item,
            offers_by_username.get(item.candidate.username.casefold()),
        )
        for item in ranked
    ]
    all_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=FINAL_ALL_COLUMNS).to_csv(
        all_csv_path,
        index=False,
        encoding="utf-8",
    )
    recommended_records = [
        record for record in records if record["decision_status"] == "recommended"
    ]
    pd.DataFrame(recommended_records, columns=FINAL_ALL_COLUMNS).to_csv(
        recommended_csv_path,
        index=False,
        encoding="utf-8",
    )

    grouped = {
        "recommended": "Рекомендованные — 70+",
        "manual_review": "Резерв для ручной проверки — 60–69",
        "rejected": "Отклонённые — ниже 60",
    }
    lines = [
        "# Итоговая оценка сохранённого v2-пула LD LATTE",
        "",
        "> Решение всегда подтверждает человек. Офферы не отправлены автоматически.",
        "",
        "`manual_review` — резерв, а не автоматическое одобрение.",
    ]
    for status, heading in grouped.items():
        lines.extend(("", f"## {heading}", ""))
        matching = [
            (item, record)
            for item, record in zip(ranked, records, strict=True)
            if record["decision_status"] == status
        ]
        if not matching:
            lines.append("Нет кандидатов в этой категории.")
            continue
        for index, (item, record) in enumerate(matching, start=1):
            candidate = item.candidate
            lines.extend(
                (
                    f"### {index}. {candidate.name} — {item.score.total_score}/100",
                    "",
                    f"- Профиль: {candidate.profile_url}",
                    f"- Статус: `{status}`",
                    f"- Data confidence: {candidate.data_confidence:.3f}",
                    f"- Evidence count: {candidate.evidence_count}",
                    f"- Ручная проверка: `{record['manual_review_status']}`",
                    f"- Причина: {item.match_reason}",
                    "",
                    "| Критерий | Баллы | Причина |",
                    "|---|---:|---|",
                )
            )
            for key, label in (
                ("topic", "Тематика"),
                ("visual", "Визуальные/текстовые сигналы"),
                ("audience", "Аудитория"),
                ("tone", "Тон"),
                ("engagement", "Вовлечённость"),
                ("ad_load", "Рекламная нагрузка"),
                ("price_segment", "Ценовой сегмент"),
                ("format", "Форматы"),
                ("brand_safety", "Brand safety"),
                ("data_confidence", "Полнота данных"),
            ):
                detail = item.criterion_details[key]
                reason = detail.reason.replace("|", "\\|").replace("\n", " ")
                if "confidence:" not in reason.casefold():
                    reason = f"{reason} confidence: {detail.confidence}"
                lines.append(
                    f"| {label} | {detail.score}/{detail.max_score} | {reason} |"
                )
            if record["personalized_offer"]:
                lines.extend(
                    (
                        "",
                        "#### Черновик оффера — не отправлен",
                        "",
                        str(record["personalized_offer"]),
                    )
                )
            lines.append("")
    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def finalize_saved_v2_pool(
    settings: Settings,
    *,
    dry_run: bool,
    enrichment_provider: ProfileEnrichmentProvider | None = None,
    offer_provider: Any | None = None,
) -> SavedPoolResult:
    """Finalize saved canonical profiles without Tavily or author resolution."""

    pool = load_saved_v2_pool(settings)
    plan = build_saved_pool_plan(settings, pool)
    if dry_run:
        return SavedPoolResult(plan, [], [], [], [], [])

    saved_enriched = _load_saved_enriched_candidates(settings)
    pool_urls = {str(item.profile_url).rstrip("/").casefold() for item in pool}
    enriched_by_url = {
        str(item.profile_url).rstrip("/").casefold(): item
        for item in saved_enriched
        if str(item.profile_url).rstrip("/").casefold() in pool_urls
    }
    pending = [
        candidate
        for candidate in pool
        if str(candidate.profile_url).rstrip("/").casefold() not in enriched_by_url
    ]
    if pending:
        active_enrichment = enrichment_provider
        if active_enrichment is None:
            if not settings.apify_api_token or not settings.apify_actor_id:
                raise FinalPipelineError(
                    "APIFY_API_TOKEN and APIFY_ACTOR_ID are required for the "
                    f"{plan.apify_required_count} uncached saved-pool profiles"
                )
            active_enrichment = create_profile_enrichment_provider(
                provider_name="apify",
                mock_fixture_path=settings.profile_enrichment_mock_path,
                apify_api_token=settings.apify_api_token,
                apify_actor_id=settings.apify_actor_id,
                timeout_seconds=max(settings.request_timeout_seconds, 300),
                apify_raw_response_path=settings.final_apify_raw_response_v2_path,
            )
        pending_audit = [
            RealCandidateAuditRow(
                raw_url=str(candidate.profile_url),
                normalized_url=str(candidate.profile_url),
                platform=candidate.platform,
                source_query=candidate.source_query,
                title=candidate.title,
                tavily_score=candidate.tavily_score,
                decision="accepted_saved_pool",
                reason="saved canonical profile awaiting enrichment",
                data_confidence=candidate.data_confidence,
                evidence_count=candidate.evidence_count,
                evidence_urls=candidate.evidence_urls,
                author_resolution_confidence=candidate.author_resolution_confidence,
            )
            for candidate in pending
        ]
        saved_pool_settings = replace(
            settings,
            max_candidates_for_apify=20,
            profile_posts_limit=3,
        )
        newly_enriched, _, _ = enrich_instagram_candidates(
            pending,
            pending_audit,
            provider=active_enrichment,
            settings=saved_pool_settings,
        )
        enriched_by_url.update(
            {
                str(item.profile_url).rstrip("/").casefold(): item
                for item in newly_enriched
            }
        )
    complete_pool = [
        enriched_by_url.get(str(candidate.profile_url).rstrip("/").casefold(), candidate)
        for candidate in pool
    ]
    ideal = _load_profile(settings.ideal_blogger_profile_json_path)
    ranked = rank_real_candidates(complete_pool, ideal)
    save_final_score_breakdown(ranked, settings.final_score_breakdown_path)
    recommended, manual_review, rejected = categorize_saved_pool(ranked)
    offer_targets = select_saved_pool_offer_targets(recommended, manual_review)
    active_offer_provider = offer_provider
    offers: list[FinalPersonalizedOffer] = []
    if offer_targets:
        if active_offer_provider is None:
            if not settings.openai_api_key:
                raise FinalPipelineError(
                    "OPENAI_API_KEY is required to draft saved-pool finalist offers"
                )
            active_offer_provider = OpenAIFinalOfferProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                timeout_seconds=settings.openai_request_timeout_seconds,
                prompt=settings.final_offer_prompt_path.read_text(encoding="utf-8"),
            )
        offer_result = generate_final_offers(offer_targets, active_offer_provider)
        offers = offer_result.offers
        if offer_result.errors:
            LOGGER.warning(
                "Saved-pool offer drafts completed with %d recoverable errors",
                len(offer_result.errors),
            )
    save_saved_pool_outputs(
        ranked,
        offers,
        all_csv_path=settings.final_all_candidates_csv_path,
        markdown_path=settings.final_all_candidates_markdown_path,
        recommended_csv_path=settings.final_recommended_bloggers_csv_path,
    )
    return SavedPoolResult(
        plan=plan,
        ranked=ranked,
        recommended=recommended,
        manual_review=manual_review,
        rejected=rejected,
        offer_targets=offer_targets,
    )

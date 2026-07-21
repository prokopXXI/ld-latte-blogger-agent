"""Normalize, enrich, filter, deduplicate, and audit public search results."""

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from src.models import (
    AuditReason,
    BrandSafetyLevel,
    CandidateProfile,
    CandidateSource,
    EnrichedCandidate,
    Platform,
    PriceSegment,
    SearchAuditRow,
    SearchHit,
    SourceBlogger,
)


SUPPORTED_HOSTS = {
    "instagram.com": Platform.INSTAGRAM,
    "www.instagram.com": Platform.INSTAGRAM,
    "youtube.com": Platform.YOUTUBE_SHORTS,
    "www.youtube.com": Platform.YOUTUBE_SHORTS,
    "m.youtube.com": Platform.YOUTUBE_SHORTS,
    "youtu.be": Platform.YOUTUBE_SHORTS,
    "t.me": Platform.TELEGRAM,
    "telegram.me": Platform.TELEGRAM,
    "www.telegram.me": Platform.TELEGRAM,
}
INSTAGRAM_NON_PROFILE_PATHS = {
    "accounts",
    "direct",
    "explore",
    "p",
    "reel",
    "reels",
    "stories",
}
TOPIC_MARKERS = {
    "женская мода": ("женская мода", "женская одежда", "fashion"),
    "капсульный гардероб": ("капсул",),
    "повседневные образы": ("повседнев", "образ на каждый день"),
    "образы для офиса": ("офисн", "деловой образ"),
    "сочетание вещей": ("сочетан", "стилизац"),
    "примерки": ("примерк",),
    "стильные подборки": ("подборк",),
    "обзоры одежды": ("обзор одежды", "обзоры одежды"),
}
VISUAL_MARKERS = {
    "минималистичный": ("минимал",),
    "светлый": ("светлый", "светлая"),
    "естественный": ("естествен", "без фильтр"),
    "чистый": ("чистый визуал", "лаконичн"),
    "структурированный": ("структур",),
    "реалистичный": ("реалист", "реальная посадка"),
    "глянцевый": ("глянц",),
    "динамичный": ("динамич",),
    "тёплый": ("теплый", "тёплый"),
}
TONE_MARKERS = {
    "доверительный": ("доверител", "честн"),
    "спокойный": ("спокойн",),
    "экспертный": ("эксперт", "стилист", "разбор"),
    "естественный": ("естествен", "без фильтр"),
    "дружелюбный": ("дружелюб",),
    "энергичный": ("энергич", "динамич"),
    "продающий": ("скидк", "промокод", "акци"),
}
INTEREST_MARKERS = {
    "капсульный гардероб": ("капсул",),
    "офисный стиль": ("офисн",),
    "покупки на маркетплейсах": ("wildberries", "ozon", "маркетплейс"),
    "доступная одежда": ("доступн", "бюджет", "масс-маркет"),
    "повседневный стиль": ("повседнев", "каждый день"),
    "сочетание вещей": ("сочетан", "стилизац"),
    "качество и посадка": ("качество", "посадк", "размер"),
}
FORMAT_MARKERS = {
    "reels": ("reel", "рилс"),
    "shorts": ("shorts",),
    "примерки": ("примерк",),
    "stories": ("stories", "сторис"),
    "карусели": ("карусел",),
    "обзоры": ("обзор",),
    "гайды": ("гайд",),
    "фотоподборки": ("фотоподбор",),
}
BRAND_MARKERS = (
    "официальный магазин",
    "интернет-магазин",
    "магазин одежды",
    "бренд одежды",
    "shop",
    "store",
    "каталог товаров",
)
AUTHOR_MARKERS = ("блог", "стилист", "автор", "образы", "примерки", "личный")
AUDIT_COLUMNS = [
    "url",
    "normalized_url",
    "source_query",
    "source_title",
    "data_confidence",
    "reason",
]


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Clean scoring candidates plus the audit decisions for all raw hits."""

    candidates: list[CandidateProfile]
    audit_rows: list[SearchAuditRow]
    total_found: int


def normalize_profile_url(url: str) -> str | None:
    """Return a stable public-profile URL or `None` for unsupported pages."""

    raw_url = url.strip()
    if not raw_url:
        return None
    if "://" not in raw_url:
        raw_url = f"https://{raw_url}"
    parsed = urlsplit(raw_url)
    host = parsed.netloc.casefold().split(":", maxsplit=1)[0]
    platform = SUPPORTED_HOSTS.get(host)
    if platform is None:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None
    if platform == Platform.INSTAGRAM:
        username = parts[0].casefold()
        if username in INSTAGRAM_NON_PROFILE_PATHS:
            return None
        return f"https://www.instagram.com/{username}/"
    if platform == Platform.TELEGRAM:
        if parts[0] == "s" and len(parts) > 1:
            parts = parts[1:]
        username = parts[0]
        if username.startswith("+") or username.casefold() in {"share", "proxy"}:
            return None
        return f"https://t.me/{username.casefold()}"
    if host == "youtu.be":
        return f"https://youtu.be/{parts[0]}"
    if parts[0].casefold() in {"watch", "results", "feed"}:
        return None
    normalized_path = "/".join(parts[:2])
    return f"https://www.youtube.com/{normalized_path}"


def platform_from_url(url: str) -> Platform | None:
    """Resolve a supported platform after URL normalization."""

    parsed = urlsplit(url)
    return SUPPORTED_HOSTS.get(parsed.netloc.casefold())


def _labels_from_text(text: str, markers: dict[str, tuple[str, ...]]) -> list[str]:
    normalized = text.casefold()
    return [
        label
        for label, variants in markers.items()
        if any(variant in normalized for variant in variants)
    ]


def _display_name(title: str | None, profile_url: str) -> str | None:
    if title:
        cleaned = re.split(r"[|•]", title, maxsplit=1)[0].strip()
        cleaned = re.sub(r"\s*[-—]\s*(Instagram|YouTube|Telegram).*$", "", cleaned, flags=re.I)
        if cleaned:
            return cleaned[:200]
    parts = [part for part in urlsplit(profile_url).path.split("/") if part]
    return parts[-1].lstrip("@")[:200] if parts else None


def _is_brand_or_store(title: str | None, snippet: str | None, profile_url: str) -> bool:
    title_text = (title or "").casefold()
    combined = f"{title or ''} {snippet or ''}".casefold()
    path = urlsplit(profile_url).path.casefold()
    has_author_marker = any(marker in combined for marker in AUTHOR_MARKERS)
    title_is_commercial = any(marker in title_text for marker in BRAND_MARKERS)
    handle_is_commercial = any(marker in path for marker in ("shop", "store", "official_store"))
    return (title_is_commercial or handle_is_commercial) and not has_author_marker


def _parse_metric(text: str, label_pattern: str) -> float | None:
    match = re.search(rf"{label_pattern}[^0-9]{{0,12}}(\d+(?:[.,]\d+)?)\s*%", text, re.I)
    return float(match.group(1).replace(",", ".")) if match else None


def _parse_count(text: str, label_pattern: str) -> int | None:
    match = re.search(
        rf"(\d+(?:[.,]\d+)?)\s*(тыс\.?|k)?\s*{label_pattern}",
        text,
        re.I,
    )
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    if match.group(2):
        value *= 1_000
    return round(value)


def _infer_price_segment(text: str) -> PriceSegment | None:
    normalized = text.casefold()
    if any(marker in normalized for marker in ("премиум", "люкс", "luxury")):
        return PriceSegment.PREMIUM
    if any(marker in normalized for marker in ("доступн", "бюджет", "масс-маркет")):
        return PriceSegment.MASS_MARKET
    if "средний ценовой" in normalized:
        return PriceSegment.MIDDLE
    return None


def _infer_brand_safety(text: str) -> BrandSafetyLevel | None:
    normalized = text.casefold()
    if "brand safety: high" in normalized:
        return BrandSafetyLevel.HIGH
    if "brand safety: medium" in normalized:
        return BrandSafetyLevel.MEDIUM
    if "brand safety: low" in normalized:
        return BrandSafetyLevel.LOW
    return None


def _confidence_score(
    *,
    name: str | None,
    title: str | None,
    snippet: str | None,
    topics: list[str] | None,
    visual_style: list[str] | None,
    tone: list[str] | None,
    audience_description: str | None,
    audience_interests: list[str] | None,
    content_formats: list[str] | None,
    has_numeric_metric: bool,
    price_segment: PriceSegment | None,
    brand_safety: BrandSafetyLevel | None,
) -> float:
    confidence = 0.0
    confidence += 0.10 if name else 0.0
    confidence += 0.10 if title else 0.0
    confidence += 0.10 if snippet else 0.0
    confidence += 0.20 if topics else 0.0
    confidence += 0.10 if visual_style else 0.0
    confidence += 0.10 if tone else 0.0
    confidence += 0.10 if audience_description or audience_interests else 0.0
    confidence += 0.10 if content_formats else 0.0
    confidence += 0.05 if has_numeric_metric else 0.0
    confidence += 0.025 if price_segment else 0.0
    confidence += 0.025 if brand_safety else 0.0
    return round(confidence, 3)


def enrich_search_hit(hit: SearchHit, normalized_url: str) -> EnrichedCandidate:
    """Infer only fields supported by a result title, snippet, and public URL."""

    if hit.prefilled_candidate is not None:
        candidate = hit.prefilled_candidate
        return EnrichedCandidate(
            name=candidate.display_name,
            platform=candidate.platform,
            profile_url=candidate.profile_url,
            title=hit.title,
            snippet=hit.snippet,
            source_query=hit.source_query,
            content_topics=candidate.content_topics,
            visual_style=candidate.visual_style,
            tone=candidate.tone,
            audience_description=candidate.audience_description,
            audience_interests=candidate.audience_interests,
            content_formats=candidate.content_formats,
            content_style=candidate.content_style,
            followers=candidate.followers,
            engagement_rate_pct=candidate.engagement_rate_pct,
            average_views=candidate.average_views,
            advertising_load_pct=candidate.advertising_load_pct,
            price_segment=candidate.price_segment,
            brand_safety=candidate.brand_safety,
            is_brand_or_store=False,
            data_confidence=1.0,
        )

    platform = platform_from_url(normalized_url)
    evidence = f"{hit.title or ''} {hit.snippet or ''}"
    name = _display_name(hit.title, normalized_url)
    topics = _labels_from_text(evidence, TOPIC_MARKERS) or None
    visual_style = _labels_from_text(evidence, VISUAL_MARKERS) or None
    tone = _labels_from_text(evidence, TONE_MARKERS) or None
    audience_interests = _labels_from_text(evidence, INTEREST_MARKERS) or None
    audience_description = (
        hit.snippet
        if hit.snippet and any(marker in evidence.casefold() for marker in ("женщ", "девуш"))
        else None
    )
    detected_formats = _labels_from_text(evidence, FORMAT_MARKERS)
    if platform == Platform.INSTAGRAM and "публикации" not in detected_formats:
        detected_formats.append("публикации")
    elif platform == Platform.TELEGRAM and "публикации" not in detected_formats:
        detected_formats.append("публикации")
    elif platform == Platform.YOUTUBE_SHORTS and "shorts" not in detected_formats:
        detected_formats.append("shorts")
    content_formats = detected_formats or None
    engagement_rate = _parse_metric(evidence, r"(?:er|вовлеч[её]нность)")
    advertising_load = _parse_metric(evidence, r"(?:рекламная нагрузка|реклама)")
    followers = _parse_count(evidence, r"(?:подписчиков|followers)")
    average_views = _parse_count(evidence, r"(?:просмотров|views)")
    price_segment = _infer_price_segment(evidence)
    brand_safety = _infer_brand_safety(evidence)
    data_confidence = _confidence_score(
        name=name,
        title=hit.title,
        snippet=hit.snippet,
        topics=topics,
        visual_style=visual_style,
        tone=tone,
        audience_description=audience_description,
        audience_interests=audience_interests,
        content_formats=content_formats,
        has_numeric_metric=any(
            value is not None
            for value in (followers, average_views, engagement_rate, advertising_load)
        ),
        price_segment=price_segment,
        brand_safety=brand_safety,
    )
    return EnrichedCandidate(
        name=name,
        platform=platform,
        profile_url=normalized_url,
        title=hit.title,
        snippet=hit.snippet,
        source_query=hit.source_query,
        content_topics=topics,
        visual_style=visual_style,
        tone=tone,
        audience_description=audience_description,
        audience_interests=audience_interests,
        content_formats=content_formats,
        content_style=hit.title[:500] if hit.title else None,
        followers=followers,
        engagement_rate_pct=engagement_rate,
        average_views=average_views,
        advertising_load_pct=advertising_load,
        price_segment=price_segment,
        brand_safety=brand_safety,
        is_brand_or_store=_is_brand_or_store(hit.title, hit.snippet, normalized_url),
        data_confidence=data_confidence,
    )


def _has_sufficient_data(candidate: EnrichedCandidate) -> bool:
    has_audience = bool(candidate.audience_description or candidate.audience_interests)
    return bool(
        candidate.name
        and candidate.platform
        and candidate.content_topics
        and candidate.content_formats
        and has_audience
    )


def _handle_from_url(profile_url: str) -> str:
    parts = [part for part in urlsplit(profile_url).path.split("/") if part]
    slug = parts[-1] if parts else "public_profile"
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "", slug).strip(".") or "public_profile"
    return f"@{cleaned[:99]}"


def _to_candidate_profile(
    enriched: EnrichedCandidate,
    prefilled: CandidateProfile | None,
) -> CandidateProfile:
    metadata = {
        "source_query": enriched.source_query,
        "source_title": enriched.title,
        "source_snippet": enriched.snippet,
        "data_confidence": enriched.data_confidence,
    }
    if prefilled is not None:
        return prefilled.model_copy(update=metadata)
    return CandidateProfile(
        handle=_handle_from_url(str(enriched.profile_url)),
        display_name=enriched.name or "Unknown public profile",
        platform=enriched.platform or Platform.OTHER,
        profile_url=enriched.profile_url,
        followers=enriched.followers,
        engagement_rate_pct=enriched.engagement_rate_pct,
        average_views=enriched.average_views,
        content_topics=enriched.content_topics,
        audience_description=enriched.audience_description,
        audience_interests=enriched.audience_interests,
        content_style=enriched.content_style,
        content_formats=enriched.content_formats,
        tone=enriched.tone,
        visual_style=enriched.visual_style,
        advertising_load_pct=enriched.advertising_load_pct,
        price_segment=enriched.price_segment,
        brand_safety=enriched.brand_safety,
        location=None,
        source=CandidateSource.MANUAL_RESEARCH,
        source_url=enriched.profile_url,
        notes=(enriched.snippet or enriched.title or "Public search result")[:1_000],
        **metadata,
    )


def discover_candidates(
    hits: list[SearchHit],
    source_bloggers: list[SourceBlogger],
    minimum_confidence: float = 0.45,
) -> DiscoveryResult:
    """Normalize, deduplicate, enrich, prefilter, and audit every search hit."""

    source_urls = {str(source.profile_url).rstrip("/").casefold() for source in source_bloggers}
    seen_urls: set[str] = set()
    candidates: list[CandidateProfile] = []
    audit_rows: list[SearchAuditRow] = []

    for hit in hits:
        normalized_url = (
            str(hit.prefilled_candidate.profile_url)
            if hit.prefilled_candidate is not None
            else normalize_profile_url(hit.url)
        )
        if normalized_url is None:
            audit_rows.append(
                SearchAuditRow(
                    url=hit.url,
                    source_query=hit.source_query,
                    source_title=hit.title,
                    reason=AuditReason.UNSUPPORTED_DOMAIN,
                )
            )
            continue

        canonical_url = normalized_url.rstrip("/").casefold()
        if canonical_url in seen_urls or canonical_url in source_urls:
            audit_rows.append(
                SearchAuditRow(
                    url=hit.url,
                    normalized_url=normalized_url,
                    source_query=hit.source_query,
                    source_title=hit.title,
                    reason=AuditReason.DUPLICATE,
                )
            )
            continue
        seen_urls.add(canonical_url)

        enriched = enrich_search_hit(hit, normalized_url)
        if enriched.is_brand_or_store:
            reason = AuditReason.BRAND_OR_STORE
        elif not _has_sufficient_data(enriched):
            reason = AuditReason.INSUFFICIENT_DATA
        elif enriched.data_confidence < minimum_confidence:
            reason = AuditReason.LOW_CONFIDENCE
        else:
            reason = AuditReason.ACCEPTED
            candidates.append(_to_candidate_profile(enriched, hit.prefilled_candidate))
        audit_rows.append(
            SearchAuditRow(
                url=hit.url,
                normalized_url=normalized_url,
                source_query=hit.source_query,
                source_title=hit.title,
                data_confidence=enriched.data_confidence,
                reason=reason,
            )
        )

    return DiscoveryResult(
        candidates=candidates,
        audit_rows=audit_rows,
        total_found=len(hits),
    )


def mark_below_min_score(
    audit_rows: list[SearchAuditRow],
    below_score_urls: set[str],
) -> list[SearchAuditRow]:
    """Replace `accepted` with `below_min_score` after deterministic scoring."""

    canonical_below = {url.rstrip("/").casefold() for url in below_score_urls}
    updated: list[SearchAuditRow] = []
    for row in audit_rows:
        normalized = (row.normalized_url or "").rstrip("/").casefold()
        if row.reason == AuditReason.ACCEPTED and normalized in canonical_below:
            updated.append(row.model_copy(update={"reason": AuditReason.BELOW_MIN_SCORE}))
        else:
            updated.append(row)
    return updated


def save_search_audit(rows: list[SearchAuditRow], path: Path) -> None:
    """Persist all search decisions in a stable CSV schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    records = [row.model_dump(mode="json") for row in rows]
    pd.DataFrame(records, columns=AUDIT_COLUMNS).to_csv(path, index=False, encoding="utf-8")

"""Batch LLM analysis of enriched source bloggers with a fully offline mock."""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from src.config import Settings
from src.models import (
    BatchBloggerInsights,
    EnrichedSourceBlogger,
    LLMAnalysisAudit,
    LLMIdealBloggerProfile,
    LLMRecentPostInput,
    LLMSourceProfileInput,
    ProfileEnrichmentStatus,
)


LOGGER = logging.getLogger(__name__)
CAPTION_LIMIT = 1_500
BIOGRAPHY_LIMIT = 1_000
ACCESSIBILITY_CAPTION_LIMIT = 800
MAX_HASHTAGS_PER_POST = 30
ESTIMATED_INSIGHT_CHARS_PER_PROFILE = 650
OPENAI_MAX_RETRIES = 2
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")


class LLMAnalysisError(RuntimeError):
    """Raised for safe, actionable LLM-analysis failures."""


@dataclass(frozen=True, slots=True)
class LLMDryRunSummary:
    """Network-free estimate printed before a possible paid analysis."""

    profile_count: int
    batch_count: int
    approximate_characters: int
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class LLMAnalysisRun:
    """Validated result of a dry-run, mock run, or OpenAI run."""

    summary: LLMDryRunSummary
    batch_insights: list[BatchBloggerInsights]
    ideal_profile: LLMIdealBloggerProfile | None
    audit: LLMAnalysisAudit


def _sanitize_text(value: str | None, limit: int) -> str | None:
    """Remove contact-like data, normalize whitespace, and cap prompt size."""

    if not value:
        return None
    cleaned = EMAIL_PATTERN.sub("[redacted-email]", value)
    cleaned = PHONE_PATTERN.sub("[redacted-phone]", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def prepare_llm_profile(
    blogger: EnrichedSourceBlogger,
    max_posts: int,
    caption_limit: int = CAPTION_LIMIT,
) -> LLMSourceProfileInput:
    """Project one enriched record onto the explicit LLM data allow-list."""

    recent_posts: list[LLMRecentPostInput] = []
    for post in blogger.recent_posts[:max_posts]:
        hashtags = [
            safe_tag
            for tag in (post.hashtags or [])[:MAX_HASHTAGS_PER_POST]
            if isinstance(tag, str)
            if (safe_tag := _sanitize_text(tag, 100)) is not None
        ]
        safe_post = LLMRecentPostInput(
            caption=_sanitize_text(post.caption, caption_limit),
            hashtags=hashtags,
            post_type=_sanitize_text(post.post_type, 100),
            accessibility_caption=_sanitize_text(
                post.accessibility_caption,
                ACCESSIBILITY_CAPTION_LIMIT,
            ),
        )
        if any(
            (
                safe_post.caption,
                safe_post.hashtags,
                safe_post.post_type,
                safe_post.accessibility_caption,
            )
        ):
            recent_posts.append(safe_post)

    return LLMSourceProfileInput(
        username=blogger.profile.username,
        full_name=_sanitize_text(blogger.profile.full_name, 300),
        biography=_sanitize_text(blogger.profile.biography, BIOGRAPHY_LIMIT),
        followers_count=blogger.profile.followers_count,
        calculated_engagement_rate=blogger.calculated_engagement_rate,
        is_private=blogger.profile.is_private,
        recent_posts=recent_posts,
    )


def load_llm_profiles(
    path: Path,
    max_profiles: int,
    max_posts: int,
    caption_limit: int = CAPTION_LIMIT,
) -> list[LLMSourceProfileInput]:
    """Load usable enriched records and return only allow-listed prompt data."""

    if not path.is_file():
        raise LLMAnalysisError(
            f"Enriched source file not found: {path}. Run --enrich-source first."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMAnalysisError(f"Cannot read enriched source file: {path}") from exc
    if not isinstance(payload, list):
        raise LLMAnalysisError(
            "Enriched source file must contain a JSON array of blogger records."
        )

    profiles: list[LLMSourceProfileInput] = []
    for index, raw_blogger in enumerate(payload, start=1):
        try:
            blogger = EnrichedSourceBlogger.model_validate(raw_blogger)
        except ValidationError as exc:
            raise LLMAnalysisError(
                f"Invalid enriched source record at row {index}: {exc.error_count()} validation error(s)."
            ) from exc
        if blogger.enrichment_status not in {
            ProfileEnrichmentStatus.SUCCESS,
            ProfileEnrichmentStatus.PARTIAL,
        }:
            continue
        profiles.append(prepare_llm_profile(blogger, max_posts, caption_limit))
        if len(profiles) >= max_profiles:
            break

    if not profiles:
        raise LLMAnalysisError(
            "No profiles with enrichment_status=success or partial were found."
        )
    LOGGER.info(
        "Prepared %d usable profiles from %s; failed/not_found records were excluded",
        len(profiles),
        path,
    )
    return profiles


def batch_profiles(
    profiles: list[LLMSourceProfileInput],
    batch_size: int,
) -> list[list[LLMSourceProfileInput]]:
    """Split profiles deterministically without dropping a final short batch."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    return [profiles[start : start + batch_size] for start in range(0, len(profiles), batch_size)]


def _read_prompt(path: Path) -> str:
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise LLMAnalysisError(f"Cannot read LLM prompt: {path}") from exc
    if not prompt:
        raise LLMAnalysisError(f"LLM prompt is empty: {path}")
    return prompt


def estimate_input_characters(
    batches: list[list[LLMSourceProfileInput]],
    batch_prompt: str,
    synthesis_prompt: str,
) -> int:
    """Estimate known input plus a conservative allowance for batch insights."""

    known_batch_input = sum(
        len(batch_prompt)
        + len(
            json.dumps(
                [profile.model_dump(mode="json") for profile in batch],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        for batch in batches
    )
    profile_count = sum(len(batch) for batch in batches)
    return (
        known_batch_input
        + len(synthesis_prompt)
        + profile_count * ESTIMATED_INSIGHT_CHARS_PER_PROFILE
    )


def _top_unique(values: list[str], limit: int) -> list[str]:
    counter = Counter(value for value in values if value)
    return [value for value, _ in counter.most_common(limit)]


def _profile_text(profile: LLMSourceProfileInput) -> str:
    parts = [profile.full_name or "", profile.biography or ""]
    for post in profile.recent_posts:
        parts.extend(
            [
                post.caption or "",
                " ".join(post.hashtags),
                post.accessibility_caption or "",
            ]
        )
    return " ".join(parts).casefold()


TOPIC_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("женская мода и стиль", ("мод", "fashion", "стил", "образ", "лук")),
    ("капсульный гардероб", ("капсул", "базовый гардероб")),
    ("стильные подборки и сочетания", ("подборк", "сочет", "гардероб")),
    ("примерки и обзоры одежды", ("примерк", "обзор одеж", "распаков")),
    ("образы для офиса", ("офис", "деловой", "workwear")),
    ("находки на маркетплейсах", ("wildberries", "ozon", "маркетплейс", " wb ")),
    ("beauty и уход", ("beauty", "космет", "макияж", "уход")),
    ("lifestyle и семья", ("lifestyle", "семь", "мам", "дет")),
)


class MockLLMProvider:
    """Transparent deterministic analyzer used offline and in tests."""

    name = "mock"

    def __init__(self, model: str = "mock-fashion-analyzer-v1") -> None:
        self.model = model
        self.retries = 0
        self.usage: dict[str, int] = {}
        self._profiles: list[LLMSourceProfileInput] = []

    def analyze_batch(
        self,
        profiles: list[LLMSourceProfileInput],
        prompt: str,
    ) -> BatchBloggerInsights:
        """Derive repeatable textual signals without a model or network call."""

        del prompt
        self._profiles.extend(profiles)
        topic_values: list[str] = []
        formats: list[str] = []
        tones: list[str] = []
        audience: list[str] = []
        price_signals: list[str] = []
        uncertainty: list[str] = []
        texts = [_profile_text(profile) for profile in profiles]

        for text in texts:
            for label, patterns in TOPIC_PATTERNS:
                if any(pattern in text for pattern in patterns):
                    topic_values.append(label)
            if any(word in text for word in ("честн", "реальн", "без прикрас")):
                tones.append("доверительная подача (inferred из текста)")
            if any(word in text for word in ("девуш", "женск", "для женщин")):
                audience.append("женская аудитория (inferred из текста)")
            if any(word in text for word in ("wildberries", "ozon", "бюджет", "доступн")):
                price_signals.append(
                    "mass-market / доступный сегмент (inferred из текстовых упоминаний)"
                )
            if any(word in text for word in ("premium", "премиум", "люкс", "luxury")):
                price_signals.append("premium-сегмент (inferred из текста)")

        for profile in profiles:
            if not profile.recent_posts:
                uncertainty.append(f"{profile.username}: нет доступных последних публикаций")
            for post in profile.recent_posts:
                post_type = (post.post_type or "").casefold()
                if "video" in post_type or "reel" in post_type:
                    formats.append("короткое вертикальное видео")
                elif "sidecar" in post_type or "carousel" in post_type:
                    formats.append("карусель")
                elif "image" in post_type or "photo" in post_type:
                    formats.append("фотопубликация")
                elif post_type:
                    formats.append(post.post_type or "")

        followers = [p.followers_count for p in profiles if p.followers_count is not None]
        engagement = [
            p.calculated_engagement_rate
            for p in profiles
            if p.calculated_engagement_rate is not None
        ]
        engagement_observations: list[str] = []
        if followers:
            engagement_observations.append(
                f"Наблюдаемый диапазон подписчиков: {min(followers)}–{max(followers)}"
            )
        if engagement:
            engagement_observations.append(
                "Наблюдаемый ER: "
                f"{min(engagement):.2f}–{max(engagement):.2f}%"
            )

        captions = [post.caption or "" for p in profiles for post in p.recent_posts]
        marked_ads = sum(
            bool(re.search(r"(?:^|\s)(?:#реклама|#ad|реклама)(?:\s|$)", caption.casefold()))
            for caption in captions
        )
        advertising = [
            f"Явные рекламные маркеры: {marked_ads} из {len(captions)} доступных captions; "
            "общую нагрузку требуется проверить вручную"
        ]
        all_text = " ".join(texts)
        store_markers = any(
            marker in all_text
            for marker in ("интернет-магазин", "заказ в директ", "доставка по россии", "каталог")
        )
        negative = ["Возможны признаки магазина/витрины"] if store_markers else []
        if not topic_values:
            uncertainty.append("Fashion-тематика не подтверждена доступным текстом")
        if not formats:
            uncertainty.append("Форматы контента не определены")

        evidence_points = sum(
            bool(value)
            for value in (topic_values, formats, engagement_observations, audience, tones)
        )
        confidence = min(0.95, 0.35 + evidence_points * 0.1)
        dominant = _top_unique(topic_values, 3)
        secondary = [value for value in _top_unique(topic_values, 8) if value not in dominant]
        return BatchBloggerInsights(
            analyzed_usernames=[profile.username for profile in profiles],
            dominant_topics=dominant,
            secondary_topics=secondary,
            content_formats=_top_unique(formats, 8),
            tone_patterns=_top_unique(tones, 8),
            audience_signals=_top_unique(audience, 8),
            price_segment_signals=_top_unique(price_signals, 8),
            engagement_observations=engagement_observations,
            advertising_load_signals=advertising,
            brand_safety_observations=[
                "Явных brand-safety рисков в ограниченном тексте не найдено; нужна ручная проверка"
            ],
            positive_patterns=dominant + _top_unique(formats, 2),
            negative_patterns=negative,
            uncertainty_notes=_top_unique(uncertainty, 12),
            confidence_score=confidence,
        )

    def synthesize(
        self,
        insights: list[BatchBloggerInsights],
        prompt: str,
    ) -> LLMIdealBloggerProfile:
        """Aggregate mock batch observations into the final strict model."""

        del prompt
        dominant = _top_unique(
            [value for item in insights for value in item.dominant_topics], 6
        )
        secondary = [
            value
            for value in _top_unique(
                [value for item in insights for value in item.secondary_topics], 8
            )
            if value not in dominant
        ]
        formats = _top_unique(
            [value for item in insights for value in item.content_formats], 8
        )
        tones = _top_unique(
            [value for item in insights for value in item.tone_patterns], 8
        )
        audience = _top_unique(
            [value for item in insights for value in item.audience_signals], 8
        )
        price = _top_unique(
            [value for item in insights for value in item.price_segment_signals], 4
        )
        positive = _top_unique(
            [value for item in insights for value in item.positive_patterns], 10
        )
        negative = _top_unique(
            [value for item in insights for value in item.negative_patterns], 10
        )
        followers = [p.followers_count for p in self._profiles if p.followers_count is not None]
        engagement = [
            p.calculated_engagement_rate
            for p in self._profiles
            if p.calculated_engagement_rate is not None
        ]
        follower_range = (
            f"{min(followers)}–{max(followers)} подписчиков; размер не является главным критерием"
            if followers
            else "Не определён по доступным данным"
        )
        engagement_range = (
            f"{min(engagement):.2f}–{max(engagement):.2f}% по доступным публикациям"
            if engagement
            else "Не определён по доступным данным"
        )
        usernames = [p.username for p in self._profiles]
        query_topics = dominant[:3] or ["женская мода"]
        search_keywords = _top_unique(
            dominant
            + secondary
            + ["fashion-блогер", "женские образы", "примерки одежды"],
            20,
        )
        confidence = (
            sum(item.confidence_score for item in insights) / len(insights)
            if insights
            else 0.0
        )
        return LLMIdealBloggerProfile(
            dominant_topics=dominant,
            secondary_topics=secondary,
            content_formats=formats,
            visual_style_signals=[
                "Визуальный стиль не подтверждён: изображения и видео не передавались"
            ],
            tone_of_voice=tones or ["Не подтверждён доступным текстом"],
            target_audience=audience or [
                "inferred: женщины, интересующиеся одеждой и готовыми образами"
            ],
            audience_interests=dominant + secondary,
            price_segment="; ".join(price) if price else "Не подтверждён; проверить вручную",
            typical_follower_range=follower_range,
            engagement_rate_range=engagement_range,
            advertising_load_preferences=[
                "Умеренная рекламная нагрузка и нативная интеграция; проверить вручную"
            ],
            preferred_integration_formats=formats or [
                "Нативная публикация с честным мнением; формат требует согласования"
            ],
            brand_safety_requirements=[
                "Авторский профиль, а не магазин или каталог",
                "Отсутствие явных репутационных рисков после ручной проверки",
                "Прозрачная маркировка рекламы по применимым требованиям",
            ],
            positive_signals=positive,
            negative_signals=negative,
            exclusion_criteria=[
                "магазин, бренд, каталог или витрина вместо автора",
                "закрытый профиль без доступных данных",
                "отсутствие подтверждаемой fashion-тематики",
                "brand-safety риск, выявленный при ручной проверке",
            ],
            search_keywords=search_keywords,
            search_queries=[
                f"публичный Instagram автор {topic} женские образы"
                for topic in query_topics
            ],
            confidence_score=round(confidence, 3),
            evidence_summary=(
                f"Портрет построен детерминированно по {len(usernames)} пригодным "
                f"профилям в {len(insights)} пакетах. Использованы только biography, "
                "метрики и текст последних публикаций; визуальные выводы не делались. "
                "Результат требует ручной проверки."
            ),
            sample_profile_usernames=usernames[:25],
        )


class OpenAILLMProvider:
    """Responses API provider with Pydantic parsing and explicit safe retries."""

    name = "openai"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        timeout_seconds: float,
        max_retries: int = OPENAI_MAX_RETRIES,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise LLMAnalysisError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                "Use --dry-run-llm to inspect the request plan without a key."
            )
        self.model = model
        self._max_retries = max_retries
        self._sleep = sleep
        self._client = client or OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.retries = 0
        self.usage: dict[str, int] = {}

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(
            exc,
            (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError),
        ):
            return True
        return isinstance(exc, APIStatusError) and exc.status_code >= 500

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        status_code = getattr(exc, "status_code", None)
        suffix = f", HTTP {status_code}" if isinstance(status_code, int) else ""
        return f"{type(exc).__name__}{suffix}"

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        if hasattr(usage, "model_dump"):
            values = usage.model_dump()
        elif isinstance(usage, dict):
            values = usage
        else:
            values = {
                key: getattr(usage, key, None)
                for key in ("input_tokens", "output_tokens", "total_tokens")
            }
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = values.get(key)
            if isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value

    def _parse_response(
        self,
        prompt: str,
        payload: list[dict[str, Any]],
        output_model: type[BaseModel],
    ) -> BaseModel:
        for attempt in range(self._max_retries + 1):
            try:
                LOGGER.info(
                    "Calling OpenAI Responses API: model=%s records=%d attempt=%d",
                    self.model,
                    len(payload),
                    attempt + 1,
                )
                response = self._client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                payload,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    ],
                    text_format=output_model,
                    store=False,
                )
                self._record_usage(response)
                parsed = getattr(response, "output_parsed", None)
                if parsed is None:
                    raise LLMAnalysisError(
                        "OpenAI returned no parsed Structured Output; no response content was logged."
                    )
                return output_model.model_validate(parsed)
            except (ValidationError, LLMAnalysisError):
                raise
            except Exception as exc:
                if self._is_retryable(exc) and attempt < self._max_retries:
                    self.retries += 1
                    delay = 1.0 * (2**attempt)
                    LOGGER.warning(
                        "OpenAI request retry %d/%d after %s; waiting %.1fs",
                        self.retries,
                        self._max_retries,
                        self._safe_error(exc),
                        delay,
                    )
                    self._sleep(delay)
                    continue
                raise LLMAnalysisError(
                    "OpenAI Responses API request failed after "
                    f"{attempt + 1} attempt(s): {self._safe_error(exc)}."
                ) from exc
        raise LLMAnalysisError("OpenAI retry loop ended unexpectedly")

    def analyze_batch(
        self,
        profiles: list[LLMSourceProfileInput],
        prompt: str,
    ) -> BatchBloggerInsights:
        parsed = self._parse_response(
            prompt,
            [profile.model_dump(mode="json") for profile in profiles],
            BatchBloggerInsights,
        )
        return BatchBloggerInsights.model_validate(parsed)

    def synthesize(
        self,
        insights: list[BatchBloggerInsights],
        prompt: str,
    ) -> LLMIdealBloggerProfile:
        parsed = self._parse_response(
            prompt,
            [insight.model_dump(mode="json") for insight in insights],
            LLMIdealBloggerProfile,
        )
        return LLMIdealBloggerProfile.model_validate(parsed)


AnalysisProvider = MockLLMProvider | OpenAILLMProvider


def create_llm_provider(settings: Settings) -> AnalysisProvider:
    """Create the configured provider without logging credentials."""

    if settings.llm_provider == "mock":
        return MockLLMProvider()
    return OpenAILLMProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout_seconds=settings.openai_request_timeout_seconds,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def save_batch_insights(insights: list[BatchBloggerInsights], path: Path) -> None:
    """Atomically persist all successful batches from the current run."""

    _write_json(path, [item.model_dump(mode="json") for item in insights])


def _format_list(values: list[str]) -> str:
    return "\n".join(f"- {value}" for value in values) if values else "- Не определено"


def save_ideal_profile(
    profile: LLMIdealBloggerProfile,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Persist both machine-readable and interview-friendly portrait formats."""

    _write_json(json_path, profile.model_dump(mode="json"))
    sections = (
        ("Доминирующие темы", profile.dominant_topics),
        ("Вторичные темы", profile.secondary_topics),
        ("Форматы", profile.content_formats),
        ("Визуальные сигналы", profile.visual_style_signals),
        ("Тон", profile.tone_of_voice),
        ("Целевая аудитория", profile.target_audience),
        ("Интересы аудитории", profile.audience_interests),
        ("Предпочтительная рекламная нагрузка", profile.advertising_load_preferences),
        ("Форматы интеграции", profile.preferred_integration_formats),
        ("Brand safety", profile.brand_safety_requirements),
        ("Положительные сигналы", profile.positive_signals),
        ("Негативные сигналы", profile.negative_signals),
        ("Критерии исключения", profile.exclusion_criteria),
        ("Поисковые ключи", profile.search_keywords),
        ("Поисковые запросы", profile.search_queries),
        ("Примеры профилей", profile.sample_profile_usernames),
    )
    lines = [
        "# Портрет идеального fashion-блогера LD LATTE",
        "",
        "> Аналитический результат требует ручной проверки перед использованием.",
        "",
        f"**Ценовой сегмент:** {profile.price_segment}",
        f"**Диапазон подписчиков:** {profile.typical_follower_range}",
        f"**Диапазон ER:** {profile.engagement_rate_range}",
        f"**Confidence:** {profile.confidence_score:.2f}",
        "",
        "## Evidence summary",
        "",
        profile.evidence_summary,
    ]
    for title, values in sections:
        lines.extend(("", f"## {title}", "", _format_list(values)))
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_audit_error(exc: Exception) -> str:
    if isinstance(exc, LLMAnalysisError):
        return str(exc)[:2_000]
    return f"{type(exc).__name__}: analysis step failed; content was not logged"


def _audit(
    *,
    settings: Settings,
    profile_count: int,
    batch_count: int,
    duration: float,
    provider: AnalysisProvider | None,
    errors: list[str],
    completed_batches: int,
    dry_run: bool,
) -> LLMAnalysisAudit:
    return LLMAnalysisAudit(
        provider=settings.llm_provider,
        model=settings.openai_model if settings.llm_provider == "openai" else "mock-fashion-analyzer-v1",
        profile_count=profile_count,
        batch_count=batch_count,
        duration=round(duration, 3),
        retries=provider.retries if provider is not None else 0,
        usage=dict(provider.usage) if provider is not None else {},
        errors=errors,
        completed_batches=completed_batches,
        dry_run=dry_run,
        inspection_timestamp=datetime.now(UTC).isoformat(),
    )


def build_ideal_profile_from_enriched(
    settings: Settings,
    *,
    dry_run: bool = False,
    provider: AnalysisProvider | None = None,
) -> LLMAnalysisRun:
    """Prepare, batch, analyze, synthesize, and persist the source portrait."""

    started = time.monotonic()
    profiles = load_llm_profiles(
        settings.enriched_source_json_path,
        max_profiles=settings.openai_max_total_profiles,
        max_posts=settings.openai_max_posts_per_profile,
    )
    batches = batch_profiles(profiles, settings.openai_max_profiles_per_batch)
    batch_prompt = _read_prompt(settings.llm_batch_prompt_path)
    synthesis_prompt = _read_prompt(settings.llm_synthesis_prompt_path)
    model = settings.openai_model if settings.llm_provider == "openai" else "mock-fashion-analyzer-v1"
    summary = LLMDryRunSummary(
        profile_count=len(profiles),
        batch_count=len(batches),
        approximate_characters=estimate_input_characters(
            batches,
            batch_prompt,
            synthesis_prompt,
        ),
        provider=settings.llm_provider,
        model=model,
    )
    if dry_run:
        audit = _audit(
            settings=settings,
            profile_count=len(profiles),
            batch_count=len(batches),
            duration=time.monotonic() - started,
            provider=None,
            errors=[],
            completed_batches=0,
            dry_run=True,
        )
        _write_json(settings.llm_analysis_audit_path, audit.model_dump(mode="json"))
        return LLMAnalysisRun(summary, [], None, audit)

    successful: list[BatchBloggerInsights] = []
    save_batch_insights(successful, settings.llm_batch_insights_path)
    try:
        active_provider = provider or create_llm_provider(settings)
    except Exception as exc:
        audit = _audit(
            settings=settings,
            profile_count=len(profiles),
            batch_count=len(batches),
            duration=time.monotonic() - started,
            provider=None,
            errors=[_safe_audit_error(exc)],
            completed_batches=0,
            dry_run=False,
        )
        _write_json(settings.llm_analysis_audit_path, audit.model_dump(mode="json"))
        raise
    try:
        for index, batch in enumerate(batches, start=1):
            LOGGER.info(
                "Analyzing LLM batch %d/%d with %d profiles via %s",
                index,
                len(batches),
                len(batch),
                active_provider.name,
            )
            insight = active_provider.analyze_batch(batch, batch_prompt)
            expected_usernames = [profile.username for profile in batch]
            if insight.analyzed_usernames != expected_usernames:
                raise LLMAnalysisError(
                    f"Batch {index} returned an unexpected analyzed_usernames list."
                )
            successful.append(insight)
            save_batch_insights(successful, settings.llm_batch_insights_path)
        ideal_profile = active_provider.synthesize(successful, synthesis_prompt)
        save_ideal_profile(
            ideal_profile,
            settings.ideal_blogger_profile_json_path,
            settings.ideal_blogger_profile_markdown_path,
        )
    except Exception as exc:
        audit = _audit(
            settings=settings,
            profile_count=len(profiles),
            batch_count=len(batches),
            duration=time.monotonic() - started,
            provider=active_provider,
            errors=[_safe_audit_error(exc)],
            completed_batches=len(successful),
            dry_run=False,
        )
        _write_json(settings.llm_analysis_audit_path, audit.model_dump(mode="json"))
        raise

    audit = _audit(
        settings=settings,
        profile_count=len(profiles),
        batch_count=len(batches),
        duration=time.monotonic() - started,
        provider=active_provider,
        errors=[],
        completed_batches=len(successful),
        dry_run=False,
    )
    _write_json(settings.llm_analysis_audit_path, audit.model_dump(mode="json"))
    return LLMAnalysisRun(summary, successful, ideal_profile, audit)

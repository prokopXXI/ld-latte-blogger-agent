"""Build an explainable ideal fashion-blogger portrait without an LLM."""

import logging
import re
from collections import Counter
from collections.abc import Iterable
from statistics import mean

from src.models import (
    BrandSafetyLevel,
    IdealBloggerProfile,
    Platform,
    PriceSegment,
    SourceBlogger,
)


LOGGER = logging.getLogger(__name__)


def _most_common(values: Iterable[str], limit: int) -> list[str]:
    """Return frequent values with source order as a deterministic tie-breaker."""

    items = [value.strip().casefold() for value in values if value.strip()]
    counts = Counter(items)
    first_position = {value: items.index(value) for value in counts}
    ordered = sorted(counts, key=lambda value: (-counts[value], first_position[value]))
    return ordered[:limit]


def _audience_traits(bloggers: list[SourceBlogger]) -> list[str]:
    descriptions = " ".join(item.audience_description.casefold() for item in bloggers)
    ages = [int(value) for value in re.findall(r"\b(?:1[89]|[2-5]\d)\b", descriptions)]
    common_interests = _most_common(
        (
            interest
            for blogger in bloggers
            for interest in blogger.audience_interests
        ),
        limit=5,
    )

    traits: list[str] = []
    if ages:
        traits.append(f"женщины {min(ages)}–{max(ages)} лет")
    else:
        traits.append("женская аудитория")
    if "город" in descriptions:
        traits.append("жительницы крупных городов")
    traits.append(f"интересы: {', '.join(common_interests)}")
    return traits


def _preferred_integrations(content_formats: list[str]) -> list[str]:
    format_descriptions = {
        "reels": "нативный Reel с готовым образом",
        "примерки": "примерка с честным мнением",
        "карусели": "карусель с вариантами сочетаний",
        "stories": "серия Stories с деталями образа",
        "shorts": "YouTube Shorts со стилизацией",
        "публикации": "публикация-подборка",
        "фотоподборки": "фотоподборка комплектов",
        "гайды": "гайд по сочетанию вещей",
    }
    return [format_descriptions.get(item, item) for item in content_formats]


def build_ideal_profile(bloggers: list[SourceBlogger]) -> IdealBloggerProfile:
    """Aggregate fashion attributes and numeric benchmarks into one model."""

    if not bloggers:
        raise ValueError("At least one source blogger is required to build a profile")

    content_topics = _most_common(
        (topic for blogger in bloggers for topic in blogger.content_topics),
        limit=15,
    )
    content_formats = _most_common(
        (item for blogger in bloggers for item in blogger.content_formats),
        limit=10,
    )
    tones = _most_common(
        (tone for blogger in bloggers for tone in blogger.tone),
        limit=10,
    )
    visual_styles = _most_common(
        (style for blogger in bloggers for style in blogger.visual_style),
        limit=10,
    )
    platforms = [
        Platform(value)
        for value in _most_common((blogger.platform.value for blogger in bloggers), limit=4)
    ]
    price_segment = PriceSegment(
        _most_common((blogger.price_segment.value for blogger in bloggers), limit=1)[0]
    )
    brand_safety = BrandSafetyLevel(
        _most_common((blogger.brand_safety.value for blogger in bloggers), limit=1)[0]
    )
    engagement_rate = round(mean(item.engagement_rate_pct for item in bloggers), 2)
    advertising_load = round(mean(item.advertising_load_pct for item in bloggers), 2)

    profile = IdealBloggerProfile(
        summary=(
            "Fashion-блогер о женской одежде и практичной стилизации с "
            f"форматами «{', '.join(content_formats[:4])}», визуалом "
            f"«{', '.join(visual_styles[:3])}» и средней вовлечённостью "
            f"{engagement_rate:.2f}%."
        ),
        content_topics=content_topics,
        content_formats=content_formats,
        visual_style=visual_styles,
        tone=tones,
        target_audience=_audience_traits(bloggers),
        price_segment=price_segment,
        engagement_rate_pct=engagement_rate,
        advertising_load_pct=advertising_load,
        brand_safety=brand_safety,
        preferred_integration_formats=_preferred_integrations(content_formats),
        target_platforms=platforms,
        followers_min=min(item.followers for item in bloggers),
        followers_max=max(item.followers for item in bloggers),
        preferred_locations=_most_common((item.location for item in bloggers), limit=20),
        must_have_traits=[
            "регулярный контент о женской одежде",
            "естественная и доверительная подача",
            "понятные примеры сочетания вещей",
            "проверяемые публичные метрики",
        ],
        red_flags=[
            "рекламная нагрузка выше 60%",
            "низкий уровень brand safety",
            "отсутствие fashion-тематики",
            "несовпадение с целевым ценовым сегментом",
        ],
    )
    LOGGER.info(
        "Built ideal fashion profile: topics=%s, engagement=%.2f%%",
        ", ".join(profile.content_topics[:4]),
        profile.engagement_rate_pct,
    )
    return profile

"""Generate deterministic Russian search queries from the ideal profile."""

import json
from pathlib import Path

from src.models import IdealBloggerProfile, Platform, PriceSegment


PLATFORM_LABELS = {
    Platform.INSTAGRAM: "Instagram",
    Platform.YOUTUBE_SHORTS: "YouTube Shorts",
    Platform.TELEGRAM: "Telegram",
    Platform.OTHER: "публичный блог",
}

PRICE_LABELS = {
    PriceSegment.MASS_MARKET: "доступный ценовой сегмент",
    PriceSegment.MIDDLE: "средний ценовой сегмент",
    PriceSegment.PREMIUM: "премиальный ценовой сегмент",
    PriceSegment.MIXED: "смешанный ценовой сегмент",
}


def generate_search_queries(profile: IdealBloggerProfile) -> list[str]:
    """Compose eight queries using topics, style, audience, formats, and platforms."""

    topics = profile.content_topics
    formats = profile.content_formats
    visuals = profile.visual_style
    platforms = [PLATFORM_LABELS[item] for item in profile.target_platforms]
    audience = profile.target_audience[0]
    price = PRICE_LABELS[profile.price_segment]

    def item(values: list[str], index: int) -> str:
        return values[index % len(values)]

    queries = [
        f"fashion-блогер {item(topics, 0)} {item(platforms, 0)}",
        f"блогер {item(topics, 1)} {item(formats, 0)} {item(platforms, 1)}",
        f"{item(topics, 2)} {audience} fashion-блог {item(platforms, 2)}",
        f"{item(topics, 3)} {item(formats, 1)} блогер {item(platforms, 0)}",
        f"{item(topics, 4)} женская одежда {price} {item(platforms, 1)}",
        f"{item(visuals, 0)} визуальный стиль {item(topics, 0)} блогер",
        f"визуальный стиль {item(visuals, 1)} {item(formats, 2)} {audience} блогер",
        f"{item(topics, 0)} {item(topics, 1)} {item(formats, 3)} {price} блогер",
    ]
    return list(dict.fromkeys(query.strip() for query in queries))[:10]


def save_search_queries(queries: list[str], path: Path) -> None:
    """Persist the exact queries used by the current run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"count": len(queries), "queries": queries}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

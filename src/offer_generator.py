"""Generate deterministic personalized fashion barter offers in mock mode."""

import logging

from src.models import BarterOffer, RankedCandidate, SourceBlogger


LOGGER = logging.getLogger(__name__)

PLATFORM_INTEGRATIONS = {
    "instagram": "нативный Reel с собранным образом или серию Stories с примеркой",
    "youtube_shorts": "YouTube Shorts с примеркой и несколькими вариантами стилизации",
    "telegram": "публикацию-подборку с сочетаниями вещей и честным мнением",
    "other": "нативный материал в привычном для вашей аудитории формате",
}


def _strongest_fashion_match(ranked: RankedCandidate) -> str:
    score = ranked.evaluation.score
    normalized_criteria = {
        "совпадение fashion-тематики": score.topic_score / 20,
        "визуальная совместимость": score.visual_score / 20,
        "релевантность женской аудитории": score.audience_score / 15,
        "естественная подача": score.tone_score / 10,
        "подходящие форматы контента": score.format_score / 5,
        "совпадение ценового сегмента": score.price_segment_score / 5,
    }
    return max(normalized_criteria, key=normalized_criteria.get)


def generate_offer(
    ranked: RankedCandidate,
    source_bloggers: list[SourceBlogger],
) -> BarterOffer:
    """Build one fashion offer only from facts in the validated profile."""

    source_by_name = {source.display_name: source for source in source_bloggers}
    similar_source = source_by_name.get(ranked.evaluation.similar_to)
    if similar_source is None:
        raise ValueError(
            f"Source blogger {ranked.evaluation.similar_to!r} was not found for offer generation"
        )

    candidate = ranked.candidate
    main_topic = (candidate.content_topics or ["женская одежда"])[0]
    visual_description = ", ".join((candidate.visual_style or ["не указан в выдаче"])[:2])
    audience_interests = ", ".join(
        (candidate.audience_interests or ["не указаны в выдаче"])[:2]
    )
    content_feature = candidate.content_style or candidate.source_title or "авторская подача"
    integration_format = PLATFORM_INTEGRATIONS[candidate.platform.value]
    strongest_match = _strongest_fashion_match(ranked)
    proposed_barter = (
        "обсудить подходящий образ из ассортимента LD LATTE и согласовать "
        "бартерный формат с правом на честное авторское мнение."
    )
    message = (
        f"Здравствуйте, {candidate.display_name}! Обратили внимание на ваш контент "
        f"о теме «{main_topic}» и особенность подачи — {content_feature}. "
        f"Визуальный стиль профиля — {visual_description}; среди интересов аудитории — "
        f"{audience_interests}. Для бренда женской одежды LD LATTE особенно релевантно "
        f"следующее: {strongest_match}. По данным нашей эталонной базы ваш профиль "
        f"ближе всего к «{similar_source.display_name}». Предлагаем {proposed_barter} "
        f"Возможный формат интеграции: "
        f"{integration_format}. Конкретный образ, состав бартера и редакционные детали "
        f"предлагаем выбрать вместе, если сотрудничество вам интересно."
    )
    return BarterOffer(
        candidate_handle=candidate.handle,
        subject=f"Идея fashion-сотрудничества LD LATTE для темы «{main_topic}»",
        message=message,
        proposed_barter=proposed_barter,
        personalization_facts=[
            f"тематика: {main_topic}",
            f"особенность контента: {content_feature}",
            f"визуальный стиль: {visual_description}",
            f"интересы аудитории: {audience_interests}",
            f"формат: {integration_format}",
            f"наиболее похожий эталон: {similar_source.display_name}",
        ],
    )


def generate_offers(
    ranked_candidates: list[RankedCandidate],
    source_bloggers: list[SourceBlogger],
) -> list[BarterOffer]:
    """Generate one independently personalized offer per selected candidate."""

    offers = [generate_offer(candidate, source_bloggers) for candidate in ranked_candidates]
    LOGGER.info("Generated %d deterministic fashion barter offers", len(offers))
    return offers

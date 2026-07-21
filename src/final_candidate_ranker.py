"""Confidence-aware deterministic scoring for real discovery candidates."""

from __future__ import annotations

import re

from src.models import (
    FinalScoreBreakdown,
    FinalScoredCandidate,
    LLMIdealBloggerProfile,
    RealCandidateProfile,
    ScoreCriterionDetail,
)


FASHION_GROUPS = (
    ("женск", "women", "женствен"),
    ("fashion", "мод", "стил"),
    ("одежд", "плать", "гардероб", "вещ"),
    ("образ", "лук", "outfit", "капсул", "офис"),
    ("пример", "try-on", "обзор", "подборк", "ugc", "unboxing"),
)
VISUAL_TEXT_GROUPS = (
    ("на себе", "on-body", "пример"),
    ("естествен", "natural", "без фильтр", "честн"),
    ("эстет", "aesthetic", "женствен"),
    ("детал", "посадк", "сочет", "стилизац"),
)
AUDIENCE_GROUPS = (
    ("женщ", "девуш", "female"),
    ("wildberries", "ozon", "маркетплейс", " вб "),
    ("доступн", "бюджет", "affordable", "находк"),
    ("покуп", "шоп", "shopping", "артикул"),
)
TONE_GROUPS = (
    ("честн", "довер", "без прикрас"),
    ("личн", "рассказыва", "показываю", "со мной"),
    ("практич", "совет", "как сочет", "подбор"),
    ("авторск", "куратор", "curatorial"),
)
PRICE_MARKERS = (
    "wildberries",
    "ozon",
    "маркетплейс",
    "доступн",
    "бюджет",
    "находк",
    "артикул",
)
RISK_MARKERS = (
    "казино",
    "ставки",
    "наркот",
    "hate",
    "экстрем",
    "18+",
)
AD_PATTERN = re.compile(r"(?:#реклама|#ad\b|реклама\b|партнерск|партнёрск)", re.I)
INSUFFICIENT_REASON = "Недостаточно данных для уверенной оценки. confidence: low"


def _candidate_text(candidate: RealCandidateProfile) -> str:
    parts = [
        candidate.name,
        candidate.title or "",
        candidate.snippet or "",
        candidate.biography or "",
        " ".join(candidate.content_formats),
    ]
    for post in candidate.recent_posts:
        parts.extend(
            (
                post.caption or "",
                " ".join(post.hashtags),
                post.post_type or "",
                post.accessibility_caption or "",
            )
        )
    return " ".join(parts).casefold().replace("ё", "е")


def _group_points(text: str, groups: tuple[tuple[str, ...], ...], maximum: int) -> int:
    matches = sum(any(marker in text for marker in group) for group in groups)
    return round(matches / len(groups) * maximum)


def _engagement_points(rate: float | None) -> int:
    if rate is None:
        return 0
    if rate > 50:
        return 1
    if 2 <= rate <= 8:
        return 10
    if 1 <= rate < 2 or 8 < rate <= 15:
        return 7
    if 0.3 <= rate < 1 or 15 < rate <= 30:
        return 4
    return 2


def _advertising_points(candidate: RealCandidateProfile) -> int:
    captions = [post.caption or "" for post in candidate.recent_posts]
    if not captions:
        return 0
    ad_count = sum(bool(AD_PATTERN.search(caption)) for caption in captions)
    if ad_count == 0:
        return 6
    ratio = ad_count / len(captions)
    if ratio <= 1 / 3:
        return 4
    if ratio < 1:
        return 2
    return 0


def _format_points(candidate: RealCandidateProfile, ideal: LLMIdealBloggerProfile) -> int:
    if not candidate.content_formats:
        return 0
    candidate_text = " ".join(candidate.content_formats).casefold()
    ideal_text = " ".join(
        ideal.content_formats + ideal.preferred_integration_formats
    ).casefold()
    markers = ("video", "reel", "short", "carousel", "sidecar", "image", "публикац")
    overlap = sum(marker in candidate_text and marker in ideal_text for marker in markers)
    return min(5, max(1, overlap * 2))


def _brand_safety_points(candidate: RealCandidateProfile, text: str) -> int:
    if any(marker in text for marker in RISK_MARKERS):
        return 0
    if candidate.is_private is True:
        return 0
    if candidate.enrichment_status == "failed":
        return 1
    if candidate.recent_posts:
        return 4
    return 2


def _adjust(raw_score: int, factor: float, maximum: int) -> int:
    return max(0, min(maximum, round(raw_score * factor)))


def _matched_groups(
    text: str,
    groups: tuple[tuple[str, ...], ...],
) -> list[str]:
    return [
        next(marker for marker in group if marker in text)
        for group in groups
        if any(marker in text for marker in group)
    ]


def _adjustment_note(raw_score: int, adjusted_score: int, factor: float) -> str:
    if raw_score == adjusted_score:
        return ""
    return (
        f" Исходные сигналы дали {raw_score} баллов; после коэффициента "
        f"полноты данных {factor:.2f} сохранено {adjusted_score}."
    )


def _criterion_details(
    candidate: RealCandidateProfile,
    text: str,
    raw: dict[str, int],
    adjusted: dict[str, int],
    confidence_factor: float,
) -> dict[str, ScoreCriterionDetail]:
    """Explain unchanged score components without introducing new inference."""

    topic_matches = _matched_groups(text, FASHION_GROUPS)
    visual_matches = _matched_groups(text, VISUAL_TEXT_GROUPS)
    audience_matches = _matched_groups(text, AUDIENCE_GROUPS)
    tone_matches = _matched_groups(text, TONE_GROUPS)
    topic_reason = (
        f"Подтверждены fashion-сигналы: {', '.join(topic_matches)}."
        + _adjustment_note(
            raw["fashion_relevance_score"],
            adjusted["fashion_relevance_score"],
            confidence_factor,
        )
        if topic_matches
        else INSUFFICIENT_REASON
    )
    visual_reason = (
        "В текстах подтверждены визуальные сигналы: "
        f"{', '.join(visual_matches)}. Изображения автоматически не анализировались."
        + _adjustment_note(
            raw["visual_text_score"],
            adjusted["visual_text_score"],
            confidence_factor,
        )
        if visual_matches
        else INSUFFICIENT_REASON
    )
    audience_reason = (
        f"В публичных текстах найдены сигналы аудитории: {', '.join(audience_matches)}. "
        "Демография требует ручной проверки."
        + _adjustment_note(
            raw["audience_score"],
            adjusted["audience_score"],
            confidence_factor,
        )
        if audience_matches
        else INSUFFICIENT_REASON
    )
    tone_reason = (
        f"Тон подтверждён текстовыми маркерами: {', '.join(tone_matches)}."
        + _adjustment_note(
            raw["tone_score"],
            adjusted["tone_score"],
            confidence_factor,
        )
        if tone_matches
        else INSUFFICIENT_REASON
    )
    if candidate.engagement_rate is None:
        engagement_reason = INSUFFICIENT_REASON
        engagement_confidence = "low"
    elif candidate.engagement_rate > 50:
        engagement_reason = (
            f"ER={candidate.engagement_rate:.2f}% выглядит аномальным, поэтому оценка ограничена."
        )
        engagement_confidence = "medium"
    else:
        engagement_reason = (
            f"Использован подтверждённый ER={candidate.engagement_rate:.2f}%; "
            "максимум даётся диапазону 2–8%."
            + _adjustment_note(
                raw["engagement_score"],
                adjusted["engagement_score"],
                confidence_factor,
            )
        )
        engagement_confidence = "high"
    captions = [post.caption or "" for post in candidate.recent_posts]
    if not captions:
        ad_reason = INSUFFICIENT_REASON
        ad_confidence = "low"
    else:
        ad_count = sum(bool(AD_PATTERN.search(caption)) for caption in captions)
        ad_reason = (
            f"В {len(captions)} последних публикациях найдено явных рекламных "
            f"маркеров: {ad_count}."
            + _adjustment_note(
                raw["advertising_load_score"],
                adjusted["advertising_load_score"],
                confidence_factor,
            )
        )
        ad_confidence = "high" if len(captions) >= 3 else "medium"
    price_matches = [marker for marker in PRICE_MARKERS if marker in text]
    price_reason = (
        "Ценовой сегмент поддерживают маркеры: "
        f"{', '.join(price_matches[:6])}."
        + _adjustment_note(
            raw["price_segment_score"],
            adjusted["price_segment_score"],
            confidence_factor,
        )
        if price_matches
        else INSUFFICIENT_REASON
    )
    if candidate.content_formats:
        format_reason = (
            "Подтверждённые форматы: "
            f"{', '.join(candidate.content_formats)}."
            + _adjustment_note(
                raw["content_format_score"],
                adjusted["content_format_score"],
                confidence_factor,
            )
        )
        format_confidence = "high" if candidate.recent_posts else "medium"
    else:
        format_reason = INSUFFICIENT_REASON
        format_confidence = "low"
    risks = [marker for marker in RISK_MARKERS if marker in text]
    if risks:
        safety_reason = f"Найдены risk-маркеры: {', '.join(risks)}."
        safety_confidence = "high"
    elif candidate.is_private is True:
        safety_reason = "Профиль закрыт; brand safety нельзя подтвердить по публикациям. confidence: low"
        safety_confidence = "low"
    elif candidate.enrichment_status == "failed":
        safety_reason = "Enrichment не завершён; brand safety подтверждён недостаточно. confidence: low"
        safety_confidence = "low"
    elif candidate.recent_posts:
        safety_reason = (
            f"В {len(candidate.recent_posts)} доступных публикациях явные risk-маркеры не найдены."
            + _adjustment_note(
                raw["brand_safety_score"],
                adjusted["brand_safety_score"],
                confidence_factor,
            )
        )
        safety_confidence = "medium"
    else:
        safety_reason = INSUFFICIENT_REASON
        safety_confidence = "low"
    data_points = adjusted["data_confidence_score"]
    confidence_reason = (
        f"data_confidence={candidate.data_confidence:.3f} даёт {data_points}/5. "
        f"Автор подтверждён с confidence={candidate.author_resolution_confidence:.2f}; "
        f"evidence_count={candidate.evidence_count}."
    )
    return {
        "topic": ScoreCriterionDetail(
            score=adjusted["fashion_relevance_score"],
            max_score=20,
            reason=topic_reason,
            confidence="high" if candidate.recent_posts else ("medium" if topic_matches else "low"),
        ),
        "visual": ScoreCriterionDetail(
            score=adjusted["visual_text_score"],
            max_score=15,
            reason=visual_reason,
            confidence="medium" if visual_matches else "low",
        ),
        "audience": ScoreCriterionDetail(
            score=adjusted["audience_score"],
            max_score=15,
            reason=audience_reason,
            confidence="medium" if audience_matches else "low",
        ),
        "tone": ScoreCriterionDetail(
            score=adjusted["tone_score"],
            max_score=10,
            reason=tone_reason,
            confidence="medium" if tone_matches else "low",
        ),
        "engagement": ScoreCriterionDetail(
            score=adjusted["engagement_score"],
            max_score=10,
            reason=engagement_reason,
            confidence=engagement_confidence,
        ),
        "ad_load": ScoreCriterionDetail(
            score=adjusted["advertising_load_score"],
            max_score=10,
            reason=ad_reason,
            confidence=ad_confidence,
        ),
        "price_segment": ScoreCriterionDetail(
            score=adjusted["price_segment_score"],
            max_score=5,
            reason=price_reason,
            confidence="medium" if price_matches else "low",
        ),
        "format": ScoreCriterionDetail(
            score=adjusted["content_format_score"],
            max_score=5,
            reason=format_reason,
            confidence=format_confidence,
        ),
        "brand_safety": ScoreCriterionDetail(
            score=adjusted["brand_safety_score"],
            max_score=5,
            reason=safety_reason,
            confidence=safety_confidence,
        ),
        "data_confidence": ScoreCriterionDetail(
            score=data_points,
            max_score=5,
            reason=confidence_reason,
            confidence=(
                "high"
                if candidate.data_confidence >= 0.8
                else "medium" if candidate.data_confidence >= 0.6 else "low"
            ),
        ),
    }


def score_real_candidate(
    candidate: RealCandidateProfile,
    ideal: LLMIdealBloggerProfile,
) -> FinalScoredCandidate:
    """Score only supported signals and make the confidence penalty visible."""

    text = _candidate_text(candidate)
    raw = {
        "fashion_relevance_score": _group_points(text, FASHION_GROUPS, 20),
        "visual_text_score": _group_points(text, VISUAL_TEXT_GROUPS, 15),
        "audience_score": _group_points(text, AUDIENCE_GROUPS, 15),
        "tone_score": _group_points(text, TONE_GROUPS, 10),
        "engagement_score": _engagement_points(candidate.engagement_rate),
        "advertising_load_score": _advertising_points(candidate),
        "price_segment_score": 5 if any(marker in text for marker in PRICE_MARKERS) else 0,
        "content_format_score": _format_points(candidate, ideal),
        "brand_safety_score": _brand_safety_points(candidate, text),
    }
    confidence_factor = 0.55 + 0.45 * candidate.data_confidence
    maximums = {
        "fashion_relevance_score": 20,
        "visual_text_score": 15,
        "audience_score": 15,
        "tone_score": 10,
        "engagement_score": 10,
        "advertising_load_score": 10,
        "price_segment_score": 5,
        "content_format_score": 5,
        "brand_safety_score": 5,
    }
    adjusted = {
        key: _adjust(value, confidence_factor, maximums[key])
        for key, value in raw.items()
    }
    adjusted["data_confidence_score"] = round(candidate.data_confidence * 5)
    total = sum(adjusted.values())
    breakdown = FinalScoreBreakdown(**adjusted, total_score=total)
    criterion_details = _criterion_details(
        candidate,
        text,
        raw,
        adjusted,
        confidence_factor,
    )

    strengths: list[str] = []
    if raw["fashion_relevance_score"] >= 12:
        strengths.append("есть подтверждённые fashion-сигналы")
    if candidate.engagement_rate is not None and candidate.engagement_rate <= 50:
        strengths.append(f"доступен ER {candidate.engagement_rate:.2f}%")
    if candidate.content_formats:
        strengths.append("известны форматы контента")
    limitations: list[str] = []
    if candidate.followers_count is None:
        limitations.append("нет подтверждённого числа подписчиков")
    if candidate.engagement_rate is None:
        limitations.append("нет подтверждённого ER")
    elif candidate.engagement_rate > 50:
        limitations.append("ER выглядит аномальным")
    if not candidate.recent_posts:
        limitations.append("нет структурированных последних публикаций")
    if candidate.data_confidence < 0.6:
        limitations.append("низкая полнота данных снижает каждый критерий")
    if candidate.author_resolution_confidence < 0.8:
        limitations.append(
            "уверенность определения автора ограничена "
            f"({candidate.author_resolution_confidence:.2f})"
        )
    reason = (
        f"Итог {total}/100 после коэффициента полноты {confidence_factor:.2f} "
        f"(data_confidence={candidate.data_confidence:.2f}). "
        f"Автор подтверждён с confidence="
        f"{candidate.author_resolution_confidence:.2f} по "
        f"{candidate.evidence_count} независимым результатам. "
        f"Сильные стороны: {', '.join(strengths) if strengths else 'ограничены'}. "
        f"Ограничения: {', '.join(limitations) if limitations else 'явных нет'}; "
        "обязательна ручная проверка."
    )
    evidence = list(candidate.evidence[:12])
    if candidate.followers_count is not None:
        evidence.append(f"followers_count={candidate.followers_count}")
    if candidate.engagement_rate is not None:
        evidence.append(f"engagement_rate={candidate.engagement_rate:.4f}%")
    if not evidence:
        evidence.append("Доступна только публичная карточка поискового результата")
    return FinalScoredCandidate(
        candidate=candidate,
        score=breakdown,
        match_reason=reason,
        evidence=evidence[:20],
        criterion_details=criterion_details,
    )


def rank_real_candidates(
    candidates: list[RealCandidateProfile],
    ideal: LLMIdealBloggerProfile,
) -> list[FinalScoredCandidate]:
    """Return stable descending order with URL as a deterministic tiebreaker."""

    scored = [score_real_candidate(candidate, ideal) for candidate in candidates]
    return sorted(
        scored,
        key=lambda item: (
            -item.score.total_score,
            -item.candidate.data_confidence,
            str(item.candidate.profile_url),
        ),
    )


def select_finalists(
    ranked: list[FinalScoredCandidate],
    min_score: int,
    maximum: int,
) -> list[FinalScoredCandidate]:
    """Never fill the shortlist with profiles below the configured threshold."""

    return [item for item in ranked if item.score.total_score >= min_score][:maximum]

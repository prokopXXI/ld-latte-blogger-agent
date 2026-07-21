"""Deterministic fashion-relevant candidate scoring for the mock MVP."""

import logging
import re
from collections.abc import Iterable

from src.models import (
    BrandSafetyLevel,
    CandidateEvaluation,
    CandidateProfile,
    IdealBloggerProfile,
    PriceSegment,
    RankedCandidate,
    ScoreCriterionDetail,
    ScoreBreakdown,
    SourceBlogger,
)


LOGGER = logging.getLogger(__name__)
WORD_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
STOP_WORDS = {
    "аудитория",
    "блогер",
    "для",
    "женщины",
    "женщин",
    "интерес",
    "интересующиеся",
    "ищущие",
    "которым",
    "прежде",
    "широкая",
}


def _normalize(value: str) -> str:
    return " ".join(WORD_RE.findall(value.casefold().replace("ё", "е")))


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in _normalize(value).split()
        if len(token) > 2 and token not in STOP_WORDS
    }


def _phrase_similarity(left: str, right: str) -> float:
    if _normalize(left) == _normalize(right):
        return 1.0
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _list_similarity(candidate_values: list[str], target_values: list[str]) -> float:
    if not candidate_values or not target_values:
        return 0.0
    best_matches = [
        max(_phrase_similarity(value, target) for target in target_values)
        for value in candidate_values
    ]
    return sum(best_matches) / len(best_matches)


def _tag_similarity(candidate_tags: Iterable[str], target_tags: Iterable[str]) -> float:
    candidate = {_normalize(value) for value in candidate_tags if _normalize(value)}
    target = {_normalize(value) for value in target_tags if _normalize(value)}
    if not candidate or not target:
        return 0.0
    return len(candidate & target) / len(candidate)


def _age_range(description: str) -> tuple[int, int] | None:
    ages = [int(value) for value in re.findall(r"\b(?:1[89]|[2-5]\d)\b", description)]
    return (min(ages), max(ages)) if ages else None


def _age_similarity(left: str, right: str) -> float:
    left_range = _age_range(left)
    right_range = _age_range(right)
    if left_range is None or right_range is None:
        return 0.5
    intersection = max(0, min(left_range[1], right_range[1]) - max(left_range[0], right_range[0]))
    union = max(left_range[1], right_range[1]) - min(left_range[0], right_range[0])
    return 1.0 if union == 0 else intersection / union


def _female_audience_marker(description: str) -> bool:
    normalized = description.casefold()
    return "женщ" in normalized or "девуш" in normalized


def _audience_similarity(
    left_description: str,
    left_interests: list[str],
    right_description: str,
    right_interests: list[str],
) -> float:
    if _female_audience_marker(left_description) == _female_audience_marker(right_description):
        gender_score = 1.0
    else:
        gender_score = 0.5
    return (
        0.35 * _age_similarity(left_description, right_description)
        + 0.15 * gender_score
        + 0.50 * _list_similarity(left_interests, right_interests)
    )


def _points(similarity: float, maximum: int) -> int:
    return round(max(0.0, min(1.0, similarity)) * maximum)


def _engagement_points(rate: float | None, benchmark: float) -> int:
    if rate is None:
        return 0
    if benchmark <= 0:
        return 10
    return _points(rate / benchmark, 10)


def _advertising_load_points(load_pct: float | None) -> int:
    if load_pct is None:
        return 0
    if load_pct <= 20:
        return 10
    if load_pct >= 60:
        return 0
    return round((60 - load_pct) / 40 * 10)


def _price_similarity(candidate: PriceSegment | None, target: PriceSegment) -> float:
    if candidate is None:
        return 0.0
    if candidate == target:
        return 1.0
    if PriceSegment.MIXED in {candidate, target}:
        return 0.75
    if {candidate, target} == {PriceSegment.MASS_MARKET, PriceSegment.MIDDLE}:
        return 0.7
    if {candidate, target} == {PriceSegment.MIDDLE, PriceSegment.PREMIUM}:
        return 0.5
    return 0.2


def _brand_safety_similarity(level: BrandSafetyLevel | None) -> float:
    if level is None:
        return 0.0
    return {
        BrandSafetyLevel.HIGH: 1.0,
        BrandSafetyLevel.MEDIUM: 0.6,
        BrandSafetyLevel.LOW: 0.0,
    }[level]


def _similarity_to_source(candidate: CandidateProfile, source: SourceBlogger) -> float:
    topic = _list_similarity(candidate.content_topics or [], source.content_topics)
    visual = _tag_similarity(candidate.visual_style or [], source.visual_style)
    audience = _audience_similarity(
        candidate.audience_description or "",
        candidate.audience_interests or [],
        source.audience_description,
        source.audience_interests,
    )
    tone = _tag_similarity(candidate.tone or [], source.tone)
    engagement = (
        max(
            0.0,
            1.0
            - abs(candidate.engagement_rate_pct - source.engagement_rate_pct)
            / max(source.engagement_rate_pct, 1.0),
        )
        if candidate.engagement_rate_pct is not None
        else 0.0
    )
    advertising_load = (
        max(
            0.0,
            1.0 - abs(candidate.advertising_load_pct - source.advertising_load_pct) / 100,
        )
        if candidate.advertising_load_pct is not None
        else 0.0
    )
    price = _price_similarity(candidate.price_segment, source.price_segment)
    content_format = _tag_similarity(candidate.content_formats or [], source.content_formats)
    brand_safety = _brand_safety_similarity(candidate.brand_safety)
    return (
        0.20 * topic
        + 0.20 * visual
        + 0.15 * audience
        + 0.10 * tone
        + 0.10 * engagement
        + 0.10 * advertising_load
        + 0.05 * price
        + 0.05 * content_format
        + 0.05 * brand_safety
    )


def _nearest_source(
    candidate: CandidateProfile,
    source_bloggers: list[SourceBlogger],
) -> SourceBlogger:
    if not source_bloggers:
        raise ValueError("At least one source blogger is required for scoring")
    return max(source_bloggers, key=lambda source: _similarity_to_source(candidate, source))


def _match_reason(score: ScoreBreakdown, similar_to: str) -> str:
    if score.total_score >= 75:
        verdict = "высокое соответствие"
    elif score.total_score >= 50:
        verdict = "среднее соответствие"
    else:
        verdict = "слабое соответствие"
    return (
        f"{verdict.capitalize()}: тематика {score.topic_score}/20, визуал "
        f"{score.visual_score}/20, аудитория {score.audience_score}/15, тон "
        f"{score.tone_score}/10, вовлечённость {score.engagement_score}/10, "
        f"рекламная нагрузка {score.ad_load_score}/10, ценовой сегмент "
        f"{score.price_segment_score}/5, форматы {score.format_score}/5, "
        f"brand safety {score.brand_safety_score}/5. Ближайший эталон — {similar_to}."
    )


def _mock_criterion_details(
    candidate: CandidateProfile,
    score: ScoreBreakdown,
) -> dict[str, ScoreCriterionDetail]:
    """Explain prepared mock fields without altering their deterministic scores."""

    insufficient = "Недостаточно данных для уверенной оценки. confidence: low"
    return {
        "topic": ScoreCriterionDetail(
            score=score.topic_score,
            max_score=20,
            reason=(
                f"Заявленные темы: {', '.join(candidate.content_topics)}."
                if candidate.content_topics
                else insufficient
            ),
            confidence="high" if candidate.content_topics else "low",
        ),
        "visual": ScoreCriterionDetail(
            score=score.visual_score,
            max_score=20,
            reason=(
                f"Заявленный визуальный стиль: {', '.join(candidate.visual_style)}."
                if candidate.visual_style
                else insufficient
            ),
            confidence="high" if candidate.visual_style else "low",
        ),
        "audience": ScoreCriterionDetail(
            score=score.audience_score,
            max_score=15,
            reason=(
                f"Описание аудитории: {candidate.audience_description}; интересы: "
                f"{', '.join(candidate.audience_interests or [])}."
                if candidate.audience_description and candidate.audience_interests
                else insufficient
            ),
            confidence=(
                "high"
                if candidate.audience_description and candidate.audience_interests
                else "low"
            ),
        ),
        "tone": ScoreCriterionDetail(
            score=score.tone_score,
            max_score=10,
            reason=(
                f"Заявленный тон: {', '.join(candidate.tone)}."
                if candidate.tone
                else insufficient
            ),
            confidence="high" if candidate.tone else "low",
        ),
        "engagement": ScoreCriterionDetail(
            score=score.engagement_score,
            max_score=10,
            reason=(
                f"ER={candidate.engagement_rate_pct:.2f}% сопоставлен с эталонным ER."
                if candidate.engagement_rate_pct is not None
                else insufficient
            ),
            confidence="high" if candidate.engagement_rate_pct is not None else "low",
        ),
        "ad_load": ScoreCriterionDetail(
            score=score.ad_load_score,
            max_score=10,
            reason=(
                f"Заявленная рекламная нагрузка: {candidate.advertising_load_pct:.1f}%."
                if candidate.advertising_load_pct is not None
                else insufficient
            ),
            confidence="high" if candidate.advertising_load_pct is not None else "low",
        ),
        "price_segment": ScoreCriterionDetail(
            score=score.price_segment_score,
            max_score=5,
            reason=(
                f"Ценовой сегмент: {candidate.price_segment.value}."
                if candidate.price_segment is not None
                else insufficient
            ),
            confidence="high" if candidate.price_segment is not None else "low",
        ),
        "format": ScoreCriterionDetail(
            score=score.format_score,
            max_score=5,
            reason=(
                f"Заявленные форматы: {', '.join(candidate.content_formats)}."
                if candidate.content_formats
                else insufficient
            ),
            confidence="high" if candidate.content_formats else "low",
        ),
        "brand_safety": ScoreCriterionDetail(
            score=score.brand_safety_score,
            max_score=5,
            reason=(
                f"Уровень brand safety: {candidate.brand_safety.value}."
                if candidate.brand_safety is not None
                else insufficient
            ),
            confidence="high" if candidate.brand_safety is not None else "low",
        ),
    }


def score_candidate(
    candidate: CandidateProfile,
    ideal_profile: IdealBloggerProfile,
    source_bloggers: list[SourceBlogger],
) -> CandidateEvaluation:
    """Score one candidate against nine fixed fashion criteria."""

    if not source_bloggers:
        raise ValueError("At least one source blogger is required for scoring")

    topic_score = _points(
        _list_similarity(candidate.content_topics or [], ideal_profile.content_topics),
        20,
    )
    visual_score = _points(
        _tag_similarity(candidate.visual_style or [], ideal_profile.visual_style),
        20,
    )
    audience_score = _points(
        max(
            _audience_similarity(
                candidate.audience_description or "",
                candidate.audience_interests or [],
                source.audience_description,
                source.audience_interests,
            )
            for source in source_bloggers
        ),
        15,
    )
    tone_score = _points(_tag_similarity(candidate.tone or [], ideal_profile.tone), 10)
    engagement_score = _engagement_points(
        candidate.engagement_rate_pct,
        ideal_profile.engagement_rate_pct,
    )
    ad_load_score = _advertising_load_points(candidate.advertising_load_pct)
    price_segment_score = _points(
        _price_similarity(candidate.price_segment, ideal_profile.price_segment),
        5,
    )
    format_score = _points(
        _tag_similarity(candidate.content_formats or [], ideal_profile.content_formats),
        5,
    )
    brand_safety_score = _points(
        _brand_safety_similarity(candidate.brand_safety),
        5,
    )
    total_score = (
        topic_score
        + visual_score
        + audience_score
        + tone_score
        + engagement_score
        + ad_load_score
        + price_segment_score
        + format_score
        + brand_safety_score
    )
    score = ScoreBreakdown(
        topic_score=topic_score,
        visual_score=visual_score,
        audience_score=audience_score,
        tone_score=tone_score,
        engagement_score=engagement_score,
        ad_load_score=ad_load_score,
        price_segment_score=price_segment_score,
        format_score=format_score,
        brand_safety_score=brand_safety_score,
        total_score=total_score,
    )
    nearest = _nearest_source(candidate, source_bloggers)
    return CandidateEvaluation(
        candidate_handle=candidate.handle,
        score=score,
        similar_to=nearest.display_name,
        match_reason=_match_reason(score, nearest.display_name),
        criterion_details=_mock_criterion_details(candidate, score),
    )


def rank_candidates(
    candidates: list[CandidateProfile],
    ideal_profile: IdealBloggerProfile,
    source_bloggers: list[SourceBlogger],
    top_k: int | None = None,
) -> list[RankedCandidate]:
    """Score candidates, sort descending, and optionally keep only `top_k`."""

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    evaluated = [
        (candidate, score_candidate(candidate, ideal_profile, source_bloggers))
        for candidate in candidates
    ]
    evaluated.sort(
        key=lambda item: (
            -item[1].score.total_score,
            item[0].display_name.casefold(),
            item[0].handle.casefold(),
        )
    )
    if top_k is not None:
        evaluated = evaluated[:top_k]

    ranked = [
        RankedCandidate(rank=rank, candidate=candidate, evaluation=evaluation)
        for rank, (candidate, evaluation) in enumerate(evaluated, start=1)
    ]
    LOGGER.info("Ranked %d fashion candidates; returning %d", len(candidates), len(ranked))
    return ranked


def select_ranked_candidates(
    ranked_candidates: list[RankedCandidate],
    min_score: int,
    top_k: int,
) -> list[RankedCandidate]:
    """Keep at most `top_k` candidates meeting the configured minimum score."""

    if not 0 <= min_score <= 100:
        raise ValueError("min_score must be between 0 and 100")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    selected = [
        candidate
        for candidate in ranked_candidates
        if candidate.evaluation.score.total_score >= min_score
    ][:top_k]
    return [
        candidate.model_copy(update={"rank": index})
        for index, candidate in enumerate(selected, start=1)
    ]

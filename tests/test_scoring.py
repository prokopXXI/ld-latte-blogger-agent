"""Behavioral tests for deterministic fashion scoring and offers."""

from pathlib import Path

from src.candidate_ranker import (
    rank_candidates,
    score_candidate,
    select_ranked_candidates,
)
from src.models import (
    CandidateProfile,
    CandidateSource,
    IdealBloggerProfile,
    SourceBlogger,
)
from src.offer_generator import generate_offers
from src.profile_analyzer import build_ideal_profile
from src.sheets_loader import load_candidate_profiles, load_source_bloggers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "source_bloggers.example.csv"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "candidates.example.csv"


def _datasets() -> tuple[
    list[SourceBlogger],
    list[CandidateProfile],
    IdealBloggerProfile,
]:
    sources = load_source_bloggers(SOURCE_PATH)
    candidates = load_candidate_profiles(CANDIDATES_PATH)
    ideal = build_ideal_profile(sources)
    return sources, candidates, ideal


def test_ideal_match_gets_high_score() -> None:
    sources, _, ideal = _datasets()
    source = sources[0]
    perfect_candidate = CandidateProfile(
        handle="@perfect_match",
        display_name="Идеальное совпадение",
        platform=source.platform,
        profile_url="https://example.com/perfect_match",
        followers=source.followers,
        engagement_rate_pct=max(source.engagement_rate_pct, ideal.engagement_rate_pct),
        average_views=source.average_views,
        content_topics=source.content_topics,
        audience_description=source.audience_description,
        audience_interests=source.audience_interests,
        content_style=source.content_style,
        content_formats=source.content_formats,
        tone=source.tone,
        visual_style=source.visual_style,
        advertising_load_pct=source.advertising_load_pct,
        price_segment=source.price_segment,
        brand_safety=source.brand_safety,
        location=source.location,
        source=CandidateSource.PREPARED_PUBLIC_LIST,
        source_url="https://example.com/public-list/perfect-match",
        notes="Синтетический профиль для проверки идеального совпадения",
    )

    evaluation = score_candidate(perfect_candidate, ideal, sources)

    assert evaluation.score.total_score >= 90


def test_weak_match_gets_low_score() -> None:
    sources, candidates, ideal = _datasets()
    weak_candidate = next(item for item in candidates if item.handle == "@trend_sale_feed")

    evaluation = score_candidate(weak_candidate, ideal, sources)

    assert evaluation.score.total_score < 50


def test_candidates_are_sorted_descending() -> None:
    sources, candidates, ideal = _datasets()

    ranked = rank_candidates(candidates, ideal, sources)
    scores = [item.evaluation.score.total_score for item in ranked]

    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_number_of_results() -> None:
    sources, candidates, ideal = _datasets()

    ranked = rank_candidates(candidates, ideal, sources, top_k=3)

    assert len(ranked) == 3
    assert [item.rank for item in ranked] == [1, 2, 3]


def test_candidate_below_min_score_is_excluded() -> None:
    sources, candidates, ideal = _datasets()
    ranked = rank_candidates(candidates, ideal, sources)

    selected = select_ranked_candidates(ranked, min_score=70, top_k=5)

    assert all(item.evaluation.score.total_score >= 70 for item in selected)
    assert all(item.candidate.handle != "@sport_shape_irina" for item in selected)


def test_three_qualified_candidates_return_three_not_five() -> None:
    sources, candidates, ideal = _datasets()
    ranked = rank_candidates(candidates, ideal, sources)

    selected = select_ranked_candidates(ranked, min_score=80, top_k=5)

    assert len(selected) == 3


def test_offers_exclude_legacy_domain_terms() -> None:
    sources, candidates, ideal = _datasets()
    ranked = rank_candidates(candidates, ideal, sources, top_k=5)
    offers = generate_offers(ranked, sources)
    forbidden_terms = (
        "\u043d\u0430\u043f\u0438\u0442\u043e\u043a",
        "\u043a\u043e\u0444\u0435",
        "\u0440\u0435\u0446\u0435\u043f\u0442",
        "ve" + "gan",
        "\u0440\u0430\u0441\u0442\u0438\u0442\u0435\u043b\u044c\u043d\u043e\u0435 "
        "\u043f\u0438\u0442\u0430\u043d\u0438\u0435",
        "latte" + "-" + "\u043d\u0430\u043f\u0438\u0442\u043a\u0438",
    )

    for offer in offers:
        normalized_message = offer.message.casefold()
        assert all(term not in normalized_message for term in forbidden_terms)

"""Offline tests for the gated Tavily → Apify → OpenAI final pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import src.final_pipeline as final_pipeline
from src.config import load_settings
from src.final_candidate_ranker import score_real_candidate, select_finalists
from src.final_offer_generator import (
    FinalOfferGenerationError,
    generate_final_offers,
)
from src.models import (
    EnrichedSourceBlogger,
    FinalPersonalizedOffer,
    FinalScoreBreakdown,
    FinalScoredCandidate,
    LLMIdealBloggerProfile,
    Platform,
    ProfileEnrichmentStatus,
    PublicInstagramPost,
    PublicInstagramProfile,
    RealCandidateAuditRow,
    RealCandidateProfile,
    SearchHit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ideal() -> LLMIdealBloggerProfile:
    return LLMIdealBloggerProfile(
        dominant_topics=["женская мода", "Wildberries / Ozon"],
        secondary_topics=["капсульный гардероб"],
        content_formats=["Reels", "YouTube Shorts", "Telegram publication"],
        visual_style_signals=["natural on-body presentation (text signal)"],
        tone_of_voice=["честный", "практичный"],
        target_audience=["inferred: женщины, покупающие одежду"],
        audience_interests=["образы", "примерки", "маркетплейсы"],
        price_segment="inferred: affordable to mid",
        typical_follower_range="micro to mid",
        engagement_rate_range="1–8%",
        advertising_load_preferences=["умеренная, требует проверки"],
        preferred_integration_formats=["Reels", "Shorts", "публикация"],
        brand_safety_requirements=["ручная проверка"],
        positive_signals=["авторский fashion-профиль"],
        negative_signals=["магазин"],
        exclusion_criteria=["магазины и каталоги"],
        search_keywords=["женская мода Wildberries"],
        search_queries=[
            "site:instagram.com женская одежда блогер",
            "site:youtube.com/shorts женская мода автор",
            "site:t.me женские образы авторский канал",
            "site:instagram.com примерки Wildberries блогер",
            "site:youtube.com/shorts капсульный гардероб",
            "site:t.me Ozon мода автор",
        ],
        confidence_score=0.72,
        evidence_summary="Evidence-based test portrait; inferred fields are marked.",
        sample_profile_usernames=["reference_author"],
    )


def _candidate(username: str, confidence: float = 0.8) -> RealCandidateProfile:
    return RealCandidateProfile(
        name=f"Автор {username}",
        username=username,
        platform=Platform.INSTAGRAM,
        profile_url=f"https://www.instagram.com/{username}/",
        title="Fashion-блог: женская одежда и примерки",
        snippet=(
            "Честно показываю женские образы, капсульный гардероб, сочетания "
            "вещей и находки Wildberries для девушек"
        ),
        source_query="site:instagram.com женская одежда блогер",
        tavily_score=0.8,
        followers_count=20_000,
        engagement_rate=3.5,
        is_private=False,
        content_formats=["reels/video"],
        evidence=["Tavily title and snippet contain fashion signals"],
        data_confidence=confidence,
        enrichment_status="success",
    )


def _audit(candidate: RealCandidateProfile) -> RealCandidateAuditRow:
    return RealCandidateAuditRow(
        raw_url=str(candidate.profile_url),
        normalized_url=str(candidate.profile_url),
        platform=candidate.platform,
        source_query=candidate.source_query,
        title=candidate.title,
        tavily_score=candidate.tavily_score,
        decision="accepted_pre_enrichment",
        reason="test candidate",
        data_confidence=candidate.data_confidence,
    )


def _enriched(username: str) -> EnrichedSourceBlogger:
    post = PublicInstagramPost(
        post_type="Video",
        caption="Честная примерка женского образа с Wildberries",
        hashtags=["fashion", "примерка"],
        likes_count=500,
        comments_count=20,
        timestamp=datetime.now(UTC),
        accessibility_caption="Автор показывает одежду на себе",
    )
    return EnrichedSourceBlogger(
        profile=PublicInstagramProfile(
            username=username,
            profile_url=f"https://www.instagram.com/{username}/",
            full_name=f"Автор {username}",
            biography="Женская мода, капсулы и находки Ozon",
            followers_count=20_000,
            following_count=300,
            posts_count=100,
            is_verified=False,
            is_private=False,
            raw_source="apify",
            fetched_at=datetime.now(UTC),
        ),
        recent_posts=[post],
        calculated_engagement_rate=2.6,
        available_post_count=1,
        data_confidence=0.9,
        missing_fields=[],
        enrichment_status=ProfileEnrichmentStatus.SUCCESS,
        enrichment_error=None,
    )


def _settings(tmp_path: Path, source_path: Path, ideal_path: Path):
    return replace(
        load_settings(),
        ideal_blogger_profile_json_path=ideal_path,
        enriched_source_json_path=source_path,
        tavily_max_queries=6,
        tavily_results_per_query=5,
        max_candidates_before_enrichment=20,
        max_candidates_for_apify=8,
        max_final_candidates=5,
        final_min_score=70,
        profile_posts_limit=3,
        profile_enrichment_concurrency=2,
        profile_enrichment_delay_seconds=0,
        real_candidate_cache_dir=tmp_path / "cache",
        real_candidates_raw_path=tmp_path / "raw.csv",
        real_candidates_audit_path=tmp_path / "audit.csv",
        real_candidates_enriched_path=tmp_path / "enriched.json",
        final_real_bloggers_csv_path=tmp_path / "final.csv",
        final_real_bloggers_markdown_path=tmp_path / "final.md",
        final_run_audit_path=tmp_path / "run_audit.json",
        final_offer_prompt_path=PROJECT_ROOT / "prompts" / "final_offer_generation.txt",
        final_apify_raw_response_path=tmp_path / "apify_raw.json",
    )


def test_reference_profiles_are_excluded() -> None:
    hit = SearchHit(
        url="https://instagram.com/reference_author/",
        title="Fashion-блог reference_author",
        snippet="Женская одежда и примерки Wildberries",
        source_query="fashion",
        provider_score=0.9,
    )

    result = final_pipeline.clean_real_search_hits(
        [hit],
        reference_urls={"https://www.instagram.com/reference_author"},
        reference_usernames={"reference_author"},
        maximum_candidates=20,
    )

    assert result.candidates == []
    assert result.audit_rows[0].reason == "source_reference_profile"


def test_duplicate_profiles_are_removed() -> None:
    hits = [
        SearchHit(
            url="https://instagram.com/new_author/",
            title="Fashion-блог new_author",
            snippet="Женская одежда и примерки",
            source_query="fashion",
        ),
        SearchHit(
            url="https://www.instagram.com/NEW_AUTHOR",
            title="Тот же fashion-блог",
            snippet="Женская мода",
            source_query="fashion duplicate",
        ),
    ]

    result = final_pipeline.clean_real_search_hits(
        hits,
        reference_urls=set(),
        reference_usernames=set(),
        maximum_candidates=20,
    )

    assert len(result.candidates) == 1
    assert [row.reason for row in result.audit_rows].count("duplicate_profile") == 1


def test_stores_are_rejected() -> None:
    hit = SearchHit(
        url="https://instagram.com/dress_shop/",
        title="Официальный магазин женской одежды",
        snippet="Каталог платьев Wildberries",
        source_query="fashion",
    )

    result = final_pipeline.clean_real_search_hits(
        [hit],
        reference_urls=set(),
        reference_usernames=set(),
        maximum_candidates=20,
    )

    assert result.candidates == []
    assert result.audit_rows[0].reason == "store_brand_or_catalog"


def test_apify_limit_is_respected(tmp_path: Path) -> None:
    candidates = [_candidate(f"author_{index}") for index in range(10)]
    source = tmp_path / "source.json"
    source.write_text("[]", encoding="utf-8")
    ideal_path = tmp_path / "ideal.json"
    ideal_path.write_text(_ideal().model_dump_json(), encoding="utf-8")
    settings = _settings(tmp_path, source, ideal_path)

    class FakeApify:
        name = "apify"
        actor_id = "test~actor"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def fetch_profile(self, username: str, profile_url: str, posts_limit: int):
            assert posts_limit == 3
            self.calls.append(username)
            return _enriched(username)

    provider = FakeApify()
    enriched, audit, sent_count = final_pipeline.enrich_instagram_candidates(
        candidates,
        [_audit(candidate) for candidate in candidates],
        provider=provider,  # type: ignore[arg-type]
        settings=settings,
    )

    assert sent_count == 8
    assert len(provider.calls) == 8
    assert len(enriched) == 8
    assert sum(row.reason == "max_candidates_for_apify_limit" for row in audit) == 2


def test_low_confidence_reduces_score() -> None:
    high = _candidate("high_confidence", confidence=1.0)
    low = RealCandidateProfile.model_validate(
        {
            **high.model_dump(mode="json"),
            "username": "low_confidence",
            "profile_url": "https://www.instagram.com/low_confidence/",
            "data_confidence": 0.2,
        }
    )

    high_score = score_real_candidate(high, _ideal())
    low_score = score_real_candidate(low, _ideal())

    assert low_score.score.total_score < high_score.score.total_score
    assert "низкая полнота данных" in low_score.match_reason


def _breakdown(total: int) -> FinalScoreBreakdown:
    maximums = [20, 15, 15, 10, 10, 10, 5, 5, 5, 5]
    values: list[int] = []
    remaining = total
    for maximum in maximums:
        value = min(maximum, remaining)
        values.append(value)
        remaining -= value
    assert remaining == 0
    return FinalScoreBreakdown(
        fashion_relevance_score=values[0],
        visual_text_score=values[1],
        audience_score=values[2],
        tone_score=values[3],
        engagement_score=values[4],
        advertising_load_score=values[5],
        price_segment_score=values[6],
        content_format_score=values[7],
        brand_safety_score=values[8],
        data_confidence_score=values[9],
        total_score=total,
    )


def _scored(username: str, total: int) -> FinalScoredCandidate:
    return FinalScoredCandidate(
        candidate=_candidate(username),
        score=_breakdown(total),
        match_reason=f"Score {total}; manual review required",
        evidence=["public test evidence"],
    )


def test_openai_is_called_only_for_finalists() -> None:
    ranked = [
        _scored("first", 90),
        _scored("second", 80),
        _scored("below", 69),
        _scored("fourth", 75),
    ]
    finalists = select_finalists(ranked, min_score=70, maximum=3)

    class FakeOffers:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate_offer(self, finalist: FinalScoredCandidate):
            username = finalist.candidate.username
            self.calls.append(username)
            return FinalPersonalizedOffer(
                candidate_username=username,
                message=f"Черновик для {username}; обсудить бартер после проверки.",
                evidence_used=["public test evidence"],
                human_review_required=True,
            )

    provider = FakeOffers()
    result = generate_final_offers(finalists, provider)

    assert provider.calls == ["first", "second", "fourth"]
    assert "below" not in provider.calls
    assert len(result.offers) == 3


def test_dry_run_does_not_create_external_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps([_enriched("reference_author").model_dump(mode="json")]),
        encoding="utf-8",
    )
    ideal_path = tmp_path / "ideal.json"
    ideal_path.write_text(_ideal().model_dump_json(), encoding="utf-8")
    settings = _settings(tmp_path, source, ideal_path)

    def forbidden(*args, **kwargs):
        raise AssertionError(f"network provider created: {args} {kwargs}")

    monkeypatch.setattr(final_pipeline, "TavilySearchProvider", forbidden)
    monkeypatch.setattr(final_pipeline, "create_profile_enrichment_provider", forbidden)
    monkeypatch.setattr(final_pipeline, "OpenAIFinalOfferProvider", forbidden)

    result = final_pipeline.run_final_pipeline(settings, dry_run=True)

    assert result.plan.query_count == 6
    assert result.audit is None
    assert not settings.final_run_audit_path.exists()


def test_partial_offer_failure_preserves_other_results() -> None:
    finalists = [_scored("works", 90), _scored("fails", 85)]

    class PartialProvider:
        def generate_offer(self, finalist: FinalScoredCandidate):
            username = finalist.candidate.username
            if username == "fails":
                raise FinalOfferGenerationError("synthetic offline failure")
            return FinalPersonalizedOffer(
                candidate_username=username,
                message="Предлагаем обсудить бартер после ручной проверки.",
                evidence_used=["public test evidence"],
                human_review_required=True,
            )

    result = generate_final_offers(finalists, PartialProvider())

    assert len(result.offers) == 2
    assert len(result.errors) == 1
    assert result.offers[0].candidate_username == "works"
    assert "Не отправлять" in result.offers[1].message


def test_explainable_breakdown_preserves_existing_score() -> None:
    scored = score_real_candidate(_candidate("explainable"), _ideal())
    row = final_pipeline.final_score_breakdown_row(scored)

    assert row.total_score == scored.score.total_score
    assert row.topic_score == scored.score.fashion_relevance_score
    assert row.visual_score == scored.score.visual_text_score
    assert row.audience_score == scored.score.audience_score
    assert row.tone_score == scored.score.tone_score
    assert row.engagement_score == scored.score.engagement_score
    assert row.ad_load_score == scored.score.advertising_load_score
    assert row.price_segment_score == scored.score.price_segment_score
    assert row.format_score == scored.score.content_format_score
    assert row.brand_safety_score == scored.score.brand_safety_score
    criterion_sum = (
        row.topic_score
        + row.visual_score
        + row.audience_score
        + row.tone_score
        + row.engagement_score
        + row.ad_load_score
        + row.price_segment_score
        + row.format_score
        + row.brand_safety_score
        + scored.score.data_confidence_score
    )
    assert criterion_sum == row.total_score


def test_missing_data_is_explained_with_low_confidence() -> None:
    base = _candidate("missing_signals")
    candidate = RealCandidateProfile.model_validate(
        {
            **base.model_dump(mode="json"),
            "title": None,
            "snippet": None,
            "biography": None,
            "engagement_rate": None,
            "recent_posts": [],
            "content_formats": [],
            "data_confidence": 0.2,
        }
    )

    scored = score_real_candidate(candidate, _ideal())
    row = final_pipeline.final_score_breakdown_row(scored)

    assert "Недостаточно данных" in row.engagement_reason
    assert "confidence: low" in row.engagement_reason
    assert "Недостаточно данных" in row.ad_load_reason
    assert "confidence: low" in row.format_reason


def test_score_breakdown_csv_and_markdown_are_explainable(tmp_path: Path) -> None:
    scored = score_real_candidate(_candidate("documented"), _ideal())
    breakdown_path = tmp_path / "breakdown.csv"
    final_pipeline.save_final_score_breakdown([scored], breakdown_path)

    header = breakdown_path.read_text(encoding="utf-8").splitlines()[0]
    assert "topic_reason" in header
    assert "confidence_reason" in header
    assert "evidence_count" in header

    offer = FinalPersonalizedOffer(
        candidate_username="documented",
        message="Черновик предложения обсудить бартер после ручной проверки.",
        evidence_used=["public test evidence"],
        human_review_required=True,
    )
    markdown_path = tmp_path / "final.md"
    final_pipeline.save_final_outputs(
        [scored],
        [offer],
        csv_path=tmp_path / "final.csv",
        markdown_path=markdown_path,
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| Критерий | Баллы | Причина |" in markdown
    assert "| Тематика |" in markdown
    assert "| Полнота данных |" in markdown


def test_saved_pool_categories_keep_final_min_score_boundaries() -> None:
    ranked = [
        _scored("recommended", 70),
        _scored("manual_high", 69),
        _scored("manual_low", 60),
        _scored("rejected", 59),
    ]

    recommended, manual, rejected = final_pipeline.categorize_saved_pool(ranked)

    assert [item.candidate.username for item in recommended] == ["recommended"]
    assert [item.candidate.username for item in manual] == ["manual_high", "manual_low"]
    assert [item.candidate.username for item in rejected] == ["rejected"]


def test_saved_pool_offer_targets_fill_to_three_from_manual_review() -> None:
    recommended = [_scored("recommended_1", 80), _scored("recommended_2", 75)]
    manual = [_scored("manual_1", 69), _scored("manual_2", 65)]

    selected = final_pipeline.select_saved_pool_offer_targets(recommended, manual)

    assert [item.candidate.username for item in selected] == [
        "recommended_1",
        "recommended_2",
        "manual_1",
    ]


def test_saved_pool_dry_run_uses_no_external_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(
        load_settings(),
        final_all_candidates_csv_path=tmp_path / "all.csv",
        final_all_candidates_markdown_path=tmp_path / "all.md",
        final_recommended_bloggers_csv_path=tmp_path / "recommended.csv",
        final_score_breakdown_path=tmp_path / "breakdown.csv",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(f"external provider created: {args} {kwargs}")

    monkeypatch.setattr(final_pipeline, "TavilySearchProvider", forbidden)
    monkeypatch.setattr(final_pipeline, "create_profile_enrichment_provider", forbidden)
    monkeypatch.setattr(final_pipeline, "OpenAIFinalOfferProvider", forbidden)

    result = final_pipeline.finalize_saved_v2_pool(settings, dry_run=True)

    assert result.plan.pool_size == 20
    assert result.plan.saved_enriched_count == 8
    assert result.plan.apify_required_count == 12
    assert result.plan.tavily_calls == 0
    assert not settings.final_all_candidates_csv_path.exists()


def test_saved_pool_reuses_eight_and_enriches_only_remaining_profiles(
    tmp_path: Path,
) -> None:
    settings = replace(
        load_settings(),
        real_candidate_cache_dir=tmp_path / "cache",
        profile_enrichment_delay_seconds=0,
        final_all_candidates_csv_path=tmp_path / "all.csv",
        final_all_candidates_markdown_path=tmp_path / "all.md",
        final_recommended_bloggers_csv_path=tmp_path / "recommended.csv",
        final_score_breakdown_path=tmp_path / "breakdown.csv",
    )
    existing = {
        item.username
        for item in final_pipeline._load_saved_enriched_candidates(settings)
    }

    class FakeApify:
        name = "apify"
        actor_id = "test~actor"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def fetch_profile(self, username: str, profile_url: str, posts_limit: int):
            assert posts_limit == 3
            self.calls.append(username)
            return _enriched(username)

    class FakeOffers:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate_offer(self, finalist: FinalScoredCandidate):
            username = finalist.candidate.username
            self.calls.append(username)
            return FinalPersonalizedOffer(
                candidate_username=username,
                message="Черновик обсудить бартер; не отправлять без ручной проверки.",
                evidence_used=["public test evidence"],
                human_review_required=True,
            )

    enrichment = FakeApify()
    offers = FakeOffers()
    result = final_pipeline.finalize_saved_v2_pool(
        settings,
        dry_run=False,
        enrichment_provider=enrichment,  # type: ignore[arg-type]
        offer_provider=offers,
    )

    assert len(result.ranked) == 20
    assert len(enrichment.calls) == 12
    assert existing.isdisjoint(enrichment.calls)
    assert len(offers.calls) <= 5
    assert settings.final_all_candidates_csv_path.is_file()
    assert settings.final_all_candidates_markdown_path.is_file()

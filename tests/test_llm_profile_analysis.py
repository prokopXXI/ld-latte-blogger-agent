"""Offline tests for enriched-source LLM analysis and dry-run safety."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import src.llm_profile_analyzer as analyzer
from src.config import load_settings
from src.llm_profile_analyzer import (
    CAPTION_LIMIT,
    LLMAnalysisError,
    MockLLMProvider,
    OpenAILLMProvider,
    batch_profiles,
    build_ideal_profile_from_enriched,
    load_llm_profiles,
)
from src.models import (
    BatchBloggerInsights,
    EnrichedSourceBlogger,
    ProfileEnrichmentStatus,
    PublicInstagramPost,
    PublicInstagramProfile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _blogger(
    username: str,
    status: ProfileEnrichmentStatus,
    *,
    caption: str | None = "Честная примерка и стильные женские образы",
    posts: bool = True,
) -> EnrichedSourceBlogger:
    recent_posts = (
        [
            PublicInstagramPost(
                post_url=f"https://www.instagram.com/p/{username}/",
                post_type="Video",
                caption=caption,
                hashtags=["fashion", "примерка"],
                likes_count=120,
                comments_count=12,
                timestamp=datetime.now(UTC),
                accessibility_caption="Автор показывает повседневный образ",
            )
        ]
        if posts
        else []
    )
    return EnrichedSourceBlogger(
        profile=PublicInstagramProfile(
            username=username,
            profile_url=f"https://www.instagram.com/{username}/",
            full_name=f"Автор {username}",
            biography="Женская мода и находки Wildberries",
            followers_count=10_000,
            following_count=100,
            posts_count=50,
            is_verified=False,
            is_private=False,
            raw_source="test",
            fetched_at=datetime.now(UTC),
        ),
        recent_posts=recent_posts,
        calculated_engagement_rate=2.5 if posts else None,
        available_post_count=len(recent_posts),
        data_confidence=0.9 if status == ProfileEnrichmentStatus.SUCCESS else 0.5,
        missing_fields=[] if posts else ["recent_posts"],
        enrichment_status=status,
        enrichment_error="not found" if status == ProfileEnrichmentStatus.FAILED else None,
    )


def _write_bloggers(path: Path, bloggers: list[EnrichedSourceBlogger]) -> None:
    path.write_text(
        json.dumps(
            [blogger.model_dump(mode="json") for blogger in bloggers],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _settings(tmp_path: Path, enriched_path: Path, **changes: object):
    base = load_settings()
    configured = replace(
        base,
        llm_provider="mock",
        openai_model="gpt-5-mini",
        openai_max_profiles_per_batch=2,
        openai_max_posts_per_profile=3,
        openai_max_total_profiles=25,
        enriched_source_json_path=enriched_path,
        llm_batch_prompt_path=PROJECT_ROOT / "prompts" / "source_batch_analysis.txt",
        llm_synthesis_prompt_path=PROJECT_ROOT / "prompts" / "ideal_profile_synthesis.txt",
        llm_batch_insights_path=tmp_path / "llm_batch_insights.json",
        ideal_blogger_profile_json_path=tmp_path / "ideal_blogger_profile.json",
        ideal_blogger_profile_markdown_path=tmp_path / "ideal_blogger_profile.md",
        llm_analysis_audit_path=tmp_path / "llm_analysis_audit.json",
    )
    return replace(configured, **changes)


def _insight(usernames: list[str]) -> BatchBloggerInsights:
    return BatchBloggerInsights(
        analyzed_usernames=usernames,
        dominant_topics=["женская мода"],
        secondary_topics=[],
        content_formats=["короткое видео"],
        tone_patterns=["доверительный"],
        audience_signals=["женская аудитория"],
        price_segment_signals=["mass-market (inferred)"],
        engagement_observations=["ER доступен"],
        advertising_load_signals=["требует проверки"],
        brand_safety_observations=["требует проверки"],
        positive_patterns=["примерки"],
        negative_patterns=[],
        uncertainty_notes=[],
        confidence_score=0.8,
    )


def test_failed_profiles_are_excluded_and_partial_is_kept(tmp_path: Path) -> None:
    source = tmp_path / "enriched.json"
    _write_bloggers(
        source,
        [
            _blogger("success_one", ProfileEnrichmentStatus.SUCCESS),
            _blogger("failed_one", ProfileEnrichmentStatus.FAILED),
            _blogger("partial_one", ProfileEnrichmentStatus.PARTIAL, posts=False),
        ],
    )

    profiles = load_llm_profiles(source, max_profiles=25, max_posts=3)

    assert [profile.username for profile in profiles] == ["success_one", "partial_one"]
    assert profiles[1].recent_posts == []


def test_batching_keeps_final_short_batch() -> None:
    bloggers = [
        _blogger(f"author_{index}", ProfileEnrichmentStatus.SUCCESS)
        for index in range(5)
    ]
    profiles = [analyzer.prepare_llm_profile(blogger, max_posts=3) for blogger in bloggers]

    batches = batch_profiles(profiles, batch_size=2)

    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_captions_are_truncated_and_contacts_are_redacted(tmp_path: Path) -> None:
    source = tmp_path / "enriched.json"
    caption = "mail owner@example.com phone +7 999 123-45-67 " + "я" * 4_000
    _write_bloggers(
        source,
        [_blogger("long_caption", ProfileEnrichmentStatus.SUCCESS, caption=caption)],
    )

    profile = load_llm_profiles(source, max_profiles=1, max_posts=3)[0]
    safe_caption = profile.recent_posts[0].caption or ""

    assert len(safe_caption) <= CAPTION_LIMIT
    assert "owner@example.com" not in safe_caption
    assert "+7 999 123-45-67" not in safe_caption
    assert "[redacted-email]" in safe_caption
    assert "[redacted-phone]" in safe_caption


def test_structured_outputs_are_validated() -> None:
    valid = _insight(["author_one"]).model_dump(mode="json")
    valid["confidence_score"] = 1.5

    with pytest.raises(ValidationError):
        BatchBloggerInsights.model_validate(valid)


def test_openai_provider_uses_pydantic_parse_without_network() -> None:
    calls: list[dict[str, object]] = []

    class FakeResponses:
        def parse(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                output_parsed=_insight(["author_one"]).model_dump(mode="json"),
                usage={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )

    fake_client = SimpleNamespace(responses=FakeResponses())
    provider = OpenAILLMProvider(
        api_key="test-only-key",
        model="gpt-5-mini",
        timeout_seconds=1,
        client=fake_client,
        sleep=lambda _: None,
    )
    profile = analyzer.prepare_llm_profile(
        _blogger("author_one", ProfileEnrichmentStatus.SUCCESS),
        max_posts=3,
    )

    result = provider.analyze_batch([profile], "safe test prompt")

    assert result.analyzed_usernames == ["author_one"]
    assert calls[0]["text_format"] is BatchBloggerInsights
    assert calls[0]["store"] is False
    assert provider.usage["total_tokens"] == 20


def test_dry_run_does_not_create_provider_or_make_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "enriched.json"
    _write_bloggers(source, [_blogger("dry_run", ProfileEnrichmentStatus.SUCCESS)])
    settings = _settings(
        tmp_path,
        source,
        llm_provider="openai",
        openai_api_key=None,
    )

    def forbidden_provider_creation(*_: object, **__: object) -> None:
        raise AssertionError("provider creation would permit a network path")

    monkeypatch.setattr(analyzer, "create_llm_provider", forbidden_provider_creation)

    run = build_ideal_profile_from_enriched(settings, dry_run=True)

    assert run.summary.profile_count == 1
    assert run.summary.model == "gpt-5-mini"
    assert run.ideal_profile is None
    assert run.audit.dry_run is True


def test_missing_openai_key_has_actionable_error() -> None:
    with pytest.raises(LLMAnalysisError, match="OPENAI_API_KEY.*--dry-run-llm"):
        OpenAILLMProvider(
            api_key=None,
            model="gpt-5-mini",
            timeout_seconds=120,
        )


def test_failed_second_batch_preserves_first_intermediate_result(tmp_path: Path) -> None:
    source = tmp_path / "enriched.json"
    _write_bloggers(
        source,
        [
            _blogger("first", ProfileEnrichmentStatus.SUCCESS),
            _blogger("second", ProfileEnrichmentStatus.SUCCESS),
            _blogger("third", ProfileEnrichmentStatus.SUCCESS),
        ],
    )
    settings = _settings(tmp_path, source)

    class FailOnSecondBatch:
        name = "mock"
        model = "failing-offline-test"
        retries = 0
        usage: dict[str, int] = {}

        def __init__(self) -> None:
            self.calls = 0

        def analyze_batch(self, profiles, prompt):
            del prompt
            self.calls += 1
            if self.calls == 2:
                raise LLMAnalysisError("Synthetic batch failure")
            return _insight([profile.username for profile in profiles])

        def synthesize(self, insights, prompt):
            raise AssertionError(f"synthesis must not run: {len(insights)} {prompt[:1]}")

    with pytest.raises(LLMAnalysisError, match="Synthetic batch failure"):
        build_ideal_profile_from_enriched(
            settings,
            provider=FailOnSecondBatch(),  # type: ignore[arg-type]
        )

    saved = json.loads(settings.llm_batch_insights_path.read_text(encoding="utf-8"))
    audit = json.loads(settings.llm_analysis_audit_path.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["analyzed_usernames"] == ["first", "second"]
    assert audit["completed_batches"] == 1
    assert audit["errors"] == ["Synthetic batch failure"]


def test_mock_provider_builds_all_output_files(tmp_path: Path) -> None:
    source = tmp_path / "enriched.json"
    _write_bloggers(
        source,
        [
            _blogger("mock_one", ProfileEnrichmentStatus.SUCCESS),
            _blogger("mock_partial", ProfileEnrichmentStatus.PARTIAL, posts=False),
        ],
    )
    settings = _settings(tmp_path, source)

    run = build_ideal_profile_from_enriched(
        settings,
        provider=MockLLMProvider(),
    )

    assert run.ideal_profile is not None
    assert run.audit.errors == []
    assert settings.llm_batch_insights_path.is_file()
    assert settings.ideal_blogger_profile_json_path.is_file()
    assert settings.ideal_blogger_profile_markdown_path.is_file()
    assert settings.llm_analysis_audit_path.is_file()

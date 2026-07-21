"""Offline tests for Instagram profile enrichment, cache, and limiting."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from src.models import (
    EnrichedSourceBlogger,
    ProfileEnrichmentAuditStatus,
    ProfileEnrichmentStatus,
    PublicInstagramPost,
    PublicInstagramProfile,
)
from src.profile_enrichment_providers import (
    ApifyProfileEnrichmentProvider,
    MockProfileEnrichmentProvider,
    ProfileEnrichmentError,
    ProfileEnrichmentProvider,
    assemble_enriched_blogger,
    build_apify_actor_input,
    calculate_engagement_rate,
    create_profile_enrichment_provider,
    enrich_profile_urls,
    extract_instagram_username,
    load_profile_cache,
    save_profile_cache,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOCK_FIXTURE = PROJECT_ROOT / "data" / "profile_enrichment_mock.json"


def _profile(username: str = "fashion_test") -> PublicInstagramProfile:
    return PublicInstagramProfile(
        username=username,
        profile_url=f"https://www.instagram.com/{username}/",
        full_name="Вымышленный профиль",
        biography="Fashion mock",
        followers_count=1_000,
        following_count=100,
        posts_count=20,
        is_verified=False,
        is_private=False,
        external_url=None,
        profile_image_url="https://example.com/profile.jpg",
        raw_source="test",
        fetched_at=datetime.now(timezone.utc),
    )


def _post(likes: int = 90, comments: int = 10) -> PublicInstagramPost:
    return PublicInstagramPost(
        post_url="https://example.com/post",
        post_type="reel",
        caption="Вымышленный fashion-пост",
        hashtags=["fashion"],
        likes_count=likes,
        comments_count=comments,
        timestamp=datetime.now(timezone.utc),
        display_url="https://example.com/post.jpg",
        video_url=None,
        accessibility_caption=None,
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://www.instagram.com/example_user/",
        "https://instagram.com/example_user",
        "https://www.instagram.com/example_user?igsh=test",
        "instagram.com/example_user/profilecard/",
    ],
)
def test_extracts_username_from_profile_urls(url: str) -> None:
    assert extract_instagram_username(url) == "example_user"


@pytest.mark.parametrize(
    "url",
    [
        "https://instagram.com/p/ABC123/",
        "https://instagram.com/reel/ABC123/",
        "https://instagram.com/stories/example_user/1/",
        "https://instagram.com/explore/",
        "https://instagram.com/accounts/login/",
        "https://instagram.com/",
        "https://example.com/example_user",
    ],
)
def test_rejects_non_profile_instagram_urls(url: str) -> None:
    assert extract_instagram_username(url) is None


def test_calculates_average_engagement_rate() -> None:
    posts = [_post(90, 10), _post(40, 10)]

    assert calculate_engagement_rate(1_000, posts) == 7.5


def test_engagement_rate_is_null_without_followers() -> None:
    assert calculate_engagement_rate(None, [_post()]) is None


def test_fresh_local_cache_is_used(tmp_path: Path) -> None:
    blogger = assemble_enriched_blogger(_profile(), [_post()], posts_limit=6)
    save_profile_cache(blogger, tmp_path)

    cached = load_profile_cache(
        "fashion_test",
        tmp_path,
        now=datetime.now(timezone.utc),
    )

    assert cached is not None
    assert cached.profile.username == "fashion_test"


def test_expired_local_cache_is_ignored(tmp_path: Path) -> None:
    blogger = assemble_enriched_blogger(_profile(), [_post()], posts_limit=6)
    cache_path = save_profile_cache(blogger, tmp_path)
    old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=25)).timestamp()
    os.utime(cache_path, (old_timestamp, old_timestamp))

    assert load_profile_cache(
        "fashion_test",
        tmp_path,
        now=datetime.now(timezone.utc),
    ) is None


class CountingProvider(ProfileEnrichmentProvider):
    name = "counting"

    def __init__(self, failing_username: str | None = None) -> None:
        self.calls: list[str] = []
        self.failing_username = failing_username

    def fetch_profile(
        self,
        username: str,
        profile_url: str,
        posts_limit: int,
    ) -> EnrichedSourceBlogger:
        self.calls.append(username)
        if username == self.failing_username:
            raise ProfileEnrichmentError("simulated profile error")
        profile = _profile(username)
        return assemble_enriched_blogger(profile, [_post()], posts_limit)


def test_one_profile_error_does_not_stop_other_profiles(tmp_path: Path) -> None:
    provider = CountingProvider(failing_username="bad_profile")

    run = enrich_profile_urls(
        [
            "https://instagram.com/good_one/",
            "https://instagram.com/bad_profile/",
            "https://instagram.com/good_two/",
        ],
        provider=provider,
        posts_limit=6,
        cache_dir=tmp_path,
        cache_enabled=False,
        refresh_profiles=False,
        limit_profiles=None,
        concurrency=2,
        delay_seconds=0,
    )

    assert len(run.bloggers) == 3
    assert [item.enrichment_status for item in run.bloggers] == [
        ProfileEnrichmentStatus.SUCCESS,
        ProfileEnrichmentStatus.FAILED,
        ProfileEnrichmentStatus.SUCCESS,
    ]
    assert any(row.status == ProfileEnrichmentAuditStatus.FAILED for row in run.audit_rows)


def test_missing_apify_token_has_clear_error() -> None:
    with pytest.raises(ProfileEnrichmentError, match="APIFY_API_TOKEN.*mock"):
        ApifyProfileEnrichmentProvider(api_token=None, actor_id="owner~actor")


def test_missing_apify_actor_id_has_clear_error() -> None:
    with pytest.raises(ProfileEnrichmentError, match="APIFY_ACTOR_ID.*apify~"):
        ApifyProfileEnrichmentProvider(api_token="apify-test", actor_id=None)


def test_filled_apify_settings_override_mock_provider() -> None:
    provider = create_profile_enrichment_provider(
        provider_name="mock",
        mock_fixture_path=MOCK_FIXTURE,
        apify_api_token="apify-test",
        apify_actor_id="apify~instagram-scraper",
        timeout_seconds=300,
    )

    assert isinstance(provider, ApifyProfileEnrichmentProvider)


def test_mock_provider_works_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_on_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network must not be used by mock enrichment")

    monkeypatch.setattr(httpx.Client, "post", fail_on_network)
    provider = MockProfileEnrichmentProvider(MOCK_FIXTURE)

    result = provider.fetch_profile(
        "offline_fashion",
        "https://www.instagram.com/offline_fashion/",
        posts_limit=2,
    )

    assert result.profile.raw_source == "mock"
    assert result.available_post_count <= 2
    assert result.calculated_engagement_rate is not None


def test_limit_profiles_restricts_provider_calls(tmp_path: Path) -> None:
    provider = CountingProvider()

    run = enrich_profile_urls(
        [f"https://instagram.com/fashion_{index}/" for index in range(5)],
        provider=provider,
        posts_limit=6,
        cache_dir=tmp_path,
        cache_enabled=False,
        refresh_profiles=False,
        limit_profiles=3,
        concurrency=2,
        delay_seconds=0,
    )

    assert len(provider.calls) == 3
    assert len(run.bloggers) == 3
    assert sum(
        row.status == ProfileEnrichmentAuditStatus.SKIPPED_LIMIT
        for row in run.audit_rows
    ) == 2


def test_apify_network_call_is_fully_mocked(tmp_path: Path) -> None:
    called_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called_paths.append(request.url.path)
        assert "token=" not in str(request.url)
        assert request.headers["Authorization"] == "Bearer apify-test"
        if request.method == "POST":
            payload = json.loads(request.content)
            assert payload == {
                "directUrls": ["https://www.instagram.com/public_fashion/"],
                "resultsType": "details",
                "resultsLimit": 2,
            }
            return httpx.Response(
                201,
                json={
                    "data": {
                        "id": "run-test",
                        "status": "READY",
                        "defaultDatasetId": "dataset-test",
                    }
                },
                request=request,
            )
        if "/actor-runs/" in request.url.path:
            assert request.url.params["waitForFinish"] == "60"
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": "run-test",
                        "status": "SUCCEEDED",
                        "defaultDatasetId": "dataset-test",
                    }
                },
                request=request,
            )
        return httpx.Response(
            200,
            json=[
                {
                    "username": "public_fashion",
                    "fullName": "Public Fashion",
                    "biography": "Женская мода, test@example.com",
                    "followersCount": 10_000,
                    "followsCount": 200,
                    "postsCount": 100,
                    "verified": False,
                    "private": False,
                    "profilePicUrl": "https://example.com/avatar.jpg",
                    "latestPosts": [
                        {
                            "url": "https://example.com/apify-post",
                            "mediaType": "reel",
                            "caption": "Fashion #образ, +7 999 123-45-67",
                            "likesCount": 900,
                            "commentsCount": 100,
                            "takenAtTimestamp": 1_752_000_000,
                        }
                    ],
                }
            ],
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = ApifyProfileEnrichmentProvider(
            api_token="apify-test",
            actor_id="apify~instagram-scraper",
            client=client,
            raw_response_path=tmp_path / "apify_raw_response.json",
        )
        result = provider.fetch_profile(
            "public_fashion",
            "https://www.instagram.com/public_fashion/",
            posts_limit=2,
        )

    assert result.calculated_engagement_rate == 10.0
    assert result.profile.following_count == 200
    assert result.recent_posts[0].timestamp is not None
    assert "test@example.com" not in (result.profile.biography or "")
    assert "+7 999 123-45-67" not in (result.recent_posts[0].caption or "")
    assert called_paths == [
        "/v2/actors/apify~instagram-scraper/runs",
        "/v2/actor-runs/run-test",
        "/v2/datasets/dataset-test/items",
    ]
    raw_text = (tmp_path / "apify_raw_response.json").read_text(encoding="utf-8")
    raw_payload = json.loads(raw_text)
    assert "apify-test" not in raw_text
    assert raw_payload["runs"][0]["actor_input"]["directUrls"] == [
        "https://www.instagram.com/public_fashion/"
    ]
    assert raw_payload["runs"][0]["dataset_items"][0]["followersCount"] == 10_000


def test_builds_documented_instagram_scraper_payload() -> None:
    assert build_apify_actor_input(
        actor_id="apify~instagram-scraper",
        username="fashion_test",
        profile_url="https://www.instagram.com/fashion_test/",
        posts_limit=3,
    ) == {
        "directUrls": ["https://www.instagram.com/fashion_test/"],
        "resultsType": "details",
        "resultsLimit": 3,
    }


def test_cache_from_other_provider_is_not_used(tmp_path: Path) -> None:
    mock_blog = assemble_enriched_blogger(_profile(), [_post()], posts_limit=6)
    save_profile_cache(mock_blog, tmp_path)

    assert load_profile_cache(
        "fashion_test",
        tmp_path,
        expected_source="apify",
    ) is None


def test_failed_profile_is_not_reused_from_cache(tmp_path: Path) -> None:
    failed = EnrichedSourceBlogger(
        profile=_profile(),
        recent_posts=[],
        calculated_engagement_rate=None,
        available_post_count=0,
        data_confidence=0,
        missing_fields=["recent_posts"],
        enrichment_status=ProfileEnrichmentStatus.FAILED,
        enrichment_error="temporary failure",
    )
    save_profile_cache(failed, tmp_path)

    assert load_profile_cache("fashion_test", tmp_path) is None


def test_apify_style_limit_replaces_failed_profile_with_next(tmp_path: Path) -> None:
    provider = CountingProvider(failing_username="missing_profile")

    run = enrich_profile_urls(
        [
            "https://instagram.com/missing_profile/",
            "https://instagram.com/good_one/",
            "https://instagram.com/good_two/",
            "https://instagram.com/good_three/",
            "https://instagram.com/not_needed/",
        ],
        provider=provider,
        posts_limit=3,
        cache_dir=tmp_path,
        cache_enabled=False,
        refresh_profiles=True,
        limit_profiles=3,
        concurrency=1,
        delay_seconds=0,
        replace_failed_with_next=True,
    )

    assert provider.calls == [
        "missing_profile",
        "good_one",
        "good_two",
        "good_three",
    ]
    assert len(run.bloggers) == 4
    assert sum(
        blogger.enrichment_status != ProfileEnrichmentStatus.FAILED
        for blogger in run.bloggers
    ) == 3
    assert sum(
        row.status == ProfileEnrichmentAuditStatus.SKIPPED_LIMIT
        for row in run.audit_rows
    ) == 1

"""Offline coverage for content URL → public author resolution and v2 gating."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import httpx
import pytest

import src.final_pipeline as final_pipeline
from src.config import load_settings
from src.content_author_resolver import (
    ApifyInstagramContentAuthorProvider,
    ContentAuthorResolver,
    InstagramContentAuthorProvider,
    YouTubeDataAuthorProvider,
)
from src.main import run_pipeline
from src.models import ContentAuthorResolution, Platform, SearchHit


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _hit(url: str, *, query: str = "женская одежда", suffix: str = "") -> SearchHit:
    return SearchHit(
        url=url,
        title=f"Fashion-блог примерки женской одежды {suffix}".strip(),
        snippet="Автор показывает капсульный гардероб и образы Wildberries",
        source_query=query,
        provider_score=0.8,
    )


def _instagram_resolution(
    url: str,
    username: str,
    confidence: float = 0.98,
) -> ContentAuthorResolution:
    return ContentAuthorResolution(
        content_url=url,
        platform=Platform.INSTAGRAM,
        resolved_profile_url=f"https://www.instagram.com/{username}/",
        resolved_username_or_channel=username,
        resolution_method="mocked_apify_owner_username",
        confidence=confidence,
        status="resolved_author",
        reason="ownerUsername supplied by an offline fixture",
    )


class MappingInstagramProvider(InstagramContentAuthorProvider):
    def __init__(self, mapping: dict[str, ContentAuthorResolution]) -> None:
        self.mapping = mapping
        self.calls: list[list[str]] = []

    def resolve_authors(
        self,
        content_urls: list[str],
    ) -> dict[str, ContentAuthorResolution]:
        self.calls.append(list(content_urls))
        return {url: self.mapping[url] for url in content_urls if url in self.mapping}


def _resolver(
    provider: InstagramContentAuthorProvider | None,
    *,
    maximum: int = 20,
    minimum: float = 0.65,
    youtube_provider=None,
) -> ContentAuthorResolver:
    return ContentAuthorResolver(
        instagram_provider=provider,
        youtube_provider=youtube_provider,
        maximum_content_urls=maximum,
        minimum_confidence=minimum,
        cache_path=None,
    )


def test_instagram_reel_resolves_to_owner_profile() -> None:
    url = "https://www.instagram.com/reel/ABC123/"
    provider = MappingInstagramProvider({url: _instagram_resolution(url, "fashion_owner")})

    result = _resolver(provider).resolve(_hit(url))

    assert str(result.resolved_profile_url) == "https://www.instagram.com/fashion_owner/"
    assert result.resolved_username_or_channel == "fashion_owner"
    assert result.status == "resolved_author"


def test_apify_content_actor_api_is_mocked_and_uses_bearer_header(
    tmp_path: Path,
) -> None:
    url = "https://www.instagram.com/reel/ACTOR123/"
    called_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called_paths.append(request.url.path)
        assert request.headers["Authorization"] == "Bearer apify-test"
        assert "apify-test" not in str(request.url)
        if request.method == "POST":
            assert request.read().decode()
            return httpx.Response(201, json={"data": {"id": "run-1", "status": "RUNNING"}})
        if "/actor-runs/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": "run-1",
                        "status": "SUCCEEDED",
                        "defaultDatasetId": "dataset-1",
                    }
                },
            )
        return httpx.Response(
            200,
            json=[{"shortCode": "ACTOR123", "ownerUsername": "actor_owner"}],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    raw_path = tmp_path / "raw.json"
    provider = ApifyInstagramContentAuthorProvider(
        api_token="apify-test",
        actor_id="apify~instagram-scraper",
        client=client,
        raw_response_path=raw_path,
    )

    result = provider.resolve_authors([url])[url]

    assert str(result.resolved_profile_url) == "https://www.instagram.com/actor_owner/"
    assert called_paths == [
        "/v2/actors/apify~instagram-scraper/runs",
        "/v2/actor-runs/run-1",
        "/v2/datasets/dataset-1/items",
    ]
    assert "apify-test" not in raw_path.read_text(encoding="utf-8")


def test_multiple_reels_from_one_author_are_merged() -> None:
    urls = [
        "https://www.instagram.com/reel/ONE/",
        "https://www.instagram.com/p/TWO/",
    ]
    provider = MappingInstagramProvider(
        {url: _instagram_resolution(url, "same_author") for url in urls}
    )
    hits = [
        _hit(urls[0], query="примерка Wildberries"),
        _hit(urls[1], query="капсульный гардероб"),
    ]
    resolutions = _resolver(provider).resolve_many(hits)

    cleaned = final_pipeline.clean_resolved_search_hits(
        hits,
        resolutions,
        reference_urls=set(),
        reference_usernames=set(),
        maximum_candidates=20,
        minimum_resolution_confidence=0.65,
    )

    assert len(cleaned.candidates) == 1
    assert cleaned.candidates[0].evidence_count == 2
    assert cleaned.candidates[0].evidence_urls == urls
    assert cleaned.candidates[0].content_formats == []
    assert any(row.decision == "merged_author_evidence" for row in cleaned.audit_rows)


def test_youtube_video_resolves_to_confirmed_channel_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Goog-Api-Key"] == "youtube-test"
        assert "youtube-test" not in str(request.url)
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "video123",
                        "snippet": {
                            "channelId": "UC_confirmed_channel",
                            "channelTitle": "Confirmed Fashion Channel",
                        },
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    youtube = YouTubeDataAuthorProvider(api_key="youtube-test", client=client)
    result = _resolver(None, youtube_provider=youtube).resolve(
        _hit("https://www.youtube.com/watch?v=video123")
    )

    assert str(result.resolved_profile_url) == (
        "https://www.youtube.com/channel/UC_confirmed_channel"
    )
    assert result.resolution_method == "youtube_data_api_channel_id"


def test_unverified_youtube_author_does_not_reach_cleaning() -> None:
    hit = _hit("https://youtu.be/unverified")
    resolution = _resolver(None).resolve(hit)

    cleaned = final_pipeline.clean_resolved_search_hits(
        [hit],
        [resolution],
        reference_urls=set(),
        reference_usernames=set(),
        maximum_candidates=20,
        minimum_resolution_confidence=0.65,
    )

    assert cleaned.candidates == []
    assert "unresolved_author" in cleaned.audit_rows[0].reason


def test_reference_author_is_excluded_after_resolution() -> None:
    url = "https://www.instagram.com/reel/REFERENCE/"
    hit = _hit(url)
    resolution = _resolver(
        MappingInstagramProvider({url: _instagram_resolution(url, "reference_author")})
    ).resolve(hit)

    cleaned = final_pipeline.clean_resolved_search_hits(
        [hit],
        [resolution],
        reference_urls=set(),
        reference_usernames={"reference_author"},
        maximum_candidates=20,
        minimum_resolution_confidence=0.65,
    )

    assert cleaned.candidates == []
    assert cleaned.audit_rows[0].reason == "source_reference_profile_after_resolution"


def test_content_resolution_limit_is_respected() -> None:
    urls = [f"https://www.instagram.com/reel/LIMIT{index}/" for index in range(3)]
    provider = MappingInstagramProvider(
        {url: _instagram_resolution(url, f"author_{index}") for index, url in enumerate(urls)}
    )

    results = _resolver(provider, maximum=2).resolve_many([_hit(url) for url in urls])

    assert provider.calls == [urls[:2]]
    assert results[2].status == "skipped_resolution_limit"


def test_low_author_confidence_is_rejected_before_scoring() -> None:
    url = "https://www.instagram.com/p/LOWCONF/"
    provider = MappingInstagramProvider(
        {url: _instagram_resolution(url, "uncertain_author", confidence=0.60)}
    )
    hit = _hit(url)
    resolution = _resolver(provider, minimum=0.65).resolve(hit)

    cleaned = final_pipeline.clean_resolved_search_hits(
        [hit],
        [resolution],
        reference_urls=set(),
        reference_usernames=set(),
        maximum_candidates=20,
        minimum_resolution_confidence=0.65,
    )

    assert resolution.status == "low_confidence"
    assert cleaned.candidates == []
    assert "low_confidence" in cleaned.audit_rows[0].reason


def test_partial_resolution_failure_preserves_other_authors() -> None:
    good = "https://www.instagram.com/reel/GOOD/"
    missing = "https://www.instagram.com/reel/MISSING/"
    provider = MappingInstagramProvider(
        {good: _instagram_resolution(good, "working_author")}
    )
    hits = [_hit(good), _hit(missing)]
    resolutions = _resolver(provider).resolve_many(hits)

    cleaned = final_pipeline.clean_resolved_search_hits(
        hits,
        resolutions,
        reference_urls=set(),
        reference_usernames=set(),
        maximum_candidates=20,
        minimum_resolution_confidence=0.65,
    )

    assert [candidate.username for candidate in cleaned.candidates] == ["working_author"]
    assert any("unresolved_author" in row.reason for row in cleaned.audit_rows)


def test_v2_dry_run_creates_no_network_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(
        load_settings(),
        ideal_blogger_profile_json_path=PROJECT_ROOT / "data" / "ideal_blogger_profile.json",
        enriched_source_json_path=PROJECT_ROOT / "data" / "enriched_source_bloggers.json",
        final_run_audit_v2_path=tmp_path / "run_v2.json",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(f"network provider created: {args} {kwargs}")

    monkeypatch.setattr(final_pipeline, "TavilySearchProvider", forbidden)
    monkeypatch.setattr(final_pipeline, "create_content_author_resolver", forbidden)
    monkeypatch.setattr(final_pipeline, "create_profile_enrichment_provider", forbidden)
    monkeypatch.setattr(final_pipeline, "OpenAIFinalOfferProvider", forbidden)

    result = final_pipeline.run_final_pipeline_v2(settings, dry_run=True)

    assert result.plan.query_count == 8
    assert result.plan.tavily_result_limit == 40
    assert result.audit is None
    assert not settings.final_run_audit_v2_path.exists()


def test_legacy_mock_pipeline_still_runs_offline(tmp_path: Path) -> None:
    settings = replace(
        load_settings(),
        mock_mode=True,
        source_provider="csv",
        source_csv_path=PROJECT_ROOT / "data" / "source_bloggers.example.csv",
        candidates_csv_path=PROJECT_ROOT / "data" / "candidates.example.csv",
        search_provider="mock",
        search_queries_path=tmp_path / "search_queries.json",
        search_audit_path=tmp_path / "search_audit.csv",
        results_csv_path=tmp_path / "results.csv",
        final_score_breakdown_path=tmp_path / "final_score_breakdown.csv",
    )

    result = run_pipeline(settings)

    assert result.selected_candidates
    assert settings.results_csv_path.is_file()
    assert settings.final_score_breakdown_path.is_file()

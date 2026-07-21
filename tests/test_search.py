"""Offline tests for provider selection, URL cleanup, and prefiltering."""

import json
from pathlib import Path

import httpx
import pytest

from src.candidate_enricher import discover_candidates
from src.models import AuditReason, SearchHit, SourceBlogger
from src.search_providers import (
    MockSearchProvider,
    SearchProviderError,
    TavilySearchProvider,
)
from src.sheets_loader import load_source_bloggers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "source_bloggers.example.csv"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "candidates.example.csv"


def _sources() -> list[SourceBlogger]:
    return load_source_bloggers(SOURCE_PATH)


def _fashion_hit(url: str, query: str = "fashion query") -> SearchHit:
    return SearchHit(
        url=url,
        title="Мария — fashion-блог и примерки в Instagram",
        snippet=(
            "Женщины 24–38, капсульный гардероб, естественный светлый визуал, "
            "честные Reels и доступная одежда"
        ),
        source_query=query,
    )


def test_duplicate_profile_urls_are_removed() -> None:
    hits = [
        _fashion_hit("https://www.instagram.com/fashion_maria/?utm_source=test"),
        _fashion_hit("https://instagram.com/fashion_maria/", query="second query"),
    ]

    result = discover_candidates(hits, _sources())

    assert len(result.candidates) == 1
    assert [row.reason for row in result.audit_rows] == [
        AuditReason.ACCEPTED,
        AuditReason.DUPLICATE,
    ]


def test_unsupported_domain_is_rejected() -> None:
    result = discover_candidates(
        [_fashion_hit("https://example.org/fashion_maria")],
        _sources(),
    )

    assert result.candidates == []
    assert result.audit_rows[0].reason == AuditReason.UNSUPPORTED_DOMAIN


def test_brand_or_store_profile_is_rejected() -> None:
    hit = SearchHit(
        url="https://www.instagram.com/moda_shop/",
        title="MODA SHOP — официальный магазин одежды",
        snippet="Женская одежда, каталог товаров, доступные цены",
        source_query="женская одежда Instagram",
    )

    result = discover_candidates([hit], _sources())

    assert result.candidates == []
    assert result.audit_rows[0].reason == AuditReason.BRAND_OR_STORE


def test_missing_tavily_key_has_clear_error() -> None:
    with pytest.raises(SearchProviderError, match="TAVILY_API_KEY.*SEARCH_PROVIDER=mock"):
        TavilySearchProvider(api_key=None)


def test_mock_provider_works_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_on_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network must not be used by mock provider")

    monkeypatch.setattr(httpx.Client, "post", fail_on_network)
    provider = MockSearchProvider(CANDIDATES_PATH)

    hits = provider.search(["offline fashion query"], max_results=30)

    assert len(hits) == 8
    assert all(hit.prefilled_candidate is not None for hit in hits)


def test_tavily_provider_uses_mocked_http_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer tvly-test"
        assert payload["search_depth"] == "basic"
        assert payload["include_raw_content"] is False
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://www.instagram.com/public_stylist/",
                        "title": "Public Stylist",
                        "content": "Женская мода и капсульный гардероб",
                        "score": 0.87,
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = TavilySearchProvider(api_key="tvly-test", client=client)
        hits = provider.search(["fashion query"], max_results=3)

    assert len(hits) == 1
    assert hits[0].source_query == "fashion query"
    assert hits[0].provider_score == 0.87

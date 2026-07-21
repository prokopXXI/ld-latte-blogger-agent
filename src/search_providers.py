"""Replaceable mock and Tavily providers for public-profile discovery."""

import math
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from src.candidate_search import get_candidates
from src.config import Settings
from src.models import SearchHit


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
ALLOWED_SEARCH_DOMAINS = [
    "instagram.com",
    "youtube.com",
    "youtu.be",
    "t.me",
    "telegram.me",
]


class SearchProviderError(RuntimeError):
    """Raised when a configured provider cannot complete discovery."""


class SearchProvider(ABC):
    """Interface implemented by every candidate discovery source."""

    name: str

    @abstractmethod
    def search(self, queries: list[str], max_results: int) -> list[SearchHit]:
        """Return raw public search hits for the supplied queries."""

        raise SearchProviderError("SearchProvider.search must be implemented by a provider")


class MockSearchProvider(SearchProvider):
    """Offline provider backed by the existing candidate CSV."""

    name = "mock"

    def __init__(self, candidates_path: Path) -> None:
        self.candidates_path = candidates_path

    def search(self, queries: list[str], max_results: int) -> list[SearchHit]:
        if not queries:
            raise SearchProviderError("Mock search requires at least one generated query")
        candidates = get_candidates(self.candidates_path)

        def best_query(candidate_index: int) -> str:
            candidate = candidates[candidate_index]
            topics = candidate.content_topics or []
            return max(
                queries,
                key=lambda query: (
                    sum(topic.casefold() in query.casefold() for topic in topics),
                    candidate.platform.value.split("_")[0] in query.casefold(),
                    -queries.index(query),
                ),
            )

        return [
            SearchHit(
                url=str(candidate.profile_url),
                title=candidate.display_name,
                snippet=candidate.notes,
                source_query=best_query(index),
                prefilled_candidate=candidate,
            )
            for index, candidate in enumerate(candidates[:max_results])
        ]


class TavilySearchProvider(SearchProvider):
    """Tavily Search API client restricted to public social-profile domains."""

    name = "tavily"

    def __init__(
        self,
        api_key: str | None,
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise SearchProviderError(
                "TAVILY_API_KEY is missing. Add it to .env or use SEARCH_PROVIDER=mock."
            )
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.client = client

    def _search_with_client(
        self,
        client: httpx.Client,
        queries: list[str],
        max_results: int,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        per_query = min(20, max(1, math.ceil(max_results / len(queries))))
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for query in queries:
            remaining = max_results - len(hits)
            if remaining <= 0:
                break
            payload = {
                "query": query,
                "search_depth": "basic",
                "topic": "general",
                "max_results": min(per_query, remaining),
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
                "include_domains": ALLOWED_SEARCH_DOMAINS,
            }
            try:
                response = client.post(TAVILY_SEARCH_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                raise SearchProviderError(
                    f"Tavily search failed with HTTP {status}. Check TAVILY_API_KEY, "
                    "API limits, or use SEARCH_PROVIDER=mock."
                ) from exc
            except (httpx.RequestError, ValueError) as exc:
                raise SearchProviderError(
                    "Tavily is unavailable or returned an invalid response. "
                    "Check the connection or use SEARCH_PROVIDER=mock."
                ) from exc

            results = data.get("results")
            if not isinstance(results, list):
                raise SearchProviderError(
                    "Tavily response has no results list. Use SEARCH_PROVIDER=mock "
                    "while the API is unavailable."
                )
            for result in results:
                if not isinstance(result, dict) or not isinstance(result.get("url"), str):
                    continue
                hits.append(
                    SearchHit(
                        url=result["url"],
                        title=result.get("title") if isinstance(result.get("title"), str) else None,
                        snippet=(
                            result.get("content")
                            if isinstance(result.get("content"), str)
                            else None
                        ),
                        source_query=query,
                        provider_score=(
                            float(result["score"])
                            if isinstance(result.get("score"), (int, float))
                            and 0 <= float(result["score"]) <= 1
                            else None
                        ),
                    )
                )
                if len(hits) >= max_results:
                    break
        return hits

    def search(self, queries: list[str], max_results: int) -> list[SearchHit]:
        if not queries:
            raise SearchProviderError("Tavily search requires at least one generated query")
        if self.client is not None:
            return self._search_with_client(self.client, queries, max_results)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return self._search_with_client(client, queries, max_results)


def create_search_provider(settings: Settings) -> SearchProvider:
    """Construct the configured provider without changing pipeline orchestration."""

    if settings.search_provider == "mock":
        return MockSearchProvider(settings.candidates_csv_path)
    if settings.search_provider == "tavily":
        return TavilySearchProvider(
            api_key=settings.tavily_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
    raise SearchProviderError(f"Unsupported search provider: {settings.search_provider}")

"""Resolve public social-content URLs to canonical author profiles.

The resolver never treats a post or video as a candidate. Instagram ownership is
read through an official Apify Actor run, while YouTube ownership is read from
the public Data API when a key is configured. All network-facing classes are
dependency-injectable so the complete flow is testable offline.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

import httpx

from src.models import ContentAuthorResolution, Platform, SearchHit


LOGGER = logging.getLogger(__name__)
APIFY_API_BASE_URL = "https://api.apify.com/v2"
YOUTUBE_VIDEOS_ENDPOINT = "https://www.googleapis.com/youtube/v3/videos"
INSTAGRAM_CONTENT_PATHS = {"p", "reel", "tv"}
INSTAGRAM_RESERVED_PATHS = INSTAGRAM_CONTENT_PATHS | {
    "accounts",
    "direct",
    "explore",
    "reels",
    "stories",
}
INSTAGRAM_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9._]{1,30}$")
YOUTUBE_CHANNEL_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?youtube\.com/"
    r"(?P<path>@[A-Za-z0-9_.-]{3,100}|channel/[A-Za-z0-9_-]{3,100}|"
    r"user/[A-Za-z0-9_.-]{3,100}|c/[A-Za-z0-9_.-]{3,100})",
    re.I,
)
YOUTUBE_HANDLE_PATTERN = re.compile(r"(?<![\w.])@([A-Za-z0-9_.-]{3,100})")


class ContentAuthorResolutionError(RuntimeError):
    """Raised when a remote author-resolution backend cannot be used."""


class InstagramContentAuthorProvider(ABC):
    """Batch interface for resolving public Instagram post owners."""

    @abstractmethod
    def resolve_authors(
        self,
        content_urls: list[str],
    ) -> dict[str, ContentAuthorResolution]:
        """Return successful or explicit failed resolutions keyed by input URL."""


class YouTubeContentAuthorProvider(ABC):
    """Batch interface for resolving public YouTube video channels."""

    @abstractmethod
    def resolve_authors(
        self,
        content_urls: list[str],
    ) -> dict[str, ContentAuthorResolution]:
        """Return confirmed channel identities keyed by input URL."""


def _prepared_url(value: str) -> str:
    candidate = value.strip()
    return candidate if "://" in candidate else f"https://{candidate}"


def _host_parts(value: str) -> tuple[str, list[str]]:
    parsed = urlsplit(_prepared_url(value))
    host = parsed.netloc.casefold().split(":", maxsplit=1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host, [part for part in parsed.path.split("/") if part]


def classify_public_url(value: str) -> tuple[str, Platform | None, str | None]:
    """Return ``profile``, ``content``, or ``unsupported`` and a canonical URL."""

    try:
        host, parts = _host_parts(value)
    except ValueError:
        return "unsupported", None, None
    if host in {"instagram.com", "m.instagram.com"}:
        if len(parts) >= 2 and parts[0].casefold() in INSTAGRAM_CONTENT_PATHS:
            return "content", Platform.INSTAGRAM, _prepared_url(value)
        if len(parts) == 1 and parts[0].casefold() not in INSTAGRAM_RESERVED_PATHS:
            username = parts[0]
            if INSTAGRAM_USERNAME_PATTERN.fullmatch(username):
                return (
                    "profile",
                    Platform.INSTAGRAM,
                    f"https://www.instagram.com/{username.casefold()}/",
                )
        return "unsupported", Platform.INSTAGRAM, None
    if host in {"youtube.com", "m.youtube.com"}:
        if parts and (
            parts[0].casefold() == "watch"
            or parts[0].casefold() in {"shorts", "live"}
        ):
            return "content", Platform.YOUTUBE_SHORTS, _prepared_url(value)
        if len(parts) == 1 and parts[0].startswith("@"):
            return (
                "profile",
                Platform.YOUTUBE_SHORTS,
                f"https://www.youtube.com/{parts[0]}",
            )
        if len(parts) == 2 and parts[0].casefold() in {"channel", "user", "c"}:
            return (
                "profile",
                Platform.YOUTUBE_SHORTS,
                f"https://www.youtube.com/{parts[0]}/{parts[1]}",
            )
        return "unsupported", Platform.YOUTUBE_SHORTS, None
    if host == "youtu.be" and parts:
        return "content", Platform.YOUTUBE_SHORTS, _prepared_url(value)
    if host in {"t.me", "telegram.me"}:
        if parts and parts[0].casefold() == "s":
            parts = parts[1:]
        if len(parts) == 1 and not parts[0].startswith("+"):
            return "profile", Platform.TELEGRAM, f"https://t.me/{parts[0].casefold()}"
        return "unsupported", Platform.TELEGRAM, None
    return "unsupported", None, None


def _instagram_shortcode(value: str) -> str | None:
    try:
        host, parts = _host_parts(value)
    except ValueError:
        return None
    if host not in {"instagram.com", "m.instagram.com"} or len(parts) < 2:
        return None
    return parts[1] if parts[0].casefold() in INSTAGRAM_CONTENT_PATHS else None


def _youtube_video_id(value: str) -> str | None:
    try:
        parsed = urlsplit(_prepared_url(value))
    except ValueError:
        return None
    host = parsed.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    parts = [part for part in parsed.path.split("/") if part]
    if host == "youtu.be":
        return parts[0] if parts else None
    if host not in {"youtube.com", "m.youtube.com"}:
        return None
    if parts and parts[0].casefold() == "watch":
        values = parse_qs(parsed.query).get("v")
        return values[0] if values else None
    if len(parts) >= 2 and parts[0].casefold() in {"shorts", "live"}:
        return parts[1]
    return None


def _resolution_failure(
    content_url: str,
    platform: Platform | None,
    *,
    method: str,
    status: str,
    reason: str,
) -> ContentAuthorResolution:
    return ContentAuthorResolution(
        content_url=content_url,
        platform=platform,
        resolved_profile_url=None,
        resolved_username_or_channel=None,
        resolution_method=method,
        confidence=0,
        status=status,
        reason=reason,
    )


class ApifyInstagramContentAuthorProvider(InstagramContentAuthorProvider):
    """Resolve Instagram owners using the official Apify Actor API."""

    def __init__(
        self,
        *,
        api_token: str | None,
        actor_id: str | None,
        timeout_seconds: float = 300,
        client: httpx.Client | None = None,
        raw_response_path: Path | None = None,
    ) -> None:
        if not api_token:
            raise ContentAuthorResolutionError(
                "APIFY_API_TOKEN is required for Instagram content-author resolution"
            )
        if not actor_id:
            raise ContentAuthorResolutionError(
                "APIFY_ACTOR_ID is required for Instagram content-author resolution"
            )
        if actor_id != "apify~instagram-scraper":
            raise ContentAuthorResolutionError(
                "Instagram content resolution currently requires "
                "APIFY_ACTOR_ID=apify~instagram-scraper because its post dataset "
                "contains ownerUsername"
            )
        self._api_token = api_token
        self.actor_id = actor_id
        self._timeout_seconds = timeout_seconds
        self._client = client
        self._raw_response_path = raw_response_path
        self.run_count = 0

    def _run_actor(self, client: httpx.Client, content_urls: list[str]) -> list[dict[str, Any]]:
        self.run_count += 1
        actor_path = quote(self.actor_id, safe="~")
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        actor_input = {
            "directUrls": content_urls,
            "resultsType": "posts",
            "resultsLimit": len(content_urls),
        }
        try:
            response = client.post(
                f"{APIFY_API_BASE_URL}/actors/{actor_path}/runs",
                headers=headers,
                json=actor_input,
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise ContentAuthorResolutionError(
                f"Apify content Actor start failed with HTTP {exc.response.status_code}"
            ) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise ContentAuthorResolutionError(
                "Apify content Actor could not start or returned invalid JSON"
            ) from exc
        run_data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(run_data, dict) or not isinstance(run_data.get("id"), str):
            raise ContentAuthorResolutionError("Apify content Actor response has no run id")
        run_id = run_data["id"]
        deadline = time.monotonic() + self._timeout_seconds
        terminal = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}
        while str(run_data.get("status", "")).upper() not in terminal:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ContentAuthorResolutionError(
                    "Apify content Actor did not finish before the configured timeout"
                )
            try:
                response = client.get(
                    f"{APIFY_API_BASE_URL}/actor-runs/{quote(run_id, safe='')}",
                    headers=headers,
                    params={"waitForFinish": max(1, min(30, int(remaining)))},
                    follow_redirects=True,
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPStatusError as exc:
                raise ContentAuthorResolutionError(
                    f"Apify content Actor status failed with HTTP {exc.response.status_code}"
                ) from exc
            except (httpx.RequestError, ValueError) as exc:
                raise ContentAuthorResolutionError(
                    "Apify content Actor status check returned invalid data"
                ) from exc
            next_data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(next_data, dict):
                raise ContentAuthorResolutionError(
                    "Apify content Actor status response has no run data"
                )
            run_data = next_data
        status = str(run_data.get("status", "")).upper()
        if status != "SUCCEEDED":
            raise ContentAuthorResolutionError(
                f"Apify content Actor finished with status {status}"
            )
        dataset_id = run_data.get("defaultDatasetId")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ContentAuthorResolutionError(
                "Apify content Actor response has no default dataset id"
            )
        try:
            response = client.get(
                f"{APIFY_API_BASE_URL}/datasets/{quote(dataset_id, safe='')}/items",
                headers=headers,
                params={"clean": "true", "format": "json"},
                follow_redirects=True,
            )
            response.raise_for_status()
            items = response.json()
        except httpx.HTTPStatusError as exc:
            raise ContentAuthorResolutionError(
                f"Apify content dataset read failed with HTTP {exc.response.status_code}"
            ) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise ContentAuthorResolutionError(
                "Apify content dataset is unavailable or invalid"
            ) from exc
        if not isinstance(items, list):
            raise ContentAuthorResolutionError("Apify content dataset is not a JSON list")
        usable = [item for item in items if isinstance(item, dict)]
        if self._raw_response_path is not None:
            self._raw_response_path.parent.mkdir(parents=True, exist_ok=True)
            self._raw_response_path.write_text(
                json.dumps(
                    {
                        "actor_id": self.actor_id,
                        "actor_input": actor_input,
                        "run_id": run_id,
                        "run_status": status,
                        "dataset_id": dataset_id,
                        "dataset_items": usable,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return usable

    @staticmethod
    def _owner_username(item: dict[str, Any]) -> str | None:
        candidates: list[object] = [
            item.get("ownerUsername"),
            item.get("owner_username"),
        ]
        owner = item.get("owner")
        if isinstance(owner, dict):
            candidates.extend((owner.get("username"), owner.get("userName")))
        for value in candidates:
            if isinstance(value, str) and INSTAGRAM_USERNAME_PATTERN.fullmatch(value.strip()):
                return value.strip()
        return None

    @staticmethod
    def _item_shortcode(item: dict[str, Any]) -> str | None:
        for key in ("shortCode", "shortcode", "code"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
        for key in ("inputUrl", "url", "postUrl"):
            value = item.get(key)
            if isinstance(value, str):
                shortcode = _instagram_shortcode(value)
                if shortcode:
                    return shortcode
        return None

    def resolve_authors(
        self,
        content_urls: list[str],
    ) -> dict[str, ContentAuthorResolution]:
        if not content_urls:
            return {}
        if len(content_urls) > 20:
            raise ContentAuthorResolutionError(
                "A single Instagram author-resolution run is limited to 20 URLs"
            )
        if self._client is not None:
            items = self._run_actor(self._client, content_urls)
        else:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                items = self._run_actor(client, content_urls)
        by_shortcode = {
            shortcode: item
            for item in items
            if (shortcode := self._item_shortcode(item)) is not None
        }
        results: dict[str, ContentAuthorResolution] = {}
        for content_url in content_urls:
            item = by_shortcode.get(_instagram_shortcode(content_url) or "")
            if item is None and len(content_urls) == 1 and len(items) == 1:
                item = items[0]
            owner = self._owner_username(item) if item is not None else None
            if owner is None:
                results[content_url] = _resolution_failure(
                    content_url,
                    Platform.INSTAGRAM,
                    method="apify_owner_username",
                    status="unresolved_author",
                    reason="Apify dataset has no confirmed ownerUsername for this content URL",
                )
                continue
            results[content_url] = ContentAuthorResolution(
                content_url=content_url,
                platform=Platform.INSTAGRAM,
                resolved_profile_url=f"https://www.instagram.com/{owner.casefold()}/",
                resolved_username_or_channel=owner,
                resolution_method="apify_owner_username",
                confidence=0.98,
                status="resolved_author",
                reason="ownerUsername confirmed by the public Apify post dataset",
            )
        return results


class YouTubeDataAuthorProvider(YouTubeContentAuthorProvider):
    """Resolve video authors through the public YouTube Data API."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = 30,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ContentAuthorResolutionError("YOUTUBE_API_KEY is empty")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._client = client

    def _request(self, client: httpx.Client, content_urls: list[str]) -> list[dict[str, Any]]:
        video_ids = [video_id for url in content_urls if (video_id := _youtube_video_id(url))]
        if not video_ids:
            return []
        try:
            response = client.get(
                YOUTUBE_VIDEOS_ENDPOINT,
                headers={"X-Goog-Api-Key": self._api_key},
                params={"part": "snippet", "id": ",".join(dict.fromkeys(video_ids))},
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise ContentAuthorResolutionError(
                f"YouTube Data API failed with HTTP {exc.response.status_code}"
            ) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise ContentAuthorResolutionError(
                "YouTube Data API is unavailable or returned invalid JSON"
            ) from exc
        items = payload.get("items") if isinstance(payload, dict) else None
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    def resolve_authors(
        self,
        content_urls: list[str],
    ) -> dict[str, ContentAuthorResolution]:
        if self._client is not None:
            items = self._request(self._client, content_urls)
        else:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                items = self._request(client, content_urls)
        by_video_id = {
            item["id"]: item
            for item in items
            if isinstance(item.get("id"), str)
        }
        results: dict[str, ContentAuthorResolution] = {}
        for content_url in content_urls:
            item = by_video_id.get(_youtube_video_id(content_url) or "")
            snippet = item.get("snippet") if isinstance(item, dict) else None
            channel_id = snippet.get("channelId") if isinstance(snippet, dict) else None
            channel_title = snippet.get("channelTitle") if isinstance(snippet, dict) else None
            if not isinstance(channel_id, str) or not channel_id.strip():
                continue
            results[content_url] = ContentAuthorResolution(
                content_url=content_url,
                platform=Platform.YOUTUBE_SHORTS,
                resolved_profile_url=f"https://www.youtube.com/channel/{channel_id.strip()}",
                resolved_username_or_channel=(
                    channel_title.strip()
                    if isinstance(channel_title, str) and channel_title.strip()
                    else channel_id.strip()
                ),
                resolution_method="youtube_data_api_channel_id",
                confidence=0.99,
                status="resolved_author",
                reason="channelId confirmed by YouTube Data API video.snippet",
            )
        return results


def _youtube_tavily_fallback(hit: SearchHit) -> ContentAuthorResolution | None:
    text = " ".join(value for value in (hit.title, hit.snippet) if value)
    urls = list(dict.fromkeys(match.group(0) for match in YOUTUBE_CHANNEL_URL_PATTERN.finditer(text)))
    if len(urls) == 1:
        match = YOUTUBE_CHANNEL_URL_PATTERN.search(urls[0])
        if match:
            path = match.group("path")
            identity = path.split("/", maxsplit=1)[-1].lstrip("@")
            return ContentAuthorResolution(
                content_url=hit.url,
                platform=Platform.YOUTUBE_SHORTS,
                resolved_profile_url=f"https://www.youtube.com/{path}",
                resolved_username_or_channel=identity,
                resolution_method="tavily_explicit_channel_url",
                confidence=0.80,
                status="resolved_author",
                reason="exactly one public YouTube channel URL is present in Tavily text",
            )
    handles = list(dict.fromkeys(YOUTUBE_HANDLE_PATTERN.findall(text)))
    if len(handles) == 1:
        handle = handles[0]
        return ContentAuthorResolution(
            content_url=hit.url,
            platform=Platform.YOUTUBE_SHORTS,
            resolved_profile_url=f"https://www.youtube.com/@{handle}",
            resolved_username_or_channel=handle,
            resolution_method="tavily_unique_channel_handle",
            confidence=0.67,
            status="resolved_author",
            reason="exactly one channel handle is present in Tavily title/snippet",
        )
    return None


class ContentAuthorResolver:
    """Classify hits and resolve at most the configured number of content URLs."""

    def __init__(
        self,
        *,
        instagram_provider: InstagramContentAuthorProvider | None,
        youtube_provider: YouTubeContentAuthorProvider | None,
        maximum_content_urls: int = 20,
        minimum_confidence: float = 0.65,
        cache_path: Path | None = None,
    ) -> None:
        if not 1 <= maximum_content_urls <= 20:
            raise ValueError("maximum_content_urls must be between 1 and 20")
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")
        self.instagram_provider = instagram_provider
        self.youtube_provider = youtube_provider
        self.maximum_content_urls = maximum_content_urls
        self.minimum_confidence = minimum_confidence
        self.cache_path = cache_path

    def _load_cache(self) -> dict[str, ContentAuthorResolution]:
        if self.cache_path is None or not self.cache_path.is_file():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            records = payload.get("resolutions") if isinstance(payload, dict) else None
            if not isinstance(records, dict):
                return {}
            return {
                url: ContentAuthorResolution.model_validate(value)
                for url, value in records.items()
                if isinstance(url, str) and isinstance(value, dict)
            }
        except (OSError, ValueError):
            LOGGER.warning("Ignoring invalid content-author cache: %s", self.cache_path)
            return {}

    def _save_cache(self, cache: dict[str, ContentAuthorResolution]) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "resolutions": {
                        url: result.model_dump(mode="json")
                        for url, result in sorted(cache.items())
                        if result.status == "resolved_author"
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _enforce_confidence(
        self,
        result: ContentAuthorResolution,
    ) -> ContentAuthorResolution:
        if result.status == "resolved_author" and result.confidence < self.minimum_confidence:
            return result.model_copy(
                update={
                    "status": "low_confidence",
                    "reason": (
                        f"author resolution confidence {result.confidence:.2f} is below "
                        f"MIN_AUTHOR_RESOLUTION_CONFIDENCE={self.minimum_confidence:.2f}"
                    ),
                }
            )
        return result

    def resolve(self, hit: SearchHit) -> ContentAuthorResolution:
        """Resolve one hit; batch callers should prefer :meth:`resolve_many`."""

        return self.resolve_many([hit])[0]

    def resolve_many(self, hits: list[SearchHit]) -> list[ContentAuthorResolution]:
        """Resolve content in one bounded batch while preserving input order."""

        results: list[ContentAuthorResolution | None] = [None] * len(hits)
        selected_content: list[tuple[int, SearchHit, Platform]] = []
        for index, hit in enumerate(hits):
            kind, platform, normalized = classify_public_url(hit.url)
            if kind == "profile" and platform is not None and normalized is not None:
                identity = [part for part in urlsplit(normalized).path.split("/") if part][-1]
                results[index] = ContentAuthorResolution(
                    content_url=hit.url,
                    platform=platform,
                    resolved_profile_url=normalized,
                    resolved_username_or_channel=identity.lstrip("@"),
                    resolution_method="direct_profile_url",
                    confidence=1.0,
                    status="profile_url",
                    reason="Tavily URL is already a canonical public author profile",
                )
            elif kind == "content" and platform is not None:
                if len(selected_content) < self.maximum_content_urls:
                    selected_content.append((index, hit, platform))
                else:
                    results[index] = _resolution_failure(
                        hit.url,
                        platform,
                        method="not_attempted",
                        status="skipped_resolution_limit",
                        reason=(
                            "content URL was not resolved because "
                            f"MAX_CONTENT_URLS_FOR_RESOLUTION={self.maximum_content_urls}"
                        ),
                    )
            else:
                results[index] = _resolution_failure(
                    hit.url,
                    platform,
                    method="not_applicable",
                    status="unsupported_url",
                    reason="URL is neither a supported public profile nor supported content",
                )

        cache = self._load_cache()
        pending_instagram: list[str] = []
        pending_youtube: list[str] = []
        selected_by_url: dict[str, list[tuple[int, SearchHit, Platform]]] = {}
        for index, hit, platform in selected_content:
            selected_by_url.setdefault(hit.url, []).append((index, hit, platform))
        for index, hit, platform in selected_content:
            cached = cache.get(hit.url)
            if cached is not None:
                results[index] = self._enforce_confidence(cached)
            elif platform == Platform.INSTAGRAM and hit.url not in pending_instagram:
                pending_instagram.append(hit.url)
            elif platform == Platform.YOUTUBE_SHORTS and hit.url not in pending_youtube:
                pending_youtube.append(hit.url)

        provider_results: dict[str, ContentAuthorResolution] = {}
        if pending_instagram:
            if self.instagram_provider is None:
                for url in pending_instagram:
                    provider_results[url] = _resolution_failure(
                        url,
                        Platform.INSTAGRAM,
                        method="apify_owner_username",
                        status="unresolved_author",
                        reason="Instagram author resolver is not configured",
                    )
            else:
                try:
                    provider_results.update(
                        self.instagram_provider.resolve_authors(pending_instagram)
                    )
                except Exception as exc:
                    safe_reason = (
                        str(exc)
                        if isinstance(exc, ContentAuthorResolutionError)
                        else f"{type(exc).__name__}: Instagram resolution batch failed"
                    )
                    for url in pending_instagram:
                        provider_results[url] = _resolution_failure(
                            url,
                            Platform.INSTAGRAM,
                            method="apify_owner_username",
                            status="unresolved_author",
                            reason=safe_reason[:2_000],
                        )

        if pending_youtube and self.youtube_provider is not None:
            try:
                provider_results.update(
                    self.youtube_provider.resolve_authors(pending_youtube)
                )
            except Exception as exc:
                LOGGER.warning(
                    "YouTube author batch could not be resolved: %s",
                    type(exc).__name__,
                )
        for url in pending_instagram + pending_youtube:
            provider_result = provider_results.get(url)
            for index, hit, platform in selected_by_url[url]:
                resolved = provider_result
                if resolved is None and platform == Platform.YOUTUBE_SHORTS:
                    resolved = _youtube_tavily_fallback(hit)
                if resolved is None:
                    resolved = _resolution_failure(
                        url,
                        platform,
                        method=(
                            "youtube_data_api_or_tavily_explicit_identity"
                            if platform == Platform.YOUTUBE_SHORTS
                            else "apify_owner_username"
                        ),
                        status="unresolved_author",
                        reason="No confirmed public author identity was returned",
                    )
                resolved = self._enforce_confidence(resolved)
                results[index] = resolved
                if resolved.status == "resolved_author":
                    cache[url] = resolved
        self._save_cache(cache)
        return [
            result
            if result is not None
            else _resolution_failure(
                hits[index].url,
                None,
                method="internal_guard",
                status="unresolved_author",
                reason="Resolution produced no result",
            )
            for index, result in enumerate(results)
        ]


def create_content_author_resolver(
    *,
    apify_api_token: str | None,
    apify_actor_id: str | None,
    youtube_api_key: str | None,
    timeout_seconds: float,
    maximum_content_urls: int,
    minimum_confidence: float,
    cache_path: Path,
    apify_raw_response_path: Path,
) -> ContentAuthorResolver:
    """Create network providers only for an explicitly requested real v2 run."""

    instagram = ApifyInstagramContentAuthorProvider(
        api_token=apify_api_token,
        actor_id=apify_actor_id,
        timeout_seconds=max(timeout_seconds, 300),
        raw_response_path=apify_raw_response_path,
    )
    youtube = (
        YouTubeDataAuthorProvider(
            api_key=youtube_api_key,
            timeout_seconds=timeout_seconds,
        )
        if youtube_api_key
        else None
    )
    return ContentAuthorResolver(
        instagram_provider=instagram,
        youtube_provider=youtube,
        maximum_content_urls=maximum_content_urls,
        minimum_confidence=minimum_confidence,
        cache_path=cache_path,
    )

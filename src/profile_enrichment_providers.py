"""Offline-testable enrichment of public Instagram source profiles."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import httpx
import pandas as pd
from pydantic import ValidationError

from src.models import (
    EnrichedSourceBlogger,
    ProfileEnrichmentAuditRow,
    ProfileEnrichmentAuditStatus,
    ProfileEnrichmentStatus,
    PublicInstagramPost,
    PublicInstagramProfile,
)


LOGGER = logging.getLogger(__name__)
APIFY_API_BASE_URL = "https://api.apify.com/v2"
CACHE_TTL = timedelta(hours=24)
INSTAGRAM_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9._]{1,30}$")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
INSTAGRAM_RESERVED_PATHS = {
    "about",
    "accounts",
    "developer",
    "direct",
    "explore",
    "p",
    "reel",
    "reels",
    "stories",
    "tv",
}
SUMMARY_COLUMNS = [
    "username",
    "profile_url",
    "full_name",
    "followers_count",
    "posts_count",
    "collected_posts",
    "engagement_rate",
    "is_private",
    "data_confidence",
    "enrichment_status",
    "enrichment_error",
]
AUDIT_COLUMNS = [
    "input_url",
    "username",
    "status",
    "reason",
    "cache_used",
    "fetched_at",
]


class ProfileEnrichmentError(RuntimeError):
    """Raised when a configured profile provider cannot return public data."""


class ProfileEnrichmentProvider(ABC):
    """Interface for a source-profile enrichment backend."""

    name: str
    actor_id: str | None = None

    @abstractmethod
    def fetch_profile(
        self,
        username: str,
        profile_url: str,
        posts_limit: int,
    ) -> EnrichedSourceBlogger:
        """Fetch one public profile without relying on Instagram credentials."""

        raise ProfileEnrichmentError(
            "ProfileEnrichmentProvider.fetch_profile must be implemented"
        )


@dataclass(frozen=True, slots=True)
class InstagramProfileInput:
    """Validated input URL and its canonical Instagram identity."""

    input_url: str
    username: str
    profile_url: str


@dataclass(frozen=True, slots=True)
class ProfileEnrichmentRun:
    """Ordered enriched records and a complete source-link audit."""

    bloggers: list[EnrichedSourceBlogger]
    audit_rows: list[ProfileEnrichmentAuditRow]


def canonical_instagram_profile_url(username: str) -> str:
    """Build a canonical public profile URL from a validated username."""

    if not INSTAGRAM_USERNAME_PATTERN.fullmatch(username):
        raise ValueError(f"Invalid Instagram username: {username!r}")
    return f"https://www.instagram.com/{username}/"


def extract_instagram_username(value: str | None) -> str | None:
    """Extract a profile username and reject post, Reel, or navigation URLs."""

    if not isinstance(value, str) or not value.strip():
        return None
    prepared = value.strip()
    if "://" not in prepared:
        prepared = f"https://{prepared}"
    try:
        parsed = urlsplit(prepared)
    except ValueError:
        return None
    host = parsed.netloc.casefold().split(":", maxsplit=1)[0]
    if host.startswith("www."):
        host = host[4:]
    if host not in {"instagram.com", "m.instagram.com"}:
        return None
    path_parts = [unquote(part) for part in parsed.path.split("/") if part]
    if not path_parts:
        return None
    username = path_parts[0]
    if username.casefold() in INSTAGRAM_RESERVED_PATHS:
        return None
    if not INSTAGRAM_USERNAME_PATTERN.fullmatch(username):
        return None
    return username


def calculate_engagement_rate(
    followers_count: int | None,
    posts: list[PublicInstagramPost],
) -> float | None:
    """Calculate average post ER, returning null when required data is absent."""

    if followers_count is None or followers_count <= 0:
        return None
    per_post_rates = [
        (post.likes_count + post.comments_count) / followers_count * 100
        for post in posts
        if post.likes_count is not None and post.comments_count is not None
    ]
    if not per_post_rates:
        return None
    return round(sum(per_post_rates) / len(per_post_rates), 4)


def _missing_profile_fields(profile: PublicInstagramProfile) -> list[str]:
    fields = [
        "full_name",
        "biography",
        "followers_count",
        "following_count",
        "posts_count",
        "is_verified",
        "is_private",
        "external_url",
        "profile_image_url",
    ]
    return [field for field in fields if getattr(profile, field) is None]


def assemble_enriched_blogger(
    profile: PublicInstagramProfile,
    posts: list[PublicInstagramPost],
    posts_limit: int,
) -> EnrichedSourceBlogger:
    """Apply common ER, confidence, missing-field, and status rules."""

    recent_posts = posts[:posts_limit]
    engagement_rate = calculate_engagement_rate(profile.followers_count, recent_posts)
    missing_fields = _missing_profile_fields(profile)
    if not recent_posts:
        missing_fields.append("recent_posts")
    if profile.followers_count is not None and profile.followers_count <= 0:
        missing_fields.append("positive_followers_count")
    if recent_posts and not any(
        post.likes_count is not None and post.comments_count is not None
        for post in recent_posts
    ):
        missing_fields.append("post_engagement_metrics")
    if engagement_rate is None:
        missing_fields.append("calculated_engagement_rate")
    missing_fields = list(dict.fromkeys(missing_fields))

    confidence_checks = [
        profile.full_name is not None,
        profile.biography is not None,
        profile.followers_count is not None,
        profile.following_count is not None,
        profile.posts_count is not None,
        profile.is_verified is not None,
        profile.is_private is not None,
        profile.profile_image_url is not None,
        bool(recent_posts),
        engagement_rate is not None,
    ]
    data_confidence = round(sum(confidence_checks) / len(confidence_checks), 2)
    essential_data_available = (
        profile.followers_count is not None
        and bool(recent_posts)
        and engagement_rate is not None
    )
    return EnrichedSourceBlogger(
        profile=profile,
        recent_posts=recent_posts,
        calculated_engagement_rate=engagement_rate,
        available_post_count=len(recent_posts),
        data_confidence=data_confidence,
        missing_fields=missing_fields,
        enrichment_status=(
            ProfileEnrichmentStatus.SUCCESS
            if essential_data_available
            else ProfileEnrichmentStatus.PARTIAL
        ),
        enrichment_error=None,
    )


def _http_url_or_none(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    prepared = value.strip()
    try:
        parsed = urlsplit(prepared)
    except ValueError:
        return None
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return None
    return prepared


def _optional_text(value: object, max_length: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text[:max_length] if text else None


def _public_content_text(value: object, max_length: int) -> str | None:
    """Keep useful public text while excluding email addresses and phone numbers."""

    text = _optional_text(value, max_length)
    if text is None:
        return None
    text = EMAIL_PATTERN.sub("[contact removed]", text)
    text = PHONE_PATTERN.sub("[contact removed]", text)
    return text


def _optional_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().replace(" ", "").replace(",", "")
        if normalized.isdigit():
            return int(normalized)
    return None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _optional_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1_000
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str) and value.strip():
        prepared = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(prepared)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _first_value(data: dict[str, Any], *keys: str) -> object:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _hashtags(raw_value: object, caption: str | None) -> list[str] | None:
    if isinstance(raw_value, list):
        values = [str(value).strip().lstrip("#") for value in raw_value]
        cleaned = [value for value in values if value]
        return cleaned or None
    if caption:
        found = re.findall(r"#([\w\u0400-\u04FF]+)", caption)
        return list(dict.fromkeys(found)) or None
    return None


def _parse_public_post(data: dict[str, Any]) -> PublicInstagramPost:
    caption = _public_content_text(_first_value(data, "caption", "text"), 20_000)
    return PublicInstagramPost(
        post_url=_http_url_or_none(_first_value(data, "url", "postUrl", "post_url")),
        post_type=_optional_text(
            _first_value(data, "type", "mediaType", "productType", "postType"), 100
        ),
        caption=caption,
        hashtags=_hashtags(_first_value(data, "hashtags", "tags"), caption),
        likes_count=_optional_count(
            _first_value(data, "likesCount", "likes", "likeCount")
        ),
        comments_count=_optional_count(
            _first_value(data, "commentsCount", "comments", "commentCount")
        ),
        timestamp=_optional_datetime(
            _first_value(
                data,
                "timestamp",
                "takenAtTimestamp",
                "takenAt",
                "taken_at",
            )
        ),
        display_url=_http_url_or_none(
            _first_value(data, "displayUrl", "display_url", "imageUrl")
        ),
        video_url=_http_url_or_none(
            _first_value(data, "videoUrl", "video_url")
        ),
        accessibility_caption=_public_content_text(
            _first_value(
                data,
                "accessibilityCaption",
                "accessibility_caption",
                "alt",
            ),
            5_000,
        ),
    )


def build_apify_actor_input(
    actor_id: str,
    username: str,
    profile_url: str,
    posts_limit: int,
) -> dict[str, object]:
    """Build a documented input payload for a supported official Apify Actor."""

    if actor_id == "apify~instagram-scraper":
        return {
            "directUrls": [profile_url],
            "resultsType": "details",
            "resultsLimit": posts_limit,
        }
    if actor_id == "apify~instagram-profile-scraper":
        return {"usernames": [username]}
    raise ProfileEnrichmentError(
        "Unsupported APIFY_ACTOR_ID input schema. Supported official Actors: "
        "apify~instagram-scraper and apify~instagram-profile-scraper."
    )


class MockProfileEnrichmentProvider(ProfileEnrichmentProvider):
    """Offline provider backed by explicit fictional profile templates."""

    name = "mock"

    def __init__(self, fixture_path: Path) -> None:
        try:
            payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProfileEnrichmentError(f"Mock enrichment file not found: {fixture_path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileEnrichmentError(
                f"Cannot read mock enrichment file {fixture_path}: {exc}"
            ) from exc
        if not isinstance(payload, list) or not payload or not all(
            isinstance(item, dict) for item in payload
        ):
            raise ProfileEnrichmentError(
                "Mock enrichment file must contain a non-empty JSON array of objects"
            )
        self._templates: list[dict[str, Any]] = payload

    def fetch_profile(
        self,
        username: str,
        profile_url: str,
        posts_limit: int,
    ) -> EnrichedSourceBlogger:
        digest = hashlib.sha256(username.casefold().encode("utf-8")).digest()
        template = self._templates[int.from_bytes(digest[:4], "big") % len(self._templates)]
        fetched_at = datetime.now(timezone.utc)
        profile = PublicInstagramProfile(
            username=username,
            profile_url=profile_url,
            full_name=_optional_text(template.get("full_name"), 500),
            biography=_public_content_text(template.get("biography"), 5_000),
            followers_count=_optional_count(template.get("followers_count")),
            following_count=_optional_count(template.get("following_count")),
            posts_count=_optional_count(template.get("posts_count")),
            is_verified=_optional_bool(template.get("is_verified")),
            is_private=_optional_bool(template.get("is_private")),
            external_url=_http_url_or_none(template.get("external_url")),
            profile_image_url=_http_url_or_none(template.get("profile_image_url")),
            raw_source=self.name,
            fetched_at=fetched_at,
        )
        raw_posts = template.get("recent_posts")
        posts = [
            _parse_public_post(item)
            for item in raw_posts if isinstance(item, dict)
        ] if isinstance(raw_posts, list) else []
        return assemble_enriched_blogger(profile, posts, posts_limit)


class ApifyProfileEnrichmentProvider(ProfileEnrichmentProvider):
    """Official Apify Actor API client for public Instagram data."""

    name = "apify"

    def __init__(
        self,
        api_token: str | None,
        actor_id: str | None,
        timeout_seconds: float = 300,
        client: httpx.Client | None = None,
        raw_response_path: Path | None = None,
    ) -> None:
        if not actor_id:
            raise ProfileEnrichmentError(
                "APIFY_ACTOR_ID is missing. Set it in .env, for example: "
                "APIFY_ACTOR_ID=apify~instagram-scraper."
            )
        if not api_token:
            raise ProfileEnrichmentError(
                "APIFY_API_TOKEN is missing. Add it to .env or use "
                "PROFILE_ENRICHMENT_PROVIDER=mock."
            )
        if not re.fullmatch(r"[A-Za-z0-9_-]+(?:~[A-Za-z0-9_-]+)?", actor_id):
            raise ProfileEnrichmentError("APIFY_ACTOR_ID has an invalid format")
        self._api_token = api_token
        self._actor_id = actor_id
        self.actor_id = actor_id
        self._timeout_seconds = timeout_seconds
        self._client = client
        self._raw_response_path = raw_response_path
        self._raw_response_lock = Lock()

    def _without_token(self, value: object) -> object:
        """Recursively redact the configured token before diagnostic persistence."""

        if isinstance(value, str):
            return value.replace(self._api_token, "[REDACTED]")
        if isinstance(value, list):
            return [self._without_token(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): self._without_token(item)
                for key, item in value.items()
                if str(key).casefold() not in {"authorization", "token", "api_token"}
            }
        return value

    def _capture_raw_response(
        self,
        *,
        username: str,
        profile_url: str,
        actor_input: dict[str, object],
        run_id: str,
        run_status: str,
        dataset_id: str,
        dataset_payload: object,
    ) -> None:
        """Save raw dataset items and request metadata, never request credentials."""

        if self._raw_response_path is None:
            return
        capture = {
            "actor_id": self._actor_id,
            "username": username,
            "profile_url": profile_url,
            "actor_input": actor_input,
            "run_id": run_id,
            "run_status": run_status,
            "dataset_id": dataset_id,
            "dataset_items": dataset_payload,
        }
        with self._raw_response_lock:
            document: dict[str, object] = {
                "actor_ids": [self._actor_id],
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "runs": [],
            }
            if self._raw_response_path.is_file():
                try:
                    existing = json.loads(
                        self._raw_response_path.read_text(encoding="utf-8")
                    )
                    if isinstance(existing, dict) and isinstance(
                        existing.get("runs"), list
                    ):
                        document = existing
                except (OSError, json.JSONDecodeError):
                    LOGGER.warning(
                        "Replacing unreadable Apify diagnostic file: %s",
                        self._raw_response_path,
                    )
            runs = document.get("runs")
            if not isinstance(runs, list):
                runs = []
                document["runs"] = runs
            runs.append(capture)
            actor_ids = {
                str(actor_id)
                for actor_id in document.get("actor_ids", [])
                if isinstance(actor_id, str) and actor_id
            }
            for run in runs:
                if not isinstance(run, dict):
                    continue
                run_actor_id = run.get("actor_id")
                if not run_actor_id and isinstance(run.get("actor_input"), dict):
                    actor_input_value = run["actor_input"]
                    if "directUrls" in actor_input_value:
                        run_actor_id = "apify~instagram-scraper"
                    elif "usernames" in actor_input_value:
                        run_actor_id = "apify~instagram-profile-scraper"
                    if run_actor_id:
                        run["actor_id"] = run_actor_id
                if run_actor_id:
                    actor_ids.add(str(run_actor_id))
            if not actor_ids and isinstance(document.get("actor_id"), str):
                actor_ids.add(str(document["actor_id"]))
            document.pop("actor_id", None)
            document["actor_ids"] = sorted(actor_ids)
            document["captured_at"] = datetime.now(timezone.utc).isoformat()
            safe_document = self._without_token(document)
            self._raw_response_path.parent.mkdir(parents=True, exist_ok=True)
            self._raw_response_path.write_text(
                json.dumps(safe_document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        LOGGER.info(
            "Saved raw Apify dataset response for %s to %s",
            username,
            self._raw_response_path,
        )

    def _fetch_with_client(
        self,
        client: httpx.Client,
        username: str,
        profile_url: str,
        posts_limit: int,
    ) -> EnrichedSourceBlogger:
        actor_path = quote(self._actor_id, safe="~")
        run_endpoint = f"{APIFY_API_BASE_URL}/actors/{actor_path}/runs"
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        actor_input = build_apify_actor_input(
            self._actor_id,
            username,
            profile_url,
            posts_limit,
        )
        try:
            response = client.post(
                run_endpoint,
                headers=headers,
                json=actor_input,
                follow_redirects=True,
            )
            response.raise_for_status()
            start_payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise ProfileEnrichmentError(
                f"Apify Actor start failed with HTTP {exc.response.status_code}. "
                "Check the token, Actor access, limits, and input contract."
            ) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise ProfileEnrichmentError(
                "Apify could not start the Actor or returned invalid JSON. "
                "Check the connection "
                "or use PROFILE_ENRICHMENT_PROVIDER=mock."
            ) from exc

        start_data = start_payload.get("data") if isinstance(start_payload, dict) else None
        if not isinstance(start_data, dict) or not isinstance(start_data.get("id"), str):
            raise ProfileEnrichmentError("Apify Actor start response has no run id")
        run_id = start_data["id"]
        run_data = start_data
        deadline = time.monotonic() + self._timeout_seconds
        terminal_statuses = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}
        while str(run_data.get("status", "")).upper() not in terminal_statuses:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise ProfileEnrichmentError(
                    f"Apify Actor run {run_id} did not finish within "
                    f"{self._timeout_seconds:g} seconds"
                )
            wait_seconds = max(1, min(60, int(remaining_seconds)))
            try:
                run_response = client.get(
                    f"{APIFY_API_BASE_URL}/actor-runs/{quote(run_id, safe='')}",
                    headers=headers,
                    params={"waitForFinish": wait_seconds},
                    follow_redirects=True,
                )
                run_response.raise_for_status()
                run_payload = run_response.json()
            except httpx.HTTPStatusError as exc:
                raise ProfileEnrichmentError(
                    f"Apify Actor status check failed with HTTP "
                    f"{exc.response.status_code}."
                ) from exc
            except (httpx.RequestError, ValueError) as exc:
                raise ProfileEnrichmentError(
                    "Apify Actor status check failed or returned invalid JSON."
                ) from exc
            next_run_data = (
                run_payload.get("data") if isinstance(run_payload, dict) else None
            )
            if not isinstance(next_run_data, dict):
                raise ProfileEnrichmentError(
                    "Apify Actor status response has no run data"
                )
            run_data = next_run_data

        final_status = str(run_data.get("status", "")).upper()
        if final_status != "SUCCEEDED":
            raise ProfileEnrichmentError(
                f"Apify Actor run {run_id} finished with status {final_status}"
            )
        dataset_id = run_data.get("defaultDatasetId") or start_data.get(
            "defaultDatasetId"
        )
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ProfileEnrichmentError(
                "Completed Apify Actor run has no defaultDatasetId"
            )
        try:
            dataset_response = client.get(
                f"{APIFY_API_BASE_URL}/datasets/{quote(dataset_id, safe='')}/items",
                headers=headers,
                params={"clean": "true", "format": "json", "limit": 10},
                follow_redirects=True,
            )
            dataset_response.raise_for_status()
            payload = dataset_response.json()
        except httpx.HTTPStatusError as exc:
            raise ProfileEnrichmentError(
                f"Apify dataset read failed with HTTP {exc.response.status_code}."
            ) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise ProfileEnrichmentError(
                "Apify dataset is unavailable or returned invalid JSON."
            ) from exc
        self._capture_raw_response(
            username=username,
            profile_url=profile_url,
            actor_input=actor_input,
            run_id=run_id,
            run_status=final_status,
            dataset_id=dataset_id,
            dataset_payload=payload,
        )
        if not isinstance(payload, list) or not payload:
            raise ProfileEnrichmentError("Apify Actor returned no dataset items")
        items = [item for item in payload if isinstance(item, dict)]
        if not items:
            raise ProfileEnrichmentError("Apify dataset contains no usable objects")
        return self._parse_items(username, profile_url, posts_limit, items)

    def _parse_items(
        self,
        username: str,
        profile_url: str,
        posts_limit: int,
        items: list[dict[str, Any]],
    ) -> EnrichedSourceBlogger:
        profile_item = max(
            items,
            key=lambda item: sum(
                key in item
                for key in (
                    "followersCount",
                    "followingCount",
                    "postsCount",
                    "biography",
                    "latestPosts",
                )
            ),
        )
        actor_error = _first_value(
            profile_item,
            "error",
            "errorDescription",
            "errorMessage",
        )
        if actor_error:
            raise ProfileEnrichmentError(
                "Apify Actor could not enrich this public profile: "
                f"{_optional_text(actor_error, 500) or 'unknown profile error'}"
            )
        fetched_at = datetime.now(timezone.utc)
        profile = PublicInstagramProfile(
            username=username,
            profile_url=profile_url,
            full_name=_optional_text(
                _first_value(profile_item, "fullName", "full_name", "name"), 500
            ),
            biography=_public_content_text(
                _first_value(profile_item, "biography", "bio"), 5_000
            ),
            followers_count=_optional_count(
                _first_value(profile_item, "followersCount", "followers")
            ),
            following_count=_optional_count(
                _first_value(
                    profile_item,
                    "followingCount",
                    "followsCount",
                    "following",
                )
            ),
            posts_count=_optional_count(
                _first_value(profile_item, "postsCount", "posts_count")
            ),
            is_verified=_optional_bool(
                _first_value(profile_item, "verified", "isVerified", "is_verified")
            ),
            is_private=_optional_bool(
                _first_value(profile_item, "private", "isPrivate", "is_private")
            ),
            external_url=_http_url_or_none(
                _first_value(profile_item, "externalUrl", "external_url")
            ),
            profile_image_url=_http_url_or_none(
                _first_value(
                    profile_item,
                    "profilePicUrlHD",
                    "profilePicUrlHd",
                    "profilePicUrl",
                    "profileImageUrl",
                    "profile_image_url",
                )
            ),
            raw_source=self.name,
            fetched_at=fetched_at,
        )

        raw_posts: list[dict[str, Any]] = []
        embedded = _first_value(
            profile_item,
            "latestPosts",
            "recentPosts",
            "recent_posts",
            "posts",
        )
        if isinstance(embedded, list):
            raw_posts.extend(item for item in embedded if isinstance(item, dict))
        for item in items:
            if item is profile_item:
                continue
            if any(key in item for key in ("likesCount", "commentsCount", "caption", "shortCode")):
                raw_posts.append(item)
        posts = [_parse_public_post(item) for item in raw_posts[:posts_limit]]
        return assemble_enriched_blogger(profile, posts, posts_limit)

    def fetch_profile(
        self,
        username: str,
        profile_url: str,
        posts_limit: int,
    ) -> EnrichedSourceBlogger:
        if self._client is not None:
            return self._fetch_with_client(
                self._client, username, profile_url, posts_limit
            )
        with httpx.Client(timeout=self._timeout_seconds) as client:
            return self._fetch_with_client(client, username, profile_url, posts_limit)


def create_profile_enrichment_provider(
    provider_name: str,
    mock_fixture_path: Path,
    apify_api_token: str | None,
    apify_actor_id: str | None,
    timeout_seconds: float,
    apify_raw_response_path: Path | None = None,
) -> ProfileEnrichmentProvider:
    """Construct the configured enrichment provider without starting a request."""

    if provider_name == "apify" or (apify_api_token and apify_actor_id):
        return ApifyProfileEnrichmentProvider(
            api_token=apify_api_token,
            actor_id=apify_actor_id,
            timeout_seconds=timeout_seconds,
            raw_response_path=apify_raw_response_path,
        )
    if provider_name == "mock":
        return MockProfileEnrichmentProvider(mock_fixture_path)
    raise ProfileEnrichmentError(f"Unsupported profile enrichment provider: {provider_name}")


def load_profile_cache(
    username: str,
    cache_dir: Path,
    now: datetime | None = None,
    ttl: timedelta = CACHE_TTL,
    expected_source: str | None = None,
) -> EnrichedSourceBlogger | None:
    """Load a validated cache item only when its file age is below the TTL."""

    cache_path = cache_dir / f"{username.casefold()}.json"
    if not cache_path.is_file():
        return None
    current_time = now or datetime.now(timezone.utc)
    try:
        modified_at = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        if current_time - modified_at >= ttl:
            return None
        cached = EnrichedSourceBlogger.model_validate_json(
            cache_path.read_text(encoding="utf-8")
        )
        if cached.enrichment_status == ProfileEnrichmentStatus.FAILED:
            LOGGER.info("Ignoring failed profile cache for %s", username)
            return None
        if expected_source is not None and cached.profile.raw_source != expected_source:
            LOGGER.info(
                "Ignoring %s cache for %s while using provider %s",
                cached.profile.raw_source,
                username,
                expected_source,
            )
            return None
        return cached
    except (OSError, ValidationError, ValueError) as exc:
        LOGGER.warning("Ignoring invalid profile cache for %s: %s", username, exc)
        return None


def save_profile_cache(blogger: EnrichedSourceBlogger, cache_dir: Path) -> Path:
    """Persist one profile without tokens, headers, or raw provider payloads."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{blogger.profile.username.casefold()}.json"
    cache_path.write_text(
        blogger.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return cache_path


def _safe_error_text(error: Exception) -> str:
    text = str(error)
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)(token=)[^&\s]+", r"\1[REDACTED]", text)
    return text[:2_000] or error.__class__.__name__


def _failed_blogger(
    profile_input: InstagramProfileInput,
    raw_source: str,
    error: Exception,
) -> EnrichedSourceBlogger:
    fetched_at = datetime.now(timezone.utc)
    return EnrichedSourceBlogger(
        profile=PublicInstagramProfile(
            username=profile_input.username,
            profile_url=profile_input.profile_url,
            full_name=None,
            biography=None,
            followers_count=None,
            following_count=None,
            posts_count=None,
            is_verified=None,
            is_private=None,
            external_url=None,
            profile_image_url=None,
            raw_source=raw_source,
            fetched_at=fetched_at,
        ),
        recent_posts=[],
        calculated_engagement_rate=None,
        available_post_count=0,
        data_confidence=0,
        missing_fields=[
            "full_name",
            "biography",
            "followers_count",
            "following_count",
            "posts_count",
            "is_verified",
            "is_private",
            "external_url",
            "profile_image_url",
            "recent_posts",
            "calculated_engagement_rate",
        ],
        enrichment_status=ProfileEnrichmentStatus.FAILED,
        enrichment_error=_safe_error_text(error),
    )


def _enrich_until_usable_limit(
    valid_inputs: list[InstagramProfileInput],
    initial_audit_rows: list[ProfileEnrichmentAuditRow],
    provider: ProfileEnrichmentProvider,
    posts_limit: int,
    cache_dir: Path,
    cache_enabled: bool,
    refresh_profiles: bool,
    limit_profiles: int,
    delay_seconds: float,
) -> ProfileEnrichmentRun:
    """Try subsequent URLs until the requested number has usable public data."""

    bloggers: list[EnrichedSourceBlogger] = []
    audit_rows = list(initial_audit_rows)
    usable_count = 0
    now = datetime.now(timezone.utc)

    for position, profile_input in enumerate(valid_inputs):
        if usable_count >= limit_profiles:
            audit_rows.append(
                ProfileEnrichmentAuditRow(
                    input_url=profile_input.input_url,
                    username=profile_input.username,
                    status=ProfileEnrichmentAuditStatus.SKIPPED_LIMIT,
                    reason="excluded_after_usable_profile_limit_reached",
                    cache_used=False,
                    fetched_at=None,
                )
            )
            continue

        cached = None
        if cache_enabled and not refresh_profiles:
            cached = load_profile_cache(
                profile_input.username,
                cache_dir,
                now=now,
                expected_source=provider.name,
            )
        cache_used = cached is not None
        if cached is not None:
            blogger = cached
        else:
            try:
                blogger = provider.fetch_profile(
                    profile_input.username,
                    profile_input.profile_url,
                    posts_limit,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Profile enrichment failed for %s; trying the next source URL: %s",
                    profile_input.username,
                    _safe_error_text(exc),
                )
                blogger = _failed_blogger(profile_input, provider.name, exc)
            if (
                cache_enabled
                and blogger.enrichment_status != ProfileEnrichmentStatus.FAILED
            ):
                try:
                    save_profile_cache(blogger, cache_dir)
                except OSError as exc:
                    LOGGER.warning(
                        "Could not cache profile %s: %s",
                        profile_input.username,
                        exc,
                    )

        bloggers.append(blogger)
        audit_rows.append(
            ProfileEnrichmentAuditRow(
                input_url=profile_input.input_url,
                username=profile_input.username,
                status=ProfileEnrichmentAuditStatus(blogger.enrichment_status.value),
                reason=(
                    "fresh_cache"
                    if cache_used
                    else blogger.enrichment_error or f"collected_by_{provider.name}"
                ),
                cache_used=cache_used,
                fetched_at=blogger.profile.fetched_at,
            )
        )
        if blogger.enrichment_status != ProfileEnrichmentStatus.FAILED:
            usable_count += 1
        if (
            not cache_used
            and delay_seconds > 0
            and usable_count < limit_profiles
            and position < len(valid_inputs) - 1
        ):
            time.sleep(delay_seconds)

    if usable_count < limit_profiles:
        LOGGER.warning(
            "Only %d usable profiles were found after checking %d valid source URLs",
            usable_count,
            len(valid_inputs),
        )
    return ProfileEnrichmentRun(bloggers=bloggers, audit_rows=audit_rows)


def enrich_profile_urls(
    input_urls: list[str | None],
    provider: ProfileEnrichmentProvider,
    posts_limit: int,
    cache_dir: Path,
    cache_enabled: bool,
    refresh_profiles: bool,
    limit_profiles: int | None,
    concurrency: int,
    delay_seconds: float,
    replace_failed_with_next: bool = False,
) -> ProfileEnrichmentRun:
    """Validate, deduplicate, limit, cache, and enrich profile URLs safely."""

    if limit_profiles is not None and limit_profiles <= 0:
        raise ValueError("--limit-profiles must be a positive integer")
    valid_inputs: list[InstagramProfileInput] = []
    audit_rows: list[ProfileEnrichmentAuditRow] = []
    seen_usernames: set[str] = set()

    for raw_url in input_urls:
        input_url = raw_url.strip() if isinstance(raw_url, str) else None
        username = extract_instagram_username(input_url)
        if username is None:
            audit_rows.append(
                ProfileEnrichmentAuditRow(
                    input_url=input_url,
                    username=None,
                    status=ProfileEnrichmentAuditStatus.INVALID_URL,
                    reason="not_a_public_instagram_profile_url",
                    cache_used=False,
                    fetched_at=None,
                )
            )
            continue
        normalized_username = username.casefold()
        if normalized_username in seen_usernames:
            audit_rows.append(
                ProfileEnrichmentAuditRow(
                    input_url=input_url,
                    username=username,
                    status=ProfileEnrichmentAuditStatus.DUPLICATE,
                    reason="duplicate_username",
                    cache_used=False,
                    fetched_at=None,
                )
            )
            continue
        seen_usernames.add(normalized_username)
        valid_inputs.append(
            InstagramProfileInput(
                input_url=input_url or "",
                username=username,
                profile_url=canonical_instagram_profile_url(username),
            )
        )

    if replace_failed_with_next and limit_profiles is not None:
        return _enrich_until_usable_limit(
            valid_inputs=valid_inputs,
            initial_audit_rows=audit_rows,
            provider=provider,
            posts_limit=posts_limit,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            refresh_profiles=refresh_profiles,
            limit_profiles=limit_profiles,
            delay_seconds=delay_seconds,
        )

    selected_count = len(valid_inputs) if limit_profiles is None else limit_profiles
    selected_inputs = valid_inputs[:selected_count]
    for skipped in valid_inputs[selected_count:]:
        audit_rows.append(
            ProfileEnrichmentAuditRow(
                input_url=skipped.input_url,
                username=skipped.username,
                status=ProfileEnrichmentAuditStatus.SKIPPED_LIMIT,
                reason="excluded_by_limit_profiles",
                cache_used=False,
                fetched_at=None,
            )
        )

    results_by_index: dict[int, EnrichedSourceBlogger] = {}
    pending: list[tuple[int, InstagramProfileInput]] = []
    now = datetime.now(timezone.utc)
    for index, profile_input in enumerate(selected_inputs):
        cached = None
        if cache_enabled and not refresh_profiles:
            cached = load_profile_cache(
                profile_input.username,
                cache_dir,
                now=now,
                expected_source=provider.name,
            )
        if cached is not None:
            results_by_index[index] = cached
            audit_rows.append(
                ProfileEnrichmentAuditRow(
                    input_url=profile_input.input_url,
                    username=profile_input.username,
                    status=ProfileEnrichmentAuditStatus(cached.enrichment_status.value),
                    reason="fresh_cache",
                    cache_used=True,
                    fetched_at=cached.profile.fetched_at,
                )
            )
        else:
            pending.append((index, profile_input))

    futures: dict[Future[EnrichedSourceBlogger], tuple[int, InstagramProfileInput]] = {}
    if pending:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for pending_index, (index, profile_input) in enumerate(pending):
                future = executor.submit(
                    provider.fetch_profile,
                    profile_input.username,
                    profile_input.profile_url,
                    posts_limit,
                )
                futures[future] = (index, profile_input)
                if delay_seconds > 0 and pending_index < len(pending) - 1:
                    time.sleep(delay_seconds)

            for future in as_completed(futures):
                index, profile_input = futures[future]
                try:
                    blogger = future.result()
                except Exception as exc:
                    LOGGER.warning(
                        "Profile enrichment failed for %s: %s",
                        profile_input.username,
                        _safe_error_text(exc),
                    )
                    blogger = _failed_blogger(profile_input, provider.name, exc)
                results_by_index[index] = blogger
                if (
                    cache_enabled
                    and blogger.enrichment_status != ProfileEnrichmentStatus.FAILED
                ):
                    try:
                        save_profile_cache(blogger, cache_dir)
                    except OSError as exc:
                        LOGGER.warning(
                            "Could not cache profile %s: %s",
                            profile_input.username,
                            exc,
                        )
                audit_rows.append(
                    ProfileEnrichmentAuditRow(
                        input_url=profile_input.input_url,
                        username=profile_input.username,
                        status=ProfileEnrichmentAuditStatus(
                            blogger.enrichment_status.value
                        ),
                        reason=(
                            blogger.enrichment_error
                            or f"collected_by_{provider.name}"
                        ),
                        cache_used=False,
                        fetched_at=blogger.profile.fetched_at,
                    )
                )

    bloggers = [results_by_index[index] for index in sorted(results_by_index)]
    return ProfileEnrichmentRun(bloggers=bloggers, audit_rows=audit_rows)


def save_enrichment_outputs(
    run: ProfileEnrichmentRun,
    json_path: Path,
    summary_path: Path,
    audit_path: Path,
) -> None:
    """Save structured JSON, a compact summary, and a safe URL-level audit."""

    for path in (json_path, summary_path, audit_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = [blogger.model_dump(mode="json") for blogger in run.bloggers]
    json_path.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_records = [
        {
            "username": blogger.profile.username,
            "profile_url": str(blogger.profile.profile_url),
            "full_name": blogger.profile.full_name,
            "followers_count": blogger.profile.followers_count,
            "posts_count": blogger.profile.posts_count,
            "collected_posts": blogger.available_post_count,
            "engagement_rate": blogger.calculated_engagement_rate,
            "is_private": blogger.profile.is_private,
            "data_confidence": blogger.data_confidence,
            "enrichment_status": blogger.enrichment_status.value,
            "enrichment_error": blogger.enrichment_error,
        }
        for blogger in run.bloggers
    ]
    pd.DataFrame(summary_records, columns=SUMMARY_COLUMNS).to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
    )
    audit_records = [row.model_dump(mode="json") for row in run.audit_rows]
    pd.DataFrame(audit_records, columns=AUDIT_COLUMNS).to_csv(
        audit_path,
        index=False,
        encoding="utf-8",
    )
    LOGGER.info(
        "Saved %d enriched profiles, %d audit rows",
        len(run.bloggers),
        len(run.audit_rows),
    )

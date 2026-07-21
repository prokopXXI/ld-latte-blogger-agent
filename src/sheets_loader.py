"""Load strict mock CSVs and inspect unknown public Google Sheets schemas."""

import io
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar
from urllib.parse import parse_qs, urlsplit

import httpx
import pandas as pd
from pydantic import BaseModel, ValidationError

from src.models import (
    CandidateProfile,
    RawPlatform,
    RawSourceBlogger,
    SourceBlogger,
    SourceInspectionReport,
    SourceValidationStatus,
)


LOGGER = logging.getLogger(__name__)


class DataLoadError(ValueError):
    """Raised when a CSV source is missing, malformed, or unavailable."""


SOURCE_REQUIRED_COLUMNS = {
    "handle",
    "display_name",
    "platform",
    "profile_url",
    "followers",
    "engagement_rate_pct",
    "average_views",
    "content_topics",
    "audience_description",
    "audience_interests",
    "content_style",
    "content_formats",
    "tone",
    "visual_style",
    "advertising_load_pct",
    "price_segment",
    "brand_safety",
    "location",
    "collaboration_notes",
}

CANDIDATE_REQUIRED_COLUMNS = {
    "handle",
    "display_name",
    "platform",
    "profile_url",
    "followers",
    "engagement_rate_pct",
    "average_views",
    "content_topics",
    "audience_description",
    "audience_interests",
    "content_style",
    "content_formats",
    "tone",
    "visual_style",
    "advertising_load_pct",
    "price_segment",
    "brand_safety",
    "location",
    "source",
    "source_url",
    "notes",
}

LIST_COLUMNS = {
    "content_topics",
    "audience_interests",
    "content_formats",
    "tone",
    "visual_style",
}
RAW_REQUIRED_FIELDS = {"name", "profile_url"}
RAW_EXPECTED_FIELDS = {
    "name",
    "profile_url",
    "platform",
    "notes",
    "engagement_rate",
}
COLUMN_SYNONYMS = {
    "name": {
        "name",
        "display_name",
        "blogger",
        "blogger_name",
        "author",
        "блогер",
        "имя",
        "имя_блогера",
        "автор",
        "название",
        "ник",
        "никнейм",
        "аккаунт",
    },
    "profile_url": {
        "profile_url",
        "url",
        "link",
        "profile",
        "profile_link",
        "ссылка",
        "профиль",
        "ссылка_на_профиль",
        "ссылка_на_блогера",
        "url_профиля",
    },
    "platform": {
        "platform",
        "social_network",
        "social",
        "channel",
        "площадка",
        "соцсеть",
        "социальная_сеть",
        "канал",
    },
    "notes": {
        "notes",
        "collaboration_notes",
        "note",
        "comment",
        "comments",
        "description",
        "заметки",
        "комментарий",
        "комментарии",
        "описание",
        "примечание",
    },
    "engagement_rate": {
        "engagement_rate",
        "engagement_rate_pct",
        "engagement",
        "er",
        "er_pct",
        "вовлеченность",
        "вовлечённость",
        "коэффициент_вовлеченности",
        "коэффициент_вовлечённости",
    },
}
RecordT = TypeVar("RecordT", bound=BaseModel)
RawScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class SourceInspectionResult:
    """Raw row models plus the aggregate inspection report."""

    rows: list[RawSourceBlogger]
    report: SourceInspectionReport


def _split_pipe_values(value: object, *, column: str, row_number: int) -> list[str]:
    if not isinstance(value, str):
        raise DataLoadError(
            f"Row {row_number}: column {column!r} must contain text separated by '|'"
        )
    items = [item.strip() for item in value.split("|") if item.strip()]
    if not items:
        raise DataLoadError(f"Row {row_number}: column {column!r} cannot be empty")
    return items


def _format_validation_error(error: ValidationError) -> str:
    details: list[str] = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"])
        details.append(f"{location}: {item['msg']}")
    return "; ".join(details)


def load_csv_frame(path: Path) -> pd.DataFrame:
    """Load any UTF-8 CSV without assuming its column schema."""

    if not path.exists():
        raise DataLoadError(f"CSV file not found: {path}")
    if not path.is_file():
        raise DataLoadError(f"CSV path is not a file: {path}")
    try:
        return pd.read_csv(path, encoding="utf-8")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Cannot read CSV file {path}: {exc}") from exc


def load_unknown_csv_frame(path: Path) -> pd.DataFrame:
    """Load a local unknown-schema CSV with the same header detection as Sheets."""

    if not path.exists():
        raise DataLoadError(f"CSV file not found: {path}")
    if not path.is_file():
        raise DataLoadError(f"CSV path is not a file: {path}")
    try:
        csv_text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Cannot read CSV file {path}: {exc}") from exc
    return _parse_downloaded_csv(csv_text, str(path))


def _load_records(
    path: Path,
    model_type: type[RecordT],
    required_columns: set[str],
) -> list[RecordT]:
    frame = load_csv_frame(path)
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise DataLoadError(f"CSV file {path} is missing required columns: {missing}")
    if frame.empty:
        raise DataLoadError(f"CSV file contains no data rows: {path}")

    records: list[RecordT] = []
    for row_number, raw_record in enumerate(frame.to_dict(orient="records"), start=2):
        record = dict(raw_record)
        for column in LIST_COLUMNS:
            record[column] = _split_pipe_values(
                record[column],
                column=column,
                row_number=row_number,
            )
        try:
            records.append(model_type.model_validate(record))
        except ValidationError as exc:
            details = _format_validation_error(exc)
            raise DataLoadError(f"Invalid row {row_number} in {path}: {details}") from exc
    return records


def extract_spreadsheet_id(sheet_url: str) -> str:
    """Extract and validate a spreadsheet id from a normal Google Sheets URL."""

    parsed = urlsplit(sheet_url.strip())
    if parsed.netloc.casefold() != "docs.google.com":
        raise DataLoadError("GOOGLE_SHEET_URL must use https://docs.google.com/spreadsheets/")
    match = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", parsed.path)
    if not match:
        raise DataLoadError("Could not extract spreadsheet_id from GOOGLE_SHEET_URL")
    return match.group(1)


def extract_google_sheet_gid(sheet_url: str, default_gid: int = 0) -> int:
    """Read `gid` from either URL query or fragment, falling back to a default."""

    parsed = urlsplit(sheet_url.strip())
    query_gid = parse_qs(parsed.query).get("gid")
    fragment_gid = parse_qs(parsed.fragment).get("gid")
    raw_gid = (query_gid or fragment_gid or [str(default_gid)])[0]
    if not raw_gid.isdigit():
        raise DataLoadError(f"Google Sheets gid must be a non-negative integer; got {raw_gid!r}")
    return int(raw_gid)


def build_google_sheets_export_url(sheet_url: str, gid: int | None = None) -> str:
    """Build a fixed HTTPS CSV export URL without copying arbitrary query params."""

    spreadsheet_id = extract_spreadsheet_id(sheet_url)
    effective_gid = extract_google_sheet_gid(sheet_url) if gid is None else gid
    if effective_gid < 0:
        raise DataLoadError("Google Sheets gid must be non-negative")
    return (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export"
        f"?format=csv&gid={effective_gid}"
    )


def _parse_downloaded_csv(csv_text: str, source_label: str) -> pd.DataFrame:
    try:
        headerless_frame = pd.read_csv(io.StringIO(csv_text), header=None)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Downloaded Google Sheet is not a readable CSV: {exc}") from exc

    first_row = headerless_frame.iloc[0].tolist()
    header_mapping = detect_column_mapping(first_row)
    first_row_contains_url = any(normalize_source_url(value) for value in first_row)
    if header_mapping or not first_row_contains_url:
        frame = pd.read_csv(io.StringIO(csv_text))
    else:
        frame = headerless_frame
        frame.columns = [f"column_{index}" for index in range(1, len(frame.columns) + 1)]
        LOGGER.info(
            "No header row detected in %s; assigned columns=%s",
            source_label,
            list(frame.columns),
        )

    frame = frame.dropna(how="all").reset_index(drop=True)
    LOGGER.info(
        "Loaded %d rows from %s; columns=%s",
        len(frame),
        source_label,
        list(frame.columns),
    )
    return frame


def _download_with_client(
    client: httpx.Client,
    export_url: str,
    destination: Path,
) -> pd.DataFrame:
    try:
        response = client.get(export_url, follow_redirects=True)
    except httpx.RequestError as exc:
        raise DataLoadError(
            "Google Sheet is unavailable because the network request failed: "
            f"{exc}. Check the connection and sharing settings."
        ) from exc

    if response.status_code in {401, 403, 404}:
        raise DataLoadError(
            f"Google Sheet is closed or unavailable (HTTP {response.status_code}). "
            "The owner must set General access to 'Anyone with the link' and role 'Viewer'."
        )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise DataLoadError(
            f"Google Sheets CSV export failed with HTTP {response.status_code}."
        ) from exc

    content_type = response.headers.get("content-type", "").casefold()
    try:
        csv_text = response.content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataLoadError("Google Sheets response is not valid UTF-8") from exc
    html_prefix = csv_text.lstrip()[:100].casefold()
    if "text/html" in content_type or html_prefix.startswith(("<!doctype html", "<html")):
        raise DataLoadError(
            "Google Sheet returned an HTML login/access page instead of CSV. "
            "The owner must enable 'Anyone with the link' with Viewer access."
        )

    frame = _parse_downloaded_csv(csv_text, export_url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(csv_text, encoding="utf-8")
    LOGGER.info("Saved a read-only local copy to %s", destination)
    return frame


def download_google_sheet_csv(
    sheet_url: str,
    gid: int,
    timeout_seconds: float,
    destination: Path,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Download a public sheet as CSV without cookies or Google credentials."""

    export_url = build_google_sheets_export_url(sheet_url, gid)
    if client is not None:
        return _download_with_client(client, export_url, destination)
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as owned_client:
        return _download_with_client(owned_client, export_url, destination)


def _normalize_column_name(column: object) -> str:
    normalized = str(column).strip().casefold().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", "_", normalized).strip("_")


def detect_column_mapping(columns: list[object]) -> dict[str, str]:
    """Map Russian and English source headers to canonical raw fields."""

    normalized_columns = {str(column): _normalize_column_name(column) for column in columns}
    mapping: dict[str, str] = {}
    for canonical, synonyms in COLUMN_SYNONYMS.items():
        normalized_synonyms = {_normalize_column_name(value) for value in synonyms}
        for original, normalized in normalized_columns.items():
            if normalized in normalized_synonyms:
                mapping[canonical] = original
                break
    return mapping


def detect_frame_mapping(frame: pd.DataFrame) -> dict[str, str]:
    """Map headers first, then infer a URL column from its actual values."""

    mapping = detect_column_mapping(list(frame.columns))
    if "profile_url" in mapping or frame.empty:
        return mapping

    best_column: str | None = None
    best_ratio = 0.0
    for column in frame.columns:
        values = frame[column].dropna().tolist()
        if not values:
            continue
        valid_count = sum(normalize_source_url(value) is not None for value in values)
        ratio = valid_count / len(values)
        if ratio > best_ratio:
            best_column = str(column)
            best_ratio = ratio
    if best_column is not None and best_ratio >= 0.5:
        mapping["profile_url"] = best_column
        LOGGER.info(
            "Inferred profile_url column %r from values (valid URL ratio %.0f%%)",
            best_column,
            best_ratio * 100,
        )
    return mapping


def detect_platform_from_source_url(url: str | None) -> RawPlatform:
    """Determine a source platform from its public URL domain only."""

    if not url:
        return RawPlatform.UNKNOWN
    prepared = url.strip()
    if "://" not in prepared:
        prepared = f"https://{prepared}"
    parsed = urlsplit(prepared)
    host = parsed.netloc.casefold().split(":", maxsplit=1)[0]
    if host.startswith("www."):
        host = host[4:]
    if "." not in host or any(character.isspace() for character in host):
        return RawPlatform.UNKNOWN
    if host == "instagram.com":
        return RawPlatform.INSTAGRAM
    if host in {"youtube.com", "youtu.be"}:
        return RawPlatform.YOUTUBE
    if host in {"t.me", "telegram.me"}:
        return RawPlatform.TELEGRAM
    return RawPlatform.UNKNOWN


def normalize_source_url(value: object) -> str | None:
    """Normalize a syntactically valid HTTP(S) URL without opening it."""

    if not isinstance(value, str) or not value.strip():
        return None
    prepared = value.strip()
    if "://" not in prepared:
        prepared = f"https://{prepared}"
    parsed = urlsplit(prepared)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return None
    host = parsed.netloc.casefold().split(":", maxsplit=1)[0]
    if host.startswith("www."):
        host = host[4:]
    if "." not in host or any(character.isspace() for character in host):
        return None
    path = "/" + "/".join(part for part in parsed.path.split("/") if part)
    if path == "/":
        path = ""
    return f"https://{host}{path}"


def _clean_scalar(value: object) -> RawScalar:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return str(value).strip() or None


def _mapped_text(record: dict[str, RawScalar], mapping: dict[str, str], field: str) -> str | None:
    column = mapping.get(field)
    value = record.get(column) if column else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_raw_source_bloggers(
    frame: pd.DataFrame,
    mapping: dict[str, str],
) -> list[RawSourceBlogger]:
    """Convert every unknown-schema row without inventing absent values."""

    rows: list[RawSourceBlogger] = []
    for raw_record in frame.to_dict(orient="records"):
        cleaned = {str(key): _clean_scalar(value) for key, value in raw_record.items()}
        original_name = _mapped_text(cleaned, mapping, "name")
        raw_url = _mapped_text(cleaned, mapping, "profile_url")
        profile_url = normalize_source_url(raw_url)
        platform = detect_platform_from_source_url(profile_url)
        source_notes = _mapped_text(cleaned, mapping, "notes")

        missing_data = [
            field
            for field in sorted(RAW_EXPECTED_FIELDS)
            if not _mapped_text(cleaned, mapping, field)
        ]
        if raw_url and profile_url is None:
            missing_data.append("valid_profile_url")
        if platform == RawPlatform.UNKNOWN:
            missing_data.append("supported_platform")
        missing_data = list(dict.fromkeys(missing_data))

        if profile_url is None:
            status = SourceValidationStatus.INVALID
        elif original_name and platform != RawPlatform.UNKNOWN:
            status = SourceValidationStatus.VALID
        else:
            status = SourceValidationStatus.PARTIAL
        rows.append(
            RawSourceBlogger(
                original_name=original_name,
                profile_url=profile_url,
                platform=platform,
                source_notes=source_notes,
                raw_fields=cleaned,
                validation_status=status,
                missing_data=missing_data,
            )
        )
    return rows


def inspect_source_frame(
    frame: pd.DataFrame,
    source_url: str,
    sheet_gid: int | None,
) -> SourceInspectionResult:
    """Inspect columns and URL coverage without running downstream business logic."""

    original_columns = [str(column) for column in frame.columns]
    mapping = detect_frame_mapping(frame)
    rows = build_raw_source_bloggers(frame, mapping)
    mapped_columns = set(mapping.values())
    platform_distribution = Counter(row.platform.value for row in rows)
    rows_with_urls = sum(row.profile_url is not None for row in rows)
    report = SourceInspectionReport(
        source_url=source_url,
        sheet_gid=sheet_gid,
        row_count=len(frame),
        original_columns=original_columns,
        detected_mapping=mapping,
        unmapped_columns=[column for column in original_columns if column not in mapped_columns],
        missing_required_fields=sorted(RAW_REQUIRED_FIELDS - set(mapping)),
        rows_with_valid_urls=rows_with_urls,
        rows_without_urls=len(rows) - rows_with_urls,
        platform_distribution=dict(sorted(platform_distribution.items())),
        inspection_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return SourceInspectionResult(rows=rows, report=report)


def save_source_inspection(report: SourceInspectionReport, path: Path) -> None:
    """Save aggregate diagnostics only; raw row contents are intentionally omitted."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def safe_source_preview(rows: list[RawSourceBlogger], limit: int = 3) -> list[dict[str, object]]:
    """Return a shortened console preview without notes or arbitrary raw fields."""

    preview: list[dict[str, object]] = []
    for index, row in enumerate(rows[:limit], start=1):
        preview.append(
            {
                "row": index,
                "name": (row.original_name[:40] if row.original_name else None),
                "profile_url": (row.profile_url[:100] if row.profile_url else None),
                "platform": row.platform.value,
                "status": row.validation_status.value,
                "missing_data": row.missing_data,
            }
        )
    return preview


def load_source_bloggers(path: Path) -> list[SourceBlogger]:
    """Load normalized source bloggers for the existing scoring pipeline."""

    return _load_records(path, SourceBlogger, SOURCE_REQUIRED_COLUMNS)


def load_candidate_profiles(path: Path) -> list[CandidateProfile]:
    """Load prepared public candidates and return validated profiles."""

    return _load_records(path, CandidateProfile, CANDIDATE_REQUIRED_COLUMNS)

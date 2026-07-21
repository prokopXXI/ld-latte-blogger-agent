"""Offline tests for public Google Sheets loading and raw schema inspection."""

from pathlib import Path

import httpx
import pandas as pd
import pytest

from src.models import RawPlatform, SourceValidationStatus
from src.sheets_loader import (
    DataLoadError,
    build_google_sheets_export_url,
    detect_column_mapping,
    detect_frame_mapping,
    detect_platform_from_source_url,
    download_google_sheet_csv,
    extract_google_sheet_gid,
    extract_spreadsheet_id,
    inspect_source_frame,
    load_source_bloggers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "source_bloggers.example.csv"
SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1J1HtHP-CMQ8skFOuFtspu2GhmSzeWPcehmrAK-P6gR4/edit#gid=42"
)


def test_extracts_spreadsheet_id() -> None:
    assert extract_spreadsheet_id(SHEET_URL) == (
        "1J1HtHP-CMQ8skFOuFtspu2GhmSzeWPcehmrAK-P6gR4"
    )


@pytest.mark.parametrize(
    ("url", "expected_gid"),
    [
        (SHEET_URL, 42),
        (SHEET_URL.replace("#gid=42", "?gid=17"), 17),
        (SHEET_URL.replace("#gid=42", ""), 0),
    ],
)
def test_extracts_gid(url: str, expected_gid: int) -> None:
    assert extract_google_sheet_gid(url) == expected_gid


def test_builds_safe_csv_export_url() -> None:
    assert build_google_sheets_export_url(SHEET_URL, gid=7) == (
        "https://docs.google.com/spreadsheets/d/"
        "1J1HtHP-CMQ8skFOuFtspu2GhmSzeWPcehmrAK-P6gR4/export?format=csv&gid=7"
    )


def test_maps_russian_and_english_column_synonyms() -> None:
    mapping = detect_column_mapping(
        ["Имя блогера", "profile_url", "Соцсеть", "Комментарий", "ER"]
    )

    assert mapping == {
        "name": "Имя блогера",
        "profile_url": "profile_url",
        "platform": "Соцсеть",
        "notes": "Комментарий",
        "engagement_rate": "ER",
    }


@pytest.mark.parametrize(
    ("url", "platform"),
    [
        ("https://instagram.com/fashion_author", RawPlatform.INSTAGRAM),
        ("https://youtube.com/@fashion_author", RawPlatform.YOUTUBE),
        ("https://youtu.be/example", RawPlatform.YOUTUBE),
        ("https://t.me/fashion_author", RawPlatform.TELEGRAM),
        ("https://example.org/fashion_author", RawPlatform.UNKNOWN),
    ],
)
def test_detects_platform_from_url(url: str, platform: RawPlatform) -> None:
    assert detect_platform_from_source_url(url) == platform


def test_missing_columns_produce_diagnostic_report() -> None:
    frame = pd.DataFrame([{"Комментарий": "Строка без имени и ссылки"}])

    inspection = inspect_source_frame(frame, "local.csv", sheet_gid=None)

    assert inspection.report.missing_required_fields == ["name", "profile_url"]
    assert inspection.report.rows_without_urls == 1
    assert inspection.rows[0].validation_status == SourceValidationStatus.INVALID
    assert "profile_url" in inspection.rows[0].missing_data


@pytest.mark.parametrize("status_code", [403, 404])
def test_closed_or_missing_sheet_has_clear_error(
    status_code: int,
    tmp_path: Path,
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status_code, request=request)
    )
    with httpx.Client(transport=transport) as client:
        with pytest.raises(DataLoadError, match="Anyone with the link.*Viewer"):
            download_google_sheet_csv(
                SHEET_URL,
                gid=0,
                timeout_seconds=5,
                destination=tmp_path / "source.csv",
                client=client,
            )


def test_google_csv_download_is_mocked_and_saved(tmp_path: Path) -> None:
    csv_body = "Блогер,Ссылка\nТест,https://instagram.com/test_author\n"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/csv; charset=utf-8"},
            text=csv_body,
            request=request,
        )
    )
    destination = tmp_path / "source.csv"

    with httpx.Client(transport=transport) as client:
        frame = download_google_sheet_csv(
            SHEET_URL,
            gid=0,
            timeout_seconds=5,
            destination=destination,
            client=client,
        )

    assert len(frame) == 1
    assert destination.read_text(encoding="utf-8") == csv_body


def test_headerless_google_csv_is_detected_and_url_column_is_inferred(
    tmp_path: Path,
) -> None:
    csv_body = (
        "1,https://instagram.com/first_author\n"
        ",\n"
        "2,https://youtube.com/@second_author\n"
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/csv; charset=utf-8"},
            text=csv_body,
            request=request,
        )
    )

    with httpx.Client(transport=transport) as client:
        frame = download_google_sheet_csv(
            SHEET_URL,
            gid=0,
            timeout_seconds=5,
            destination=tmp_path / "headerless.csv",
            client=client,
        )

    assert list(frame.columns) == ["column_1", "column_2"]
    assert len(frame) == 2
    assert detect_frame_mapping(frame) == {"profile_url": "column_2"}


def test_existing_csv_source_mode_still_loads() -> None:
    bloggers = load_source_bloggers(SOURCE_PATH)

    assert len(bloggers) == 5

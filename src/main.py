"""CLI for source inspection and the blogger-selection pipeline."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.candidate_enricher import (
    discover_candidates,
    mark_below_min_score,
    save_search_audit,
)
from src.candidate_ranker import rank_candidates, select_ranked_candidates
from src.config import ConfigurationError, Settings, configure_logging, load_settings
from src.content_author_resolver import ContentAuthorResolutionError
from src.llm_profile_analyzer import (
    LLMAnalysisError,
    build_ideal_profile_from_enriched,
)
from src.final_offer_generator import FinalOfferGenerationError
from src.final_pipeline import (
    FinalPipelineError,
    FinalRunPlan,
    SCORE_BREAKDOWN_COLUMNS,
    SavedPoolPlan,
    finalize_saved_v2_pool,
    run_final_pipeline,
    run_final_pipeline_v2,
)
from src.models import (
    BarterOffer,
    FinalScoreBreakdownRow,
    PipelineResult,
    RankedCandidate,
    ResultRow,
    SourceBlogger,
)
from src.offer_generator import generate_offers
from src.profile_analyzer import build_ideal_profile
from src.profile_enrichment_providers import (
    ProfileEnrichmentError,
    create_profile_enrichment_provider,
    enrich_profile_urls,
    save_enrichment_outputs,
)
from src.search_providers import SearchProviderError, create_search_provider
from src.search_queries import generate_search_queries, save_search_queries
from src.sheets_loader import DataLoadError, load_source_bloggers
from src.sheets_loader import (
    RAW_EXPECTED_FIELDS,
    download_google_sheet_csv,
    detect_frame_mapping,
    inspect_source_frame,
    load_csv_frame,
    load_unknown_csv_frame,
    safe_source_preview,
    save_source_inspection,
)


LOGGER = logging.getLogger(__name__)
RESULT_COLUMNS = [
    "name",
    "platform",
    "profile_url",
    "total_score",
    "topic_score",
    "visual_score",
    "audience_score",
    "tone_score",
    "engagement_score",
    "ad_load_score",
    "price_segment_score",
    "format_score",
    "brand_safety_score",
    "similar_to",
    "match_reason",
    "personalized_offer",
    "source_query",
    "source_title",
    "source_snippet",
    "data_confidence",
]


def _result_rows(
    ranked_candidates: list[RankedCandidate],
    offers: list[BarterOffer],
) -> list[ResultRow]:
    offers_by_handle = {offer.candidate_handle: offer for offer in offers}
    if len(offers_by_handle) != len(offers):
        raise ValueError("Generated offers contain duplicate candidate handles")

    rows: list[ResultRow] = []
    for ranked in ranked_candidates:
        offer = offers_by_handle.get(ranked.candidate.handle)
        if offer is None:
            raise ValueError(f"No offer generated for {ranked.candidate.handle}")
        candidate = ranked.candidate
        score = ranked.evaluation.score
        rows.append(
            ResultRow(
                name=candidate.display_name,
                platform=candidate.platform,
                profile_url=candidate.profile_url,
                total_score=score.total_score,
                topic_score=score.topic_score,
                visual_score=score.visual_score,
                audience_score=score.audience_score,
                tone_score=score.tone_score,
                engagement_score=score.engagement_score,
                ad_load_score=score.ad_load_score,
                price_segment_score=score.price_segment_score,
                format_score=score.format_score,
                brand_safety_score=score.brand_safety_score,
                similar_to=ranked.evaluation.similar_to,
                match_reason=ranked.evaluation.match_reason,
                personalized_offer=offer.message,
                source_query=candidate.source_query,
                source_title=candidate.source_title,
                source_snippet=candidate.source_snippet,
                data_confidence=candidate.data_confidence,
            )
        )
    return rows


def save_results(rows: list[ResultRow], output_path: Path) -> None:
    """Write validated rows with a stable, interview-friendly column order."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [row.model_dump(mode="json") for row in rows]
    pd.DataFrame(records, columns=RESULT_COLUMNS).to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )
    LOGGER.info("Saved %d rows to %s", len(rows), output_path)


def save_mock_score_breakdown(
    ranked_candidates: list[RankedCandidate],
    output_path: Path,
) -> None:
    """Save the same explainable schema for the fully offline mock pipeline."""

    rows: list[FinalScoreBreakdownRow] = []
    for ranked in ranked_candidates:
        candidate = ranked.candidate
        evaluation = ranked.evaluation
        score = evaluation.score

        def reason(key: str) -> str:
            detail = evaluation.criterion_details.get(key)
            if detail is None:
                return "Недостаточно данных для уверенной оценки. confidence: low"
            if "confidence:" in detail.reason.casefold():
                return detail.reason
            return f"{detail.reason} confidence: {detail.confidence}"

        rows.append(
            FinalScoreBreakdownRow(
                profile_url=candidate.profile_url,
                platform=candidate.platform,
                total_score=score.total_score,
                max_score=100,
                evidence_count=1,
                topic_score=score.topic_score,
                topic_reason=reason("topic"),
                visual_score=score.visual_score,
                visual_reason=reason("visual"),
                audience_score=score.audience_score,
                audience_reason=reason("audience"),
                tone_score=score.tone_score,
                tone_reason=reason("tone"),
                engagement_score=score.engagement_score,
                engagement_reason=reason("engagement"),
                ad_load_score=score.ad_load_score,
                ad_load_reason=reason("ad_load"),
                price_segment_score=score.price_segment_score,
                price_segment_reason=reason("price_segment"),
                format_score=score.format_score,
                format_reason=reason("format"),
                brand_safety_score=score.brand_safety_score,
                brand_safety_reason=reason("brand_safety"),
                data_confidence=candidate.data_confidence,
                confidence_reason=(
                    f"data_confidence={candidate.data_confidence:.3f}; mock-score "
                    "не добавляет отдельные баллы за полноту данных."
                ),
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [row.model_dump(mode="json") for row in rows],
        columns=SCORE_BREAKDOWN_COLUMNS,
    ).to_csv(output_path, index=False, encoding="utf-8")


def _load_pipeline_source(settings: Settings) -> list[SourceBlogger]:
    if settings.source_provider == "csv":
        return load_source_bloggers(settings.source_csv_path)
    if not settings.google_sheet_url:
        raise ConfigurationError(
            "GOOGLE_SHEET_URL is required when SOURCE_PROVIDER=google_sheets"
        )
    download_google_sheet_csv(
        sheet_url=settings.google_sheet_url,
        gid=settings.google_sheet_gid,
        timeout_seconds=settings.request_timeout_seconds,
        destination=settings.source_real_csv_path,
    )
    try:
        return load_source_bloggers(settings.source_real_csv_path)
    except DataLoadError as exc:
        raise DataLoadError(
            "The Google Sheet was downloaded, but its raw schema is not ready for "
            "the scoring pipeline. Run `python -m src.main --inspect-source` first. "
            f"Details: {exc}"
        ) from exc


def inspect_selected_source(settings: Settings) -> None:
    """Inspect the selected source and exit before search, scoring, or offers."""

    if settings.source_provider == "csv":
        frame = load_csv_frame(settings.source_csv_path)
        source_url = str(settings.source_csv_path)
        sheet_gid = None
    else:
        if not settings.google_sheet_url:
            raise ConfigurationError(
                "GOOGLE_SHEET_URL is required when SOURCE_PROVIDER=google_sheets"
            )
        frame = download_google_sheet_csv(
            sheet_url=settings.google_sheet_url,
            gid=settings.google_sheet_gid,
            timeout_seconds=settings.request_timeout_seconds,
            destination=settings.source_real_csv_path,
        )
        source_url = settings.google_sheet_url
        sheet_gid = settings.google_sheet_gid

    inspection = inspect_source_frame(frame, source_url, sheet_gid)
    save_source_inspection(inspection.report, settings.source_inspection_path)
    detected_fields = set(inspection.report.detected_mapping)
    fields_for_later_enrichment = sorted(RAW_EXPECTED_FIELDS - detected_fields)

    print("\nДиагностика исходной базы")
    print(f"Source provider: {settings.source_provider}")
    print(f"Колонки ({len(inspection.report.original_columns)}):")
    print(json.dumps(inspection.report.original_columns, ensure_ascii=False))
    print(f"Строк: {inspection.report.row_count}")
    print("Detected mapping:")
    print(json.dumps(inspection.report.detected_mapping, ensure_ascii=False, indent=2))
    print(
        "Не найдены обязательные поля: "
        f"{inspection.report.missing_required_fields or 'нет'}"
    )
    print(
        "Поля для последующего обогащения: "
        f"{fields_for_later_enrichment or 'нет'}"
    )
    print("Первые 3 строки (сокращённо, без заметок и дополнительных полей):")
    print(json.dumps(safe_source_preview(inspection.rows), ensure_ascii=False, indent=2))
    print(f"Отчёт сохранён: {settings.source_inspection_path}")


def _load_enrichment_source_frame(settings: Settings) -> pd.DataFrame:
    """Load live Google data when selected, otherwise prefer its local snapshot."""

    if settings.source_provider == "google_sheets":
        if not settings.google_sheet_url:
            raise ConfigurationError(
                "GOOGLE_SHEET_URL is required when SOURCE_PROVIDER=google_sheets"
            )
        return download_google_sheet_csv(
            sheet_url=settings.google_sheet_url,
            gid=settings.google_sheet_gid,
            timeout_seconds=settings.request_timeout_seconds,
            destination=settings.source_real_csv_path,
        )
    if settings.source_real_csv_path.is_file():
        LOGGER.info(
            "Using the local Google Sheets snapshot for enrichment: %s",
            settings.source_real_csv_path,
        )
        return load_unknown_csv_frame(settings.source_real_csv_path)
    return load_csv_frame(settings.source_csv_path)


def enrich_selected_source(
    settings: Settings,
    limit_profiles: int | None,
    refresh_profiles: bool,
) -> None:
    """Enrich source Instagram links without search, scoring, LLM, or offers."""

    frame = _load_enrichment_source_frame(settings)
    mapping = detect_frame_mapping(frame)
    profile_url_column = mapping.get("profile_url")
    if profile_url_column is None:
        input_urls: list[str | None] = [None] * len(frame)
    else:
        input_urls = [
            value.strip() if isinstance(value, str) and value.strip() else None
            for value in frame[profile_url_column].tolist()
        ]

    provider = create_profile_enrichment_provider(
        provider_name=settings.profile_enrichment_provider,
        mock_fixture_path=settings.profile_enrichment_mock_path,
        apify_api_token=settings.apify_api_token,
        apify_actor_id=settings.apify_actor_id,
        timeout_seconds=max(settings.request_timeout_seconds, 300),
        apify_raw_response_path=settings.apify_raw_response_path,
    )
    run = enrich_profile_urls(
        input_urls=input_urls,
        provider=provider,
        posts_limit=settings.profile_posts_limit,
        cache_dir=settings.profile_cache_dir,
        cache_enabled=settings.profile_cache_enabled,
        refresh_profiles=refresh_profiles,
        limit_profiles=limit_profiles,
        concurrency=settings.profile_enrichment_concurrency,
        delay_seconds=settings.profile_enrichment_delay_seconds,
        replace_failed_with_next=provider.name == "apify",
    )
    save_enrichment_outputs(
        run,
        json_path=settings.enriched_source_json_path,
        summary_path=settings.enriched_source_summary_path,
        audit_path=settings.profile_enrichment_audit_path,
    )
    if not run.bloggers:
        raise DataLoadError(
            "No valid Instagram profile URLs were available for enrichment. "
            f"Review {settings.profile_enrichment_audit_path}."
        )

    cached_count = sum(row.cache_used for row in run.audit_rows)
    failed_count = sum(blogger.enrichment_status.value == "failed" for blogger in run.bloggers)
    usable_count = len(run.bloggers) - failed_count
    invalid_count = sum(row.status.value == "invalid_url" for row in run.audit_rows)
    skipped_count = sum(row.status.value == "skipped_limit" for row in run.audit_rows)
    print("\nОбогащение исходных Instagram-профилей завершено")
    print(f"Profile enrichment provider: {provider.name}")
    if provider.actor_id:
        print(f"Apify Actor: {provider.actor_id}")
    print(f"Проверено профилей: {len(run.bloggers)}")
    print(f"Получены пригодные данные: {usable_count}")
    print(f"Использовано из кэша: {cached_count}")
    print(f"Ошибок профилей: {failed_count}")
    print(f"Невалидных исходных ссылок: {invalid_count}")
    print(f"Пропущено из-за --limit-profiles: {skipped_count}")
    print(f"JSON сохранён: {settings.enriched_source_json_path}")
    print(f"Сводка сохранена: {settings.enriched_source_summary_path}")
    print(f"Аудит сохранён: {settings.profile_enrichment_audit_path}")
    if provider.name == "apify":
        print(f"Сырой ответ Apify сохранён: {settings.apify_raw_response_path}")


def run_pipeline(settings: Settings) -> PipelineResult:
    """Execute source analysis, discovery, audit, scoring, and offer generation."""

    if not settings.mock_mode:
        raise ConfigurationError(
            "This version keeps LLM features disabled. Set MOCK_MODE=true; "
            "SEARCH_PROVIDER may still be mock or tavily."
        )

    source_bloggers = _load_pipeline_source(settings)
    LOGGER.info("Loaded %d source bloggers", len(source_bloggers))
    ideal_profile = build_ideal_profile(source_bloggers)

    queries = generate_search_queries(ideal_profile)
    save_search_queries(queries, settings.search_queries_path)
    provider = create_search_provider(settings)
    hits = provider.search(queries, settings.search_max_results)
    discovery = discover_candidates(hits, source_bloggers)

    ranked_all = rank_candidates(
        discovery.candidates,
        ideal_profile,
        source_bloggers,
    )
    save_mock_score_breakdown(ranked_all, settings.final_score_breakdown_path)
    qualified_count = sum(
        item.evaluation.score.total_score >= settings.min_score for item in ranked_all
    )
    selected = select_ranked_candidates(
        ranked_all,
        min_score=settings.min_score,
        top_k=settings.top_k,
    )
    below_score_urls = {
        str(item.candidate.profile_url)
        for item in ranked_all
        if item.evaluation.score.total_score < settings.min_score
    }
    audit_rows = mark_below_min_score(discovery.audit_rows, below_score_urls)
    save_search_audit(audit_rows, settings.search_audit_path)

    offers = generate_offers(selected, source_bloggers)
    pipeline_result = PipelineResult(
        ideal_profile=ideal_profile,
        selected_candidates=selected,
        offers=offers,
    )
    save_results(_result_rows(selected, offers), settings.results_csv_path)

    print("\nПайплайн завершён успешно")
    print(f"Search provider: {provider.name}")
    print(f"Поисковых запросов создано: {len(queries)}")
    print(f"Результатов найдено: {discovery.total_found}")
    print(f"После очистки осталось: {len(discovery.candidates)}")
    print(f"Прошли MIN_SCORE={settings.min_score}: {qualified_count}")
    print(f"В итоговый результат вошло: {len(selected)} из максимум {settings.top_k}")
    for item in selected:
        print(
            f"  {item.rank}. {item.candidate.display_name} — "
            f"{item.evaluation.score.total_score}/100"
        )
    print(f"Результат сохранён: {settings.results_csv_path}")
    print(f"Аудит поиска сохранён: {settings.search_audit_path}")
    return pipeline_result


def build_selected_ideal_profile(settings: Settings, dry_run: bool) -> None:
    """Build or inspect the source LLM portrait without search or scoring."""

    run = build_ideal_profile_from_enriched(settings, dry_run=dry_run)
    title = "Dry-run LLM-анализа завершён" if dry_run else "Портрет блогера построен"
    print(f"\n{title}")
    print(f"LLM provider: {run.summary.provider}")
    print(f"Модель: {run.summary.model}")
    print(f"Пригодных профилей: {run.summary.profile_count}")
    print(f"Пакетов: {run.summary.batch_count}")
    print(f"Примерный объём входа: {run.summary.approximate_characters} символов")
    if dry_run:
        print("Сетевые запросы: 0")
        print("Платные вызовы OpenAI не выполнялись")
        print(f"Dry-run audit сохранён: {settings.llm_analysis_audit_path}")
        return
    print(f"Успешно обработано пакетов: {run.audit.completed_batches}")
    print(f"Batch insights: {settings.llm_batch_insights_path}")
    print(f"JSON-портрет: {settings.ideal_blogger_profile_json_path}")
    print(f"Markdown-портрет: {settings.ideal_blogger_profile_markdown_path}")
    print(f"Audit: {settings.llm_analysis_audit_path}")


def _print_final_plan(plan: FinalRunPlan) -> None:
    """Show hard cost ceilings before any external client is created."""

    print(f"\nПлан полного реального запуска ({plan.pipeline_version})")
    print(f"Поисковых запросов Tavily: {plan.query_count}")
    print(f"Максимум Tavily-результатов: {plan.tavily_result_limit}")
    print(
        "Кандидатов до enrichment: не более "
        f"{plan.max_candidates_before_enrichment}"
    )
    print(f"Instagram-профилей для Apify: не более {plan.max_apify_candidates}")
    if plan.pipeline_version == "v2":
        print(
            "Content URLs для определения автора: не более "
            f"{plan.max_content_urls_for_resolution}"
        )
        print(
            "Apify Actor runs для content resolution: не более "
            f"{plan.max_apify_content_resolution_runs} (один пакет)"
        )
        print(
            "Apify Actor runs для profile enrichment: не более "
            f"{plan.max_apify_candidates}"
        )
        print(
            "MIN_AUTHOR_RESOLUTION_CONFIDENCE: "
            f"{plan.min_author_resolution_confidence:.2f}"
        )
    print(f"OpenAI-офферов: не более {plan.max_openai_offers}")
    print(f"FINAL_MIN_SCORE: {plan.final_min_score}")
    print(f"OpenAI model: {plan.openai_model}")
    print(
        "Готовность ключей: "
        + (
            f"не заполнены {', '.join(plan.missing_credentials)}"
            if plan.missing_credentials
            else "все необходимые переменные заполнены"
        )
    )


def find_real_bloggers(settings: Settings, dry_run: bool, use_v2: bool = True) -> None:
    """Print the plan, then execute only after an explicit non-dry command."""

    runner = run_final_pipeline_v2 if use_v2 else run_final_pipeline
    preview = runner(settings, dry_run=True)
    _print_final_plan(preview.plan)
    if dry_run:
        print("Сетевые запросы: 0")
        print("Tavily, Apify и OpenAI не запускались")
        print("Файлы результатов не изменялись")
        return
    result = runner(settings, dry_run=False)
    print("\nРеальный поиск завершён")
    print(f"Финалистов: {len(result.finalists)}")
    for index, finalist in enumerate(result.finalists, start=1):
        print(
            f"  {index}. {finalist.candidate.name} — "
            f"{finalist.score.total_score}/100, needs_review"
        )
    print(
        "CSV: "
        f"{settings.final_real_bloggers_csv_v2_path if use_v2 else settings.final_real_bloggers_csv_path}"
    )
    print(
        "Markdown: "
        f"{settings.final_real_bloggers_markdown_v2_path if use_v2 else settings.final_real_bloggers_markdown_path}"
    )
    print("Сообщения не отправлялись")


def _print_saved_pool_plan(plan: SavedPoolPlan) -> None:
    print("\nПлан финализации сохранённого v2-пула")
    print(f"Canonical profiles в пуле: {plan.pool_size}")
    print(f"Готовый сохранённый enrichment: {plan.saved_enriched_count}")
    print(f"Дополнительный свежий cache: {plan.fresh_cache_count}")
    print(
        "Всего профилей без нового Apify-вызова: "
        f"{plan.saved_enriched_count + plan.fresh_cache_count}"
    )
    print(f"Потребуют нового Apify-вызова: {plan.apify_required_count}")
    print(
        "MAX_CANDIDATES_FOR_APIFY: "
        f"{plan.max_candidates_for_apify} (только сохранённый пул)"
    )
    print(f"Публикаций на профиль: максимум {plan.posts_per_profile}")
    print(f"OpenAI-офферов: максимум {plan.max_openai_offers}")
    print(f"Tavily-вызовов: {plan.tavily_calls}")
    print("Порог recommended: 70+")
    print("Резерв manual_review: 60–69")
    print("Rejected: ниже 60")


def finalize_saved_pool_command(settings: Settings, dry_run: bool) -> None:
    result = finalize_saved_v2_pool(settings, dry_run=dry_run)
    _print_saved_pool_plan(result.plan)
    if dry_run:
        print("Сетевые запросы: 0")
        print("Tavily, Apify и OpenAI не запускались")
        print("Итоговые файлы не изменялись")
        return
    print("\nСохранённый пул обработан")
    print(f"Оценено: {len(result.ranked)}")
    print(f"Recommended: {len(result.recommended)}")
    print(f"Manual review: {len(result.manual_review)}")
    print(f"Rejected: {len(result.rejected)}")
    print(f"Черновиков офферов: {len(result.offer_targets)}")
    print(f"Все кандидаты: {settings.final_all_candidates_csv_path}")
    print(f"Markdown: {settings.final_all_candidates_markdown_path}")
    print(f"Recommended CSV: {settings.final_recommended_bloggers_csv_path}")
    print("Сообщения не отправлялись")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LD LATTE fashion blogger agent")
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument(
        "--inspect-source",
        action="store_true",
        help="inspect the selected source and stop before search and scoring",
    )
    commands.add_argument(
        "--enrich-source",
        action="store_true",
        help="enrich source Instagram profiles and stop before search and scoring",
    )
    commands.add_argument(
        "--build-ideal-profile",
        action="store_true",
        help="analyze enriched source profiles and build a structured ideal portrait",
    )
    commands.add_argument(
        "--find-real-bloggers",
        action="store_true",
        help="run Tavily, Apify, scoring, and OpenAI drafts for real candidates",
    )
    commands.add_argument(
        "--finalize-saved-pool",
        action="store_true",
        help="score the saved v2 canonical pool without running Tavily",
    )
    parser.add_argument(
        "--limit-profiles",
        type=int,
        default=None,
        help="process only the first N valid Instagram profiles",
    )
    parser.add_argument(
        "--refresh-profiles",
        action="store_true",
        help="ignore profile cache and fetch selected profiles again",
    )
    parser.add_argument(
        "--dry-run-llm",
        action="store_true",
        help="estimate ideal-profile LLM requests without making network calls",
    )
    parser.add_argument(
        "--dry-run-final",
        action="store_true",
        help="show real-search cost ceilings without external API calls",
    )
    parser.add_argument(
        "--dry-run-final-v2",
        action="store_true",
        help="show content-author-resolution v2 ceilings without external API calls",
    )
    parser.add_argument(
        "--dry-run-saved-pool",
        action="store_true",
        help="show saved-pool cache and API counts without external calls",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Configure the app, report expected errors clearly, and return an exit code."""

    try:
        args = _parse_args(argv)
        settings = load_settings()
        configure_logging(settings.log_level)
        if args.limit_profiles is not None and args.limit_profiles <= 0:
            raise ConfigurationError("--limit-profiles must be a positive integer")
        if (args.limit_profiles is not None or args.refresh_profiles) and not args.enrich_source:
            raise ConfigurationError(
                "--limit-profiles and --refresh-profiles require --enrich-source"
            )
        if args.dry_run_llm and not args.build_ideal_profile:
            raise ConfigurationError(
                "--dry-run-llm requires --build-ideal-profile"
            )
        if args.dry_run_final and not args.find_real_bloggers:
            raise ConfigurationError(
                "--dry-run-final requires --find-real-bloggers"
            )
        if args.dry_run_final_v2 and not args.find_real_bloggers:
            raise ConfigurationError(
                "--dry-run-final-v2 requires --find-real-bloggers"
            )
        if args.dry_run_final and args.dry_run_final_v2:
            raise ConfigurationError(
                "Choose either --dry-run-final or --dry-run-final-v2"
            )
        if args.dry_run_saved_pool and not args.finalize_saved_pool:
            raise ConfigurationError(
                "--dry-run-saved-pool requires --finalize-saved-pool"
            )
        if args.finalize_saved_pool and (
            args.dry_run_final or args.dry_run_final_v2
        ):
            raise ConfigurationError(
                "Final-search dry-run flags cannot be combined with --finalize-saved-pool"
            )
        if args.inspect_source:
            inspect_selected_source(settings)
        elif args.enrich_source:
            enrich_selected_source(
                settings,
                limit_profiles=args.limit_profiles,
                refresh_profiles=args.refresh_profiles,
            )
        elif args.build_ideal_profile:
            build_selected_ideal_profile(settings, dry_run=args.dry_run_llm)
        elif args.find_real_bloggers:
            find_real_bloggers(
                settings,
                dry_run=args.dry_run_final or args.dry_run_final_v2,
                use_v2=args.dry_run_final_v2 or not args.dry_run_final,
            )
        elif args.finalize_saved_pool:
            finalize_saved_pool_command(
                settings,
                dry_run=args.dry_run_saved_pool,
            )
        else:
            run_pipeline(settings)
        return 0
    except (
        ConfigurationError,
        ContentAuthorResolutionError,
        DataLoadError,
        LLMAnalysisError,
        FinalPipelineError,
        FinalOfferGenerationError,
        ProfileEnrichmentError,
        SearchProviderError,
        ValidationError,
        OSError,
        ValueError,
    ) as exc:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")
        LOGGER.error("Pipeline failed: %s", exc)
        return 1
    except Exception:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")
        LOGGER.exception("Pipeline failed because of an unexpected error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

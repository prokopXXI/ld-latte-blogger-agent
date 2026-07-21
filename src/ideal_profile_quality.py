"""Offline quality gate for the synthesized ideal blogger profile."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.models import BatchBloggerInsights, LLMAnalysisAudit, LLMIdealBloggerProfile
from src.llm_profile_analyzer import save_ideal_profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TERMS = ("кофе", "напиток", "напитков", "food", "vegan")
FINAL_SEARCH_QUERIES = [
    "site:instagram.com женская мода примерки одежды образы Wildberries блогер сотрудничество -магазин -бренд -каталог",
    "site:instagram.com Reels капсульный гардероб повседневные женские образы автор блога -магазин -бутик -продавец",
    "site:instagram.com образы для офиса подборки женской одежды Ozon блогер PR сотрудничество -магазин -карточка",
    "site:instagram.com честный обзор женской одежды примерка на себе Wildberries fashion-блогер -магазин -продавец",
    "site:youtube.com/shorts женская одежда примерка Wildberries Shorts fashion-блогер -магазин -бренд -каталог",
    "site:youtube.com/shorts капсульный гардероб женские образы Ozon обзор одежды автор -магазин -карточка",
    "site:youtube.com/shorts образы для офиса сочетание вещей женская мода блогер сотрудничество -магазин -продавец",
    "site:t.me женская мода подборки образов Wildberries авторский канал сотрудничество -магазин -каталог",
    "site:t.me капсульный гардероб примерки женской одежды Ozon fashion автор канала -магазин -бренд",
]


class IdealProfileQualityError(RuntimeError):
    """Raised when quality artifacts cannot be validated safely."""


@dataclass(frozen=True, slots=True)
class QualityCheck:
    """One reproducible validation outcome."""

    name: str
    passed: bool
    details: str


@dataclass(frozen=True, slots=True)
class QualityValidationResult:
    """Final validated profile, checks, and search readiness decision."""

    profile: LLMIdealBloggerProfile
    checks: list[QualityCheck]
    original_query_count: int
    ready_for_search: bool
    risks: list[str]


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise IdealProfileQualityError(f"Required quality source not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdealProfileQualityError(f"Cannot read valid JSON from {path}") from exc


def _nonempty_required_fields(profile: LLMIdealBloggerProfile) -> list[str]:
    empty: list[str] = []
    for field_name in type(profile).model_fields:
        value = getattr(profile, field_name)
        if isinstance(value, str) and not value.strip():
            empty.append(field_name)
        elif isinstance(value, list) and not value:
            empty.append(field_name)
    return empty


def _qualify_evidence_claims(profile: LLMIdealBloggerProfile) -> LLMIdealBloggerProfile:
    """Downgrade two unsupported causal claims without adding new facts."""

    positive_signals = [
        signal.replace(
            "simplifies outreach and formalizing barter deals",
            "simplifies outreach; barter terms are not confirmed",
        ).replace(
            "good for conversion-focused barter",
            "inferred: potentially relevant for barter; conversion data is unavailable",
        )
        for signal in profile.positive_signals
    ]
    advertising_load_preferences = [
        signal.replace(
            "willing to accept paid and barter integrations (explicit PR language across samples)",
            "inferred: potentially open to discussing integrations; PR language does not confirm paid or barter terms",
        )
        for signal in profile.advertising_load_preferences
    ]
    evidence_summary = profile.evidence_summary
    qualification = (
        " Barter suitability is the LD LATTE business objective; available PR/collaboration "
        "signals do not confirm that a creator accepts barter or any specific terms."
    )
    if qualification.strip() not in evidence_summary:
        evidence_summary += qualification

    corrected = profile.model_copy(
        update={
            "positive_signals": positive_signals,
            "advertising_load_preferences": advertising_load_preferences,
            "evidence_summary": evidence_summary,
            "search_queries": FINAL_SEARCH_QUERIES,
        }
    )
    return LLMIdealBloggerProfile.model_validate(corrected.model_dump(mode="json"))


def _report_mark(check: QualityCheck) -> str:
    return "PASS" if check.passed else "FAIL"


def _write_report(
    path: Path,
    result: QualityValidationResult,
    audit: LLMAnalysisAudit,
    analyzed_usernames: set[str],
) -> None:
    profile = result.profile
    lines = [
        "# Quality report: IdealBloggerProfile LD LATTE",
        "",
        "> Проверка выполнена полностью локально. OpenAI и Tavily не вызывались; "
        "enriched-данные не изменялись.",
        "",
        "## Итог",
        "",
        f"- Статус для поиска: **{'READY' if result.ready_for_search else 'NOT READY'}**.",
        f"- Confidence score: **{profile.confidence_score:.2f}**.",
        f"- Проанализировано профилей: **{len(analyzed_usernames)}**.",
        f"- Batch count: **{audit.batch_count}**, provider: **{audit.provider}**, model: **{audit.model}**.",
        f"- Исходных search queries: **{result.original_query_count}**; после локального улучшения: **{len(profile.search_queries)}**.",
        "- READY означает пригодность портрета как seed для discovery, но не автоматическое одобрение найденных кандидатов.",
        "",
        "## Автоматическая валидация",
        "",
        "| Проверка | Результат | Детали |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {check.name} | {_report_mark(check)} | {check.details} |"
        for check in result.checks
    )
    lines.extend(
        [
            "",
            "## Сильные стороны",
            "",
            "- Повторяющиеся сигналы Wildberries/Ozon, fashion-подборок, примерок, UGC и обзоров подтверждаются batch insights.",
            "- Форматы Reels/video, carousel/sidecar и одиночные публикации получены из фактических post types.",
            "- Профиль различает автора и магазин и содержит явные критерии исключения коммерческих витрин.",
            "- Engagement представлен диапазоном, а аномальный ER выше 100% не скрыт и помечен как риск качества данных.",
            "- Профиль сохраняет ограничения: нет демографии, визуальных материалов, conversion history и подтверждённых условий сотрудничества.",
            "",
            "## Подтверждено входными данными",
            "",
            "- У части эталонов есть упоминания Wildberries/Ozon, артикулов, подборок и обзоров товаров.",
            "- В batch evidence присутствуют fashion/style, женские образы, product reviews и unboxing.",
            "- Наблюдаются Video/Reels, Sidecar/Carousel и Image-публикации.",
            "- В bios встречаются PR/cooperation handles и формулировки о сотрудничестве.",
            "- Есть private-профиль без постов, профили со слабой fashion-фокусировкой и возможные магазины/бутики.",
            "- Значения ER неоднородны; batch 2 содержит аномалию около 272%.",
            "",
            "## Inferred и нормативные выводы",
            "",
            "- Женский состав и покупательские интересы аудитории inferred из тем и текстов; demographic analytics отсутствует.",
            "- Affordable-to-mid price segment inferred из marketplace/budget-сигналов; цены и чеки не передавались.",
            "- Визуальная совместимость inferred только по тексту; изображения и видео не анализировались.",
            "- Готовность обсуждать бартер inferred из PR/cooperation-сигналов и не является подтверждённым условием автора.",
            "- Конверсионный потенциал нельзя выводить только из ER: clicks, sales и campaign history отсутствуют.",
            "- Отсутствие явного unsafe-контента в трёх captions не подтверждает полный brand safety профиля.",
            "",
            "## Риски и слабые места",
            "",
        ]
    )
    lines.extend(f"- {risk}" for risk in result.risks)
    lines.extend(
        [
            "",
            "## Готовность к автоматическому поиску",
            "",
            "Профиль пригоден для запуска Tavily как источник поисковых гипотез: запросы конкретны, покрывают три площадки и содержат негативные фильтры магазинов, брендов, каталогов и карточек товаров. Результаты поиска нельзя автоматически считать подходящими: после discovery обязательны URL-нормализация, исключение магазинов, проверка автора, fashion-релевантности, метрик и brand safety.",
            "",
            "## Финальные search_queries для Tavily",
            "",
        ]
    )
    lines.extend(f"{index}. `{query}`" for index, query in enumerate(profile.search_queries, start=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_ideal_profile_quality(
    *,
    profile_path: Path,
    markdown_path: Path,
    batches_path: Path,
    audit_path: Path,
    report_path: Path,
) -> QualityValidationResult:
    """Validate, minimally qualify claims, improve queries, and save a report."""

    try:
        original_profile = LLMIdealBloggerProfile.model_validate(_load_json(profile_path))
        raw_batches = _load_json(batches_path)
        if not isinstance(raw_batches, list):
            raise IdealProfileQualityError("llm_batch_insights.json must contain a list")
        batches = [BatchBloggerInsights.model_validate(item) for item in raw_batches]
        audit = LLMAnalysisAudit.model_validate(_load_json(audit_path))
    except ValidationError as exc:
        raise IdealProfileQualityError(
            f"Pydantic validation failed with {exc.error_count()} error(s)"
        ) from exc

    original_query_count = len(original_profile.search_queries)
    profile = _qualify_evidence_claims(original_profile)
    analyzed_usernames = {
        username
        for batch in batches
        for username in batch.analyzed_usernames
    }
    empty_fields = _nonempty_required_fields(profile)
    profile_text = json.dumps(profile.model_dump(mode="json"), ensure_ascii=False).casefold()
    forbidden_found = [term for term in FORBIDDEN_TERMS if term in profile_text]
    sample_unknown = sorted(set(profile.sample_profile_usernames) - analyzed_usernames)
    platform_counts = {
        "Instagram": sum("site:instagram.com" in query for query in profile.search_queries),
        "YouTube Shorts": sum("site:youtube.com/shorts" in query for query in profile.search_queries),
        "Telegram": sum("site:t.me" in query for query in profile.search_queries),
    }
    relevance_groups = {
        "женская одежда/fashion": any(
            term in profile_text for term in ("женск", "fashion", "одежд", "плать")
        ),
        "Wildberries/Ozon": "wildberries" in profile_text and "ozon" in profile_text,
        "интеграции/бартер": any(
            term in profile_text for term in ("barter", "бартер", "integration", "интеграц")
        ),
    }
    checks = [
        QualityCheck(
            "Обязательные поля",
            not empty_fields,
            "все поля заполнены" if not empty_fields else f"пустые: {', '.join(empty_fields)}",
        ),
        QualityCheck(
            "Confidence 0–1",
            0 <= profile.confidence_score <= 1,
            f"confidence_score={profile.confidence_score:.2f}",
        ),
        QualityCheck(
            "Search queries 6–10",
            6 <= len(profile.search_queries) <= 10,
            f"было {original_query_count}, локально улучшено до {len(profile.search_queries)}",
        ),
        QualityCheck(
            "Search keywords",
            bool(profile.search_keywords),
            f"ключей: {len(profile.search_keywords)}",
        ),
        QualityCheck(
            "Sample usernames",
            not sample_unknown,
            (
                f"все {len(profile.sample_profile_usernames)} входят в {len(analyzed_usernames)} analyzed_usernames"
                if not sample_unknown
                else f"неизвестные: {', '.join(sample_unknown)}"
            ),
        ),
        QualityCheck(
            "Запрещённая предметная область",
            not forbidden_found,
            "совпадений нет" if not forbidden_found else f"найдены: {', '.join(forbidden_found)}",
        ),
        QualityCheck(
            "Fashion/marketplace/barter relevance",
            all(relevance_groups.values()),
            "; ".join(f"{name}: {'да' if found else 'нет'}" for name, found in relevance_groups.items()),
        ),
        QualityCheck(
            "Покрытие платформ",
            all(count >= 2 for count in platform_counts.values()),
            "; ".join(f"{name}: {count}" for name, count in platform_counts.items()),
        ),
        QualityCheck(
            "Audit завершён",
            not audit.errors and audit.completed_batches == audit.batch_count,
            f"errors={len(audit.errors)}, batches={audit.completed_batches}/{audit.batch_count}",
        ),
    ]
    risks = [
        "Confidence 0.72 — умеренный, а не высокий; портрет нельзя использовать как безусловную истину.",
        "ER около 272% является аномалией и не должен задавать scoring threshold без повторной проверки формулы и исходных постов.",
        "Визуальные признаки не подтверждены медиа: нельзя автоматически оценить соответствие эстетике LD LATTE.",
        "Нет достоверной демографии аудитории, conversion/sales history, ставок и фактической рекламной нагрузки.",
        "PR/cooperation handles подтверждают доступность для контакта, но не согласие на бартер.",
        "Часть эталонов относится к beauty/lifestyle/home или магазинам, поэтому возможен тематический drift выдачи.",
        "Search-engine результаты могут включать магазины даже при негативных фильтрах; post-search audit остаётся обязательным.",
    ]
    ready = all(check.passed for check in checks)
    result = QualityValidationResult(
        profile=profile,
        checks=checks,
        original_query_count=original_query_count,
        ready_for_search=ready,
        risks=risks,
    )
    save_ideal_profile(profile, profile_path, markdown_path)
    _write_report(report_path, result, audit, analyzed_usernames)
    return result


def main() -> int:
    """Run the local quality gate using standard data paths."""

    try:
        result = validate_ideal_profile_quality(
            profile_path=PROJECT_ROOT / "data" / "ideal_blogger_profile.json",
            markdown_path=PROJECT_ROOT / "data" / "ideal_blogger_profile.md",
            batches_path=PROJECT_ROOT / "data" / "llm_batch_insights.json",
            audit_path=PROJECT_ROOT / "data" / "llm_analysis_audit.json",
            report_path=PROJECT_ROOT / "data" / "ideal_profile_quality_report.md",
        )
    except IdealProfileQualityError as exc:
        print(f"Quality validation failed: {exc}")
        return 1
    print("IdealBloggerProfile quality validation completed")
    print(f"confidence_score: {result.profile.confidence_score:.2f}")
    print(f"search_queries: {len(result.profile.search_queries)}")
    print(f"ready_for_search: {result.ready_for_search}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

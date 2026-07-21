# LD LATTE Fashion Blogger Agent

Демонстрационный Python-пайплайн для поиска fashion-блогеров, объяснимого
ранжирования и подготовки предложений о бартерном сотрудничестве. LD LATTE —
бренд женской одежды, представленный на Wildberries и Ozon.

Проект поддерживает полностью офлайн-режим, чтение публичной Google-таблицы
через CSV export, обогащение профилей, пакетный LLM-анализ и сменные provider-ы
поиска. Публичный Google CSV export, Apify и OpenAI-анализ эталонного портрета
были реально проверены. Также реально запускались Tavily-поиск новых кандидатов,
author resolution, Apify enrichment найденных профилей, OpenAI-анализ и
генерация черновиков офферов. Google Sheets API не используется, таблица
читается без service account и только на чтение.

Главная точка входа в тестовое задание — [`START.md`](START.md).

## Текущий статус

| Компонент | Статус | Проверка |
|---|---|---|
| Google Sheets | Реально подключён public CSV export | `--inspect-source` и `data/source_inspection.json` |
| Instagram enrichment | Реально проверено через Apify | 33 профиля: 19 пригодных, 14 недоступных, 3 cache hits; 1 невалидная исходная строка |
| Scoring | Полностью работает в mock-режиме | `python -m src.main`, `MIN_SCORE`, `TOP_K`, audit |
| Поиск новых блогеров | Реальный Tavily → Author Resolution → Apify → scoring контур выполнен | Обработано 20 кандидатов: 0 recommended, 5 manual_review, 15 rejected |
| LLM-анализ | Реальный IdealBloggerProfile построен через Responses API; mock и dry-run сохранены | Audit: `data/llm_analysis_audit.json` |
| Офферы | OpenAI Structured Output реально запускался для итогового пула | Сформировано 3 черновика; сообщения автоматически не отправлялись |

## Реально проверенные интеграции

Публичная Google-таблица вернула 34 строки: 33 валидные Instagram-ссылки и
один невалидный источник. Через Apify проверены все 33 эталонных профиля
компании: пригодные данные получены по 19 (18 `success`, 1 `partial`), 14
профилей недоступны или не найдены, 3 ответа взяты из свежего кэша. Токены не
сохраняются в логах, raw response, audit и результатах.

Реальные Instagram-ссылки относятся к исходной эталонной базе LD LATTE. Они не
называются новыми кандидатами, найденными Tavily. Локальный snapshot, profile
cache, полные enrichment-результаты и сырой Apify response исключены из Git.

## Что делает пайплайн

1. Загружает пять эталонных fashion-блогеров из CSV.
2. Строит Pydantic-модель `IdealBloggerProfile`.
3. Формирует восемь поисковых запросов из тем, форматов, визуального стиля,
   аудитории, ценового сегмента и площадок портрета.
4. Получает кандидатов через сменный `SearchProvider`.
5. Нормализует ссылки, удаляет дубли и исключает нерелевантные результаты.
6. Обогащает профили только информацией из публичной поисковой выдачи.
7. Оценивает кандидатов по детерминированной шкале 0–100.
8. Оставляет не более `TOP_K` профилей с результатом не ниже `MIN_SCORE`.
9. Создаёт персонализированные mock-офферы и сохраняет CSV.

## Установка

Требуется Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Для Windows команда активации: `.venv\Scripts\activate`.

## Mock-режим без сети

Настройки по умолчанию:

```dotenv
MOCK_MODE=true
SOURCE_PROVIDER=csv
SEARCH_PROVIDER=mock
TOP_K=5
MIN_SCORE=70
SEARCH_MAX_RESULTS=30
```

Запуск:

```bash
python -m src.main
```

`MockSearchProvider` читает `data/candidates.example.csv`, не требует ключей и
не выполняет сетевых запросов. Если порог прошли четыре кандидата, итоговый CSV
будет содержать четыре строки, а не дополняться слабым пятым профилем.

## Источник эталонных блогеров

Поддерживаются два независимых режима:

- `SOURCE_PROVIDER=csv` читает нормализованный
  `data/source_bloggers.example.csv` и полностью работает без сети;
- `SOURCE_PROVIDER=google_sheets` только читает публичный лист через CSV export,
  сохраняет локальный снимок в `data/source_bloggers.real.csv` и позволяет
  исследовать неизвестную структуру.

Чтобы открыть исходную таблицу без service account, её владелец должен выбрать
в Google Sheets «Настройки доступа» → «Общий доступ» → «Все, у кого есть
ссылка» с ролью «Читатель». Режим не хранит cookies, не обходит авторизацию и не
запрашивает Google credentials.

Добавьте в локальный `.env`:

```dotenv
SOURCE_PROVIDER=google_sheets
GOOGLE_SHEET_URL=https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit
GOOGLE_SHEET_GID=0
```

Диагностика источника:

```bash
python -m src.main --inspect-source
```

Команда выводит колонки, число непустых строк, найденное сопоставление полей и
три сокращённые строки, затем сохраняет агрегированный отчёт в
`data/source_inspection.json`. Она завершается до поиска, scoring и генерации
офферов и не изменяет `data/results.csv`.

Обычная ссылка преобразуется в адрес вида
`/spreadsheets/d/<id>/export?format=csv&gid=<gid>`. Публичный CSV export — это
простое read-only скачивание одного листа без аутентификации. Google Sheets API
предоставляет более широкие операции чтения и записи, обычно требует отдельной
настройки доступа и на этом этапе намеренно не подключён. Проект ничего не
записывает обратно: сначала необходимо понять и нормализовать реальную схему,
а изменение исходной базы не входит в задачу.

## Обогащение исходных Instagram-профилей

Реальная Google-таблица содержит в основном ссылки, поэтому её недостаточно для
расчёта ER и последующего анализа контента. Команда enrichment извлекает
Instagram username, отбрасывает ссылки на посты, Reels, Stories и служебные
страницы, затем получает ограниченный набор публичных полей профиля и последних
публикаций.

Два provider-режима:

- `PROFILE_ENRICHMENT_PROVIDER=mock` использует вымышленные шаблоны из
  `data/profile_enrichment_mock.json` и не обращается в сеть;
- `PROFILE_ENRICHMENT_PROVIDER=apify` запускает указанный пользователем Actor
  через официальный Apify API. Рекомендуемый Actor поддерживается Apify и
  принимает `directUrls`, `resultsType` и `resultsLimit`.

Если `SOURCE_PROVIDER=csv`, enrichment сначала использует уже сохранённый
`data/source_bloggers.real.csv`. Поэтому полностью offline-проверка возможна
после однократной команды `--inspect-source`.

Тестовый запуск первых трёх профилей:

```bash
PROFILE_ENRICHMENT_PROVIDER=mock \
python -m src.main --enrich-source --limit-profiles 3
```

Для повторного запуска через Apify добавьте секреты только в локальный `.env`:

```dotenv
PROFILE_ENRICHMENT_PROVIDER=apify
APIFY_API_TOKEN=your-private-token
APIFY_ACTOR_ID=apify~instagram-scraper
PROFILE_POSTS_LIMIT=3
PROFILE_ENRICHMENT_CONCURRENCY=2
PROFILE_ENRICHMENT_DELAY_SECONDS=1
PROFILE_CACHE_ENABLED=true
```

Интеграция уже была проверена на реальной эталонной базе. Повторный запуск
может быть платным и для офлайн-демонстрации не нужен. Если он необходим,
сначала обязательно сохраните лимит `--limit-profiles 3`. Принудительное
обновление, игнорирующее свежий кэш:

```bash
python -m src.main --enrich-source --limit-profiles 3 --refresh-profiles
```

Кэш каждого профиля хранится в `data/profile_cache/<username>.json` и действует
24 часа. Ошибка одного профиля фиксируется в audit и не останавливает остальные.
Кэш другого provider автоматически игнорируется.
Команда enrichment не запускает Tavily, scoring, OpenAI или офферы и не меняет
`data/results.csv`.

Apify и Tavily решают разные задачи: Tavily ищет публичные страницы новых
кандидатов, а Apify структурирует данные заранее известных публичных профилей.
Ни один режим не использует Instagram cookies, логин, пароль или браузерную
автоматизацию. Email и телефоноподобные значения удаляются из сохраняемого
публичного текста.

## Полный реальный поиск новых блогеров

Контур использует готовый `data/ideal_blogger_profile.json` и никогда не читает
mock-кандидатов. Перед любой платной операцией выполните dry-run:

```bash
python -m src.main --find-real-bloggers --dry-run-final
```

Команда только читает локальный портрет и эталонные identities, показывает
лимиты и наличие обязательных переменных. Tavily, Apify и OpenAI не создаются,
а файлы результатов не изменяются.

```dotenv
TAVILY_MAX_QUERIES=8
TAVILY_RESULTS_PER_QUERY=5
MAX_CANDIDATES_BEFORE_ENRICHMENT=20
MAX_CANDIDATES_FOR_APIFY=8
MAX_FINAL_CANDIDATES=5
FINAL_MIN_SCORE=70
```

Максимальный план v2-запуска: 8 запросов и 40 результатов Tavily, до 20
очищенных кандидатов, до 8 Instagram-профилей в Apify и до 5 OpenAI-офферов.
Фактическое число может быть меньше после фильтрации и `FINAL_MIN_SCORE`.

Для отдельно подтверждённого запуска заполните локальный `.env`:

```dotenv
TAVILY_API_KEY=your-private-key
APIFY_API_TOKEN=your-private-token
APIFY_ACTOR_ID=apify~instagram-scraper
OPENAI_API_KEY=your-private-key
OPENAI_MODEL=gpt-5-mini
```

Точная команда реального запуска:

```bash
python -m src.main --find-real-bloggers
```

Последовательность: IdealBloggerProfile → Tavily → Author Resolution для
post/video URL → нормализация, dedup и исключение эталонов/магазинов → Apify
только для Instagram → confidence-aware scoring → категории результата → OpenAI
Structured Output drafts. YouTube и Telegram остаются на публичном evidence;
отсутствующие followers и ER сохраняются как `null`.

OpenAI вызывается только после scoring: для `recommended`, а если их меньше
трёх — для лучших `manual_review`, максимум для пяти профилей. Текст предлагает
обсудить возможный бартер, не обещает конкретный товар или коллекцию и не
отправляется автоматически. Итог всегда подтверждает человек.

Стоимость зависит от тарифов и фактического числа запросов Tavily, запусков
Apify Actor, OpenAI-модели, retries и размера evidence. Dry-run показывает
верхние количественные пределы, но не рассчитывает денежную стоимость.

### Финализация уже сохранённого v2-пула без Tavily

После первого v2-поиска повторять Tavily не требуется. Режим ниже читает
сохранённые raw/resolution audit, восстанавливает 20 canonical profiles и
использует уже готовый enrichment/cache. Лимит
`MAX_CANDIDATES_FOR_APIFY=20` применяется только внутри этого режима; максимум
три последние публикации запрашиваются только для ещё не обогащённых профилей.

Сначала выполните полностью локальный dry-run:

```bash
python -m src.main --finalize-saved-pool --dry-run-saved-pool
```

Он показывает число сохранённых cache hits, потенциальных новых Apify-вызовов
и максимум OpenAI-офферов. Tavily, Apify и OpenAI не запускаются, результаты не
изменяются. Подтверждённый запуск:

```bash
python -m src.main --finalize-saved-pool
```

Категории результата не меняют score и `FINAL_MIN_SCORE=70`:

- `recommended`: 70 и выше;
- `manual_review`: 60–69, резерв без автоматического одобрения;
- `rejected`: ниже 60.

Офферы создаются для `recommended`. Если таких меньше трёх, список дополняется
лучшими `manual_review`, но не более чем до пяти черновиков. Сообщения никогда
не отправляются автоматически. Даже `recommended` подтверждает человек.

Результаты сохраняются в `data/final_all_candidates.csv`,
`data/final_all_candidates.md` и `data/final_recommended_bloggers.csv`.

Фактический итог реального запуска сохранённого пула:

- обработано 20 кандидатов;
- `recommended`: 0;
- `manual_review`: 5;
- `rejected`: 15;
- через OpenAI сформировано 3 черновика офферов;
- сообщения автоматически не отправлялись.

## Анализ реальной эталонной базы через LLM

Источник — `data/enriched_source_bloggers.json`. В анализ попадают только записи
со статусом `success` или `partial`, максимум
`OPENAI_MAX_TOTAL_PROFILES`. Профили делятся на пакеты по
`OPENAI_MAX_PROFILES_PER_BATCH`; после каждого успешного пакета текущие
`BatchBloggerInsights` атомарно сохраняются. Отдельный финальный запрос
синтезирует Pydantic Structured Output `IdealBloggerProfile`.

Сначала всегда можно посмотреть план без ключа и без сети:

```bash
LLM_PROVIDER=openai python -m src.main --build-ideal-profile --dry-run-llm
```

Dry-run показывает provider, модель, число пригодных профилей, пакетов и
примерный объём символов. Он не создаёт OpenAI-клиент и делает ноль сетевых
запросов. Полностью офлайн-анализ с прозрачными keyword/rule-based правилами:

```bash
LLM_PROVIDER=mock python -m src.main --build-ideal-profile
```

Для отдельно подтверждённого реального запуска укажите ключ только в локальном
`.env` и смените provider:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your-private-key
OPENAI_MODEL=gpt-5-mini
OPENAI_MAX_PROFILES_PER_BATCH=8
OPENAI_MAX_POSTS_PER_PROFILE=3
OPENAI_MAX_TOTAL_PROFILES=25
OPENAI_REQUEST_TIMEOUT_SECONDS=120
```

```bash
python -m src.main --build-ideal-profile
```

OpenAI provider использует Responses API `responses.parse`, Pydantic Structured
Outputs, timeout и две явные повторные попытки с экспоненциальной задержкой.
Встроенные retries SDK отключены, чтобы число повторов было видно в audit.

В OpenAI передаются только: `username`, `full_name`, `biography`, число
подписчиков, рассчитанный ER, признак private и не более трёх последних записей
с сокращёнными `caption`, hashtags, `post_type` и `accessibility_caption`.
Email и телефоноподобные значения дополнительно редактируются.

Не передаются raw JSON, изображения, видео, URL медиа, email, телефоны, внешние
ссылки, токены и остальные поля enrichment. Длинный caption ограничен 1500
символами. Промпты запрещают придумывать визуальные признаки и требуют
помечать косвенные выводы как `inferred`.

Стоимость зависит от выбранной модели, числа профилей и объёма captions. Dry-run
даёт только оценку символов, а не точную цену или токены. Любой итоговый портрет
нужно вручную проверить: ER может содержать выбросы, текст не подтверждает
визуальный стиль, а небольшой набор публикаций не доказывает рекламную нагрузку
или brand safety. Реальный анализ эталонного портрета через OpenAI, Tavily-поиск,
author resolution, Apify enrichment новых кандидатов и генерация трёх черновиков
офферов выполнены. Все результаты требуют ручной проверки; сообщения не
отправлялись.

## Безопасность поиска

Проект не открывает закрытые профили, не обходит авторизацию Instagram и не
пытается эмулировать действия пользователя. Он анализирует только URL, title и
snippet, которые вернул легальный поисковый API.

Отбрасываются:

- неподдерживаемые домены и страницы без профиля или канала;
- повторяющиеся ссылки и профили из эталонной базы;
- магазины, бренды, каталоги и официальные страницы маркетплейсов;
- результаты без признаков женской fashion-тематики;
- профили с `data_confidence < 0.45`.

Если поисковая выдача не содержит значения, оно остаётся `null`. Неизвестная
метрика не подменяется вымышленным числом и получает ноль по соответствующему
критерию scoring.

## Входные и служебные файлы

- `data/source_bloggers.example.csv` — эталонная база;
- `data/source_bloggers.real.csv` — локальный read-only снимок публичного листа;
- `data/source_inspection.json` — агрегированная диагностика структуры;
- `data/profile_enrichment_mock.json` — вымышленные offline-шаблоны;
- `data/profile_cache/<username>.json` — кэш одного профиля на 24 часа;
- `data/enriched_source_bloggers.json` — полные Pydantic-структуры enrichment;
- `data/enriched_source_summary.csv` — компактная таблица профилей и ER;
- `data/profile_enrichment_audit.csv` — результат обработки каждой ссылки;
- `data/llm_batch_insights.json` — результаты успешных пакетов LLM-анализа;
- `data/ideal_blogger_profile.json` — итоговый структурированный портрет;
- `data/ideal_blogger_profile.md` — читаемая версия портрета;
- `data/llm_analysis_audit.json` — provider, модель, длительность, retries,
  usage и безопасные ошибки;
- `data/real_candidates_raw.csv` — исходные Tavily title/url/content/query/score;
- `data/real_candidates_audit.csv` — очистка, enrichment, confidence penalty,
  score и причина каждого решения;
- `data/real_candidates_enriched.json` — безопасное объединение Tavily и Apify;
- `data/final_real_bloggers.csv` — финалисты со статусом `needs_review`;
- `data/final_real_bloggers.md` — shortlist и неотправленные черновики;
- `data/final_run_audit.json` — лимиты, counts, errors и OpenAI usage;
- `data/final_score_breakdown.csv` — баллы и текстовая причина по каждому
  критерию для всех кандидатов, дошедших до scoring;
- `data/final_all_candidates.csv` — все 20 кандидатов сохранённого v2-пула с
  категориями и полным scoring breakdown;
- `data/final_all_candidates.md` — human-readable группы recommended,
  manual_review и rejected;
- `data/final_recommended_bloggers.csv` — только кандидаты с score 70+;
- `data/candidates.example.csv` — кандидаты offline-провайдера;
- `data/search_queries.json` — запросы текущего запуска;
- `data/search_audit.csv` — все найденные ссылки и причины решений;
- `data/results.csv` — финальные кандидаты и офферы.

Списочные поля CSV разделяются символом `|`. Все mock-профили вымышлены и
используют `example.com`.

Все артефакты реального поиска и отдельный candidate cache находятся в
`.gitignore`. Финальный shortlist можно осознанно добавить в репозиторий только
после ручной проверки каждого профиля и текста.

Основные переменные `.env`:

```dotenv
SOURCE_PROVIDER=csv
GOOGLE_SHEET_URL=
GOOGLE_SHEET_GID=0
PROFILE_ENRICHMENT_PROVIDER=mock
APIFY_API_TOKEN=
APIFY_ACTOR_ID=apify~instagram-scraper
PROFILE_POSTS_LIMIT=3
PROFILE_ENRICHMENT_CONCURRENCY=2
PROFILE_ENRICHMENT_DELAY_SECONDS=1
PROFILE_CACHE_ENABLED=true
SOURCE_CSV_PATH=data/source_bloggers.example.csv
SOURCE_REAL_CSV_PATH=data/source_bloggers.real.csv
SOURCE_INSPECTION_PATH=data/source_inspection.json
CANDIDATES_CSV_PATH=data/candidates.example.csv
RESULTS_CSV_PATH=data/results.csv
SEARCH_QUERIES_PATH=data/search_queries.json
SEARCH_AUDIT_PATH=data/search_audit.csv
SEARCH_PROVIDER=mock
TAVILY_API_KEY=
SEARCH_MAX_RESULTS=30
TAVILY_MAX_QUERIES=9
TAVILY_RESULTS_PER_QUERY=5
MAX_CANDIDATES_BEFORE_ENRICHMENT=20
MAX_CANDIDATES_FOR_APIFY=8
MAX_FINAL_CANDIDATES=5
FINAL_MIN_SCORE=70
LLM_PROVIDER=mock
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-mini
OPENAI_MAX_PROFILES_PER_BATCH=8
OPENAI_MAX_POSTS_PER_PROFILE=3
OPENAI_MAX_TOTAL_PROFILES=25
OPENAI_REQUEST_TIMEOUT_SECONDS=120
TOP_K=5
MIN_SCORE=70
LOG_LEVEL=INFO
```

`TOP_K` допускает значения 3–5, `MIN_SCORE` — 0–100.

Финальный scoring использует отдельную шкалу 0–100: fashion relevance,
текстовые/визуальные сигналы, аудиторию, тон, ER, рекламную нагрузку, price
segment, формат, brand safety и data confidence. Confidence даёт до 5 баллов и
пропорционально снижает остальные критерии; причина записывается в audit.

## Почему кандидат получил именно такой score

После scoring проект сохраняет `data/final_score_breakdown.csv`. В файл входят
все оценённые кандидаты, включая тех, кто не прошёл `FINAL_MIN_SCORE`. Для
тематики, визуальных/текстовых сигналов, аудитории, тона, ER, рекламной
нагрузки, ценового сегмента, форматов и brand safety сохраняются фактические
баллы и отдельное объяснение на основании доступных публичных данных.

Если сигнал нельзя подтвердить, причина явно содержит «Недостаточно данных для
уверенной оценки» и `confidence: low`: нулевой или ограниченный балл больше не
остаётся необъяснённым. `data_confidence`, `evidence_count` и
`confidence_reason` показывают полноту основания оценки.

Формулы не меняются. В real-контуре девять предметных критериев дают максимум
95 баллов: visual/text имеет максимум 15. Ещё до 5 баллов даёт полнота данных;
этот вклад указан в `confidence_reason`, поэтому сумма точно сходится с
`total_score`. В mock-контуре visual имеет максимум 20 и отдельного бонуса за
полноту данных нет. Markdown-shortlist дополнительно показывает после каждого
финалиста таблицу `Критерий / Баллы / Причина`.

## Портрет и scoring

`IdealBloggerProfile` содержит темы и форматы контента, визуальный стиль, тон,
целевую аудиторию, ценовой сегмент, engagement rate, рекламную нагрузку, brand
safety и предпочтительные форматы интеграции.

| Критерий | Максимум |
|---|---:|
| Тематика | 20 |
| Визуальная совместимость | 20 |
| Целевая аудитория | 15 |
| Тон и подача | 10 |
| Вовлечённость | 10 |
| Рекламная нагрузка | 10 |
| Ценовой сегмент | 5 |
| Форматы контента | 5 |
| Brand safety | 5 |

## Результаты и аудит

`results.csv` дополнительно содержит `source_query`, `source_title`,
`source_snippet` и `data_confidence`.

`search_audit.csv` фиксирует одно итоговое решение для каждого результата:

- `accepted`;
- `duplicate`;
- `unsupported_domain`;
- `brand_or_store`;
- `insufficient_data`;
- `low_confidence`;
- `below_min_score`.

## Тесты

```bash
python -m pytest -q
```

Тесты не обращаются в интернет. Они проверяют scoring, `MIN_SCORE`, реальное
число финалистов, нормализацию и дубли ссылок, фильтрацию доменов и магазинов,
Google Sheets URL/gid/export, сопоставление колонок, HTTP-ошибки через mock и
offline-работу CSV/mock-провайдера. Отдельные тесты проверяют Instagram username,
исключение post/Reel URL, ER, null при недостатке метрик, 24-часовой кэш,
изоляцию ошибок профилей, Apify-конфигурацию и `--limit-profiles`. LLM-тесты
отдельно проверяют фильтрацию статусов, batching, сокращение captions, partial
без постов, Pydantic Structured Outputs, отсутствие сети в dry-run, понятную
ошибку ключа, сохранение готовых пакетов после сбоя и mock provider.

## Модули

- `src/search_queries.py` — генерация запросов из идеального портрета;
- `src/search_providers.py` — абстракция, mock и Tavily;
- `src/candidate_enricher.py` — нормализация, enrichment, фильтры и аудит;
- `src/candidate_ranker.py` — scoring, `MIN_SCORE` и `TOP_K`;
- `src/sheets_loader.py` — строгие CSV, публичный Google CSV export и инспекция;
- `src/profile_enrichment_providers.py` — mock/Apify, ER, cache и audit;
- `src/llm_profile_analyzer.py` — allow-list данных, batching, mock/OpenAI,
  Structured Outputs, retry, промежуточные файлы и audit;
- `src/final_pipeline.py` — cost gate и полный реальный orchestration;
- `src/final_candidate_ranker.py` — confidence-aware scoring реальных данных;
- `src/final_offer_generator.py` — Responses API drafts только для финалистов;
- `src/main.py` — полный orchestration;
- остальные модули отвечают за конфигурацию, модели, CSV и офферы.

## Ограничения и ручная проверка

Title и snippet поисковой выдачи могут быть неполными или устаревшими. По ним
нельзя надёжно подтвердить текущую аудиторию, ER, долю рекламы или brand safety.
Поэтому `data_confidence` — оценка полноты данных, а не гарантия качества
профиля.

Перед отправкой любого оффера сотрудник LD LATTE должен вручную открыть
публичный профиль, проверить актуальность контента и метрик, убедиться в brand
safety и отдельно согласовать образ и условия сотрудничества.

Instagram может ограничивать публично доступные поля, а private-профили не дают
достаточно публикаций для анализа. Данные Actor могут быть неполными или
устаревшими; `data_confidence` отражает полноту, но не подтверждает достоверность.
Перед реальным запуском нужно проверить схему и стоимость выбранного Actor, его
условия использования и соответствие правилам платформы.

# RAG по учебнику ТФКП

**Чилилова Аминат Ахмедовна**, группа М05-603в  
Курс LLM, МФТИ, 26f. Домашнее задание №1 — RAG.

Модуль представляет собой RAG-систему, отвечающую на вопросы с вариантами A–D по учебнику В. В. Горяйнова и Е. С. Половинкина «Лекции по теории функций комплексного переменного» (2025). Система строит гибридный индекс (BM25 + dense) по корпусу из PDF, находит релевантные фрагменты, генерирует ответ моделью **Qwen3.5-4B** и возвращает строгий JSON: `answer_ids`, `abstain`, `rationale`, `citations`. Когда данных в корпусе недостаточно, система корректно отказывается отвечать.


## Структура репозитория

```text
.
├── parse_corpus.py              # PDF → corpus.jsonl
├── build_index.py               # corpus.jsonl → index/
├── predict.py                   # локальный запуск Qwen3.5-4B (Hugging Face, 4-bit)
├── predict_on_host.py           # HTTP-режим через OpenAI-совместимый endpoint
├── evaluate.py                  # оценка против gold-файлов
├── run_experiments.py           # 5 конфигураций + experiments.csv
├── config.json                  # параметры моделей, retrieval, лимиты
├── requirements.txt
├── report.md
├── README.md
│
├── examples_of_module_output/   # примеры выхода модуля
│   ├── dev_answers.jsonl        # итоговый прогон итоговой системы
│   └── dev_trace.jsonl          # трейс вызовов LLM (calls, tokens, время)
│
├── experiments_result/          # результаты прогонов экспериментов
│   └── experiments.csv          # 5 конфигураций: baseline, RAG, final, 2 абляции
│
└── questions_and_golds_set/     # вопросы и эталоны
    ├── dev_questions.jsonl      # 15 dev-вопросов (формат задания)
    ├── dev_gold.jsonl           # эталоны для dev
    ├── own_questions.jsonl      # 8 авторских вопросов
    └── own_gold.jsonl           # эталоны для авторских вопросов
```
## Быстрый запуск
Клонируем репозиторий и переходим в папку проекта:
```bash
git clone https://github.com/ChAminat/LLM-mipt-26f.git

cd LLM-mipt-26f
```
Ставим зависимости:

```bash
python -m pip install -r requirements.txt
```

Скачиваем PDF учебника одельно (в репозитории его нет), после чего собираем корпус:

```bash
wget -O book.pdf "https://new.mipt.ru/upload/mathematics/study/uchebniki/%D0%9B_%D0%A2%D0%A4%D0%9A%D0%9F_%D0%93%D0%BE%D1%80%D1%8F%D0%B9%D0%BD%D0%BE%D0%B2_%D0%9F%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD_2025.pdf"

python parse_corpus.py --pdf book.pdf --output corpus.jsonl
```
Получается ~370 блоков, `corpus.jsonl` ~1 MB. По нему собирается индекс (~3 минуты на CPU, `index/` ~2 MB):

```bash
python build_index.py --corpus /path/to/corpus.jsonl --output ./index --config config.json
```

Дальше - предсказание. В локальном режиме модель Qwen3.5-4B грузится с Hugging Face (4-bit, ~8 GB, 2–4 минуты при первом запуске); на 15 вопросов уходит ~7 минут на T4:

```bash
python predict.py \
    --index ./index \
    --questions /path/to/questions.jsonl \
    --output ./answers.jsonl \
    --trace ./trace.jsonl \
    --config config.json
```

В HTTP-режиме модель пашет на сервере, локально считаются только эмбеддинги для retrieval. Нужно задать окружение и вызвать второй скрипт:

```bash
export LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LLM_MODEL="Qwen/Qwen3.5-4B"
export LLM_API_KEY="dummy"
export SUBMISSION_ID="mipt-tfkp-rag"

python predict_on_host.py \
    --index ./index \
    --questions /path/to/questions.jsonl \
    --output ./answers.jsonl \
    --trace ./trace.jsonl \
    --config config.json
```

Качество проверяется против эталонов:

```bash
python evaluate.py --answers ./answers.jsonl --gold dev_gold.jsonl
```

Печатается таблица «эталон / ответ / совпало», accuracy и `Q` по 4 категориям. Наконец, эксперименты — 5 конфигураций через `predict.py`, ~40 минут, результат в `experiments.csv`:

```bash
python run_experiments.py \
    --index ./index \
    --questions dev_questions.jsonl \
    --gold dev_gold.jsonl \
    --config config.json \
    --workdir experiments
```

Промежуточные JSONL складываются в `experiments/`.

# Особенности реализации

#### Два режима предсказания

Логика retrieval, промптов, валидации и отказов общая. Различается только способ вызова LLM:

- **`predict.py`** — загружает Qwen3.5-4B локально с Hugging Face (4-bit). Не требует сервера, работает «из коробки». Используется для воспроизводимости и dev-экспериментов.
- **`predict_on_host.py`** — тот же пайплайн, но вызовы идут по HTTP на OpenAI-совместимый endpoint. Параметры берутся из окружения (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `SUBMISSION_ID`). Используется на проверке.

У обоих файлов одинаковый CLI (`--index`, `--questions`, `--output`, `--trace`, `--config`), одинаковый формат выхода и одинаковые флаги экспериментов и лимиты из `config.json`, поэтому их можно менять местами без изменения остального пайплайна.

#### Архитектура

**Парсер (`parse_corpus.py`).** PyMuPDF, постраничное извлечение, восстановление cp1251-кириллицы. Склейка строк-формул в `$...$` с текстовым дублем для BM25. Нарезка по маркерам: `Теорема X.Y`, `Определение`, `Следствие`, `Замечание`, `Пример`, `Доказательство`. Фильтр оглавления. Выход: `corpus.jsonl` с полями `block_id, page, section, text, text_search`.

**Индекс (`build_index.py`).** BM25 (`rank_bm25`, k1=1.5, b=0.75) по `text_search`. Dense: `intfloat/multilingual-e5-small`, L2-нормированные эмбеддинги, префиксы `query:` / `passage:`. Сохранение: `bm25.pkl`, `embeddings.npy` (370×384), `meta.json`, `texts.jsonl`.

**Retrieval.** BM25 + dense → RRF (k=60) → top-6 + соседние блоки (±1 по `block_ids`). Порог отказа: max BM25 < 1.0 → `abstain` без вызова LLM.

**Генерация.** Qwen3.5-4B, `enable_thinking=False` (локально через `transformers` + `bitsandbytes`, либо по HTTP). Строгий JSON: `{answer_ids, abstain, rationale, citations}`. Валидатор проверяет schema, `citations ⊆ retrieved` (≤4), консистентность `abstain`, `rationale` ≤1200. При невалидном JSON — retry 1 раз.

**Отказы.** Retrieval: max BM25 < 1.0 → `abstain` без вызова LLM. LLM вернул `abstain=true` → `abstain` с обоснованием.

#### Эксперименты

См. `experiments.csv` и разделы 4–5 в `report.md`. Кратко:

- Baseline без контекста: accuracy 0.600.
- RAG с одним поиском (BM25-only): **accuracy 0.800**.
- Итоговая система (hybrid + соседи): accuracy 0.733.
- Абляция 1 (без dense): accuracy 0.800.
- Абляция 2 (без соседей): accuracy 0.800.

**Вывод:** RAG даёт +13–20 п.п. к accuracy. На этом корпусе BM25-only работает лучше hybrid+dense из-за точной математической терминологии.

#!/usr/bin/env python3
"""
RAG по учебнику ТФКП: BM25 + dense → RRF → Qwen3.5-4B (локально) → structured JSON.

Модель загружается локально с Hugging Face. Никаких HTTP, env, API-ключей.

Установка:
    pip install pymupdf rank_bm25 sentence-transformers transformers accelerate bitsandbytes numpy

Запуск:
    python predict.py --index index --questions dev_questions.jsonl \
        --output dev_answers.jsonl --trace dev_trace.jsonl --config config.json
"""
import argparse
import json
import pickle
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------- Локальная модель ----------

MODEL_NAME = "Qwen/Qwen3.5-4B"


def load_local_model():
    """Загружает Qwen3.5-4B в 4-bit (влезает в T4)."""
    print(f"Загрузка токенизатора {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Загрузка модели (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Модель загружена на {model.device}")
    return model, tokenizer


def local_generate(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 1024,
) -> dict:
    """Генерирует ответ локально. Возвращает {content, input_tokens, output_tokens}."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # отключаем <think> для чистого JSON
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs.input_ids.shape[1]:]
    content = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "content": content,
        "input_tokens": int(inputs.input_ids.shape[1]),
        "output_tokens": int(generated.shape[0]),
    }


# ---------- Загрузка индекса ----------

def load_index(index_dir: Path, config: dict):
    with (index_dir / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)["bm25"]
    embeddings = np.load(index_dir / "embeddings.npy")
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    texts = []
    with (index_dir / "texts.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line))
    model = SentenceTransformer(
        meta["embedding"]["model"],
        revision=meta["embedding"]["revision"],
        device="cpu",
    )
    return {
        "bm25": bm25,
        "embeddings": embeddings,
        "block_ids": meta["block_ids"],
        "id2idx": {bid: i for i, bid in enumerate(meta["block_ids"])},
        "texts": texts,
        "embed_model": model,
        "query_prefix": meta["embedding"]["prefix_query"],
    }


# ---------- Токенизация ----------

TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)
STOP_RU = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или
ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они
тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе
под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти
мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой
хоть после над больше тот через эти нас про всего них какая много разве три эту моя
впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда
конечно всю между
""".split())
STOP_EN = set("""
the a an and or of to in on is are was were be been being what which who when where
how does did for with
""".split())
STOP = STOP_RU | STOP_EN


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower().replace("ё", "е"))
            if t not in STOP]


# ---------- Retrieval ----------

def rrf(rankings: list[list[str]], k: int = 60, top_n: int = 12) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, bid in enumerate(dict.fromkeys(ranking), 1):
            scores[bid] = scores.get(bid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda b: (-scores[b], b))[:top_n]


def retrieve(question: str, idx: dict, config: dict) -> tuple[list[str], float]:
    exp = config.get("experiments", {})
    use_bm25 = exp.get("use_bm25", True)
    use_dense = exp.get("use_dense", True)
    use_neighbors = exp.get("use_neighbors", True)

    top_k = config["retrieval"]["top_k"]
    candidates = config["retrieval"]["candidates"]
    rrf_k = config["retrieval"]["rrf_k"]

    rankings = []

    if use_bm25:
        q_tokens = tokenize(question)
        bm25_scores = idx["bm25"].get_scores(q_tokens)
        bm25_order = np.argsort(-bm25_scores).tolist()[:candidates]
        rankings.append([idx["block_ids"][i] for i in bm25_order])
        max_bm25 = float(bm25_scores.max()) if len(bm25_scores) else 0.0
    else:
        bm25_scores = np.zeros(len(idx["block_ids"]))
        max_bm25 = 0.0

    if use_dense:
        q_vec = idx["embed_model"].encode(
            [idx["query_prefix"] + question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        sims = (idx["embeddings"] @ q_vec.T).ravel()
        dense_order = np.argsort(-sims).tolist()[:candidates]
        rankings.append([idx["block_ids"][i] for i in dense_order])

    if len(rankings) == 1:
        fused = rankings[0][:candidates]
    elif len(rankings) >= 2:
        fused = rrf(rankings, k=rrf_k, top_n=candidates)
    else:
        fused = []

    if use_neighbors:
        block_ids = idx["block_ids"]
        pos = {bid: i for i, bid in enumerate(block_ids)}
        expanded = []
        for bid in fused[:top_k]:
            i = pos[bid]
            for j in (max(0, i - 1), i, min(len(block_ids) - 1, i + 1)):
                nb = block_ids[j]
                if nb not in expanded:
                    expanded.append(nb)
        top = expanded[:top_k]
    else:
        top = fused[:top_k]

    return top, max_bm25

def pack_context(block_ids: list[str], idx: dict, max_chars: int = 6000) -> str:
    parts = []
    used = 0
    for bid in block_ids:
        i = idx["id2idx"][bid]
        rec = idx["texts"][i]
        chunk = f"[{bid}] page={rec['page']} section={rec['section']}\n{rec['text']}"
        if used + len(chunk) > max_chars:
            continue
        parts.append(chunk)
        used += len(chunk) + 2
    return "\n\n".join(parts)


# ---------- Prompt ----------

SYSTEM_PROMPT = """Ты — ассистент по теории функций комплексного переменного (ТФКП).
Отвечай ТОЛЬКО по предоставленному контексту из учебника Горяйнова–Половинкина.
Не используй знания из памяти, если их нет в контексте.

Вопрос содержит варианты A, B, C, D. Правильных может быть несколько.
Выбери ВСЕ правильные варианты. Если данных в контексте недостаточно —
верни abstain=true и пустые списки.

Формат ответа — строгий JSON без markdown:
{
  "answer_ids": ["A", "C"],
  "abstain": false,
  "rationale": "краткое обоснование (≤1200 символов)",
  "citations": ["b_0011", "b_0012"]
}

Правила:
- answer_ids: уникальные ID вариантов (A, B, C, D, E, F), порядок неважен.
- citations: ≤4 уникальных block_id ИЗ ПРЕДОСТАВЛЕННОГО КОНТЕКСТА.
- rationale: кратко, с опорой на контекст.
- Для вычислительных вопросов (интеграл, коэффициент ряда) обязательно
  цитируй блок с формулой или теоремой, из которой следует ответ.
- Внимательно проверь знак перед i f_y: для ∂f/∂z̄ это +i f_y,
  для ∂f/∂z это −i f_y. Не путай варианты.
- Если утверждение следует из теоремы о производных всех порядков,
  выбери его тоже.
- При abstain=true: answer_ids=[], citations=[], в rationale — чего не хватает.
- Не цитируй block_id, которых нет в контексте.
"""


def build_user_prompt(question: dict, context: str) -> str:
    options = "\n".join(f"{o['id']}. {o['text']}" for o in question["options"])
    return f"""Вопрос: {question['question']}

Варианты:
{options}

Контекст из учебника:
{context}
"""


# ---------- Парсинг и валидация ----------

def parse_json_response(content: str) -> Optional[dict]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def validate_answer(answer: dict, retrieved_ids: set[str], question: dict) -> tuple[bool, str]:
    if not isinstance(answer, dict):
        return False, "ответ не dict"
    if "answer_ids" not in answer or "abstain" not in answer:
        return False, "нет answer_ids или abstain"
    if not isinstance(answer["abstain"], bool):
        return False, "abstain не bool"
    if not isinstance(answer.get("answer_ids"), list):
        return False, "answer_ids не list"
    if not isinstance(answer.get("citations"), list):
        return False, "citations не list"
    if not isinstance(answer.get("rationale"), str):
        return False, "rationale не str"

    if answer["abstain"]:
        if answer["answer_ids"] or answer["citations"]:
            return False, "при abstain=true списки должны быть пусты"
        return True, "ok"

    if not answer["answer_ids"]:
        return False, "abstain=false, но answer_ids пуст"

    valid_option_ids = {o["id"] for o in question["options"]}
    if not set(answer["answer_ids"]) <= valid_option_ids:
        return False, "answer_ids содержит недопустимые ID"
    if len(set(answer["answer_ids"])) != len(answer["answer_ids"]):
        return False, "answer_ids с дублями"
    if not answer["citations"]:
        return False, "abstain=false, но citations пуст"
    if len(answer["citations"]) > 4:
        return False, "citations > 4"
    if not set(answer["citations"]) <= retrieved_ids:
        return False, "citations содержат block_id вне retrieved"
    if len(answer["rationale"]) > 1200:
        return False, "rationale > 1200"
    return True, "ok"


# ---------- Обработка вопроса ----------

def process_question(
    question: dict,
    idx: dict,
    model,
    tokenizer,
    config: dict,
) -> tuple[dict, dict]:
    """Обрабатывает один вопрос. Возвращает (answer_record, trace_record)."""
    qid = question["id"]
    t0 = time.perf_counter()
    exp = config.get("experiments", {})
    use_context = exp.get("use_context", True)

    total_in = 0
    total_out = 0
    n_calls = 0
    trace_events = []

    # ---------- Baseline: без контекста ----------
    if not use_context:
        options = "\n".join(f"{o['id']}. {o['text']}" for o in question["options"])
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — ассистент по теории функций комплексного переменного. "
                    "Отвечай строгим JSON без markdown:\n"
                    '{"answer_ids": ["A"], "abstain": false, '
                    '"rationale": "кратко", "citations": []}\n'
                    "Если не уверен — abstain=true, списки пусты."
                ),
            },
            {
                "role": "user",
                "content": f"Вопрос: {question['question']}\n\nВарианты:\n{options}",
            },
        ]

        parsed = None
        last_error = None
        for attempt in range(2):
            if n_calls >= 4:
                break
            n_calls += 1
            resp = local_generate(model, tokenizer, messages, max_new_tokens=512)
            total_in += resp["input_tokens"]
            total_out += resp["output_tokens"]
            trace_events.append({
                "event": "llm_call_no_context",
                "attempt": attempt + 1,
                "input_tokens": resp["input_tokens"],
                "output_tokens": resp["output_tokens"],
            })
            candidate = parse_json_response(resp["content"])
            if candidate is None:
                last_error = "JSON не распознан"
                messages.append({"role": "assistant", "content": resp["content"]})
                messages.append({
                    "role": "user",
                    "content": "Ответ не является валидным JSON. Верни только JSON по схеме.",
                })
                continue
            # Минимальная валидация для baseline: answer_ids ⊆ options, abstain bool
            if not isinstance(candidate.get("abstain"), bool):
                last_error = "abstain не bool"
                continue
            if candidate["abstain"]:
                parsed = candidate
                break
            valid_ids = {o["id"] for o in question["options"]}
            ids = candidate.get("answer_ids", [])
            if not isinstance(ids, list) or not set(ids) <= valid_ids:
                last_error = "answer_ids невалидны"
                continue
            parsed = candidate
            break

        if parsed is None:
            answer = {
                "id": qid,
                "answer_ids": [],
                "abstain": True,
                "rationale": f"Baseline: не удалось получить валидный ответ ({last_error})",
                "citations": [],
            }
            trace_events.append({"event": "abstain", "reason": "baseline_invalid"})
        else:
            answer = {
                "id": qid,
                "answer_ids": parsed.get("answer_ids", []) if not parsed.get("abstain") else [],
                "abstain": parsed.get("abstain", True),
                "rationale": parsed.get("rationale", "")[:1200],
                "citations": [],
            }

        trace = {
            "id": qid,
            "llm_calls": n_calls,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "retrieved_block_ids": [],
            "events": trace_events,
        }
        return answer, trace

    # ---------- RAG-режим (с контекстом) ----------
    retrieved, max_bm25 = retrieve(question["question"], idx, config)
    context = pack_context(retrieved, idx)
    trace_events.append({
        "event": "retrieval",
        "n_blocks": len(retrieved),
        "max_bm25": round(max_bm25, 3),
        "block_ids": retrieved,
    })

    # Отказ №1: BM25 слишком низкий
    if max_bm25 < 1.0:
        answer = {
            "id": qid,
            "answer_ids": [],
            "abstain": True,
            "rationale": "В корпусе не найдено релевантных фрагментов (низкий BM25).",
            "citations": [],
        }
        trace_events.append({"event": "abstain", "reason": "low_bm25"})
        return answer, {
            "id": qid,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "retrieved_block_ids": retrieved,
            "events": trace_events,
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, context)},
    ]

    last_error = None
    final_answer = None

    for attempt in range(2):
        if n_calls >= 4:
            break
        n_calls += 1
        resp = local_generate(model, tokenizer, messages, max_new_tokens=1024)
        total_in += resp["input_tokens"]
        total_out += resp["output_tokens"]
        trace_events.append({
            "event": "llm_call",
            "attempt": attempt + 1,
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
        })
        parsed = parse_json_response(resp["content"])
        if parsed is None:
            last_error = "JSON не распознан"
            messages.append({"role": "assistant", "content": resp["content"]})
            messages.append({
                "role": "user",
                "content": "Ответ не является валидным JSON. Верни только JSON по схеме.",
            })
            continue
        ok, reason = validate_answer(parsed, set(retrieved), question)
        if ok:
            final_answer = parsed
            break
        last_error = reason
        messages.append({"role": "assistant", "content": resp["content"]})
        messages.append({
            "role": "user",
            "content": f"Ответ не прошёл валидацию: {reason}. Исправь и верни только JSON.",
        })

    if final_answer is None:
        answer = {
            "id": qid,
            "answer_ids": [],
            "abstain": True,
            "rationale": f"Не удалось получить валидный ответ: {last_error}",
            "citations": [],
        }
        trace_events.append({"event": "abstain", "reason": "invalid_llm_response"})
    else:
        answer = {"id": qid, **final_answer}
        if answer["abstain"]:
            trace_events.append({"event": "abstain", "reason": "llm_abstained"})

    trace = {
        "id": qid,
        "llm_calls": n_calls,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "retrieved_block_ids": retrieved,
        "events": trace_events,
    }
    return answer, trace

# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))

    print("Загрузка индекса...")
    idx = load_index(args.index, config)
    print(f"  {len(idx['block_ids'])} блоков")

    print("Загрузка локальной модели...")
    model, tokenizer = load_local_model()

    questions = []
    with args.questions.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"Вопросов: {len(questions)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out_f, \
         args.trace.open("w", encoding="utf-8") as tr_f:
        for i, q in enumerate(questions, 1):
            answer, trace = process_question(q, idx, model, tokenizer, config)
            out_f.write(json.dumps(answer, ensure_ascii=False) + "\n")
            tr_f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            status = "ABSTAIN" if answer["abstain"] else ",".join(answer["answer_ids"])
            print(f"  [{i}/{len(questions)}] {q['id']}: {status} "
                  f"({trace['llm_calls']} calls, {trace['elapsed_ms']}ms)")

    print(f"Готово: {args.output}, {args.trace}")


if __name__ == "__main__":
    main()
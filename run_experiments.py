#!/usr/bin/env python3
"""
Прогоняет 5 конфигураций predict.py и собирает experiments.csv.

Запуск:
    python run_experiments.py --index ../artifacts/index \
        --questions dev_questions.jsonl --gold dev_gold.jsonl \
        --config config.json --workdir experiments

Конфигурации:
    1. baseline_no_context  — Qwen по памяти
    2. rag_one_search       — только BM25, top-3, без соседей
    3. final                — BM25 + dense + RRF + соседи
    4. ablation_no_dense    — только BM25 + соседи
    5. ablation_no_neighbors — BM25 + dense + RRF, без соседей
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


CONFIGS = [
    {"name": "baseline_no_context", "flags": {"use_bm25": False, "use_dense": False, "use_neighbors": False, "use_context": False}},
    {"name": "rag_one_search", "flags": {"use_bm25": True, "use_dense": False, "use_neighbors": False, "use_context": True}},
    {"name": "final", "flags": {"use_bm25": True, "use_dense": True, "use_neighbors": True, "use_context": True}},
    {"name": "ablation_no_dense", "flags": {"use_bm25": True, "use_dense": False, "use_neighbors": True, "use_context": True}},
    {"name": "ablation_no_neighbors", "flags": {"use_bm25": True, "use_dense": True, "use_neighbors": False, "use_context": True}},
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate(answers: list[dict], gold: list[dict]) -> tuple[float, float]:
    """Возвращает (accuracy, Q)."""
    gold_map = {g["id"]: g for g in gold}
    cats = {
        "условия": ["q_001", "q_002", "q_005", "q_006", "q_008"],
        "сопоставление": ["q_003", "q_004", "q_009", "q_012"],
        "вычисления": ["q_007", "q_010", "q_011"],
        "недостаточность": ["q_013", "q_014", "q_015"],
    }
    cat_scores: dict[str, list[float]] = {k: [] for k in cats}
    total = 0.0
    for a in answers:
        qid = a["id"]
        g = gold_map.get(qid)
        if not g:
            continue
        if g["abstain"]:
            score = 1.0 if a.get("abstain") else 0.0
        else:
            exact = (set(g["answer_ids"]) == set(a.get("answer_ids", []))) and not a.get("abstain", True)
            score = 1.0 if exact else 0.0
        total += score
        for cat, ids in cats.items():
            if qid in ids:
                cat_scores[cat].append(score)
    accuracy = total / len(gold) if gold else 0.0
    cat_means = [sum(v) / len(v) for v in cat_scores.values() if v]
    Q = sum(cat_means) / len(cat_means) if cat_means else 0.0
    return accuracy, Q


def mean_cost(trace: list[dict]) -> float:
    """C_i = input + 4*output + 256*calls, среднее по вопросам."""
    if not trace:
        return 0.0
    costs = [t["input_tokens"] + 4 * t["output_tokens"] + 256 * t["llm_calls"] for t in trace]
    return statistics.mean(costs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, default=Path("experiments"))
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    gold = load_jsonl(args.gold)
    base_config = json.loads(args.config.read_text(encoding="utf-8"))

    rows = []
    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n{'='*60}\nКонфигурация: {name}\n{'='*60}")

        # Собираем конфиг с флагами
        run_config = dict(base_config)
        run_config["experiments"] = cfg["flags"]
        cfg_path = args.workdir / f"config_{name}.json"
        cfg_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

        answers_path = args.workdir / f"{name}_answers.jsonl"
        trace_path = args.workdir / f"{name}_trace.jsonl"

        cmd = [
            sys.executable, "predict.py",
            "--index", str(args.index),
            "--questions", str(args.questions),
            "--output", str(answers_path),
            "--trace", str(trace_path),
            "--config", str(cfg_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] {name}: {result.stderr[-1000:]}")
            continue

        answers = load_jsonl(answers_path)
        trace = load_jsonl(trace_path)
        accuracy, Q = evaluate(answers, gold)
        mean_in = statistics.mean(t["input_tokens"] for t in trace)
        mean_out = statistics.mean(t["output_tokens"] for t in trace)
        mean_calls = statistics.mean(t["llm_calls"] for t in trace)
        cost = mean_cost(trace)
        median_ms = statistics.median(t["elapsed_ms"] for t in trace)

        rows.append({
            "name": name,
            "answer_accuracy": round(accuracy, 4),
            "Q": "" if name == "baseline_no_context" else round(Q, 4),
            "mean_input_tokens": round(mean_in, 1),
            "mean_output_tokens": round(mean_out, 1),
            "mean_llm_calls": round(mean_calls, 2),
            "mean_cost": round(cost, 1),
            "median_latency_ms": int(median_ms),
        })
        print(f"  accuracy={accuracy:.3f} Q={Q:.3f} "
              f"in={mean_in:.0f} out={mean_out:.0f} calls={mean_calls:.1f} "
              f"cost={cost:.0f} latency={median_ms:.0f}ms")

    # Записываем CSV
    out_csv = Path("experiments.csv")
    fieldnames = ["name", "answer_accuracy", "Q", "mean_input_tokens",
                  "mean_output_tokens", "mean_llm_calls", "mean_cost", "median_latency_ms"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n{'='*60}")
    print(f"experiments.csv записан: {out_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
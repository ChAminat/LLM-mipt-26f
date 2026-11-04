#!/usr/bin/env python3
"""
Оценка dev_answers.jsonl против dev_gold.jsonl.

Запуск:
    python evaluate.py --answers dev_answers.jsonl --gold dev_gold.jsonl
"""
import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    args = ap.parse_args()

    answers = {r["id"]: r for r in load_jsonl(args.answers)}
    gold = {r["id"]: r for r in load_jsonl(args.gold)}

    # Категории: 1-4 = условия, 5-8 = сопоставление/вычисления,
    # 9-12 = вычисления/сопоставление, 13-15 = недостаточность
    categories = {
        "условия": ["q_001", "q_002", "q_005", "q_006", "q_008"],
        "сопоставление": ["q_003", "q_004", "q_009", "q_012"],
        "вычисления": ["q_007", "q_010", "q_011"],
        "недостаточность": ["q_013", "q_014", "q_015"],
    }

    total_score = 0
    cat_scores: dict[str, list[float]] = {k: [] for k in categories}

    print(f"{'ID':<8} {'Эталон':<15} {'Ответ':<15} {'A':<3} {'abstain':<8}")
    print("-" * 60)

    for qid in sorted(gold):
        g = gold[qid]
        a = answers.get(qid, {})
        gold_ids = set(g["answer_ids"])
        ans_ids = set(a.get("answer_ids", []))
        abstain_ok = (g["abstain"] == a.get("abstain", False))

        if g["abstain"]:
            score = 1.0 if a.get("abstain") else 0.0
        else:
            exact = (gold_ids == ans_ids) and not a.get("abstain", True)
            score = 1.0 if exact else 0.0

        for cat, ids in categories.items():
            if qid in ids:
                cat_scores[cat].append(score)

        total_score += score
        print(f"{qid:<8} {str(sorted(gold_ids)):<15} {str(sorted(ans_ids)):<15} "
              f"{int(score):<3} {'✓' if abstain_ok else '✗':<8}")

    n = len(gold)
    accuracy = total_score / n
    print(f"\nТочность: {total_score:.0f}/{n} = {accuracy:.3f}")
    print("\nПо категориям (Q):")
    cat_means = {}
    for cat, scores in cat_scores.items():
        if scores:
            m = sum(scores) / len(scores)
            cat_means[cat] = m
            print(f"  {cat:<20} {m:.3f}  ({len(scores)} вопр.)")
    if cat_means:
        Q = sum(cat_means.values()) / len(cat_means)
        print(f"\nQ (среднее по категориям): {Q:.3f}")


if __name__ == "__main__":
    main()
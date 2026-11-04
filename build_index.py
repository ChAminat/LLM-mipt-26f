#!/usr/bin/env python3
"""
Строит гибридный индекс (BM25 + dense) по corpus.jsonl.

Установка:
    pip install pymupdf rank_bm25 sentence-transformers numpy

Запуск:
    python build_index.py --corpus corpus.jsonl --output index --config config.json
"""
import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ---------- Токенизация для BM25 ----------

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


# ---------- Сборка ----------

def build(corpus_path: Path, output: Path, config: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)

    embed_model = config["embedding"]["model"]
    embed_revision = config["embedding"].get("revision")

    print(f"[1/4] Читаем корпус: {corpus_path}")
    records = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    n = len(records)
    print(f"      {n} блоков")

    print("[2/4] BM25")
    t0 = time.perf_counter()
    tokenized = [tokenize(r["text_search"]) for r in records]
    bm25 = BM25Okapi(tokenized)
    bm25_sec = time.perf_counter() - t0
    print(f"      {bm25_sec:.1f}s")

    print(f"[3/4] Dense: {embed_model} (rev={embed_revision})")
    t0 = time.perf_counter()
    passages = [f"passage: {r['text_search']}" for r in records]
    model = SentenceTransformer(embed_model, revision=embed_revision, device="cpu")
    embeddings = model.encode(
        passages,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    dense_sec = time.perf_counter() - t0
    print(f"      {dense_sec:.1f}s, shape={embeddings.shape}")

    print(f"[4/4] Сохраняем в {output}")
    with (output / "bm25.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "k1": 1.5, "b": 0.75}, f)

    np.save(output / "embeddings.npy", embeddings)

    meta = {
        "n_blocks": n,
        "block_ids": [r["block_id"] for r in records],
        "embedding": {
            "model": embed_model,
            "revision": embed_revision,
            "dim": int(embeddings.shape[1]),
            "prefix_passage": "passage: ",
            "prefix_query": "query: ",
        },
        "bm25": {"k1": 1.5, "b": 0.75},
        "build_seconds": {"bm25": bm25_sec, "dense": dense_sec},
    }
    (output / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output / "texts.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "block_id": r["block_id"],
                "page": r["page"],
                "section": r["section"],
                "text": r["text"],
                "text_search": r["text_search"],
            }, ensure_ascii=False) + "\n")

    size_mb = sum(p.stat().st_size for p in output.iterdir()) / 1e6
    print(f"Готово: BM25 {bm25_sec:.1f}s + dense {dense_sec:.1f}s, index/ = {size_mb:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    build(args.corpus, args.output, config)


if __name__ == "__main__":
    main()
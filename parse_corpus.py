#!/usr/bin/env python3
"""
Простой парсер ТФКП: PyMuPDF + восстановление кириллицы + склейка формул.

Установка:
    pip install pymupdf

Запуск:
    python parse_corpus.py --pdf book.pdf --output corpus.jsonl
"""
import argparse
import json
import re
from pathlib import Path

import pymupdf


# ---------- Кириллица ----------

def fix_cyrillic(s: str) -> str:
    """cp1251-байты, прочитанные как latin-1, → нормальная кириллица."""
    out = []
    for ch in s:
        code = ord(ch)
        if code < 0x100:
            try:
                out.append(bytes([code]).decode("cp1251"))
            except UnicodeDecodeError:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


# ---------- Формулы ----------

MATH_CHARS = set("∂∫∑∏πζ∇≤≥≠±∞√∈∉⊂⊆→⇒⇔·×÷′″⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉αβγδεθλμνξρστφχψωΓΔΘΛΞΠΣΦΨΩ⩽⩾")
FORMULA_MARKERS = MATH_CHARS | set("^_{}\\") | {"\x0c"}


def looks_like_formula(line: str) -> bool:
    if not line.strip():
        return False
    hits = sum(1 for ch in line if ch in FORMULA_MARKERS)
    return hits >= 1 and hits / max(len(line), 1) > 0.02


def formula_to_text(line: str) -> str:
    """Текстовый дубль формулы для BM25."""
    repl = {
        "∂": "d", "∫": "int ", "∑": "sum ", "∏": "prod ",
        "π": "pi", "ζ": "zeta", "∇": "nabla", "≤": "<=", "≥": ">=",
        "⩽": "<=", "⩾": ">=", "≠": "!=", "±": "+-", "∞": "inf",
        "√": "sqrt", "∈": " in ", "∉": " not in ",
        "⊂": " subset ", "⊆": " subseteq ",
        "→": "->", "⇒": "=>", "⇔": "<=>", "·": "*", "×": "*", "÷": "/",
        "′": "'", "″": "''",
    }
    out = []
    for ch in line:
        if ch == "\x0c":
            out.append(" ")
        elif ch in repl:
            out.append(repl[ch])
        elif ch == "̄":
            out.append("_bar")
        elif ch in "⁰¹²³⁴⁵⁶⁷⁸⁹":
            out.append("^" + "0123456789"["⁰¹²³⁴⁵⁶⁷⁸⁹".index(ch)])
        elif ch in "₀₁₂₃₄₅₆₇₈₉":
            out.append("_" + "0123456789"["₀₁₂₃₄₅₆₇₈₉".index(ch)])
        else:
            out.append(ch)
    return "".join(out)


def merge_formula_lines(lines):
    """Склеивает подряд идущие строки-формулы. Возвращает [(clean, search_or_None)]."""
    result = []
    buf = []

    def flush():
        nonlocal buf
        if not buf:
            return
        joined = " ".join(s.strip() for s in buf if s.strip())
        if not joined:
            buf = []
            return
        plain = formula_to_text(joined).strip()
        clean = f"${joined}$"
        search = f"${joined}$ [formula: {plain}]" if plain and plain != joined else clean
        result.append((clean, search))
        buf = []

    for line in lines:
        if looks_like_formula(line):
            buf.append(line)
        else:
            flush()
            result.append((line, None))
    flush()
    return result


# ---------- Нарезка на блоки ----------

BLOCK_RE = re.compile(
    r"(?m)^(?:"
    r"(?P<theorem>Теорема\s+\d+(?:\.\d+)?)"
    r"|(?P<definition>Определение\s+\d+(?:\.\d+)?)"
    r"|(?P<corollary>Следствие\s+\d+(?:\.\d+)?)"
    r"|(?P<remark>Замечание\s+\d+(?:\.\d+)?)"
    r"|(?P<example>Пример\s+\d+(?:\.\d+)?)"
    r"|(?P<proof>Доказательство)"
    r")\b"
)

SECTION_RE = re.compile(r"(?m)^(\d+(?:\.\d+)?)\.\s+[А-ЯЁ]")
TOC_RE = re.compile(r"\.\s*\.\s*\.\s*\.")


def is_toc(text: str) -> bool:
    if "Содержание" in text[:200]:
        return True
    dots = len(TOC_RE.findall(text))
    return dots >= 3 and dots / max(len(text.splitlines()), 1) > 0.15


def extract_blocks(page_text, page_no, current_section):
    """Делит страницу на блоки по маркерам."""
    for m in SECTION_RE.finditer(page_text):
        current_section = m.group(1)

    matches = list(BLOCK_RE.finditer(page_text))
    if not matches:
        blocks = [{"page": page_no, "section": current_section,
                   "raw_text": page_text.strip()}] if page_text.strip() else []
        return blocks, current_section

    blocks = []
    pre = page_text[: matches[0].start()].strip()
    if len(pre) > 50:
        blocks.append({"page": page_no, "section": current_section, "raw_text": pre})

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
        block_text = page_text[start:end].strip()
        if len(block_text) < 20:
            continue
        blocks.append({"page": page_no, "section": current_section, "raw_text": block_text})
    return blocks, current_section


# ---------- Сборка ----------

def build_corpus(pdf_path: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    current_section = ""
    n = n_formula = 0

    with output.open("w", encoding="utf-8") as f:
        for pno in range(doc.page_count):
            raw = doc[pno].get_text()
            fixed = fix_cyrillic(raw)
            blocks, current_section = extract_blocks(fixed, pno + 1, current_section)
            for block in blocks:
                if is_toc(block["raw_text"]):
                    continue
                merged = merge_formula_lines(block["raw_text"].split("\n"))
                clean_parts, search_parts = [], []
                for clean, search in merged:
                    clean_parts.append(clean)
                    search_parts.append(search if search is not None else clean)
                clean_text = "\n".join(clean_parts).strip()
                search_text = "\n".join(search_parts).strip()
                if len(clean_text) < 30:
                    continue
                n += 1
                if "[formula:" in search_text:
                    n_formula += 1
                f.write(json.dumps({
                    "block_id": f"b_{n:04d}",
                    "page": block["page"],
                    "section": block["section"],
                    "text": clean_text,
                    "text_search": search_text,
                }, ensure_ascii=False) + "\n")

    doc.close()
    print(f"Блоков: {n}")
    print(f"С формулами: {n_formula}")
    print(f"Выход: {output} ({output.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    build_corpus(args.pdf, args.output)


if __name__ == "__main__":
    main()
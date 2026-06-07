"""
Construction d'un jeu d'évaluation (question, réponse de référence, passage-or).

Approche : pour des passages tirés au hasard dans le corpus, on demande au LLM
de générer UNE question factuelle dont la réponse se trouve dans le passage,
ainsi qu'une réponse courte. On garde le `chunk_id` source comme "gold" pour
mesurer la qualité de la récupération.
"""
from __future__ import annotations

import json
import random
import re

from .config import CONFIG, CHUNKS_PATH, EVAL_PATH
from .data_loader import load_jsonl, save_jsonl
from .generator import Generator

_QA_PROMPT = (
    "Voici un extrait de documentation technique :\n\n"
    "\"\"\"{passage}\"\"\"\n\n"
    "Génère UNE question factuelle en français dont la réponse se trouve "
    "explicitement dans cet extrait, puis donne la réponse courte.\n"
    "Réponds STRICTEMENT au format :\n"
    "Question: <ta question>\n"
    "Réponse: <ta réponse courte>"
)

_Q = re.compile(r"Question\s*:\s*(.+)", re.IGNORECASE)
_A = re.compile(r"R[ée]ponse\s*:\s*(.+)", re.IGNORECASE)


def _parse_qa(text: str) -> tuple[str, str] | None:
    q = _Q.search(text)
    a = _A.search(text)
    if not q or not a:
        return None
    question, answer = q.group(1).strip(), a.group(1).strip()
    if len(question) < 8 or len(answer) < 1:
        return None
    return question, answer


def build_eval_set(n: int = 100, seed: int | None = None) -> list[dict]:
    random.seed(seed or CONFIG.seed)
    chunks = load_jsonl(CHUNKS_PATH)
    # on privilégie les passages assez longs pour contenir un fait clair
    candidates = [c for c in chunks if len(c["text"]) > 350]
    random.shuffle(candidates)

    gen = Generator()
    rows: list[dict] = []
    for c in candidates:
        if len(rows) >= n:
            break
        raw = gen._generate([{  # noqa: SLF001 (usage interne assumé)
            "role": "user",
            "content": _QA_PROMPT.format(passage=c["text"][:1200]),
        }])
        parsed = _parse_qa(raw)
        if parsed is None:
            continue
        question, answer = parsed
        rows.append({
            "question": question,
            "answer": answer,
            "gold_chunk_id": c["chunk_id"],
            "gold_doc_id": c["doc_id"],
            "url": c["url"],
        })
    print(f"[eval] {len(rows)} paires question/réponse générées")
    return rows


if __name__ == "__main__":
    save_jsonl(build_eval_set(n=100), EVAL_PATH)

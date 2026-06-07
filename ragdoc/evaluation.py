"""
Métriques d'évaluation.

Récupération (on connaît le passage "gold" de chaque question) :
  - Hit@k    : le passage gold est-il dans le top-k ?
  - Recall@k : identique au Hit@k ici (un seul passage pertinent par question)
  - MRR      : moyenne de 1/rang du premier passage pertinent

Génération (réponse prédite vs réponse de référence) :
  - Exact Match (EM) normalisé
  - F1 au niveau des tokens
  - ROUGE-L (si la librairie `rouge-score` est installée)
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter


# Normalisation de texte (pour EM / F1)
def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")  # enlève les accents
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(le|la|les|un|une|des|de|du|et|ou|a|au|aux)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Métriques de génération
def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def token_f1(pred: str, ref: str) -> float:
    p_tokens = normalize(pred).split()
    r_tokens = normalize(ref).split()
    if not p_tokens or not r_tokens:
        return float(p_tokens == r_tokens)
    common = Counter(p_tokens) & Counter(r_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(r_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return scorer.score(ref, pred)["rougeL"].fmeasure


# Métriques de récupération
def retrieval_metrics(retrieved_ids: list[str], gold_id: str, k_values) -> dict:
    out = {}
    rank = next((i for i, cid in enumerate(retrieved_ids) if cid == gold_id), None)
    for k in k_values:
        out[f"hit@{k}"] = float(rank is not None and rank < k)
    out["mrr"] = 0.0 if rank is None else 1.0 / (rank + 1)
    return out


# Agrégation
def mean(values: list[float]) -> float:
    valid = [v for v in values if v == v]  # filtre les NaN
    return sum(valid) / len(valid) if valid else float("nan")


def aggregate(per_example: list[dict]) -> dict:
    if not per_example:
        return {}
    keys = per_example[0].keys()
    return {k: round(mean([row[k] for row in per_example]), 4) for k in keys}

"""
Étape 4 — Évaluation et expériences.

Produit :
  (A) Métriques de RÉCUPÉRATION pour chaque k (Hit@k, MRR) sur tout le jeu.
  (B) Métriques de GÉNÉRATION comparant closed-book vs RAG (EM, F1, ROUGE-L)
      sur un sous-échantillon (la génération est coûteuse).

Lance avec l'index de base ou l'index spécialisé (--finetuned) pour comparer.
Les résultats sont écrits dans results/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragdoc.config import CONFIG, EVAL_PATH, INDEX_DIR, RESULTS_DIR
from ragdoc.data_loader import load_jsonl
from ragdoc.evaluation import (aggregate, exact_match, retrieval_metrics,
                               rouge_l, token_f1)
from ragdoc.pipeline import RagPipeline
from ragdoc.retriever import Retriever


def eval_retrieval(retriever: Retriever, eval_rows: list[dict]) -> dict:
    k_values = CONFIG.retrieval.k_values
    max_k = max(k_values)
    per_example = []
    for row in eval_rows:
        hits = retriever.search(row["question"], top_k=max_k)
        ids = [h["chunk_id"] for h in hits]
        per_example.append(retrieval_metrics(ids, row["gold_chunk_id"], k_values))
    return aggregate(per_example)


def eval_generation(pipeline: RagPipeline, eval_rows: list[dict]) -> dict:
    rag_rows, cb_rows, examples = [], [], []
    for row in eval_rows:
        ref = row["answer"]
        rag = pipeline.answer(row["question"])
        cb = pipeline.answer_closed_book(row["question"])

        rag_rows.append({"EM": exact_match(rag["answer"], ref),
                         "F1": token_f1(rag["answer"], ref),
                         "ROUGE_L": rouge_l(rag["answer"], ref)})
        cb_rows.append({"EM": exact_match(cb["answer"], ref),
                        "F1": token_f1(cb["answer"], ref),
                        "ROUGE_L": rouge_l(cb["answer"], ref)})
        examples.append({"question": row["question"], "reference": ref,
                         "rag_answer": rag["answer"], "closed_book_answer": cb["answer"]})
    return {"rag": aggregate(rag_rows),
            "closed_book": aggregate(cb_rows),
            "examples": examples[:10]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", action="store_true",
                        help="évaluer avec l'index spécialisé")
    parser.add_argument("--n_gen", type=int, default=30,
                        help="nb d'exemples pour l'évaluation de génération")
    parser.add_argument("--skip_generation", action="store_true")
    args = parser.parse_args()

    index_dir = (INDEX_DIR.parent / "faiss_index_finetuned"
                 if args.finetuned else INDEX_DIR)
    tag = "finetuned" if args.finetuned else "base"

    eval_rows = load_jsonl(EVAL_PATH)
    print(f"[eval] {len(eval_rows)} exemples — index = {tag}")

    retriever = Retriever.load(index_dir)
    t0 = time.time()
    report = {"tag": tag, "n_eval": len(eval_rows),
              "retrieval": eval_retrieval(retriever, eval_rows)}
    print("\n=== RÉCUPÉRATION ===")
    print(json.dumps(report["retrieval"], indent=2, ensure_ascii=False))

    if not args.skip_generation:
        from ragdoc.generator import Generator
        pipeline = RagPipeline(retriever, Generator())
        subset = eval_rows[:args.n_gen]
        report["generation"] = eval_generation(pipeline, subset)
        report["generation"]["n_gen"] = len(subset)
        print("\n=== GÉNÉRATION (RAG vs closed-book) ===")
        print("RAG        :", report["generation"]["rag"])
        print("Closed-book:", report["generation"]["closed_book"])

    report["elapsed_sec"] = round(time.time() - t0, 1)
    out = RESULTS_DIR / f"report_{tag}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[eval] Rapport écrit -> {out}")

"""
Spécialisation (fine-tuning) du modèle d'embedding sur le domaine.

On entraîne le retrieveur à rapprocher chaque question de SON passage gold,
avec la perte MultipleNegativesRankingLoss (les autres passages du batch
servent de négatifs). C'est la méthode de référence pour spécialiser un
bi-encodeur de recherche sémantique.

Entrée  : eval_set.jsonl (question -> gold_chunk_id) + chunks.jsonl (textes)
Sortie  : un modèle sauvegardé dans CONFIG.model.finetuned_embedding_dir
"""
from __future__ import annotations

import random

from .config import CONFIG, CHUNKS_PATH, EVAL_PATH
from .data_loader import load_jsonl


def build_training_pairs(holdout_ratio: float = 0.3):
    """Sépare le jeu QA en train/test et construit les paires (question, passage)."""
    chunks = {c["chunk_id"]: c["text"] for c in load_jsonl(CHUNKS_PATH)}
    qa = [r for r in load_jsonl(EVAL_PATH) if r["gold_chunk_id"] in chunks]
    random.seed(CONFIG.seed)
    random.shuffle(qa)

    n_test = int(len(qa) * holdout_ratio)
    test, train = qa[:n_test], qa[n_test:]

    # préfixes e5 cohérents avec le retrieveur
    qp = "query: " if CONFIG.model.use_e5_prefixes else ""
    pp = "passage: " if CONFIG.model.use_e5_prefixes else ""
    pairs = [(qp + r["question"], pp + chunks[r["gold_chunk_id"]]) for r in train]
    return pairs, train, test


def finetune(epochs: int = 2, batch_size: int = 16):
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    pairs, train, test = build_training_pairs()
    print(f"[finetune] {len(pairs)} paires d'entraînement, {len(test)} en test")

    model = SentenceTransformer(CONFIG.model.embedding_model)
    examples = [InputExample(texts=[q, p]) for q, p in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=int(0.1 * len(loader) * epochs),
        show_progress_bar=True,
    )
    out_dir = CONFIG.model.finetuned_embedding_dir
    model.save(out_dir)
    print(f"[finetune] Modèle spécialisé sauvegardé -> {out_dir}")
    return out_dir, test


if __name__ == "__main__":
    finetune()

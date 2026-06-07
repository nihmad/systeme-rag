"""
Retrieveur : encodage des passages avec un modèle d'embedding (sentence-
transformers) et recherche par similarité cosinus via FAISS.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CONFIG, INDEX_DIR


class Retriever:
    def __init__(self, model_name: str | None = None, use_e5_prefixes: bool | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or CONFIG.model.embedding_model
        self.use_e5_prefixes = (
            CONFIG.model.use_e5_prefixes if use_e5_prefixes is None else use_e5_prefixes
        )
        print(f"[retriever] Chargement de l'embedding model : {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.chunks: list[dict] = []

    # préfixes e5 
    def _prep(self, texts: list[str], kind: str) -> list[str]:
        if not self.use_e5_prefixes:
            return texts
        prefix = "query: " if kind == "query" else "passage: "
        return [prefix + t for t in texts]

    def _encode(self, texts: list[str], kind: str) -> np.ndarray:
        emb = self.model.encode(
            self._prep(texts, kind),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True, 
            show_progress_bar=len(texts) > 64,
        )
        return emb.astype("float32")

    #construction / persistance de l'index
    def build(self, chunks: list[dict]) -> "Retriever":
        import faiss

        self.chunks = chunks
        emb = self._encode([c["text"] for c in chunks], kind="passage")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        print(f"[retriever] Index construit : {self.index.ntotal} vecteurs, dim={emb.shape[1]}")
        return self

    def save(self, index_dir: Path = INDEX_DIR) -> None:
        import faiss

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (index_dir / "meta.json").write_text(
            json.dumps({"model_name": self.model_name,
                        "use_e5_prefixes": self.use_e5_prefixes}, ensure_ascii=False))
        print(f"[retriever] Index sauvegardé dans {index_dir}")

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "Retriever":
        import faiss

        meta = json.loads((index_dir / "meta.json").read_text())
        obj = cls(model_name=meta["model_name"], use_e5_prefixes=meta["use_e5_prefixes"])
        obj.index = faiss.read_index(str(index_dir / "index.faiss"))
        with (index_dir / "chunks.jsonl").open(encoding="utf-8") as f:
            obj.chunks = [json.loads(line) for line in f if line.strip()]
        print(f"[retriever] Index chargé : {obj.index.ntotal} vecteurs")
        return obj

    # recherche
    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or CONFIG.retrieval.top_k
        q = self._encode([query], kind="query")
        scores, idx = self.index.search(q, top_k)
        results = []
        for rank, (i, s) in enumerate(zip(idx[0], scores[0])):
            c = dict(self.chunks[int(i)])
            c["score"] = float(s)
            c["rank"] = rank
            results.append(c)
        return results

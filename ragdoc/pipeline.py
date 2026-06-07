"""
Pipeline RAG : assemble le retrieveur et le générateur.

    rag = RagPipeline(retriever, generator)
    out = rag.answer("À quoi sert la balise <article> ?")
    print(out["answer"], out["contexts"])
"""
from __future__ import annotations

from .config import CONFIG
from .generator import Generator
from .retriever import Retriever


class RagPipeline:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str, top_k: int | None = None) -> dict:
        top_k = top_k or CONFIG.retrieval.top_k
        hits = self.retriever.search(question, top_k=top_k)
        contexts = [h["text"] for h in hits]
        answer = self.generator.answer_with_context(question, contexts)
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
            "sources": [{"title": h["title"], "url": h["url"], "score": h["score"]}
                        for h in hits],
        }

    def answer_closed_book(self, question: str) -> dict:
        """Réponse sans récupération — pour la comparaison closed-book vs RAG."""
        return {"question": question,
                "answer": self.generator.answer_closed_book(question),
                "contexts": [], "retrieved_chunk_ids": [], "sources": []}

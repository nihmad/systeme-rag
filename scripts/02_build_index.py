"""Étape 2 — Encode les passages et construit l'index vectoriel FAISS.

Option : --finetuned pour indexer avec le modèle d'embedding spécialisé.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragdoc.config import CHUNKS_PATH, CONFIG, INDEX_DIR
from ragdoc.data_loader import load_jsonl
from ragdoc.retriever import Retriever

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", action="store_true",
                        help="utiliser le modèle d'embedding spécialisé")
    args = parser.parse_args()

    model_name = (CONFIG.model.finetuned_embedding_dir if args.finetuned
                  else CONFIG.model.embedding_model)
    index_dir = INDEX_DIR.parent / ("faiss_index_finetuned" if args.finetuned
                                    else "faiss_index")

    chunks = load_jsonl(CHUNKS_PATH)
    retriever = Retriever(model_name=model_name).build(chunks)
    retriever.save(index_dir)

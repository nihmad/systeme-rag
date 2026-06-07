"""Étape 1 — Construit le corpus de passages à partir de MDN FR."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragdoc.config import CHUNKS_PATH
from ragdoc.data_loader import save_jsonl
from ragdoc.preprocessing import build_chunks

if __name__ == "__main__":
    save_jsonl(build_chunks(), CHUNKS_PATH)

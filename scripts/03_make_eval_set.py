"""Étape 3 — Génère le jeu d'évaluation (questions/réponses + passage gold)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragdoc.config import EVAL_PATH
from ragdoc.data_loader import save_jsonl
from ragdoc.eval_set import build_eval_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100, help="nombre de paires Q/R")
    args = parser.parse_args()
    save_jsonl(build_eval_set(n=args.n), EVAL_PATH)

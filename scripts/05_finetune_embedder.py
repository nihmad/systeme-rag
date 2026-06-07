"""Étape 5 (optionnelle) — Spécialise le modèle d'embedding sur le domaine."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragdoc.finetune_embedder import finetune

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    finetune(epochs=args.epochs, batch_size=args.batch_size)

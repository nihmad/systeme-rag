"""Configuration centrale du projet."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Chemins
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

# Corpus brut (clone partiel de MDN) et corpus nettoyé/découpé
MDN_CLONE_DIR = DATA_DIR / "mdn_clone"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"          # corpus découpé en passages
EVAL_PATH = DATA_DIR / "eval_set.jsonl"          # jeu d'évaluation Q/R
INDEX_DIR = DATA_DIR / "faiss_index"             # index vectoriel + métadonnées

for _d in (DATA_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# Corpus : sous-ensemble de MDN translated-content (français)
# Dépôt : https://github.com/mdn/translated-content
# On ne clone qu'un sous-dossier pour rester léger.
MDN_REPO_URL = "https://github.com/mdn/translated-content.git"
# Dossiers à récupérer via sparse-checkout (chemins relatifs dans le dépôt).
MDN_SPARSE_PATHS: tuple[str, ...] = (
    "files/fr/web/html",
    "files/fr/web/css",
    "files/fr/web/javascript/guide",
)


# Découpage en passages (chunking)
@dataclass
class ChunkConfig:
    chunk_size: int = 800        # taille cible d'un passage (en caractères)
    chunk_overlap: int = 120     # recouvrement entre deux passages consécutifs
    min_chunk_chars: int = 200   # on retire les passages trop courts


# Modèles
@dataclass
class ModelConfig:
    # Modèle d'embedding (retrieveur). multilingual-e5 attend les préfixes
    # "query: " et "passage: " — gérés dans retriever.py.
    embedding_model: str = "intfloat/multilingual-e5-base"
    use_e5_prefixes: bool = True

    # Modèle d'embedding spécialisé (rempli après fine-tuning)
    finetuned_embedding_dir: str = str(DATA_DIR / "embedder_finetuned")

    # Modèle générateur.
    generator_model: str = "unsloth/mistral-7b-instruct-v0.3"
    load_in_4bit: bool = True

    max_new_tokens: int = 256
    temperature: float = 0.3


# Récupération (retrieval)
@dataclass
class RetrievalConfig:
    top_k: int = 4         
    k_values: tuple[int, ...] = (1, 3, 5, 10)  # valeurs de k


# Objet de configuration global
@dataclass
class Config:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    seed: int = 42


CONFIG = Config()

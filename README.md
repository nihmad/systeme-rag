# RAG sur de la documentation technique francophone

Système **RAG (Retrieval-Augmented Generation)** appliqué à la documentation
web de **MDN en français**. Projet réalisé dans le cadre du cours *Large Language
Models*.

L'objectif : montrer concrètement qu'un LLM relié à une base documentaire
(via récupération sémantique) répond de façon plus **fidèle** et **exacte**
qu'un LLM seul, et étudier ce qui fait varier cette performance.


## 1. Problématique

> Un système RAG sur une documentation technique française permet-il à un LLM de
> répondre de façon plus fidèle et exacte qu'en *closed-book* (sans contexte) ?
> Quel est l'impact **(a)** du nombre de passages récupérés `k` et
> **(b)** de la spécialisation du retrieveur par fine-tuning ?

On y répond par trois expériences comparatives :

1. **closed-book vs RAG** — gain apporté par la récupération (EM, F1, ROUGE-L).
2. **ablation sur `k`** — évolution de Hit@k et du MRR selon le nombre de passages.
3. **retrieveur de base vs spécialisé** — gain du fine-tuning de l'embedding model.


## 2. Cas d'usage et données

- **Corpus** : sous-ensemble de [MDN translated-content](https://github.com/mdn/translated-content)
  en français (documentation HTML / CSS / guide JavaScript), licence **CC-BY-SA**.
  Récupéré par *clone partiel* (shallow + sparse-checkout) pour rester léger.
- **Exploration & nettoyage** : suppression des macros `{{...}}`, blocs de code,
  balises HTML et liens Markdown, puis découpage en passages de ~800 caractères
  avec recouvrement (voir `ragdoc/preprocessing.py`).
- **Jeu d'évaluation** : généré automatiquement à partir du corpus — pour des
  passages tirés au hasard, le LLM produit une question factuelle et sa réponse ;
  le passage source sert de **vérité terrain** pour mesurer la récupération
  (`ragdoc/eval_set.py`).


## 3. Architecture

```
Question ──► [Retrieveur] ──► top-k passages ──► [Générateur LLM] ──► Réponse
                 │                                       │
        embeddings + FAISS                    prompt = contexte + question
```

| Composant      | Choix par défaut                              | Pourquoi |
|----------------|-----------------------------------------------|----------|
| Embeddings     | `intfloat/multilingual-e5-base`               | multilingue, fort en français |
| Index          | FAISS `IndexFlatIP` (cosinus)                 | simple, exact, local |
| Générateur     | `croissantllm/CroissantLLMChat-v0.1` (~1.3B)  | libre, bilingue FR/EN, tient sur un T4 gratuit |
| Spécialisation | fine-tuning de l'embedder (`MultipleNegativesRankingLoss`) | rapproche question ↔ passage du domaine |

> Alternative générateur plus puissante : `mistralai/Mistral-7B-Instruct-v0.3`
> avec quantization 4 bits (`CONFIG.model.load_in_4bit = True`). Ce modèle est
> *gated* sur Hugging Face : il faut accepter sa licence et fournir un token.


## 4. Installation

### Option A — Google Colab (recommandée)
Ouvrez `notebooks/RAG_doc_technique_fr_colab.ipynb` dans Colab, activez le GPU
(*Exécution → Modifier le type d'exécution → GPU T4*) et exécutez les cellules
dans l'ordre.

### Option B — En local
```bash
git clone https://github.com/VOTRE_USER/rag-doc-technique-fr.git
cd rag-doc-technique-fr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Un GPU est fortement recommandé pour le générateur (CPU possible mais lent).


## 5. Reproduire le projet (pas à pas)

```bash
# 1. Construire le corpus de passages (clone MDN + nettoyage + découpage)
python scripts/01_build_corpus.py

# 2. Encoder et indexer les passages (FAISS)
python scripts/02_build_index.py

# 3. Générer le jeu d'évaluation question/réponse
python scripts/03_make_eval_set.py -n 100

# 4. Évaluer : RAG vs closed-book + ablation sur k  -> results/report_base.json
python scripts/04_run_evaluation.py --n_gen 30

# 5. (Optionnel) Spécialiser le retrieveur, réindexer, réévaluer
python scripts/05_finetune_embedder.py --epochs 2
python scripts/02_build_index.py --finetuned
python scripts/04_run_evaluation.py --finetuned --n_gen 30
```

Démo rapide en Python :
```python
from ragdoc.retriever import Retriever
from ragdoc.generator import Generator
from ragdoc.pipeline import RagPipeline

rag = RagPipeline(Retriever.load(), Generator())
print(rag.answer("À quoi sert l'élément HTML <article> ?")["answer"])
```


## 6. Structure du dépôt

```
rag-doc-technique-fr/
├── README.md
├── requirements.txt
├── ragdoc/                     # bibliothèque du projet
│   ├── config.py               # tous les réglages (modèles, chemins, k…)
│   ├── data_loader.py          # clone partiel MDN + parsing Markdown
│   ├── preprocessing.py        # nettoyage + découpage en passages
│   ├── retriever.py            # embeddings + index FAISS + recherche
│   ├── generator.py            # chargement du LLM + génération
│   ├── pipeline.py             # orchestration RAG (et closed-book)
│   ├── eval_set.py             # génération du jeu d'évaluation
│   ├── finetune_embedder.py    # spécialisation du retrieveur
│   └── evaluation.py           # métriques (Hit@k, MRR, EM, F1, ROUGE-L)
├── scripts/                    # points d'entrée en ligne de commande (01→05)
├── notebooks/
│   └── RAG_doc_technique_fr_colab.ipynb   # exécution de bout en bout
├── data/                       # corpus, index, jeu d'éval (régénérables)
└── results/                    # rapports d'évaluation (JSON)
```


## 7. Métriques

- **Récupération** : `Hit@k` (le bon passage est-il dans le top-k ?), `MRR`.
- **Génération** : `Exact Match` et `F1` au niveau des tokens (normalisés :
  minuscules, sans accents ni ponctuation), `ROUGE-L`.
- **(Optionnel)** métriques RAG avancées via [RAGAS](https://github.com/explodinggradients/ragas)
  (*faithfulness*, *answer relevancy*, *context precision/recall*) — nécessite un
  LLM juge ; dépendance commentée dans `requirements.txt`.


## 8. Résultats

> À compléter avec vos chiffres après exécution (les rapports sont écrits dans
> `results/`). Exemple de tableau à remplir :

| Configuration            | Hit@1 | Hit@3 | MRR  | F1 (RAG) | F1 (closed-book) |
|--------------------------|-------|-------|------|----------|------------------|
| Embedder de base         |  …    |  …    |  …   |   …      |        …         |
| Embedder spécialisé      |  …    |  …    |  …   |   …      |        —         |

Pistes d'analyse : gain du RAG vs closed-book, palier de `k`, apport du
fine-tuning, cas d'échec (qualité du petit LLM, bruit du corpus…).


## 9. Limites

- Le générateur par défaut (~1.3B) reste modeste : utile pédagogiquement, mais un
  modèle 7B quantifié donne de meilleures réponses.
- Le jeu d'évaluation est généré par un LLM : il peut contenir des questions
  imparfaites — un filtrage/relecture d'un échantillon est recommandé.
- Le contenu de MDN évolue dans le temps ; figez éventuellement un commit du
  dépôt source pour une reproductibilité parfaite.


## 10. Crédits et licence

- Corpus : [MDN Web Docs](https://developer.mozilla.org/), Mozilla Contributors,
  licence **CC-BY-SA**.
- Modèles : CroissantLLM (CentraleSupélec et al.), e5 (Microsoft), via Hugging Face.
- Code de ce projet : libre d'utilisation à des fins pédagogiques.

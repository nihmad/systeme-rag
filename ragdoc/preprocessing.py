"""
Nettoyage et découpage du corpus en passages (chunks).

Le Markdown de MDN contient beaucoup de bruit pour un RAG : macros `{{...}}`,
blocs de code, balises HTML résiduelles, liens. On nettoie, puis on découpe
chaque document en passages de taille fixe avec recouvrement.
"""
from __future__ import annotations

import re

from .config import CONFIG
from .data_loader import load_documents, save_jsonl
from .config import CHUNKS_PATH

#  expressions régulières de nettoyage 
_MACRO = re.compile(r"\{\{.*?\}\}")              # macros Kuma : {{HTMLElement}}
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)  # blocs de code
_HTML_TAG = re.compile(r"<[^>]+>")               # balises HTML
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # [texte](url) -> texte
_MD_HEADER = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Retire le balisage et le bruit, garde le texte lisible."""
    text = _CODE_BLOCK.sub(" ", text)
    text = _MACRO.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub(" ", text)
    text = _MD_HEADER.sub("", text)
    text = text.replace("`", "").replace("*", "").replace(">", " ")
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Découpe un texte en fenêtres de `size` caractères avec recouvrement.

    On essaie de couper sur une frontière de phrase proche pour ne pas
    casser les mots/phrases au milieu.
    """
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        # on cherche une fin de phrase entre end-100 et end pour couper proprement
        if end < n:
            window = text[max(start, end - 100):end]
            cut = max(window.rfind(". "), window.rfind("\n"))
            if cut != -1:
                end = max(start, end - 100) + cut + 1
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - overlap
    return chunks


def build_chunks() -> list[dict]:
    """Pipeline complet : documents -> nettoyage -> passages indexables."""
    cfg = CONFIG.chunk
    docs = load_documents()
    rows: list[dict] = []
    for doc in docs:
        cleaned = clean_text(doc["text"])
        if len(cleaned) < cfg.min_chunk_chars:
            continue
        for i, chunk in enumerate(chunk_text(cleaned, cfg.chunk_size, cfg.chunk_overlap)):
            if len(chunk) < cfg.min_chunk_chars:
                continue
            rows.append({
                "chunk_id": f"{doc['id']}::{i}",
                "doc_id": doc["id"],
                "title": doc["title"],
                "url": doc["url"],
                "text": chunk,
            })
    print(f"[chunking] {len(docs)} documents -> {len(rows)} passages")
    return rows


if __name__ == "__main__":
    chunks = build_chunks()
    save_jsonl(chunks, CHUNKS_PATH)

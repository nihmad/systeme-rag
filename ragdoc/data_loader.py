"""
Chargement du corpus : documentation technique MDN en français.

Étapes :
  1. clone *partiel* (shallow + sparse-checkout) du dépôt MDN translated-content,
     limité aux sous-dossiers français listés dans config.MDN_SPARSE_PATHS ;
  2. lecture des fichiers Markdown (`index.md`) et extraction du front-matter
     (titre + slug) et du corps.

On obtient une liste de documents { "id", "title", "url", "text" }.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .config import CONFIG, MDN_CLONE_DIR, MDN_REPO_URL, MDN_SPARSE_PATHS


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    """Exécute une commande shell en levant une exception si elle échoue."""
    print("  $", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_mdn(force: bool = False) -> Path:
    """Clone partiel du dépôt MDN (uniquement les dossiers FR voulus)."""
    if MDN_CLONE_DIR.exists() and not force:
        print(f"[corpus] Clone déjà présent : {MDN_CLONE_DIR}")
        return MDN_CLONE_DIR

    if MDN_CLONE_DIR.exists():
        _run(["rm", "-rf", str(MDN_CLONE_DIR)])

    print("[corpus] Clone partiel de MDN (cela peut prendre 1-2 min)…")
    # --filter=blob:none + sparse-checkout = on ne télécharge que ce qu'il faut.
    _run(["git", "clone", "--depth", "1", "--filter=blob:none",
          "--sparse", MDN_REPO_URL, str(MDN_CLONE_DIR)])
    _run(["git", "sparse-checkout", "set", *MDN_SPARSE_PATHS], cwd=MDN_CLONE_DIR)
    return MDN_CLONE_DIR


_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    """Sépare le front-matter YAML (titre, slug…) du corps Markdown."""
    m = _FRONT_MATTER.match(raw)
    if not m:
        return {}, raw
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip('"').strip("'")
    return meta, raw[m.end():]


def load_documents() -> list[dict]:
    """Parcourt le clone et renvoie la liste des documents bruts."""
    root = clone_mdn()
    docs: list[dict] = []
    md_files = sorted(root.rglob("index.md"))
    print(f"[corpus] {len(md_files)} fichiers index.md trouvés")

    for path in md_files:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        meta, body = _parse_front_matter(raw)
        slug = meta.get("slug", str(path.relative_to(root)))
        docs.append({
            "id": slug,
            "title": meta.get("title", slug),
            "url": f"https://developer.mozilla.org/fr/docs/{slug}",
            "text": body,
        })
    return docs


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[io] {len(rows)} lignes écrites -> {path}")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    documents = load_documents()
    print(f"\nExemple de document :\n{json.dumps(documents[0], ensure_ascii=False)[:500]}…")

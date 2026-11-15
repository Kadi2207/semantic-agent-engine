"""Fine-tuning contrastif de multilingual-e5-small sur les triplets (ancre, positif, negatif dur).

Entree  : data/processed/triplets_dev.jsonl + data/processed/triplets_llm.jsonl (fusionnes, dedoublonnes)
Sortie  : models/e5-finetuned (non versionne)
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
TRIPLET_FILES = [
    ROOT / "data" / "processed" / "triplets_dev.jsonl",
    ROOT / "data" / "processed" / "triplets_llm.jsonl",
]
OUTPUT_DIR = ROOT / "models" / "e5-finetuned"

BASE_MODEL = "intfloat/multilingual-e5-small"
EPOCHS = 3
BATCH_SIZE = 16
WARMUP_RATIO = 0.10
LEARNING_RATE = 2e-5  # valeur par defaut de sentence-transformers, explicitee pour l'affichage
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_triplets(paths: list[Path]) -> tuple[list[dict[str, str]], dict[str, tuple[int, int]]]:
    """Fusionne les fichiers et dedoublonne sur (anchor, positive, negative).

    Retourne les triplets et, par fichier, (lignes lues, triplets nouveaux apportes).
    """
    missing = [p for p in paths if not p.exists()]
    if missing:
        names = ", ".join(str(p.relative_to(ROOT)) for p in missing)
        raise FileNotFoundError(
            f"fichier(s) de triplets absent(s) : {names}. "
            "Lance d'abord scripts/build_triplets_llm.py --write."
        )
    seen: set[tuple[str, str, str]] = set()
    triplets: list[dict[str, str]] = []
    stats: dict[str, tuple[int, int]] = {}
    for path in paths:
        read = added = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            read += 1
            record = json.loads(line)
            key = (record["anchor"], record["positive"], record["negative"])
            if key not in seen:
                seen.add(key)
                triplets.append(record)
                added += 1
        stats[path.name] = (read, added)
    return triplets, stats


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        triplets, stats = load_triplets(TRIPLET_FILES)
    except FileNotFoundError as exc:
        sys.exit(f"ERREUR : {exc}")
    for name, (read, added) in stats.items():
        print(f"{name:<22} {read:>4} lignes lues, {added:>4} retenues apres dedoublonnage")
    print(f"TOTAL exemples d'entrainement : {len(triplets)}")
    if len(triplets) < BATCH_SIZE:
        sys.exit(f"ERREUR : moins de {BATCH_SIZE} exemples, impossible de former un batch.")

    set_seed(SEED)
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.datasets import NoDuplicatesDataLoader

    examples = [
        InputExample(texts=[f"query: {t['anchor']}", f"passage: {t['positive']}", f"passage: {t['negative']}"])
        for t in triplets
    ]
    model = SentenceTransformer(BASE_MODEL, device="cpu")
    # MultipleNegativesRankingLoss prend les autres exemples du batch comme negatifs : un meme positif
    # (jusqu'a 4 ancres par unite) present deux fois dans un batch deviendrait un faux negatif.
    # NoDuplicatesDataLoader interdit tout texte en double dans un batch (melange via `random`, donc seede).
    loader = NoDuplicatesDataLoader(examples, batch_size=BATCH_SIZE)
    loss = losses.MultipleNegativesRankingLoss(model)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = math.ceil(WARMUP_RATIO * total_steps)

    print("\nConfiguration :")
    print(f"  modele de base    : {BASE_MODEL} (cpu)")
    print("  perte             : MultipleNegativesRankingLoss (negatif dur + negatifs du batch)")
    print(f"  chargeur          : NoDuplicatesDataLoader (aucun texte en double par batch)")
    print(f"  epochs            : {EPOCHS}")
    print(f"  batch             : {BATCH_SIZE}")
    print(f"  pas / epoch       : {steps_per_epoch}  (total {total_steps})")
    print(f"  warmup            : {warmup_steps} pas ({WARMUP_RATIO:.0%})")
    print(f"  learning rate     : {LEARNING_RATE}")
    print(f"  graine            : {SEED}")
    print(f"  sortie            : {OUTPUT_DIR.relative_to(ROOT)}")

    start = time.perf_counter()
    # old_fit : API InputExample + DataLoader, sans dependance a `datasets` / `accelerate`
    # (que fit() exige depuis sentence-transformers 3.x et qui ne sont pas installes).
    model.old_fit(
        train_objectives=[(loader, loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LEARNING_RATE},
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - start

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_DIR))
    print(f"\nTemps d'entrainement : {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"Modele sauvegarde : {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

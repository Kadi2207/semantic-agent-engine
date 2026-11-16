"""Comparaison d'architectures : meme protocole exact que scripts/train_embeddings.py, autres modeles de base.

Memes triplets (fusionnes, dedoublonnes), graine 42, 3 epoques, batch 16, lr 2e-5, warmup 0,10, perte
MultipleNegativesRankingLoss, chargeur NoDuplicatesDataLoader. Seules changent la base et ses conventions d'entree :
- e5 (small, base) : prefixes "query: " / "passage: " ;
- paraphrase-multilingual-MiniLM-L12-v2 : aucun prefixe (convention du modele).
Ablation sur e5-small : "sans negatifs difficiles" = paires (ancre, positif) seules, les negatifs sont les autres
exemples du lot.

Le modele de production (models/e5-finetuned) n'est jamais touche : sorties dans models/arch/<config>/.

    uv run python scripts/train_architectures.py e5_small_sans_negatifs_durs minilm_l12 e5_base
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_embeddings import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    TRIPLET_FILES,
    WARMUP_RATIO,
    load_triplets,
    set_seed,
)

ARCH_DIR = ROOT / "models" / "arch"
E5_PREFIXES = ("query: ", "passage: ")
NO_PREFIXES = ("", "")

# config -> (modele de base, prefixes (requete, passage), negatifs difficiles)
CONFIGS: dict[str, tuple[str, tuple[str, str], bool]] = {
    "e5_small_sans_negatifs_durs": ("intfloat/multilingual-e5-small", E5_PREFIXES, False),
    "minilm_l12": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", NO_PREFIXES, True),
    "e5_base": ("intfloat/multilingual-e5-base", E5_PREFIXES, True),
}


def train(config: str) -> dict[str, object]:
    base, (q_prefix, p_prefix), hard_negatives = CONFIGS[config]
    triplets, _ = load_triplets(TRIPLET_FILES)
    set_seed(SEED)
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.datasets import NoDuplicatesDataLoader

    examples = [
        InputExample(
            texts=[f"{q_prefix}{t['anchor']}", f"{p_prefix}{t['positive']}"]
            + ([f"{p_prefix}{t['negative']}"] if hard_negatives else [])
        )
        for t in triplets
    ]
    model = SentenceTransformer(base, device="cpu")
    loader = NoDuplicatesDataLoader(examples, batch_size=BATCH_SIZE)
    loss = losses.MultipleNegativesRankingLoss(model)
    total_steps = len(loader) * EPOCHS
    warmup_steps = math.ceil(WARMUP_RATIO * total_steps)
    print(f"[{config}] {base} : {len(examples)} exemples, {total_steps} pas, negatifs durs={hard_negatives}", flush=True)
    start = time.perf_counter()
    model.old_fit(
        train_objectives=[(loader, loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LEARNING_RATE},
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - start
    out = ARCH_DIR / config
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    log = {
        "config": config,
        "base_model": base,
        "prefixes": {"query": q_prefix, "passage": p_prefix},
        "hard_negatives": hard_negatives,
        "examples": len(examples),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": WARMUP_RATIO,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "seed": SEED,
        "loss": "MultipleNegativesRankingLoss",
        "train_seconds": round(elapsed, 1),
    }
    (out / "training_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{config}] termine en {elapsed / 60:.1f} min -> {out.relative_to(ROOT)}", flush=True)
    return log


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for config in sys.argv[1:] or list(CONFIGS):
        train(config)


if __name__ == "__main__":
    main()

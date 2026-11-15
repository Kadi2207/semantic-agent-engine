"""Compare e5-small de base et e5 fine-tune sur queries_dev (meme procedure que run_baselines.py)."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_baselines import DATA_DIR, METRICS, QUERIES_PATH, TOP_K, TYPES, mean, score_query  # noqa: E402
from src.memory.corpus import load_corpus, load_queries  # noqa: E402
from src.retrieval.dense_baseline import MODEL_NAME, DenseRetriever  # noqa: E402

FINETUNED_PATH = ROOT / "models" / "e5-finetuned"


def evaluate(retriever: DenseRetriever, queries: list[dict[str, object]]) -> dict[str, list[list[float]]]:
    """Scores par requete, regroupes sous 'all' et sous chaque type de piege."""
    scores: dict[str, list[list[float]]] = defaultdict(list)
    for query in queries:
        retrieved = retriever.search(str(query["query"]), TOP_K, str(query["project_id"]))
        row = score_query(retrieved, set(query["relevant_unit_ids"]))  # type: ignore[arg-type]
        scores["all"].append(row)
        scores[str(query["piege"])].append(row)
    return scores


def print_block(label: str, before: list[list[float]], after: list[list[float]]) -> None:
    mean_before, mean_after = mean(before), mean(after)
    print(f"\n{label}  (n = {len(before)})")
    print(f"  {'Metrique':<10}  {'AVANT':>7}  {'APRES':>7}  {'DELTA':>7}")
    for metric, b, a in zip(METRICS, mean_before, mean_after):
        print(f"  {metric:<10}  {b:>7.3f}  {a:>7.3f}  {a - b:>+7.3f}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not (FINETUNED_PATH / "config.json").exists():
        sys.exit(f"ERREUR : modele fine-tune introuvable dans {FINETUNED_PATH}. Lance d'abord scripts/train_embeddings.py.")

    # Meme filtre que run_baselines.py : requetes avec reponse, hors piege sans_reponse.
    queries = [
        query
        for query in load_queries(QUERIES_PATH)
        if query["relevant_unit_ids"] and query["piege"] != "sans_reponse"
    ]
    corpus = load_corpus(DATA_DIR)

    before = evaluate(DenseRetriever(corpus, MODEL_NAME), queries)
    after = evaluate(DenseRetriever(corpus, FINETUNED_PATH), queries)

    print(f"Requetes DEV evaluees : {len(queries)} (sans_reponse exclues)")
    print(f"AVANT = {MODEL_NAME}")
    print(f"APRES = {FINETUNED_PATH.relative_to(ROOT)}")
    print_block("GLOBAL", before["all"], after["all"])
    for piege in TYPES:
        if before[piege]:
            print_block(piege, before[piege], after[piege])


if __name__ == "__main__":
    main()

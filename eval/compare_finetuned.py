"""Compare e5-small de base et e5 fine-tune sur un jeu de requetes (meme procedure que run_baselines.py).

Par defaut queries_dev ; --queries permet de pointer vers un autre fichier (ex. queries_test.json).
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=QUERIES_PATH,
        help="fichier de requetes (defaut : data/eval/queries_dev.json)",
    )
    parser.add_argument("--out", type=Path, help="ecrit aussi les resultats en JSON (section compare_finetuned)")
    parser.add_argument("--expect", nargs="*", default=[], help="valeurs publiees cle=valeur ; aucun ecart tolere avant ecriture")
    args = parser.parse_args()
    queries_path: Path = args.queries if args.queries.is_absolute() else ROOT / args.queries

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not (FINETUNED_PATH / "config.json").exists():
        sys.exit(f"ERREUR : modele fine-tune introuvable dans {FINETUNED_PATH}. Lance d'abord scripts/train_embeddings.py.")

    # Meme filtre que run_baselines.py : requetes avec reponse, hors piege sans_reponse.
    queries = [
        query
        for query in load_queries(queries_path)
        if query["relevant_unit_ids"] and query["piege"] != "sans_reponse"
    ]
    corpus = load_corpus(DATA_DIR)

    before = evaluate(DenseRetriever(corpus, MODEL_NAME), queries)
    after = evaluate(DenseRetriever(corpus, FINETUNED_PATH), queries)

    print(f"Requetes evaluees ({queries_path.name}) : {len(queries)} (sans_reponse exclues)")
    print(f"AVANT = {MODEL_NAME}")
    print(f"APRES = {FINETUNED_PATH.relative_to(ROOT)}")
    print_block("GLOBAL", before["all"], after["all"])
    for piege in TYPES:
        if before[piege]:
            print_block(piege, before[piege], after[piege])

    if args.out:  # ecriture apres les calculs et l'affichage, a partir des memes scores
        from eval import results_io

        results = {
            name: {
                "global": results_io.metric_block(scores["all"], METRICS),
                "per_category": {p: results_io.metric_block(scores[p], METRICS) for p in TYPES if scores[p]},
            }
            for name, scores in (("e5_base", before), ("e5_finetuned", after))
        }
        observed = {f"{s}.{c}.{m}": v for s, r in results.items() for c, b in [("global", r["global"]), *r["per_category"].items()]
                    for m, v in b.items() if m != "n"}
        problems = results_io.check_expected(observed, args.expect)
        if problems:
            sys.exit("ECART avec les valeurs publiees, rien n'est ecrit :\n  " + "\n  ".join(problems))
        is_test = queries_path.name == "queries_test.json"
        all_queries = load_queries(queries_path)
        query_info = results_io.query_set_info(queries_path, all_queries, queries)
        results_io.write_experiment(
            args.out,
            "compare_finetuned",
            {
                "experiment": "e5-small de base contre e5 fine-tune",
                "corpus": f"corpus interne (data/raw), projets {', '.join(query_info['projects'])}",
                "split": "test" if is_test else "dev",
                "script": "eval/compare_finetuned.py",
                "status": (
                    "re-execution d'enregistrement de resultats historiques, aucun reglage effectue ; "
                    "ce test ne sert jamais a regler quoi que ce soit"
                    if is_test
                    else "resultats de developpement (le dev a aussi servi au reglage de la generation des triplets)"
                ),
                "queries": query_info,
                "models": {"e5_base": results_io.model_info(MODEL_NAME), "e5_finetuned": results_io.model_info(FINETUNED_PATH)},
                "params": results_io.retrieval_params(TOP_K, METRICS),
                "results": results,
            },
            "Resultats de recherche sur le corpus interne, enregistres par eval/results_io.py",
        )
        print(f"\nResultats ecrits dans {args.out}")


if __name__ == "__main__":
    main()

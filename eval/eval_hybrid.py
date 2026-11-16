"""Recherche hybride de l'agent (BM25 + e5 fine-tune, fusion RRF k=60) sur un jeu de requetes.

Meme procedure et memes metriques que run_baselines.py et compare_finetuned.py ; le classement evalue est celui
de HybridRetriever.search, la classe utilisee par l'agent (aucune copie de la fusion ici).

    uv run python eval/eval_hybrid.py --out eval/results/retrieval_dev.json
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_baselines import DATA_DIR, METRICS, QUERIES_PATH, TOP_K, TYPES, format_row, mean, score_query  # noqa: E402
from src.agent.retriever import FINETUNED_PATH, RRF_K, HybridRetriever  # noqa: E402
from src.memory.corpus import load_corpus, load_queries  # noqa: E402

SECTION = "hybrid_rrf"
RRF_NOTE = "RRF_K de src/agent/retriever.py (HybridRetriever, la recherche utilisee par l'agent)"
TEST_STATUS = (
    "execution unique d'enregistrement sur le test scelle, aucun reglage effectue ; "
    "ce test ne sert jamais a regler quoi que ce soit"
)
DEV_STATUS = "resultats de developpement (le dev a aussi servi au reglage de la generation des triplets)"


def hybrid_params() -> dict[str, object]:
    """Parametres communs aux scripts de recherche, avec le k de fusion reellement utilise."""
    from eval import results_io

    params = results_io.retrieval_params(TOP_K, METRICS)
    params["rrf_k"] = RRF_K
    params["rrf_k_note"] = RRF_NOTE
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--out", type=Path, help=f"ecrit aussi les resultats en JSON (section {SECTION})")
    args = parser.parse_args()
    queries_path: Path = args.queries if args.queries.is_absolute() else ROOT / args.queries
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    all_queries = load_queries(queries_path)
    queries = [q for q in all_queries if q["relevant_unit_ids"] and q["piege"] != "sans_reponse"]  # meme filtre
    retriever = HybridRetriever(load_corpus(DATA_DIR))

    scores: dict[str, list[list[float]]] = defaultdict(list)
    for query in queries:
        retrieved = [hit.unit.unit_id for hit in retriever.search(str(query["query"]), str(query["project_id"]), k=TOP_K)]
        row = score_query(retrieved, set(query["relevant_unit_ids"]))  # type: ignore[arg-type]
        scores["all"].append(row)
        scores[str(query["piege"])].append(row)

    width = 24
    print(f"Requetes evaluees ({queries_path.name}) : {len(queries)} (sans_reponse exclues) ; RRF k={RRF_K}")
    print(f"{'Hybride RRF':<{width}}  {'n':>3}  " + "  ".join(f"{m:>8}" for m in METRICS))
    print(format_row("global", len(scores["all"]), mean(scores["all"]), width))
    for piege in TYPES:
        if scores[piege]:
            print(format_row(piege, len(scores[piege]), mean(scores[piege]), width))

    if args.out:
        from eval import results_io

        is_test = queries_path.name == "queries_test.json"
        query_info = results_io.query_set_info(queries_path, all_queries, queries)
        results_io.write_experiment(
            args.out,
            SECTION,
            {
                "experiment": "recherche hybride de l'agent : BM25 + e5 fine-tune, fusion RRF",
                "corpus": f"corpus interne (data/raw), projets {', '.join(query_info['projects'])}",
                "split": "test" if is_test else "dev",
                "script": "eval/eval_hybrid.py",
                "status": TEST_STATUS if is_test else DEV_STATUS,
                "queries": query_info,
                "models": {
                    "bm25": {"model": "rank_bm25.BM25Okapi", "training": None},
                    "e5_finetuned": results_io.model_info(FINETUNED_PATH),
                },
                "params": hybrid_params(),
                "results": {
                    SECTION: {
                        "global": results_io.metric_block(scores["all"], METRICS),
                        "per_category": {p: results_io.metric_block(scores[p], METRICS) for p in TYPES if scores[p]},
                    }
                },
            },
            "Resultats de recherche sur le corpus interne, enregistres par eval/results_io.py",
        )
        print(f"\nResultats ecrits dans {args.out}")


if __name__ == "__main__":
    main()

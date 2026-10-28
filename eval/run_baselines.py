from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402
from src.memory.corpus import load_corpus, load_queries  # noqa: E402
from src.retrieval.bm25_baseline import BM25Retriever  # noqa: E402
from src.retrieval.dense_baseline import DenseRetriever  # noqa: E402

QUERIES_PATH = ROOT / "data" / "eval" / "queries_dev.json"
DATA_DIR = ROOT / "data" / "raw"
TYPES = ["decision_active", "reformulation", "responsable_reel", "negation", "engagement"]
METRICS = ["Recall@3", "Recall@5", "MRR", "nDCG@5"]
TOP_K = 5


class Retriever(Protocol):
    def search(self, query: str, k: int, project_id: str) -> list[str]: ...


def score_query(retrieved: list[str], relevant: set[str]) -> list[float]:
    return [
        recall_at_k(retrieved, relevant, 3),
        recall_at_k(retrieved, relevant, 5),
        mrr(retrieved, relevant),
        ndcg_at_k(retrieved, relevant, 5),
    ]


def mean(rows: list[list[float]]) -> list[float]:
    return [sum(column) / len(rows) for column in zip(*rows)]


def format_row(label: str, count: int, values: list[float], width: int) -> str:
    cells = "  ".join(f"{value:>8.3f}" for value in values)
    return f"{label:<{width}}  {count:>3}  {cells}"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    queries = [
        query
        for query in load_queries(QUERIES_PATH)
        if query["relevant_unit_ids"] and query["piege"] != "sans_reponse"
    ]
    corpus = load_corpus(DATA_DIR)
    retrievers: dict[str, Retriever] = {
        "BM25": BM25Retriever(corpus),
        "Dense (e5-small)": DenseRetriever(corpus),
    }

    scores: dict[str, dict[str, list[list[float]]]] = {
        name: defaultdict(list) for name in retrievers
    }
    for name, retriever in retrievers.items():
        for query in queries:
            retrieved = retriever.search(str(query["query"]), TOP_K, str(query["project_id"]))
            row = score_query(retrieved, set(query["relevant_unit_ids"]))
            scores[name]["all"].append(row)
            scores[name][str(query["piege"])].append(row)

    header = "  ".join(f"{metric:>8}" for metric in METRICS)
    width = 24

    print(f"Requetes DEV evaluees : {len(queries)} (sans_reponse exclues)")
    print()
    print("Scores globaux")
    print(f"{'Baseline':<{width}}  {'n':>3}  {header}")
    for name in retrievers:
        rows = scores[name]["all"]
        print(format_row(name, len(rows), mean(rows), width))

    print()
    print("Scores par type de piege")
    print(f"{'Type / baseline':<{width}}  {'n':>3}  {header}")
    for piege in TYPES:
        for name in retrievers:
            rows = scores[name][piege]
            if rows:
                print(format_row(f"{piege} / {name.split()[0]}", len(rows), mean(rows), width))


if __name__ == "__main__":
    main()

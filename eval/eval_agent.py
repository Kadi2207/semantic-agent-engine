"""Reponse finale de l'agent (answer() : recherche hybride, abstention, resolution des relations) sur un jeu de requetes.

Trois mesures, sans aucun reglage :
- final_answer : pour les requetes avec reponse, l'unite finalement affichee (apres resolution) est-elle une
  unite pertinente ? Globale et par categorie. Un conflit non tranche est compte juste si une unite pertinente
  figure parmi les affirmations affichees ; ces cas sont comptes a part.
- sans_reponse : comportement sur les requetes sans reponse du jeu (abstention attendue).
- abstention_probes (dev seulement) : sondes exploratoires de data/eval/probes_abstention.json.

    uv run python eval/eval_agent.py --out eval/results/agent_dev.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import results_io  # noqa: E402
from eval.eval_hybrid import DEV_STATUS, RRF_NOTE, TEST_STATUS  # noqa: E402
from eval.run_baselines import QUERIES_PATH, TYPES  # noqa: E402
from src.agent.answer import ABSTENTION_THRESHOLD, Answer, answer  # noqa: E402
from src.agent.retriever import FINETUNED_PATH, RRF_K  # noqa: E402
from src.memory.corpus import load_queries  # noqa: E402

PROBES_PATH = ROOT / "data" / "eval" / "probes_abstention.json"


def describe(result: Answer) -> dict[str, object]:
    """Ce que l'agent a renvoye, tel quel (aucun jugement ici)."""
    retrieved = result.resolution[0].split(" ")[0] if result.resolution else result.unit_id
    return {
        "abstained": result.abstained,
        "best_dense_score": round(result.best_dense_score, 6),
        "retrieved_unit_id": retrieved,  # meilleure unite de la recherche, avant resolution
        "final_unit_id": result.unit_id,
        "final_unit_status": result.unit_status,
        "resolved": bool(result.resolution),
        "conflict_with": list(result.conflict_with),
        "statement": result.statement,
    }


def final_answer_row(query: dict[str, object]) -> dict[str, object]:
    relevant = set(query["relevant_unit_ids"])  # type: ignore[arg-type]
    row = describe(answer(str(query["query"]), str(query["project_id"])))
    shown = {row["final_unit_id"], *row["conflict_with"]} - {None}  # type: ignore[misc]
    row["correct"] = not row["abstained"] and bool(shown & relevant)
    row["retrieved_was_relevant"] = row["retrieved_unit_id"] in relevant
    return {"query_id": query["query_id"], "project_id": query["project_id"], "piege": query["piege"],
            "relevant_unit_ids": sorted(relevant), **row}


def accuracy_block(rows: list[dict[str, object]]) -> dict[str, object]:
    correct = sum(bool(r["correct"]) for r in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "abstained": sum(bool(r["abstained"]) for r in rows),
        "resolved": sum(bool(r["resolved"]) for r in rows),
        "correct_thanks_to_resolution": sum(bool(r["correct"]) and not r["retrieved_was_relevant"] for r in rows),
        "unresolved_conflict": sum(bool(r["conflict_with"]) for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    queries_path: Path = args.queries if args.queries.is_absolute() else ROOT / args.queries
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    is_test = queries_path.name == "queries_test.json"

    all_queries = load_queries(queries_path)
    answerable = [q for q in all_queries if q["relevant_unit_ids"] and q["piege"] != "sans_reponse"]
    unanswerable = [q for q in all_queries if q not in answerable]

    rows = [final_answer_row(q) for q in answerable]
    final = {
        "global": accuracy_block(rows),
        "per_category": {p: accuracy_block([r for r in rows if r["piege"] == p]) for p in TYPES if any(r["piege"] == p for r in rows)},
        "per_query": rows,
    }
    no_answer_rows = [
        {"query_id": q["query_id"], "project_id": q["project_id"], **describe(answer(str(q["query"]), str(q["project_id"])))}
        for q in unanswerable
    ]
    no_answer = {
        "n": len(no_answer_rows),
        "expected": "abstention",
        "abstained": sum(bool(r["abstained"]) for r in no_answer_rows),
        "answered": sum(not r["abstained"] for r in no_answer_rows),
        "per_query": no_answer_rows,
    }

    status = TEST_STATUS if is_test else DEV_STATUS
    experiments: dict[str, object] = {
        "final_answer": {
            "experiment": "justesse de la reponse finale de l'agent, apres abstention et resolution des relations",
            "definition": "juste = l'agent ne s'abstient pas et l'unite affichee apres resolution est une unite pertinente ; "
            "pour un conflit non tranche, une unite pertinente figure parmi les affirmations affichees",
            "results": final,
            "sans_reponse": no_answer,
        }
    }
    if not is_test:
        spec = json.loads(PROBES_PATH.read_text(encoding="utf-8"))
        probe_rows = []
        for probe in spec["probes"]:
            row = describe(answer(probe["query"], probe["project_id"]))
            row["wrongly_answered"] = not row["abstained"] and row["final_unit_id"] not in probe["acceptable_unit_ids"]
            probe_rows.append({"probe_id": probe["probe_id"], "project_id": probe["project_id"], "query": probe["query"], **row})
        experiments["abstention_probes"] = {
            "experiment": "sondes d'abstention : sujet present dans le projet, reponse absente des sources",
            "status": spec["status"],
            "file": PROBES_PATH.relative_to(ROOT).as_posix(),
            "sha256": results_io.sha256_text_file(PROBES_PATH),
            "rule": spec["rule"],
            "results": {
                "n": len(probe_rows),
                "abstained": sum(bool(r["abstained"]) for r in probe_rows),
                "wrongly_answered": sum(bool(r["wrongly_answered"]) for r in probe_rows),
                "per_probe": probe_rows,
            },
        }

    data = {
        "description": "Reponse finale de l'agent sur le corpus interne, enregistree par eval/eval_agent.py",
        "generated_on": date.today().isoformat(),
        "split": "test" if is_test else "dev",
        "script": "eval/eval_agent.py",
        "status": status,
        "queries": results_io.query_set_info(queries_path, all_queries, answerable),
        "models": {"bm25": {"model": "rank_bm25.BM25Okapi", "training": None}, "e5_finetuned": results_io.model_info(FINETUNED_PATH)},
        "params": {
            "abstention_threshold": ABSTENTION_THRESHOLD,
            "abstention_threshold_note": "ABSTENTION_THRESHOLD de src/agent/answer.py, compare au meilleur cosinus e5 du projet",
            "rrf_k": RRF_K,
            "rrf_k_note": RRF_NOTE,
            "search_scope": "unites du projet de la requete",
        },
        "experiments": experiments,
    }
    out: Path = args.out if args.out.is_absolute() else ROOT / args.out
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    g = final["global"]
    print(f"Reponse finale ({queries_path.name}) : {g['correct']}/{g['n']} justes ({g['accuracy']:.3f}), "
          f"{g['abstained']} abstentions, {g['resolved']} resolutions, {g['unresolved_conflict']} conflit(s) non tranche(s)")
    for piege, block in final["per_category"].items():
        print(f"  {piege:<18} {block['correct']}/{block['n']}")
    for r in rows:
        if not r["correct"]:
            print(f"  FAUX {r['query_id']} ({r['piege']}) : renvoie {r['final_unit_id']}, attendu {r['relevant_unit_ids']}")
    print(f"Sans reponse : {no_answer['abstained']}/{no_answer['n']} abstentions")
    if not is_test:
        p = experiments["abstention_probes"]["results"]  # type: ignore[index]
        print(f"Sondes : {p['wrongly_answered']}/{p['n']} reponses a tort, {p['abstained']} abstention(s)")
        for r in p["per_probe"]:
            print(f"  {r['probe_id']} cos {r['best_dense_score']:.3f} -> {'ABSTENTION' if r['abstained'] else r['final_unit_id']} : {r['statement']}")
    print(f"Resultats ecrits dans {out}")


if __name__ == "__main__":
    main()

"""Evaluation du RAG controle (src/agent/rag.py) sur un jeu de requetes, avec criteres d'acceptation fixes.

Les criteres sont fixes dans ce fichier AVANT toute execution (commit separe) et recopies dans le JSON :
- C1 : au moins 23 des 25 requetes avec reponse du dev restent repondues avec une unite attendue parmi les
  citations affichees (synthese acceptee : identifiants cites ; sinon, repli extractif : unite affichee) ;
- C2 : au moins 4 des 5 requetes sans_reponse du dev deviennent des abstentions ;
- C3 : 100 % des citations des syntheses affichees sont des identifiants fournis au LLM ;
- C4 : parcours Atlas identique en preuve extractive (answer() inchange, verifie par
  tests/test_interface_unchanged_answers.py et scripts/verify_demo.py).
Les sondes d'abstention sont mesurees seulement. Si C1 a C4 sont atteints, le mode est active par defaut dans
l'interface ; sinon il reste desactive et le resultat negatif est documente.

    uv run python eval/eval_rag.py --out eval/results/rag_dev.json
    uv run python eval/eval_rag.py --queries data/eval/queries_test.json --out eval/results/rag_test.json
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
from eval.eval_agent import PROBES_PATH  # noqa: E402
from eval.eval_hybrid import DEV_STATUS, TEST_STATUS  # noqa: E402
from eval.run_baselines import QUERIES_PATH, TYPES  # noqa: E402
from src.agent.rag import (  # noqa: E402
    RAG_EXTRA,
    RAG_K,
    RAG_MAX_COMPLETION_TOKENS,
    RAG_MODEL,
    RAG_SYSTEM_PROMPT,
    RagResult,
    get_rag_client,
    rag_answer,
)
from src.memory.corpus import load_queries  # noqa: E402

CRITERIA = {
    "C1_answered_with_expected_unit": {"min": 23, "of": 25, "text": "au moins 23 des 25 requetes avec reponse restent repondues avec une unite attendue parmi les citations"},
    "C2_unanswerable_abstained": {"min": 4, "of": 5, "text": "au moins 4 des 5 requetes sans_reponse deviennent des abstentions"},
    "C3_citations_valid": {"min_rate": 1.0, "text": "100 % des citations des syntheses affichees sont des identifiants fournis au LLM"},
    "C4_atlas_identical": {"text": "parcours Atlas identique en preuve extractive (answer() inchange)"},
}


def shown_ids(r: RagResult) -> list[str]:
    """Unites presentees comme reponse : citations de la synthese, sinon unite(s) de la reponse extractive."""
    if r.abstained:
        return []
    if r.status == "synthese":
        return list(r.cited_ids)
    return [u for u in (r.base.unit_id, *r.base.conflict_with) if u]


def row(query: str, project_id: str, r: RagResult) -> dict[str, object]:
    return {
        "project_id": project_id,
        "query": query,
        "status": r.status,
        "abstained": r.abstained,
        "best_dense_score": round(r.base.best_dense_score, 6),
        "extractive_unit_id": r.base.unit_id,
        "provided_ids": list(r.provided_ids),
        "cited_ids": list(r.cited_ids),
        "shown_ids": shown_ids(r),
        "synthesis": r.synthesis or None,
        "rejection": r.rejection,
        "citations_valid": all(i in r.provided_ids for i in r.cited_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-only", action="store_true", help="aucun appel Groq : relit le cache seulement")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    queries_path: Path = args.queries if args.queries.is_absolute() else ROOT / args.queries
    is_test = queries_path.name == "queries_test.json"
    client = get_rag_client(cache_only=args.cache_only)

    all_queries = load_queries(queries_path)
    rows = []
    for q in all_queries:
        r = rag_answer(str(q["query"]), str(q["project_id"]), client=client)
        x = {"query_id": q["query_id"], "piege": q["piege"], "relevant_unit_ids": q["relevant_unit_ids"], **row(str(q["query"]), str(q["project_id"]), r)}
        answerable = bool(q["relevant_unit_ids"]) and q["piege"] != "sans_reponse"
        x["correct"] = (not x["abstained"] and bool(set(x["shown_ids"]) & set(q["relevant_unit_ids"]))) if answerable else bool(x["abstained"])
        rows.append(x)
        print(f"{x['query_id']} {x['piege']:<16} {x['status']:<15} juste={x['correct']} cite={x['cited_ids']} attendu={q['relevant_unit_ids']}"
              + (f" rejet={x['rejection']}" if x["rejection"] else ""), flush=True)

    answer_rows = [x for x in rows if x["relevant_unit_ids"] and x["piege"] != "sans_reponse"]
    none_rows = [x for x in rows if x not in answer_rows]
    syntheses = [x for x in rows if x["status"] == "synthese"]
    results: dict[str, object] = {
        "answerable": {
            "n": len(answer_rows),
            "correct": sum(x["correct"] for x in answer_rows),
            "per_category": {p: {"n": len(s), "correct": sum(x["correct"] for x in s)}
                             for p in TYPES if (s := [x for x in answer_rows if x["piege"] == p])},
            "abstained": sum(x["abstained"] for x in answer_rows),
        },
        "sans_reponse": {"n": len(none_rows), "abstained": sum(x["abstained"] for x in none_rows)},
        "statuses": {s: sum(x["status"] == s for x in rows) for s in sorted({x["status"] for x in rows})},
        "syntheses": len(syntheses),
        "citations": sum(len(x["cited_ids"]) for x in syntheses),
        "citations_valid": sum(sum(i in x["provided_ids"] for i in x["cited_ids"]) for x in syntheses),
        "rejections": [{"query_id": x["query_id"], "reason": x["rejection"]} for x in rows if x["status"] == "rejetee"],
        "per_query": rows,
    }
    if not is_test:
        spec = json.loads(PROBES_PATH.read_text(encoding="utf-8"))
        probe_rows = []
        for p in spec["probes"]:
            r = rag_answer(p["query"], p["project_id"], client=client)
            x = {"probe_id": p["probe_id"], **row(p["query"], p["project_id"], r)}
            x["wrongly_answered"] = not x["abstained"] and not set(x["shown_ids"]) & set(p["acceptable_unit_ids"])
            probe_rows.append(x)
        results["abstention_probes"] = {
            "status": spec["status"],
            "n": len(probe_rows),
            "abstained": sum(x["abstained"] for x in probe_rows),
            "wrongly_answered": sum(x["wrongly_answered"] for x in probe_rows),
            "per_probe": probe_rows,
        }
        a, s = results["answerable"], results["sans_reponse"]
        rate = results["citations_valid"] / results["citations"] if results["citations"] else 1.0
        met = {
            "C1_answered_with_expected_unit": a["correct"] >= 23,  # type: ignore[index]
            "C2_unanswerable_abstained": s["abstained"] >= 4,  # type: ignore[index]
            "C3_citations_valid": rate >= 1.0,
            "C4_atlas_identical": None,  # verifie par les tests et verify_demo, rapporte a part
        }
        results["criteria_met"] = met
        results["citations_valid_rate"] = rate

    snap = client.snapshot()
    data = {
        "description": "RAG controle : synthese LLM a partir des unites retrouvees, citations verifiees par le code",
        "generated_on": date.today().isoformat(),
        "split": "test" if is_test else "dev",
        "script": "eval/eval_rag.py",
        "status": TEST_STATUS if is_test else DEV_STATUS + " ; criteres fixes avant l'execution, aucun reglage du prompt apres resultats",
        "criteria": CRITERIA,
        "queries": {"file": queries_path.relative_to(ROOT).as_posix(), "sha256": results_io.sha256_text_file(queries_path), "total": len(all_queries)},
        "llm": {
            "model": RAG_MODEL, "temperature": 0.0, "json_mode": True, "k_units": RAG_K,
            "max_completion_tokens": RAG_MAX_COMPLETION_TOKENS, "extra": RAG_EXTRA,
            "system_prompt_sha256": results_io.sha256_params({"prompt": RAG_SYSTEM_PROMPT}),
        },
        "groq_usage_this_run": snap,
        "results": results,
    }
    out: Path = args.out if args.out.is_absolute() else ROOT / args.out
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    a, s = results["answerable"], results["sans_reponse"]
    print(f"avec reponse : {a['correct']}/{a['n']} ; sans reponse : {s['abstained']}/{s['n']} abstentions ; statuts {results['statuses']}")  # type: ignore[index]
    if not is_test:
        print(f"sondes : {results['abstention_probes']['wrongly_answered']}/6 a tort ; criteres : {results['criteria_met']}")  # type: ignore[index]
    print(f"Groq : {snap}")
    print(f"Resultats ecrits dans {out}")


if __name__ == "__main__":
    main()

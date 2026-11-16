"""Chaine complete : reponse finale de l'agent sur la memoire EXTRAITE, comparee a la memoire annotee.

Aucun reglage. Requetes : celles du dev qui portent sur les projets 04 a 10 (les projets 01 a 03 ont servi a regler
la consigne d'extraction ; les projets 11 et 12, test scelle, ne servent a aucune mesure ici).
- memoire extraite (data/extracted) : meme agent (answer), autre memoire. Une reponse est jugee juste si l'unite
  affichee est appariee (Matcher de eval/eval_extraction.py : recouvrement de preuve, puis cosinus e5) a une
  unite attendue. Pour un conflit non tranche, il suffit qu'une des unites affichees le soit.
- memoire annotee : memes requetes, meme regle que eval/eval_agent.py.

    uv run python eval/eval_agent_extracted.py --out eval/results/agent_extracted_dev.json
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
from eval.eval_agent import accuracy_block, describe, final_answer_row, resolution_breakdown  # noqa: E402
from eval.eval_extraction import COS_THRESHOLD, MIN_OVERLAP, Matcher  # noqa: E402
from eval.run_baselines import QUERIES_PATH, TYPES  # noqa: E402
from src.agent.answer import ABSTENTION_THRESHOLD, answer  # noqa: E402
from src.agent.retriever import FINETUNED_PATH, RRF_K  # noqa: E402
from src.memory.corpus import load_corpus, load_queries, load_relations  # noqa: E402
from src.memory.extracted import EXTRACTED_DIR, MENTION, get_extracted_graph, get_extracted_retriever  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
DEV_PROJECTS = [f"projet_{n:02d}" for n in range(4, 11)]


def unit_mapping(matcher: Matcher) -> dict[str, str]:
    """unite extraite -> unite annotee appariee, pour les projets de developpement 04 a 10."""
    mapping: dict[str, str] = {}
    gold_all, pred_all = load_corpus(RAW_DIR), load_corpus(EXTRACTED_DIR)
    for path in sorted(RAW_DIR.glob("projet_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["project_id"] not in DEV_PROJECTS:
            continue
        gold = [c.unit for c in gold_all if c.project_id == data["project_id"]]
        pred = [c.unit for c in pred_all if c.project_id == data["project_id"]]
        transcripts = {str(m["source_id"]): str(m["transcript"]) for m in data["meetings"]}
        mapping.update(matcher.match(gold, pred, transcripts)[0])
    return mapping


def extracted_row(query: dict[str, object], mapping: dict[str, str]) -> dict[str, object]:
    relevant = set(query["relevant_unit_ids"])  # type: ignore[arg-type]
    result = answer(str(query["query"]), str(query["project_id"]), retriever=get_extracted_retriever(), graph=get_extracted_graph())
    row = describe(result)
    shown = [u for u in (row["final_unit_id"], *row["conflict_with"]) if u]  # type: ignore[misc]
    matched = [mapping.get(u) for u in shown]
    row["matched_annotated_unit_ids"] = matched  # None : unite extraite sans equivalent annote
    row["correct"] = not row["abstained"] and bool(set(matched) & relevant)
    row["retrieved_was_relevant"] = mapping.get(str(row["retrieved_unit_id"])) in relevant
    return {"query_id": query["query_id"], "project_id": query["project_id"], "piege": query["piege"],
            "relevant_unit_ids": sorted(relevant), **row}


def block(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "global": accuracy_block(rows),
        "per_category": {p: accuracy_block([r for r in rows if r["piege"] == p]) for p in TYPES if any(r["piege"] == p for r in rows)},
        "resolution_breakdown": resolution_breakdown(rows),
        "per_query": rows,
    }


def no_answer_block(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"n": len(rows), "expected": "abstention", "abstained": sum(bool(r["abstained"]) for r in rows),
            "answered": sum(not r["abstained"] for r in rows), "per_query": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    all_queries = load_queries(QUERIES_PATH)
    in_scope = [q for q in all_queries if q["project_id"] in DEV_PROJECTS]
    answerable = [q for q in in_scope if q["relevant_unit_ids"] and q["piege"] != "sans_reponse"]
    unanswerable = [q for q in in_scope if q not in answerable]

    mapping = unit_mapping(Matcher())
    extracted_rows = [extracted_row(q, mapping) for q in answerable]
    annotated_rows = [final_answer_row(q) for q in answerable]
    extracted_units = [c for c in load_corpus(EXTRACTED_DIR) if c.project_id in DEV_PROJECTS]
    extracted_ids = {c.unit.unit_id for c in extracted_units}
    relations = [r for r in load_relations(EXTRACTED_DIR) if r.from_id in extracted_ids]
    matched_gold = set(mapping.values())
    extracted, annotated = block(extracted_rows), block(annotated_rows)
    both = sum(bool(e["correct"]) and bool(a["correct"]) for e, a in zip(extracted_rows, annotated_rows))

    data = {
        "description": "Chaine complete : reponse finale de l'agent sur la memoire extraite, comparee a la memoire annotee",
        "generated_on": date.today().isoformat(),
        "split": "dev",
        "script": "eval/eval_agent_extracted.py",
        "status": "mesure sans aucun reglage, projets de developpement 04 a 10 ; les projets 11 et 12 (test scelle) ne sont pas utilises",
        "queries": {
            "file": QUERIES_PATH.relative_to(ROOT).as_posix(),
            "sha256": results_io.sha256_text_file(QUERIES_PATH),
            "total_in_file": len(all_queries),
            "projects": DEV_PROJECTS,
            "in_scope": len(in_scope),
            "evaluated": len(answerable),
            "evaluated_query_ids": [q["query_id"] for q in answerable],
            "sans_reponse_query_ids": [q["query_id"] for q in unanswerable],
            "excluded": "requetes des projets 01 a 03 (projets ayant servi a regler la consigne d'extraction)",
        },
        "extracted_memory": {
            "directory": EXTRACTED_DIR.relative_to(ROOT).as_posix(),
            "mention": MENTION,
            "built_by": "scripts/build_extracted_memory.py (rejeu du cache Groq local, aucun appel)",
            "units": len(extracted_units),
            "relations": len(relations),
            "units_matched_to_annotation": len(mapping),
            "expected_units_with_extracted_match": sum(u in matched_gold for q in answerable for u in q["relevant_unit_ids"]),  # type: ignore[union-attr]
            "expected_units": sum(len(q["relevant_unit_ids"]) for q in answerable),  # type: ignore[arg-type]
        },
        "matching": {
            "method": "Matcher de eval/eval_extraction.py : recouvrement de preuve dans le transcript, a defaut cosinus e5 des enonces",
            "min_overlap": MIN_OVERLAP,
            "cos_threshold": COS_THRESHOLD,
        },
        "models": {"bm25": {"model": "rank_bm25.BM25Okapi", "training": None}, "e5_finetuned": results_io.model_info(FINETUNED_PATH)},
        "params": {"abstention_threshold": ABSTENTION_THRESHOLD, "rrf_k": RRF_K, "search_scope": "unites du projet de la requete"},
        "definition": "juste = l'agent ne s'abstient pas et l'unite affichee est (memoire annotee) une unite attendue, ou (memoire "
        "extraite) appariee a une unite attendue ; pour un conflit non tranche, une des unites affichees suffit",
        "results": {
            "extracted": extracted,
            "annotated": annotated,
            "comparison": {
                "n": len(answerable),
                "correct_extracted": extracted["global"]["correct"],  # type: ignore[index]
                "correct_annotated": annotated["global"]["correct"],  # type: ignore[index]
                "correct_in_both": both,
                "correct_only_annotated": [a["query_id"] for e, a in zip(extracted_rows, annotated_rows) if a["correct"] and not e["correct"]],
                "correct_only_extracted": [a["query_id"] for e, a in zip(extracted_rows, annotated_rows) if e["correct"] and not a["correct"]],
            },
        },
        "sans_reponse": {
            "extracted": no_answer_block([
                {"query_id": q["query_id"], "project_id": q["project_id"],
                 **describe(answer(str(q["query"]), str(q["project_id"]), retriever=get_extracted_retriever(), graph=get_extracted_graph()))}
                for q in unanswerable
            ]),
            "annotated": no_answer_block([
                {"query_id": q["query_id"], "project_id": q["project_id"], **describe(answer(str(q["query"]), str(q["project_id"])))}
                for q in unanswerable
            ]),
        },
    }
    out: Path = args.out if args.out.is_absolute() else ROOT / args.out
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, b in (("extraite", extracted), ("annotee", annotated)):
        g = b["global"]
        print(f"memoire {name:<8} : {g['correct']}/{g['n']} justes ({g['accuracy']:.3f}), {g['abstained']} abstention(s), "  # type: ignore[index]
              f"{g['resolved']} resolution(s) ; " + ", ".join(f"{p} {v['correct']}/{v['n']}" for p, v in b["per_category"].items()))  # type: ignore[index, union-attr]
    for e, a in zip(extracted_rows, annotated_rows):
        print(f"  {e['query_id']} {e['piege']:<17} extraite {'OK ' if e['correct'] else 'non'} ({e['final_unit_id']} -> {e['matched_annotated_unit_ids']})"
              f" | annotee {'OK ' if a['correct'] else 'non'} ({a['final_unit_id']}) | attendu {e['relevant_unit_ids']}")
    s = data["sans_reponse"]
    print(f"sans reponse : abstentions {s['extracted']['abstained']}/{s['extracted']['n']} (extraite), {s['annotated']['abstained']}/{s['annotated']['n']} (annotee)")  # type: ignore[index]
    print(f"Resultats ecrits dans {out}")


if __name__ == "__main__":
    main()

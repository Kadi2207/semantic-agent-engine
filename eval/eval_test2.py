"""Test scelle elargi (projets 13 a 16, data/test2) : une seule execution d'enregistrement par mesure, sans reglage.

Le jeu (data/test2, data/eval/queries_test2.json) a ete gele avant toute execution (data/eval/test2_manifest.json) ;
le script refuse de tourner si une empreinte a change. Configuration actuelle, modele fine-tune NON reentraine.

Etapes (chacune ecrit sa section de eval/results/test2.json et refuse d'etre relancee, sauf --cache-only qui rejoue
depuis le cache sans reseau pour verifier) :
- offline : recherche (BM25, e5 de base, e5 fine-tune, hybride RRF) sur les requetes avec reponse (simples et
  complexes) ; reponse finale extractive (20 requetes simples) ; abstention (8 sans reponse) ; questions complexes
  (part des unites attendues affichees) ; dossiers (rappel des unites attendues, aucune unite sans source) ;
- rag : RAG controle sur les memes requetes (cache separe data/processed/groq_rag_test2_cache.jsonl) ;
- extraction : extraction figee sur les 4 projets (cache d'extraction habituel), metriques des unites et relations ;
- relations : seconde passe de liaison, configuration retenue sur le dev 04-10 (relations_v2.json), sur la sortie
  de l'extraction.

Quota Groq : limite journaliere de 200 000 tokens. QUOTA_BASELINE_TODAY = tokens deja consommes le jour de
l'execution avant ce script (caches RAG et liaison ecrits ce jour-la) ; s'y ajoutent les couts enregistres par les
etapes precedentes. Une etape n'est lancee que si son estimation (avec marge) tient dans le reste.

    uv run python eval/eval_test2.py --stage offline
    uv run python eval/eval_test2.py --stage rag
    uv run python eval/eval_test2.py --stage extraction
    uv run python eval/eval_test2.py --stage relations
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import results_io  # noqa: E402
from eval.run_baselines import METRICS, TOP_K, score_query  # noqa: E402
from src.memory.corpus import load_corpus, load_participants, load_queries, load_relations  # noqa: E402

TEST2_DIR = ROOT / "data" / "test2"
QUERIES_PATH = ROOT / "data" / "eval" / "queries_test2.json"
MANIFEST = ROOT / "data" / "eval" / "test2_manifest.json"
OUT_PATH = ROOT / "eval" / "results" / "test2.json"
RAG_TEST2_CACHE = ROOT / "data" / "processed" / "groq_rag_test2_cache.jsonl"
SIMPLE = ["decision_active", "reformulation", "responsable_reel", "negation", "engagement"]
QUOTA_LIMIT = 200_000
QUOTA_BASELINE_TODAY = 97_283  # 2026-11-16 : cache RAG (25 895) + cache de liaison (71 388), ecrits ce jour-la
ESTIMATES = {"rag": 36 * 900, "extraction": 12 * 6_000, "relations": 4 * 5_000}  # tokens, marge comprise
STATUS = "execution unique d'enregistrement sur le test scelle elargi, configuration actuelle, aucun reglage"


def check_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changed = [p for p, sha in manifest["files"].items() if results_io.sha256_text_file(ROOT / p) != sha]
    if changed:
        sys.exit(f"ECART : fichiers du test scelle elargi modifies depuis le gel : {changed}")


def queries() -> list[dict[str, object]]:
    return load_queries(QUERIES_PATH)


def project_names() -> dict[str, str]:
    return {json.loads(p.read_text(encoding="utf-8"))["project_id"]: json.loads(p.read_text(encoding="utf-8"))["project_name"]
            for p in sorted(TEST2_DIR.glob("projet_*.json"))}


def mean_block(rows: list[list[float]]) -> dict[str, object]:
    return results_io.metric_block(rows, METRICS)


# --------------------------------------------------------------------------- hors ligne


def stage_offline() -> dict[str, object]:
    from src.agent.answer import ReplacementGraph, answer
    from src.agent.dossier import build_dossier
    from src.agent.retriever import FINETUNED_PATH, HybridRetriever
    from src.retrieval.bm25_baseline import BM25Retriever
    from src.retrieval.dense_baseline import MODEL_NAME, DenseRetriever
    from eval.eval_agent import describe

    corpus = load_corpus(TEST2_DIR)
    relations = load_relations(TEST2_DIR)
    hybrid = HybridRetriever(corpus)
    graph = ReplacementGraph(relations)
    qs = queries()

    # recherche
    searchable = [q for q in qs if q["piege"] in SIMPLE + ["complexe"]]
    systems = {
        "e5_base": DenseRetriever(corpus, MODEL_NAME),
        "e5_finetuned": DenseRetriever(corpus, FINETUNED_PATH),
    }
    bm25 = BM25Retriever(corpus)
    retrieval: dict[str, object] = {}
    per_query_rows: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for name in ("bm25", "e5_base", "e5_finetuned", "hybrid_rrf"):
        rows: dict[str, list[list[float]]] = defaultdict(list)
        for q in searchable:
            if name == "bm25":
                ids = bm25.search(str(q["query"]), TOP_K, str(q["project_id"]))
            elif name == "hybrid_rrf":
                ids = [h.unit.unit_id for h in hybrid.search(str(q["query"]), str(q["project_id"]), k=TOP_K)]
            else:
                ids = systems[name].search(str(q["query"]), TOP_K, str(q["project_id"]))  # type: ignore[union-attr]
            row = score_query(ids, set(q["relevant_unit_ids"]))  # type: ignore[arg-type]
            rows["all"].append(row)
            if q["piege"] in SIMPLE:
                rows["simples"].append(row)
                rows[str(q["piege"])].append(row)
            else:
                rows["complexe"].append(row)
            per_query_rows[name][str(q["query_id"])] = row
        retrieval[name] = {"global": mean_block(rows["all"]), "simples": mean_block(rows["simples"]),
                           "complexe": mean_block(rows["complexe"]),
                           "per_category": {p: mean_block(rows[p]) for p in SIMPLE if rows[p]}}
    retrieval["per_query"] = per_query_rows

    # reponse finale extractive, abstention, questions complexes
    def run(q: dict[str, object]) -> dict[str, object]:
        row = describe(answer(str(q["query"]), str(q["project_id"]), retriever=hybrid, graph=graph))
        shown = [u for u in (row["final_unit_id"], *row["conflict_with"]) if u]  # type: ignore[misc]
        relevant = set(q["relevant_unit_ids"])  # type: ignore[arg-type]
        row["shown_ids"] = shown
        row["correct"] = not row["abstained"] and bool(set(shown) & relevant)
        row["expected_shown"] = len(set(shown) & relevant)
        return {"query_id": q["query_id"], "project_id": q["project_id"], "piege": q["piege"], "relevant_unit_ids": sorted(relevant), **row}

    simple_rows = [run(q) for q in qs if q["piege"] in SIMPLE]
    none_rows = [run(q) for q in qs if q["piege"] == "sans_reponse"]
    complex_rows = [run(q) for q in qs if q["piege"] == "complexe"]

    def acc(rows: list[dict[str, object]]) -> dict[str, object]:
        return {"n": len(rows), "correct": sum(bool(r["correct"]) for r in rows), "abstained": sum(bool(r["abstained"]) for r in rows),
                "resolved": sum(bool(r["resolved"]) for r in rows), "unresolved_conflict": sum(bool(r["conflict_with"]) for r in rows)}

    expected_total = sum(len(r["relevant_unit_ids"]) for r in complex_rows)  # type: ignore[arg-type]
    complex_block = {
        "n": len(complex_rows), "expected_units": expected_total,
        "expected_units_shown": sum(int(r["expected_shown"]) for r in complex_rows),
        "share_expected_shown": round(sum(int(r["expected_shown"]) for r in complex_rows) / expected_total, 4),
        "queries_with_all_expected_shown": sum(int(r["expected_shown"]) == len(r["relevant_unit_ids"]) for r in complex_rows),  # type: ignore[arg-type]
        "definition": "reponse extractive : unite affichee (et unites en conflit) ; part = unites attendues affichees / unites attendues",
        "per_query": complex_rows,
    }

    # dossiers
    names = project_names()
    participants = load_participants(TEST2_DIR)
    transcripts = {m["source_id"]: m["transcript"] for p in sorted(TEST2_DIR.glob("projet_*.json"))
                   for m in json.loads(p.read_text(encoding="utf-8"))["meetings"]}
    dossier_rows = []
    for q in (q for q in qs if q["piege"] == "dossier"):
        d = build_dossier(str(q["query"]), str(q["project_id"]), names[str(q["project_id"])], retriever=hybrid,
                          relations=relations, participants=participants)
        cited = d.cited_ids()
        expected = set(q["relevant_unit_ids"])  # type: ignore[arg-type]
        without_source = [u.unit_id for u in d.units if u.source_id not in transcripts or u.evidence not in transcripts[u.source_id]]
        dossier_rows.append({
            "query_id": q["query_id"], "project_id": q["project_id"], "query": q["query"], "scope": d.scope, "abstained": d.abstained,
            "expected": sorted(expected), "cited": sorted(cited), "expected_cited": len(cited & expected),
            "recall": round(len(cited & expected) / len(expected), 4), "cited_not_expected": sorted(cited - expected),
            "units_without_source": without_source,
        })
    dossiers = {
        "n": len(dossier_rows),
        "expected_units": sum(len(r["expected"]) for r in dossier_rows),
        "expected_units_cited": sum(r["expected_cited"] for r in dossier_rows),
        "micro_recall": round(sum(r["expected_cited"] for r in dossier_rows) / sum(len(r["expected"]) for r in dossier_rows), 4),
        "units_without_source": sum(len(r["units_without_source"]) for r in dossier_rows),
        "definition": "unites attendues = rubriques documentees du dossier (decisions en vigueur, historique, conflits, actions et "
        "engagements, risques, questions ouvertes), fixees avant toute execution ; source = preuve retrouvee dans le transcript",
        "per_query": dossier_rows,
    }
    return {
        "status": STATUS,
        "retrieval": {"queries": len(searchable), "top_k": TOP_K, "metrics": METRICS, "results": retrieval},
        "final_answer": {"global": acc(simple_rows), "per_category": {p: acc([r for r in simple_rows if r["piege"] == p]) for p in SIMPLE},
                         "per_query": simple_rows},
        "abstention": {"n": len(none_rows), "abstained": sum(bool(r["abstained"]) for r in none_rows), "per_query": none_rows},
        "complex": complex_block,
        "dossiers": dossiers,
        "groq_tokens": 0,
    }


# --------------------------------------------------------------------------- RAG


def stage_rag(cache_only: bool) -> dict[str, object]:
    from eval.eval_rag import row as rag_row
    from src.agent import rag as rag_module
    from src.agent.answer import ReplacementGraph
    from src.agent.retriever import HybridRetriever
    from src.ingest.groq_client import CachedGroq

    client = CachedGroq(model=rag_module.RAG_MODEL, temperature=0.0, cache_path=RAG_TEST2_CACHE,
                        min_seconds_between_calls=rag_module.SECONDS_BETWEEN_CALLS, json_mode=True, cache_only=cache_only,
                        max_completion_tokens=rag_module.RAG_MAX_COMPLETION_TOKENS, extra_body=rag_module.RAG_EXTRA)
    hybrid = HybridRetriever(load_corpus(TEST2_DIR))
    graph = ReplacementGraph(load_relations(TEST2_DIR))
    rows = []
    for q in queries():
        if q["piege"] == "dossier":
            continue
        r = rag_module.rag_answer(str(q["query"]), str(q["project_id"]), retriever=hybrid, graph=graph, client=client)
        row = {"query_id": q["query_id"], "piege": q["piege"], "relevant_unit_ids": q["relevant_unit_ids"], **rag_row(str(q["query"]), str(q["project_id"]), r)}
        relevant = set(q["relevant_unit_ids"])  # type: ignore[arg-type]
        row["correct"] = not row["abstained"] and bool(set(row["shown_ids"]) & relevant)
        row["expected_shown"] = len(set(row["shown_ids"]) & relevant)
        rows.append(row)
    simple = [r for r in rows if r["piege"] in SIMPLE]
    none = [r for r in rows if r["piege"] == "sans_reponse"]
    cplx = [r for r in rows if r["piege"] == "complexe"]
    expected_total = sum(len(r["relevant_unit_ids"]) for r in cplx)
    cited = [i for r in rows if r["status"] == "synthese" for i in r["cited_ids"]]
    snap = client.snapshot()
    tokens = snap["prompt_tokens"] + snap["completion_tokens"] + snap["cached_prompt_tokens"] + snap["cached_completion_tokens"]
    return {
        "status": STATUS,
        "cache": RAG_TEST2_CACHE.relative_to(ROOT).as_posix() + " (non versionne)",
        "answerable": {"n": len(simple), "correct": sum(bool(r["correct"]) for r in simple),
                       "per_category": {p: {"n": sum(r["piege"] == p for r in simple), "correct": sum(bool(r["correct"]) for r in simple if r["piege"] == p)} for p in SIMPLE}},
        "sans_reponse": {"n": len(none), "abstained": sum(bool(r["abstained"]) for r in none)},
        "complex": {"n": len(cplx), "expected_units": expected_total, "expected_units_shown": sum(r["expected_shown"] for r in cplx),
                    "share_expected_shown": round(sum(r["expected_shown"] for r in cplx) / expected_total, 4),
                    "queries_with_all_expected_shown": sum(r["expected_shown"] == len(r["relevant_unit_ids"]) for r in cplx)},
        "citations": {"total": len(cited), "valid": sum(bool(r["citations_valid"]) for r in rows if r["status"] == "synthese"),
                      "syntheses": sum(r["status"] == "synthese" for r in rows)},
        "statuses": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
        "groq_calls": snap["calls"] + snap["cache_hits"], "api_calls_this_run": snap["calls"],
        "groq_tokens": tokens,
        "per_query": rows,
    }


# --------------------------------------------------------------------------- extraction et relations


PROJECT_ORDER = ["projet_13", "projet_14", "projet_15", "projet_16"]  # ordre fixe de traitement
WORST_CALL_TOKENS = 7_000  # pire cas d'un appel d'extraction (maximum observe dans le cache : 6 482)


def extraction_results(client, projects: list[str] | None = None, budget_left: int | None = None) -> tuple[list, list[str]]:  # noqa: ANN001
    """Extraction projet par projet, dans l'ordre fixe ; avec budget_left, un projet n'est entame que si le pire cas
    de ses appels non mis en cache tient dans le budget restant (sinon arret : seuls les projets complets comptent)."""
    from eval.eval_extraction import COST_KEYS, Matcher, ProjectResult
    from src.ingest.extract import extract_project
    from src.schemas import Relation, SemanticUnit

    matcher = Matcher()
    results, stopped = [], []
    paths = {json.loads(p.read_text(encoding="utf-8"))["project_id"]: p for p in TEST2_DIR.glob("projet_*.json")}
    for pid in projects or PROJECT_ORDER:
        path = paths[pid]
        data = json.loads(path.read_text(encoding="utf-8"))
        if budget_left is not None:
            uncached = len(data["meetings"])  # majorant : chaque reunion peut demander un appel
            spent = client.snapshot()["prompt_tokens"] + client.snapshot()["completion_tokens"]
            if spent + uncached * WORST_CALL_TOKENS > budget_left:
                stopped = [p for p in (projects or PROJECT_ORDER)][len(results):]
                break
        meetings = [{k: m[k] for k in ("source_id", "source_date", "participants", "transcript")} for m in data["meetings"]]
        before = client.snapshot()
        with redirect_stdout(io.StringIO()):
            extraction = extract_project(data["project_id"], meetings, client)
        after = client.snapshot()
        gold = [SemanticUnit.model_validate(u) for u in data["units"]]
        mapping, _, missing = matcher.match(gold, extraction.units, {m["source_id"]: m["transcript"] for m in meetings})
        results.append(ProjectResult(
            project_id=data["project_id"], gold=gold, pred=extraction.units, mapping=mapping,
            gold_rels=[Relation.model_validate(r) for r in data["relations"]], pred_rels=extraction.relations,
            rejections=extraction.rejections, notes=extraction.notes,
            proposed_units=sum(m.proposed_units for m in extraction.meetings),
            proposed_relations=sum(m.proposed_relations for m in extraction.meetings),
            gold_missing_evidence=missing, cost={k: after[k] - before[k] for k in COST_KEYS},
        ))
        if any("reunion_non_traitee" in k for k in extraction.rejections):
            sys.exit(f"ECHEC d'appel sur {pid} : rien n'est ecrit (les reponses obtenues restent en cache)")
    return results, stopped


def stage_extraction(cache_only: bool, budget_left: int, measured: list[str] | None = None) -> dict[str, object]:
    from eval.eval_extraction import compute_metrics
    from src.ingest.extract import make_client

    client = make_client(cache_only=cache_only)
    results, stopped = extraction_results(client, measured, None if cache_only else budget_left)
    if not results:
        sys.exit("QUOTA : aucun projet ne peut etre extrait avec marge ; rien n'est ecrit")
    metrics = compute_metrics(results)
    per_project = {r.project_id: {k: v for k, v in compute_metrics([r]).items() if k in ("units", "relations")} for r in results}
    snap = client.snapshot()
    return {"status": STATUS, "projects_measured": [r.project_id for r in results], "projects_not_measured": stopped,
            "order": "ordre fixe 13, 14, 15, 16 ; arret au dernier projet complet si le quota restant ne suffit pas",
            "metrics": metrics, "per_project": per_project, "groq_tokens": snap["prompt_tokens"] + snap["completion_tokens"],
            "production_tokens_of_used_responses": metrics["cost"]["total_tokens"], "api_calls_this_run": snap["calls"]}


def stage_relations(cache_only: bool, existing: dict[str, object]) -> dict[str, object]:
    from eval.eval_extraction import compute_metrics
    from eval.eval_relations_v2 import embed, variant_relations
    from src.ingest.extract import make_client
    from src.ingest.link import candidate_pairs, link_project, make_link_client

    selected = json.loads((ROOT / "eval/results/relations_v2.json").read_text(encoding="utf-8"))["dev"]["selected"]
    measured = existing["stages"]["extraction"]["projects_measured"]  # type: ignore[index]
    results, _ = extraction_results(make_client(cache_only=True), measured)  # rejeu de l'etape extraction, 0 appel
    emb = embed(results)
    client = make_link_client(cache_only=cache_only)
    after = {}
    for r in results:
        pairs = candidate_pairs(r.pred, r.pred_rels, emb, int(selected["k"]))
        linked = link_project(r.project_id, r.pred, pairs, client, version=str(selected["prompt"]), existing=list(r.pred_rels))
        after[r.project_id] = variant_relations(list(r.pred_rels), linked.relations, {frozenset(p) for p in pairs}, str(selected["mode"]))
    before = compute_metrics(results)["relations"]
    swapped = compute_metrics([dataclasses.replace(r, pred_rels=after[r.project_id]) for r in results])["relations"]
    snap = client.snapshot()
    tokens = snap["prompt_tokens"] + snap["completion_tokens"] + snap["cached_prompt_tokens"] + snap["cached_completion_tokens"]
    per_project = {r.project_id: {"before": compute_metrics([r])["relations"]["strict"],
                                  "after": compute_metrics([dataclasses.replace(r, pred_rels=after[r.project_id])])["relations"]["strict"],
                                  "annotated": len(r.gold_rels), "predicted_before": len(r.pred_rels), "predicted_after": len(after[r.project_id])}
                   for r in results}
    return {"status": STATUS, "configuration": selected, "projects_measured": measured, "before": before, "after": swapped,
            "per_project": per_project, "groq_calls": snap["calls"] + snap["cache_hits"], "groq_tokens": tokens}


# --------------------------------------------------------------------------- principal


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["offline", "rag", "extraction", "relations"], required=True)
    parser.add_argument("--cache-only", action="store_true", help="rejoue depuis le cache sans reseau (verification)")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--projects", nargs="*", help="extraction : projets complets a enregistrer depuis le cache (arret anticipe)")
    parser.add_argument("--interrupted-run", type=str, help="extraction : JSON {tokens, note} de l'execution interrompue")
    args = parser.parse_args()
    check_manifest()
    out: Path = args.out if args.out.is_absolute() else ROOT / args.out
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    stages = data.get("stages", {})
    if args.stage in stages and not args.cache_only:
        sys.exit(f"Etape {args.stage} deja enregistree : execution unique, rien n'est relance.")
    if args.stage == "relations" and "extraction" not in stages:
        sys.exit("L'etape relations a besoin de l'etape extraction.")

    used = QUOTA_BASELINE_TODAY + sum(int(s.get("groq_tokens", 0)) for name, s in stages.items() if name != args.stage)
    estimate = ESTIMATES.get(args.stage, 0)
    if args.stage == "relations":
        estimate = len(stages["extraction"]["projects_measured"]) * WORST_CALL_TOKENS
    if args.stage in ("rag", "relations") and not args.cache_only and used + estimate > QUOTA_LIMIT:
        sys.exit(f"QUOTA : {used} tokens consommes aujourd'hui, l'etape {args.stage} en demande jusqu'a {estimate} "
                 f"(limite {QUOTA_LIMIT}) ; non lancee.")

    if args.stage == "offline":
        section = stage_offline()
    elif args.stage == "rag":
        section = stage_rag(args.cache_only)
    elif args.stage == "extraction":
        previous = args.projects or (stages.get("extraction", {}).get("projects_measured") if args.cache_only else None)
        section = stage_extraction(args.cache_only, QUOTA_LIMIT - used, previous)
        if args.interrupted_run:  # execution interrompue : projets complets enregistres depuis le cache, sans appel
            run_info = json.loads(args.interrupted_run)
            section["projects_not_measured"] = [p for p in PROJECT_ORDER if p not in section["projects_measured"]]
            section["groq_tokens"] = int(run_info["tokens"])
            section["interrupted_run"] = run_info
    else:
        section = stage_relations(args.cache_only, data)
    old = stages.get(args.stage, {})
    if args.cache_only and old and not args.interrupted_run:  # rejeu : consommation et date de l'execution d'origine
        for key in ("groq_tokens", "api_calls_this_run"):
            if key in old:
                section[key] = old[key]
    section["executed_on"] = old.get("executed_on", date.today().isoformat())
    stages[args.stage] = section
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    all_q = queries()
    data.update({
        "description": "Test scelle elargi : projets 13 a 16, redige par l'assistant de developpement avant toute execution",
        "authorship_caveat": ("jeu redige par l'assistant de developpement apres lecture des interfaces de l'agent, ce qui peut le "
                              "rendre plus favorable au systeme qu'un jeu redige par un tiers"),
        "script": "eval/eval_test2.py",
        "frozen_files": manifest["files"],
        "queries": {"file": QUERIES_PATH.relative_to(ROOT).as_posix(), "total": len(all_q),
                    "by_type": {p: sum(q["piege"] == p for q in all_q) for p in SIMPLE + ["sans_reponse", "complexe", "dossier"]}},
        "models": {"e5_finetuned": results_io.model_info(ROOT / "models" / "e5-finetuned"), "retrained": False},
        "quota": {"limit_tokens_per_day": QUOTA_LIMIT, "baseline_today_before_test2": QUOTA_BASELINE_TODAY,
                  "test2_tokens": {name: int(s.get("groq_tokens", 0)) for name, s in stages.items()}},
        "stages": stages,
    })
    data["quota"]["total_today_after_test2"] = QUOTA_BASELINE_TODAY + sum(data["quota"]["test2_tokens"].values())
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in section.items() if k not in ("per_query",)}, ensure_ascii=False, default=str)[:4000])
    print(f"Resultats ecrits dans {out}")


if __name__ == "__main__":
    main()

"""Intervalles de confiance a 95 % (bootstrap) et tests apparies, sur dev, test et test2.

- Intervalle : bootstrap percentile, 10 000 tirages avec remise, graine fixe (SEED). Unite de tirage : la requete pour
  les metriques par requete (MRR, Recall@5, reponse finale, abstention) ; le projet pour les F1 d'extraction (unites,
  relations), calcules sur les comptes sommes des projets tires (bootstrap par grappes). Avec 2 projets (test 11-12)
  ou 4 (test2), l'intervalle par projet est tres grossier ; il est donne tel quel.
- Test apparie : difference moyenne A - B sur les memes requetes (ou projets), intervalle bootstrap apparie de la
  difference, et p-valeur bilaterale d'un test de permutation par inversion de signe (10 000 permutations, meme
  graine). "Significatif" = p < 0,05 ET intervalle de la difference qui exclut 0.

Les valeurs par requete de la recherche (dev, test) ne sont pas enregistrees dans retrieval_*.json : elles sont
recalculees ici avec le meme code deterministe, et le script verifie que leurs moyennes redonnent exactement les
valeurs publiees (sinon il s'arrete sans rien ecrire). Les relations v2 par projet sont rejouees depuis les caches
locaux (0 appel). Aucun reglage, aucune nouvelle execution de systeme sur un jeu de test.

    uv run python eval/stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_baselines import METRICS, TOP_K, score_query  # noqa: E402
from src.memory.corpus import load_corpus, load_queries  # noqa: E402

RESULTS = ROOT / "eval" / "results"
OUT_PATH = RESULTS / "stats.json"
SEED = 20261116
N_RESAMPLES = 10_000
ALPHA = 0.05
MRR, R5 = METRICS.index("MRR"), METRICS.index("Recall@5")


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- statistiques


def ci_mean(values: list[float]) -> dict[str, object]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(N_RESAMPLES, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": len(x), "estimate": round(float(x.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def f1_from(counts: np.ndarray) -> float:
    tp, pred, gold = counts
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def ci_f1_clusters(per_project: dict[str, tuple[int, int, int]]) -> dict[str, object]:
    """per_project : projet -> (justes, predites, annotees) ; bootstrap par projet du F1 sur les comptes sommes."""
    counts = np.asarray(list(per_project.values()), dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(counts), size=(N_RESAMPLES, len(counts)))
    f1s = np.array([f1_from(counts[i].sum(axis=0)) for i in idx])
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return {"n_projects": len(counts), "estimate": round(f1_from(counts.sum(axis=0)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def paired(a: list[float], b: list[float]) -> dict[str, object]:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(N_RESAMPLES, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    signs = np.random.default_rng(SEED).choice([-1.0, 1.0], size=(N_RESAMPLES, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    p = (1 + int((perm >= abs(d.mean()) - 1e-12).sum())) / (N_RESAMPLES + 1)
    significant = bool(p < ALPHA and (lo > 0 or hi < 0))
    return {"n": len(d), "mean_a": round(float(np.mean(a)), 4), "mean_b": round(float(np.mean(b)), 4),
            "difference": round(float(d.mean()), 4), "ci95_difference": [round(float(lo), 4), round(float(hi), 4)],
            "p_value_permutation": round(p, 4), "significant": significant, "ties": int((d == 0).sum())}


def paired_f1_clusters(before: dict[str, tuple[int, int, int]], after: dict[str, tuple[int, int, int]]) -> dict[str, object]:
    projects = sorted(before)
    b = np.asarray([before[p] for p in projects], dtype=float)
    a = np.asarray([after[p] for p in projects], dtype=float)
    diff = f1_from(a.sum(axis=0)) - f1_from(b.sum(axis=0))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(projects), size=(N_RESAMPLES, len(projects)))
    boot = np.array([f1_from(a[i].sum(axis=0)) - f1_from(b[i].sum(axis=0)) for i in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # permutation : echange aleatoire, projet par projet, des comptes avant/apres
    swaps = np.random.default_rng(SEED).integers(0, 2, size=(N_RESAMPLES, len(projects))).astype(bool)
    perm = np.array([f1_from(np.where(s[:, None], b, a).sum(axis=0)) - f1_from(np.where(s[:, None], a, b).sum(axis=0)) for s in swaps])
    p = (1 + int((np.abs(perm) >= abs(diff) - 1e-12).sum())) / (N_RESAMPLES + 1)
    return {"n_projects": len(projects), "f1_a": round(f1_from(a.sum(axis=0)), 4), "f1_b": round(f1_from(b.sum(axis=0)), 4),
            "difference": round(diff, 4), "ci95_difference": [round(float(lo), 4), round(float(hi), 4)],
            "p_value_permutation": round(p, 4), "significant": bool(p < ALPHA and (lo > 0 or hi < 0))}


# --------------------------------------------------------------------------- donnees par requete


def retrieval_rows(split: str) -> dict[str, dict[str, list[float]]]:
    """Valeurs par requete (dev, test : recalculees ; test2 : lues dans test2.json), cle = query_id."""
    if split == "test2":
        rows = load("test2.json")["stages"]["offline"]["retrieval"]["results"]["per_query"]
        return {name: dict(r) for name, r in rows.items()}
    from src.agent.retriever import FINETUNED_PATH, HybridRetriever
    from src.retrieval.bm25_baseline import BM25Retriever
    from src.retrieval.dense_baseline import MODEL_NAME, DenseRetriever

    corpus = load_corpus(ROOT / "data" / "raw")
    queries = [q for q in load_queries(ROOT / "data" / "eval" / f"queries_{split}.json") if q["relevant_unit_ids"] and q["piege"] != "sans_reponse"]
    bm25, base, ft, hybrid = BM25Retriever(corpus), DenseRetriever(corpus, MODEL_NAME), DenseRetriever(corpus, FINETUNED_PATH), HybridRetriever(corpus)
    out: dict[str, dict[str, list[float]]] = {"bm25": {}, "e5_base": {}, "e5_finetuned": {}, "hybrid_rrf": {}}
    for q in queries:
        text, pid, rel = str(q["query"]), str(q["project_id"]), set(q["relevant_unit_ids"])  # type: ignore[arg-type]
        out["bm25"][q["query_id"]] = score_query(bm25.search(text, TOP_K, pid), rel)
        out["e5_base"][q["query_id"]] = score_query(base.search(text, TOP_K, pid), rel)
        out["e5_finetuned"][q["query_id"]] = score_query(ft.search(text, TOP_K, pid), rel)
        out["hybrid_rrf"][q["query_id"]] = score_query([h.unit.unit_id for h in hybrid.search(text, pid, k=TOP_K)], rel)
    published = load(f"retrieval_{split}.json")["experiments"]
    checks = {"e5_base": published["compare_finetuned"]["results"]["e5_base"], "e5_finetuned": published["compare_finetuned"]["results"]["e5_finetuned"],
              "hybrid_rrf": published["hybrid_rrf"]["results"]["hybrid_rrf"]}
    if "run_baselines" in published:
        checks["bm25"] = published["run_baselines"]["results"]["bm25"]
    for name, ref in checks.items():
        for m in ("MRR", "Recall@5"):
            ours = float(np.mean([r[METRICS.index(m)] for r in out[name].values()]))
            if round(ours, 6) != round(ref["global"][m], 6):
                sys.exit(f"ECART {split} {name} {m} : recalcul {ours} contre {ref['global'][m]} publie ; rien n'est ecrit")
    return out


def final_answer(split: str) -> tuple[dict[str, float], dict[str, float]]:
    """(juste par requete avec reponse, abstention par requete sans reponse) de la reponse extractive."""
    if split == "test2":
        s = load("test2.json")["stages"]["offline"]
        return ({r["query_id"]: float(r["correct"]) for r in s["final_answer"]["per_query"]},
                {r["query_id"]: float(r["abstained"]) for r in s["abstention"]["per_query"]})
    fa = load(f"agent_{split}.json")["experiments"]["final_answer"]
    return ({r["query_id"]: float(r["correct"]) for r in fa["results"]["per_query"]},
            {r["query_id"]: float(r["abstained"]) for r in fa["sans_reponse"]["per_query"]})


def rag(split: str) -> tuple[dict[str, float], dict[str, float]] | None:
    if split == "test2":
        stage = load("test2.json")["stages"].get("rag")
        if not stage:
            return None
        rows = [r for r in stage["per_query"] if r["piege"] != "complexe"]
    else:
        rows = load(f"rag_{split}.json")["results"]["per_query"]
    return ({r["query_id"]: float(r["correct"]) for r in rows if r["piege"] != "sans_reponse"},
            {r["query_id"]: float(r["abstained"]) for r in rows if r["piege"] == "sans_reponse"})


def extraction_counts(split: str) -> tuple[dict, dict] | None:
    """(unites, relations strictes) par projet : (justes, predites, annotees)."""
    if split == "test2":
        stage = load("test2.json")["stages"].get("extraction")
        if not stage:
            return None
        per = stage["per_project"]
    else:
        projects = [f"projet_{n:02d}" for n in (range(4, 11) if split == "dev" else range(11, 13))]
        per = {p: load("extraction.json")["per_project"][p] for p in projects}
    units = {p: (v["units"]["matched"], v["units"]["extracted"], v["units"]["annotated"]) for p, v in per.items()}
    rels = {p: (v["relations"]["strict"]["correct"], v["relations"]["extracted"], v["relations"]["annotated"]) for p, v in per.items()}
    return units, rels


def relations_v2_counts(split: str) -> tuple[dict, dict] | None:
    """(avant, apres) par projet pour la configuration retenue ; rejeu depuis les caches locaux, 0 appel."""
    if split == "test2":
        stage = load("test2.json")["stages"].get("relations")
        if not stage:
            return None
        per = stage["per_project"]
        return ({p: (v["before"]["correct"], v["predicted_before"], v["annotated"]) for p, v in per.items()},
                {p: (v["after"]["correct"], v["predicted_after"], v["annotated"]) for p, v in per.items()})
    import dataclasses
    import io
    from contextlib import redirect_stdout

    from eval.eval_extraction import compute_metrics
    from eval.eval_relations_v2 import run

    selected = load("relations_v2.json")["dev"]["selected"]
    try:
        with redirect_stdout(io.StringIO()):
            data = run(split, (selected["prompt"],), cache_only=True)
    except SystemExit:
        return None
    rels = data["variants"][selected["variant"]]["_rels"]
    before, after = {}, {}
    for r in data["results"]:
        b = compute_metrics([r])["relations"]
        a = compute_metrics([dataclasses.replace(r, pred_rels=rels[r.project_id])])["relations"]
        before[r.project_id] = (b["strict"]["correct"], b["extracted"], b["annotated"])
        after[r.project_id] = (a["strict"]["correct"], a["extracted"], a["annotated"])
    published = load("relations_v2.json")[split]
    if (round(f1_from(np.sum(list(after.values()), axis=0)), 4) != published["after"]["strict"]["f1"]
            or round(f1_from(np.sum(list(before.values()), axis=0)), 4) != published["before"]["strict"]["f1"]):
        sys.exit(f"ECART relations v2 {split} : le rejeu ne redonne pas relations_v2.json ; rien n'est ecrit")
    return before, after


# --------------------------------------------------------------------------- principal


def common(a: dict[str, float], b: dict[str, float]) -> tuple[list[float], list[float]]:
    keys = sorted(set(a) & set(b))
    return [a[k] for k in keys], [b[k] for k in keys]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    intervals: dict[str, dict[str, object]] = {}
    tests: dict[str, dict[str, object]] = {}
    for split in ("dev", "test", "test2"):
        iv: dict[str, object] = {}
        ret = retrieval_rows(split)
        for name in ("bm25", "e5_base", "e5_finetuned", "hybrid_rrf"):
            if ret.get(name):
                iv[f"MRR {name}"] = ci_mean([r[MRR] for r in ret[name].values()])
                iv[f"Recall@5 {name}"] = ci_mean([r[R5] for r in ret[name].values()])
        correct, abstained = final_answer(split)
        iv["reponse finale extractive"] = ci_mean(list(correct.values()))
        iv["abstention extractive"] = ci_mean(list(abstained.values()))
        r = rag(split)
        if r:
            iv["reponse finale RAG"] = ci_mean(list(r[0].values()))
            iv["abstention RAG"] = ci_mean(list(r[1].values()))
        ex = extraction_counts(split)
        if ex:
            iv["F1 unites"] = ci_f1_clusters(ex[0])
            iv["F1 relations strict"] = ci_f1_clusters(ex[1])
        intervals[split] = iv

        t: dict[str, object] = {}
        ft, base = common({k: v[MRR] for k, v in ret["e5_finetuned"].items()}, {k: v[MRR] for k, v in ret["e5_base"].items()})
        t["MRR e5 fine-tune contre e5 de base"] = paired(ft, base)
        hy, de = common({k: v[MRR] for k, v in ret["hybrid_rrf"].items()}, {k: v[MRR] for k, v in ret["e5_finetuned"].items()})
        t["MRR hybride contre dense fine-tune"] = paired(hy, de)
        if r:
            a, b = common(r[0], correct)
            t["reponse finale RAG contre extractive"] = paired(a, b)
            a, b = common(r[1], abstained)
            t["abstention RAG contre extractive"] = paired(a, b)
        rv = relations_v2_counts(split)
        if rv:
            t["F1 relations v2 contre v1 (extraction seule)"] = paired_f1_clusters(rv[0], rv[1])
        tests[split] = t
        print(f"== {split}")
        for k, v in iv.items():
            print(f"  {k:<32} {v['estimate']:.3f}  IC95 [{v['ci95'][0]:.3f} ; {v['ci95'][1]:.3f}]")
        for k, v in t.items():
            print(f"  {k:<48} diff {v['difference']:+.3f}  IC95 [{v['ci95_difference'][0]:+.3f} ; {v['ci95_difference'][1]:+.3f}]  "
                  f"p {v['p_value_permutation']:.4f}  {'SIGNIFICATIF' if v['significant'] else 'non significatif'}")

    payload = {
        "description": "Intervalles de confiance a 95 % et tests apparies",
        "script": "eval/stats.py",
        "method": {"bootstrap": "percentile, tirage avec remise", "n_resamples": N_RESAMPLES, "seed": SEED,
                   "unit": "requete pour les metriques par requete ; projet (grappes) pour les F1 d'extraction",
                   "paired_test": "permutation par inversion de signe (requetes) ou echange avant/apres (projets), bilaterale",
                   "significant": "p < 0,05 et intervalle bootstrap de la difference qui exclut 0",
                   "retrieval_per_query": "dev et test recalcules avec le meme code et verifies contre retrieval_*.json ; test2 lu dans test2.json"},
        "intervals": intervals,
        "paired_tests": tests,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultats ecrits dans {OUT_PATH}")


if __name__ == "__main__":
    main()

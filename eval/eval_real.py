"""Evaluation externe de la recherche de passages sur QMSum (reunions reelles ou jouees, en anglais).

PORTEE : ces chiffres mesurent la RECHERCHE DE PASSAGES dans des reunions en anglais (corpus externe).
Ils ne mesurent PAS la detection des decisions, engagements, contradictions ou responsabilites, evaluee
separement sur le corpus interne (eval/run_baselines.py, eval/eval_extraction.py).

Donnees (voir data/external/README.md pour les sources et licences) :
- requetes : `specific_query_list` de QMSum ayant des `relevant_text_span`, domaines Product (AMI),
  Academic (ICSI) et Committee (Canada `covid_*`, Pays de Galles `education_*`) ;
- split : TEST pour la mesure. VAL est reserve a un eventuel reglage ; aucun reglage n'a ete fait.
- passages candidats : les tours de parole de LA reunion de la requete (un passage par tour), texte
  "<locuteur>: <contenu>", balises {vocalsound}, {disfmarker}... retirees. Tous les tours sont gardes
  (meme vides apres nettoyage) pour que l'indice d'un passage reste sa position dans la reunion.
- pertinence : un passage est pertinent si son indice (position dans `meeting_transcripts`, a partir de 0)
  tombe dans l'un des intervalles [debut, fin] de la requete, BORNES INCLUSES. Tous les passages de tous
  les intervalles comptent. Regle verifiee a la main sur un exemple par domaine.
- requetes exclues : sans intervalle, intervalle hors des bornes de la reunion, ou dont tous les passages
  pertinents sont vides apres nettoyage (aucun systeme ne peut alors les trouver). Elles sont listees.

Systemes (memes reunions, requetes, passages candidats et regles de pertinence pour tous) :
- bm25          : BM25Okapi, tokenisation identique au corpus interne (src/retrieval/bm25_baseline.tokenize) ;
- e5_base       : intfloat/multilingual-e5-small, prefixes "query: " / "passage: ", cosinus ;
- e5_finetuned  : models/e5-finetuned (fine-tune sur le corpus interne en francais), memes prefixes ;
- hybrid_rrf    : fusion Reciprocal Rank Fusion (k = 60) des classements complets bm25 et e5_finetuned,
                  comme l'agent (src/agent/retriever.py).
Egalites departagees par l'indice du tour (ordre stable) : relancer donne les memes classements.

Metriques (calculees PAR REQUETE puis moyennees ; fonctions de eval/metrics.py, identiques au corpus interne) :
- Recall@K : part des passages pertinents retrouves dans les K premiers (pertinents dans le top K / nombre
  total de passages pertinents). Quand une requete a plus de K passages pertinents, Recall@K ne peut pas
  atteindre 1 : le plafond moyen est affiche a cote ("recall@K_plafond").
- Hit@K : part des requetes ayant au moins un passage pertinent dans les K premiers (1 ou 0 par requete).
- MRR : inverse du rang du premier passage pertinent (1/rang), 0 si aucun.
- nDCG@10 : gain cumule actualise des 10 premiers (gain 1 par passage pertinent, actualisation 1/log2(rang+1)),
  rapporte au gain du classement ideal (tous les pertinents en tete).

Memoire limitee : les modeles sont charges et evalues l'un apres l'autre, puis liberes.

    uv run python eval/eval_real.py                      # test, ecrit eval/results/real_qmsum.json
    uv run python eval/eval_real.py --check-reproducible # relance complete et compare les chiffres
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402
from src.retrieval.bm25_baseline import tokenize  # noqa: E402

QMSUM_DIR = ROOT / "data" / "external" / "qmsum" / "data"
OUT_PATH = ROOT / "eval" / "results" / "real_qmsum.json"
DOMAINS = ["Product", "Academic", "Committee"]
E5_BASE = "intfloat/multilingual-e5-small"
E5_FINETUNED = ROOT / "models" / "e5-finetuned"
RRF_K = 60
KS = (5, 10)
SYSTEMS = ["bm25", "e5_base", "e5_finetuned", "hybrid_rrf"]
SCOPE = (
    "Recherche de passages dans des reunions en anglais (corpus externe QMSum). Ne mesure pas la detection "
    "des decisions, engagements, contradictions ou responsabilites, evaluee sur le corpus interne."
)
_MARKUP = re.compile(r"\{[^}]*\}")
_SPACES = re.compile(r"\s+")


@dataclass
class Meeting:
    meeting_id: str  # "<Domaine>/<nom de fichier sans .json>"
    domain: str
    subgroup: str  # Product, Academic, Committee/Canada, Committee/Pays de Galles
    passages: list[str]


@dataclass
class Query:
    query_id: str
    meeting_id: str
    domain: str
    subgroup: str
    text: str
    relevant: set[int] = field(default_factory=set)


# --------------------------------------------------------------------------- preparation


def clean_turn(content: str) -> str:
    """Retire les balises de transcription ({vocalsound}, {disfmarker}...) et normalise les espaces."""
    return _SPACES.sub(" ", _MARKUP.sub(" ", content)).strip()


def passage_text(turn: dict[str, str]) -> str:
    content = clean_turn(str(turn.get("content", "")))
    return f"{turn.get('speaker', '').strip()}: {content}" if content else ""


def relevant_indices(spans: list[list[str]] | None, n_turns: int) -> tuple[set[int], str | None]:
    """Indices des tours pertinents (bornes incluses), ou (vide, motif d'exclusion)."""
    if not spans:
        return set(), "aucun intervalle"
    indices: set[int] = set()
    for span in spans:
        start, end = int(span[0]), int(span[1])
        if start < 0 or end < start or end >= n_turns:
            return set(), f"intervalle hors bornes [{start}, {end}] pour {n_turns} tours"
        indices.update(range(start, end + 1))
    return indices, None


def subgroup_of(domain: str, name: str) -> str:
    if domain != "Committee":
        return domain
    return "Committee/Canada" if name.startswith("covid") else "Committee/Pays de Galles"


def load_split(split: str) -> tuple[list[Meeting], list[Query], list[dict[str, str]]]:
    meetings: list[Meeting] = []
    queries: list[Query] = []
    excluded: list[dict[str, str]] = []
    for domain in DOMAINS:
        for path in sorted((QMSUM_DIR / domain / split).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            turns = data["meeting_transcripts"]
            meeting = Meeting(f"{domain}/{path.stem}", domain, subgroup_of(domain, path.stem), [passage_text(t) for t in turns])
            meetings.append(meeting)
            for i, item in enumerate(data["specific_query_list"]):
                query_id = f"{meeting.meeting_id}#{i}"
                relevant, reason = relevant_indices(item.get("relevant_text_span"), len(turns))
                if reason is None and not any(meeting.passages[j] for j in relevant):
                    reason = "tous les passages pertinents sont vides apres nettoyage"
                if reason:
                    excluded.append({"query_id": query_id, "query": item["query"], "reason": reason})
                    continue
                queries.append(Query(query_id, meeting.meeting_id, domain, meeting.subgroup, item["query"], relevant))
    return meetings, queries, excluded


# --------------------------------------------------------------------------- classements


def rank_by_scores(scores: list[float]) -> list[int]:
    """Indices tries par score decroissant, egalites departagees par l'indice (deterministe)."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda i: (-fused[i], i))


def bm25_rankings(meetings: list[Meeting], queries: list[Query]) -> dict[str, list[int]]:
    from rank_bm25 import BM25Okapi

    by_meeting = {m.meeting_id: m for m in meetings}
    indexes = {m.meeting_id: BM25Okapi([tokenize(p) or ["∅"] for p in m.passages]) for m in meetings}
    return {
        q.query_id: rank_by_scores(list(indexes[q.meeting_id].get_scores(tokenize(q.text))))
        for q in queries
        if q.meeting_id in by_meeting
    }


def dense_rankings(model_name_or_path: str | Path, meetings: list[Meeting], queries: list[Query]) -> dict[str, list[int]]:
    """Charge un modele, classe toutes les requetes, puis le libere (un seul modele en memoire a la fois)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_name_or_path), device="cpu")
    rankings: dict[str, list[int]] = {}
    queries_by_meeting: dict[str, list[Query]] = {}
    for q in queries:
        queries_by_meeting.setdefault(q.meeting_id, []).append(q)
    for meeting in meetings:
        meeting_queries = queries_by_meeting.get(meeting.meeting_id, [])
        if not meeting_queries:
            continue
        passages = model.encode(
            [f"passage: {p}" for p in meeting.passages], batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        q_emb = model.encode([f"query: {q.text}" for q in meeting_queries], normalize_embeddings=True, show_progress_bar=False)
        scores = np.asarray(q_emb, dtype=np.float32) @ np.asarray(passages, dtype=np.float32).T
        for q, row in zip(meeting_queries, scores):
            rankings[q.query_id] = rank_by_scores([round(float(s), 6) for s in row])
        print(f"    {meeting.meeting_id}: {len(meeting.passages)} passages, {len(meeting_queries)} requetes", flush=True)
    del model
    gc.collect()
    return rankings


# --------------------------------------------------------------------------- metriques


def query_metrics(ranking: list[int], relevant: set[int]) -> dict[str, float]:
    ids = [str(i) for i in ranking]
    rel = {str(i) for i in relevant}
    row = {f"recall@{k}": recall_at_k(ids, rel, k) for k in KS}
    for k in KS:
        row[f"hit@{k}"] = 1.0 if rel & set(ids[:k]) else 0.0
    row["mrr"] = mrr(ids, rel)
    row["ndcg@10"] = ndcg_at_k(ids, rel, 10)
    for k in KS:
        row[f"recall@{k}_plafond"] = min(k, len(rel)) / len(rel)
    return row


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"n_queries": 0}
    keys = rows[0].keys()
    return {"n_queries": len(rows), **{k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}}


GROUPS: dict[str, callable] = {  # type: ignore[valid-type]
    "Product": lambda q: q.domain == "Product",
    "Academic": lambda q: q.domain == "Academic",
    "Committee": lambda q: q.domain == "Committee",
    "Committee/Canada": lambda q: q.subgroup == "Committee/Canada",
    "Committee/Pays de Galles": lambda q: q.subgroup == "Committee/Pays de Galles",
    "global": lambda q: True,
    "global hors Pays de Galles": lambda q: q.subgroup != "Committee/Pays de Galles",
}

# Statut de licence des transcriptions de chaque groupe (detail et sources : data/external/README.md).
WALES_STATUS = "NON VERIFIEE pour les comites du Senedd (ni interdite ni accordee explicitement) ; jamais presentee comme juridiquement validee"
LICENCE_STATUS = {
    "Product": "AMI : CC BY 4.0, verifiee sur la page officielle",
    "Academic": "ICSI : CC BY 4.0, verifiee sur la page officielle",
    "Committee": "mixte : Canada verifiee (reproduction non commerciale) ; Pays de Galles " + WALES_STATUS,
    "Committee/Canada": "Chambre des communes : reproduction exacte, non officielle, non commerciale autorisee, verifiee",
    "Committee/Pays de Galles": WALES_STATUS,
    "global": "mixte : inclut le Pays de Galles, " + WALES_STATUS,
    "global hors Pays de Galles": "AMI, ICSI et Canada : licences verifiees",
}


def evaluate(split: str) -> dict[str, object]:
    meetings, queries, excluded = load_split(split)
    print(f"Split {split} : {len(meetings)} reunions, {len(queries)} requetes evaluables, {len(excluded)} exclues", flush=True)

    rankings: dict[str, dict[str, list[int]]] = {}
    print("[1/4] BM25", flush=True)
    rankings["bm25"] = bm25_rankings(meetings, queries)
    print("[2/4] e5 de base", flush=True)
    rankings["e5_base"] = dense_rankings(E5_BASE, meetings, queries)
    print("[3/4] e5 fine-tune", flush=True)
    rankings["e5_finetuned"] = dense_rankings(E5_FINETUNED, meetings, queries)
    print("[4/4] hybride RRF (bm25 + e5 fine-tune)", flush=True)
    rankings["hybrid_rrf"] = {q.query_id: rrf([rankings["bm25"][q.query_id], rankings["e5_finetuned"][q.query_id]]) for q in queries}

    per_query = {s: {q.query_id: query_metrics(rankings[s][q.query_id], q.relevant) for q in queries} for s in SYSTEMS}
    results = {
        s: {name: aggregate([per_query[s][q.query_id] for q in queries if keep(q)]) for name, keep in GROUPS.items()}
        for s in SYSTEMS
    }
    counts = {name: sum(keep(q) for q in queries) for name, keep in GROUPS.items()}
    excluded_by_domain = {d: sum(e["query_id"].startswith(f"{d}/") for e in excluded) for d in DOMAINS}
    return {
        "scope": SCOPE,
        "generated_on": date.today().isoformat(),
        "dataset": {
            "name": "QMSum",
            "source": "https://github.com/Yale-LILY/QMSum",
            "licences": "voir data/external/README.md (AMI et ICSI CC BY 4.0 ; Canada : reproduction non commerciale ; "
            "Pays de Galles : licence des comites non verifiee, sous-groupe rapporte a part)",
            "split_measured": split,
            "split_tuning": "val (aucun reglage effectue)",
            "queries": "specific_query_list avec relevant_text_span",
            "domains": DOMAINS,
            "meetings": len(meetings),
        },
        "protocol": {
            "passages": "un passage = un tour de parole de la reunion de la requete ; texte '<locuteur>: <contenu>', "
            "balises {..} retirees ; tours vides conserves pour garder les indices",
            "relevance": "indice du tour (position dans meeting_transcripts, a partir de 0) dans un relevant_text_span, "
            "bornes incluses ; tous les intervalles comptent",
            "metrics": "par requete puis moyenne ; Recall@K = pertinents dans le top K / pertinents ; MRR = 1/rang du "
            "premier pertinent ; nDCG@10 rapporte au classement ideal ; recall@K_plafond = moyenne de min(K, |pertinents|)/|pertinents|",
            "systems": {
                "bm25": "BM25Okapi, tokenisation de src/retrieval/bm25_baseline.py",
                "e5_base": E5_BASE,
                "e5_finetuned": str(E5_FINETUNED.relative_to(ROOT).as_posix()),
                "hybrid_rrf": f"RRF k={RRF_K} sur les classements complets bm25 et e5_finetuned",
            },
            "e5_prefixes": {"query": "query: ", "passage": "passage: "},
            "tie_break": "indice du tour croissant",
        },
        "licence_status": LICENCE_STATUS,
        "counts": {"evaluated_queries": counts, "excluded_queries_by_domain": excluded_by_domain},
        "excluded_queries": excluded,
        "results": results,
        "per_query": per_query,
    }


def print_table(report: dict[str, object]) -> None:
    print(f"\n{report['scope']}\n")
    results = report["results"]  # type: ignore[index]
    header = (
        f"{'groupe':<27} {'n':>4} {'systeme':<13} {'R@5':>6} {'R@10':>6} {'Hit@5':>6} {'Hit@10':>6} {'MRR':>6} "
        f"{'nDCG@10':>8}   plafond R@5/R@10"
    )
    print(header)
    print("-" * len(header))
    for group in GROUPS:
        print(f"[licence] {LICENCE_STATUS[group]}")
        for s in SYSTEMS:
            r = results[s][group]
            if not r["n_queries"]:
                continue
            print(
                f"{group:<27} {r['n_queries']:>4} {s:<13} {r['recall@5']:>6.3f} {r['recall@10']:>6.3f} {r['hit@5']:>6.3f} "
                f"{r['hit@10']:>6.3f} {r['mrr']:>6.3f} {r['ndcg@10']:>8.3f}   {r['recall@5_plafond']:.3f}/{r['recall@10_plafond']:.3f}"
            )
        print()
    excluded = report["excluded_queries"]  # type: ignore[index]
    print(f"Requetes exclues : {len(excluded)}")
    for e in excluded:
        print(f"  - {e['query_id']} : {e['reason']} | {e['query']}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--check-reproducible", action="store_true", help="relance tout et verifie l'egalite des chiffres")
    args = parser.parse_args()
    if not QMSUM_DIR.exists():
        sys.exit(f"ERREUR : QMSum introuvable dans {QMSUM_DIR} (voir data/external/README.md).")
    if not (E5_FINETUNED / "config.json").exists():
        sys.exit("ERREUR : models/e5-finetuned introuvable (lance scripts/train_embeddings.py).")

    report = evaluate(args.split)
    print_table(report)
    if args.check_reproducible:
        print("\nRelance complete pour verifier la reproductibilite…", flush=True)
        again = evaluate(args.split)
        same = again["results"] == report["results"] and again["per_query"] == report["per_query"]
        diffs = [
            abs(v - again["per_query"][s][qid][k])
            for s, by_query in report["per_query"].items()
            for qid, row in by_query.items()
            for k, v in row.items()
        ]
        report["reproducibility"] = {
            "runs": 2,
            "rerun_identical": same,
            "compared_per_query_values": len(diffs),
            "max_abs_difference": max(diffs) if diffs else 0.0,
            "same_query_sets": again["counts"] == report["counts"] and again["excluded_queries"] == report["excluded_queries"],
            "checked_on": date.today().isoformat(),
        }
        print(f"Relance identique : {same} ({len(diffs)} valeurs par requete comparees, ecart max {report['reproducibility']['max_abs_difference']})")
        if not same:
            sys.exit("ERREUR : la relance ne donne pas les memes chiffres ; resultats non ecrits.")
    if args.split == "test":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResultats ecrits dans {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

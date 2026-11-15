"""Evalue l'extraction d'unites/relations (src/ingest/extract.py) contre les annotations du corpus.

L'extraction ne recoit que les reunions brutes (source_id, source_date, participants, transcript) :
les unites et relations annotees ne servent qu'a la comparaison.

Appariement extrait/annote, dans une meme reunion, un pour un (glouton par score decroissant) :
- recouvrement des evidences : les deux sont des extraits exacts du transcript, on mesure le
  chevauchement de leurs positions dans la meme prise de parole (longueur commune / plus courte) ;
  apparie si >= MIN_OVERLAP ;
- sinon similarite e5 (base, prefixe "query: ") des statements >= COS_THRESHOLD.
  0.93 : sur les unites annotees, 2.8 % des paires d'unites DISTINCTES d'une meme reunion depassent
  ce seuil, tandis que 81 % des paires (statement, sa propre evidence) le depassent. Au-dessous
  (0.90 : 15 % de faux appariements), le critere ne discrimine plus. C'est un critere de secours :
  le critere principal reste le recouvrement d'evidence.

Exemples :
    uv run python eval/eval_extraction.py --projects 1 --show
    uv run python eval/eval_extraction.py --projects 1 2 3 --groups p01=1 p02=2 p03=3
    uv run python eval/eval_extraction.py --projects 4-12 --groups dev_04_10=4-10 test_11_12=11-12 \\
        --out eval/results/extraction.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingest import extract as extract_module  # noqa: E402
from src.ingest.extract import _norm, _turns, extract_project, make_client  # noqa: E402
from src.ingest.groq_client import MissingKeyError  # noqa: E402
from src.schemas import Relation, SemanticUnit  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
E5_MODEL = "intfloat/multilingual-e5-small"
MIN_OVERLAP = 0.5
COS_THRESHOLD = 0.93
TYPES = ["decision", "action", "commitment", "question", "answer", "fact", "risk", "idea"]
RELATION_TYPES = ["answers", "creates_action", "supports", "supersedes", "contradicts"]
COST_KEYS = ["calls", "cache_hits", "prompt_tokens", "completion_tokens", "cached_prompt_tokens", "cached_completion_tokens"]


@dataclass(frozen=True)
class Span:
    turn: int
    start: int
    end: int


@dataclass
class ProjectResult:
    project_id: str
    gold: list[SemanticUnit]
    pred: list[SemanticUnit]
    mapping: dict[str, str]  # pred unit_id -> gold unit_id
    gold_rels: list[Relation]
    pred_rels: list[Relation]
    rejections: Counter[str] = field(default_factory=Counter)
    notes: Counter[str] = field(default_factory=Counter)
    proposed_units: int = 0
    proposed_relations: int = 0
    gold_missing_evidence: int = 0
    cost: dict[str, int] = field(default_factory=dict)


def locate(evidence: str, turns: list[tuple[str, str]]) -> Span | None:
    needle = _norm(evidence).strip()
    for i, (_, text) in enumerate(turns):
        start = _norm(text).find(needle)
        if start >= 0:
            return Span(i, start, start + len(needle))
    return None


def overlap(a: Span | None, b: Span | None) -> float:
    if a is None or b is None or a.turn != b.turn:
        return 0.0
    common = min(a.end, b.end) - max(a.start, b.start)
    return max(common, 0) / max(min(a.end - a.start, b.end - b.start), 1)


def prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


# --------------------------------------------------------------------------- appariement


class Matcher:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(E5_MODEL, device="cpu")

    def match(
        self, gold: list[SemanticUnit], pred: list[SemanticUnit], transcripts: dict[str, str]
    ) -> tuple[dict[str, str], dict[tuple[str, str], tuple[float, float]], int]:
        """pred_id -> gold_id, details (overlap, cos) par paire retenue, nb d'evidences annotees introuvables."""
        mapping: dict[str, str] = {}
        details: dict[tuple[str, str], tuple[float, float]] = {}
        gold_not_found = 0
        for source_id, transcript in transcripts.items():
            turns = _turns(transcript)
            g = [u for u in gold if u.source_id == source_id]
            p = [u for u in pred if u.source_id == source_id]
            g_spans = [locate(u.evidence, turns) for u in g]
            gold_not_found += sum(s is None for s in g_spans)
            if not g or not p:
                continue
            p_spans = [locate(u.evidence, turns) for u in p]
            g_emb = self._model.encode([f"query: {u.statement}" for u in g], normalize_embeddings=True)
            p_emb = self._model.encode([f"query: {u.statement}" for u in p], normalize_embeddings=True)
            cos = p_emb @ g_emb.T
            candidates = []
            for i in range(len(p)):
                for j in range(len(g)):
                    ov, c = overlap(p_spans[i], g_spans[j]), float(cos[i, j])
                    if ov >= MIN_OVERLAP or c >= COS_THRESHOLD:
                        candidates.append((ov + c, i, j, ov, c))
            used_p, used_g = set(), set()
            for _, i, j, ov, c in sorted(candidates, reverse=True):
                if i in used_p or j in used_g:
                    continue
                used_p.add(i)
                used_g.add(j)
                mapping[p[i].unit_id] = g[j].unit_id
                details[(p[i].unit_id, g[j].unit_id)] = (ov, c)
        return mapping, details, gold_not_found


# --------------------------------------------------------------------------- metriques


def compute_metrics(results: list[ProjectResult]) -> dict[str, object]:
    gold = [u for r in results for u in r.gold]
    pred = [u for r in results for u in r.pred]
    gold_by_id = {u.unit_id: u for u in gold}
    mapping = {k: v for r in results for k, v in r.mapping.items()}
    pairs = [(p, gold_by_id[mapping[p.unit_id]]) for p in pred if p.unit_id in mapping]

    per_type = {}
    for t in TYPES:
        n_gold = sum(u.type == t for u in gold)
        n_pred = sum(u.type == t for u in pred)
        tp = sum(a.type == t and b.type == t for a, b in pairs)
        per_type[t] = {"annotated": n_gold, "extracted": n_pred, "correct": tp, **prf(tp, n_pred, n_gold)}

    type_correct = sum(a.type == b.type for a, b in pairs)
    status_correct = sum(a.status == b.status for a, b in pairs)
    units = {
        "annotated": len(gold),
        "extracted": len(pred),
        "matched": len(pairs),
        **prf(len(pairs), len(pred), len(gold)),
        # Exactitudes calculees sur les seules unites appariees (denominateur = accuracy_denominator).
        "type_correct": type_correct,
        "status_correct": status_correct,
        "accuracy_denominator": len(pairs),
        "type_accuracy": round(type_correct / len(pairs), 4) if pairs else 0.0,
        "status_accuracy": round(status_correct / len(pairs), 4) if pairs else 0.0,
        "per_type": per_type,
        "type_confusions": {f"{g}->{e}": n for (g, e), n in Counter((b.type, a.type) for a, b in pairs if a.type != b.type).most_common()},
        "status_errors": {f"{g}->{e}": n for (g, e), n in Counter((b.status, a.status) for a, b in pairs if a.status != b.status).most_common()},
    }

    gold_rels = [rel for r in results for rel in r.gold_rels]
    pred_rels = [rel for r in results for rel in r.pred_rels]
    gold_set = {(x.from_id, x.to_id, x.relation) for x in gold_rels}
    gold_pairs = {frozenset((x.from_id, x.to_id)) for x in gold_rels}
    mapped = {(mapping.get(x.from_id), mapping.get(x.to_id), x.relation) for x in pred_rels}
    tp_strict = sum(m in gold_set for m in mapped)
    tp_loose = sum(frozenset((a, b)) in gold_pairs for a, b, _ in mapped if a and b)
    rel_per_type = {}
    for t in RELATION_TYPES:
        n_gold = sum(x.relation == t for x in gold_rels)
        n_pred = sum(x.relation == t for x in pred_rels)
        tp = sum(m in gold_set for m in mapped if m[2] == t)
        rel_per_type[t] = {"annotated": n_gold, "extracted": n_pred, "correct": tp, **prf(tp, n_pred, n_gold)}
    relations = {
        "annotated": len(gold_rels),
        "extracted": len(pred_rels),
        "strict": {"correct": tp_strict, **prf(tp_strict, len(pred_rels), len(gold_rels))},
        "loose_endpoints_only": {"correct": tp_loose, **prf(tp_loose, len(pred_rels), len(gold_rels))},
        "per_type": rel_per_type,
    }

    validation = {
        "proposed_units": sum(r.proposed_units for r in results),
        "accepted_units": len(pred),
        "proposed_relations": sum(r.proposed_relations for r in results),
        "accepted_relations": len(pred_rels),
        "rejections": dict(sum((r.rejections for r in results), Counter()).most_common()),
        "notes": dict(sum((r.notes for r in results), Counter()).most_common()),
        "annotated_evidence_not_found": sum(r.gold_missing_evidence for r in results),
    }
    raw = {k: sum(r.cost.get(k, 0) for r in results) for k in COST_KEYS}
    # Cout de production des reponses utilisees : chaque reponse vient d'un appel Groq, fait pendant ce
    # run ou lors d'un run precedent (tokens alors relus dans le cache). Une regeneration --cache-only
    # donne donc le meme cout que le run d'origine ; les lectures du cache ne sont pas de nouveaux appels.
    cost = {
        "groq_calls": raw["calls"] + raw["cache_hits"],
        "prompt_tokens": raw["prompt_tokens"] + raw["cached_prompt_tokens"],
        "completion_tokens": raw["completion_tokens"] + raw["cached_completion_tokens"],
        "api_calls_this_run": raw["calls"],
        "cache_reads_this_run": raw["cache_hits"],
    }
    cost["total_tokens"] = cost["prompt_tokens"] + cost["completion_tokens"]
    return {
        "projects": [r.project_id for r in results],
        "units": units,
        "relations": relations,
        "validation": validation,
        "cost": cost,
    }


def print_metrics(name: str, m: dict[str, object]) -> None:
    u, rel, val, cost = m["units"], m["relations"], m["validation"], m["cost"]  # type: ignore[index]
    print(f"\n{'=' * 78}\n{name}  ({', '.join(m['projects'])})\n{'=' * 78}")  # type: ignore[arg-type]
    print(f"UNITES  annotees {u['annotated']}, extraites {u['extracted']}, appariees {u['matched']}")
    n = u["accuracy_denominator"]
    print(
        f"  P {u['precision']:.3f}  R {u['recall']:.3f}  F1 {u['f1']:.3f}   |  "
        f"type exact {u['type_correct']}/{n} = {u['type_accuracy']:.3f}   "
        f"statut exact {u['status_correct']}/{n} = {u['status_accuracy']:.3f}"
    )
    print(f"  {'type':<11} {'annot.':>6} {'extr.':>6} {'P':>6} {'R':>6} {'F1':>6}")
    for t, s in u["per_type"].items():
        if s["annotated"] or s["extracted"]:
            print(f"  {t:<11} {s['annotated']:>6} {s['extracted']:>6} {s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f}")
    if u["type_confusions"]:
        print("  confusions de type (annote->extrait) : " + ", ".join(f"{k} x{v}" for k, v in u["type_confusions"].items()))
    if u["status_errors"]:
        print("  erreurs de statut (annote->extrait)  : " + ", ".join(f"{k} x{v}" for k, v in u["status_errors"].items()))
    s, lo = rel["strict"], rel["loose_endpoints_only"]
    print(f"RELATIONS  annotees {rel['annotated']}, extraites {rel['extracted']}")
    print(f"  strict P {s['precision']:.3f}  R {s['recall']:.3f}  F1 {s['f1']:.3f}   |  extremites seules P {lo['precision']:.3f}  R {lo['recall']:.3f}  F1 {lo['f1']:.3f}")
    print(f"  {'relation':<15} {'annot.':>6} {'extr.':>6} {'P':>6} {'R':>6} {'F1':>6}")
    for t, x in rel["per_type"].items():
        if x["annotated"] or x["extracted"]:
            print(f"  {t:<15} {x['annotated']:>6} {x['extracted']:>6} {x['precision']:>6.3f} {x['recall']:>6.3f} {x['f1']:>6.3f}")
    rej = ", ".join(f"{k} x{v}" for k, v in val["rejections"].items()) or "aucun"
    notes = ", ".join(f"{k} x{v}" for k, v in val["notes"].items()) or "aucune"
    print(f"VALIDATION  unites {val['accepted_units']}/{val['proposed_units']} acceptees, relations {val['accepted_relations']}/{val['proposed_relations']}")
    print(f"  rejets : {rej}\n  notes  : {notes}")
    print(
        f"COUT  {cost['groq_calls']} appel(s) Groq ; tokens {cost['total_tokens']} "
        f"(entree {cost['prompt_tokens']}, sortie {cost['completion_tokens']})  "
        f"[ce run : {cost['api_calls_this_run']} nouvel(s) appel(s), {cost['cache_reads_this_run']} lecture(s) du cache]"
    )


# --------------------------------------------------------------------------- affichage cote a cote


def show_side_by_side(meetings: list[dict[str, object]], r: ProjectResult, details: dict[tuple[str, str], tuple[float, float]]) -> None:
    gold_to_pred = {g: p for p, g in r.mapping.items()}
    pred_by_id = {u.unit_id: u for u in r.pred}
    print(f"\n{'=' * 100}\n{r.project_id} : unites annotees (A) et extraites (E)\n{'=' * 100}")
    for meeting in meetings:
        sid = str(meeting["source_id"])
        print(f"\n--- {sid} ({meeting['source_date']}) ---")
        for g in [u for u in r.gold if u.source_id == sid]:
            print(f"A {g.unit_id:<12} {g.type:<10} {g.status:<10} {g.statement}")
            p_id = gold_to_pred.get(g.unit_id)
            if p_id is None:
                print(f"  E {'(manquee)':<12}")
                continue
            p = pred_by_id[p_id]
            ov, c = details[(p_id, g.unit_id)]
            flags = [f"TYPE {p.type}"] * (p.type != g.type) + [f"STATUT {p.status}"] * (p.status != g.status)
            flag = f"  <<< {', '.join(flags)}" if flags else ""
            print(f"  E {p.unit_id:<12} {p.type:<10} {p.status:<10} {p.statement}   [recouvr. {ov:.2f}, cos {c:.2f}]{flag}")
        for p in [u for u in r.pred if u.source_id == sid and u.unit_id not in r.mapping]:
            print(f"E+ {p.unit_id:<11} {p.type:<10} {p.status:<10} {p.statement}   (sans equivalent annote)")
            print(f"     evidence : \"{p.evidence}\"")
    gold_set = {(x.from_id, x.to_id, x.relation) for x in r.gold_rels}
    mapped_pred = {(r.mapping.get(x.from_id), r.mapping.get(x.to_id), x.relation): x for x in r.pred_rels}
    print("\n--- relations ---")
    for x in r.gold_rels:
        ok = "OK " if (x.from_id, x.to_id, x.relation) in mapped_pred else "-- "
        print(f"A {ok} {x.from_id} -{x.relation}-> {x.to_id}")
    for key, x in mapped_pred.items():
        if key not in gold_set:
            via = f" (= {key[0]} -> {key[1]})" if key[0] and key[1] else " (extremite non appariee)"
            print(f"E+    {x.from_id} -{x.relation}-> {x.to_id}{via}")


# --------------------------------------------------------------------------- main


def parse_numbers(spec: str) -> list[int]:
    """'4-10' -> [4..10] ; '11,12' -> [11, 12] ; '3' -> [3]."""
    numbers: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            numbers += list(range(int(lo), int(hi) + 1))
        elif part:
            numbers.append(int(part))
    return numbers


def run_project(number: int, client, matcher: Matcher, show: bool) -> ProjectResult:  # noqa: ANN001
    path = next(RAW_DIR.glob(f"projet_{number:02d}_*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    project_id = data["project_id"]
    # Entree de l'extraction : les reunions brutes SEULES (aucune unite ni relation annotee).
    meetings = [{k: m[k] for k in ("source_id", "source_date", "participants", "transcript")} for m in data["meetings"]]
    print(f"[{project_id}] extraction de {len(meetings)} reunion(s)…", flush=True)
    before = client.snapshot()
    extraction = extract_project(project_id, meetings, client)
    after = client.snapshot()

    gold = [SemanticUnit.model_validate(u) for u in data["units"]]
    gold_rels = [Relation.model_validate(r) for r in data["relations"]]
    transcripts = {str(m["source_id"]): str(m["transcript"]) for m in meetings}
    mapping, details, missing = matcher.match(gold, extraction.units, transcripts)
    result = ProjectResult(
        project_id=project_id,
        gold=gold,
        pred=extraction.units,
        mapping=mapping,
        gold_rels=gold_rels,
        pred_rels=extraction.relations,
        rejections=extraction.rejections,
        notes=extraction.notes,
        proposed_units=sum(m.proposed_units for m in extraction.meetings),
        proposed_relations=sum(m.proposed_relations for m in extraction.meetings),
        gold_missing_evidence=missing,
        cost={k: after[k] - before[k] for k in COST_KEYS},
    )
    if show:
        show_side_by_side(meetings, result, details)
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--projects", nargs="+", default=["1-12"], help="projets a extraire (ex. 1 2 3, ou 4-12)")
    parser.add_argument("--groups", nargs="*", default=[], help="groupes de rapport nom=projets (ex. dev=4-10 test=11-12)")
    parser.add_argument("--show", action="store_true", help="affiche les unites extraites a cote des annotees")
    parser.add_argument("--cache-only", action="store_true", help="n'appelle jamais Groq (reponses en cache seulement)")
    parser.add_argument("--out", type=Path, help="ecrit les metriques en JSON")
    args = parser.parse_args()

    numbers = sorted({n for spec in args.projects for n in parse_numbers(spec)})
    groups = {g.split("=", 1)[0]: parse_numbers(g.split("=", 1)[1]) for g in args.groups}

    client = make_client(cache_only=args.cache_only)
    matcher = Matcher()
    results: dict[int, ProjectResult] = {}
    try:
        for n in numbers:
            results[n] = run_project(n, client, matcher, args.show)
    except MissingKeyError as exc:
        sys.exit(f"ERREUR : {exc}")

    report: dict[str, object] = {}
    for name, members in groups.items():
        subset = [results[n] for n in members if n in results]
        if subset:
            report[name] = compute_metrics(subset)
    report[f"global_{numbers[0]:02d}_{numbers[-1]:02d}"] = compute_metrics(list(results.values()))
    for name, metrics in report.items():
        print_metrics(name, metrics)

    if args.out:
        prompt_sha = hashlib.sha256(extract_module.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        payload = {
            "generated_on": date.today().isoformat(),
            "extraction": {
                "model": extract_module.GROQ_MODEL,
                "temperature": extract_module.GROQ_TEMPERATURE,
                "max_completion_tokens": extract_module.MAX_COMPLETION_TOKENS,
                "system_prompt_sha256": prompt_sha,
                "prompt_tuned_on": ["projet_01", "projet_02", "projet_03"],
            },
            "matching": {"min_evidence_overlap": MIN_OVERLAP, "e5_cosine_threshold": COS_THRESHOLD, "e5_model": E5_MODEL},
            "groups": report,
            "per_project": {r.project_id: compute_metrics([r]) for r in results.values()},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResultats ecrits dans {args.out}")


if __name__ == "__main__":
    main()

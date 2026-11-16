"""Analyse des erreurs d'extraction des relations sur le dev 04-10 (analyse seulement, aucun reglage).

L'extraction est rejouee UNIQUEMENT depuis le cache Groq local (aucun appel), avec la meme logique
d'appariement que eval/eval_extraction.py (run_project, Matcher). Avant toute ecriture, le script verifie
que le rejeu redonne exactement les metriques du groupe dev_04_10 de eval/results/extraction.json.

Categories (extremites traduites en unites annotees via l'appariement des unites) :
- relation predite : juste (memes extremites, meme sens, meme type), sinon, dans cet ordre :
  extremite_non_annotee (une extremite n'est appariee a aucune unite annotee),
  mauvais_sens (la relation annotee existe en sens inverse, meme type),
  mauvais_type (la paire est reliee dans l'annotation, mais par un autre type),
  mauvaise_cible (meme type et meme source, ou meme cible, mais l'autre extremite differe),
  faux_positif_pur (aucun des cas precedents) ;
- relation annotee : retrouvee, sinon, dans cet ordre :
  extremite_non_extraite (une extremite n'a pas d'unite extraite appariee),
  autre_type_ou_sens (la paire est reliee dans l'extraction, avec un autre type ou dans l'autre sens),
  autre_cible (le meme type est predit depuis la meme source, ou vers la meme cible, mais pas sur cette paire),
  jamais_proposee (rien de tout cela).

    uv run python eval/analyze_relation_errors.py
    uv run python eval/analyze_relation_errors.py --expect totals.annotated=84 totals.predicted=54
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "eval" / "results" / "relation_errors_dev.json"
EXTRACTION_JSON = ROOT / "eval" / "results" / "extraction.json"
DEV_PROJECTS = list(range(4, 11))
RELATION_TYPES = ["creates_action", "supports", "answers", "supersedes", "contradicts"]
PRED_ERRORS = ["extremite_non_annotee", "mauvais_sens", "mauvais_type", "mauvaise_cible", "faux_positif_pur"]
GOLD_MISSES = ["extremite_non_extraite", "autre_type_ou_sens", "autre_cible", "jamais_proposee"]


def classify(results: list) -> dict[str, dict[str, object]]:  # noqa: ANN001
    """Comptes par type de relation, a partir des ProjectResult de eval_extraction.run_project."""
    by_type = {
        t: {"annotated": 0, "predicted": 0, "correct": 0, "found": 0,
            "predicted_errors": Counter(), "missed": Counter()}
        for t in RELATION_TYPES
    }
    for r in results:
        m = r.mapping
        gold = {(g.from_id, g.to_id, g.relation) for g in r.gold_rels}
        gold_pairs = {frozenset((g.from_id, g.to_id)) for g in r.gold_rels}
        gold_from_type = {(g.from_id, g.relation) for g in r.gold_rels}
        gold_to_type = {(g.to_id, g.relation) for g in r.gold_rels}
        matched_gold = set(m.values())
        mapped = set()
        for p in r.pred_rels:
            a, b, t = m.get(p.from_id), m.get(p.to_id), p.relation
            mapped.add((a, b, t))
            entry = by_type[t]
            entry["predicted"] += 1
            if (a, b, t) in gold:
                entry["correct"] += 1
            elif a is None or b is None:
                entry["predicted_errors"]["extremite_non_annotee"] += 1
            elif (b, a, t) in gold:
                entry["predicted_errors"]["mauvais_sens"] += 1
            elif frozenset((a, b)) in gold_pairs:
                entry["predicted_errors"]["mauvais_type"] += 1
            elif (a, t) in gold_from_type or (b, t) in gold_to_type:
                entry["predicted_errors"]["mauvaise_cible"] += 1
            else:
                entry["predicted_errors"]["faux_positif_pur"] += 1
        for g in r.gold_rels:
            entry = by_type[g.relation]
            entry["annotated"] += 1
            if (g.from_id, g.to_id, g.relation) in mapped:
                entry["found"] += 1
            elif g.from_id not in matched_gold or g.to_id not in matched_gold:
                entry["missed"]["extremite_non_extraite"] += 1
            elif any(x[0] and x[1] and {x[0], x[1]} == {g.from_id, g.to_id} for x in mapped):
                entry["missed"]["autre_type_ou_sens"] += 1
            elif any((x[0] == g.from_id or x[1] == g.to_id) and x[2] == g.relation for x in mapped):
                entry["missed"]["autre_cible"] += 1
            else:
                entry["missed"]["jamais_proposee"] += 1
    return {
        t: {
            "annotated": e["annotated"],
            "predicted": e["predicted"],
            "correct": e["correct"],
            "predicted_errors": {k: e["predicted_errors"][k] for k in PRED_ERRORS},
            "found": e["found"],
            "missed": {k: e["missed"][k] for k in GOLD_MISSES},
        }
        for t, e in by_type.items()
    }


def totals(by_type: dict[str, dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {k: sum(e[k] for e in by_type.values()) for k in ("annotated", "predicted", "correct", "found")}  # type: ignore[misc]
    out["predicted_errors"] = {k: sum(e["predicted_errors"][k] for e in by_type.values()) for k in PRED_ERRORS}  # type: ignore[index]
    out["missed"] = {k: sum(e["missed"][k] for e in by_type.values()) for k in GOLD_MISSES}  # type: ignore[index]
    out["missed_total"] = sum(out["missed"].values())  # type: ignore[union-attr]
    return out


def analyze() -> tuple[dict[str, object], dict[str, int], bool]:
    """Rejoue le dev 04-10 depuis le cache ; renvoie (analyse, compteurs d'appels, coherence avec extraction.json)."""
    from eval.eval_extraction import Matcher, compute_metrics, run_project
    from src.ingest.extract import make_client

    client = make_client(cache_only=True)
    matcher = Matcher()
    with redirect_stdout(io.StringIO()):
        results = [run_project(n, client, matcher, show=False) for n in DEV_PROJECTS]
    reference = json.loads(EXTRACTION_JSON.read_text(encoding="utf-8"))["groups"]["dev_04_10"]
    replay = compute_metrics(results)
    consistent = replay["units"] == reference["units"] and replay["relations"] == reference["relations"]
    by_type = classify(results)
    return {"by_type": by_type, "totals": totals(by_type)}, {"api_calls": client.calls, "cache_reads": client.cache_hits}, consistent


def lookup(analysis: dict[str, object], key: str) -> object:
    node: object = analysis
    for part in key.split("."):
        node = node[part]  # type: ignore[index]
    return node


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--expect", nargs="*", default=[], help="valeurs attendues cle.sous_cle=entier ; aucun ecart tolere avant ecriture")
    args = parser.parse_args()

    from src.ingest.extract import SYSTEM_PROMPT

    analysis, calls, consistent = analyze()
    problems = []
    if calls["api_calls"]:
        problems.append(f"{calls['api_calls']} appel(s) Groq effectue(s) : le rejeu doit venir du cache seul")
    if not consistent:
        problems.append("le rejeu ne redonne pas les metriques dev_04_10 de extraction.json")
    for item in args.expect:
        key, _, value = item.partition("=")
        if lookup(analysis, key) != int(value):
            problems.append(f"{key} : attendu {value}, obtenu {lookup(analysis, key)}")
    t = analysis["totals"]
    print(f"annotees {t['annotated']}, predites {t['predicted']}, justes {t['correct']}, retrouvees {t['found']}, manquees {t['missed_total']}")  # type: ignore[index]
    print(f"erreurs sur les predites : {t['predicted_errors']}")  # type: ignore[index]
    print(f"manquees : {t['missed']}")  # type: ignore[index]
    print(f"rejeu : {calls['api_calls']} appel Groq, {calls['cache_reads']} lectures du cache ; coherent avec extraction.json : {consistent}")
    if problems:
        sys.exit("ECART, rien n'est ecrit :\n  " + "\n  ".join(problems))

    extraction_meta = json.loads(EXTRACTION_JSON.read_text(encoding="utf-8"))["extraction"]
    payload = {
        "experiment": "analyse des erreurs d'extraction des relations",
        "generated_on": date.today().isoformat(),
        "script": "eval/analyze_relation_errors.py",
        "split": "dev",
        "projects": [f"projet_{n:02d}" for n in DEV_PROJECTS],
        "status": "analyse seulement, aucun reglage ; le test scelle (projets 11-12) n'est pas utilise",
        "source": "cache local des reponses Groq (data/processed/groq_extract_cache.jsonl, non versionne), aucun appel",
        "replay": {**calls, "consistent_with_extraction_json_dev_04_10": consistent},
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "system_prompt_sha256_in_extraction_json": extraction_meta["system_prompt_sha256"],
        "matching": "identique a eval/eval_extraction.py (recouvrement d'evidence, puis cosinus e5)",
        "categories": {
            "predicted_errors": PRED_ERRORS,
            "missed": GOLD_MISSES,
            "definitions": "voir la docstring de eval/analyze_relation_errors.py",
        },
        **analysis,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultats ecrits dans {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

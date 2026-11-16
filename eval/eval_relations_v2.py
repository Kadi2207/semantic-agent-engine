"""Relations v2 : seconde passe de liaison (src/ingest/link.py) sur les unites deja extraites, mesuree a cote.

Rien de l'extraction n'est modifie (consigne gelee, extraction.json, relation_errors_dev.json) : l'extraction est
rejouee depuis son cache (0 appel, verifie contre extraction.json), puis la passe de liaison propose de nouvelles
relations, ecrites dans eval/results/relations_v2.json seulement.

Critere d'acceptation, fixe avant de voir les resultats : sur le dev 04-10, F1 strict des relations >= 0,456
(0,406 + 0,05) avec une precision stricte >= 0,45. Reglage (consigne de liaison, k, mode de fusion) sur les projets
04 a 10 UNIQUEMENT ; puis une seule execution d'enregistrement sur le test scelle 11-12, avec la configuration
retenue sur le dev.

Consignes de liaison : v1, puis v2 (seule revision, apres analyse des erreurs de la v1 sur le dev). Pour chaque
version, un seul appel par projet, sur les paires candidates a K_MAX voisins : seule k = K_MAX est une configuration
reellement executee. Trois modes de fusion, appliques par le code a la sortie de ce meme appel :
- "remplacement" : seules les relations de la passe de liaison (qui rejuge aussi les paires deja reliees) ;
- "union" : relations de l'extraction + relations de la passe de liaison ;
- "complement" : relations de l'extraction + relations de la passe de liaison sur les seules paires que
  l'extraction n'a pas reliees (l'extraction garde la main sur ses paires).
Les variantes k < K_MAX sont INDICATIVES : obtenues par filtrage a posteriori de la reponse k = K_MAX (decisions
gardees sur les seules paires candidates a k voisins), elles ne sont pas des executions (un appel sur moins de paires
pourrait repondre autrement) et ne peuvent pas etre retenues.
Choix sur le dev, parmi les configurations executees (version, k = K_MAX, mode) : meilleur F1 strict parmi celles de
precision >= 0,45 (a egalite, le moins de relations predites) ; si aucune n'a cette precision, meilleur F1. Toutes
les variantes sont enregistrees avec leurs scores, ainsi que le plafond du rappel strict (relations annotees dont les
deux extremites ont ete extraites).

Budget : 100 000 tokens Groq pour tout le lot, comptes depuis le cache de liaison (chaque appel y est enregistre avec
son usage) ; avant chaque appel non mis en cache, le script s'arrete si le budget risque d'etre depasse.

    uv run python eval/eval_relations_v2.py --split dev
    uv run python eval/eval_relations_v2.py --split test
    uv run python eval/eval_relations_v2.py --split dev --cache-only   # rejeu sans reseau
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import results_io  # noqa: E402
from eval.analyze_relation_errors import classify, totals  # noqa: E402
from eval.eval_extraction import Matcher, compute_metrics, run_project  # noqa: E402
from src.ingest import link as link_module  # noqa: E402
from src.ingest.extract import SYSTEM_PROMPT, make_client  # noqa: E402
from src.ingest.link import LINK_CACHE, LINK_PROMPTS, candidate_pairs, link_project, link_prompts, make_link_client  # noqa: E402
from src.schemas import Relation  # noqa: E402

OUT_PATH = ROOT / "eval" / "results" / "relations_v2.json"
EXTRACTION_JSON = ROOT / "eval" / "results" / "extraction.json"
RELATION_ERRORS_JSON = ROOT / "eval" / "results" / "relation_errors_dev.json"
E5_FINETUNED = ROOT / "models" / "e5-finetuned"
SPLITS = {"dev": (list(range(4, 11)), "dev_04_10"), "test": (list(range(11, 13)), "test_11_12")}
VERSIONS = ("v1", "v2")
K_MAX = 3
MODES = ("remplacement", "union", "complement")
F1_MIN = 0.456
PRECISION_MIN = 0.45
BUDGET_TOKENS = 100_000
CALL_ESTIMATE_TOKENS = 12_000  # marge pour un appel de liaison (entree + sortie, raisonnement compris)


def lot_tokens() -> tuple[int, int]:
    """(tokens, appels) de tous les appels de liaison deja faits, lus dans le cache (usage enregistre par appel)."""
    if not LINK_CACHE.exists():
        return 0, 0
    entries = [json.loads(line) for line in LINK_CACHE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sum(int(e["usage"].get("prompt_tokens", 0)) + int(e["usage"].get("completion_tokens", 0)) for e in entries), len(entries)


def replay_extraction(numbers: list[int], group: str) -> list:
    client = make_client(cache_only=True)
    matcher = Matcher()
    with redirect_stdout(io.StringIO()):
        results = [run_project(n, client, matcher, show=False) for n in numbers]
    reference = json.loads(EXTRACTION_JSON.read_text(encoding="utf-8"))["groups"][group]
    replay = compute_metrics(results)
    if client.calls or replay["units"] != reference["units"] or replay["relations"] != reference["relations"]:
        sys.exit(f"ECART : le rejeu de l'extraction ({client.calls} appel) ne redonne pas {group} de extraction.json")
    return results


def embed(results: list) -> dict[str, object]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(E5_FINETUNED), device="cpu")
    emb: dict[str, object] = {}
    for r in results:
        vectors = model.encode([f"query: {u.statement}" for u in r.pred], normalize_embeddings=True, show_progress_bar=False)
        emb.update({u.unit_id: v for u, v in zip(r.pred, vectors)})
    return emb


def variant_relations(original: list[Relation], linked: list[Relation], allowed: set[frozenset[str]], mode: str) -> list[Relation]:
    kept = [r for r in linked if frozenset((r.from_id, r.to_id)) in allowed]
    if mode == "remplacement":
        return kept
    if mode == "complement":
        related = {frozenset((r.from_id, r.to_id)) for r in original}
        kept = [r for r in kept if frozenset((r.from_id, r.to_id)) not in related]
    seen = {(r.from_id, r.to_id, r.relation) for r in original}
    return list(original) + [r for r in kept if (r.from_id, r.to_id, r.relation) not in seen]


def relations_block(results: list, relations_by_project: dict[str, list[Relation]]) -> dict[str, object]:
    swapped = [dataclasses.replace(r, pred_rels=relations_by_project[r.project_id]) for r in results]
    return compute_metrics(swapped)["relations"]


def run(split: str, versions: tuple[str, ...], cache_only: bool) -> dict[str, object]:
    numbers, group = SPLITS[split]
    results = replay_extraction(numbers, group)
    emb = embed(results)
    client = make_link_client(cache_only=cache_only)
    pairs = {r.project_id: candidate_pairs(r.pred, r.pred_rels, emb, K_MAX) for r in results}
    original = {r.project_id: list(r.pred_rels) for r in results}
    linked: dict[str, dict[str, list[Relation]]] = {}
    per_project: dict[str, dict[str, object]] = {}
    costs: dict[str, dict[str, int]] = {}
    for version in versions:
        linked[version] = {}
        before = client.snapshot()
        for r in results:
            system, user = link_prompts(version, r.pred, pairs[r.project_id], original[r.project_id])
            if not cache_only and client.key(system, user) not in client._cache:  # noqa: SLF001
                used, _ = lot_tokens()
                if used + CALL_ESTIMATE_TOKENS > BUDGET_TOKENS:
                    sys.exit(f"BUDGET : {used} tokens deja consommes, un appel de plus risque de depasser {BUDGET_TOKENS} ; rien n'est ecrit")
                print(f"[{version} {r.project_id}] appel de liaison : {len(pairs[r.project_id])} paires", flush=True)
            result = link_project(r.project_id, r.pred, pairs[r.project_id], client, version=version, existing=original[r.project_id])
            linked[version][r.project_id] = result.relations
            per_project.setdefault(version, {})[r.project_id] = {  # type: ignore[union-attr]
                "units": len(r.pred), "candidate_pairs": len(result.pairs), "decisions": dict(result.decisions.most_common()),
                "rejections": dict(result.rejections.most_common()),
            }
        after = client.snapshot()
        d = {k: after[k] - before[k] for k in after}
        costs[version] = {"groq_calls": d["calls"] + d["cache_hits"], "api_calls_this_run": d["calls"],
                          "prompt_tokens": d["prompt_tokens"] + d["cached_prompt_tokens"],
                          "completion_tokens": d["completion_tokens"] + d["cached_completion_tokens"]}

    variants = {}
    for version in versions:
        for k in range(K_MAX + 1):
            allowed = {r.project_id: {frozenset(p) for p in candidate_pairs(r.pred, r.pred_rels, emb, k)} for r in results}
            for mode in MODES:
                rels = {pid: variant_relations(original[pid], linked[version][pid], allowed[pid], mode) for pid in original}
                variants[f"{version} k={k} {mode}"] = {
                    "prompt": version, "k": k, "mode": mode, "executed": k == K_MAX,
                    "nature": "configuration executee (un appel par projet sur ces paires)" if k == K_MAX
                    else f"indicative : filtrage a posteriori de la passe k={K_MAX}, pas une execution",
                    "relations": relations_block(results, rels), "_rels": rels,
                }
    annotated = sum(len(r.gold_rels) for r in results)
    reachable = sum(1 for r in results for g in r.gold_rels if g.from_id in set(r.mapping.values()) and g.to_id in set(r.mapping.values()))
    return {
        "results": results, "variants": variants, "original": original, "per_project": per_project, "cost": costs,
        "before": relations_block(results, original),
        "ceiling": {"annotated_relations": annotated, "both_endpoints_extracted": reachable,
                    "max_strict_recall": round(reachable / annotated, 4) if annotated else 0.0,
                    "definition": "une relation annotee ne peut etre retrouvee que si ses deux extremites sont appariees a des unites extraites"},
    }


def select(variants: dict[str, dict[str, object]]) -> str:
    def strict(name: str) -> dict[str, float]:
        return variants[name]["relations"]["strict"]  # type: ignore[index,return-value]

    executed = [n for n in variants if variants[n]["executed"]]
    eligible = [n for n in executed if strict(n)["precision"] >= PRECISION_MIN] or executed
    return min(eligible, key=lambda n: (-strict(n)["f1"], variants[n]["relations"]["extracted"]))  # type: ignore[index]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=list(SPLITS), required=True)
    parser.add_argument("--cache-only", action="store_true", help="n'appelle jamais Groq (cache de liaison seulement)")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    out: Path = args.out if args.out.is_absolute() else ROOT / args.out
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    if args.split == "test":
        if "dev" not in existing:
            sys.exit("Le test scelle se mesure apres le dev, avec la configuration retenue sur le dev.")
        if "test" in existing and not args.cache_only:
            sys.exit("Le test scelle est deja enregistre : execution unique, rien n'est relance.")
        chosen = existing["dev"]["selected"]["variant"]
        data = run("test", (existing["dev"]["selected"]["prompt"],), args.cache_only)
    else:
        data = run("dev", VERSIONS, args.cache_only)
        chosen = select(data["variants"])
    variants = data["variants"]
    after = variants[chosen]
    for name, v in variants.items():
        s = v["relations"]["strict"]
        print(f"{name:<24} {'executee  ' if v['executed'] else 'indicative'} predites {v['relations']['extracted']:3d}  "
              f"P {s['precision']:.3f}  R {s['recall']:.3f}  F1 {s['f1']:.3f}" + ("   <- retenue" if name == chosen else ""))
    before_s, after_s = data["before"]["strict"], after["relations"]["strict"]
    met = after_s["f1"] >= F1_MIN and after_s["precision"] >= PRECISION_MIN

    section: dict[str, object] = {
        "projects": [r.project_id for r in data["results"]],
        "status": ("reglage de la consigne de liaison, de k et du mode de fusion sur ces projets uniquement" if args.split == "dev"
                   else "execution unique d'enregistrement sur le test scelle, configuration retenue sur le dev, aucun reglage"),
        "before": data["before"],
        "after": after["relations"],
        "selected": {"variant": chosen, "prompt": after["prompt"], "k": after["k"], "mode": after["mode"], "executed": after["executed"]},
        "ceiling": data["ceiling"],
        "per_project": data["per_project"],
        "cost": data["cost"],
    }
    if args.split == "dev":
        section["criterion_met"] = met
        section["selection_rule"] = (f"parmi les configurations executees (consignes {', '.join(VERSIONS)}, k={K_MAX}, modes "
                                     f"{', '.join(MODES)}) : meilleur F1 strict a precision >= 0,45 ; les variantes k < {K_MAX} "
                                     "sont indicatives et ne peuvent pas etre retenues")
        n_executed = sum(v["executed"] for v in variants.values())
        section["caveats"] = {
            "complement_mode": "le mode de fusion complement a ete introduit apres l'analyse des resultats de la v1 sur le dev",
            "prompt_v2": "la v2 est la seule et derniere revision de la consigne de liaison ; aucune v3",
            "selection": f"choix parmi {n_executed} configurations executees sur le dev : le score dev de la configuration retenue est donc optimiste",
            "unbiased_measure": "seul le test scelle, execute une fois avec la configuration retenue, sert de mesure non biaisee",
        }
        section["best_per_prompt"] = {
            version: min(((n, v) for n, v in variants.items() if v["executed"] and v["prompt"] == version),
                         key=lambda item: (-item[1]["relations"]["strict"]["f1"], item[1]["relations"]["extracted"]))[0]
            for version in VERSIONS
        }
        section["variants"] = {n: {k: v[k] for k in ("prompt", "k", "mode", "executed", "nature", "relations")} for n, v in variants.items()}
        analysis = classify([dataclasses.replace(r, pred_rels=after["_rels"][r.project_id]) for r in data["results"]])
        reference = json.loads(RELATION_ERRORS_JSON.read_text(encoding="utf-8"))
        section["error_analysis"] = {
            "categories": "memes categories que eval/analyze_relation_errors.py",
            "before": {"by_type": reference["by_type"], "totals": reference["totals"]},
            "after": {"by_type": analysis, "totals": totals(analysis)},
        }
    else:
        section["criterion_met_on_dev"] = existing["dev"]["criterion_met"]

    used, calls = lot_tokens()
    payload = existing or {}
    payload.update({
        "experiment": "relations v2 : seconde passe de liaison sur les unites extraites",
        "generated_on": date.today().isoformat(),
        "script": "eval/eval_relations_v2.py",
        "link": {
            "module": "src/ingest/link.py", "model": link_module.LINK_MODEL, "temperature": link_module.LINK_TEMPERATURE,
            "json_mode": True, "extra": link_module.LINK_EXTRA, "max_completion_tokens": link_module.LINK_MAX_COMPLETION_TOKENS,
            "prompts_sha256": {v: results_io.sha256_params({"prompt": p}) for v, p in LINK_PROMPTS.items()},
            "prompt_history": {"v1": "premiere consigne", "v2": "seule revision, apres analyse des erreurs de la v1 sur le dev 04-10"},
            "candidates": f"k plus proches anterieures (cosinus e5 fine-tune, meme projet) + paires deja reliees ; appel sur k={K_MAX}",
            "calls": "un appel par projet et par version de consigne", "tuned_on": [f"projet_{n:02d}" for n in SPLITS["dev"][0]],
        },
        "extraction": {"system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                       "source": "rejeu du cache d'extraction, 0 appel, verifie contre extraction.json", "unchanged": True},
        "criterion": {"rule": "dev 04-10 : F1 strict des relations >= 0,456 (0,406 + 0,05) et precision stricte >= 0,45",
                      "f1_min": F1_MIN, "precision_min": PRECISION_MIN, "fixed_before_results": True},
        "budget": {"limit_tokens": BUDGET_TOKENS, "lot_tokens_consumed": used, "lot_groq_calls": calls,
                   "source": "usage enregistre de chaque appel dans data/processed/groq_link_cache.jsonl"},
    })
    payload[args.split] = section
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"avant : P {before_s['precision']:.3f} R {before_s['recall']:.3f} F1 {before_s['f1']:.3f} ; "
          f"apres ({chosen}) : P {after_s['precision']:.3f} R {after_s['recall']:.3f} F1 {after_s['f1']:.3f} ; "
          f"critere {'atteint' if met else 'NON atteint'} ; lot : {used} tokens, {calls} appels")
    print(f"Resultats ecrits dans {out}")


if __name__ == "__main__":
    main()

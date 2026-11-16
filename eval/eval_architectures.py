"""Comparaison d'architectures sur le corpus interne : zero-shot et fine-tune, meme protocole d'evaluation.

Modeles : e5-small (production, models/e5-finetuned), e5-small sans negatifs difficiles (ablation contrastive),
e5-base et paraphrase-multilingual-MiniLM-L12-v2 (scripts/train_architectures.py). Chaque modele est interroge avec
ses propres conventions d'entree (prefixes e5, aucun prefixe pour MiniLM). Memes requetes, memes metriques et meme
filtre que eval/run_baselines.py ; recherche dense dans les unites du projet de la requete.

Regle de choix (fixee avant les resultats, dev seulement) : le modele de production ne change que si un candidat
fine-tune bat e5-small fine-tune sur le MRR dev, sans recul de plus de 0,1 du MRR sur aucune categorie, ET que
verify_demo et le parcours Atlas restent identiques ; sinon e5-small reste.

    uv run python eval/eval_architectures.py --out eval/results/architectures_dev.json
    uv run python eval/eval_architectures.py --queries data/eval/queries_test.json --out eval/results/architectures_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import results_io  # noqa: E402
from eval.eval_hybrid import DEV_STATUS, TEST_STATUS  # noqa: E402
from eval.run_baselines import DATA_DIR, METRICS, QUERIES_PATH, TOP_K, TYPES, score_query  # noqa: E402
from scripts.train_architectures import ARCH_DIR, CONFIGS, E5_PREFIXES, NO_PREFIXES  # noqa: E402
from src.memory.corpus import load_corpus, load_queries  # noqa: E402

PRODUCTION = ROOT / "models" / "e5-finetuned"
MAX_CATEGORY_DROP = 0.1
# nom -> (modele ou dossier, prefixes, configuration d'entrainement ou None pour un zero-shot)
MODELS: dict[str, tuple[str, tuple[str, str], str | None]] = {
    "e5_small_base": ("intfloat/multilingual-e5-small", E5_PREFIXES, None),
    "e5_small_finetuned": (str(PRODUCTION), E5_PREFIXES, "production"),
    "e5_small_sans_negatifs_durs": (str(ARCH_DIR / "e5_small_sans_negatifs_durs"), E5_PREFIXES, "e5_small_sans_negatifs_durs"),
    "e5_base_base": ("intfloat/multilingual-e5-base", E5_PREFIXES, None),
    "e5_base_finetuned": (str(ARCH_DIR / "e5_base"), E5_PREFIXES, "e5_base"),
    "minilm_l12_base": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", NO_PREFIXES, None),
    "minilm_l12_finetuned": (str(ARCH_DIR / "minilm_l12"), NO_PREFIXES, "minilm_l12"),
}
CANDIDATES = ["e5_small_sans_negatifs_durs", "e5_base_finetuned", "minilm_l12_finetuned"]


def size_info(model, name_or_path: str) -> dict[str, object]:  # noqa: ANN001
    params = sum(p.numel() for p in model.parameters())
    path = Path(name_or_path)
    if not path.exists():  # modele du cache Hugging Face
        cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{name_or_path.replace('/', '--')}"
        path = cache if cache.exists() else path
    weights = [p for p in path.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".bin") and "onnx" not in p.parts and "openvino" not in p.parts]
    return {"parameters": params, "weights_mb": round(sum(p.stat().st_size for p in weights) / 2**20, 1) if weights else None}


def evaluate(name: str, queries: list[dict[str, object]]) -> dict[str, object]:
    from sentence_transformers import SentenceTransformer

    source, (q_prefix, p_prefix), training = MODELS[name]
    model = SentenceTransformer(source, device="cpu")
    corpus = load_corpus(DATA_DIR)
    grouped: dict[str, list] = defaultdict(list)
    for item in corpus:
        grouped[item.project_id].append(item)
    index = {
        pid: ([i.unit.unit_id for i in items], model.encode([f"{p_prefix}{i.text}" for i in items], normalize_embeddings=True, show_progress_bar=False))
        for pid, items in grouped.items()
    }
    texts = [f"{q_prefix}{q['query']}" for q in queries]
    model.encode(texts[:2], normalize_embeddings=True, show_progress_bar=False)  # chauffe
    start = time.perf_counter()
    q_emb = np.vstack([model.encode([t], normalize_embeddings=True, show_progress_bar=False) for t in texts])
    latency_ms = (time.perf_counter() - start) / len(texts) * 1000
    scores: dict[str, list[list[float]]] = defaultdict(list)
    for query, emb in zip(queries, q_emb):
        ids, mat = index[str(query["project_id"])]
        order = np.argsort(-(mat @ emb), kind="stable")[:TOP_K]
        row = score_query([ids[i] for i in order], set(query["relevant_unit_ids"]))  # type: ignore[arg-type]
        scores["all"].append(row)
        scores[str(query["piege"])].append(row)
    log_path = ARCH_DIR / str(training) / "training_log.json" if training and training != "production" else None
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path and log_path.exists() else None
    return {
        "model": source if not Path(source).exists() else Path(source).relative_to(ROOT).as_posix(),
        "prefixes": {"query": q_prefix, "passage": p_prefix},
        "fine_tuned": training is not None,
        "training": log if log else ({"source": "scripts/train_embeddings.py", **results_io.training_config(), "train_seconds": None,
                                      "train_seconds_note": "duree non enregistree lors de l'entrainement d'origine"}
                                     if training == "production" else None),
        "size": size_info(model, source),
        "encode_latency_ms_per_query": round(latency_ms, 2),
        "global": results_io.metric_block(scores["all"], METRICS),
        "per_category": {p: results_io.metric_block(scores[p], METRICS) for p in TYPES if scores[p]},
    }


def choose(results: dict[str, dict[str, object]]) -> dict[str, object]:
    ref = results["e5_small_finetuned"]
    verdicts = {}
    for name in CANDIDATES:
        cand = results[name]
        better = cand["global"]["MRR"] > ref["global"]["MRR"]  # type: ignore[index]
        drops = {p: round(ref["per_category"][p]["MRR"] - cand["per_category"][p]["MRR"], 4) for p in ref["per_category"]}  # type: ignore[index]
        worst = max(drops.values())
        verdicts[name] = {"mrr": cand["global"]["MRR"], "beats_production_mrr": better, "max_category_mrr_drop": worst,  # type: ignore[index]
                          "eligible": better and worst <= MAX_CATEGORY_DROP}
    eligible = [n for n, v in verdicts.items() if v["eligible"] and n != "e5_small_sans_negatifs_durs"]
    return {
        "rule": "un candidat fine-tune remplace e5-small seulement s'il bat son MRR dev sans recul de plus de 0,1 du MRR "
        "sur aucune categorie, et si verify_demo et le parcours Atlas restent identiques",
        "reference": {"name": "e5_small_finetuned", "mrr": ref["global"]["MRR"]},  # type: ignore[index]
        "candidates": verdicts,
        "eligible_architectures": eligible,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    queries_path: Path = args.queries if args.queries.is_absolute() else ROOT / args.queries
    is_test = queries_path.name == "queries_test.json"
    all_queries = load_queries(queries_path)
    queries = [q for q in all_queries if q["relevant_unit_ids"] and q["piege"] != "sans_reponse"]

    results = {}
    for name in MODELS:
        results[name] = evaluate(name, queries)
        g = results[name]["global"]
        print(f"{name:<28} " + "  ".join(f"{m} {g[m]:.3f}" for m in METRICS)  # type: ignore[index]
              + f"  latence {results[name]['encode_latency_ms_per_query']} ms", flush=True)
    data = {
        "description": "Comparaison d'architectures d'embeddings (zero-shot et fine-tunees), recherche dense",
        "generated_on": date.today().isoformat(),
        "split": "test" if is_test else "dev",
        "script": "eval/eval_architectures.py",
        "status": TEST_STATUS if is_test else DEV_STATUS + " ; regle de choix fixee avant les resultats",
        "queries": results_io.query_set_info(queries_path, all_queries, queries),
        "protocol": {"training": "scripts/train_architectures.py (meme protocole que scripts/train_embeddings.py)",
                     "configs": {k: {"base": v[0], "hard_negatives": v[2]} for k, v in CONFIGS.items()},
                     "top_k": TOP_K, "metrics": METRICS, "search": "dense seule, unites du projet de la requete",
                     "latency": "encodage d'une requete a la fois, CPU, moyenne sur les requetes evaluees"},
        "results": results,
    }
    if not is_test:
        data["choice"] = choose(results)
        print(json.dumps(data["choice"], ensure_ascii=False, indent=1))
    out: Path = args.out if args.out.is_absolute() else ROOT / args.out
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultats ecrits dans {out}")


if __name__ == "__main__":
    main()

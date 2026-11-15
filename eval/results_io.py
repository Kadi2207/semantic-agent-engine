"""Enregistrement tracable des resultats de recherche (run_baselines.py, compare_finetuned.py) en JSON.

Ce module ne calcule aucune metrique : il recoit les scores par requete deja calcules par les scripts
d'evaluation et les ecrit avec leurs metadonnees (corpus, split, requetes, modeles, configuration,
empreintes SHA-256). Une valeur qui n'a pas de source (ex. k RRF dans un script sans fusion) est
enregistree a null avec une explication, jamais inventee.

Un fichier de resultats regroupe plusieurs experiences sur le meme jeu de requetes, une section par
script ("experiments": {"run_baselines": ..., "compare_finetuned": ...}).
"""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = ROOT / "scripts" / "train_embeddings.py"
TRAINING_KEYS = ("BASE_MODEL", "EPOCHS", "BATCH_SIZE", "WARMUP_RATIO", "LEARNING_RATE", "SEED")
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Prefixes e5 : src/retrieval/dense_baseline.py les ecrit en dur (f"query: ...", f"passage: ...") sans constante
# exportee. Ils sont donc DECLARES ici, une seule fois, et un test verifie qu'ils figurent bien dans ce fichier.
E5_PREFIXES = {"query": "query: ", "passage": "passage: "}
E5_PREFIXES_NOTE = "declares ; verifies par test contre src/retrieval/dense_baseline.py"
FINGERPRINT_COVERS = (
    "params_sha256 = SHA-256 du JSON canonique (cles triees, sans espaces) de {'params': params, 'training': "
    "{modele: parametres d'entrainement}} ; params contient top_k, metrics, prefixes e5, portee de recherche, rrf_k "
    "et seed d'evaluation ; training contient, pour chaque modele fine-tune, les constantes lues dans "
    "scripts/train_embeddings.py (base_model, epochs, batch_size, warmup_ratio, learning_rate, seed)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


TEXT_SHA256_NORMALISATION = (
    "fins de ligne normalisees en LF avant calcul (CRLF et CR convertis en LF) : l'empreinte ne depend pas de "
    "core.autocrlf ni du systeme d'exploitation de la copie du depot"
)


def normalized_text_bytes(data: bytes) -> bytes:
    """Contenu texte avec fins de ligne normalisees : CRLF -> LF, puis CR isole -> LF."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def sha256_text_file(path: Path) -> str:
    """Empreinte d'un fichier TEXTE, independante des fins de ligne (ne pas utiliser pour un binaire)."""
    return hashlib.sha256(normalized_text_bytes(path.read_bytes())).hexdigest()


def sha256_params(params: dict[str, object]) -> str:
    """Empreinte de la configuration (JSON canonique : cles triees, sans espaces)."""
    canonical = json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def retrieval_params(top_k: int, metrics: list[str]) -> dict[str, object]:
    """Parametres d'evaluation communs a run_baselines.py et compare_finetuned.py (constantes passees par le code)."""
    return {
        "top_k": top_k,
        "metrics": list(metrics),
        "search_scope": "unites du projet de la requete",
        "e5_prefixes": dict(E5_PREFIXES),
        "e5_prefixes_note": E5_PREFIXES_NOTE,
        "rrf_k": None,
        "rrf_k_note": "aucune fusion RRF dans ce script",
        "evaluation_seed": None,
        "evaluation_seed_note": "evaluation deterministe, aucun tirage aleatoire",
    }


def fingerprint_input(params: dict[str, object], models: dict[str, dict[str, object]]) -> dict[str, object]:
    """Structure couverte par l'empreinte : params + parametres d'entrainement des modeles fine-tunes."""
    training = {name: info["training"] for name, info in sorted(models.items()) if info.get("training")}
    return {"params": params, "training": training}


def params_fingerprint(params: dict[str, object], models: dict[str, dict[str, object]]) -> str:
    return sha256_params(fingerprint_input(params, models))


def training_config() -> dict[str, object]:
    """Constantes d'entrainement lues dans scripts/train_embeddings.py (sans l'importer ni l'executer)."""
    values: dict[str, object] = {}
    for node in ast.parse(TRAIN_SCRIPT.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in TRAINING_KEYS:
                values[name.lower()] = ast.literal_eval(node.value)
    return {"source": "scripts/train_embeddings.py", **values}


def hf_revision(model_name: str) -> str | None:
    """Revision (commit) du modele Hugging Face en cache local, ou None si introuvable."""
    ref = HF_CACHE / f"models--{model_name.replace('/', '--')}" / "refs" / "main"
    return ref.read_text(encoding="utf-8").strip() if ref.exists() else None


def model_info(name_or_path: str | Path) -> dict[str, object]:
    path = Path(name_or_path)
    if path.exists():
        weights = path / "model.safetensors"
        return {
            "model": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
            "weights_sha256": sha256_file(weights) if weights.exists() else None,
            "training": training_config(),
        }
    revision = hf_revision(str(name_or_path))
    return {
        "model": str(name_or_path),
        "huggingface_revision": revision,
        "huggingface_revision_note": None if revision else "revision du cache local introuvable",
        "training": None,
    }


def metric_block(rows: list[list[float]], metric_names: list[str]) -> dict[str, object]:
    """Moyenne par metrique de scores par requete deja calcules (meme moyenne que les scripts)."""
    means = [sum(column) / len(rows) for column in zip(*rows)]
    return {"n": len(rows), **dict(zip(metric_names, means))}


def query_set_info(queries_path: Path, all_queries: list[dict[str, object]], evaluated: list[dict[str, object]]) -> dict[str, object]:
    evaluated_ids = {str(q["query_id"]) for q in evaluated}
    return {
        "file": queries_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_text_file(queries_path),
        "sha256_normalisation": TEXT_SHA256_NORMALISATION,
        "total": len(all_queries),
        "evaluated": len(evaluated),
        "excluded": [
            {"query_id": q["query_id"], "reason": "piege sans_reponse (aucune unite pertinente)"}
            for q in all_queries
            if str(q["query_id"]) not in evaluated_ids
        ],
        "projects": sorted({str(q["project_id"]) for q in all_queries}),
    }


def check_expected(observed: dict[str, float], expected: list[str]) -> list[str]:
    """Compare des valeurs publiees ("cle=valeur", arrondi a 3 decimales) ; renvoie la liste des ecarts."""
    problems = []
    for item in expected:
        key, _, value = item.partition("=")
        if key not in observed:
            problems.append(f"{key} : valeur attendue {value}, cle absente des resultats")
        elif f"{observed[key]:.3f}" != f"{float(value):.3f}":
            problems.append(f"{key} : publie {float(value):.3f}, obtenu {observed[key]:.3f}")
    return problems


def write_experiment(out: Path, section: str, experiment: dict[str, object], description: str) -> None:
    """Ajoute ou remplace la section `section` du fichier, en verifiant qu'il porte sur le meme jeu de requetes."""
    data: dict[str, object] = {"description": description, "experiments": {}}
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
        for name, other in data.get("experiments", {}).items():  # type: ignore[union-attr]
            if name != section and other["queries"]["sha256"] != experiment["queries"]["sha256"]:  # type: ignore[index]
                raise ValueError(f"{out} contient deja une experience sur un autre fichier de requetes ({name})")
    experiment = {"generated_on": date.today().isoformat(), **experiment}
    experiment["params_sha256"] = params_fingerprint(experiment["params"], experiment["models"])  # type: ignore[arg-type]
    experiment["params_sha256_covers"] = FINGERPRINT_COVERS
    data["experiments"][section] = experiment  # type: ignore[index]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


METADATA_ONLY_MENTION = "metadonnees mises a jour apres controle, valeurs inchangees"


def update_metadata(
    out: Path,
    section: str,
    params: dict[str, object],
    models: dict[str, dict[str, object]],
) -> tuple[bytes, bytes]:
    """Met a jour params, modeles et empreinte d'une section EXISTANTE, sans aucune evaluation.

    La section "results" n'est jamais reconstruite : elle est relue puis reecrite telle quelle. Renvoie sa
    serialisation avant et apres ; l'appelant verifie qu'elles sont identiques octet pour octet.
    """
    data = json.loads(out.read_text(encoding="utf-8"))
    experiment = data["experiments"][section]
    before = json.dumps(experiment["results"], ensure_ascii=False, indent=2).encode("utf-8")
    experiment["params"] = params
    experiment["models"] = models
    experiment["params_sha256"] = params_fingerprint(params, models)
    experiment["params_sha256_covers"] = FINGERPRINT_COVERS
    experiment["metadata_updated_on"] = date.today().isoformat()
    if METADATA_ONLY_MENTION not in experiment["status"]:
        experiment["status"] = f"{experiment['status']} ; {METADATA_ONLY_MENTION}"
    after = json.dumps(experiment["results"], ensure_ascii=False, indent=2).encode("utf-8")
    if before != after:
        raise RuntimeError("la section results a change : rien n'est ecrit")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return before, after


def refresh_query_fingerprints(out: Path) -> list[tuple[str, bool]]:
    """Recalcule l'empreinte (normalisee) du fichier de requetes de chaque section, sans aucune evaluation.

    Seuls les champs queries.sha256, queries.sha256_normalisation et metadata_updated_on changent ; la section
    "results" est relue puis reecrite telle quelle. Renvoie, par section, si results est identique octet pour octet.
    """
    data = json.loads(out.read_text(encoding="utf-8"))
    checks = []
    for name, experiment in data["experiments"].items():
        before = json.dumps(experiment["results"], ensure_ascii=False, indent=2).encode("utf-8")
        queries = experiment["queries"]
        queries["sha256"] = sha256_text_file(ROOT / queries["file"])
        queries["sha256_normalisation"] = TEXT_SHA256_NORMALISATION
        experiment["metadata_updated_on"] = date.today().isoformat()
        after = json.dumps(experiment["results"], ensure_ascii=False, indent=2).encode("utf-8")
        checks.append((name, before == after))
    if not all(same for _, same in checks):
        raise RuntimeError("une section results a change : rien n'est ecrit")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return checks


def main() -> None:
    """Mise a jour des metadonnees de fichiers de resultats existants, sans nouvelle execution d'evaluation."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=main.__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--refresh-test", type=Path, help="params, modeles et empreinte de la section test existante")
    group.add_argument("--refresh-query-fingerprints", type=Path, nargs="+", help="empreintes normalisees des requetes")
    args = parser.parse_args()
    if args.refresh_query_fingerprints:
        for path in args.refresh_query_fingerprints:
            checks = refresh_query_fingerprints(path)
            print(f"{path} : " + ", ".join(f"{name} results identique={same}" for name, same in checks))
        return
    sys.path.insert(0, str(ROOT))
    # Constantes reelles du code d'evaluation (import sans executer d'evaluation ni charger de modele).
    from eval.compare_finetuned import FINETUNED_PATH, MODEL_NAME
    from eval.run_baselines import METRICS, TOP_K

    raw_before = json.loads(args.refresh_test.read_text(encoding="utf-8"))["experiments"]["compare_finetuned"]["results"]
    before, after = update_metadata(
        args.refresh_test,
        "compare_finetuned",
        retrieval_params(TOP_K, METRICS),
        {"e5_base": model_info(MODEL_NAME), "e5_finetuned": model_info(FINETUNED_PATH)},
    )
    raw_after = json.loads(args.refresh_test.read_text(encoding="utf-8"))["experiments"]["compare_finetuned"]["results"]
    same = before == after and raw_before == raw_after
    print(f"section results identique octet pour octet : {before == after} ({len(before)} octets) ; objets egaux : {raw_before == raw_after}")
    if not same:
        sys.exit("ERREUR : valeurs modifiees")


if __name__ == "__main__":
    main()

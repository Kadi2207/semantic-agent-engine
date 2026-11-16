"""Tracabilite des resultats de recherche : eval/results/retrieval_dev.json et retrieval_test.json.

Ces tests ne recalculent rien (aucun modele charge) : ils verifient le format, les metadonnees, la coherence
des nombres de requetes, l'egalite exacte avec les chiffres publies dans le README, et qu'aucune valeur
sans source n'a ete inventee.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

from eval.results_io import (
    E5_PREFIXES,
    E5_PREFIXES_NOTE,
    METADATA_ONLY_MENTION,
    TEXT_SHA256_NORMALISATION,
    sha256_text_file,
    params_fingerprint,
    retrieval_params,
    sha256_params,
    training_config,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = {"dev": ROOT / "eval/results/retrieval_dev.json", "test": ROOT / "eval/results/retrieval_test.json"}
QUERIES = {"dev": ROOT / "data/eval/queries_dev.json", "test": ROOT / "data/eval/queries_test.json"}
EXPECTED_SECTIONS = {"dev": {"run_baselines", "compare_finetuned", "hybrid_rrf"}, "test": {"compare_finetuned", "hybrid_rrf"}}
EXPERIMENT_KEYS = {"experiment", "generated_on", "corpus", "split", "script", "status", "queries", "models", "params", "params_sha256", "results"}
QUERY_KEYS = {"file", "sha256", "total", "evaluated", "excluded", "projects"}
METRICS = ["Recall@3", "Recall@5", "MRR", "nDCG@5"]

# Chiffres publies dans le README (section Evaluation), (split, section, systeme, categorie, metrique) -> valeur
PUBLISHED = {
    ("dev", "run_baselines", "bm25", "global", "Recall@3"): 0.800,
    ("dev", "run_baselines", "bm25", "global", "MRR"): 0.631,
    ("dev", "compare_finetuned", "e5_base", "global", "Recall@3"): 0.680,
    ("dev", "compare_finetuned", "e5_base", "global", "Recall@5"): 0.880,
    ("dev", "compare_finetuned", "e5_base", "global", "MRR"): 0.526,
    ("dev", "compare_finetuned", "e5_base", "global", "nDCG@5"): 0.614,
    ("dev", "compare_finetuned", "e5_finetuned", "global", "Recall@3"): 0.840,
    ("dev", "compare_finetuned", "e5_finetuned", "global", "Recall@5"): 1.000,
    ("dev", "compare_finetuned", "e5_finetuned", "global", "MRR"): 0.716,
    ("dev", "compare_finetuned", "e5_finetuned", "global", "nDCG@5"): 0.786,
    ("dev", "run_baselines", "bm25", "global", "Recall@5"): 0.880,
    ("dev", "run_baselines", "bm25", "global", "nDCG@5"): 0.695,
    ("dev", "compare_finetuned", "e5_base", "negation", "MRR"): 0.207,
    ("dev", "compare_finetuned", "e5_finetuned", "negation", "MRR"): 0.507,
    ("dev", "compare_finetuned", "e5_base", "engagement", "MRR"): 0.550,
    ("dev", "compare_finetuned", "e5_finetuned", "engagement", "MRR"): 1.000,
    ("dev", "compare_finetuned", "e5_base", "decision_active", "MRR"): 0.800,
    ("dev", "compare_finetuned", "e5_finetuned", "decision_active", "MRR"): 0.707,
    ("test", "compare_finetuned", "e5_base", "global", "MRR"): 0.640,
    ("test", "compare_finetuned", "e5_finetuned", "global", "MRR"): 0.750,
}


def load(split: str) -> dict:
    return json.loads(RESULTS[split].read_text(encoding="utf-8"))


def queries(split: str) -> list[dict]:
    return json.loads(QUERIES[split].read_text(encoding="utf-8"))["queries"]


def value(split: str, section: str, system: str, category: str, metric: str) -> float:
    result = load(split)["experiments"][section]["results"][system]
    block = result["global"] if category == "global" else result["per_category"][category]
    return block[metric]


@pytest.mark.parametrize("split", ["dev", "test"])
def test_format_and_metadata_are_complete(split: str) -> None:
    data = load(split)
    assert set(data["experiments"]) == EXPECTED_SECTIONS[split]
    for name, experiment in data["experiments"].items():
        assert EXPERIMENT_KEYS <= set(experiment), name
        assert QUERY_KEYS <= set(experiment["queries"]), name
        assert experiment["split"] == split
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", experiment["generated_on"])
        assert (ROOT / experiment["script"]).exists()
        assert experiment["params"]["metrics"] == METRICS
        assert experiment["params_sha256"] == params_fingerprint(experiment["params"], experiment["models"])
        assert "training" in experiment["params_sha256_covers"] and "seed" in experiment["params_sha256_covers"]


def test_test_file_is_marked_as_historical_rerecording() -> None:
    status = load("test")["experiments"]["compare_finetuned"]["status"]
    assert "re-execution d'enregistrement de resultats historiques" in status
    assert "aucun reglage effectue" in status


@pytest.mark.parametrize("split, total, evaluated", [("dev", 30, 25), ("test", 6, 5)])
def test_query_counts_and_fingerprints_match_the_query_files(split: str, total: int, evaluated: int) -> None:
    raw = queries(split)
    expected_excluded = sorted(q["query_id"] for q in raw if not q["relevant_unit_ids"] or q["piege"] == "sans_reponse")
    # normalisation des fins de ligne faite ici independamment de results_io (CRLF puis CR -> LF)
    sha = hashlib.sha256(QUERIES[split].read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
    for experiment in load(split)["experiments"].values():
        info = experiment["queries"]
        assert info["sha256"] == sha  # le fichier de requetes n'a pas change depuis l'enregistrement
        assert info["sha256_normalisation"] == TEXT_SHA256_NORMALISATION
        assert (info["total"], info["evaluated"]) == (len(raw), len(raw) - len(expected_excluded)) == (total, evaluated)
        assert sorted(x["query_id"] for x in info["excluded"]) == expected_excluded
        assert info["projects"] == sorted({q["project_id"] for q in raw})
        for result in experiment["results"].values():
            assert result["global"]["n"] == evaluated
            assert sum(block["n"] for block in result["per_category"].values()) == evaluated


@pytest.mark.parametrize("split", ["dev", "test"])
def test_no_invented_category_or_value(split: str) -> None:
    pieges = {q["piege"] for q in queries(split) if q["relevant_unit_ids"] and q["piege"] != "sans_reponse"}
    for section, experiment in load(split)["experiments"].items():
        params = experiment["params"]
        if section == "hybrid_rrf":  # seule section avec fusion : le k vient de src/agent/retriever.py
            from src.agent.retriever import RRF_K

            assert params["rrf_k"] == RRF_K == 60 and "src/agent/retriever.py" in params["rrf_k_note"]
        else:
            assert params["rrf_k"] is None and params["rrf_k_note"]  # pas de fusion RRF dans ces scripts
        assert params["evaluation_seed"] is None and params["evaluation_seed_note"]
        for result in experiment["results"].values():
            assert set(result["per_category"]) == pieges  # seulement les categories presentes dans les requetes
            for block in [result["global"], *result["per_category"].values()]:
                assert block["n"] > 0
                assert set(block) == {"n", *METRICS}
                assert all(isinstance(block[m], float) and math.isfinite(block[m]) and 0.0 <= block[m] <= 1.0 for m in METRICS)


def test_model_metadata_comes_from_real_sources() -> None:
    for split in ("dev", "test"):
        models = load(split)["experiments"]["compare_finetuned"]["models"]
        finetuned = models["e5_finetuned"]
        assert finetuned["model"] == "models/e5-finetuned"
        assert finetuned["weights_sha256"] is None or re.fullmatch(r"[0-9a-f]{64}", finetuned["weights_sha256"])
        assert finetuned["training"] == training_config()  # seed, epochs... lus dans scripts/train_embeddings.py
        assert finetuned["training"]["seed"] == 42
        base = models["e5_base"]
        assert base["training"] is None
        assert base["huggingface_revision"] or base["huggingface_revision_note"]
    assert load("dev")["experiments"]["run_baselines"]["models"]["bm25"]["training"] is None


def test_published_readme_values_are_exactly_the_recorded_ones() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for key, published in PUBLISHED.items():
        french = f"{published:.3f}".replace(".", ",")  # le README ecrit les decimales a la francaise
        assert french in readme, f"{key} : {french} absent du README"
        assert f"{value(*key):.3f}" == f"{published:.3f}", key


@pytest.mark.parametrize("split", ["dev", "test"])
def test_fingerprint_is_recomputed_from_code_constants_not_from_the_json(split: str) -> None:
    """(a) L'empreinte attendue est reconstruite a partir des constantes importees du code d'evaluation."""
    from eval.run_baselines import METRICS as CODE_METRICS
    from eval.run_baselines import TOP_K as CODE_TOP_K

    from eval.eval_hybrid import hybrid_params

    for section, experiment in load(split)["experiments"].items():
        expected_params = hybrid_params() if section == "hybrid_rrf" else retrieval_params(CODE_TOP_K, CODE_METRICS)
        training = {"e5_finetuned": training_config()} if section in ("compare_finetuned", "hybrid_rrf") else {}
        expected = sha256_params({"params": expected_params, "training": training})
        assert experiment["params"] == expected_params, section
        assert experiment["params_sha256"] == expected, section


def test_declared_e5_prefixes_are_those_of_dense_baseline() -> None:
    """(b) Les prefixes declares sont bien ceux ecrits dans src/retrieval/dense_baseline.py."""
    source = (ROOT / "src/retrieval/dense_baseline.py").read_text(encoding="utf-8")
    assert E5_PREFIXES == {"query": "query: ", "passage": "passage: "}
    for prefix in E5_PREFIXES.values():
        assert f'f"{prefix}{{' in source, prefix
    for split in ("dev", "test"):
        for experiment in load(split)["experiments"].values():
            assert experiment["params"]["e5_prefixes"] == E5_PREFIXES
            assert experiment["params"]["e5_prefixes_note"] == E5_PREFIXES_NOTE


def test_sealed_test_values_unchanged_after_metadata_update() -> None:
    """(c) Les valeurs du test restent celles publiees (MRR 0.640 puis 0.750) ; seule la metadonnee a change."""
    experiment = load("test")["experiments"]["compare_finetuned"]
    assert f"{experiment['results']['e5_base']['global']['MRR']:.3f}" == "0.640"
    assert f"{experiment['results']['e5_finetuned']['global']['MRR']:.3f}" == "0.750"
    assert METADATA_ONLY_MENTION in experiment["status"]
    assert experiment["generated_on"] <= experiment["metadata_updated_on"]


def test_text_fingerprint_is_the_same_with_lf_crlf_and_cr_line_endings(tmp_path: Path) -> None:
    """Le meme fichier donne la meme empreinte quelles que soient ses fins de ligne (core.autocrlf, OS)."""
    content = QUERIES["test"].read_bytes().replace(b"\r\n", b"\n")  # contenu reel des requetes test, en LF
    variants = {"lf": content, "crlf": content.replace(b"\n", b"\r\n"), "cr": content.replace(b"\n", b"\r")}
    assert len(set(variants.values())) == 3  # trois fichiers differents octet pour octet
    digests = set()
    for name, data in variants.items():
        path = tmp_path / f"queries_{name}.json"
        path.write_bytes(data)
        digests.add(sha256_text_file(path))
    assert digests == {load("test")["experiments"]["compare_finetuned"]["queries"]["sha256"]}
    other = tmp_path / "other.json"
    other.write_bytes(content.replace(b"q_test_001", b"q_test_999"))
    assert sha256_text_file(other) not in digests  # un vrai changement de contenu change l'empreinte


def test_shared_baseline_is_identical_in_both_dev_sections() -> None:
    experiments = load("dev")["experiments"]
    assert experiments["run_baselines"]["results"]["e5_base"] == experiments["compare_finetuned"]["results"]["e5_base"]

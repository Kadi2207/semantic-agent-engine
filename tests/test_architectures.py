"""Comparaison d'architectures : coherence des resultats enregistres, protocole identique, regle de choix appliquee."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV = json.loads((ROOT / "eval/results/architectures_dev.json").read_text(encoding="utf-8"))
TEST = json.loads((ROOT / "eval/results/architectures_test.json").read_text(encoding="utf-8"))
RETRIEVAL_DEV = json.loads((ROOT / "eval/results/retrieval_dev.json").read_text(encoding="utf-8"))
MODELS = {"e5_small_base", "e5_small_finetuned", "e5_small_sans_negatifs_durs", "e5_base_base", "e5_base_finetuned",
          "minilm_l12_base", "minilm_l12_finetuned"}


def test_all_models_are_evaluated_on_the_same_queries() -> None:
    for data, n in ((DEV, 25), (TEST, 5)):
        assert set(data["results"]) == MODELS
        for name, r in data["results"].items():
            assert r["global"]["n"] == n == data["queries"]["evaluated"], name
            assert sum(b["n"] for b in r["per_category"].values()) == n
            assert r["encode_latency_ms_per_query"] > 0 and r["size"]["parameters"] > 0


def test_e5_small_numbers_reproduce_the_published_dense_evaluation() -> None:
    """Meme modele, meme protocole : les chiffres de e5-small doivent egaler ceux de compare_finetuned."""
    published = RETRIEVAL_DEV["experiments"]["compare_finetuned"]["results"]
    for ours, theirs in (("e5_small_base", "e5_base"), ("e5_small_finetuned", "e5_finetuned")):
        for metric in ("Recall@3", "Recall@5", "MRR", "nDCG@5"):
            assert round(DEV["results"][ours]["global"][metric], 6) == round(published[theirs]["global"][metric], 6), (ours, metric)


def test_same_training_protocol_for_every_fine_tuned_model() -> None:
    from scripts.train_embeddings import BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED, WARMUP_RATIO

    for name in ("e5_small_sans_negatifs_durs", "e5_base_finetuned", "minilm_l12_finetuned"):
        log = DEV["results"][name]["training"]
        assert (log["epochs"], log["batch_size"], log["learning_rate"], log["warmup_ratio"], log["seed"]) == (EPOCHS, BATCH_SIZE, LEARNING_RATE, WARMUP_RATIO, SEED)
        assert log["loss"] == "MultipleNegativesRankingLoss" and log["examples"] == 371 and log["train_seconds"] > 0
    assert DEV["results"]["e5_small_sans_negatifs_durs"]["training"]["hard_negatives"] is False
    assert DEV["results"]["minilm_l12_finetuned"]["prefixes"] == {"query": "", "passage": ""}
    assert DEV["results"]["e5_base_finetuned"]["prefixes"] == {"query": "query: ", "passage": "passage: "}


def test_choice_rule_is_applied_on_dev_only() -> None:
    choice = DEV["choice"]
    ref = DEV["results"]["e5_small_finetuned"]
    for name, verdict in choice["candidates"].items():
        cand = DEV["results"][name]
        drop = max(ref["per_category"][p]["MRR"] - cand["per_category"][p]["MRR"] for p in ref["per_category"])
        assert verdict["eligible"] == (cand["global"]["MRR"] > ref["global"]["MRR"] and drop <= 0.1), name
    assert "choice" not in TEST and "execution unique d'enregistrement sur le test scelle" in TEST["status"]
    production = (ROOT / "src/agent/retriever.py").read_text(encoding="utf-8")
    assert 'FINETUNED_PATH = ROOT / "models" / "e5-finetuned"' in production  # e5-small reste en production
    assert choice["eligible_architectures"] == []  # aucun candidat ne remplit la regle : e5-small reste

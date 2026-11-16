"""Intervalles de confiance et tests apparies : methode fixee, coherence avec les resultats publies."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATS = json.loads((ROOT / "eval/results/stats.json").read_text(encoding="utf-8"))


def load(name: str) -> dict:
    return json.loads((ROOT / "eval/results" / name).read_text(encoding="utf-8"))


def test_method_is_fixed() -> None:
    from eval.stats import N_RESAMPLES, SEED

    m = STATS["method"]
    assert (m["n_resamples"], m["seed"]) == (N_RESAMPLES, SEED) == (10_000, 20261116)
    assert set(STATS["intervals"]) == set(STATS["paired_tests"]) == {"dev", "test", "test2"}


def test_point_estimates_match_published_results() -> None:
    for split in ("dev", "test"):
        published = load(f"retrieval_{split}.json")["experiments"]
        iv = STATS["intervals"][split]
        assert iv["MRR e5_finetuned"]["estimate"] == round(published["compare_finetuned"]["results"]["e5_finetuned"]["global"]["MRR"], 4)
        assert iv["MRR hybrid_rrf"]["estimate"] == round(published["hybrid_rrf"]["results"]["hybrid_rrf"]["global"]["MRR"], 4)
        fa = load(f"agent_{split}.json")["experiments"]["final_answer"]["results"]["global"]
        assert iv["reponse finale extractive"]["estimate"] == round(fa["correct"] / fa["n"], 4)
    ex = load("extraction.json")["groups"]
    assert STATS["intervals"]["dev"]["F1 unites"]["estimate"] == round(ex["dev_04_10"]["units"]["f1"], 4)
    assert STATS["intervals"]["test"]["F1 relations strict"]["estimate"] == round(ex["test_11_12"]["relations"]["strict"]["f1"], 4)


def test_intervals_contain_the_estimate_and_significance_rule_is_applied() -> None:
    for split, block in STATS["intervals"].items():
        for name, v in block.items():
            assert v["ci95"][0] <= v["estimate"] <= v["ci95"][1], (split, name)
    for split, block in STATS["paired_tests"].items():
        for name, t in block.items():
            lo, hi = t["ci95_difference"]
            assert t["significant"] == (t["p_value_permutation"] < 0.05 and (lo > 0 or hi < 0)), (split, name)

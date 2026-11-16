"""Controle NLI de fidelite : protocole fige, comptes coherents (le JSON seul, sans le modele)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "eval/results/rag_nli.json").read_text(encoding="utf-8"))
LABELS = ["entailment", "neutral", "contradiction"]


def test_protocol_is_fixed_and_recorded() -> None:
    p = DATA["protocol"]
    assert p["model"] == "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" and len(p["revision"]) == 40
    assert p["fixed_before_execution"] is True and "aucun seuil" in p["prediction"]
    assert "ne remplace pas une evaluation humaine" in DATA["warning"]


def test_counts_rates_and_not_entailed_list_are_consistent() -> None:
    for split in ("dev", "test2"):
        s = DATA["splits"][split]
        assert sum(s["counts"].values()) == s["sentences"] > 0
        for lab in LABELS:
            assert s["rates"][lab] == round(s["counts"][lab] / s["sentences"], 4)
        assert len(s["not_entailed"]) == s["sentences"] - s["counts"]["entailment"]
        for item in s["not_entailed"]:
            assert item["label"] in ("neutral", "contradiction") and item["cited_ids"] and item["premise"]
            assert max(item["scores"], key=item["scores"].get) == item["label"]

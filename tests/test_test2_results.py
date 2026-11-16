"""Test scelle elargi : resultats enregistres une seule fois, sur le jeu gele, sous le quota du jour."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "eval/results/test2.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((ROOT / "data/eval/test2_manifest.json").read_text(encoding="utf-8"))


def test_results_refer_to_the_frozen_files() -> None:
    assert DATA["frozen_files"] == MANIFEST["files"]
    assert "apres lecture des interfaces de l'agent" in DATA["authorship_caveat"]
    assert DATA["models"]["retrained"] is False


def test_each_stage_is_a_single_recording() -> None:
    for name, stage in DATA["stages"].items():
        assert "execution unique d'enregistrement" in stage["status"], name


def test_counts_are_consistent() -> None:
    off = DATA["stages"]["offline"]
    assert off["retrieval"]["queries"] == 28 and off["final_answer"]["global"]["n"] == 20
    assert off["abstention"]["n"] == 8 and off["complex"]["n"] == 8 and off["dossiers"]["n"] == 4
    for block in ("global", "simples", "complexe"):
        assert off["retrieval"]["results"]["hybrid_rrf"][block]["n"] == {"global": 28, "simples": 20, "complexe": 8}[block]
    rag = DATA["stages"]["rag"]
    assert rag["citations"]["valid"] == rag["citations"]["syntheses"]  # toutes les syntheses ont des citations valides


def test_quota_is_respected_and_accounted() -> None:
    q = DATA["quota"]
    assert q["total_today_after_test2"] == q["baseline_today_before_test2"] + sum(q["test2_tokens"].values()) <= q["limit_tokens_per_day"]
    ex = DATA["stages"]["extraction"]
    assert ex["interrupted_run"]["tokens"] == ex["groq_tokens"]

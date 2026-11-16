"""Analyse d'erreurs des relations (eval/results/relation_errors_dev.json) : format, coherence, rejeu."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "eval" / "results" / "relation_errors_dev.json"
CACHE = ROOT / "data" / "processed" / "groq_extract_cache.jsonl"
RELATION_TYPES = ["creates_action", "supports", "answers", "supersedes", "contradicts"]


def load() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_format_and_metadata() -> None:
    from src.ingest.extract import SYSTEM_PROMPT

    data = load()
    for key in ("experiment", "generated_on", "script", "split", "projects", "status", "source", "replay",
                "system_prompt_sha256", "categories", "by_type", "totals"):
        assert key in data, key
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["generated_on"])
    assert (ROOT / data["script"]).exists()
    assert data["split"] == "dev"
    assert data["projects"] == [f"projet_{n:02d}" for n in range(4, 11)]  # jamais le test scelle
    assert "cache local" in data["source"] and data["replay"]["api_calls"] == 0
    assert data["replay"]["consistent_with_extraction_json_dev_04_10"] is True
    sha = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert data["system_prompt_sha256"] == sha == data["system_prompt_sha256_in_extraction_json"]
    assert set(data["by_type"]) == set(RELATION_TYPES)


def test_counts_are_consistent() -> None:
    data = load()
    errors, misses = data["categories"]["predicted_errors"], data["categories"]["missed"]
    for name, e in data["by_type"].items():
        assert e["correct"] + sum(e["predicted_errors"][k] for k in errors) == e["predicted"], name
        assert e["found"] + sum(e["missed"][k] for k in misses) == e["annotated"], name
        assert e["found"] == e["correct"], name  # une relation juste est une relation annotee retrouvee
    t = data["totals"]
    for key in ("annotated", "predicted", "correct", "found"):
        assert t[key] == sum(e[key] for e in data["by_type"].values()), key
    for key in errors:
        assert t["predicted_errors"][key] == sum(e["predicted_errors"][key] for e in data["by_type"].values())
    for key in misses:
        assert t["missed"][key] == sum(e["missed"][key] for e in data["by_type"].values())
    assert t["missed_total"] == t["annotated"] - t["found"]
    # memes totaux que l'evaluation de reference (extraction.json, groupe dev_04_10)
    reference = json.loads((ROOT / "eval/results/extraction.json").read_text(encoding="utf-8"))["groups"]["dev_04_10"]["relations"]
    assert (t["annotated"], t["predicted"], t["correct"]) == (reference["annotated"], reference["extracted"], reference["strict"]["correct"])
    for name, e in data["by_type"].items():
        ref = reference["per_type"][name]
        assert (e["annotated"], e["predicted"], e["correct"]) == (ref["annotated"], ref["extracted"], ref["correct"]), name


@pytest.mark.skipif(not CACHE.exists(), reason="cache Groq local absent (clone frais) : rejeu impossible sans appel")
def test_replay_from_local_cache_gives_the_same_analysis() -> None:
    from eval.analyze_relation_errors import analyze

    analysis, calls, consistent = analyze()
    data = load()
    assert calls["api_calls"] == 0
    assert consistent is True
    assert analysis["by_type"] == data["by_type"]
    assert analysis["totals"] == data["totals"]

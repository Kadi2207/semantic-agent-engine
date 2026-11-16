"""Mesures de l'agent : eval/results/agent_dev.json et la section hybrid_rrf de retrieval_dev.json.

Aucun modele charge : on verifie la coherence interne des fichiers, leurs empreintes et le statut des sondes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from eval.results_io import sha256_text_file

ROOT = Path(__file__).resolve().parent.parent
AGENT = json.loads((ROOT / "eval/results/agent_dev.json").read_text(encoding="utf-8"))
PROBES = json.loads((ROOT / "data/eval/probes_abstention.json").read_text(encoding="utf-8"))
QUERIES = json.loads((ROOT / "data/eval/queries_dev.json").read_text(encoding="utf-8"))["queries"]
DEV_PROJECTS = {f"projet_{n:02d}" for n in range(1, 11)}


def test_final_answer_counts_are_consistent_with_the_per_query_rows() -> None:
    final = AGENT["experiments"]["final_answer"]["results"]
    rows = final["per_query"]
    answerable = {q["query_id"]: q for q in QUERIES if q["relevant_unit_ids"] and q["piege"] != "sans_reponse"}
    assert {r["query_id"] for r in rows} == set(answerable) and len(rows) == 25
    for row in rows:  # la justesse est recalculee a partir de ce que l'agent a renvoye
        shown = {row["final_unit_id"], *row["conflict_with"]}
        assert row["correct"] == (not row["abstained"] and bool(shown & set(answerable[row["query_id"]]["relevant_unit_ids"])))
    blocks = {"global": rows, **{p: [r for r in rows if r["piege"] == p] for p in final["per_category"]}}
    for name, subset in blocks.items():
        block = final["global"] if name == "global" else final["per_category"][name]
        assert (block["n"], block["correct"]) == (len(subset), sum(r["correct"] for r in subset)), name
        assert block["accuracy"] == block["correct"] / block["n"]
    assert sum(b["n"] for b in final["per_category"].values()) == final["global"]["n"]
    assert "decision_active" in final["per_category"]


def test_unanswerable_queries_are_reported_as_they_are() -> None:
    block = AGENT["experiments"]["final_answer"]["sans_reponse"]
    expected = {q["query_id"] for q in QUERIES if q["piege"] == "sans_reponse"}
    assert {r["query_id"] for r in block["per_query"]} == expected and block["n"] == 5
    assert block["abstained"] == sum(r["abstained"] for r in block["per_query"])
    assert block["abstained"] + block["answered"] == block["n"]


def test_probes_are_exploratory_development_only_and_match_the_recorded_result() -> None:
    assert PROBES["status"] == "sondes exploratoires, projets de developpement, jamais utilisees pour regler"
    assert len(PROBES["probes"]) == 6 and {p["project_id"] for p in PROBES["probes"]} <= DEV_PROJECTS
    section = AGENT["experiments"]["abstention_probes"]
    assert section["status"] == PROBES["status"]
    assert section["sha256"] == sha256_text_file(ROOT / section["file"])
    results = section["results"]
    by_id = {p["probe_id"]: p for p in PROBES["probes"]}
    for row in results["per_probe"]:
        probe = by_id[row["probe_id"]]
        assert row["query"] == probe["query"]
        assert row["wrongly_answered"] == (not row["abstained"] and row["final_unit_id"] not in probe["acceptable_unit_ids"])
    assert results["wrongly_answered"] == sum(r["wrongly_answered"] for r in results["per_probe"])
    assert (results["wrongly_answered"], results["n"], results["abstained"]) == (5, 6, 0)  # limite mesuree, non corrigee
    summary = section["summary"]  # trois issues : abstention, reponse a tort, reponse acceptable sur le fond
    assert summary["abstained"] + summary["wrongly_answered"] + summary["acceptable_on_substance"] == summary["n"] == 6
    assert summary["acceptable_probe_ids"] == [p["probe_id"] for p in PROBES["probes"] if p["acceptable_unit_ids"]] == ["probe_02"]
    assert section["rule"] == PROBES["rule"] and section["independence_note"] == PROBES["independence_note"]
    dev_unanswerable = {q["query_id"] for q in QUERIES if q["piege"] == "sans_reponse"}
    assert {p["same_as_dev_query"] for p in PROBES["probes"] if "same_as_dev_query" in p} == dev_unanswerable
    assert "ce n'est pas une mesure independante" in PROBES["independence_note"]


def test_resolution_breakdown_separates_real_resolutions_from_the_conflict() -> None:
    for name in ("agent_dev.json", "agent_test.json"):
        final = json.loads((ROOT / "eval/results" / name).read_text(encoding="utf-8"))["experiments"]["final_answer"]
        rows, block, g = final["results"]["per_query"], final["resolution_breakdown"], final["results"]["global"]
        assert block["resolved"] == g["resolved"] == sum(r["resolved"] for r in rows)
        assert block["resolved_correct"] == sum(r["resolved"] and r["correct"] for r in rows)
        conflicts = [r["query_id"] for r in rows if r["conflict_with"] and r["correct"]]
        assert block["unresolved_conflict_query_ids"] == conflicts
        assert block["unresolved_conflict_counted_correct"] == len(conflicts)
        assert block["correct"] == g["correct"]
        assert block["correct_if_unresolved_conflict_not_counted"] == g["correct"] - len(conflicts)
        assert block["resolved_correct"] + len(conflicts) == g["correct_thanks_to_resolution"]
    dev = AGENT["experiments"]["final_answer"]["resolution_breakdown"]
    assert (dev["resolved_correct"], dev["unresolved_conflict_query_ids"], dev["correct_if_unresolved_conflict_not_counted"]) == (3, ["q_dev_026"], 16)


def test_metadata_come_from_the_code_and_the_query_file() -> None:
    source = (ROOT / "src/agent/answer.py").read_text(encoding="utf-8")
    threshold = float(re.search(r"^ABSTENTION_THRESHOLD = ([0-9.]+)", source, re.MULTILINE).group(1))
    assert AGENT["params"]["abstention_threshold"] == threshold
    assert AGENT["params"]["rrf_k"] == 60
    assert AGENT["split"] == "dev" and (ROOT / AGENT["script"]).exists()
    assert AGENT["queries"]["sha256"] == sha256_text_file(ROOT / AGENT["queries"]["file"])
    assert (AGENT["queries"]["total"], AGENT["queries"]["evaluated"]) == (30, 25)


def test_sealed_test_run_is_a_single_recording_without_tuning() -> None:
    test = json.loads((ROOT / "eval/results/agent_test.json").read_text(encoding="utf-8"))
    assert test["split"] == "test" and "abstention_probes" not in test["experiments"]  # les sondes sont dev seulement
    assert "execution unique d'enregistrement sur le test scelle, aucun reglage effectue" in test["status"]
    assert (test["queries"]["total"], test["queries"]["evaluated"]) == (6, 5)
    assert test["queries"]["sha256"] == sha256_text_file(ROOT / test["queries"]["file"])

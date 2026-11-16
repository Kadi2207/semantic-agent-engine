"""Relations v2 : resultats enregistres coherents, extraction inchangee, choix sur le dev, test scelle unique."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "eval/results/relations_v2.json").read_text(encoding="utf-8"))
EXTRACTION = json.loads((ROOT / "eval/results/extraction.json").read_text(encoding="utf-8"))
DEV, TEST = DATA["dev"], DATA["test"]


def test_extraction_is_untouched_and_is_the_before_column() -> None:
    from src.ingest.extract import SYSTEM_PROMPT

    sha = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert DATA["extraction"]["system_prompt_sha256"] == sha == EXTRACTION["extraction"]["system_prompt_sha256"]
    assert DEV["before"] == EXTRACTION["groups"]["dev_04_10"]["relations"]
    assert TEST["before"] == EXTRACTION["groups"]["test_11_12"]["relations"]


def test_link_prompts_are_versioned_and_separate() -> None:
    from eval.results_io import sha256_params
    from src.ingest.link import LINK_PROMPTS

    assert DATA["link"]["prompts_sha256"] == {v: sha256_params({"prompt": p}) for v, p in LINK_PROMPTS.items()}
    assert set(DATA["link"]["prompts_sha256"]) == {"v1", "v2"}


def test_ceiling_of_strict_recall() -> None:
    c = DEV["ceiling"]
    assert (c["annotated_relations"], c["both_endpoints_extracted"]) == (84, 73)
    assert c["max_strict_recall"] == round(73 / 84, 4) == 0.869
    for name, v in DEV["variants"].items():
        assert v["relations"]["strict"]["recall"] <= c["max_strict_recall"], name


def test_only_executed_configurations_can_be_selected_and_the_rule_is_applied() -> None:
    variants = DEV["variants"]
    assert len(variants) == 2 * 4 * 3  # 2 consignes x k = 0..3 x 3 modes, toutes listees
    for name, v in variants.items():
        assert v["executed"] == (v["k"] == 3), name
        assert ("indicative" in v["nature"]) == (not v["executed"]), name
    executed = {n: v for n, v in variants.items() if v["executed"]}
    eligible = {n: v for n, v in executed.items() if v["relations"]["strict"]["precision"] >= 0.45} or executed
    best = max(v["relations"]["strict"]["f1"] for v in eligible.values())
    chosen = DEV["selected"]["variant"]
    assert chosen in eligible and eligible[chosen]["relations"]["strict"]["f1"] == best and DEV["selected"]["executed"]
    assert DEV["after"] == variants[chosen]["relations"]
    strict = DEV["after"]["strict"]
    assert DEV["criterion_met"] == (strict["f1"] >= 0.456 and strict["precision"] >= 0.45)


def test_caveats_on_the_dev_selection_are_recorded() -> None:
    caveats = DEV["caveats"]
    assert "introduit apres l'analyse des resultats de la v1" in caveats["complement_mode"]
    assert "6 configurations executees" in caveats["selection"] and "optimiste" in caveats["selection"]
    assert "seul le test scelle" in caveats["unbiased_measure"] and "aucune v3" in caveats["prompt_v2"]
    assert sum(v["executed"] for v in DEV["variants"].values()) == 6
    assert set(DEV["best_per_prompt"]) == {"v1", "v2"}


def test_sealed_test_is_a_single_recording_with_the_dev_configuration() -> None:
    assert "execution unique d'enregistrement sur le test scelle" in TEST["status"]
    assert TEST["selected"] == DEV["selected"] and "variants" not in TEST
    assert TEST["criterion_met_on_dev"] == DEV["criterion_met"]
    assert set(TEST["per_project"]) == {DEV["selected"]["prompt"]}


def test_error_analysis_uses_the_same_categories() -> None:
    from eval.analyze_relation_errors import GOLD_MISSES, PRED_ERRORS

    after = DEV["error_analysis"]["after"]["totals"]
    assert list(after["predicted_errors"]) == PRED_ERRORS and list(after["missed"]) == GOLD_MISSES
    assert after["predicted"] == DEV["after"]["extracted"] and after["correct"] == DEV["after"]["strict"]["correct"]
    before = json.loads((ROOT / "eval/results/relation_errors_dev.json").read_text(encoding="utf-8"))["totals"]
    assert DEV["error_analysis"]["before"]["totals"] == before


def test_budget_is_respected() -> None:
    b = DATA["budget"]
    assert b["limit_tokens"] == 100_000 and 0 < b["lot_tokens_consumed"] <= b["limit_tokens"]


def test_extracted_memory_uses_v2_relations_only_if_the_criterion_is_met() -> None:
    from src.memory.extracted import EXTRACTED_DIR

    if DEV["criterion_met"]:
        assert EXTRACTED_DIR.name == "extracted_v2"
    else:
        assert EXTRACTED_DIR.name == "extracted"  # resultat negatif : rien n'est active

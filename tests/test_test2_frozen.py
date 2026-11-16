"""Test scelle elargi : fichiers geles avant toute execution (empreintes du manifeste), contenu conforme au cahier."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from eval.results_io import sha256_text_file

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "data/eval/test2_manifest.json").read_text(encoding="utf-8"))
QUERIES = json.loads((ROOT / "data/eval/queries_test2.json").read_text(encoding="utf-8"))["queries"]


def test_frozen_files_match_the_manifest() -> None:
    assert len(MANIFEST["files"]) == 5
    for path, sha in MANIFEST["files"].items():
        assert sha256_text_file(ROOT / path) == sha, path
    assert "avant toute execution" in MANIFEST["authored_by"] and "jamais utilise pour un reglage" in MANIFEST["use"]


def test_four_new_projects_of_three_meetings_valid_and_separate_from_data_raw() -> None:
    from scripts.validate_corpus import validate_file

    paths = sorted((ROOT / "data/test2").glob("projet_*.json"))
    assert [p.name[:10] for p in paths] == ["projet_13_", "projet_14_", "projet_15_", "projet_16_"]
    raw_ids = {json.loads(p.read_text(encoding="utf-8"))["project_id"] for p in (ROOT / "data/raw").glob("projet_*.json")}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["meetings"]) == 3 and data["project_id"] not in raw_ids
        problems, _, _ = validate_file(path)
        assert problems == [], problems


def test_query_set_composition() -> None:
    counts = Counter(q["piege"] for q in QUERIES)
    assert len(QUERIES) == 40
    assert (counts["sans_reponse"], counts["complexe"], counts["dossier"]) == (8, 8, 4)
    units = {u["unit_id"] for p in (ROOT / "data/test2").glob("projet_*.json") for u in json.loads(p.read_text(encoding="utf-8"))["units"]}
    for q in QUERIES:
        assert set(q["relevant_unit_ids"]) <= units, q["query_id"]
        assert bool(q["relevant_unit_ids"]) == (q["piege"] != "sans_reponse"), q["query_id"]
        if q["piege"] == "complexe":
            assert len(q["relevant_unit_ids"]) >= 2, q["query_id"]

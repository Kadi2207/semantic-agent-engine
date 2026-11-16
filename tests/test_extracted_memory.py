"""Memoire extraite (chaine complete) : chargement, coherence des structures, mesure enregistree, interface.

La memoire extraite (data/extracted) est versionnee : elle se charge sans le cache Groq ni aucun appel.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from src.agent.sources import source_passage
from src.memory.corpus import load_corpus, load_participants, load_relations
from src.memory.extracted import EXTRACTED_DIR, EXTRACTED_PROJECTS, MENTION

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "e5-finetuned" / "config.json"
EXTRACTION = json.loads((ROOT / "eval/results/extraction.json").read_text(encoding="utf-8"))
RESULT = json.loads((ROOT / "eval/results/agent_extracted_dev.json").read_text(encoding="utf-8"))
CORPUS = load_corpus(EXTRACTED_DIR)
RELATIONS = load_relations(EXTRACTED_DIR)
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="modele fine-tune absent (clone frais)")


def test_only_projects_04_to_12_with_the_same_meetings_as_the_annotated_memory() -> None:
    projects = {c.project_id for c in CORPUS}
    assert projects == {f"projet_{n:02d}" for n in EXTRACTED_PROJECTS}
    for path in sorted(EXTRACTED_DIR.glob("projet_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = json.loads((ROOT / "data/raw" / path.name).read_text(encoding="utf-8"))
        assert data["mention"] == MENTION and data["memory"] == "extraite"
        assert data["system_prompt_sha256"] == EXTRACTION["extraction"]["system_prompt_sha256"]
        assert [m["transcript"] for m in data["meetings"]] == [m["transcript"] for m in raw["meetings"]]
    assert load_participants(EXTRACTED_DIR) == {p: v for p, v in load_participants(ROOT / "data/raw").items() if p in projects}


def test_counts_are_those_of_extraction_json() -> None:
    count = Counter(c.project_id for c in CORPUS)
    ids = {c.unit.unit_id: c.project_id for c in CORPUS}
    rel = Counter(ids[r.from_id] for r in RELATIONS)
    for group, numbers in (("dev_04_10", range(4, 11)), ("test_11_12", range(11, 13))):
        names = [f"projet_{n:02d}" for n in numbers]
        expected = EXTRACTION["groups"][group]
        assert sum(count[p] for p in names) == expected["units"]["extracted"]
        assert sum(rel[p] for p in names) == expected["relations"]["extracted"]


def test_structures_are_coherent() -> None:
    ids = {c.unit.unit_id: c for c in CORPUS}
    assert len(ids) == len(CORPUS)  # identifiants uniques
    for r in RELATIONS:  # relations entre unites existantes d'un meme projet
        assert r.from_id in ids and r.to_id in ids and r.from_id != r.to_id
        assert ids[r.from_id].project_id == ids[r.to_id].project_id
    superseded_targets = {r.to_id for r in RELATIONS if r.relation == "supersedes"}
    assert all(ids[u].unit.status != "active" for u in superseded_targets)  # regle de coherence du pipeline
    for c in CORPUS:  # chaque unite a une preuve retrouvee dans le transcript de sa reunion
        assert c.unit.unit_id.startswith(c.unit.source_id) and c.unit.evidence.strip()
        assert source_passage(c.unit).found, c.unit.unit_id


def test_recorded_measure_is_internally_consistent() -> None:
    r = RESULT["results"]
    for memory in ("extracted", "annotated"):
        rows, g = r[memory]["per_query"], r[memory]["global"]
        assert g["n"] == len(rows) == RESULT["queries"]["evaluated"] == 20
        assert g["correct"] == sum(x["correct"] for x in rows)
        assert {x["project_id"] for x in rows} <= {f"projet_{n:02d}" for n in range(4, 11)}
    extracted = {x["query_id"]: x for x in r["extracted"]["per_query"]}
    for qid, row in extracted.items():
        assert row["correct"] == (not row["abstained"] and bool(set(row["matched_annotated_unit_ids"]) & set(row["relevant_unit_ids"]))), qid
    assert [x["query_id"] for x in r["annotated"]["per_query"]] == list(extracted)  # exactement les memes requetes
    assert "11" not in " ".join(RESULT["queries"]["projects"]) and "test scelle" in RESULT["status"]
    # la memoire annotee redonne, sur ces requetes, exactement les lignes de agent_dev.json
    dev = {x["query_id"]: x for x in json.loads((ROOT / "eval/results/agent_dev.json").read_text(encoding="utf-8"))["experiments"]["final_answer"]["results"]["per_query"]}
    for row in r["annotated"]["per_query"]:
        assert row == dev[row["query_id"]], row["query_id"]


def run_app(memory: str | None, project: str | None = None, question: str | None = None):  # noqa: ANN201
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600).run()
    if memory:
        at.sidebar.radio[0].set_value(memory).run()
    if project:
        at.sidebar.radio[1].set_value(project).run()
    if question:
        at.text_input(key="question").set_value(question).run()
        next(b for b in at.button if b.label == "Envoyer").click().run()
    assert not at.exception
    return at


@needs_model
def test_interface_defaults_to_annotated_memory_with_all_projects() -> None:
    at = run_app(None)
    assert at.sidebar.radio[0].value == "annotée (référence)"
    assert len(at.sidebar.radio[1].options) == 12 and at.sidebar.radio[1].value == "projet_01"
    at.button(key="demo_0").click().run()  # le parcours Atlas reste sur la memoire annotee
    next(b for b in at.button if b.label == "Envoyer").click().run()
    assert "p01_u20" in "\n".join(m.value for m in at.markdown)


@needs_model
def test_extracted_mode_offers_projects_04_to_12_with_the_mention_and_answers() -> None:
    at = run_app("extraite automatiquement", "projet_04", "A-t-on perdu des fiches pendant l'essai de reprise des données ?")
    assert [o[:2] for o in at.sidebar.radio[1].options] == [f"{n:02d}" for n in range(4, 13)]
    assert any(MENTION in i.value for i in at.info)
    page = "\n".join(m.value for m in at.markdown)
    assert "p04_m02_x02" in page  # unite extraite (appariee a p04_u08 dans la mesure)

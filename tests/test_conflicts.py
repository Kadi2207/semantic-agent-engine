"""Conflit non tranche : une contradiction n'est une resolution que si l'unite contredite n'est plus active.

Trois cas, sur les projets de developpement :
- projet_10 : p10_u15 contredit p10_u14, toutes deux actives -> conflit non tranche, rien dans l'historique ;
- projet_02 : p02_u08 contredit p02_u01 (superseded) -> resolution et historique inchanges ;
- projet_01 (Atlas) : chaine de deux remplacements -> inchangee.
Les tests de graphe n'ont pas besoin du modele ; la comparaison avant/apres de answer() en a besoin.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from src.agent.answer import ReplacementGraph
from src.memory.corpus import load_corpus, load_relations

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "fixtures" / "answers_before_conflict_fix.json"
MODEL = ROOT / "models" / "e5-finetuned" / "config.json"


class Lookup:
    """Acces aux unites par identifiant, sans modele (meme interface que HybridRetriever.unit)."""

    def __init__(self) -> None:
        self._units = {c.unit.unit_id: c.unit for c in load_corpus(ROOT / "data" / "raw")}

    def unit(self, unit_id: str):  # noqa: ANN201
        return self._units[unit_id]


@pytest.fixture(scope="module")
def lookup() -> Lookup:
    return Lookup()


@pytest.fixture(scope="module")
def graph() -> ReplacementGraph:
    return ReplacementGraph(load_relations(ROOT / "data" / "raw"))


def test_two_active_contradicting_units_are_an_open_conflict(graph: ReplacementGraph, lookup: Lookup) -> None:
    u14, u15 = lookup.unit("p10_u14"), lookup.unit("p10_u15")
    assert u14.status == u15.status == "active"
    assert graph.open_conflicts(u15, lookup) == ("p10_u14",)
    assert graph.open_conflicts(u14, lookup) == ("p10_u15",)  # detecte dans les deux sens
    assert graph.history("p10_u15", lookup) == ()  # p10_u14 n'est pas une information perimee
    assert graph.resolve(u15, lookup) == (u15, ())
    assert graph.resolve(u14, lookup) == (u14, ())


def test_contradiction_of_a_superseded_unit_is_still_a_resolution(graph: ReplacementGraph, lookup: Lookup) -> None:
    u01, u08 = lookup.unit("p02_u01"), lookup.unit("p02_u08")
    assert (u01.status, u08.status) == ("superseded", "active")
    resolved, path = graph.resolve(u01, lookup)
    assert resolved.unit_id == "p02_u08"
    assert path == ("p02_u01 (superseded) -contradicts-> p02_u08 (active)",)
    assert graph.history("p02_u08", lookup) == (("p02_u01", "contradicts", "p02_u08"),)
    assert graph.open_conflicts(u08, lookup) == ()
    assert graph.open_conflicts(u01, lookup) == ()


def test_atlas_two_step_replacement_chain_is_unchanged(graph: ReplacementGraph, lookup: Lookup) -> None:
    resolved, path = graph.resolve(lookup.unit("p01_u01"), lookup)
    assert resolved.unit_id == "p01_u20"
    assert path == (
        "p01_u01 (superseded) -supersedes-> p01_u12 (superseded)",
        "p01_u12 (superseded) -supersedes-> p01_u20 (active)",
    )
    expected = (("p01_u12", "supersedes", "p01_u20"), ("p01_u01", "supersedes", "p01_u12"))
    assert graph.history("p01_u20", lookup) == expected == graph.history("p01_u20")
    assert graph.open_conflicts(lookup.unit("p01_u20"), lookup) == ()


def test_only_projet_10_has_an_open_conflict_in_the_corpus(graph: ReplacementGraph, lookup: Lookup) -> None:
    open_pairs = {
        (unit_id, other)
        for unit_id in lookup._units
        for other in graph.open_conflicts(lookup.unit(unit_id), lookup)
    }
    assert open_pairs == {("p10_u14", "p10_u15"), ("p10_u15", "p10_u14")}


@pytest.mark.skipif(not MODEL.exists(), reason="modele fine-tune absent (clone frais) : answer() ne peut pas tourner")
def test_answers_for_atlas_and_projet_02_are_identical_before_and_after() -> None:
    """Compare answer() aux reponses enregistrees sur main avant la correction (champ par champ)."""
    from src.agent.answer import answer

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["answers"]
    assert {a["project_id"] for a in baseline} == {"projet_01", "projet_02"}
    for before in baseline:
        after = dataclasses.asdict(answer(before["query"], before["project_id"]))
        assert after.pop("conflict_with") == (), before["query"]  # seul champ nouveau, vide ici
        after["source_date"] = str(after["source_date"]) if after["source_date"] else None
        after["best_dense_score"] = round(after["best_dense_score"], 6)
        after["fused_score"] = round(after["fused_score"], 9) if after["fused_score"] is not None else None
        after["resolution"] = list(after["resolution"])
        after["history"] = [list(h) for h in after["history"]]
        assert after == before, before["query"]


@pytest.mark.skipif(not MODEL.exists(), reason="modele fine-tune absent (clone frais) : answer() ne peut pas tourner")
def test_answer_reports_the_orvane_conflict_without_concluding() -> None:
    from src.agent.answer import answer

    result = answer("La facture de mars d'Orvane est-elle réglée ?", "projet_10")
    assert result.unresolved_conflict
    assert {result.unit_id, *result.conflict_with} == {"p10_u14", "p10_u15"}
    assert result.history == () and result.resolution == ()

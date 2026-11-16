"""Dossier documente : intention, constitution deterministe depuis la memoire, aucune unite sans source."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.intent import detect_intent

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "e5-finetuned" / "config.json"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="modele fine-tune absent (clone frais)")
NAMES = {"projet_01": "Atlas", "projet_02": "Budget marketing", "projet_10": "Relances clients et recouvrement"}


@pytest.mark.parametrize(
    "query, intent",
    [
        ("Constitue le dossier sur la mise en production d'Atlas", "dossier"),
        ("Fais-moi le dossier du projet Budget marketing", "dossier"),
        ("Peux-tu préparer le dossier du recrutement ?", "dossier"),
        ("Qui suit le dossier Orvane ?", "answer"),  # question SUR un dossier, pas une demande
        ("Écris un mail à la personne qui doit signer le devis d'Horizon", "draft_email"),
        ("Quelle est la date de mise en production d'Atlas ?", "answer"),
    ],
)
def test_dossier_intent(query: str, intent: str) -> None:
    assert detect_intent(query) == intent


def test_named_project_and_topic() -> None:
    from src.agent.dossier import dossier_topic, named_project

    assert named_project("Fais-moi le dossier du projet Budget marketing", NAMES) == "projet_02"
    assert named_project("Constitue le dossier sur la mise en production", NAMES) is None
    assert dossier_topic("Constitue le dossier sur la mise en production d'Atlas") == "la mise en production d'atlas"


def build(query: str, project_id: str):  # noqa: ANN201
    from src.agent.dossier import build_dossier

    return build_dossier(query, project_id, NAMES[project_id])


def assert_every_unit_is_sourced(dossier) -> None:  # noqa: ANN001
    from src.agent.dossier import to_markdown
    from src.agent.sources import source_passage

    assert dossier.units and {u.unit_id for u in dossier.units} == dossier.cited_ids()
    markdown = to_markdown(dossier)
    sources = markdown.split("## Sources citées")[1]
    for unit in dossier.units:  # chaque unite citee a sa preuve retrouvee dans le transcript, et figure aux sources
        assert source_passage(unit).found, unit.unit_id
        assert f"- {unit.unit_id}, réunion {unit.source_id}" in sources and unit.evidence in sources
    body = markdown.split("## Sources citées")[0]
    for line in body.splitlines():  # chaque element de liste renvoie a une unite et a sa reunion
        if line.startswith("- ") and "Deux affirmations" not in line:
            assert "réunion p" in line, line


@needs_model
def test_atlas_dossier() -> None:
    from src.agent.dossier import to_markdown

    d = build("Constitue le dossier sur la mise en production d'Atlas", "projet_01")
    assert d.scope == "sujet" and not d.abstained
    assert [i.unit.unit_id for i in d.decisions] == ["p01_u20"]  # decision du 5 mai
    history = {i.unit.unit_id: i for i in d.history}
    assert set(history) == {"p01_u12", "p01_u01"}  # le 14 avril, valide puis propose
    assert history["p01_u12"].detail == "La décision du 31/03/2026 (14 avril) a été remplacée par celle du 09/04/2026 (5 mai)."
    tasks = {i.unit.unit_id: i.detail for i in d.tasks}
    assert tasks["p01_u21"] == "responsable : Claire ; échéance : le jour même, soit le 9 avril 2026, date de la réunion ; statut : en vigueur"
    assert d.conflicts == []
    markdown = to_markdown(d)
    assert "La mise en production d'Atlas est reportée au 5 mai." in markdown and "14 avril" in markdown
    assert_every_unit_is_sourced(d)


@needs_model
def test_budget_dossier_shows_the_resolved_contradiction() -> None:
    d = build("Fais-moi le dossier du projet Budget marketing", "projet_02")
    assert d.scope == "projet"
    assert "p02_u08" in {i.unit.unit_id for i in d.decisions} and "p02_u01" not in {i.unit.unit_id for i in d.decisions}
    contradiction = next(i for i in d.history if i.unit.unit_id == "p02_u01")
    assert contradiction.detail == "La décision du 04/02/2026 a été contredite par celle du 18/02/2026." and contradiction.replaced_by == "p02_u08"
    assert d.conflicts == []
    cancelled = next(i for i in d.tasks if i.unit.unit_id == "p02_u04")
    assert cancelled.detail.endswith("statut : annulée")
    assert_every_unit_is_sourced(d)


@needs_model
def test_relances_dossier_flags_the_open_conflict_without_concluding() -> None:
    from src.agent.dossier import to_markdown

    d = build("Constitue le dossier du projet Relances clients et recouvrement", "projet_10")
    assert [{a.unit.unit_id, b.unit.unit_id} for a, b in d.conflicts] == [{"p10_u14", "p10_u15"}]
    assert not {"p10_u14", "p10_u15"} & {i.unit.unit_id for i in d.history}  # aucune des deux n'est presentee comme perimee
    assert "les sources ne permettent pas de trancher" in to_markdown(d)
    assert_every_unit_is_sourced(d)


@needs_model
def test_off_topic_dossier_abstains() -> None:
    d = build("Constitue le dossier sur le salaire du nouveau directeur technique", "projet_01")
    assert d.abstained and d.units == []

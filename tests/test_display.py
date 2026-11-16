"""Libelles d'affichage : type en francais, proposition non validee, echeance "le jour même" datee."""
from __future__ import annotations

from pathlib import Path
from typing import get_args

from src.agent.display import TYPE_LABELS, deadline_note, french_date, type_label
from src.memory.corpus import load_corpus
from src.schemas import SemanticUnit

UNITS = {c.unit.unit_id: c.unit for c in load_corpus(Path(__file__).resolve().parent.parent / "data" / "raw")}


def test_every_unit_type_has_a_french_label() -> None:
    assert set(TYPE_LABELS) == set(get_args(SemanticUnit.model_fields["type"].annotation))
    assert type_label(UNITS["p01_u20"]) == "décision"
    assert type_label(UNITS["p01_u07"]) == "engagement"


def test_a_proposal_is_labelled_as_not_validated() -> None:
    assert UNITS["p01_u01"].type == "idea"
    assert type_label(UNITS["p01_u01"]) == "proposition, non validée"


def test_same_day_deadline_gets_the_meeting_date_and_nothing_else_is_computed() -> None:
    assert deadline_note(UNITS["p01_u21"]) == "le jour même, soit le 9 avril 2026, date de la réunion"
    assert deadline_note(UNITS["p01_u13"]) is None  # "d'ici vendredi" : aucune date calculee
    assert deadline_note(UNITS["p04_u16"]) is None  # "d'ici le 10 juillet" : deja une date
    assert deadline_note(UNITS["p01_u20"]) is None  # une decision n'a pas d'echeance
    assert french_date(UNITS["p01_u21"].source_date) == "9 avril 2026"

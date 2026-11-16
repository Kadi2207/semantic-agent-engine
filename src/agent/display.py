"""Libelles d'affichage (aucune logique de recherche ni de resolution).

- type de l'unite en francais ; une unite de type idea est une "proposition, non validée" ;
- echeance relative "le jour même" : on ajoute la date de la reunion, sans rien calculer d'autre.
"""
from __future__ import annotations

from datetime import date

from src.schemas import SemanticUnit

TYPE_LABELS = {
    "decision": "décision",
    "action": "action",
    "commitment": "engagement",
    "question": "question",
    "answer": "réponse",
    "fact": "fait",
    "risk": "risque",
    "idea": "proposition, non validée",
}
_MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
SAME_DAY = "le jour même"


def type_label(unit: SemanticUnit) -> str:
    return TYPE_LABELS[unit.type]


def french_date(day: date) -> str:
    return f"{'1er' if day.day == 1 else day.day} {_MONTHS[day.month - 1]} {day.year}"


def deadline_note(unit: SemanticUnit) -> str | None:
    """Pour une action ou un engagement a faire "le jour même" : la date de la reunion, sinon None."""
    if unit.type in ("action", "commitment") and SAME_DAY in unit.statement:
        return f"{SAME_DAY}, soit le {french_date(unit.source_date)}, date de la réunion"
    return None

"""Libelles d'affichage (aucune logique de recherche ni de resolution).

- type de l'unite en francais ; une unite de type idea est une "proposition, non validée" ;
- echeance relative "le jour même" : on ajoute la date de la reunion, sans rien calculer d'autre.
"""
from __future__ import annotations

import re
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


# --------------------------------------------------------------------------- libelles de l'interface

STATUS_LABELS = {"active": "en vigueur", "superseded": "remplacée", "cancelled": "annulée"}
INTENT_LABELS = {"answer": "Répondre", "draft_email": "Rédiger un email", "dossier": "Constituer un dossier"}
# type -> (nom avec article en debut de phrase, nom avec article en cours de phrase, genre)
_TYPE_NOUNS = {
    "decision": ("La décision", "la décision", "f"),
    "action": ("L'action", "l'action", "f"),
    "commitment": ("L'engagement", "l'engagement", "m"),
    "question": ("La question", "la question", "f"),
    "answer": ("La réponse", "la réponse", "f"),
    "fact": ("Le fait", "le fait", "m"),
    "risk": ("Le risque", "le risque", "m"),
    "idea": ("La proposition", "la proposition", "f"),
}
_DAY_MONTH = re.compile(rf"\b\d{{1,2}}(?:er)? (?:{'|'.join(_MONTHS)})\b")
_STEP = re.compile(r"^(\S+) \(\w+\) -(\w+)-> (\S+) \(\w+\)$")


def status_label(status: str) -> str:
    return STATUS_LABELS[status]


def intent_label(intent: str) -> str:
    return INTENT_LABELS[intent]


def french_number(value: float, digits: int = 3) -> str:
    """Decimales a la francaise : 0.09 -> '0,090'."""
    return f"{value:.{digits}f}".replace(".", ",")


def short_date(day: date) -> str:
    return f"{day:%d/%m/%Y}"


def _mention(unit: SemanticUnit) -> str:
    """Date citee dans l'enonce (ex. '14 avril'), entre parentheses, si l'enonce en contient une."""
    found = _DAY_MONTH.search(unit.statement)
    return f" ({found.group(0)})" if found else ""


def replacement_sentence(old: SemanticUnit, new: SemanticUnit, relation: str) -> str:
    """'La décision du 31/03/2026 (14 avril) a été remplacée par celle du 09/04/2026 (5 mai).'"""
    subject, _, gender = _TYPE_NOUNS[old.type]
    _, new_noun, new_gender = _TYPE_NOUNS[new.type]
    verb = ("contredit" if relation == "contradicts" else "remplacé") + ("e" if gender == "f" else "")
    by = ("celle" if new_gender == "f" else "celui") if new.type == old.type else new_noun
    return (
        f"{subject} du {short_date(old.source_date)}{_mention(old)} a été {verb} par {by} du "
        f"{short_date(new.source_date)}{_mention(new)}."
    )


def resolution_sentences(resolution: tuple[str, ...], lookup) -> list[str]:  # noqa: ANN001
    """Une phrase lisible par etape du chemin de resolution (le chemin brut reste disponible a part)."""
    sentences = []
    for step in resolution:
        older_id, relation, newer_id = _STEP.match(step).groups()  # type: ignore[union-attr]
        sentences.append(replacement_sentence(lookup.unit(older_id), lookup.unit(newer_id), relation))
    return sentences

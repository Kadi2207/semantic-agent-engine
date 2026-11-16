"""Brouillon d'email a la personne qui doit faire une tache, ou demande de clarification.

Le destinataire est la personne qui DOIT faire la tache, lue dans le statement des unites
action/commitment (premier participant nomme = sujet de la phrase), jamais le speaker :
dans "Karim doit signer le devis", c'est Nadia qui parle mais Karim qui signe.
Aucune adresse n'est jamais inventee : recipient.email reste toujours None.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from src.agent.answer import ABSTENTION_THRESHOLD, DATA_DIR, get_graph, get_retriever
from src.agent.intent import strip_instruction
from src.agent.retriever import Hit
from src.memory.corpus import load_participants
from src.schemas import EmailDraft, Recipient, SemanticUnit

TASK_TYPES = frozenset({"action", "commitment"})
TOP_N = 3  # on ne cherche le responsable que parmi les 3 unites les plus pertinentes
FOLLOW_UP_SUBJECT = "Suivi"  # objet d'un brouillon sur une tache en vigueur
UPDATED_SUBJECT = "Point mis à jour"  # objet quand la tache ciblee a ete remplacee ou annulee
TOOL_CANDIDATE = "send_email"  # outil qui enverrait le mail apres confirmation ; jamais appele ici

_MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]


@dataclass(frozen=True)
class Candidate:
    person: str | None  # None : tache sans responsable designe
    unit: SemanticUnit  # la tache a jour
    origin_ids: tuple[str, ...] = ()  # taches perimees retrouvees par la recherche et remplacees par `unit`
    updates: tuple[tuple[SemanticUnit, str], ...] = ()  # unites non-taches qui ont modifie `unit` depuis


@dataclass(frozen=True)
class Clarification:
    query: str
    project_id: str
    reason: str  # "ambiguous" | "no_person" | "off_topic"
    question: str  # la question a poser a l'utilisateur
    candidates: tuple[Candidate, ...] = ()


@lru_cache(maxsize=1)
def get_participants() -> dict[str, list[str]]:
    return load_participants(DATA_DIR)


def responsible_person(statement: str, people: list[str]) -> str | None:
    """Premier participant nomme dans le statement (le sujet de la phrase), ou None."""
    positions = []
    for name in people:
        match = re.search(rf"(?<!\w){re.escape(name)}(?!\w)", statement)
        if match:
            positions.append((match.start(), name))
    return min(positions)[1] if positions else None


def _named_in_query(query: str, people: list[str]) -> set[str]:
    return {name for name in people if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", query, re.IGNORECASE)}


def _french_date(unit: SemanticUnit) -> str:
    d = unit.source_date
    return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"


def _task_phrase(statement: str, person: str) -> str:
    """'Karim doit signer le devis ...' -> 'signer le devis ...' (pour l'objet du mail)."""
    task = statement.strip().rstrip(".")
    task = re.sub(rf"^{re.escape(person)}\s+", "", task)
    task = re.sub(r"^(doit|devra|s'engage à|s’engage à)\s+", "", task)
    return task


def _build_draft(candidate: Candidate) -> EmailDraft:
    unit, person, extra_units = candidate.unit, candidate.person, list(candidate.updates)
    lines = [
        f"Bonjour {person},",
        "",
        f"Lors de la réunion du {_french_date(unit)} ({unit.source_id}), le point suivant a été noté :",
        f"« {unit.statement} »",
    ]
    for later, relation in extra_units:
        verb = "remplacé" if relation == "supersedes" else "contredit"
        lines += [
            "",
            f"Point d'attention : ce point a depuis été {verb} lors de la réunion du "
            f"{_french_date(later)} ({later.source_id}) :",
            f"« {later.statement} »",
        ]
    lines += ["", "Pourriez-vous me faire un retour sur ce sujet ?", "", "Bien cordialement,"]

    source_ids = list(dict.fromkeys([*candidate.origin_ids, unit.unit_id, *(later.unit_id for later, _ in extra_units)]))
    return EmailDraft(
        intent="draft_email",
        tool_candidate=TOOL_CANDIDATE,
        recipient=Recipient(
            name=person,
            email=None,  # jamais inventee : a completer par l'utilisateur
            evidence_unit_ids=[unit.unit_id],
            resolution_status="email_missing",
        ),
        # Tache remplacee ou annulee depuis : l'objet le dit, le corps (qui detaille la mise a jour) ne change pas.
        subject=f"{UPDATED_SUBJECT if candidate.updates or candidate.origin_ids else FOLLOW_UP_SUBJECT} : "
        f"{_task_phrase(unit.statement, person)}",
        body="\n".join(lines),
        source_unit_ids=source_ids,
        needs_confirmation=True,
        executed=False,
    )


def _task_candidate(unit: SemanticUnit, people: list[str]) -> Candidate:
    """Tache a jour + son responsable.

    - tache active : son responsable ;
    - tache perimee remplacee par une autre TACHE : c'est la nouvelle tache qui fait foi, avec son
      responsable a elle (eventuellement aucun : "sans responsable designe") ;
    - tache perimee remplacee par autre chose (ex. decision "signature mise en attente") : la personne
      reste la meme, mais la mise a jour est signalee dans le mail.
    """
    resolved, path = get_graph().resolve(unit, get_retriever())
    if resolved.unit_id == unit.unit_id:
        return Candidate(responsible_person(unit.statement, people), unit)
    relation = "contradicts" if "-contradicts->" in path[-1] else "supersedes"
    if resolved.type in TASK_TYPES:
        return Candidate(responsible_person(resolved.statement, people), resolved, origin_ids=(unit.unit_id,))
    return Candidate(responsible_person(unit.statement, people), unit, updates=((resolved, relation),))


def draft_email(query: str, project_id: str, threshold: float = ABSTENTION_THRESHOLD) -> EmailDraft | Clarification:
    retriever = get_retriever()
    people = get_participants().get(project_id, [])
    topic = strip_instruction(query)

    if retriever.max_dense_score(topic, project_id) < threshold:
        return Clarification(
            query, project_id, "off_topic",
            "Je ne trouve pas ce sujet dans les sources du projet. De quelle tâche, ou à quelle personne, veux-tu écrire ?",
        )

    hits: list[Hit] = retriever.search(topic, project_id, k=TOP_N)
    tasks: list[Candidate] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.unit.type in TASK_TYPES:
            candidate = _task_candidate(hit.unit, people)
            if candidate.unit.unit_id not in seen:  # deux hits peuvent mener a la meme tache a jour
                seen.add(candidate.unit.unit_id)
                tasks.append(candidate)

    # Si la demande nomme explicitement quelqu'un, on ne garde que ses taches (si elle en a).
    named = _named_in_query(query, people)
    if named and any(c.person in named for c in tasks):
        tasks = [c for c in tasks if c.person in named]

    if not tasks or tasks[0].person is None:
        detail = f" La tâche « {tasks[0].unit.statement.rstrip('.')} » n'a pas de responsable désigné." if tasks else ""
        return Clarification(
            query, project_id, "no_person",
            f"Je ne trouve dans les sources aucune personne chargée de cette tâche.{detail} À qui veux-tu écrire ?",
            tuple(tasks[:1]),
        )

    candidates = [c for c in tasks if c.person is not None]
    persons = list(dict.fromkeys(c.person for c in candidates))  # ordre de pertinence, sans doublon
    if len(persons) > 1:
        return Clarification(
            query, project_id, "ambiguous",
            f"Plusieurs personnes peuvent correspondre ({', '.join(persons)}). À laquelle veux-tu écrire ?",
            tuple(candidates),
        )
    return _build_draft(candidates[0])  # la tache la plus pertinente de l'unique responsable

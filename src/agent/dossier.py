"""Dossier documente : constitution deterministe, a partir de la memoire, d'un dossier sur un sujet ou un projet.

1. Perimetre : si la demande nomme le projet ("le dossier du projet Budget marketing") ou ne precise aucun sujet,
   toutes les unites du projet ; sinon, les unites retrouvees par la recherche hybride (memes regles que l'agent :
   seuil d'abstention, questions ecartees des points de depart), puis l'expansion par les relations du graphe
   (dans les deux sens, jusqu'a EXPANSION_DEPTH liens).
2. Sections : objet, decisions en vigueur, historique des changements, conflits non tranches, actions et
   engagements, risques, questions ouvertes, sources citees. Chaque element est une unite de la memoire, avec sa
   reunion, sa date et sa preuve ; aucune phrase n'est generee hors des libelles de section.
Aucun appel a un LLM.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from src.agent.answer import ABSTENTION_THRESHOLD, DATA_DIR, NON_ANSWER_TYPES, get_retriever
from src.agent.display import deadline_note, replacement_sentence, short_date, status_label, type_label
from src.agent.email import get_participants, responsible_person
from src.agent.sources import source_passage
from src.memory.corpus import load_relations
from src.schemas import Relation, SemanticUnit

SEED_COUNT = 3  # points de depart retenus par la recherche
EXPANSION_DEPTH = 2  # liens suivis depuis chaque point de depart
TASK_TYPES = ("action", "commitment")
# Echeance ecrite dans l'enonce : on la cite telle quelle, sans rien calculer (sauf "le jour même", date de reunion).
DEADLINE = re.compile(
    r"(le jour même|le lendemain(?: matin)?|d'ici (?:le )?[\w ]+?(?=[,.]|$)|avant (?:le )?[\w ]+?(?=[,.]|$)|"
    r"au plus tard (?:le )?[\w ]+?(?=[,.]|$)|pour (?:le )?(?:comité du )?\d{1,2} \w+|\d{1,2} (?:janvier|février|mars|"
    r"avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)(?: \d{4})?)",
    re.IGNORECASE,
)
_PREFIX = re.compile(
    r"^\s*(?:(?:peux-tu|pourrais-tu|pouvez-vous|merci de)\s+)?"
    r"(?:constitue[rz]?|fais|faites|faire|prepare[rz]?|monte[rz]?|etabli[rs]?|redige[rz]?|donne|donnez|"
    r"compile[rz]?|assemble[rz]?|genere[rz]?)(?:-moi|-nous)?\s+"
    r"(?:(?:un|le|ce)\s+)?dossier\s*(?:documente\s*)?(?:(?:sur|du|de la|de l'|des|de|concernant|a propos de)\s*)?",
)


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def dossier_topic(query: str) -> str:
    """'Constitue le dossier sur la mise en production d'Atlas' -> 'la mise en production d'atlas' (sans accents)."""
    return _PREFIX.sub("", _plain(query), count=1).strip(" ?.!")


@dataclass(frozen=True)
class Item:
    unit: SemanticUnit
    detail: str = ""  # libelle complementaire (responsable, echeance, remplacement...)
    replaced_by: str = ""  # historique : unite qui remplace ou contredit celle-ci


@dataclass
class Dossier:
    query: str
    project_id: str
    project_name: str
    scope: str  # "projet" (toutes les unites) ou "sujet" (recherche + expansion)
    topic: str
    abstained: bool = False
    seeds: tuple[str, ...] = ()
    decisions: list[Item] = field(default_factory=list)
    history: list[Item] = field(default_factory=list)
    conflicts: list[tuple[Item, Item]] = field(default_factory=list)
    tasks: list[Item] = field(default_factory=list)
    risks: list[Item] = field(default_factory=list)
    open_questions: list[Item] = field(default_factory=list)
    units: list[SemanticUnit] = field(default_factory=list)  # toutes les unites citees, ordre chronologique

    def cited_ids(self) -> set[str]:
        ids = {i.unit.unit_id for section in (self.decisions, self.history, self.tasks, self.risks, self.open_questions) for i in section}
        ids |= {i.unit.unit_id for pair in self.conflicts for i in pair}
        return ids


@lru_cache(maxsize=1)
def _annotated_relations() -> tuple[Relation, ...]:
    return tuple(load_relations(DATA_DIR))


def _ref(unit: SemanticUnit) -> str:
    return f"{unit.unit_id}, réunion {unit.source_id} du {short_date(unit.source_date)}"


def named_project(query: str, project_names: dict[str, str]) -> str | None:
    """Projet nomme explicitement dans la demande ("dossier du projet Budget marketing"), sinon None."""
    text = _plain(query)
    found = [pid for pid, name in project_names.items() if re.search(rf"(?<!\w){re.escape(_plain(name))}(?!\w)", text)]
    return max(found, key=lambda pid: len(project_names[pid])) if found else None


def deadline_of(unit: SemanticUnit) -> str:
    note = deadline_note(unit)
    if note:
        return note
    found = DEADLINE.search(unit.statement)
    return found.group(0) if found else "non précisée dans les sources"


def build_dossier(
    query: str,
    project_id: str,
    project_name: str,
    retriever=None,  # noqa: ANN001
    relations: list[Relation] | tuple[Relation, ...] | None = None,
    participants: dict[str, list[str]] | None = None,
    threshold: float = ABSTENTION_THRESHOLD,
) -> Dossier:
    retriever = retriever or get_retriever()
    relations = _annotated_relations() if relations is None else relations
    people = (participants if participants is not None else get_participants()).get(project_id, [])
    project_units = {h.unit.unit_id: h.unit for h in retriever.search(query, project_id, k=None)}
    topic = dossier_topic(query)
    subject = re.sub(r"^(?:(?:du|le|ce)\s+)?projet(?!\w)\s*", "", topic).strip()
    project_wide = not subject or subject == _plain(project_name)  # "le dossier du projet Atlas" : tout le projet

    dossier = Dossier(query, project_id, project_name, "projet" if project_wide else "sujet", topic)
    if project_wide:
        selected = set(project_units)
    else:
        if retriever.max_dense_score(topic, project_id) < threshold:
            dossier.abstained = True
            return dossier
        hits = [h for h in retriever.search(topic, project_id, k=None) if h.unit.type not in NON_ANSWER_TYPES]
        dossier.seeds = tuple(h.unit.unit_id for h in hits[:SEED_COUNT])
        selected, frontier = set(dossier.seeds), set(dossier.seeds)
        for _ in range(EXPANSION_DEPTH):
            frontier = {
                other
                for r in relations
                for this, other in ((r.from_id, r.to_id), (r.to_id, r.from_id))
                if this in frontier and other in project_units and other not in selected
            }
            selected |= frontier
    units = sorted((project_units[u] for u in selected), key=lambda u: (u.source_date, u.unit_id))
    local = [r for r in relations if r.from_id in selected and r.to_id in selected]

    contradictions = [r for r in local if r.relation == "contradicts"]
    open_pairs = [r for r in contradictions if project_units[r.from_id].status == "active" == project_units[r.to_id].status]
    in_conflict = {u for r in open_pairs for u in (r.from_id, r.to_id)}
    dossier.conflicts = [(Item(project_units[r.to_id]), Item(project_units[r.from_id])) for r in open_pairs]
    dossier.decisions = [Item(u) for u in units if u.type == "decision" and u.status == "active" and u.unit_id not in in_conflict]
    for r in sorted(local, key=lambda r: (project_units[r.to_id].source_date, r.to_id)):
        older, newer = project_units[r.to_id], project_units[r.from_id]
        if r.relation == "supersedes" or (r.relation == "contradicts" and older.status != "active"):
            dossier.history.append(Item(older, replacement_sentence(older, newer, r.relation), newer.unit_id))
    for u in units:
        if u.type in TASK_TYPES:
            person = responsible_person(u.statement, people) or "aucun responsable désigné"
            dossier.tasks.append(Item(u, f"responsable : {person} ; échéance : {deadline_of(u)} ; statut : {status_label(u.status)}"))
        elif u.type == "risk":
            dossier.risks.append(Item(u, f"statut : {status_label(u.status)}"))
    answered = {r.to_id for r in relations if r.relation == "answers"}
    dossier.open_questions = [Item(u) for u in units if u.type == "question" and u.status == "active" and u.unit_id not in answered]
    cited = dossier.cited_ids()
    dossier.units = [u for u in units if u.unit_id in cited]
    return dossier


def to_markdown(d: Dossier) -> str:
    """Dossier complet en Markdown (telechargement) : chaque element renvoie a son unite et a sa source."""
    lines = [f"# Dossier : {d.query.strip()}", ""]
    lines += ["## Objet", ""]
    scope = "toutes les unités du projet" if d.scope == "projet" else "unités retrouvées par la recherche, puis reliées par les relations du graphe"
    lines += [f"Projet {d.project_name} ({d.project_id}). Périmètre : {scope}."]
    if d.abstained:
        return "\n".join(lines + ["", "Information absente ou incertaine dans les sources : aucun dossier n'est constitué.", ""])
    if d.units:
        lines += [f"{len(d.units)} unités citées, du {short_date(d.units[0].source_date)} au {short_date(d.units[-1].source_date)}."]

    def section(title: str, items: list[Item], empty: str) -> None:
        lines.extend(["", f"## {title}", ""])
        if not items:
            lines.append(empty)
        for item in items:
            extra = f" ({item.detail})" if item.detail else ""
            lines.append(f"- {item.unit.statement}{extra} [{_ref(item.unit)}]")

    section("Décisions en vigueur", d.decisions, "Aucune décision en vigueur dans ce périmètre.")
    lines += ["", "## Historique des changements", ""]
    if not d.history:
        lines.append("Aucun remplacement ni contradiction résolue dans ce périmètre.")
    for item in d.history:
        lines.append(
            f"- {item.detail} Ancienne version ({type_label(item.unit)}) : {item.unit.statement} [{_ref(item.unit)}], "
            f"remplacée dans la mémoire par {item.replaced_by}"
        )
    lines += ["", "## Conflits non tranchés", ""]
    if not d.conflicts:
        lines.append("Aucun conflit non tranché dans ce périmètre.")
    for a, b in d.conflicts:
        lines.append("- Deux affirmations contradictoires, toutes deux en vigueur ; les sources ne permettent pas de trancher :")
        lines.append(f"  - {a.unit.statement} [{_ref(a.unit)}]")
        lines.append(f"  - {b.unit.statement} [{_ref(b.unit)}]")
    section("Actions et engagements", d.tasks, "Aucune action ni engagement dans ce périmètre.")
    section("Risques", d.risks, "Aucun risque dans ce périmètre.")
    section("Questions ouvertes", d.open_questions, "Aucune question restée sans réponse dans ce périmètre.")
    lines += ["", "## Sources citées", ""]
    for u in d.units:
        passage = source_passage(u, context=0)
        speaker = passage.lines[0].speaker if passage.found else (u.speaker or "inconnu")
        lines.append(f"- {u.unit_id}, réunion {u.source_id} du {short_date(u.source_date)}, {speaker} : « {u.evidence} »")
    return "\n".join(lines) + "\n"

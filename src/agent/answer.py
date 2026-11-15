"""Reponse sourcee a une question, avec abstention quand les sources ne couvrent pas le sujet.

Deux garde-fous apres la recherche hybride :
- une unite de type "question" n'est jamais une reponse : elle est ecartee des candidats ;
- si la meilleure unite est perimee (superseded/cancelled), on suit les relations
  supersedes/contradicts jusqu'a l'unite active qui la remplace (from_id = la plus recente).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from src.agent.retriever import ROOT, HybridRetriever
from src.memory.corpus import load_corpus, load_relations
from src.schemas import Relation, SemanticUnit

DATA_DIR = ROOT / "data" / "raw"
# Le cosinus bi-encodeur ne separe pas 'hors sujet' de 'sur le sujet mais sans reponse dans les sources'
# (les deux groupes se chevauchent sur le dev). 0.45 ecarte de maniere fiable le hors-sujet. Distinguer
# finement une question sans reponse releve d'un cross-encoder / NLI (piste documentee dans le rapport).
ABSTENTION_THRESHOLD = 0.45
ABSTENTION_MESSAGE = "Information absente ou incertaine dans les sources"
NON_ANSWER_TYPES = frozenset({"question"})
REPLACING_RELATIONS = frozenset({"supersedes", "contradicts"})


@dataclass(frozen=True)
class Answer:
    query: str
    project_id: str
    abstained: bool
    best_dense_score: float  # meilleur cosinus e5 sur le projet, compare au seuil
    threshold: float
    statement: str | None = None
    evidence: str | None = None
    source_id: str | None = None
    source_date: date | None = None
    speaker: str | None = None
    unit_id: str | None = None
    unit_status: str | None = None
    fused_score: float | None = None  # score RRF de l'unite retrouvee (avant resolution)
    # Chemin de resolution si l'unite retrouvee etait perimee, ex. ["p02_u01 -contradicts-> p02_u08"].
    resolution: tuple[str, ...] = ()
    message: str | None = None


class ReplacementGraph:
    """Pour chaque unite, les unites plus recentes qui la remplacent (supersedes) ou la contredisent."""

    def __init__(self, relations: list[Relation]) -> None:
        self._replaced_by: dict[str, list[tuple[str, str]]] = {}
        for rel in relations:
            if rel.relation in REPLACING_RELATIONS:
                self._replaced_by.setdefault(rel.to_id, []).append((rel.from_id, rel.relation))

    def resolve(self, start: SemanticUnit, lookup: HybridRetriever) -> tuple[SemanticUnit, tuple[str, ...]]:
        """Remonte la chaine depuis une unite perimee jusqu'a une unite active.

        Parcours en largeur ; parmi plusieurs remplacants, l'actif le plus recent gagne.
        Si aucune unite active n'est atteignable, renvoie l'unite de depart (son statut reste visible).
        """
        if start.status == "active":
            return start, ()
        frontier = [(start, ())]
        seen = {start.unit_id}
        while frontier:
            next_frontier = []
            found = []
            for unit, path in frontier:
                for replacer_id, relation in self._replaced_by.get(unit.unit_id, []):
                    if replacer_id in seen:
                        continue
                    seen.add(replacer_id)
                    replacer = lookup.unit(replacer_id)
                    step = path + (f"{unit.unit_id} ({unit.status}) -{relation}-> {replacer_id} ({replacer.status})",)
                    (found if replacer.status == "active" else next_frontier).append((replacer, step))
            if found:
                return max(found, key=lambda item: (item[0].source_date, item[0].unit_id))
            frontier = next_frontier
        return start, ()


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Construit une seule fois le retriever (chargement du modele + indexation)."""
    return HybridRetriever(load_corpus(DATA_DIR))


@lru_cache(maxsize=1)
def get_graph() -> ReplacementGraph:
    return ReplacementGraph(load_relations(DATA_DIR))


def answer(
    query: str,
    project_id: str,
    threshold: float = ABSTENTION_THRESHOLD,
    retriever: HybridRetriever | None = None,
    graph: ReplacementGraph | None = None,
) -> Answer:
    retriever = retriever or get_retriever()
    graph = graph or get_graph()
    candidates = [hit for hit in retriever.search(query, project_id, k=None) if hit.unit.type not in NON_ANSWER_TYPES]
    best_dense = retriever.max_dense_score(query, project_id)

    if not candidates or best_dense < threshold:
        return Answer(
            query=query,
            project_id=project_id,
            abstained=True,
            best_dense_score=best_dense,
            threshold=threshold,
            message=ABSTENTION_MESSAGE,
        )

    top = candidates[0]
    unit, resolution = graph.resolve(top.unit, retriever)
    return Answer(
        query=query,
        project_id=project_id,
        abstained=False,
        best_dense_score=best_dense,
        threshold=threshold,
        statement=unit.statement,
        evidence=unit.evidence,
        source_id=unit.source_id,
        source_date=unit.source_date,
        speaker=unit.speaker,
        unit_id=unit.unit_id,
        unit_status=unit.status,
        fused_score=top.score,
        resolution=resolution,
    )

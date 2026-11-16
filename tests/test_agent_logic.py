"""Logique de l'agent sans modele d'embeddings : intention, brouillon d'email, responsable, fusion RRF, resolution.

Projets de developpement uniquement (01 a 10). La recherche est remplacee par un classement fixe, pour tester
la logique qui suit la recherche independamment du modele.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agent import email as email_module
from src.agent.answer import ReplacementGraph
from src.agent.email import Clarification, draft_email, responsible_person
from src.agent.intent import detect_intent, strip_instruction
from src.agent.retriever import RRF_K, Hit, HybridRetriever
from src.memory.corpus import load_corpus, load_participants, load_relations
from src.schemas import EmailDraft

ROOT = Path(__file__).resolve().parent.parent
CORPUS = load_corpus(ROOT / "data" / "raw")
UNITS = {c.unit.unit_id: c.unit for c in CORPUS}
PEOPLE = load_participants(ROOT / "data" / "raw")


class FixedRetriever:
    """Classement impose : meme interface que HybridRetriever pour draft_email et ReplacementGraph."""

    def __init__(self, ranking: list[str], dense: float = 0.9) -> None:
        self.ranking, self.dense = ranking, dense

    def unit(self, unit_id: str):  # noqa: ANN201
        return UNITS[unit_id]

    def max_dense_score(self, query: str, project_id: str) -> float:
        return self.dense

    def search(self, query: str, project_id: str, k: int | None = 5) -> list[Hit]:
        hits = [Hit(UNITS[u], 1.0 / (RRF_K + i), self.dense, i, i) for i, u in enumerate(self.ranking, start=1)]
        return hits[:k]


@pytest.fixture
def fixed(monkeypatch):  # noqa: ANN001, ANN201
    def install(ranking: list[str], dense: float = 0.9) -> None:
        monkeypatch.setattr(email_module, "get_retriever", lambda: FixedRetriever(ranking, dense))

    return install


# --------------------------------------------------------------------------- intention


@pytest.mark.parametrize(
    "query, intent",
    [
        ("Écris un mail à la personne qui doit signer le devis d'Horizon", "draft_email"),
        ("Rédige un message pour Karim", "draft_email"),
        ("Contacte le responsable du salon", "draft_email"),
        ("Envoie un e-mail à Julien", "draft_email"),
        ("Peux-tu envoyer un mail à Karim ?", "draft_email"),
        ("Qui doit envoyer le mail aux utilisateurs ?", "answer"),  # question SUR un mail, pas une demande
        ("Quelle est la date de mise en production d'Atlas ?", "answer"),
        ("Qui écrit le script de purge ?", "answer"),
    ],
)
def test_detect_intent(query: str, intent: str) -> None:
    assert detect_intent(query) == intent


def test_strip_instruction_keeps_only_the_topic() -> None:
    assert strip_instruction("Écris un mail à la personne qui doit signer le devis d'Horizon") == "doit signer le devis d'Horizon"
    assert strip_instruction("Quelle est la date ?") == "Quelle est la date ?"


# --------------------------------------------------------------------------- brouillon d'email


def test_draft_email_normal_case_never_sends_and_never_invents_an_address(fixed) -> None:  # noqa: ANN001
    fixed(["p02_u04", "p02_u11", "p02_u10"])
    draft = draft_email("Écris un mail à la personne qui doit signer le devis d'Horizon", "projet_02")
    assert isinstance(draft, EmailDraft)
    assert draft.recipient.name == "Karim"  # celui qui doit signer, pas Nadia qui le demande
    assert UNITS["p02_u04"].speaker == "Nadia"
    assert draft.recipient.email is None and draft.recipient.resolution_status == "email_missing"
    assert draft.needs_confirmation is True and draft.executed is False
    assert draft.source_unit_ids == ["p02_u04", "p02_u11"]  # la tache annulee et la decision qui la met en attente
    assert "mise en attente" in draft.body


def test_draft_email_ambiguous_recipient_asks_instead_of_drafting(fixed) -> None:  # noqa: ANN001
    fixed(["p03_u10", "p03_u11", "p03_u09"])
    result = draft_email("Écris un mail à la personne qui s'occupe de la grille d'évaluation", "projet_03")
    assert isinstance(result, Clarification) and result.reason == "ambiguous"
    assert [c.person for c in result.candidates] == ["Olivier", "Inès"]


def test_draft_email_without_any_responsible_asks_instead_of_inventing(fixed) -> None:  # noqa: ANN001
    fixed(["p02_u08", "p02_u09", "p02_u01"])  # aucune action ni engagement parmi les unites retrouvees
    result = draft_email("Écris un mail au responsable du budget", "projet_02")
    assert isinstance(result, Clarification) and result.reason == "no_person"
    assert result.candidates == ()


def test_draft_email_off_topic_asks_for_clarification(fixed) -> None:  # noqa: ANN001
    fixed(["p02_u04"], dense=0.1)
    result = draft_email("Écris un mail au directeur technique", "projet_02")
    assert isinstance(result, Clarification) and result.reason == "off_topic"


# --------------------------------------------------------------------------- regle du responsable


def test_responsible_is_the_first_participant_named_in_the_statement() -> None:
    assert responsible_person(UNITS["p02_u04"].statement, PEOPLE["projet_02"]) == "Karim"
    assert responsible_person(UNITS["p03_u07"].statement, PEOPLE["projet_03"]) == "Inès"  # "Inès est sollicitée par Olivier"
    assert responsible_person(UNITS["p07_u16"].statement, PEOPLE["projet_07"]) == "Manon"  # "... à Sébastien"
    assert responsible_person("La salle doit être réservée avant vendredi.", PEOPLE["projet_02"]) is None


# --------------------------------------------------------------------------- fusion RRF


class _Bm25:
    def __init__(self, ranking: list[str]) -> None:
        self.ranking = ranking

    def search(self, query: str, k: int, project_id: str) -> list[str]:
        return self.ranking[:k]


class _Dense:
    def __init__(self, ranking: list[str]) -> None:
        self.ranking = ranking

    def search_with_scores(self, query: str, k: int, project_id: str) -> list[tuple[str, float]]:
        return [(u, 1.0 - 0.1 * i) for i, u in enumerate(self.ranking[:k])]


def hybrid(bm25: list[str], dense: list[str]) -> HybridRetriever:
    """HybridRetriever reel (meme code de fusion), avec deux classements imposes a la place des modeles."""
    retriever = object.__new__(HybridRetriever)
    retriever._rrf_k = RRF_K
    retriever._bm25, retriever._dense = _Bm25(bm25), _Dense(dense)
    retriever._units = UNITS
    retriever._project_sizes = {"projet_01": len(bm25)}
    return retriever


def test_rrf_scores_and_order() -> None:
    assert RRF_K == 60
    hits = hybrid(["p01_u20", "p01_u21", "p01_u02"], ["p01_u21", "p01_u02", "p01_u20"]).search("q", "projet_01", k=None)
    assert [h.unit.unit_id for h in hits] == ["p01_u21", "p01_u20", "p01_u02"]
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert hits[1].score == pytest.approx(1 / 61 + 1 / 63)
    assert (hits[0].bm25_rank, hits[0].dense_rank) == (2, 1)


def test_rrf_ties_prefer_active_units_then_the_most_recent_source() -> None:
    # p01_u12 (superseded) et p01_u20 (active) ont le meme score fusionne : 1/61 + 1/62
    hits = hybrid(["p01_u12", "p01_u20"], ["p01_u20", "p01_u12"]).search("q", "projet_01", k=None)
    assert hits[0].score == pytest.approx(hits[1].score)
    assert [h.unit.unit_id for h in hits] == ["p01_u20", "p01_u12"]
    # deux unites actives a egalite : la source la plus recente d'abord (p01_u21, 9 avril, avant p01_u13, 31 mars)
    hits = hybrid(["p01_u13", "p01_u21"], ["p01_u21", "p01_u13"]).search("q", "projet_01", k=None)
    assert [h.unit.unit_id for h in hits] == ["p01_u21", "p01_u13"]


def test_unknown_project_is_rejected() -> None:
    with pytest.raises(KeyError):
        hybrid(["p01_u20"], ["p01_u20"]).search("q", "projet_99")


# --------------------------------------------------------------------------- ReplacementGraph.resolve


def test_resolve_on_development_projects() -> None:
    graph, lookup = ReplacementGraph(load_relations(ROOT / "data" / "raw")), FixedRetriever([])
    assert graph.resolve(UNITS["p01_u20"], lookup) == (UNITS["p01_u20"], ())  # deja active : rien a faire
    assert graph.resolve(UNITS["p01_u12"], lookup)[0].unit_id == "p01_u20"  # un remplacement
    assert graph.resolve(UNITS["p01_u01"], lookup)[0].unit_id == "p01_u20"  # deux remplacements
    resolved, path = graph.resolve(UNITS["p02_u04"], lookup)  # action annulee -> decision de mise en attente
    assert resolved.unit_id == "p02_u11" and path == ("p02_u04 (cancelled) -supersedes-> p02_u11 (active)",)
    for unit in UNITS.values():  # toute unite perimee des projets de developpement mene a une unite active
        if unit.status != "active" and unit.unit_id[:3] not in ("p11", "p12"):
            assert graph.resolve(unit, lookup)[0].status == "active", unit.unit_id

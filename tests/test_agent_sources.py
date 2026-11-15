"""Historique des decisions remplacees et passages sources (sans modele d'embeddings)."""
from __future__ import annotations

from src.agent.answer import ReplacementGraph
from src.agent.sources import source_passage
from src.memory.corpus import load_corpus, load_relations
from src.schemas import Relation


def test_history_walks_back_the_whole_replacement_chain() -> None:
    graph = ReplacementGraph(
        [
            Relation(from_id="u12", to_id="u01", relation="supersedes", evidence_source_id="m2"),
            Relation(from_id="u20", to_id="u12", relation="supersedes", evidence_source_id="m3"),
            Relation(from_id="u20", to_id="u21", relation="creates_action", evidence_source_id="m3"),  # ignoree
        ]
    )
    assert graph.history("u20") == (("u12", "supersedes", "u20"), ("u01", "supersedes", "u12"))
    assert graph.history("u01") == ()


def test_atlas_demo_chain_and_every_evidence_has_a_source_passage() -> None:
    graph = ReplacementGraph(load_relations("data/raw"))
    assert graph.history("p01_u20") == (("p01_u12", "supersedes", "p01_u20"), ("p01_u01", "supersedes", "p01_u12"))
    corpus = load_corpus("data/raw")
    missing = [c.unit.unit_id for c in corpus if not source_passage(c.unit).found]
    assert missing == []
    passage = source_passage(next(c.unit for c in corpus if c.unit.unit_id == "p01_u20"), context=1)
    assert [line.is_evidence for line in passage.lines] == [False, True, False]
    assert passage.lines[1].speaker == "Aminata"

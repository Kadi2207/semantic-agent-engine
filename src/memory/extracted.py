"""Memoire extraite : alternative a la memoire annotee, lue par le meme chargeur et le meme agent.

data/extracted/ contient, pour les projets 04 a 12, les unites et relations produites par le pipeline
d'extraction (rejeu du cache, scripts/build_extracted_memory.py), sans correction manuelle, au format de data/raw.
Les projets 01 a 03 n'y figurent pas : ils ont servi a regler la consigne d'extraction.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXTRACTED_DIR = ROOT / "data" / "extracted"
EXTRACTED_PROJECTS = tuple(range(4, 13))
MENTION = "unités et relations produites par le pipeline d'extraction, sans correction manuelle"


@lru_cache(maxsize=1)
def get_extracted_retriever():  # noqa: ANN201
    """Meme retriever hybride que l'agent, construit sur la memoire extraite."""
    from src.agent.retriever import HybridRetriever
    from src.memory.corpus import load_corpus

    return HybridRetriever(load_corpus(EXTRACTED_DIR))


@lru_cache(maxsize=1)
def get_extracted_graph():  # noqa: ANN201
    from src.agent.answer import ReplacementGraph
    from src.memory.corpus import load_relations

    return ReplacementGraph(load_relations(EXTRACTED_DIR))

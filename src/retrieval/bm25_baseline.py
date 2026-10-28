from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.memory.corpus import CorpusUnit

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, corpus: list[CorpusUnit]) -> None:
        grouped: dict[str, list[CorpusUnit]] = {}
        for item in corpus:
            grouped.setdefault(item.project_id, []).append(item)
        self._ids: dict[str, list[str]] = {}
        self._indexes: dict[str, BM25Okapi] = {}
        for project_id, items in grouped.items():
            self._ids[project_id] = [item.unit.unit_id for item in items]
            self._indexes[project_id] = BM25Okapi([tokenize(item.text) for item in items])

    def search(self, query: str, k: int, project_id: str) -> list[str]:
        scores = self._indexes[project_id].get_scores(tokenize(query))
        ids = self._ids[project_id]
        order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
        return [ids[i] for i in order[:k]]

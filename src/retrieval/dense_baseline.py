from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.memory.corpus import CorpusUnit

MODEL_NAME = "intfloat/multilingual-e5-small"


class DenseRetriever:
    def __init__(self, corpus: list[CorpusUnit], model_name_or_path: str | Path = MODEL_NAME) -> None:
        # Nom de modele Hugging Face ou dossier local (ex. models/e5-finetuned) ;
        # les prefixes e5 "query: " / "passage: " s'appliquent dans les deux cas.
        self._model = SentenceTransformer(str(model_name_or_path), device="cpu")
        grouped: dict[str, list[CorpusUnit]] = {}
        for item in corpus:
            grouped.setdefault(item.project_id, []).append(item)
        self._ids: dict[str, list[str]] = {}
        self._indexes: dict[str, faiss.IndexFlatIP] = {}
        for project_id, items in grouped.items():
            embeddings = self._encode([f"passage: {item.text}" for item in items])
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self._ids[project_id] = [item.unit.unit_id for item in items]
            self._indexes[project_id] = index

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def search(self, query: str, k: int, project_id: str) -> list[str]:
        return [unit_id for unit_id, _ in self.search_with_scores(query, k, project_id)]

    def search_with_scores(self, query: str, k: int, project_id: str) -> list[tuple[str, float]]:
        """Comme search, mais renvoie aussi le cosinus (embeddings normalises => produit scalaire)."""
        index = self._indexes[project_id]
        ids = self._ids[project_id]
        scores, positions = index.search(self._encode([f"query: {query}"]), min(k, len(ids)))
        return [(ids[i], float(s)) for i, s in zip(positions[0], scores[0]) if i >= 0]

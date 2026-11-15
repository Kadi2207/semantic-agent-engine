"""Recherche hybride BM25 + dense (e5 fine-tune) fusionnee par Reciprocal Rank Fusion."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.memory.corpus import CorpusUnit
from src.retrieval.bm25_baseline import BM25Retriever
from src.retrieval.dense_baseline import DenseRetriever
from src.schemas import SemanticUnit

ROOT = Path(__file__).resolve().parent.parent.parent
FINETUNED_PATH = ROOT / "models" / "e5-finetuned"
RRF_K = 60

# A score fusionne egal, l'etat actuel (active) passe devant l'historique.
_STATUS_PRIORITY = {"active": 0, "superseded": 1, "cancelled": 2}
# Les scores RRF sont des sommes de fractions : on arrondit pour que deux
# scores mathematiquement egaux soient bien traites comme egaux.
_TIE_DECIMALS = 9


@dataclass(frozen=True)
class Hit:
    unit: SemanticUnit
    score: float  # score RRF fusionne
    dense_score: float  # cosinus e5 de cette unite avec la question
    bm25_rank: int  # rang 1-based dans le classement BM25
    dense_rank: int  # rang 1-based dans le classement dense


class HybridRetriever:
    def __init__(
        self,
        corpus: list[CorpusUnit],
        model_path: str | Path = FINETUNED_PATH,
        rrf_k: int = RRF_K,
    ) -> None:
        if not (Path(model_path) / "config.json").exists():
            raise FileNotFoundError(
                f"Modele fine-tune introuvable dans {model_path}. Lance d'abord scripts/train_embeddings.py."
            )
        self._rrf_k = rrf_k
        self._bm25 = BM25Retriever(corpus)
        self._dense = DenseRetriever(corpus, model_path)
        self._units: dict[str, SemanticUnit] = {item.unit.unit_id: item.unit for item in corpus}
        self._project_sizes: dict[str, int] = {}
        for item in corpus:
            self._project_sizes[item.project_id] = self._project_sizes.get(item.project_id, 0) + 1

    @property
    def project_ids(self) -> list[str]:
        return sorted(self._project_sizes)

    def unit(self, unit_id: str) -> SemanticUnit:
        return self._units[unit_id]

    def search(self, query: str, project_id: str, k: int | None = 5) -> list[Hit]:
        """Les k unites du projet les mieux classees apres fusion RRF (k=None : tout le projet)."""
        if project_id not in self._project_sizes:
            raise KeyError(f"project_id inconnu : {project_id!r} (connus : {', '.join(self.project_ids)})")
        n = self._project_sizes[project_id]

        bm25_ranking = self._bm25.search(query, n, project_id)
        dense_ranking = self._dense.search_with_scores(query, n, project_id)

        bm25_rank = {unit_id: rank for rank, unit_id in enumerate(bm25_ranking, start=1)}
        dense_rank = {unit_id: rank for rank, (unit_id, _) in enumerate(dense_ranking, start=1)}
        dense_score = dict(dense_ranking)

        hits = []
        for unit_id in bm25_rank.keys() | dense_rank.keys():
            # Une unite absente d'un classement n'y contribue pas (cas impossible ici : classements complets).
            fused = sum(1.0 / (self._rrf_k + ranks[unit_id]) for ranks in (bm25_rank, dense_rank) if unit_id in ranks)
            hits.append(
                Hit(
                    unit=self._units[unit_id],
                    score=fused,
                    dense_score=dense_score.get(unit_id, float("nan")),
                    bm25_rank=bm25_rank.get(unit_id, n + 1),
                    dense_rank=dense_rank.get(unit_id, n + 1),
                )
            )

        hits.sort(
            key=lambda hit: (
                -round(hit.score, _TIE_DECIMALS),
                _STATUS_PRIORITY[hit.unit.status],
                -hit.unit.source_date.toordinal(),  # puis la source la plus recente
                hit.unit.unit_id,
            )
        )
        return hits[:k]

    def max_dense_score(self, query: str, project_id: str) -> float:
        """Meilleur cosinus e5 entre la question et une unite du projet."""
        best = self._dense.search_with_scores(query, 1, project_id)
        return best[0][1] if best else float("-inf")

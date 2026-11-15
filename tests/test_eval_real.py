"""Correspondance des intervalles QMSum et metriques a plusieurs passages pertinents (sans donnees externes)."""
from __future__ import annotations

import math

from eval.eval_real import clean_turn, passage_text, query_metrics, rank_by_scores, relevant_indices, rrf


def test_spans_are_inclusive_turn_indices_given_as_strings() -> None:
    indices, reason = relevant_indices([["2", "4"]], n_turns=10)
    assert reason is None
    assert indices == {2, 3, 4}  # borne de fin incluse


def test_single_turn_span_and_union_of_all_intervals() -> None:
    indices, reason = relevant_indices([["0", "0"], ["5", "6"], ["6", "7"]], n_turns=8)
    assert reason is None
    assert indices == {0, 5, 6, 7}  # dernier tour (7) inclus, chevauchement compte une fois


def test_invalid_spans_are_reported_not_silently_kept() -> None:
    assert relevant_indices([], n_turns=5) == (set(), "aucun intervalle")
    indices, reason = relevant_indices([["3", "5"]], n_turns=5)  # 5 = nombre de tours : hors bornes
    assert indices == set() and "hors bornes" in reason
    indices, reason = relevant_indices([["4", "2"]], n_turns=10)
    assert indices == set() and "hors bornes" in reason


def test_markup_is_removed_but_empty_turns_stay_empty() -> None:
    assert clean_turn("Yeah {vocalsound} , okay {disfmarker}  so") == "Yeah , okay so"
    assert passage_text({"speaker": "Marketing", "content": "{vocalsound}"}) == ""
    assert passage_text({"speaker": "Marketing", "content": "Fine ."}) == "Marketing: Fine ."


def test_metrics_count_every_relevant_passage() -> None:
    ranking = [7, 2, 9, 3, 1, 0, 4, 5, 6, 8]
    relevant = {2, 3, 4, 8}  # 4 pertinents, aux rangs 2, 4, 7 et 10
    m = query_metrics(ranking, relevant)
    assert m["recall@5"] == 2 / 4
    assert m["recall@10"] == 4 / 4
    assert m["hit@5"] == 1.0 and m["hit@10"] == 1.0
    miss = query_metrics([0, 1, 5, 6, 7, 9, 2], {2})  # seul pertinent au rang 7
    assert miss["hit@5"] == 0.0 and miss["recall@5"] == 0.0 and miss["mrr"] == 1 / 7
    assert m["mrr"] == 1 / 2
    dcg = 1 / math.log2(3) + 1 / math.log2(5) + 1 / math.log2(8) + 1 / math.log2(11)
    idcg = sum(1 / math.log2(r + 1) for r in range(1, 5))
    assert math.isclose(m["ndcg@10"], dcg / idcg)
    assert m["recall@5_plafond"] == 1.0


def test_recall_ceiling_when_more_relevant_than_k() -> None:
    relevant = set(range(20))
    m = query_metrics(list(range(30)), relevant)  # classement parfait
    assert m["recall@5"] == m["recall@5_plafond"] == 5 / 20
    assert m["ndcg@10"] == 1.0 and m["mrr"] == 1.0


def test_rankings_are_deterministic_and_rrf_fuses_full_rankings() -> None:
    assert rank_by_scores([0.5, 0.9, 0.5, 0.1]) == [1, 0, 2, 3]  # egalite departagee par l'indice
    fused = rrf([[0, 1, 2], [2, 1, 0]])
    # 0 et 2 : 1/61 + 1/63 = 0.032266 (egalite, departagee par l'indice) ; 1 : 2/62 = 0.032258
    assert fused == [0, 2, 1]
    assert rrf([[3, 1], [1, 3]]) == [1, 3]  # egalite parfaite : plus petit indice d'abord

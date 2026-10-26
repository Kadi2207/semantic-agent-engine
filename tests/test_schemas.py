from __future__ import annotations

import math
from datetime import date

import pytest
from pydantic import ValidationError

from eval.metrics import macro_f1, mrr, ndcg_at_k, recall_at_k
from src.schemas import EmailDraft, Recipient, SemanticUnit


def _unit_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "unit_id": "u1",
        "source_id": "s1",
        "source_date": date(2025, 1, 15),
        "type": "decision",
        "statement": "Le lancement est reporte au mois prochain.",
        "speaker": "Alice",
        "evidence": "On reporte le lancement.",
        "status": "active",
    }
    payload.update(overrides)
    return payload


def _draft_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "intent": "draft_email",
        "tool_candidate": "send_email",
        "recipient": Recipient(
            name="Bob",
            email="bob@example.com",
            evidence_unit_ids=["u1"],
            resolution_status="resolved",
        ),
        "subject": "Report du lancement",
        "body": "Bonjour Bob, le lancement est reporte.",
        "source_unit_ids": ["u1"],
    }
    payload.update(overrides)
    return payload


def test_semantic_unit_valid() -> None:
    unit = SemanticUnit(**_unit_payload())
    assert unit.type == "decision"
    assert unit.speaker == "Alice"
    assert unit.source_date == date(2025, 1, 15)


def test_semantic_unit_rejects_invalid_type() -> None:
    with pytest.raises(ValidationError):
        SemanticUnit(**_unit_payload(type="opinion"))


def test_email_draft_defaults() -> None:
    draft = EmailDraft(**_draft_payload())
    assert draft.needs_confirmation is True
    assert draft.executed is False


def test_email_draft_rejects_executed_true() -> None:
    with pytest.raises(ValidationError):
        EmailDraft(**_draft_payload(executed=True))


def test_recall_at_k() -> None:
    # 2 pertinents sur 3 retrouves dans le top-3 : 2/3.
    assert recall_at_k(["a", "b", "c", "d"], {"a", "c", "e"}, 3) == pytest.approx(2 / 3)
    assert recall_at_k([], {"a"}, 3) == 0.0
    assert recall_at_k(["a"], set(), 3) == 0.0


def test_mrr() -> None:
    # Premier pertinent au rang 3 : 1/3.
    assert mrr(["x", "y", "a"], {"a", "b"}) == pytest.approx(1 / 3)
    assert mrr(["x", "y"], {"a"}) == 0.0
    assert mrr([], {"a"}) == 0.0


def test_ndcg_at_k() -> None:
    # DCG = 1/log2(2) + 1/log2(4) = 1.5 ; IDCG = 1 + 1/log2(3).
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(expected)
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["x"], {"a"}, 3) == 0.0
    assert ndcg_at_k([], set(), 3) == 0.0


def test_macro_f1() -> None:
    # Classe a : P=1, R=1/2, F1=2/3. Classe b : P=2/3, R=1, F1=4/5.
    # Moyenne : (2/3 + 4/5) / 2 = 11/15.
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    assert macro_f1(y_true, y_pred) == pytest.approx(11 / 15)
    assert macro_f1([], []) == 0.0
    assert macro_f1(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

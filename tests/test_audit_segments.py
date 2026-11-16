"""Decoupage des reunions longues pour l'audit : indices absolus, pas de chevauchement, ordre conserve."""
from __future__ import annotations

from eval import audit_real_extraction as audit


def test_segments_cover_every_turn_once_with_absolute_indices(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(audit, "SEGMENT_TURNS", 4)
    turns = [{"speaker": "PM" if i % 2 else "UI", "content": f"phrase {i}"} for i in range(10)]
    turns[5]["content"] = "{vocalsound}"  # vide apres nettoyage
    segments, lines = audit.segment_meeting("IS1003a", turns)

    assert [s["source_id"] for s in segments] == ["IS1003a#t0000-0003", "IS1003a#t0004-0007", "IS1003a#t0008-0009"]
    assert sorted(s["source_id"] for s in segments) == [s["source_id"] for s in segments]  # ordre chronologique
    covered = [i for seg in lines.values() for i, _ in seg]
    assert covered == [0, 1, 2, 3, 4, 6, 7, 8, 9]  # chaque tour non vide une seule fois, indice absolu
    assert segments[1]["transcript"] == "UI: phrase 4\nUI: phrase 6\nPM: phrase 7"
    assert segments[0]["participants"] == ["PM", "UI"]


def test_groq_failure_aborts_the_whole_audit_without_leaking_the_error_body() -> None:
    import pytest

    from src.ingest.extract import extract_project
    from src.ingest.groq_client import GroqError

    class Failing:
        cache_only = False

        def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            raise GroqError("HTTP 429 : Rate limit reached ... in organization `org_secret123` ... tokens per day (TPD)")

    segments, _ = audit.segment_meeting("IS1003a", [{"speaker": "PM", "content": "Hello ."}] * 3)
    with pytest.raises(audit.AuditAborted) as info:
        extract_project("IS1003a", segments, audit.FailFastClient(Failing()))  # type: ignore[arg-type]
    assert "quota quotidien" in str(info.value)
    assert "org_secret123" not in str(info.value)


def test_turn_index_is_absolute() -> None:
    lines = [(40, "PM: Okay, let's start the project."), (41, "UI: We need a remote control.")]
    assert audit.turn_index("We need a remote control.", lines) == 41
    assert audit.turn_index("Something else.", lines) is None

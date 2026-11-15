"""Validation de l'extraction (src/ingest/extract.py) sans appel reseau : le LLM est simule."""
from __future__ import annotations

import json

from src.ingest.extract import extract_project, extract_units

MEETING = {
    "source_id": "m1",
    "source_date": "2026-01-10",
    "participants": ["Paul", "Nina"],
    "transcript": (
        "Paul: Bon, on commence.\n"
        "Paul: Nina, tu pourrais relancer le fournisseur ?\n"
        "Nina: Oui, je le relance demain matin.\n"
        "Paul: Les tests de charge passent, donc on lance lundi."
    ),
}


class FakeClient:
    """Renvoie des reponses fixees dans l'ordre ; meme interface que CachedGroq.complete."""

    cache_only = False

    def __init__(self, *responses: dict[str, object]) -> None:
        self._responses = [json.dumps(r, ensure_ascii=False) for r in responses]
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str, label: str = "", refresh: bool = False) -> str:
        self.calls += 1
        return self._responses[min(self.calls, len(self._responses)) - 1]


def unit(uid: str, type_: str, evidence: str, speaker: str = "Paul", status: str = "active") -> dict[str, str]:
    return {"id": uid, "type": type_, "statement": f"Enonce {uid}.", "speaker": speaker, "evidence": evidence, "status": status}


def test_evidence_must_be_in_transcript_and_rejections_are_counted() -> None:
    client = FakeClient(
        {
            "units": [
                unit("u1", "action", "Oui, je le relance demain matin.", speaker="Paul"),  # speaker faux -> corrige
                unit("u2", "fact", "Les tests passent tous."),  # reformulee -> rejetee
                unit("u3", "decision", "on lance lundi", status="bogus"),  # statut invalide -> rejetee
            ],
            "relations": [{"from": "u2", "to": "u1", "relation": "supports"}],
        }
    )
    result = extract_units(MEETING, client)
    assert [u.unit_id for u in result.units] == ["m1_x01"]
    assert result.units[0].speaker == "Nina"
    assert result.notes["speaker_corrige_depuis_transcript"] == 1
    assert result.rejections["evidence_absente_du_transcript"] == 1
    assert result.rejections["schema_invalide (status)"] == 1
    assert result.rejections["relation_vers_unite_rejetee"] == 1
    assert result.relations == []


def test_typographic_variants_are_tolerated_and_exact_text_is_kept() -> None:
    client = FakeClient({"units": [unit("u1", "question", "Nina, tu pourrais relancer le fournisseur ?")], "relations": []})
    result = extract_units(MEETING, client)
    assert result.units[0].evidence == "Nina, tu pourrais relancer le fournisseur ?"
    assert result.notes["evidence_typographie_normalisee"] == 1


def test_relations_ids_are_checked() -> None:
    client = FakeClient(
        {
            "units": [unit("u1", "fact", "Les tests de charge passent"), unit("u2", "decision", "on lance lundi")],
            "relations": [
                {"from": "u1", "to": "u2", "relation": "supports"},
                {"from": "u1", "to": "u2", "relation": "supports"},  # doublon
                {"from": "u1", "to": "u9", "relation": "supports"},  # id inconnu
                {"from": "u1", "to": "u1", "relation": "supports"},  # boucle
                {"from": "u2", "to": "u1", "relation": "explains"},  # type invalide
            ],
        }
    )
    result = extract_units(MEETING, client)
    assert [(r.from_id, r.to_id, r.relation) for r in result.relations] == [("m1_x01", "m1_x02", "supports")]
    assert result.rejections == {
        "relation_doublon": 1,
        "relation_id_inconnu": 1,
        "relation_boucle": 1,
        "relation_type_invalide": 1,
    }


def test_incomplete_response_is_retried_once() -> None:
    client = FakeClient({"units": []}, {"units": [unit("u1", "fact", "Les tests de charge passent")], "relations": []})
    result = extract_units(MEETING, client)
    assert client.calls == 2
    assert len(result.units) == 1
    assert result.notes["reponse_incomplete_relancee"] == 1


def test_project_supersedes_marks_previous_unit_superseded() -> None:
    later = {**MEETING, "source_id": "m2", "source_date": "2026-01-20", "transcript": "Paul: Finalement on lance mardi."}
    client = FakeClient(
        {"units": [unit("u1", "decision", "on lance lundi")], "relations": []},
        {"units": [unit("u1", "decision", "Finalement on lance mardi.")], "relations": [{"from": "u1", "to": "m1_x01", "relation": "supersedes"}]},
    )
    project = extract_project("p", [later, MEETING], client)  # ordre volontairement inverse : tri chronologique
    statuses = {u.unit_id: u.status for u in project.units}
    assert statuses == {"m1_x01": "superseded", "m2_x01": "active"}
    assert [(r.from_id, r.to_id) for r in project.relations] == [("m2_x01", "m1_x01")]

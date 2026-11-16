"""Seconde passe de liaison : candidats, validation stricte de la reponse du modele (aucun appel reseau)."""
from __future__ import annotations

import json
from datetime import date

import numpy as np

from src.ingest.link import LINK_SYSTEM_PROMPT, build_link_prompt, candidate_pairs, parse_links
from src.ingest.extract import SYSTEM_PROMPT
from src.schemas import Relation, SemanticUnit


def unit(uid: str, day: int, source: str, kind: str = "decision") -> SemanticUnit:
    return SemanticUnit(unit_id=uid, source_id=source, source_date=date(2026, 3, day), type=kind, statement=f"Énoncé {uid}.",
                        speaker="Paul", evidence="x", status="active")


UNITS = [unit("m1_x01", 1, "m1"), unit("m1_x02", 1, "m1"), unit("m2_x01", 8, "m2"), unit("m2_x02", 8, "m2")]
EMB = {"m1_x01": np.array([1.0, 0.0]), "m1_x02": np.array([0.0, 1.0]), "m2_x01": np.array([0.9, 0.1]), "m2_x02": np.array([0.1, 0.9])}


def test_candidates_are_earlier_neighbours_plus_existing_pairs() -> None:
    existing = [Relation(from_id="m2_x02", to_id="m1_x01", relation="supports", evidence_source_id="m2")]
    assert candidate_pairs(UNITS, existing, EMB, 0) == [("m1_x01", "m2_x02")]  # k=0 : paires deja reliees seulement
    pairs = candidate_pairs(UNITS, [], EMB, 1)
    assert all(a < b for a, b in pairs)  # A toujours anterieure a B (meme ordre que l'ordre chronologique ici)
    assert ("m1_x01", "m2_x01") in pairs and ("m1_x02", "m2_x02") in pairs  # le plus proche anterieur
    assert len(pairs) == 3  # une paire par unite qui a au moins une anterieure


def test_prompt_lists_units_and_numbered_pairs_and_is_separate_from_the_frozen_extraction_prompt() -> None:
    prompt = build_link_prompt(UNITS, [("m1_x01", "m2_x01")])
    assert "1 : A = m1_x01 ; B = m2_x01" in prompt and "m2_x02 | m2 du 2026-03-08 | decision | active" in prompt
    assert LINK_SYSTEM_PROMPT != SYSTEM_PROMPT and "aucun" in LINK_SYSTEM_PROMPT


def test_only_valid_pairs_types_and_directions_are_kept() -> None:
    pairs = [("m1_x01", "m2_x01"), ("m1_x02", "m2_x02"), ("m1_x01", "m1_x02")]
    raw = json.dumps({"links": [
        {"pair": 1, "relation": "supersedes", "direction": "B->A"},
        {"pair": 2, "relation": "supports", "direction": "A->B"},
        {"pair": 2, "relation": "answers", "direction": "A->B"},  # doublon
        {"pair": 3, "relation": "aucun"},
        {"pair": 9, "relation": "supports", "direction": "A->B"},  # paire inconnue
    ]})
    result = parse_links(raw, "p", pairs, {u.unit_id: u.source_id for u in UNITS})
    assert [(r.from_id, r.to_id, r.relation) for r in result.relations] == [("m2_x01", "m1_x01", "supersedes"), ("m1_x02", "m2_x02", "supports")]
    assert result.relations[0].evidence_source_id == "m2"
    assert result.rejections == {"paire_en_double": 1, "numero_de_paire_inconnu": 1}
    assert result.decisions["aucun"] == 1


def test_invalid_type_direction_or_json_never_produce_a_relation() -> None:
    pairs = [("m1_x01", "m2_x01"), ("m1_x02", "m2_x02")]
    raw = json.dumps({"links": [{"pair": 1, "relation": "unrelated", "direction": "A->B"}, {"pair": 2, "relation": "supports"}]})
    result = parse_links(raw, "p", pairs, {u.unit_id: u.source_id for u in UNITS})
    assert result.relations == [] and result.rejections == {"type_invalide": 1, "sens_invalide": 1}
    assert parse_links("pas du json", "p", pairs, {}).rejections == {"reponse_json_invalide": 1}
    assert parse_links(json.dumps({"links": []}), "p", pairs, {}).rejections == {"paires_sans_reponse": 2}

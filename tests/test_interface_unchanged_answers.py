"""La finition de l'interface ne change aucune reponse de l'agent.

Les reponses du parcours Atlas et des autres exemples de l'interface ont ete enregistrees sur main AVANT la
finition (tests/fixtures/answers_before_interface.json) ; elles sont comparees champ par champ a celles d'aujourd'hui.
Les questions sont lues dans app.py : si le parcours change, le test le voit.
"""
from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "answers_before_interface.json"
MODEL = ROOT / "models" / "e5-finetuned" / "config.json"


def app_scenarios() -> list[tuple[str, str]]:
    """(projet, question) du parcours Atlas puis des autres exemples, lus dans app.py sans l'executer."""
    constants: dict[str, object] = {}
    for node in ast.parse((ROOT / "app.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("DEMO_PROJECT", "DEMO_STEPS", "OTHER_EXAMPLES"):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    # etapes 1 a 5 du parcours (les etapes 6 email et 7 dossier, ajoutees ensuite, ont leurs propres tests)
    demo = [(constants["DEMO_PROJECT"], question) for _, question, _ in constants["DEMO_STEPS"][:5]]  # type: ignore[index]
    other = [(project, question) for _, project, question in constants["OTHER_EXAMPLES"]]  # type: ignore[union-attr]
    return demo + other  # type: ignore[return-value]


def snapshot(project_id: str, question: str) -> dict[str, object]:
    """Ce que l'agent renvoie pour une question, sous une forme comparable (meme routage que app.py)."""
    from src.agent.answer import answer
    from src.agent.email import Clarification, draft_email
    from src.agent.intent import detect_intent

    intent = detect_intent(question)
    record: dict[str, object] = {"project_id": project_id, "question": question, "intent": intent}
    if intent == "draft_email":
        result = draft_email(question, project_id)
        if isinstance(result, Clarification):
            record["clarification"] = {
                "reason": result.reason,
                "question": result.question,
                "candidates": [[c.person, c.unit.unit_id, list(c.origin_ids)] for c in result.candidates],
            }
        else:
            record["draft"] = result.model_dump(mode="json")
        return record
    data = dataclasses.asdict(answer(question, project_id))
    data["source_date"] = str(data["source_date"]) if data["source_date"] else None
    data["best_dense_score"] = round(data["best_dense_score"], 6)
    data["fused_score"] = round(data["fused_score"], 9) if data["fused_score"] is not None else None
    for key in ("resolution", "conflict_with"):
        data[key] = list(data[key])
    data["history"] = [list(step) for step in data["history"]]
    record["answer"] = data
    return record


def test_steps_6_and_7_are_added_after_the_five_original_steps() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    steps = ast.literal_eval(next(n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
                                  and getattr(n.targets[0], "id", "") == "DEMO_STEPS"))
    assert [label for label, _, _ in steps][5:] == ["(6) Email", "(7) Dossier"]


def test_fixture_covers_the_atlas_path_and_the_other_examples() -> None:
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))["answers"]
    assert [(r["project_id"], r["question"]) for r in recorded] == app_scenarios()
    assert len(recorded) == 8  # 5 etapes Atlas + 3 autres exemples


@pytest.mark.skipif(not MODEL.exists(), reason="modele fine-tune absent (clone frais) : l'agent ne peut pas tourner")
def test_agent_answers_are_exactly_the_same_as_before_the_interface_polish() -> None:
    for before in json.loads(FIXTURE.read_text(encoding="utf-8"))["answers"]:
        after = snapshot(before["project_id"], before["question"])
        if "draft" in before:  # seul changement voulu depuis : l'objet d'un brouillon sur une action annulee
            assert before["draft"]["subject"] == "Suivi : signer le devis de l'agence Horizon d'ici vendredi"
            assert after["draft"].pop("subject") == "Point mis à jour : signer le devis de l'agence Horizon d'ici vendredi"
            before = {**before, "draft": {k: v for k, v in before["draft"].items() if k != "subject"}}
        assert after == before, before["question"]

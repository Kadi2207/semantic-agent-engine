"""Finitions : objet du brouillon d'email pour une action remplacee, aucun emoji dans le code ni la documentation."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from src.agent import email as email_module
from src.agent.answer import ReplacementGraph
from src.agent.email import UPDATED_SUBJECT, draft_email
from src.agent.retriever import RRF_K, Hit
from src.schemas import EmailDraft, Relation, SemanticUnit

ROOT = Path(__file__).resolve().parent.parent
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")


def unit(uid: str, day: int, statement: str, status: str) -> SemanticUnit:
    return SemanticUnit(unit_id=uid, source_id=f"m{day}", source_date=date(2026, 5, day), type="action",
                        statement=statement, speaker="Paul", evidence="x", status=status)


OLD = unit("t_u01", 4, "Nina doit envoyer le bon de commande au fournisseur vendredi.", "superseded")
NEW = unit("t_u02", 11, "Nina doit envoyer le bon de commande au fournisseur lundi.", "active")


class Retriever:
    def unit(self, unit_id: str) -> SemanticUnit:
        return {"t_u01": OLD, "t_u02": NEW}[unit_id]

    def max_dense_score(self, query: str, project_id: str) -> float:
        return 0.9

    def search(self, query: str, project_id: str, k: int | None = 5) -> list[Hit]:
        return [Hit(OLD, 1 / (RRF_K + 1), 0.9, 1, 1), Hit(NEW, 1 / (RRF_K + 2), 0.8, 2, 2)][:k]


def test_subject_says_when_the_targeted_action_was_replaced(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(email_module, "get_participants", lambda: {"p": ["Nina", "Paul"]})
    graph = ReplacementGraph([Relation(from_id="t_u02", to_id="t_u01", relation="supersedes", evidence_source_id="m11")])
    draft = draft_email("Écris un mail à la personne qui doit envoyer le bon de commande", "p", retriever=Retriever(), graph=graph)
    assert isinstance(draft, EmailDraft) and draft.recipient.name == "Nina"
    assert draft.subject.startswith(f"{UPDATED_SUBJECT} : ")  # la tache visee a ete remplacee
    assert draft.needs_confirmation is True and draft.executed is False


def test_no_emoji_in_code_docs_or_streamlit_icons() -> None:
    files = [ROOT / "app.py", ROOT / "README.md"] + [p for d in ("src", "scripts", "docs") for p in (ROOT / d).rglob("*") if p.suffix in (".py", ".md")]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not EMOJI.search(text), path.relative_to(ROOT)
        if path.suffix == ".py":
            assert not re.search(r"\b(?:page_)?icon\s*=", text), path.relative_to(ROOT)  # aucune icone st.*

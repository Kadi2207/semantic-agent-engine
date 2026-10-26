from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, field_validator


class SemanticUnit(BaseModel):
    unit_id: str
    source_id: str
    source_date: date
    type: Literal[
        "decision",
        "action",
        "commitment",
        "question",
        "answer",
        "fact",
        "risk",
        "idea",
    ]
    statement: str
    speaker: str | None
    evidence: str
    status: Literal["active", "superseded", "cancelled"]


class Relation(BaseModel):
    from_id: str
    to_id: str
    relation: Literal[
        "supports",
        "contradicts",
        "supersedes",
        "creates_action",
        "answers",
        "unrelated",
    ]
    evidence_source_id: str


class Recipient(BaseModel):
    name: str
    email: str | None
    evidence_unit_ids: list[str]
    resolution_status: Literal["resolved", "email_missing", "ambiguous"]


class EmailDraft(BaseModel):
    intent: Literal["draft_email"]
    tool_candidate: str
    recipient: Recipient
    subject: str
    body: str
    source_unit_ids: list[str]
    needs_confirmation: bool = True
    executed: bool = False

    @field_validator("executed")
    @classmethod
    def _reject_executed(cls, value: bool) -> bool:
        # Un brouillon ne doit jamais etre marque comme envoye par le moteur.
        if value:
            raise ValueError("an EmailDraft cannot have executed=True")
        return value

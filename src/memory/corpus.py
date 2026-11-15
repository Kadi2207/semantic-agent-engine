from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.schemas import Relation, SemanticUnit


@dataclass(frozen=True)
class CorpusUnit:
    project_id: str
    unit: SemanticUnit

    @property
    def text(self) -> str:
        return self.unit.statement


def load_corpus(data_dir: str | Path = "data/raw") -> list[CorpusUnit]:
    corpus: list[CorpusUnit] = []
    for path in sorted(Path(data_dir).glob("projet_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        project_id = data["project_id"]
        for raw_unit in data["units"]:
            corpus.append(CorpusUnit(project_id, SemanticUnit.model_validate(raw_unit)))
    return corpus


def load_relations(data_dir: str | Path = "data/raw") -> list[Relation]:
    relations: list[Relation] = []
    for path in sorted(Path(data_dir).glob("projet_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        relations += [Relation.model_validate(raw) for raw in data.get("relations", [])]
    return relations


def load_participants(data_dir: str | Path = "data/raw") -> dict[str, list[str]]:
    """Participants des reunions de chaque projet (project_id -> noms, tries)."""
    participants: dict[str, list[str]] = {}
    for path in sorted(Path(data_dir).glob("projet_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        names = {name for meeting in data.get("meetings", []) for name in meeting["participants"]}
        participants[data["project_id"]] = sorted(names)
    return participants


def load_queries(path: str | Path) -> list[dict[str, object]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["queries"]

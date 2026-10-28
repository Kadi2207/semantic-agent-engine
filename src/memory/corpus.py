from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.schemas import SemanticUnit


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


def load_queries(path: str | Path) -> list[dict[str, object]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["queries"]

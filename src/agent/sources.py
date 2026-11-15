"""Passages sources : retrouve, dans le transcript de la reunion, la prise de parole qui contient l'evidence."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.agent.retriever import ROOT
from src.schemas import SemanticUnit

DATA_DIR = ROOT / "data" / "raw"
_TYPO = str.maketrans({"’": "'", "‘": "'", " ": " ", " ": " ", "‑": "-"})


@dataclass(frozen=True)
class SourceLine:
    turn: int  # position de la prise de parole dans le transcript (a partir de 0)
    speaker: str
    text: str
    is_evidence: bool


@dataclass(frozen=True)
class SourcePassage:
    source_id: str
    source_date: str
    lines: tuple[SourceLine, ...]
    found: bool  # False si l'evidence n'a pas ete retrouvee dans le transcript


@lru_cache(maxsize=1)
def load_transcripts(data_dir: str = str(DATA_DIR)) -> dict[str, tuple[str, list[tuple[str, str]]]]:
    """source_id -> (date, [(locuteur, texte)]) pour toutes les reunions du corpus."""
    meetings: dict[str, tuple[str, list[tuple[str, str]]]] = {}
    for path in sorted(Path(data_dir).glob("projet_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for meeting in data.get("meetings", []):
            turns = []
            for line in str(meeting["transcript"]).splitlines():
                speaker, _, text = line.partition(":")
                turns.append((speaker.strip(), text.strip()))
            meetings[meeting["source_id"]] = (str(meeting["source_date"]), turns)
    return meetings


def source_passage(unit: SemanticUnit, context: int = 1) -> SourcePassage:
    """La prise de parole contenant l'evidence, avec `context` prises de parole avant et apres."""
    date, turns = load_transcripts().get(unit.source_id, (str(unit.source_date), []))
    needle = unit.evidence.translate(_TYPO).strip()
    for i, (_, text) in enumerate(turns):
        if needle and needle in text.translate(_TYPO):
            lo, hi = max(i - context, 0), min(i + context, len(turns) - 1)
            lines = tuple(SourceLine(j, turns[j][0], turns[j][1], j == i) for j in range(lo, hi + 1))
            return SourcePassage(unit.source_id, date, lines, True)
    return SourcePassage(unit.source_id, date, (), False)

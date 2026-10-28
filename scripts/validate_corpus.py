from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schemas import Relation, SemanticUnit  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"


def _format_validation_error(error: ValidationError) -> str:
    parts = []
    for detail in error.errors():
        location = ".".join(str(item) for item in detail["loc"]) or "<racine>"
        parts.append(f"{location}: {detail['msg']}")
    return "; ".join(parts)


def validate_file(path: Path) -> tuple[list[str], int, int]:
    problems: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"lecture JSON impossible: {exc}"], 0, 0

    meetings = data.get("meetings", [])
    units = data.get("units", [])
    relations = data.get("relations", [])
    transcripts = {m.get("source_id"): m.get("transcript", "") for m in meetings}

    unit_ids: set[str] = set()
    for index, raw_unit in enumerate(units):
        label = raw_unit.get("unit_id", f"units[{index}]")
        try:
            SemanticUnit.model_validate(raw_unit)
        except ValidationError as exc:
            problems.append(f"schema SemanticUnit {label}: {_format_validation_error(exc)}")

        unit_id = raw_unit.get("unit_id")
        if unit_id is not None:
            if unit_id in unit_ids:
                problems.append(f"unit_id en double: {unit_id}")
            unit_ids.add(unit_id)

        source_id = raw_unit.get("source_id")
        evidence = raw_unit.get("evidence")
        if source_id not in transcripts:
            problems.append(f"{label}: source_id inconnu {source_id!r}")
        elif not isinstance(evidence, str) or evidence not in transcripts[source_id]:
            problems.append(f"{label}: evidence introuvable dans {source_id}: {evidence!r}")

    for index, raw_relation in enumerate(relations):
        label = f"relations[{index}]"
        try:
            Relation.model_validate(raw_relation)
        except ValidationError as exc:
            problems.append(f"schema Relation {label}: {_format_validation_error(exc)}")

        for field in ("from_id", "to_id"):
            target = raw_relation.get(field)
            if target not in unit_ids:
                problems.append(f"{label}: {field} orphelin {target!r}")

    return problems, len(units), len(relations)


def main() -> int:
    # La console Windows n'est pas toujours en UTF-8, on force pour les accents.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    paths = sorted(RAW_DIR.glob("projet_*.json"))
    total_units = 0
    total_relations = 0
    total_problems = 0

    for path in paths:
        problems, unit_count, relation_count = validate_file(path)
        total_units += unit_count
        total_relations += relation_count
        total_problems += len(problems)
        if problems:
            print(f"{path.name}: {len(problems)} probleme(s)")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"{path.name}: OK ({unit_count} unites, {relation_count} relations)")

    print()
    print("Bilan global")
    print(f"  fichiers  : {len(paths)}")
    print(f"  unites    : {total_units}")
    print(f"  relations : {total_relations}")
    print(f"  problemes : {total_problems}")
    return 1 if total_problems else 0


if __name__ == "__main__":
    sys.exit(main())

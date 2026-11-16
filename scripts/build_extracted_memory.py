"""Construit la memoire extraite (projets 04 a 12) a partir du rejeu de l'extraction depuis le cache Groq local.

Aucun appel Groq : le rejeu vient du cache seul (meme rejeu que eval/analyze_relation_errors.py). Les unites et
relations extraites sont ecrites telles quelles, sans correction manuelle, dans data/extracted/, au meme format
que data/raw (reunions, units, relations) : l'agent les lit avec le meme chargeur (src/memory/corpus.py).
Avant d'ecrire, le script verifie que le rejeu redonne les nombres d'unites et de relations de extraction.json.

    uv run python scripts/build_extracted_memory.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingest.extract import SYSTEM_PROMPT, extract_project, make_client  # noqa: E402
from src.memory.extracted import EXTRACTED_DIR, EXTRACTED_PROJECTS, MENTION  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
EXTRACTION_JSON = ROOT / "eval" / "results" / "extraction.json"
GROUPS = {"dev_04_10": range(4, 11), "test_11_12": range(11, 13)}


def main() -> None:
    client = make_client(cache_only=True)
    reference = json.loads(EXTRACTION_JSON.read_text(encoding="utf-8"))
    prompt_sha = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if prompt_sha != reference["extraction"]["system_prompt_sha256"]:
        sys.exit("ECART : la consigne n'est pas celle de extraction.json, rien n'est ecrit")

    projects: dict[int, dict[str, object]] = {}
    for number in EXTRACTED_PROJECTS:
        raw = json.loads(next(RAW_DIR.glob(f"projet_{number:02d}_*.json")).read_text(encoding="utf-8"))
        # Entree de l'extraction : les reunions brutes SEULES (aucune unite ni relation annotee).
        meetings = [{k: m[k] for k in ("source_id", "source_date", "participants", "transcript")} for m in raw["meetings"]]
        extraction = extract_project(raw["project_id"], meetings, client)
        projects[number] = {
            "project_id": raw["project_id"],
            "project_name": raw["project_name"],
            "memory": "extraite",
            "mention": MENTION,
            "source": "rejeu de l'extraction depuis le cache Groq local (scripts/build_extracted_memory.py), consigne figee",
            "system_prompt_sha256": prompt_sha,
            "meetings": meetings,
            "units": [u.model_dump(mode="json") for u in extraction.units],
            "relations": [r.model_dump(mode="json") for r in extraction.relations],
        }

    problems = []
    if client.calls:
        problems.append(f"{client.calls} appel(s) Groq effectue(s) : le rejeu doit venir du cache seul")
    for group, numbers in GROUPS.items():
        units = sum(len(projects[n]["units"]) for n in numbers)  # type: ignore[arg-type]
        relations = sum(len(projects[n]["relations"]) for n in numbers)  # type: ignore[arg-type]
        expected = reference["groups"][group]
        if (units, relations) != (expected["units"]["extracted"], expected["relations"]["extracted"]):
            problems.append(f"{group} : {units} unites et {relations} relations rejouees, extraction.json en compte "
                            f"{expected['units']['extracted']} et {expected['relations']['extracted']}")
    if problems:
        sys.exit("ECART, rien n'est ecrit :\n  " + "\n  ".join(problems))

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    for number, project in projects.items():
        name = next(RAW_DIR.glob(f"projet_{number:02d}_*.json")).name
        (EXTRACTED_DIR / name).write_text(json.dumps(project, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"{project['project_id']} : {len(project['units'])} unites, {len(project['relations'])} relations -> data/extracted/{name}")  # type: ignore[arg-type]
    print(f"rejeu : {client.calls} appel Groq, {client.cache_hits} lectures du cache ; coherent avec extraction.json")


if __name__ == "__main__":
    main()

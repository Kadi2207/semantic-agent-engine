"""Prechauffe le cache du RAG controle pour les questions de la demonstration (aucun appel si deja en cache).

Le cache (data/processed/groq_rag_cache.jsonl) est versionne : la demonstration rejoue alors les syntheses sans
reseau. Questions : parcours Atlas (etapes 1 a 5), autres exemples, question de la capture en memoire extraite.

    uv run python scripts/prewarm_rag.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.rag import get_rag_client, rag_answer  # noqa: E402
from src.memory.extracted import get_extracted_graph, get_extracted_retriever  # noqa: E402

EXTRACTED_QUESTIONS = [("projet_04", "A-t-on perdu des fiches pendant l'essai de reprise des données ?")]


def app_questions() -> list[tuple[str, str]]:
    constants = {}
    for node in ast.parse((ROOT / "app.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("DEMO_PROJECT", "DEMO_STEPS", "OTHER_EXAMPLES"):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    demo = [(constants["DEMO_PROJECT"], q) for _, q, _ in constants["DEMO_STEPS"][:5]]
    return demo + [(pid, q) for _, pid, q in constants["OTHER_EXAMPLES"]]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = get_rag_client()
    for project_id, question in app_questions():
        if question.lower().startswith(("écris", "ecris")):
            continue  # demandes d'email : pas de RAG
        r = rag_answer(question, project_id, client=client)
        print(f"[annotee] {project_id} {r.status:<15} {list(r.cited_ids)} {question}")
    for project_id, question in EXTRACTED_QUESTIONS:
        r = rag_answer(question, project_id, retriever=get_extracted_retriever(), graph=get_extracted_graph(), client=client)
        print(f"[extraite] {project_id} {r.status:<15} {list(r.cited_ids)} {question}")
    print(f"Groq : {client.snapshot()}")


if __name__ == "__main__":
    main()

"""docs/ARCHITECTURE.md et docs/DECISIONS.md : liens depuis le README, fichiers cites existants, chiffres tires des JSON."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
DOCS = {name: (ROOT / "docs" / name).read_text(encoding="utf-8") for name in ("ARCHITECTURE.md", "DECISIONS.md")}


def numbers(node: object) -> set[str]:
    if isinstance(node, bool):
        return set()
    if isinstance(node, (int, float)):
        return {f"{abs(node):.3f}".replace(".", ",")}
    if isinstance(node, dict):
        return set().union(*(numbers(v) for v in node.values())) if node else set()
    if isinstance(node, list):
        return set().union(*(numbers(v) for v in node)) if node else set()
    return set()


def test_readme_links_both_documents() -> None:
    assert "(docs/ARCHITECTURE.md)" in README and "(docs/DECISIONS.md)" in README


def test_every_cited_path_exists() -> None:
    for name, text in DOCS.items():
        for path in re.findall(r"`((?:src|eval|scripts|tests|data|docs)/[\w./-]+|app\.py)`", text):
            assert (ROOT / path).exists(), (name, path)
        for json_name in re.findall(r"`(\w+\.json)`", text):
            assert (ROOT / "eval/results" / json_name).exists() or list(ROOT.glob(f"data/**/{json_name}")), (name, json_name)


def test_every_three_decimal_number_comes_from_a_results_json() -> None:
    known = set().union(*(numbers(json.loads(p.read_text(encoding="utf-8"))) for p in (ROOT / "eval/results").glob("*.json")))
    for name, text in DOCS.items():
        cited = set(re.findall(r"(?<![\d,])\d,\d{3}(?![\d])", text))
        assert cited <= known, (name, sorted(cited - known))


def test_architecture_diagram_covers_the_chain() -> None:
    text = DOCS["ARCHITECTURE.md"]
    assert "```mermaid" in text
    for module in ("extract.py", "retriever.py", "answer.py", "rag.py", "email.py", "dossier.py", "app.py", "train_embeddings.py"):
        assert module in text, module


def test_decisions_have_alternative_and_proof() -> None:
    sections = DOCS["DECISIONS.md"].split("\n## ")[1:]
    assert len(sections) >= 10
    for s in sections:
        assert re.search(r"Alternatives? écartées? :", s) and ("Preuve" in s or "Limite chiffrée" in s), s[:60]


def test_mit_license_and_data_note() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License") and "Copyright (c) 2026 Kadidiatou Ibrahima Bagayoko" in license_text
    section = README.split("\n## Licence\n")[1].split("\n## ")[0]
    assert "(LICENSE)" in section and "QMSum" in section and "ne sont pas redistribuées" in section
    projects = sorted(ROOT.glob("data/raw/projet_*.json")) + sorted(ROOT.glob("data/test2/projet_*.json"))
    assert len(projects) == 16 and "Les 16 projets du corpus" in section and "fictifs" in section


def test_ci_workflow_and_single_demo_command() -> None:
    workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    assert 'python-version: "3.12"' in workflow and "https://download.pytorch.org/whl/cpu" in workflow
    assert workflow.index("download.pytorch.org") < workflow.index("pip install -r requirements.txt")
    assert "pytest" in workflow
    head = README.split("## Évaluer le projet en 5 minutes")[0] + README.split("## Évaluer le projet en 5 minutes")[1].split("\n## ")[0]
    assert "actions/workflows/tests.yml/badge.svg" in README.split("## ")[0]
    assert "uv run python scripts/run_demo.py" in head and (ROOT / "scripts/run_demo.py").exists()

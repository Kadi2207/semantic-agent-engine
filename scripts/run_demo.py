"""Lance la demonstration en une commande : environnement verifie, modele entraine s'il manque, verify_demo, Streamlit.

    uv run python scripts/run_demo.py            # verifie, entraine si besoin, verifie la demo, lance l'interface
    uv run python scripts/run_demo.py --check    # tout sauf le lancement de l'interface

Aucune cle n'est necessaire : les syntheses du RAG de la demonstration sont rejouees depuis le cache versionne.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "e5-finetuned"
PACKAGES = ["torch", "sentence_transformers", "rank_bm25", "pydantic", "streamlit", "numpy"]


def step(title: str) -> None:
    print(f"\n== {title}", flush=True)


def check_environment() -> list[str]:
    problems = []
    if sys.version_info[:2] != (3, 12):
        problems.append(f"Python 3.12 attendu, {sys.version.split()[0]} trouve (uv venv --python 3.12)")
    missing = [p for p in PACKAGES if importlib.util.find_spec(p) is None]
    if missing:
        problems.append(f"paquets manquants : {', '.join(missing)} (uv pip install -r requirements.txt)")
    for path in ("app.py", "data/raw", "data/processed/groq_rag_cache.jsonl"):
        if not (ROOT / path).exists():
            problems.append(f"{path} introuvable : lancer depuis un clone complet du depot")
    return problems


def run(args: list[str]) -> int:
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="ne lance pas l'interface")
    args = parser.parse_args()

    step("Environnement")
    problems = check_environment()
    if problems:
        print("\n".join(f"  - {p}" for p in problems))
        return 1
    print(f"  Python {sys.version.split()[0]}, dependances presentes")

    step("Modele fine-tune")
    if (MODEL / "config.json").exists():
        print(f"  present : {MODEL.relative_to(ROOT)}")
    else:
        print("  absent : entrainement (quelques minutes sur CPU, telechargement du modele de base)")
        if run(["scripts/train_embeddings.py"]) != 0 or not (MODEL / "config.json").exists():
            print("  ECHEC de l'entrainement")
            return 1

    step("Verification de la demonstration (scripts/verify_demo.py)")
    if run(["scripts/verify_demo.py"]) != 0:
        print("  ECHEC : la demonstration ne passe pas ses verifications, l'interface n'est pas lancee")
        return 1

    if args.check:
        print("\nTout est pret. Interface : uv run streamlit run app.py --server.address localhost")
        return 0
    step("Interface Streamlit (http://localhost:8501, Ctrl+C pour arreter)")
    return run(["-m", "streamlit", "run", "app.py", "--server.address", "localhost"])


if __name__ == "__main__":
    sys.exit(main())

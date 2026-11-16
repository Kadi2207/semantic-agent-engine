"""Enregistre une video du parcours de demonstration (filet de securite pour la soutenance).

Prerequis : l'application tourne en local (uv run streamlit run app.py --server.address localhost) et le
paquet playwright est installe ; le navigateur utilise est Microsoft Edge deja present sur la machine.

    uv run python scripts/record_demo.py --out demo/demo_atlas.webm
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
# (bouton du parcours, texte qui prouve que la bonne reponse est affichee)
STEPS = [
    ("(1) Décision en vigueur", "Passage source"),
    ("(2) Action : qui et quand", "Claire envoie un message aux utilisateurs"),
    ("(3) Décision remplacée", "Information mise à jour"),
    ("(4) Passages sources", "Source de p01_u12"),
    ("(5) Hors sources", "Information absente ou incertaine dans les sources"),
]
PAUSE_MS = 4500


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("demo/demo_atlas.webm"))
    parser.add_argument("--channel", default="msedge", help="navigateur installe a piloter (msedge ou chrome)")
    args = parser.parse_args()
    video_dir = Path(tempfile.mkdtemp(prefix="demo_video_"))
    size = {"width": 1280, "height": 900}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True)
        context = browser.new_context(viewport=size, record_video_dir=str(video_dir), record_video_size=size)
        page = context.new_page()
        page.goto(URL)
        page.get_by_text("Embedding Lab").first.wait_for(timeout=600_000)  # chargement du modele
        page.get_by_role("button", name=STEPS[0][0]).wait_for(timeout=600_000)
        page.wait_for_timeout(2500)
        for label, proof in STEPS:
            page.get_by_role("button", name=label).click()
            page.wait_for_timeout(1200)  # la question pre-remplie est visible
            page.get_by_role("button", name="Envoyer").click()
            page.get_by_text(proof).first.wait_for(timeout=120_000)
            page.wait_for_timeout(PAUSE_MS)
            page.mouse.move(800, 600)  # au-dessus de la zone principale (la molette agit sous le curseur)
            page.mouse.wheel(0, 700)  # faire defiler jusqu'aux sources, a la mise a jour et a l'historique
            page.wait_for_timeout(PAUSE_MS)
            page.mouse.wheel(0, -3000)
            page.wait_for_timeout(800)
            print(f"  etape affichee : {label}", flush=True)
        video_path = Path(page.video.path())
        context.close()  # finalise le fichier video
        browser.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(video_path), args.out)
    shutil.rmtree(video_dir, ignore_errors=True)
    print(f"Video enregistree : {args.out} ({args.out.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    sys.exit(main())

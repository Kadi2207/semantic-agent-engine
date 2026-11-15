"""Verifie le parcours de demonstration (projet Atlas) avant de le declarer pret.

V1 le supersedes d'Atlas reussit, de facon reproductible (plusieurs formulations, plusieurs passages) ;
V2 les passages sources s'affichent dans l'interface (Streamlit AppTest, rendu reel de app.py) ;
V3 l'abstention se declenche sur une question hors sources ;
V4 l'action a un responsable (lu dans l'enonce, pas le locuteur) et une echeance.

    uv run python scripts/verify_demo.py --out <fichier.json>
Lancer deux fois et comparer les deux fichiers verifie la reproductibilite entre processus.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.answer import answer, get_retriever  # noqa: E402
from src.agent.email import get_participants, responsible_person  # noqa: E402
from src.agent.sources import source_passage  # noqa: E402
from src.memory.corpus import load_corpus  # noqa: E402

PROJECT = "projet_01"
DECISION_QUESTIONS = [
    "La mise en production d'Atlas est-elle toujours prévue le 14 avril ?",
    "La mise en production d'Atlas est-elle validée pour le 14 avril ?",
    "Le lancement d'Atlas a-t-il lieu le 14 avril ?",
]
CURRENT_DECISION_QUESTION = "Quelle est la date de mise en production d'Atlas ?"
ACTION_QUESTION = "Qui doit annoncer le report aux utilisateurs, et quand ?"
OFF_SOURCES_QUESTION = "Quel est le salaire du nouveau directeur technique ?"
EXPECTED_SUPERSEDES = "p01_u12 (superseded) -supersedes-> p01_u20 (active)"
DEADLINE = re.compile(
    r"\b(le jour même|d'ici (?:le )?\w+|avant (?:le )?\w+|au plus tard (?:le )?[\w ]+|"
    r"\d{1,2} (?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)|"
    r"lundi|mardi|mercredi|jeudi|vendredi)\b",
    re.IGNORECASE,
)
REPEATS = 3


def v1_supersedes() -> dict[str, object]:
    runs = []
    for _ in range(REPEATS):
        for q in DECISION_QUESTIONS:
            a = answer(q, PROJECT)
            runs.append({"question": q, "unit_id": a.unit_id, "resolution": list(a.resolution), "abstained": a.abstained})
    current = answer(CURRENT_DECISION_QUESTION, PROJECT)
    ok_runs = all(r["unit_id"] == "p01_u20" and EXPECTED_SUPERSEDES in r["resolution"] for r in runs)
    history = [list(step) for step in current.history]
    ok_history = ["p01_u12", "supersedes", "p01_u20"] in history and ["p01_u01", "supersedes", "p01_u12"] in history
    return {
        "ok": ok_runs and ok_history and current.unit_id == "p01_u20",
        "detail": f"{sum(r['unit_id'] == 'p01_u20' and EXPECTED_SUPERSEDES in r['resolution'] for r in runs)}/{len(runs)} "
        f"reponses ({len(DECISION_QUESTIONS)} formulations x {REPEATS} passages) resolues par {EXPECTED_SUPERSEDES} ; "
        f"decision en vigueur {current.unit_id}, historique {history}",
        "runs": runs,
        "current_decision": {"unit_id": current.unit_id, "statement": current.statement, "history": history},
    }


def v2_sources_displayed() -> dict[str, object]:
    from streamlit.testing.v1 import AppTest

    # (a) toutes les evidences du corpus sont retrouvees dans leur transcript
    corpus = load_corpus(ROOT / "data" / "raw")
    missing = [c.unit.unit_id for c in corpus if not source_passage(c.unit).found]
    # (b) rendu reel de l'interface pour les etapes 1, 3 et 4 du parcours
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600).run()
    seen: dict[str, dict[str, bool]] = {}
    expected = {
        "1": ["<mark>La mise en production d&#x27;Atlas est reportée au 5 mai.</mark>", "Passage source"],
        "3": ["<mark>La mise en production d&#x27;Atlas est reportée au 5 mai.</mark>", "Historique", "Information mise à jour"],
        "4": [
            "<mark>La mise en production d&#x27;Atlas est reportée au 5 mai.</mark>",
            "<mark>C&#x27;est donc acté, mise en production d&#x27;Atlas le 14 avril.</mark>",
            "<mark>Je propose qu&#x27;on vise une mise en production le 14 avril.</mark>",
        ],
    }
    for step, needles in expected.items():
        at.button(key=f"demo_{int(step) - 1}").click().run()
        next(b for b in at.button if b.label == "Envoyer").click().run()
        page = "\n".join(m.value for m in at.markdown) + "\n".join(i.value for i in at.info)
        seen[step] = {n: n in page for n in needles}
    ok = not missing and not at.exception and all(all(v.values()) for v in seen.values())
    return {
        "ok": ok,
        "detail": f"evidences retrouvees dans leur transcript : {len(corpus) - len(missing)}/{len(corpus)} ; "
        f"elements affiches (etapes 1, 3, 4) : {sum(sum(v.values()) for v in seen.values())}/{sum(len(v) for v in seen.values())}",
        "missing_passages": missing,
        "rendered": seen,
        "app_exception": [str(e) for e in at.exception],
    }


def v3_abstention() -> dict[str, object]:
    a = answer(OFF_SOURCES_QUESTION, PROJECT)
    return {
        "ok": a.abstained,
        "detail": f"« {OFF_SOURCES_QUESTION} » -> {'abstention' if a.abstained else a.unit_id} "
        f"(meilleur cosinus {a.best_dense_score:.3f} < seuil {a.threshold:.2f})",
        "message": a.message,
    }


def v4_action() -> dict[str, object]:
    a = answer(ACTION_QUESTION, PROJECT)
    unit = get_retriever().unit(a.unit_id) if a.unit_id else None
    person = responsible_person(unit.statement, get_participants()[PROJECT]) if unit else None
    deadline = DEADLINE.search(unit.statement) if unit else None
    ok = bool(unit and unit.type in ("action", "commitment") and person and deadline and unit.status == "active")
    return {
        "ok": ok,
        "detail": f"{a.unit_id} [{unit.type if unit else '-'}, {unit.status if unit else '-'}] « {a.statement} » ; "
        f"responsable : {person} ; echeance : « {deadline.group(0) if deadline else None} » "
        f"(reunion {a.source_id} du {a.source_date})",
        "unit_id": a.unit_id,
        "responsible": person,
        "deadline": deadline.group(0) if deadline else None,
        "source_date": str(a.source_date),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {"V1_supersedes": v1_supersedes(), "V3_abstention": v3_abstention(), "V4_action": v4_action(), "V2_sources": v2_sources_displayed()}
    for name in ("V1_supersedes", "V2_sources", "V3_abstention", "V4_action"):
        print(f"{name:<15} {'OK ' if report[name]['ok'] else 'ECHEC'}  {report[name]['detail']}")
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.exit(0 if all(r["ok"] for r in report.values()) else 1)


if __name__ == "__main__":
    main()

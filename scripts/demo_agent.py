"""CLI de demonstration : route la demande selon l'intention.

- question -> reponse sourcee (ou abstention) ;
- demande d'ecrire a quelqu'un -> brouillon d'email a confirmer (jamais envoye), ou demande de clarification.

Exemples :
    uv run python scripts/demo_agent.py "Quand a lieu la mise en production d'Atlas ?" projet_01
    uv run python scripts/demo_agent.py "Ecris un mail a la personne qui doit signer le devis d'Horizon" 02
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.answer import ABSTENTION_THRESHOLD, Answer, answer  # noqa: E402
from src.agent.email import Clarification, draft_email  # noqa: E402
from src.agent.intent import detect_intent  # noqa: E402
from src.schemas import EmailDraft  # noqa: E402


def normalize_project_id(raw: str) -> str:
    """Accepte 'projet_01', '01' ou '1'."""
    return f"projet_{int(raw):02d}" if raw.isdigit() else raw


def render_answer(result: Answer) -> str:
    lines = [f"Question : {result.query}", f"Projet   : {result.project_id}", "Intention: answer", ""]
    if result.abstained:
        lines.append(f"Reponse  : {result.message}")
    elif result.conflict_with:
        from src.agent.answer import get_retriever

        lines.append("CONFLIT NON TRANCHE : affirmations contradictoires, toutes en vigueur ; l'agent ne choisit pas.")
        for number, unit_id in enumerate((result.unit_id, *result.conflict_with), start=1):
            unit = get_retriever().unit(unit_id)
            lines += [
                f"Affirmation {number} : {unit.statement}",
                f"  Preuve : \"{unit.evidence}\"",
                f"  Source : {unit.source_id} du {unit.source_date}, dit par {unit.speaker or 'inconnu'} ; unite {unit.unit_id} (statut {unit.status})",
            ]
    else:
        lines += [
            f"Reponse  : {result.statement}",
            f"Preuve   : \"{result.evidence}\"",
            f"Source   : {result.source_id} du {result.source_date}, dit par {result.speaker or 'inconnu'}",
            f"Unite    : {result.unit_id} (statut {result.unit_status})",
        ]
        for step in result.resolution:
            lines.append(f"Mise a jour : {step}")
    lines.append("")
    verdict = "abstention" if result.abstained else "reponse"
    lines.append(
        f"[confiance] meilleur cosinus dense = {result.best_dense_score:.3f}, "
        f"seuil = {result.threshold:.2f} -> {verdict}"
        + (f" ; score RRF = {result.fused_score:.4f}" if result.fused_score is not None else "")
    )
    return "\n".join(lines)


def render_draft(query: str, project_id: str, draft: EmailDraft) -> str:
    recipient = draft.recipient
    email = recipient.email or "(adresse inconnue : a renseigner, jamais inventee)"
    lines = [
        f"Demande  : {query}",
        f"Projet   : {project_id}",
        "Intention: draft_email",
        "",
        "===== BROUILLON D'EMAIL (A CONFIRMER, NON ENVOYE) =====",
        f"Destinataire : {recipient.name} <{email}>  [statut : {recipient.resolution_status}]",
        f"Objet        : {draft.subject}",
        "",
        draft.body,
        "",
        f"References   : {', '.join(draft.source_unit_ids)} (responsable identifie dans {', '.join(recipient.evidence_unit_ids)})",
        f"Outil prevu  : {draft.tool_candidate} | needs_confirmation={draft.needs_confirmation} | executed={draft.executed}",
        "=> A confirmer : relis le brouillon et renseigne l'adresse avant tout envoi.",
    ]
    return "\n".join(lines)


def render_clarification(result: Clarification) -> str:
    lines = [
        f"Demande  : {result.query}",
        f"Projet   : {result.project_id}",
        "Intention: draft_email",
        "",
        f"CLARIFICATION NECESSAIRE ({result.reason}) : aucun brouillon genere.",
        f"Question : {result.question}",
    ]
    for c in result.candidates:
        lines.append(
            f"  - {c.person or 'aucun responsable designe'} : {c.unit.statement} "
            f"[{c.unit.unit_id}, {c.unit.source_id} du {c.unit.source_date}]"
        )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", help="la question ou la demande en langage naturel")
    parser.add_argument("project_id", help="identifiant du projet (projet_01, 01 ou 1)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=ABSTENTION_THRESHOLD,
        help=f"cosinus minimal pour repondre (defaut {ABSTENTION_THRESHOLD})",
    )
    args = parser.parse_args()
    project_id = normalize_project_id(args.project_id)

    try:
        if detect_intent(args.question) == "draft_email":
            result = draft_email(args.question, project_id, threshold=args.threshold)
            if isinstance(result, Clarification):
                print(render_clarification(result))
            else:
                print(render_draft(args.question, project_id, result))
        else:
            print(render_answer(answer(args.question, project_id, threshold=args.threshold)))
    except (KeyError, FileNotFoundError) as exc:
        sys.exit(f"ERREUR : {exc}")


if __name__ == "__main__":
    main()

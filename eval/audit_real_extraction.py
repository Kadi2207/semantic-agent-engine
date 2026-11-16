"""Audit qualitatif EXPLORATOIRE de l'extraction (consigne gelee du bloc 1) sur 3 reunions AMI du split test QMSum.

Ce n'est pas une validation statistique : 3 reunions de lancement de projet, jouees selon un scenario (AMI),
en anglais alors que la consigne est en francais. Le CSV produit est destine a une relecture humaine
(colonnes correct_oui_non et commentaire laissees vides).

Decoupage : une reunion AMI depasse la taille d'un appel Groq (limite de debit ~8000 tokens/min avec la sortie
gelee a 5500 tokens). Chaque reunion est donc decoupee en segments consecutifs de SEGMENT_TURNS tours, sans
chevauchement. Les segments ne sont PAS traites comme des reunions independantes : ils sont enchaines par
extract_project(), qui fournit au LLM toutes les unites deja extraites des segments precedents de la meme
reunion ; une relation ou une mise a jour de statut peut donc viser une unite d'un segment precedent.
- aucun tour n'est coupe : une evidence (extrait d'une seule prise de parole) ne peut pas traverser une frontiere ;
- pas de chevauchement : un meme tour n'est vu qu'une fois, donc pas de doublon d'evidence aux frontieres ;
  les doublons de sens entre segments voisins sont recherches et signales ;
- provenance conservee : identifiant de reunion d'origine, segment, indice ABSOLU du tour de l'evidence.
La consigne, le modele et les parametres d'extraction sont ceux du bloc 1, inchanges.

Arret au premier echec : si un segment ne peut pas etre traite (ex. quota Groq atteint), tout s'arrete sans
rien ecrire, plutot que de poursuivre l'enchainement sans le contexte de ce segment. Les reponses deja obtenues
restent en cache et sont reutilisees a la relance. Les messages d'erreur bruts de Groq (qui contiennent des
identifiants de compte) ne sont jamais ecrits : seul un motif court l'est.

    uv run python eval/audit_real_extraction.py
    uv run python eval/audit_real_extraction.py --cache-only --csv <autre.csv> --meta <autre.json>   # reproductibilite
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.eval_real import QMSUM_DIR, clean_turn  # noqa: E402
from src.ingest import extract as extract_module  # noqa: E402
from src.ingest.extract import _norm, extract_project, make_client  # noqa: E402
from src.ingest.groq_client import CachedGroq, GroqError  # noqa: E402

MEETINGS = ["IS1003a", "ES2011a", "ES2004a"]  # les 3 reunions AMI du test les plus courtes, 3 equipes differentes
SEGMENT_TURNS = 40
PLACEHOLDER_DATE = "2000-01-01"  # QMSum ne donne pas la date des reunions ; date fictive commune
BOUNDARY_WINDOW = 3  # tours de part et d'autre d'une frontiere examines pour les controles
CSV_PATH = ROOT / "eval" / "results" / "real_extraction_sample.csv"
META_PATH = ROOT / "eval" / "results" / "real_extraction_sample_meta.json"
CSV_COLUMNS = ["source_id", "type", "statement", "evidence", "status", "correct_oui_non", "commentaire",
               "unit_id", "segment", "turn_index", "speaker"]


class AuditAborted(RuntimeError):
    """Volontairement PAS une GroqError : extract_units ne l'intercepte pas, l'audit s'arrete aussitot."""


def short_reason(exc: Exception) -> str:
    """Motif court, sans le corps de l'erreur Groq (qui contient l'identifiant d'organisation)."""
    text = str(exc)
    if "429" in text and "per day" in text:
        return "HTTP 429 (quota quotidien de tokens Groq atteint)"
    if "429" in text:
        return "HTTP 429 (limite de debit Groq)"
    match = re.match(r"(HTTP \d{3}|reseau|absente du cache)", text)
    return match.group(1) if match else type(exc).__name__


class FailFastClient:
    """Enveloppe CachedGroq : toute erreur Groq interrompt l'audit (meme interface pour extract_units)."""

    def __init__(self, inner: CachedGroq) -> None:
        self.inner = inner
        self.cache_only = inner.cache_only

    def complete(self, system_prompt: str, user_prompt: str, label: str = "", refresh: bool = False) -> str:
        try:
            return self.inner.complete(system_prompt, user_prompt, label=label, refresh=refresh)
        except GroqError as exc:
            raise AuditAborted(f"segment {label} non traite : {short_reason(exc)}") from None

    def snapshot(self) -> dict[str, int]:
        return self.inner.snapshot()


def segment_meeting(meeting_id: str, turns: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, list[tuple[int, str]]]]:
    """Segments consecutifs de SEGMENT_TURNS tours ; pour chacun, (indice absolu, texte) des tours non vides."""
    participants = sorted({t["speaker"].strip() for t in turns if t["speaker"].strip()})
    segments, lines_by_segment = [], {}
    for start in range(0, len(turns), SEGMENT_TURNS):
        end = min(start + SEGMENT_TURNS, len(turns)) - 1
        source_id = f"{meeting_id}#t{start:04d}-{end:04d}"
        lines = [(i, clean_turn(turns[i]["content"])) for i in range(start, end + 1)]
        lines = [(i, f"{turns[i]['speaker'].strip()}: {text}") for i, text in lines if text]
        lines_by_segment[source_id] = lines
        segments.append(
            {
                "source_id": source_id,
                "source_date": PLACEHOLDER_DATE,
                "participants": participants,
                "transcript": "\n".join(line for _, line in lines),
            }
        )
    return segments, lines_by_segment


def turn_index(evidence: str, lines: list[tuple[int, str]]) -> int | None:
    needle = _norm(evidence)
    for index, line in lines:
        if needle in _norm(line.split(":", 1)[1] if ":" in line else line):
            return index
    return None


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-only", action="store_true", help="n'appelle jamais Groq (reponses en cache seulement)")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--meta", type=Path, default=META_PATH)
    args = parser.parse_args()
    try:
        rows, meta = run_audit(FailFastClient(make_client(cache_only=args.cache_only)))
    except AuditAborted as exc:
        sys.exit(f"\nAUDIT INTERROMPU, aucun fichier ecrit : {exc}. Les reponses deja obtenues restent en cache.")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(rows)} unites ecrites dans {args.csv} ; controles dans {args.meta}")


def run_audit(client: FailFastClient) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    meta: dict[str, object] = {
        "nature": "audit qualitatif exploratoire, pas une validation statistique",
        "corpus": "AMI (CC BY 4.0, attribution : AMI Project), via QMSum split test, domaine Product",
        "meetings": MEETINGS,
        "segment_turns": SEGMENT_TURNS,
        "segment_overlap": 0,
        "source_date": f"{PLACEHOLDER_DATE} (fictive : QMSum ne donne pas de date)",
        "extraction": {
            "model": extract_module.GROQ_MODEL,
            "temperature": extract_module.GROQ_TEMPERATURE,
            "max_completion_tokens": extract_module.MAX_COMPLETION_TOKENS,
            "system_prompt_sha256": hashlib.sha256(extract_module.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        },
        "per_meeting": {},
    }
    for meeting_id in MEETINGS:
        data = json.loads((QMSUM_DIR / "Product" / "test" / f"{meeting_id}.json").read_text(encoding="utf-8"))
        turns = data["meeting_transcripts"]
        segments, lines_by_segment = segment_meeting(meeting_id, turns)
        print(f"[{meeting_id}] {len(turns)} tours, {len(segments)} segments", flush=True)
        before = client.snapshot()
        extraction = extract_project(meeting_id, segments, client)
        after = client.snapshot()

        segment_of = {u.unit_id: u.source_id for u in extraction.units}
        index_of = {u.unit_id: turn_index(u.evidence, lines_by_segment[u.source_id]) for u in extraction.units}
        for u in extraction.units:
            rows.append(
                {"source_id": meeting_id, "type": u.type, "statement": u.statement, "evidence": u.evidence,
                 "status": u.status, "correct_oui_non": "", "commentaire": "", "unit_id": u.unit_id,
                 "segment": u.source_id, "turn_index": index_of[u.unit_id], "speaker": u.speaker}
            )

        # ---- controles aux frontieres
        boundaries = [int(s["source_id"].split("#t")[1].split("-")[0]) for s in segments[1:]]  # 1er tour de chaque segment
        cross = [r for r in extraction.relations if segment_of.get(r.from_id) != segment_of.get(r.to_id)]
        near = [u for u in extraction.units if index_of[u.unit_id] is not None
                and any(b - BOUNDARY_WINDOW <= index_of[u.unit_id] < b + BOUNDARY_WINDOW for b in boundaries)]
        answered = {r.to_id for r in extraction.relations if r.relation == "answers"}
        open_questions_at_boundary = [
            {"unit_id": u.unit_id, "turn_index": index_of[u.unit_id], "statement": u.statement}
            for u in near if u.type == "question" and u.unit_id not in answered
            and any(b - BOUNDARY_WINDOW <= index_of[u.unit_id] < b for b in boundaries)
        ]
        evidence_seen: dict[tuple[int | None, str], str] = {}
        duplicate_evidence, near_duplicates = [], []
        units = extraction.units
        for u in units:
            key = (index_of[u.unit_id], _norm(u.evidence))
            if key in evidence_seen:
                duplicate_evidence.append([evidence_seen[key], u.unit_id])
            evidence_seen.setdefault(key, u.unit_id)
        for i, a in enumerate(units):
            for b in units[i + 1:]:
                if segment_of[a.unit_id] == segment_of[b.unit_id] or a.type != b.type:
                    continue
                wa, wb = words(a.statement), words(b.statement)
                if wa and wb and len(wa & wb) / len(wa | wb) >= 0.6:
                    near_duplicates.append([a.unit_id, b.unit_id])
        cost = {k: after[k] - before[k] for k in after}
        meta["per_meeting"][meeting_id] = {  # type: ignore[index]
            "turns": len(turns),
            "segments": [s["source_id"] for s in segments],
            "units": len(units),
            "relations": len(extraction.relations),
            "units_without_turn_index": [u.unit_id for u in units if index_of[u.unit_id] is None],
            "segments_not_processed": [m.source_id for m in extraction.meetings if any(k.startswith("reunion_") for k in m.rejections)],
            "rejections": dict(extraction.rejections),
            "notes": dict(extraction.notes),
            "boundary_checks": {
                "boundaries_first_turn": boundaries,
                "relations_crossing_a_boundary": [[r.from_id, r.relation, r.to_id] for r in cross],
                "units_within_window_of_a_boundary": len(near),
                "questions_just_before_a_boundary_without_answer": open_questions_at_boundary,
                "duplicate_evidence_across_segments": duplicate_evidence,
                "near_duplicate_statements_across_segments_jaccard_0.6": near_duplicates,
            },
            "relations_list": [[r.from_id, r.relation, r.to_id] for r in extraction.relations],
            "cost": cost,
        }
        print(f"    {len(units)} unites, {len(extraction.relations)} relations ({len(cross)} traversent une frontiere), "
              f"rejets {dict(extraction.rejections) or 'aucun'}, appels {cost['calls']}, cache {cost['cache_hits']}", flush=True)
    return rows, meta


if __name__ == "__main__":
    main()

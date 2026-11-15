"""Extraction d'unites semantiques et de relations a partir d'un transcript brut, via Groq.

CONSIGNE GELEE LE 2026-11-15, REGLEE SUR LES PROJETS 01-03 (deux iterations). Ne plus modifier
SYSTEM_PROMPT : les resultats des projets 04-12 (dont le test scelle 11-12) ont ete mesures avec
cette version exacte (empreinte sha256 enregistree dans eval/results/extraction.json).

extract_units(meeting, prior_units) traite UNE reunion. Les unites deja extraites des reunions
precedentes du meme projet peuvent etre fournies en contexte : le modele peut alors relier une
nouvelle unite a une ancienne (ex. une decision qui en remplace une autre) et declarer une ancienne
unite perimee. extract_project() enchaine les reunions d'un projet dans l'ordre chronologique.

Validation stricte de chaque unite proposee par le LLM, avec comptage des rejets par motif :
- schema pydantic SemanticUnit (type, statut, champs obligatoires) ;
- evidence = phrase EXACTE du transcript, contenue dans une seule prise de parole (sinon rejet) ;
  seules les variantes typographiques (apostrophes, guillemets, espaces insecables) sont tolerees,
  et l'evidence conservee est alors le texte exact du transcript ;
- le speaker est celui de la prise de parole qui contient l'evidence (corrige si le LLM se trompe) ;
- ids coherents : relations vers des unites existantes et acceptees, pas de boucle ni de doublon.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from src.ingest.groq_client import ROOT, CachedGroq, GroqError
from src.schemas import Relation, SemanticUnit

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_TEMPERATURE = 0.0
CACHE_PATH = ROOT / "data" / "processed" / "groq_extract_cache.jsonl"  # non versionne (data/processed/*)
# Une reunion = ~1 appel de plusieurs milliers de tokens (raisonnement compris) ; limite Groq ~8000 tokens/min.
MIN_SECONDS_BETWEEN_CALLS = 60.0
# Sans limite explicite, Groq coupe la sortie vers 3072 tokens (finish_reason=length observe) ; une reunion
# en demande ~2700 raisonnement compris. 5500 laisse de la marge tout en restant sous ~8000 tokens/appel.
MAX_COMPLETION_TOKENS = 5500

UNIT_TYPES = get_args(SemanticUnit.model_fields["type"].annotation)
STATUSES = get_args(SemanticUnit.model_fields["status"].annotation)
RELATION_TYPES = tuple(r for r in get_args(Relation.model_fields["relation"].annotation) if r != "unrelated")

SYSTEM_PROMPT = """Tu es un annotateur de comptes rendus de réunion. À partir du transcript d'UNE réunion, tu extrais les unités sémantiques et les relations entre elles. Tu réponds uniquement par un objet JSON.

TYPES D'UNITÉ
- decision : un choix acté par le groupe ou un responsable (« c'est acté », « on part sur », « va pour »).
- action : une tâche CONFIÉE PAR UNE PERSONNE À UNE AUTRE.
  Exemple — Paul : « Nina, tu envoies le devis au client avant jeudi ? » → action, statement « Nina doit envoyer le devis au client avant jeudi. »
- commitment : un engagement que la personne prend D'ELLE-MÊME, sans qu'on le lui ait demandé (« je m'engage à », « je m'en occupe », « je prépare »).
  Exemple — Nina : « De mon côté, je prépare le planning pour lundi. » → commitment, statement « Nina prépare le planning pour lundi. »
- question : une question posée qui appelle une réponse sur le fond. Une DEMANDE de tâche formulée comme une question (« tu peux… ? », « tu veux bien… ? », « est-ce que tu pourrais… ? ») n'est PAS une question : c'est une action (voir convention 1).
- answer : la réponse de fond apportée à une question posée juste avant, même si elle prend la forme d'un constat, d'un chiffre ou d'une condition ; relie-la à la question par la relation answers.
  Exemple — Paul : « Il reste combien sur l'enveloppe ? » / Nina : « Environ 12 000 euros. » → answer « Il reste environ 12 000 euros sur l'enveloppe. » answers la question.
- fact : un constat, un état des lieux, une information factuelle qui ne répond pas à une question.
  Un statut officiel annoncé (approuvé, refusé, validé, signé, gelé) est une decision, pas un fact : « Le budget est approuvé. » → decision.
- risk : une alerte, un risque ou une menace potentielle.
- idea : une proposition ou suggestion non encore actée.

CONVENTIONS
1. Une demande suivie de son acceptation = UNE SEULE unité. La tâche ayant été confiée par quelqu'un d'autre, c'est une action ; le sujet du statement est la personne qui fera la tâche ; l'evidence est la phrase qui énonce le plus précisément la tâche (souvent l'acceptation).
   Exemple — Paul : « Nina, tu pourrais relancer le fournisseur ? » / Nina : « Oui, je le relance demain matin. » → une seule unité : action « Nina relance le fournisseur le lendemain matin. », evidence « Oui, je le relance demain matin. » (et non une action + un commitment).
2. Ignore la logistique de réunion : ouverture et organisation de la séance, ordre du jour, objectif de la réunion, passage au point suivant, date du prochain point, remerciements.
   Exemples à NE PAS extraire : « Bon, on commence. », « Premier point, le budget. », « L'objectif aujourd'hui, c'est de faire le point sur Orion. », « On refait un point vendredi. », « On garde la réunion hebdomadaire. »
3. Relation supports : propose-la chaque fois qu'un fait, une réponse ou un constat justifie une décision ou une autre affirmation (from = ce qui justifie, to = ce qui est justifié).
   Exemple — Lina : « Les tests de charge passent. » / Paul : « Alors on lance lundi. » → fact « Les tests de charge passent. » supports decision « Le lancement est fixé à lundi. »
4. Relation creates_action : chaque action ou engagement qui découle d'une décision ou d'une réponse doit y être relié (from = la décision ou la réponse, to = l'action ou l'engagement).
   Exemple — Paul : « Va pour le fournisseur B. » / Paul : « Nina, tu lui envoies le bon de commande ? » / Nina : « Je l'envoie ce soir. » → decision « Le fournisseur B est retenu. » creates_action action « Nina envoie le bon de commande au fournisseur B le soir même. »

RÈGLES POUR CHAQUE UNITÉ
- statement : une phrase autonome en français, reformulée à la 3e personne, qui nomme explicitement les personnes, objets et dates (compréhensible sans le transcript). Pour une action ou un engagement, le sujet de la phrase est la personne qui DOIT faire la tâche, pas celle qui la demande.
- speaker : le prénom de la personne qui prononce l'evidence.
- evidence : un extrait COPIÉ À L'IDENTIQUE du transcript, caractère pour caractère, tiré d'une seule prise de parole, sans le nom du locuteur ni les deux-points. Jamais de reformulation, jamais de coupe au milieu avec « … ».
- status : « active » par défaut ; « superseded » si, dans cette même réunion, l'unité est remplacée par une information plus récente ; « cancelled » si une action ou un engagement est annulé ou suspendu.
- Une unité par information utile. Ignore les politesses, l'animation de réunion et les phrases sans contenu.

RELATIONS (from -> to)
- answers : from = la réponse, to = la question.
- creates_action : from = la décision ou réponse, to = l'action ou l'engagement qu'elle entraîne.
- supports : from = l'élément qui justifie, to = l'unité justifiée.
- supersedes : from = la nouvelle unité, to = l'ancienne qu'elle remplace (date, responsable, décision qui change).
- contradicts : from = l'unité qui contredit, to = l'unité contredite.
Une relation peut viser une unité des réunions précédentes (liste fournie, avec ses identifiants). Si une unité précédente devient périmée ou annulée à cause de cette réunion, indique-le dans status_updates.

FORMAT DE SORTIE (JSON strict)
{
  "units": [
    {"id": "u1", "type": "...", "statement": "...", "speaker": "...", "evidence": "...", "status": "active"}
  ],
  "relations": [
    {"from": "u2", "to": "u1", "relation": "answers"}
  ],
  "status_updates": [
    {"id": "<identifiant d'une unité précédente>", "status": "superseded"}
  ]
}
Les identifiants des nouvelles unités sont u1, u2, u3… dans l'ordre du transcript."""

_TYPO = str.maketrans({"’": "'", "‘": "'", "‛": "'", "´": "'", "“": '"', "”": '"', "«": '"', "»": '"', " ": " ", " ": " ", "‐": "-", "‑": "-", "–": "-", "—": "-"})
_SPEAKER_LINE = re.compile(r"^\s*([^:\n]{1,40}?)\s*:\s*(.*)$")
_LOCAL_ID = re.compile(r"^u\d+$")


def _norm(text: str) -> str:
    """Normalisation caractere pour caractere (longueur conservee) : seules les variantes typographiques."""
    return text.translate(_TYPO)


@dataclass
class MeetingExtraction:
    source_id: str
    units: list[SemanticUnit]
    relations: list[Relation]
    status_updates: dict[str, str]  # unit_id d'une reunion precedente -> nouveau statut
    rejections: Counter[str] = field(default_factory=Counter)
    notes: Counter[str] = field(default_factory=Counter)  # corrections non bloquantes
    proposed_units: int = 0
    proposed_relations: int = 0


@dataclass
class ProjectExtraction:
    project_id: str
    units: list[SemanticUnit]
    relations: list[Relation]
    meetings: list[MeetingExtraction]

    @property
    def rejections(self) -> Counter[str]:
        return sum((m.rejections for m in self.meetings), Counter())

    @property
    def notes(self) -> Counter[str]:
        return sum((m.notes for m in self.meetings), Counter())


# --------------------------------------------------------------------------- prompt


def build_user_prompt(meeting: dict[str, object], prior_units: list[SemanticUnit]) -> str:
    lines = [
        f"Réunion {meeting['source_id']} du {meeting['source_date']}",
        f"Participants : {', '.join(meeting['participants'])}",  # type: ignore[arg-type]
        "",
    ]
    if prior_units:
        lines.append("Unités extraites des réunions précédentes du même projet (identifiant | type | statut | énoncé) :")
        lines += [f"{u.unit_id} | {u.type} | {u.status} | {u.statement}" for u in prior_units]
    else:
        lines.append("Aucune réunion précédente pour ce projet.")
    lines += ["", "Transcript :", str(meeting["transcript"])]
    return "\n".join(lines)


def _parse_json(raw: str) -> dict[str, object]:
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("la racine JSON n'est pas un objet")
    return data


# --------------------------------------------------------------------------- validation


def _locate_evidence(evidence: str, turns: list[tuple[str, str]]) -> tuple[str, str] | None:
    """(speaker, texte exact du transcript) de la prise de parole qui contient l'evidence, sinon None."""
    needle = _norm(evidence).strip().strip('"').strip()
    if len(needle) < 3:
        return None
    for speaker, text in turns:
        start = _norm(text).find(needle)
        if start >= 0:
            return speaker, text[start : start + len(needle)]
    return None


def _turns(transcript: str) -> list[tuple[str, str]]:
    turns = []
    for line in transcript.splitlines():
        match = _SPEAKER_LINE.match(line)
        if match:
            turns.append((match.group(1).strip(), match.group(2)))
    return turns


def extract_units(
    meeting: dict[str, object],
    client: CachedGroq | None = None,
    prior_units: list[SemanticUnit] | None = None,
) -> MeetingExtraction:
    """Extrait et valide les unites et relations d'une reunion (transcript seul)."""
    client = client or make_client()
    prior_units = prior_units or []
    source_id = str(meeting["source_id"])
    source_date = date.fromisoformat(str(meeting["source_date"]))
    turns = _turns(str(meeting["transcript"]))
    result = MeetingExtraction(source_id, [], [], {})

    user_prompt = build_user_prompt(meeting, prior_units)
    data: dict[str, object] | None = None
    # Groq n'est pas deterministe meme a temperature 0 : il arrive que le modele s'arrete avant
    # d'avoir liste les relations. Une reponse sans cle "relations" est jugee incomplete et relancee
    # une fois hors cache (la nouvelle reponse remplace l'ancienne dans le cache).
    for attempt in (1, 2):
        try:
            raw = client.complete(SYSTEM_PROMPT, user_prompt, label=source_id, refresh=attempt == 2)
            data = _parse_json(raw)
        except GroqError as exc:
            result.rejections[f"reunion_non_traitee ({exc})"] += 1
            return result
        except ValueError:
            data = None
        if data is not None and "relations" in data:
            break
        if attempt == 1 and not client.cache_only:
            result.notes["reponse_incomplete_relancee"] += 1
            continue
        if data is None:
            result.rejections["reunion_json_invalide"] += 1
            return result
        result.notes["reponse_incomplete_conservee"] += 1
        break
    assert data is not None

    # ---- unites
    raw_units = data.get("units") or []
    raw_units = raw_units if isinstance(raw_units, list) else []
    result.proposed_units = len(raw_units)
    local_to_full: dict[str, str] = {}
    rejected_local: set[str] = set()
    seen_spans: set[tuple[str, str]] = set()
    for index, item in enumerate(raw_units, start=1):
        if not isinstance(item, dict):
            result.rejections["unite_mal_formee"] += 1
            continue
        local_id = str(item.get("id", f"u{index}")).strip()
        if not _LOCAL_ID.match(local_id) or local_id in local_to_full or local_id in rejected_local:
            result.rejections["id_invalide_ou_duplique"] += 1
            rejected_local.add(local_id)
            continue
        statement = str(item.get("statement") or "").strip()
        if not statement:
            result.rejections["statement_vide"] += 1
            rejected_local.add(local_id)
            continue
        located = _locate_evidence(str(item.get("evidence") or ""), turns)
        if located is None:
            result.rejections["evidence_absente_du_transcript"] += 1
            rejected_local.add(local_id)
            continue
        speaker, exact_evidence = located
        if exact_evidence != str(item.get("evidence", "")).strip().strip('"'):
            result.notes["evidence_typographie_normalisee"] += 1
        claimed_speaker = str(item.get("speaker") or "").strip()
        if claimed_speaker != speaker:
            result.notes["speaker_corrige_depuis_transcript"] += 1
        span_key = (str(item.get("type")), _norm(exact_evidence))
        if span_key in seen_spans:
            result.rejections["doublon_meme_type_meme_evidence"] += 1
            rejected_local.add(local_id)
            continue
        full_id = f"{source_id}_x{len(local_to_full) + 1:02d}"
        try:
            unit = SemanticUnit(
                unit_id=full_id,
                source_id=source_id,
                source_date=source_date,
                type=item.get("type"),
                statement=statement,
                speaker=speaker,
                evidence=exact_evidence,
                status=item.get("status") or "active",
            )
        except ValidationError as exc:
            field_name = exc.errors()[0]["loc"][0] if exc.errors() else "?"
            result.rejections[f"schema_invalide ({field_name})"] += 1
            rejected_local.add(local_id)
            continue
        seen_spans.add(span_key)
        local_to_full[local_id] = full_id
        result.units.append(unit)

    # ---- relations
    known = {u.unit_id for u in prior_units} | set(local_to_full.values())
    current = set(local_to_full.values())
    raw_relations = data.get("relations") or []
    raw_relations = raw_relations if isinstance(raw_relations, list) else []
    result.proposed_relations = len(raw_relations)
    seen_relations: set[tuple[str, str, str]] = set()
    for item in raw_relations:
        if not isinstance(item, dict):
            result.rejections["relation_mal_formee"] += 1
            continue
        ends = []
        for key in ("from", "to"):
            ref = str(item.get(key, "")).strip()
            if ref in rejected_local:
                ends.append(None)
                result.rejections["relation_vers_unite_rejetee"] += 1
                break
            ends.append(local_to_full.get(ref, ref if ref in known else ""))
        if None in ends:
            continue
        from_id, to_id = ends
        relation_type = str(item.get("relation", "")).strip()
        if not from_id or not to_id:
            result.rejections["relation_id_inconnu"] += 1
        elif relation_type not in RELATION_TYPES:
            result.rejections["relation_type_invalide"] += 1
        elif from_id == to_id:
            result.rejections["relation_boucle"] += 1
        elif from_id not in current and to_id not in current:
            result.rejections["relation_hors_reunion"] += 1
        elif (from_id, to_id, relation_type) in seen_relations:
            result.rejections["relation_doublon"] += 1
        else:
            seen_relations.add((from_id, to_id, relation_type))
            result.relations.append(
                Relation(from_id=from_id, to_id=to_id, relation=relation_type, evidence_source_id=source_id)
            )

    # ---- mises a jour de statut d'unites precedentes
    prior_ids = {u.unit_id for u in prior_units}
    raw_updates = data.get("status_updates") or []
    for item in raw_updates if isinstance(raw_updates, list) else []:
        if not isinstance(item, dict):
            result.rejections["mise_a_jour_mal_formee"] += 1
            continue
        unit_id, status = str(item.get("id", "")).strip(), str(item.get("status", "")).strip()
        if unit_id not in prior_ids:
            result.rejections["mise_a_jour_id_inconnu"] += 1
        elif status not in STATUSES:
            result.rejections["mise_a_jour_statut_invalide"] += 1
        else:
            result.status_updates[unit_id] = status
    return result


def extract_project(project_id: str, meetings: list[dict[str, object]], client: CachedGroq) -> ProjectExtraction:
    """Enchaine les reunions dans l'ordre chronologique, en donnant au LLM les unites deja extraites."""
    units: list[SemanticUnit] = []
    relations: list[Relation] = []
    extractions: list[MeetingExtraction] = []
    for meeting in sorted(meetings, key=lambda m: (str(m["source_date"]), str(m["source_id"]))):
        extraction = extract_units(meeting, client, prior_units=units)
        extractions.append(extraction)
        units = [
            u.model_copy(update={"status": extraction.status_updates[u.unit_id]}) if u.unit_id in extraction.status_updates else u
            for u in units
        ] + extraction.units
        relations += extraction.relations

    # Coherence : une unite remplacee (cible d'un supersedes) ne peut pas rester active.
    superseded = {r.to_id for r in relations if r.relation == "supersedes"}
    fixed = Counter()
    for i, unit in enumerate(units):
        if unit.unit_id in superseded and unit.status == "active":
            units[i] = unit.model_copy(update={"status": "superseded"})
            fixed["statut_deduit_de_supersedes"] += 1
    if extractions:
        extractions[-1].notes.update(fixed)
    return ProjectExtraction(project_id, units, relations, extractions)


def make_client(cache_only: bool = False, cache_path: Path = CACHE_PATH) -> CachedGroq:
    return CachedGroq(
        model=GROQ_MODEL,
        temperature=GROQ_TEMPERATURE,
        cache_path=cache_path,
        min_seconds_between_calls=MIN_SECONDS_BETWEEN_CALLS,
        json_mode=True,
        cache_only=cache_only,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )

"""Seconde passe de liaison des relations (lot relations v2), a cote de l'extraction figee.

Entree : les unites et relations deja produites par l'extraction (consigne gelee de src/ingest/extract.py, rejouee
depuis son cache, aucun appel). La consigne d'extraction n'est pas modifiee : cette passe a sa propre consigne
(LINK_SYSTEM_PROMPT), son propre cache (data/processed/groq_link_cache.jsonl) et ses propres resultats.

1. Candidats (candidate_pairs) : pour chaque unite extraite, les k unites extraites les plus proches du meme projet
   (cosinus e5 fine-tune) parmi celles qui la precedent dans l'ordre chronologique (reunion plus ancienne, ou meme
   reunion et rang plus petit), plus toutes les paires deja reliees par l'extraction.
2. Liaison (link_project) : UN appel Groq par projet (gpt-oss-120b, JSON, temperature 0) qui recoit les unites du
   projet et la liste numerotee des paires candidates, et rend pour chaque paire un type parmi supersedes,
   contradicts, creates_action, answers, supports, ou aucun, avec son sens.
3. Validation par le code (parse_links) : numero de paire connu, type autorise, sens A->B ou B->A, pas de doublon ;
   tout le reste est rejete et compte par motif. Le modele ne peut relier que des paires candidates.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from src.ingest.groq_client import ROOT, CachedGroq, GroqError
from src.schemas import Relation, SemanticUnit

LINK_MODEL = "openai/gpt-oss-120b"
LINK_TEMPERATURE = 0.0
LINK_CACHE = ROOT / "data" / "processed" / "groq_link_cache.jsonl"
LINK_EXTRA = {"reasoning_effort": "low"}
LINK_MAX_COMPLETION_TOKENS = 6000
SECONDS_BETWEEN_CALLS = 60.0  # limite Groq ~8000 tokens/min ; un appel en consomme plusieurs milliers
LINK_TYPES = ("supersedes", "contradicts", "creates_action", "answers", "supports")
NONE = "aucun"

LINK_PROMPT_V1 = """Tu relies entre elles les unités sémantiques extraites des comptes rendus de réunion d'UN projet. Tu réponds uniquement par un objet JSON.

Tu reçois la liste des unités (identifiant | réunion et date | type | statut | énoncé), puis une liste numérotée de paires candidates (A, B), où A est l'unité la plus ancienne. Pour CHAQUE paire, tu choisis exactement une relation, ou « aucun ».

RELATIONS (from -> to)
- supersedes : from = la nouvelle unité, to = l'ancienne qu'elle remplace sur le même point (date, responsable, montant, choix qui change). Les deux unités parlent de la même chose et la plus récente rend l'autre périmée.
- contradicts : from = l'unité qui contredit, to = l'unité contredite, quand les deux affirmations sont incompatibles sans que l'une remplace explicitement l'autre.
- creates_action : from = la décision ou la réponse, to = l'action ou l'engagement qui en découle directement.
- answers : from = la réponse, to = la question à laquelle elle répond.
- supports : from = l'élément qui justifie (fait, réponse, constat, risque), to = l'unité qu'il justifie (souvent une décision).
- aucun : les deux unités ne sont pas liées par l'une de ces relations, ou le lien est seulement thématique.

RÈGLES
- Une relation exige un lien direct et explicite entre les deux énoncés ; partager un sujet ne suffit pas. En cas de doute, réponds « aucun ».
- Respecte les types : answers va vers une question ; creates_action va vers une action ou un engagement ; supersedes relie deux unités du même type ou presque (décision et décision, action et action).
- Le sens compte : « A->B » signifie from = A, to = B ; « B->A » signifie from = B, to = A. B est l'unité la plus récente : une unité qui en remplace une autre, une réponse ou une action qui découle d'une décision est en général B.

FORMAT DE SORTIE (JSON strict, une entrée par paire, dans l'ordre des numéros)
{"links": [{"pair": 1, "relation": "aucun"}, {"pair": 2, "relation": "supersedes", "direction": "B->A"}]}"""

# v2 : seule revision, faite apres l'analyse des erreurs de la v1 sur le dev 04-10 (trop de relations proposees :
# contradicts et supports surtout). Plus prudente ; les paires deja reliees par l'extraction sont signalees.
LINK_PROMPT_V2 = """Tu relies entre elles les unités sémantiques extraites des comptes rendus de réunion d'UN projet. Tu réponds uniquement par un objet JSON.

Tu reçois la liste des unités (identifiant | réunion et date | type | statut | énoncé), puis une liste numérotée de paires candidates (A, B), où A est l'unité la plus ancienne. Ces paires sont seulement proches par le sens : la grande majorité ne sont liées par AUCUNE relation. Pour CHAQUE paire, tu choisis exactement une relation, ou « aucun ».

RELATIONS (from -> to) ET CONDITIONS STRICTES
- answers : from = la réponse, to = la question. B est une answer ou un fact qui répond précisément à la question A (type question).
- creates_action : from = la décision (ou la réponse), to = l'action ou l'engagement. L'action ou l'engagement exécute concrètement cette décision précise ; le même objet est nommé dans les deux énoncés.
- supersedes : from = la nouvelle unité, to = l'ancienne. Les deux énoncés portent sur le MÊME point (même date, même responsable, même montant ou même choix) et la nouvelle valeur remplace l'ancienne. Deux unités sur le même sujet mais sur des points différents ne relèvent pas de supersedes.
- contradicts : très rare. Seulement si deux affirmations sur le même point sont incompatibles ET qu'aucune ne remplace explicitement l'autre. Une décision qui change une décision antérieure relève de supersedes, pas de contradicts. Un risque, une idée ou une question ne contredit rien.
- supports : from = l'élément qui justifie, to = l'unité justifiée. Seulement si l'énoncé de from est donné comme la RAISON de to (« parce que », « donc », « vu que », chiffre qui motive le choix). Deux unités sur le même sujet ne se justifient pas pour autant.
- aucun : dans tous les autres cas, et en cas de doute.

RÈGLES
- Une paire marquée « déjà reliée par l'extraction » a déjà sa relation : réponds « aucun » pour elle, sauf si le sens indiqué est manifestement inversé.
- Respecte les types : answers va vers une question ; creates_action va d'une decision ou answer vers une action ou un commitment ; supersedes relie deux unités du même type.
- Le sens compte : « A->B » signifie from = A, to = B ; « B->A » signifie from = B, to = A.

FORMAT DE SORTIE (JSON strict, une entrée par paire, dans l'ordre des numéros)
{"links": [{"pair": 1, "relation": "aucun"}, {"pair": 2, "relation": "supersedes", "direction": "B->A"}]}"""

LINK_PROMPTS = {"v1": LINK_PROMPT_V1, "v2": LINK_PROMPT_V2}
LINK_SYSTEM_PROMPT = LINK_PROMPT_V2  # derniere version ; la version retenue est enregistree dans relations_v2.json


@dataclass
class LinkResult:
    project_id: str
    relations: list[Relation]
    pairs: list[tuple[str, str]]
    rejections: Counter[str] = field(default_factory=Counter)
    decisions: Counter[str] = field(default_factory=Counter)


def chronological_order(units: list[SemanticUnit]) -> list[SemanticUnit]:
    """Ordre de reunion (date, puis source_id), puis ordre d'extraction dans la reunion (suffixe _xNN)."""
    return sorted(units, key=lambda u: (u.source_date.isoformat(), u.source_id, u.unit_id))


def candidate_pairs(
    units: list[SemanticUnit], relations: list[Relation], embeddings: dict[str, np.ndarray], k: int
) -> list[tuple[str, str]]:
    """Paires (A plus ancienne, B plus recente) : k plus proches anterieures de chaque unite + paires deja reliees."""
    ordered = chronological_order(units)
    rank = {u.unit_id: i for i, u in enumerate(ordered)}
    pairs: set[tuple[str, str]] = set()
    for i, unit in enumerate(ordered):
        earlier = ordered[:i]
        if not earlier or k <= 0:
            continue
        scores = np.array([float(embeddings[e.unit_id] @ embeddings[unit.unit_id]) for e in earlier])
        for j in np.argsort(-scores, kind="stable")[:k]:
            pairs.add((earlier[int(j)].unit_id, unit.unit_id))
    for rel in relations:
        if rel.from_id in rank and rel.to_id in rank and rel.from_id != rel.to_id:
            a, b = sorted((rel.from_id, rel.to_id), key=rank.__getitem__)
            pairs.add((a, b))
    return sorted(pairs, key=lambda p: (rank[p[1]], rank[p[0]]))


def build_link_prompt(units: list[SemanticUnit], pairs: list[tuple[str, str]], existing: list[Relation] | None = None) -> str:
    """existing (consigne v2 seulement) : signale les paires deja reliees par l'extraction, avec type et sens."""
    lines = ["Unités du projet (identifiant | réunion et date | type | statut | énoncé) :"]
    lines += [f"{u.unit_id} | {u.source_id} du {u.source_date.isoformat()} | {u.type} | {u.status} | {u.statement}" for u in chronological_order(units)]
    lines += ["", "Paires candidates (numéro : A = plus ancienne, B = plus récente) :"]
    for n, (a, b) in enumerate(pairs, start=1):
        line = f"{n} : A = {a} ; B = {b}"
        if existing is not None:
            for rel in existing:
                if {rel.from_id, rel.to_id} == {a, b}:
                    line += f" (déjà reliée par l'extraction : {rel.relation} {'A->B' if rel.from_id == a else 'B->A'})"
                    break
        lines.append(line)
    return "\n".join(lines)


def _parse_json(raw: str) -> dict[str, object]:
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    data = json.loads(fenced.group(1) if fenced else text)
    if not isinstance(data, dict):
        raise ValueError("la racine JSON n'est pas un objet")
    return data


def parse_links(raw: str, project_id: str, pairs: list[tuple[str, str]], source_of: dict[str, str]) -> LinkResult:
    """Valide la reponse du modele : seules des paires candidates, des types autorises et un sens explicite."""
    result = LinkResult(project_id, [], pairs)
    try:
        data = _parse_json(raw)
    except (ValueError, json.JSONDecodeError):
        result.rejections["reponse_json_invalide"] += 1
        return result
    items = data.get("links")
    if not isinstance(items, list):
        result.rejections["cle_links_absente"] += 1
        return result
    seen_pairs: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            result.rejections["entree_mal_formee"] += 1
            continue
        try:
            number = int(item.get("pair"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            result.rejections["numero_de_paire_invalide"] += 1
            continue
        if not 1 <= number <= len(pairs):
            result.rejections["numero_de_paire_inconnu"] += 1
            continue
        if number in seen_pairs:
            result.rejections["paire_en_double"] += 1
            continue
        seen_pairs.add(number)
        relation = str(item.get("relation", "")).strip()
        if relation == NONE:
            result.decisions[NONE] += 1
            continue
        if relation not in LINK_TYPES:
            result.rejections["type_invalide"] += 1
            continue
        direction = str(item.get("direction", "")).replace(" ", "").replace("→", "->")
        a, b = pairs[number - 1]
        if direction == "A->B":
            from_id, to_id = a, b
        elif direction == "B->A":
            from_id, to_id = b, a
        else:
            result.rejections["sens_invalide"] += 1
            continue
        result.decisions[relation] += 1
        result.relations.append(Relation(from_id=from_id, to_id=to_id, relation=relation, evidence_source_id=source_of[from_id]))
    result.rejections["paires_sans_reponse"] += len(pairs) - len(seen_pairs)
    if not result.rejections["paires_sans_reponse"]:
        del result.rejections["paires_sans_reponse"]
    return result


def link_prompts(version: str, units: list[SemanticUnit], pairs: list[tuple[str, str]], existing: list[Relation]) -> tuple[str, str]:
    """(consigne systeme, message) d'une version ; la v1 ne signale pas les paires deja reliees."""
    return LINK_PROMPTS[version], build_link_prompt(units, pairs, existing if version != "v1" else None)


def link_project(
    project_id: str, units: list[SemanticUnit], pairs: list[tuple[str, str]], client: CachedGroq,
    version: str = "v2", existing: list[Relation] | None = None,
) -> LinkResult:
    """Un appel par projet ; en cas d'echec Groq, aucune relation (et le motif est compte)."""
    if not pairs:
        return LinkResult(project_id, [], pairs)
    system, user = link_prompts(version, units, pairs, existing or [])
    try:
        raw = client.complete(system, user, label=f"link {version} {project_id}")
    except GroqError:  # le message brut de Groq peut contenir des identifiants de compte : jamais enregistre
        result = LinkResult(project_id, [], pairs)
        result.rejections["appel_non_traite"] += 1
        return result
    return parse_links(raw, project_id, pairs, {u.unit_id: u.source_id for u in units})


def make_link_client(cache_only: bool = False) -> CachedGroq:
    return CachedGroq(
        model=LINK_MODEL,
        temperature=LINK_TEMPERATURE,
        cache_path=LINK_CACHE,
        min_seconds_between_calls=SECONDS_BETWEEN_CALLS,
        json_mode=True,
        cache_only=cache_only,
        max_completion_tokens=LINK_MAX_COMPLETION_TOKENS,
        extra_body=LINK_EXTRA,
    )

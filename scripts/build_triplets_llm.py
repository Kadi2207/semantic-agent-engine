"""Triplets d'entrainement enrichis : ancres reformulees par Groq + hard negative mine (projets 01-10).

Anti-fuite : aucune unite citee dans queries_dev.json (relevant_unit_ids) n'est utilisee,
ni comme positif ni comme negatif.

Les reponses brutes de Groq sont mises en cache (data/processed/groq_cache.jsonl, non versionne) :
un nouveau run ne rappelle Groq que pour les unites absentes du cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schemas import SemanticUnit  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "processed" / "triplets_llm.jsonl"
CACHE_PATH = ROOT / "data" / "processed" / "groq_cache.jsonl"
QUERIES_PATH = ROOT / "data" / "eval" / "queries_dev.json"
ENV_PATH = ROOT / ".env"
DEV_PROJECTS = range(1, 11)

MODEL_NAME = "intfloat/multilingual-e5-small"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile n'est plus servi par Groq (404 model_not_found)
GROQ_TEMPERATURE = 0.4
QUESTIONS_PER_UNIT = 4
REQUEST_TIMEOUT = 30
DEFAULT_RETRY_WAIT = 30.0
# Limite Groq de 8000 tokens/minute, ~1200-2300 tokens par appel : ~5 appels/minute au plus.
MIN_SECONDS_BETWEEN_CALLS = 12.0
# Sur un 429, le retry-after annonce par Groq (< 1 s) ne suffit pas a vider la fenetre d'une minute.
MIN_RATE_LIMIT_WAIT = 30.0

MIN_NEGATIVE_SIMILARITY = 0.70
MAX_NEGATIVE_SIMILARITY = 0.97
MAX_COPY_RATIO = 0.75
MIN_ANCHOR_WORDS = 4
MAX_ANCHOR_WORDS = 30

POSITIVE_TYPES = {"decision", "action", "commitment", "fact", "risk", "idea"}
# Relations dont l'autre extremite peut aussi repondre a l'ancre : jamais negatif.
FALSE_NEGATIVE_RELATIONS = {"supports", "answers", "creates_action"}

SYSTEM_PROMPT = (
    "Tu recois une information factuelle en francais. Genere exactement 4 questions differentes et "
    "naturelles dont la reponse est precisement cette information. Regles : chaque question doit "
    "vraiment trouver sa reponse dans l'information donnee ; ne recopie pas la phrase ; ne revele pas "
    "la reponse dans la question (aucun nom de personne, aucune date, aucune duree, aucun montant "
    "chiffre qui donnerait la reponse) ; garde les termes metier naturels (par exemple API de "
    "paiement, support, export comptable) au lieu de synonymes alambiques ; chaque question doit se "
    "comprendre seule, sans contexte. Reponds avec seulement les 4 questions, une par ligne, sans "
    "numerotation."
)

LIST_PREFIX = re.compile(r"^\s*(?:[-*•–]+|\(?\d+[.)]|[a-d][.)])\s*")
# Typographie du LLM -> typographie du corpus : trait d'union insecable et apostrophes courbes.
TYPOGRAPHY = str.maketrans({"‑": "-", "’": "'", "‘": "'"})
# Demonstratif isole : la question depend d'un contexte absent ("cet engagement").
DEMONSTRATIVE = re.compile(r"\b(?:cet|cette|ces)\b", re.IGNORECASE)
# Question "meta" portant sur la phrase elle-meme ("Quel enonce resume...") et non sur les faits.
META = re.compile(
    r"\b(?:enonce|énoncé|phrase|affirmation|formuler|formule|resume|résume|reformuler)\b", re.IGNORECASE
)


class MissingKeyError(RuntimeError):
    pass


class GroqError(RuntimeError):
    pass


@dataclass(frozen=True)
class Positive:
    unit: SemanticUnit
    origin: SemanticUnit  # la question d'origine si le positif est une reponse, sinon l'unite elle-meme
    project_id: str


@dataclass(frozen=True)
class Triplet:
    anchor: str
    positive: str
    negative: str
    source: str

    def as_record(self) -> dict[str, str]:
        return {"anchor": self.anchor, "positive": self.positive, "negative": self.negative, "source": self.source}


# --------------------------------------------------------------------------- cle API

def load_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip().removeprefix("export ").strip() == "GROQ_API_KEY":
                key = value.strip().strip("\"'")
                if key:
                    return key
    raise MissingKeyError(
        "GROQ_API_KEY introuvable : ni dans l'environnement, ni (non vide) dans .env. "
        "Colle ta cle dans .env (GROQ_API_KEY=...) puis relance."
    )


# --------------------------------------------------------------------------- Groq

_last_call_at = 0.0


def _pace() -> None:
    """Espace les appels d'au moins MIN_SECONDS_BETWEEN_CALLS (limite de tokens par minute)."""
    global _last_call_at
    wait = _last_call_at + MIN_SECONDS_BETWEEN_CALLS - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def cache_key(user_prompt: str) -> str:
    """Cle dependant du modele, de la temperature, de la consigne et de l'enonce : tout changement invalide."""
    material = json.dumps([GROQ_MODEL, GROQ_TEMPERATURE, SYSTEM_PROMPT, user_prompt], ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def load_cache() -> dict[str, str]:
    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                cache[entry["key"]] = entry["raw"]
    return cache


def append_cache(key: str, unit_id: str, raw: str) -> None:
    """Ecrit chaque reponse des sa reception : un run interrompu garde ce qui est deja obtenu."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {"key": key, "unit_id": unit_id, "model": GROQ_MODEL, "raw": raw}
    with CACHE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def call_groq(api_key: str, user_prompt: str) -> str:
    payload = json.dumps(
        {
            "model": GROQ_MODEL,
            "temperature": GROQ_TEMPERATURE,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
    ).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "triplet-builder/1.0",
    }
    for attempt in (1, 2):
        request = urllib.request.Request(GROQ_URL, data=payload, headers=headers, method="POST")
        _pace()
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                body = json.loads(response.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as exc:
            detail = _error_message(exc)
            if exc.code in (401, 403):
                raise MissingKeyError(f"Groq refuse la cle (HTTP {exc.code}, {detail}) : verifie GROQ_API_KEY.") from None
            if exc.code == 429 and attempt == 1:
                wait = max(_retry_after(exc.headers.get("retry-after")), MIN_RATE_LIMIT_WAIT)
                print(f"    [429] limite de debit, attente {wait:.0f}s puis nouvel essai")
                time.sleep(wait)
                continue
            raise GroqError(f"HTTP {exc.code} : {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == 1:
                print(f"    [reseau] {type(exc).__name__}, nouvel essai dans 5s")
                time.sleep(5)
                continue
            raise GroqError(f"reseau : {type(exc).__name__}") from None
        except (KeyError, IndexError, ValueError):
            raise GroqError("reponse Groq illisible") from None
    raise GroqError("echec apres nouvel essai")


def _error_message(exc: urllib.error.HTTPError) -> str:
    """Message d'erreur renvoye par Groq (le corps ne contient jamais la cle)."""
    try:
        body = exc.read().decode("utf-8", "replace")
        return str(json.loads(body)["error"]["message"])[:300]
    except Exception:
        return "corps d'erreur illisible"


def _retry_after(value: str | None) -> float:
    try:
        return min(max(float(value), 1.0), 120.0) if value else DEFAULT_RETRY_WAIT
    except ValueError:
        return DEFAULT_RETRY_WAIT


def build_prompt(positive: Positive) -> str:
    prompt = f"Enonce : {positive.unit.statement}"
    if positive.origin.unit_id != positive.unit.unit_id:
        # Une reponse seule ("Oui, c'est valide") est souvent incomprehensible sans sa question.
        prompt += f"\n(Contexte : cet enonce repond a la question \"{positive.origin.statement}\".)"
    return prompt


# --------------------------------------------------------------------------- filtrage des ancres

def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def copy_ratio(anchor: str, statement: str) -> float:
    anchor_words, statement_words = set(words(anchor)), words(statement)
    if not statement_words:
        return 0.0
    return sum(1 for w in statement_words if w in anchor_words) / len(statement_words)


def names_pattern(names: set[str]) -> re.Pattern[str] | None:
    """Noms de personnes du projet, en mot entier et sensible a la casse ("Claire" oui, "claire" non)."""
    if not names:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")\b")


def parse_questions(
    raw: str, statement: str, leaked_queries: set[str], names: re.Pattern[str] | None
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Retourne (ancres gardees, [(ancre, code de rejet, detail)])."""
    kept: list[str] = []
    rejected: list[tuple[str, str, str]] = []
    for line in raw.splitlines():
        text = LIST_PREFIX.sub("", line.translate(TYPOGRAPHY)).strip().strip("\"'«»“” ").strip()
        if not text:
            continue
        n_words = len(words(text))
        ratio = copy_ratio(text, statement)
        name = names.search(text) if names else None
        demonstrative = DEMONSTRATIVE.search(text)
        meta = META.search(text)
        if not text.endswith("?"):
            rejected.append((text, "ancre_sans_point_interrogation", "pas de ?"))
        elif not MIN_ANCHOR_WORDS <= n_words <= MAX_ANCHOR_WORDS:
            rejected.append((text, "ancre_longueur", f"{n_words} mots"))
        elif name:
            rejected.append((text, "ancre_nom_personne", name.group(0)))
        elif demonstrative:
            rejected.append((text, "ancre_demonstratif", demonstrative.group(0)))
        elif meta:
            rejected.append((text, "ancre_meta", meta.group(0)))
        elif ratio > MAX_COPY_RATIO:
            rejected.append((text, "ancre_recopie", f"{ratio:.0%} de l'enonce"))
        elif normalize(text) in leaked_queries:
            rejected.append((text, "ancre_identique_dev", "requete dev"))
        elif text in kept:
            rejected.append((text, "ancre_doublon", "doublon"))
        elif len(kept) >= QUESTIONS_PER_UNIT:
            rejected.append((text, "ancre_au_dela_de_4", "au-dela de 4"))
        else:
            kept.append(text)
    return kept, rejected


# --------------------------------------------------------------------------- donnees

def load_dev() -> tuple[set[str], set[str]]:
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    leaked_ids = {unit_id for q in queries for unit_id in q["relevant_unit_ids"]}
    leaked_queries = {normalize(q["query"]) for q in queries}
    return leaked_ids, leaked_queries


def load_projects() -> list[dict[str, object]]:
    projects = []
    for path in sorted(RAW_DIR.glob("projet_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(str(data["project_id"]).split("_")[1]) in DEV_PROJECTS:
            projects.append(data)
    return projects


def select_positives(
    project_id: str, units: dict[str, SemanticUnit], relations: list[dict[str, str]], leaked_ids: set[str], dropped: Counter[str]
) -> list[Positive]:
    answer_of = {r["to_id"]: r["from_id"] for r in relations if r["relation"] == "answers"}
    positives: list[Positive] = []
    used: set[str] = set()
    for unit in units.values():
        if unit.status != "active":
            continue
        if unit.type == "question":
            answer = units.get(answer_of.get(unit.unit_id, ""))
            if answer is None or answer.status != "active":
                dropped["question_sans_reponse_active"] += 1
                continue
            if unit.unit_id in leaked_ids or answer.unit_id in leaked_ids:
                dropped["positif_leaked"] += 1
                continue
            candidate = Positive(answer, unit, project_id)
        elif unit.type in POSITIVE_TYPES:
            if unit.unit_id in leaked_ids:
                dropped["positif_leaked"] += 1
                continue
            candidate = Positive(unit, unit, project_id)
        else:
            continue
        if candidate.unit.unit_id not in used:
            used.add(candidate.unit.unit_id)
            positives.append(candidate)
    return positives


# --------------------------------------------------------------------------- main

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="ne traiter que N unites positives (test)")
    parser.add_argument("--write", action="store_true", help="ecrire data/processed/triplets_dev.jsonl")
    parser.add_argument("--sample", type=int, default=0, help="afficher N triplets tires au hasard a la fin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="n'appelle jamais Groq : seules les unites presentes dans groq_cache.jsonl sont traitees",
    )
    args = parser.parse_args()

    api_key = ""
    if args.cache_only:
        print("Mode --cache-only : aucun appel Groq, unites absentes du cache ignorees.")
    else:
        try:
            api_key = load_api_key()
        except MissingKeyError as exc:
            sys.exit(f"ERREUR : {exc}")

    leaked_ids, leaked_queries = load_dev()
    dropped: Counter[str] = Counter()

    # 1. selection des positifs (tous projets 01-10), puis --limit
    projects = {str(d["project_id"]): d for d in load_projects()}
    project_units: dict[str, dict[str, SemanticUnit]] = {}
    project_names: dict[str, re.Pattern[str] | None] = {}
    positives: list[Positive] = []
    for project_id, data in projects.items():
        units = {raw["unit_id"]: SemanticUnit.model_validate(raw) for raw in data["units"]}  # type: ignore[index]
        project_units[project_id] = units
        names = {n for m in data["meetings"] for n in m["participants"]} | {  # type: ignore[union-attr,index]
            u.speaker for u in units.values() if u.speaker
        }
        project_names[project_id] = names_pattern(names)
        positives += select_positives(project_id, units, data["relations"], leaked_ids, dropped)  # type: ignore[arg-type]
    print(f"Positifs eligibles (projets 01-10, hors fuite) : {len(positives)}")
    if args.limit is not None:
        positives = positives[: args.limit]
        print(f"--limit {args.limit} : {len(positives)} unites traitees")

    # 2. generation des ancres par Groq (cache d'abord)
    cache = load_cache()
    cache_hits = groq_calls = 0
    anchors: dict[str, list[str]] = {}
    for n, positive in enumerate(positives, 1):
        unit = positive.unit
        print(f"\n[{n}/{len(positives)}] {unit.unit_id} ({unit.type}) : {unit.statement}", flush=True)
        if positive.origin is not unit:
            print(f"    via la question {positive.origin.unit_id} : {positive.origin.statement}")
        prompt = build_prompt(positive)
        key = cache_key(prompt)
        raw = cache.get(key)
        if raw is not None:
            cache_hits += 1
            print("    (cache)")
        elif args.cache_only:
            print("    (absente du cache : ignoree)")
            dropped["absente_du_cache"] += 1
            continue
        else:
            try:
                raw = call_groq(api_key, prompt)
            except MissingKeyError as exc:
                sys.exit(f"ERREUR : {exc}")
            except GroqError as exc:
                print(f"    ECHEC Groq ({exc}) : unite ignoree")
                dropped["echec_groq"] += 1
                continue
            groq_calls += 1
            cache[key] = raw
            append_cache(key, unit.unit_id, raw)
        kept, rejected = parse_questions(raw, unit.statement, leaked_queries, project_names[positive.project_id])
        for question in kept:
            print(f"    + {question}")
        for question, code, detail in rejected:
            print(f"    - {question}   [rejetee : {code} ({detail})]")
            dropped[code] += 1
        anchors[unit.unit_id] = kept

    # 3. mining du hard negative dans le meme projet
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    triplets: list[Triplet] = []
    seen: set[tuple[str, str, str]] = set()
    neg_above_pos = 0
    print("\nMining des negatifs :")
    for project_id in sorted({p.project_id for p in positives}):
        units = project_units[project_id]
        relations = projects[project_id]["relations"]
        related: dict[str, set[str]] = {}
        for rel in relations:  # type: ignore[union-attr]
            if rel["relation"] in FALSE_NEGATIVE_RELATIONS:
                related.setdefault(rel["from_id"], set()).add(rel["to_id"])
                related.setdefault(rel["to_id"], set()).add(rel["from_id"])
        ids = list(units)
        passages = np.asarray(
            model.encode([f"passage: {units[i].statement}" for i in ids], normalize_embeddings=True, show_progress_bar=False)
        )
        for positive in (p for p in positives if p.project_id == project_id):
            questions = anchors.get(positive.unit.unit_id, [])
            if not questions:
                continue
            pos_id = positive.unit.unit_id
            banned = {pos_id, positive.origin.unit_id} | leaked_ids | related.get(pos_id, set())
            banned |= {i for i, u in units.items() if u.type == "question"}
            candidates = [j for j, i in enumerate(ids) if i not in banned and units[i].statement != positive.unit.statement]
            encoded = np.asarray(
                model.encode([f"query: {q}" for q in questions], normalize_embeddings=True, show_progress_bar=False)
            )
            for question, query_vec in zip(questions, encoded):
                scores = passages @ query_vec
                sim_pos = float(scores[ids.index(pos_id)])
                if not candidates:
                    dropped["aucun_negatif_possible"] += 1
                    continue
                best = max(candidates, key=lambda j: scores[j])
                sim_neg = float(scores[best])
                verdict = "OK" if sim_neg <= sim_pos else "OK (neg > pos)"
                if sim_neg < MIN_NEGATIVE_SIMILARITY:
                    verdict = "rejet : negatif faible"
                    dropped["negatif_faible"] += 1
                elif sim_neg > MAX_NEGATIVE_SIMILARITY:
                    verdict = "rejet : negatif quasi identique"
                    dropped["negatif_trop_proche"] += 1
                else:
                    triplet = Triplet(question, positive.unit.statement, units[ids[best]].statement, f"{project_id}:llm_mining")
                    key = (triplet.anchor, triplet.positive, triplet.negative)
                    if key in seen:
                        verdict = "rejet : doublon"
                        dropped["doublon"] += 1
                    else:
                        seen.add(key)
                        triplets.append(triplet)
                        neg_above_pos += sim_neg > sim_pos
                print(
                    f"  {pos_id} | pos {sim_pos:.3f} neg {sim_neg:.3f} ({ids[best]}) | {verdict}\n"
                    f"      ancre   : {question}\n"
                    f"      negatif : {units[ids[best]].statement}"
                )

    print("\n" + "=" * 70 + "\nRESUME")
    print(f"TOTAL triplets retenus : {len(triplets)}")
    print(f"Unites positives traitees : {len(positives)}, dont avec au moins une ancre : {sum(1 for a in anchors.values() if a)}")
    print(f"Reponses Groq : {groq_calls} appels reussis, {cache_hits} lues dans le cache ({CACHE_PATH.name})")
    print(f"Unites couvertes (reponse Groq disponible) : {groq_calls + cache_hits}/{len(positives)}")
    print("Par projet :")
    for project_id, count in sorted(Counter(t.source.split(":")[0] for t in triplets).items()):
        print(f"  {project_id:<12} {count:>4}")
    print(f"Triplets 'neg > pos' (gardes) : {neg_above_pos}")
    print("Rejets par motif :")
    for code, count in sorted(dropped.items()):
        print(f"  {code:<32} {count:>4}")
    if args.sample and triplets:
        shown = random.Random(args.seed).sample(triplets, min(args.sample, len(triplets)))
        print(f"\n{len(shown)} triplets tires au hasard (seed {args.seed}) :")
        for n, t in enumerate(shown, 1):
            print(f"[{n}] {t.source}\n  anchor   : {t.anchor}\n  positive : {t.positive}\n  negative : {t.negative}")

    if args.write:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            "\n".join(json.dumps(t.as_record(), ensure_ascii=False) for t in triplets) + "\n", encoding="utf-8"
        )
        print(f"ecrit : {OUTPUT_PATH}")
    else:
        print("(sans --write : rien n'est ecrit)")


if __name__ == "__main__":
    main()

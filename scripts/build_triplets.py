from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schemas import SemanticUnit  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "processed" / "triplets_dev.jsonl"
QUERIES_PATH = ROOT / "data" / "eval" / "queries_dev.json"
DEV_PROJECTS = range(1, 11)
MODEL_NAME = "intfloat/multilingual-e5-small"
MIN_NEGATIVE_SIMILARITY = 0.75
MIN_MARGIN = 0.03
MAX_TASK_OVERLAP = 0.60
MIN_TASK_CONTENT_WORDS = 2
MAX_SUBJECT_WORDS = 10

# Relations dont l'autre extremite peut aussi repondre a l'ancre : jamais negatif.
FALSE_NEGATIVE_RELATIONS = {"supports", "answers", "creates_action"}

# Ancres redigees a la main pour les paires annotees (les autres passent par les templates).
SUPERSEDES_ANCHORS = {
    ("p01_u20", "p01_u12"): "Quelle est la date de mise en production d'Atlas à ce jour ?",
    ("p02_u11", "p02_u04"): "Où en est la signature du devis de l'agence Horizon ?",
    ("p04_u12", "p04_u03"): "La migration vers le nouveau CRM est-elle soumise à une condition ?",
    ("p05_u12", "p05_u04"): "À quelle date la journée clients a-t-elle finalement lieu ?",
    ("p05_u15", "p05_u05"): "Où se déroulera finalement la journée clients ?",
    ("p09_u11", "p09_u02"): "Sur quels téléphones peut-on utiliser Chantio aujourd'hui ?",
    ("p09_u12", "p09_u05"): "Combien d'utilisateurs terrain doit-on équiper ?",
    ("p09_u13", "p09_u06"): "Qui est concerné par le déploiement de Chantio ?",
    ("p10_u13", "p10_u04"): "Quel délai s'applique aujourd'hui avant la première relance ?",
}
CONTRADICTS_ANCHORS = {
    ("p02_u08", "p02_u01"): "Le budget de la campagne de printemps est-il validé ?",
    ("p10_u15", "p10_u14"): "Que dit le comptable d'Orvane à propos de la facture de mars ?",
    ("p10_u14", "p10_u15"): "Franck a-t-il vu un virement d'Orvane pour la facture de mars ?",
}

# Ancres (texte exact) dont le triplet est exclu a la main : ancre maladroite ou negatif hors-sujet.
REJECTED_ANCHORS: set[str] = {
    "Une formation des équipes support est-elle prévue avant le lancement ?",
    "Que sait-on à propos d'un changement de l'API du prestataire de paiement ?",
    "Qui doit s'occuper d'un premier brief créatif à l'équipe ?",
    "Quelle action est prévue pour un premier brief créatif à l'équipe ?",
    "Que sait-on à propos de la grille d'évaluation date de l'an dernier ?",
    "Quelle action est prévue pour ses remarques sur la grille d'évaluation ?",
    "Quelle action est prévue pour ses remarques à jeudi ?",
    "Que sait-on à propos du traiteur de l'an dernier ?",
    "Le paiement en plusieurs fois est-il conservé ?",
    "Que sait-on à propos d'aucune décision ?",
    "Que sait-on à propos des commerciaux risquent de mal réagir s'ils ?",
    "Faut-il mettre en place une séquence de relance automatique dans l'outil ?",
    "Que sait-on à propos du salon professionnel de Lyon ?",
}

PREDICATE = re.compile(
    r"\s+(?:n'|ne\s+)?(?:est|sont|sera|seront|serait|a|ont|avait|reste|restent|passe|passent|"
    r"doit|doivent|pourrait|pourraient|va|vont|vient|viennent|devient|fait|font|attend|attendent|"
    r"propose|estime|pense|craint|signale|confirme)\b"
)
DEADLINE = re.compile(
    r"\s+(?:d'ici|avant|au plus tard|dès|jusqu'|le lendemain|le jour même|le soir|"
    r"dans la journée|ce soir|pas avant)\b.*$"
)
BAD_SUBJECT_START = re.compile(r"(?:il|elle|ils|on|ce|cela|sur|dans|pour|selon|si|après|avant|d'après|depuis|lorsque|comme)", re.IGNORECASE)
TRAILING_PREPOSITIONS = {"à", "de", "d'", "pour", "auprès", "avec", "et", "par", "chez", "du", "des"}


@dataclass(frozen=True)
class Candidate:
    anchor: str
    positive: SemanticUnit
    negative: SemanticUnit
    source: str

    def as_record(self) -> dict[str, str]:
        return {
            "anchor": self.anchor,
            "positive": self.positive.statement,
            "negative": self.negative.statement,
            "source": self.source,
        }


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def strip_names(text: str, names: set[str]) -> str:
    if not names:
        return text
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")\b")
    return re.sub(r"\s+", " ", pattern.sub("", text)).strip()


def clean_tail(text: str) -> str:
    tokens = text.rstrip(" .?!,;").split()
    tokens = tokens[:MAX_SUBJECT_WORDS]
    while tokens and tokens[-1].lower() in TRAILING_PREPOSITIONS:
        tokens.pop()
    return " ".join(tokens)


def lower_first(text: str) -> str:
    first, _, rest = text.partition(" ")
    return (first if first.isupper() else first[:1].lower() + first[1:]) + (" " + rest if rest else "")


def with_de(subject: str) -> str:
    """Prefixe 'de' en contractant avec l'article (de le -> du, de les -> des, de un -> d'un)."""
    subject = lower_first(subject)
    if re.match(r"le\s", subject):
        return "du " + subject[3:]
    if re.match(r"les\s", subject):
        return "des " + subject[4:]
    if subject.startswith("l'"):
        return "de " + subject
    if re.match(r"[aeiouyhàâéèêîôû]", subject, re.IGNORECASE):
        return "d'" + subject
    return "de " + subject


def topic_subject(statement: str, names: set[str]) -> str | None:
    """Sujet d'un enonce factuel : la partie avant le predicat, sans noms de personnes."""
    text = strip_names(statement.rstrip(" ."), names)
    match = PREDICATE.search(text)
    if match is None:
        return None
    prefix = text[: match.start()]
    if "," in prefix or BAD_SUBJECT_START.match(prefix) or len(prefix.split()) < 2:
        return None
    return clean_tail(prefix) or None


def task_subject(statement: str, names: set[str]) -> str | None:
    """Tache d'une action / d'un engagement : l'objet apres le verbe, sans echeance ni nom."""
    text = statement.rstrip(" .")
    first = text.split()[0]
    if first in names:
        text = text[len(first):].strip()
        text = re.sub(r"^(?:doit|doivent|s'engage à)\s+", "", text)
        text = text.split(" ", 1)[1] if " " in text else ""  # retire le verbe
        text = re.sub(r"^(?:à|de|d')\s*", "", text)
    else:
        match = PREDICATE.search(text)
        if match is None:
            return None
        text = text[: match.start()]
    text = DEADLINE.sub("", text)
    return clean_tail(strip_names(text, names)) or None


def anchors_for(unit: SemanticUnit, names: set[str]) -> list[str]:
    if unit.type == "decision":
        subject = topic_subject(unit.statement, names)
        if subject is None:
            return []
        return [
            f"Quelle décision a été prise au sujet {with_de(subject)} ?",
            f"Qu'a-t-on finalement décidé concernant {lower_first(subject)} ?",
        ]
    if unit.type in {"action", "commitment"}:
        task = task_subject(unit.statement, names)
        if task is None:
            return []
        return [
            f"Qui doit s'occuper {with_de(task)} ?",
            f"Quelle action est prévue pour {lower_first(task)} ?",
        ]
    if unit.type in {"fact", "risk", "idea"}:
        subject = topic_subject(unit.statement, names)
        if subject is None:
            return []
        return [f"Que sait-on à propos {with_de(subject)} ?"]
    return []


def awkward_task(unit: SemanticUnit, names: set[str]) -> bool:
    """Tache trop courte ou qui recopie trop le statement : l'ancre serait maladroite."""
    task = task_subject(unit.statement, names)
    if task is None:
        return False
    task_words = words(task)
    content = [w for w in task_words if len(w) >= 4]
    statement_words = words(unit.statement)
    overlap = sum(1 for w in statement_words if w in set(task_words))
    return len(content) < MIN_TASK_CONTENT_WORDS or overlap / len(statement_words) > MAX_TASK_OVERLAP


def is_copy(anchor: str, unit: SemanticUnit) -> bool:
    anchor_words, statement_words = set(words(anchor)), words(unit.statement)
    overlap = sum(1 for w in statement_words if w in anchor_words)
    return unit.statement.lower().rstrip(" .") in anchor.lower() or overlap / len(statement_words) >= 0.75


def load_dev() -> tuple[set[str], set[str]]:
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    ids = {unit_id for q in queries for unit_id in q["relevant_unit_ids"]}
    texts = {re.sub(r"\W+", " ", q["query"].lower()).strip() for q in queries}
    return ids, texts


def load_projects() -> list[dict[str, object]]:
    projects = []
    for path in sorted(RAW_DIR.glob("projet_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(str(data["project_id"]).split("_")[1]) in DEV_PROJECTS:
            projects.append(data)
    return projects


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--sample", type=int, default=8)
    parser.add_argument("--all", action="store_true", help="liste numerotee complete, dans l'ordre")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    leaked_ids, leaked_queries = load_dev()
    dropped: Counter[str] = Counter()
    excluded_units: set[str] = set()
    triplets: list[Candidate] = []
    seen: set[tuple[str, str, str]] = set()

    def add(candidate: Candidate) -> None:
        key = (candidate.anchor, candidate.positive.statement, candidate.negative.statement)
        if candidate.anchor in REJECTED_ANCHORS:
            dropped["rejet_manuel"] += 1
        elif key in seen:
            dropped["doublon"] += 1
        elif re.sub(r"\W+", " ", candidate.anchor.lower()).strip() in leaked_queries:
            dropped["ancre_identique_dev"] += 1
        else:
            seen.add(key)
            triplets.append(candidate)

    for data in load_projects():
        project_id = str(data["project_id"])
        units = {raw["unit_id"]: SemanticUnit.model_validate(raw) for raw in data["units"]}
        names = {n for m in data["meetings"] for n in m["participants"]} | {
            u.speaker for u in units.values() if u.speaker
        }
        relations = data["relations"]
        related: dict[str, set[str]] = {}
        for rel in relations:
            if rel["relation"] in FALSE_NEGATIVE_RELATIONS:
                related.setdefault(rel["from_id"], set()).add(rel["to_id"])
                related.setdefault(rel["to_id"], set()).add(rel["from_id"])
        answers = {r["to_id"]: r["from_id"] for r in relations if r["relation"] == "answers"}

        # --- ancres par templates : (ancre, positive, unite d'origine) ---
        pending: list[tuple[str, SemanticUnit, SemanticUnit]] = []
        for unit in units.values():
            if unit.status != "active" or unit.type == "answer":
                continue
            if unit.unit_id in leaked_ids:
                excluded_units.add(unit.unit_id)
                continue
            if unit.type == "question":
                answer = units.get(answers.get(unit.unit_id, ""))
                if answer is None or answer.status != "active":
                    dropped["question_sans_reponse"] += 1
                elif answer.unit_id in leaked_ids:
                    excluded_units.add(answer.unit_id)
                else:
                    pending.append((unit.statement, answer, unit))
                continue
            if unit.type in {"action", "commitment"} and awkward_task(unit, names):
                dropped["ancre_maladroite"] += 1
                continue
            produced = anchors_for(unit, names)
            if not produced:
                dropped["sujet_non_extrait"] += 1
            for anchor in produced:
                if is_copy(anchor, unit):
                    dropped["ancre_recopiee"] += 1
                else:
                    pending.append((anchor, unit, unit))

        # --- mining du hard negative (meme projet) ---
        ids = list(units)
        passages = model.encode([f"passage: {units[i].statement}" for i in ids], normalize_embeddings=True)
        queries = model.encode([f"query: {a}" for a, _, _ in pending], normalize_embeddings=True) if pending else []
        for row, (anchor, positive, origin) in enumerate(pending):
            banned = {positive.unit_id, origin.unit_id} | leaked_ids
            banned |= {i for i, u in units.items() if u.type == "question"}
            banned |= related.get(positive.unit_id, set()) | related.get(origin.unit_id, set())
            scores = np.asarray(passages) @ np.asarray(queries[row])
            best = max(
                (j for j, i in enumerate(ids) if i not in banned and units[i].statement != positive.statement),
                key=lambda j: scores[j],
                default=None,
            )
            sim_pos = scores[ids.index(positive.unit_id)]
            if best is None or scores[best] < MIN_NEGATIVE_SIMILARITY:
                dropped["negatif_faible"] += 1
            elif sim_pos - scores[best] < MIN_MARGIN:
                dropped["ecart_insuffisant"] += 1
            else:
                add(Candidate(anchor, positive, units[ids[best]], f"{project_id}:template_mining"))

        # --- triplets annotes ---
        for rel in relations:
            if rel["relation"] == "supersedes":
                pairs = [(units[rel["from_id"]], units[rel["to_id"]], SUPERSEDES_ANCHORS)]
            elif rel["relation"] == "contradicts":
                pair = (units[rel["from_id"]], units[rel["to_id"]])
                pairs = [(*pair, CONTRADICTS_ANCHORS), (*pair[::-1], CONTRADICTS_ANCHORS)]
            else:
                continue
            for positive, negative, table in pairs:
                if positive.status != "active":
                    continue
                if positive.unit_id in leaked_ids or negative.unit_id in leaked_ids:
                    excluded_units.add(positive.unit_id if positive.unit_id in leaked_ids else negative.unit_id)
                    dropped["annote_fuite"] += 1
                    continue
                anchor = table.get((positive.unit_id, negative.unit_id))
                if anchor is None:
                    produced = anchors_for(positive, names)
                    anchor = produced[0] if produced else None
                if anchor is None or is_copy(anchor, positive):
                    dropped["annote_sans_ancre"] += 1
                    continue
                add(Candidate(anchor, positive, negative, f"{project_id}:annotated"))

    method = Counter(t.source.split(":")[1] for t in triplets)
    project = Counter(t.source.split(":")[0] for t in triplets)
    print(f"TOTAL triplets : {len(triplets)}\n")
    print("Par methode :")
    for k, v in sorted(method.items()):
        print(f"  {k:<18} {v}")
    print("\nPar projet :")
    for k in sorted(project):
        sub = Counter(t.source.split(":")[1] for t in triplets if t.source.startswith(k + ":"))
        print(f"  {k:<10} {project[k]:>3}  (mining {sub['template_mining']}, annotated {sub['annotated']})")
    print(f"\nUnites exclues pour fuite (reponses de queries_dev rencontrees, projets 01-10) : {len(excluded_units)}")
    print(f"  reponses dev totales : {len(leaked_ids)}")
    print("Rejets :", dict(dropped))
    if args.all:
        shown = triplets
        print("\nListe complete :")
    else:
        shown = random.Random(args.seed).sample(triplets, min(args.sample, len(triplets)))
        print(f"\n{args.sample} triplets au hasard (seed {args.seed}) :")
    for n, t in enumerate(shown, 1):
        print(f"[{n}] {t.source}\n  anchor   : {t.anchor}\n  positive : {t.positive.statement}\n  negative : {t.negative.statement}")

    if args.write:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            "\n".join(json.dumps(t.as_record(), ensure_ascii=False) for t in triplets) + "\n", encoding="utf-8"
        )
        print(f"ecrit : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

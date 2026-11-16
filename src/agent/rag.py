"""RAG controle : synthese par LLM a partir des seules unites retrouvees, verifiee par le code.

Apres la recherche hybride, l'abstention et la resolution (answer(), inchange), un appel Groq (gpt-oss-120b, JSON,
temperature 0) recoit la question et les k unites en vigueur retrouvees, avec leurs identifiants. Il renvoie :
answerable (oui/non), la reponse en phrases, chacune avec les identifiants qui la justifient, et les identifiants
utilises. Le code verifie ensuite :
- chaque identifiant cite existe parmi les unites fournies ;
- chaque phrase cite au moins un identifiant, et ne contient qu'une phrase ;
sinon la synthese est rejetee et l'agent garde la reponse extractive. Si answerable = non, l'agent s'abstient.
Sans reseau ni cle (ou si Groq echoue), l'agent fonctionne exactement comme sans RAG : reponse extractive.
Les reponses de Groq sont mises en cache (data/processed/groq_rag_cache.jsonl, versionne pour rejouer hors ligne).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from src.agent.answer import NON_ANSWER_TYPES, Answer, ReplacementGraph, answer, get_graph, get_retriever
from src.agent.retriever import ROOT, HybridRetriever
from src.ingest.groq_client import CachedGroq, GroqError, MissingKeyError
from src.schemas import SemanticUnit

RAG_CACHE = ROOT / "data" / "processed" / "groq_rag_cache.jsonl"
RAG_MODEL = "openai/gpt-oss-120b"
RAG_K = 5  # unites fournies au LLM
RAG_MAX_COMPLETION_TOKENS = 1500
RAG_EXTRA = {"reasoning_effort": "low"}
SECONDS_BETWEEN_CALLS = 15.0  # environ 2 000 tokens par appel : sous 8 000 tokens par minute
RAG_ABSTENTION_MESSAGE = "Les sources trouvées ne contiennent pas cette information"
SYNTHESIS_MENTION = "générée à partir des sources citées"

RAG_SYSTEM_PROMPT = """Tu réponds à une question à partir d'extraits de comptes rendus de réunion, et de rien d'autre.
Chaque extrait est une unité identifiée par un identifiant entre crochets, par exemple [p01_u20].
Règles :
- N'utilise que les informations des unités fournies. N'ajoute aucune connaissance extérieure, aucune supposition.
- Si les unités ne permettent pas de répondre précisément à la question, réponds answerable = false et une liste de phrases vide.
- Sinon, réponds en français, en une à trois phrases courtes. Chaque phrase est un élément de la liste "sentences",
  avec les identifiants des unités qui la justifient (au moins un). Une seule phrase par élément.
- N'écris jamais d'identifiant dans le texte des phrases.
Réponds uniquement par un objet JSON de la forme :
{"answerable": true, "sentences": [{"text": "...", "unit_ids": ["p01_u20"]}], "used_unit_ids": ["p01_u20"]}"""

_SENTENCE_END = re.compile(r"[.!?…](?:\s+)(?=[A-ZÀ-Ý])")
_ID_IN_TEXT = re.compile(r"\b[pP]\d{2}_(?:m\d{2}_x\d{2}|u\d{2})\b")


@dataclass(frozen=True)
class Sentence:
    text: str
    unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class RagResult:
    base: Answer  # reponse extractive (answer()), toujours calculee et affichee comme preuve
    status: str  # synthese | abstention_llm | rejetee | indisponible | non_appelee | conflit
    sentences: tuple[Sentence, ...] = ()
    provided_ids: tuple[str, ...] = ()
    cited_ids: tuple[str, ...] = ()
    rejection: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.base.abstained or self.status == "abstention_llm"

    @property
    def synthesis(self) -> str:
        return " ".join(s.text for s in self.sentences)


@lru_cache(maxsize=2)
def get_rag_client(cache_only: bool = False) -> CachedGroq:
    return CachedGroq(
        model=RAG_MODEL,
        temperature=0.0,
        cache_path=RAG_CACHE,
        min_seconds_between_calls=SECONDS_BETWEEN_CALLS,
        json_mode=True,
        cache_only=cache_only,
        max_completion_tokens=RAG_MAX_COMPLETION_TOKENS,
        extra_body=RAG_EXTRA,
    )


def context_units(query: str, project_id: str, base: Answer, retriever: HybridRetriever, graph: ReplacementGraph) -> list[SemanticUnit]:
    """Les RAG_K premieres unites de la recherche, remplacees par leur version en vigueur, sans doublon."""
    units: list[SemanticUnit] = []
    if base.unit_id:  # l'unite que l'agent afficherait passe en premier
        units.append(retriever.unit(base.unit_id))
    for hit in retriever.search(query, project_id, k=None):
        if len(units) >= RAG_K:
            break
        if hit.unit.type in NON_ANSWER_TYPES:
            continue
        resolved, _ = graph.resolve(hit.unit, retriever)
        if resolved.status == "active" and resolved.unit_id not in {u.unit_id for u in units}:
            units.append(resolved)
    return units


def build_user_prompt(query: str, units: list[SemanticUnit]) -> str:
    lines = [f"Question : {query.strip()}", "", "Unités (en vigueur) :"]
    for u in units:
        lines.append(f"[{u.unit_id}] ({u.type}, réunion du {u.source_date:%d/%m/%Y}, {u.speaker or 'locuteur inconnu'}) {u.statement}")
    return "\n".join(lines)


def check_synthesis(raw: str, provided: list[str]) -> tuple[bool | None, tuple[Sentence, ...], str | None]:
    """(answerable, phrases, motif de rejet). Toute anomalie rejette la synthese (motif non nul)."""
    try:
        data = json.loads(raw)
    except ValueError:
        return None, (), "JSON illisible"
    if not isinstance(data, dict) or not isinstance(data.get("answerable"), bool):
        return None, (), "champ answerable absent ou invalide"
    if not data["answerable"]:
        return False, (), None
    sentences = []
    for item in data.get("sentences") or []:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            return True, (), "phrase vide ou mal formee"
        ids = item.get("unit_ids") or []
        if not isinstance(ids, list) or not ids:
            return True, (), "phrase sans citation"
        unknown = [i for i in ids if i not in provided]
        if unknown:
            return True, (), f"identifiant cite absent des unites fournies : {unknown}"
        text = item["text"].strip()
        if _SENTENCE_END.search(text):
            return True, (), "plusieurs phrases dans un meme element"
        if _ID_IN_TEXT.search(text):
            return True, (), "identifiant ecrit dans le texte"
        sentences.append(Sentence(text, tuple(dict.fromkeys(ids))))
    if not sentences:
        return True, (), "answerable sans aucune phrase"
    used = data.get("used_unit_ids") or []
    if not isinstance(used, list) or any(i not in provided for i in used):
        return True, (), "used_unit_ids contient un identifiant absent des unites fournies"
    return True, tuple(sentences), None


def rag_answer(
    query: str,
    project_id: str,
    retriever: HybridRetriever | None = None,
    graph: ReplacementGraph | None = None,
    client: CachedGroq | None = None,
) -> RagResult:
    retriever = retriever or get_retriever()
    graph = graph or get_graph()
    base = answer(query, project_id, retriever=retriever, graph=graph)
    if base.abstained:  # la recherche s'abstient deja : aucun appel
        return RagResult(base, "non_appelee")
    if base.conflict_with:  # conflit non tranche : on n'appelle pas le LLM, qui pourrait conclure
        return RagResult(base, "conflit")
    units = context_units(query, project_id, base, retriever, graph)
    provided = [u.unit_id for u in units]
    client = client or get_rag_client()
    before = client.snapshot()
    try:
        raw = client.complete(RAG_SYSTEM_PROMPT, build_user_prompt(query, units), label=f"rag {project_id}")
    except (GroqError, MissingKeyError, OSError) as exc:  # pas de reseau, pas de cle, quota : reponse extractive
        return RagResult(base, "indisponible", provided_ids=tuple(provided), rejection=type(exc).__name__)
    after = client.snapshot()
    usage = {k: after[k] - before[k] for k in after}
    answerable, sentences, rejection = check_synthesis(raw, provided)
    if rejection:
        return RagResult(base, "rejetee", provided_ids=tuple(provided), rejection=rejection, usage=usage)
    if answerable is False:
        return RagResult(base, "abstention_llm", provided_ids=tuple(provided), usage=usage)
    cited = tuple(dict.fromkeys(i for s in sentences for i in s.unit_ids))
    return RagResult(base, "synthese", sentences, tuple(provided), cited, usage=usage)

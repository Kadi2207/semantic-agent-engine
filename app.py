"""Interface de demonstration Streamlit de l'agent (aucun backend separe : appelle src/agent directement).

Lancement : streamlit run app.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agent.answer import ABSTENTION_THRESHOLD, DATA_DIR, Answer, answer, get_graph, get_retriever  # noqa: E402
from src.agent.dossier import Dossier, Item, build_dossier, named_project, to_markdown  # noqa: E402
from src.agent.email import Clarification, draft_email, get_participants  # noqa: E402
from src.agent.display import (  # noqa: E402
    deadline_note,
    french_number,
    intent_label,
    resolution_sentences,
    short_date,
    status_label,
    type_label,
)
from src.agent.intent import detect_intent  # noqa: E402
from src.agent.rag import RAG_ABSTENTION_MESSAGE, RAG_DEFAULT, SYNTHESIS_MENTION, RagResult, rag_answer  # noqa: E402
from src.agent.retriever import FINETUNED_PATH, HybridRetriever  # noqa: E402
from src.agent.sources import load_transcripts, source_passage  # noqa: E402
from src.memory.corpus import load_relations  # noqa: E402
from src.memory.extracted import EXTRACTED_DIR, EXTRACTED_PROJECTS, MENTION, get_extracted_graph, get_extracted_retriever  # noqa: E402
from src.schemas import EmailDraft, SemanticUnit  # noqa: E402

st.set_page_config(page_title="Embedding Lab", layout="centered")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 820px; }
      .evidence { border-left: 3px solid #C8901A; padding: .35rem .9rem; margin: .6rem 0;
                  font-style: italic; color: inherit; opacity: .9; }
      h1 { color: #1E3A5F; }
      .subtitle { font-size: 1.05rem; color: #1E3A5F; margin: -.6rem 0 1rem 0; }
      [data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stDecoration"] { display: none; }
      .meta { font-size: .85rem; opacity: .75; }
      .email-body { white-space: pre-wrap; font-family: inherit; line-height: 1.5;
                    border: 1px solid rgba(128,128,128,.35); border-radius: 6px; padding: .8rem 1rem; }
      .src { border: 1px solid rgba(128,128,128,.3); border-radius: 6px; padding: .5rem .8rem; margin: .4rem 0;
             font-size: .92rem; line-height: 1.45; }
      .src .ctx { opacity: .6; }
      .src mark { background: rgba(200, 144, 26, .30); color: inherit; padding: 0 .15rem; border-radius: 3px; }
      .old { text-decoration: line-through; opacity: .75; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- chargement unique


@st.cache_resource(show_spinner="Chargement du modèle fine-tuné et indexation du corpus…")
def load_agent() -> HybridRetriever:
    """Construit une seule fois retriever hybride, graphe de relations et participants."""
    get_graph()
    get_participants()
    load_transcripts()
    return get_retriever()


@st.cache_resource(show_spinner="Chargement de la mémoire extraite…")
def load_extracted_memory():  # noqa: ANN201
    """Meme agent, autre memoire : unites et relations produites par l'extraction (projets 04 a 12)."""
    return get_extracted_retriever(), get_extracted_graph()


@st.cache_data
def load_project_names() -> dict[str, str]:
    names = {}
    for path in sorted(DATA_DIR.glob("projet_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        names[data["project_id"]] = data["project_name"]
    return names


# --------------------------------------------------------------------------- scenarios

# Parcours de demonstration en 5 temps, sur le projet Atlas (projet_01, 3 reunions) :
# la mise en production est proposee puis validee au 14 avril, puis reportee au 5 mai.
DEMO_PROJECT = "projet_01"
# Libelles "(1) ..." : un libelle "1. ..." serait rendu par Streamlit comme une liste numerotee.
DEMO_STEPS = [
    ("(1) Décision en vigueur", "Quelle est la date de mise en production d'Atlas ?", False),
    ("(2) Action : qui et quand", "Qui doit annoncer le report aux utilisateurs, et quand ?", False),
    ("(3) Décision remplacée", "La mise en production d'Atlas est-elle toujours prévue le 14 avril ?", False),
    ("(4) Passages sources", "Quelle est la date de mise en production d'Atlas ?", True),
    ("(5) Hors sources", "Quel est le salaire du nouveau directeur technique ?", False),
    ("(6) Email", "Écris un mail à la personne qui doit annoncer le report aux utilisateurs", False),
    ("(7) Dossier", "Constitue le dossier sur la mise en production d'Atlas", False),
]
OTHER_EXAMPLES = [
    ("Piège négation", "projet_02", "Le budget de la campagne de printemps est-il approuvé ?"),
    ("Demande d'email", "projet_02", "Écris un mail à la personne qui doit signer le devis d'Horizon"),
    ("Demande ambiguë", "projet_03", "Écris un mail à la personne qui s'occupe de la grille d'évaluation"),
]


MEMORY_ANNOTATED = "annotée (référence)"
MEMORY_EXTRACTED = "extraite automatiquement"
EXTRACTED_IDS = [f"projet_{n:02d}" for n in EXTRACTED_PROJECTS]


def apply_scenario(project_id: str, question: str, detail_sources: bool = False) -> None:
    st.session_state["memory"] = MEMORY_ANNOTATED  # le parcours et les exemples restent sur la memoire annotee
    st.session_state["project"] = project_id
    st.session_state["question"] = question
    st.session_state["detail_sources"] = detail_sources


# --------------------------------------------------------------------------- rendu


def technical_details(result: Answer) -> None:
    """Volet repliable : chemin brut de resolution et ligne de confiance (cosinus, score RRF, modele)."""
    with st.expander("Détails techniques"):
        if result.resolution:
            path = "".join(f"<div><code>{html.escape(step)}</code></div>" for step in result.resolution)
            st.markdown(f"<div class='meta'>Chemin de résolution :{path}</div>", unsafe_allow_html=True)
        lines = [
            f"Meilleur cosinus dense : {french_number(result.best_dense_score)} "
            f"(seuil d'abstention : {french_number(ABSTENTION_THRESHOLD, 2)})"
        ]
        if result.fused_score is not None:
            lines.append(f"Score RRF : {french_number(result.fused_score, 4)}")
        lines.append(f"Recherche hybride BM25 et e5 fine-tuné (<code>{FINETUNED_PATH.relative_to(ROOT).as_posix()}</code>)")
        st.markdown(f"<div class='meta'>{'<br>'.join(lines)}</div>", unsafe_allow_html=True)


def unit_meta(unit: SemanticUnit) -> str:
    """Ligne de source d'une unite : type, reunion, locuteur, identifiant et statut, en francais."""
    return (
        f"<div class='meta'>Type : <b>{type_label(unit)}</b>, réunion <b>{unit.source_id}</b> du "
        f"{short_date(unit.source_date)}, locuteur : {unit.speaker or 'inconnu'}, unité "
        f"<code>{unit.unit_id}</code> (statut <b>{status_label(unit.status)}</b>)</div>"
    )


def render_source(unit: SemanticUnit, context: int, title: str) -> None:
    """Prise de parole contenant l'evidence (surlignee), avec le contexte immediat du transcript."""
    passage = source_passage(unit, context=context)
    if not passage.found:
        st.caption(f"{title} : passage introuvable dans le transcript de {unit.source_id}.")
        return
    lines = []
    for line in passage.lines:
        text = html.escape(line.text)
        if line.is_evidence:
            evidence = html.escape(unit.evidence)
            text = text.replace(evidence, f"<mark>{evidence}</mark>", 1) if evidence in text else f"<mark>{text}</mark>"
            lines.append(f"<div><b>{html.escape(line.speaker)}</b> : {text}</div>")
        else:
            lines.append(f"<div class='ctx'>{html.escape(line.speaker)} : {text}</div>")
    st.markdown(
        f"<div class='meta'>{title}, transcript de la réunion <b>{passage.source_id}</b> du {short_date(unit.source_date)}</div>"
        f"<div class='src'>{''.join(lines)}</div>",
        unsafe_allow_html=True,
    )


def render_conflict(result: Answer, retriever: HybridRetriever, detail_sources: bool = False) -> None:
    """Conflit non tranche : les affirmations contradictoires, toutes en vigueur, a egalite et sans conclusion."""
    st.warning(
        "**Conflit non tranché** : les sources contiennent des affirmations contradictoires, toutes encore en "
        "vigueur. L'agent ne choisit pas entre elles.",
    )
    for number, unit_id in enumerate((result.unit_id, *result.conflict_with), start=1):
        unit = retriever.unit(unit_id)
        with st.container(border=True):
            st.markdown(f"##### Affirmation {number}")
            st.markdown(f"**{html.escape(unit.statement)}**", unsafe_allow_html=True)
            st.markdown(f"<div class='evidence'>« {html.escape(unit.evidence)} »</div>", unsafe_allow_html=True)
            st.markdown(unit_meta(unit), unsafe_allow_html=True)
            render_source(unit, context=2 if detail_sources else 1, title="Passage source")
    technical_details(result)


def render_answer(result: Answer, retriever: HybridRetriever, detail_sources: bool = False) -> None:
    if result.abstained:
        st.warning(f"**{result.message}**")
        st.caption("L'agent préfère ne pas répondre plutôt que d'inventer.")
        technical_details(result)
        return

    if result.conflict_with:
        render_conflict(result, retriever, detail_sources)
        return

    with st.container(border=True):
        st.markdown("##### Réponse")
        st.markdown(f"**{result.statement}**")
        st.markdown(f"<div class='evidence'>« {html.escape(result.evidence or '')} »</div>", unsafe_allow_html=True)
        answer_unit = retriever.unit(result.unit_id)
        note = deadline_note(answer_unit)
        if note:
            st.markdown(f"<div class='meta'>Échéance : <b>{note}</b></div>", unsafe_allow_html=True)
        st.markdown(unit_meta(answer_unit), unsafe_allow_html=True)
        render_source(answer_unit, context=2 if detail_sources else 1, title="Passage source")

    if result.resolution:
        contradicted = any("-contradicts->" in step for step in result.resolution)
        title = "Information contredite" if contradicted else "Information mise à jour"
        sentences = " ".join(resolution_sentences(result.resolution, retriever))
        st.info(f"**{title}** : {sentences}")

    if result.history:
        with st.container(border=True):
            st.markdown("##### Historique : ce que cette information remplace")
            for older_id, relation, newer_id in result.history:
                older = retriever.unit(older_id)
                if relation == "supersedes" and older.status == "superseded":
                    status = f"statut : <b>remplacée</b> par <code>{newer_id}</code>"
                else:
                    verb = "contredite par" if relation == "contradicts" else "remplacée par"
                    status = f"statut : <b>{status_label(older.status)}</b>, {verb} <code>{newer_id}</code>"
                st.markdown(
                    f"<span class='old'>{html.escape(older.statement)}</span> "
                    f"<span class='meta'>({type_label(older)}, {older.source_id} du {short_date(older.source_date)}, "
                    f"{status})</span>",
                    unsafe_allow_html=True,
                )
                if detail_sources:
                    render_source(older, context=1, title=f"Source de {older_id}")
    technical_details(result)


def render_rag(result: RagResult, retriever: HybridRetriever, detail_sources: bool = False) -> None:
    """Synthese controlee au-dessus de la preuve extractive ; repli extractif si elle est absente ou rejetee."""
    if result.status == "abstention_llm":
        st.warning(f"**{RAG_ABSTENTION_MESSAGE}**")
        st.caption("Le modèle de langage a lu les unités retrouvées et n'y trouve pas la réponse : l'agent s'abstient.")
        technical_details(result.base)
        with st.expander("Unités lues par le modèle de langage"):
            for unit_id in result.provided_ids:
                unit = retriever.unit(unit_id)
                st.markdown(f"- {html.escape(unit.statement)} <span class='meta'>(<code>{unit_id}</code>)</span>", unsafe_allow_html=True)
        return
    if result.status == "synthese":
        with st.container(border=True):
            st.markdown("##### Synthèse")
            for sentence in result.sentences:
                refs = ", ".join(f"<code>{u}</code>" for u in sentence.unit_ids)
                st.markdown(f"{html.escape(sentence.text)} <span class='meta'>({refs})</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='meta'>Synthèse {SYNTHESIS_MENTION} ; chaque phrase renvoie aux unités "
                        "qui la justifient. La preuve exacte suit.</div>", unsafe_allow_html=True)
    elif result.status == "rejetee":
        st.markdown(f"<div class='meta'>Synthèse rejetée par les contrôles ({html.escape(result.rejection or '')}) : "
                    "réponse extractive.</div>", unsafe_allow_html=True)
    render_answer(result.base, retriever, detail_sources)


def render_draft(draft: EmailDraft) -> None:
    recipient = draft.recipient
    with st.container(border=True):
        st.markdown("##### Brouillon à confirmer, non envoyé")
        col_name, col_email = st.columns(2)
        col_name.text_input("Destinataire", recipient.name, disabled=True)
        col_email.text_input(
            "Adresse",
            "",
            placeholder="à renseigner (jamais inventée)",
            disabled=True,
            help="Adresse absente des sources : elle n'est jamais inventée.",
        )
        st.text_input("Objet", draft.subject, disabled=True)
        body_html = html.escape(draft.body).replace("\n", "<br>")
        st.markdown(f"<div class='email-body'>{body_html}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='meta' style='margin-top:.7rem'>Références : "
            f"{', '.join(f'<code>{u}</code>' for u in draft.source_unit_ids)}, responsable identifié dans "
            f"{', '.join(f'<code>{u}</code>' for u in recipient.evidence_unit_ids)}<br>"
            f"Confirmation requise avant tout envoi : <b>{'oui' if draft.needs_confirmation else 'non'}</b> "
            f"(<code>needs_confirmation = {draft.needs_confirmation}</code>)<br>"
            f"Email envoyé : <b>{'oui' if draft.executed else 'non'}</b> (<code>executed = {draft.executed}</code>), "
            f"l'outil d'envoi <code>{draft.tool_candidate}</code> n'est jamais appelé<br>"
            f"Adresse du destinataire : <b>{'à renseigner' if recipient.email is None else recipient.email}</b></div>",
            unsafe_allow_html=True,
        )
    st.caption("Rien n'est envoyé : relis le brouillon et renseigne l'adresse avant tout envoi.")


def render_dossier(dossier: Dossier, retriever: HybridRetriever) -> None:
    """Dossier documente : chaque element est une unite de la memoire, avec sa reunion, sa date et sa preuve."""
    if dossier.abstained:
        st.warning("**Information absente ou incertaine dans les sources** : aucun dossier n'est constitué.")
        return

    def ref(unit: SemanticUnit) -> str:
        return f"<span class='meta'>(<code>{unit.unit_id}</code>, réunion {unit.source_id} du {short_date(unit.source_date)})</span>"

    def items(entries: list[Item], empty: str) -> None:
        if not entries:
            st.markdown(f"<div class='meta'>{empty}</div>", unsafe_allow_html=True)
        for item in entries:
            detail = f" <span class='meta'>{html.escape(item.detail)}</span>" if item.detail else ""
            st.markdown(f"- {html.escape(item.unit.statement)}{detail} {ref(item.unit)}", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"##### Dossier : projet {dossier.project_name}")
        scope = "toutes les unités du projet" if dossier.scope == "projet" else (
            "unités retrouvées par la recherche, puis reliées par les relations du graphe")
        st.markdown(f"<div class='meta'>Objet : {html.escape(dossier.query.strip())}<br>Périmètre : {scope}, "
                    f"{len(dossier.units)} unités citées.</div>", unsafe_allow_html=True)
        st.markdown("**Décisions en vigueur**")
        items(dossier.decisions, "Aucune décision en vigueur dans ce périmètre.")
        st.markdown("**Historique des changements**")
        if not dossier.history:
            st.markdown("<div class='meta'>Aucun remplacement ni contradiction résolue dans ce périmètre.</div>", unsafe_allow_html=True)
        for item in dossier.history:
            st.markdown(f"- {html.escape(item.detail)} <span class='old'>{html.escape(item.unit.statement)}</span> {ref(item.unit)}",
                        unsafe_allow_html=True)
        st.markdown("**Conflits non tranchés**")
        if not dossier.conflicts:
            st.markdown("<div class='meta'>Aucun conflit non tranché dans ce périmètre.</div>", unsafe_allow_html=True)
        for a, b in dossier.conflicts:
            st.markdown("<div class='meta'>Deux affirmations contradictoires, toutes deux en vigueur ; les sources ne "
                        "permettent pas de trancher :</div>", unsafe_allow_html=True)
            items([a, b], "")
        st.markdown("**Actions et engagements**")
        items(dossier.tasks, "Aucune action ni engagement dans ce périmètre.")
        st.markdown("**Risques**")
        items(dossier.risks, "Aucun risque dans ce périmètre.")
        st.markdown("**Questions ouvertes**")
        items(dossier.open_questions, "Aucune question restée sans réponse dans ce périmètre.")
        with st.expander(f"Sources citées ({len(dossier.units)} passages de transcript)"):
            for unit in dossier.units:
                render_source(unit, context=0, title=f"Source de {unit.unit_id}")
    st.download_button(
        "Télécharger le dossier (Markdown)",
        to_markdown(dossier),
        file_name=f"dossier_{dossier.project_id}.md",
        mime="text/markdown",
    )


def render_clarification(result: Clarification) -> None:
    labels = {"ambiguous": "Demande ambiguë", "no_person": "Responsable introuvable", "off_topic": "Sujet introuvable"}
    st.warning(f"**Clarification nécessaire : {labels.get(result.reason, result.reason).lower()}**\n\n{result.question}")
    if result.candidates:
        st.markdown("**Candidats trouvés dans les sources**")
        for c in result.candidates:
            st.markdown(
                f"- **{c.person or 'aucun responsable désigné'}** : {c.unit.statement} "
                f"<span class='meta'>(<code>{c.unit.unit_id}</code>, {c.unit.source_id} du "
                f"{short_date(c.unit.source_date)})</span>",
                unsafe_allow_html=True,
            )
    st.caption("Aucun brouillon n'est généré tant que le destinataire n'est pas clair.")


# --------------------------------------------------------------------------- page

project_names = load_project_names()
project_ids = list(project_names)
st.session_state.setdefault("memory", MEMORY_ANNOTATED)
st.session_state.setdefault("rag", RAG_DEFAULT)
st.session_state.setdefault("project", DEMO_PROJECT)
st.session_state.setdefault("question", "")
st.session_state.setdefault("detail_sources", False)

extracted_mode = st.session_state["memory"] == MEMORY_EXTRACTED
if extracted_mode:  # seuls les projets 04 a 12 ont une memoire extraite
    project_ids = [pid for pid in project_ids if pid in EXTRACTED_IDS]
    if st.session_state["project"] not in project_ids:
        st.session_state["project"] = project_ids[0]

with st.sidebar:
    st.markdown("### Mémoire")
    st.radio("Mémoire", [MEMORY_ANNOTATED, MEMORY_EXTRACTED], key="memory", label_visibility="collapsed")
    if extracted_mode:
        st.caption(f"Mémoire extraite : {MENTION}. Projets 04 à 12 seulement.")
    st.toggle("Synthèse contrôlée (RAG)", key="rag", help="Synthèse par un modèle de langage à partir des seules "
              "unités retrouvées ; chaque phrase doit citer ses sources, sinon la réponse extractive est gardée.")
    st.markdown("### Projet")
    st.radio(
        "Projet",
        project_ids,
        key="project",
        format_func=lambda pid: f"{pid[-2:]} {project_names[pid]}",
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        "L'agent répond en citant sa preuve, distingue une décision en vigueur d'une décision remplacée "
        "ou contredite, s'abstient quand le meilleur passage trouvé est trop éloigné de la question, et "
        "prépare des brouillons d'email sans jamais les envoyer, et constitue des dossiers documentés à partir des sources."
    )

try:
    retriever = load_agent()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

graph = get_graph()
if extracted_mode:
    retriever, graph = load_extracted_memory()

project_id = st.session_state["project"]
st.title("Embedding Lab")
st.markdown("<div class='subtitle'>Mémoire de réunions : décisions, actions et sources</div>", unsafe_allow_html=True)
st.caption(f"Projet actif : **{project_names[project_id]}** (`{project_id}`)")

if extracted_mode:
    st.info(f"**Mémoire extraite automatiquement** : {MENTION}.")

st.markdown(
    f"<div class='meta'>Parcours de démonstration : projet {project_names[DEMO_PROJECT]} "
    f"(<code>{DEMO_PROJECT}</code>, 3 réunions)</div>",
    unsafe_allow_html=True,
)
for row in (DEMO_STEPS[:4], DEMO_STEPS[4:]):  # deux rangees : 7 boutons ne tiennent pas sur une ligne
    for col, (label, question, detail) in zip(st.columns(4), row):
        i = DEMO_STEPS.index((label, question, detail))
        col.button(label, key=f"demo_{i}", on_click=apply_scenario, args=(DEMO_PROJECT, question, detail), use_container_width=True)
with st.expander("Autres exemples (négation, email, clarification)"):
    for i, (col, (label, pid, question)) in enumerate(zip(st.columns(len(OTHER_EXAMPLES)), OTHER_EXAMPLES)):
        col.button(label, key=f"other_{i}", on_click=apply_scenario, args=(pid, question), use_container_width=True)

with st.form("ask", border=False):
    question = st.text_input("Question", key="question", placeholder="Pose une question, demande un email ou un dossier…")
    submitted = st.form_submit_button("Envoyer", type="primary")

if submitted:
    if not question.strip():
        st.info("Saisis une question.")
    else:
        intent = detect_intent(question)
        st.markdown(f"<div class='meta'>Intention détectée : <b>{intent_label(intent)}</b></div>", unsafe_allow_html=True)
        with st.spinner("Recherche dans les sources…"):
            if intent == "dossier":
                visible = {pid: project_names[pid] for pid in project_ids}
                target = named_project(question, visible) or project_id
                relations = load_relations(EXTRACTED_DIR if extracted_mode else DATA_DIR)
                result = build_dossier(question, target, project_names[target], retriever=retriever, relations=relations)
            elif intent == "draft_email":
                result = draft_email(question, project_id, retriever=retriever, graph=graph)
            elif st.session_state.get("rag"):
                result = rag_answer(question, project_id, retriever=retriever, graph=graph)
            else:
                result = answer(question, project_id, retriever=retriever, graph=graph)
        if isinstance(result, Clarification):
            render_clarification(result)
        elif isinstance(result, EmailDraft):
            render_draft(result)
        elif isinstance(result, Dossier):
            render_dossier(result, retriever)
        elif isinstance(result, RagResult):
            render_rag(result, retriever, detail_sources=st.session_state.get("detail_sources", False))
        else:
            render_answer(result, retriever, detail_sources=st.session_state.get("detail_sources", False))

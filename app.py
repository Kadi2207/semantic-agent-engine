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
from src.agent.email import Clarification, draft_email, get_participants  # noqa: E402
from src.agent.display import deadline_note, type_label  # noqa: E402
from src.agent.intent import detect_intent  # noqa: E402
from src.agent.retriever import FINETUNED_PATH, HybridRetriever  # noqa: E402
from src.agent.sources import load_transcripts, source_passage  # noqa: E402
from src.schemas import EmailDraft, SemanticUnit  # noqa: E402

st.set_page_config(page_title="Agent sémantique", page_icon="🔎", layout="centered")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 820px; }
      .evidence { border-left: 3px solid #8a8f98; padding: .35rem .9rem; margin: .6rem 0;
                  font-style: italic; color: inherit; opacity: .9; }
      .meta { font-size: .85rem; opacity: .75; }
      .email-body { white-space: pre-wrap; font-family: inherit; line-height: 1.5;
                    border: 1px solid rgba(128,128,128,.35); border-radius: 6px; padding: .8rem 1rem; }
      .src { border: 1px solid rgba(128,128,128,.3); border-radius: 6px; padding: .5rem .8rem; margin: .4rem 0;
             font-size: .92rem; line-height: 1.45; }
      .src .ctx { opacity: .6; }
      .src mark { background: rgba(255, 210, 0, .35); color: inherit; padding: 0 .15rem; border-radius: 3px; }
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
DEMO_STEPS = [
    ("1 · Décision en vigueur", "Quelle est la date de mise en production d'Atlas ?", False),
    ("2 · Action : qui et quand", "Qui doit annoncer le report aux utilisateurs, et quand ?", False),
    ("3 · Décision remplacée", "La mise en production d'Atlas est-elle toujours prévue le 14 avril ?", False),
    ("4 · Passages sources", "Quelle est la date de mise en production d'Atlas ?", True),
    ("5 · Hors sources", "Quel est le salaire du nouveau directeur technique ?", False),
]
OTHER_EXAMPLES = [
    ("Piège négation", "projet_02", "Le budget de la campagne de printemps est-il approuvé ?"),
    ("Demande d'email", "projet_02", "Écris un mail à la personne qui doit signer le devis d'Horizon"),
    ("Demande ambiguë", "projet_03", "Écris un mail à la personne qui s'occupe de la grille d'évaluation"),
]


def apply_scenario(project_id: str, question: str, detail_sources: bool = False) -> None:
    st.session_state["project"] = project_id
    st.session_state["question"] = question
    st.session_state["detail_sources"] = detail_sources


# --------------------------------------------------------------------------- rendu


def confidence_caption(score: float, extra: str = "") -> None:
    st.markdown(
        f"<div class='meta'>Confiance : meilleur cosinus dense = {score:.3f} "
        f"(seuil d'abstention {ABSTENTION_THRESHOLD:.2f}){extra} · recherche hybride BM25 + "
        f"e5 fine-tuné (<code>{FINETUNED_PATH.relative_to(ROOT).as_posix()}</code>)</div>",
        unsafe_allow_html=True,
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
        f"<div class='meta'>{title} — transcript de la réunion <b>{passage.source_id}</b> du {unit.source_date:%d/%m/%Y}</div>"
        f"<div class='src'>{''.join(lines)}</div>",
        unsafe_allow_html=True,
    )


def render_conflict(result: Answer, retriever: HybridRetriever, detail_sources: bool = False) -> None:
    """Conflit non tranche : les affirmations contradictoires, toutes en vigueur, a egalite et sans conclusion."""
    st.warning(
        "**Conflit non tranché** : les sources contiennent des affirmations contradictoires, toutes encore en "
        "vigueur. L'agent ne choisit pas entre elles.",
        icon="⚖️",
    )
    for number, unit_id in enumerate((result.unit_id, *result.conflict_with), start=1):
        unit = retriever.unit(unit_id)
        with st.container(border=True):
            st.markdown(f"##### Affirmation {number}")
            st.markdown(f"**{html.escape(unit.statement)}**", unsafe_allow_html=True)
            st.markdown(f"<div class='evidence'>« {html.escape(unit.evidence)} »</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='meta'>Type : <b>{type_label(unit)}</b> · réunion <b>{unit.source_id}</b> du "
                f"{unit.source_date:%d/%m/%Y} · locuteur : {unit.speaker or 'inconnu'} · unité "
                f"<code>{unit.unit_id}</code> (statut <b>{unit.status}</b>)</div>",
                unsafe_allow_html=True,
            )
            render_source(unit, context=2 if detail_sources else 1, title="Passage source")
    rrf = f" · score RRF {result.fused_score:.4f}" if result.fused_score is not None else ""
    confidence_caption(result.best_dense_score, rrf)


def render_answer(result: Answer, retriever: HybridRetriever, detail_sources: bool = False) -> None:
    if result.abstained:
        st.warning(f"**{result.message}**", icon="🤷")
        st.caption("L'agent préfère ne pas répondre plutôt que d'inventer.")
        confidence_caption(result.best_dense_score)
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
        st.markdown(
            f"<div class='meta'>Type : <b>{type_label(answer_unit)}</b> · réunion <b>{result.source_id}</b> du "
            f"{result.source_date:%d/%m/%Y} · locuteur : {result.speaker or 'inconnu'} · unité "
            f"<code>{result.unit_id}</code> (statut <b>{result.unit_status}</b>)</div>",
            unsafe_allow_html=True,
        )
        render_source(answer_unit, context=2 if detail_sources else 1, title="Passage source")

    if result.resolution:
        contradicted = any("-contradicts->" in step for step in result.resolution)
        title = "Information contredite" if contradicted else "Information mise à jour"
        path = "\n".join(f"- `{step}`" for step in result.resolution)
        st.info(
            f"**{title}** : l'unité la mieux classée était périmée ; "
            f"l'agent a suivi les relations jusqu'à l'information en vigueur.\n\n{path}",
            icon="🔁",
        )

    if result.history:
        with st.container(border=True):
            st.markdown("##### Historique : ce que cette information remplace")
            for older_id, relation, newer_id in result.history:
                older = retriever.unit(older_id)
                verb = "contredite par" if relation == "contradicts" else "remplacée par"
                st.markdown(
                    f"<span class='old'>{html.escape(older.statement)}</span> "
                    f"<span class='meta'>— {type_label(older)}, {older.source_id} du {older.source_date:%d/%m/%Y}, "
                    f"statut <b>{older.status}</b>, "
                    f"{verb} <code>{newer_id}</code></span>",
                    unsafe_allow_html=True,
                )
                if detail_sources:
                    render_source(older, context=1, title=f"Source de {older_id}")
    rrf = f" · score RRF {result.fused_score:.4f}" if result.fused_score is not None else ""
    confidence_caption(result.best_dense_score, rrf)


def render_draft(draft: EmailDraft) -> None:
    recipient = draft.recipient
    with st.container(border=True):
        st.markdown("##### ✉️ BROUILLON — À CONFIRMER, NON ENVOYÉ")
        col_name, col_email = st.columns(2)
        col_name.text_input("Destinataire", recipient.name, disabled=True)
        col_email.text_input(
            "Adresse",
            "",
            placeholder="à renseigner (jamais inventée)",
            disabled=True,
            help=f"Statut de résolution : {recipient.resolution_status}",
        )
        st.text_input("Objet", draft.subject, disabled=True)
        body_html = html.escape(draft.body).replace("\n", "<br>")
        st.markdown(f"<div class='email-body'>{body_html}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='meta' style='margin-top:.7rem'>Références : "
            f"{', '.join(f'<code>{u}</code>' for u in draft.source_unit_ids)} · responsable identifié dans "
            f"{', '.join(f'<code>{u}</code>' for u in recipient.evidence_unit_ids)}<br>"
            f"Outil prévu : <code>{draft.tool_candidate}</code> · needs_confirmation = <b>{draft.needs_confirmation}</b>"
            f" · executed = <b>{draft.executed}</b> · statut destinataire : <b>{recipient.resolution_status}</b></div>",
            unsafe_allow_html=True,
        )
    st.caption("Rien n'est envoyé : relis le brouillon et renseigne l'adresse avant tout envoi.")


def render_clarification(result: Clarification) -> None:
    labels = {"ambiguous": "Demande ambiguë", "no_person": "Responsable introuvable", "off_topic": "Sujet introuvable"}
    st.warning(f"**Clarification nécessaire — {labels.get(result.reason, result.reason)}**\n\n{result.question}", icon="❓")
    if result.candidates:
        st.markdown("**Candidats trouvés dans les sources**")
        for c in result.candidates:
            st.markdown(
                f"- **{c.person or 'aucun responsable désigné'}** : {c.unit.statement} "
                f"<span class='meta'>(<code>{c.unit.unit_id}</code>, {c.unit.source_id} du "
                f"{c.unit.source_date:%d/%m/%Y})</span>",
                unsafe_allow_html=True,
            )
    st.caption("Aucun brouillon n'est généré tant que le destinataire n'est pas clair.")


# --------------------------------------------------------------------------- page

project_names = load_project_names()
project_ids = list(project_names)
st.session_state.setdefault("project", DEMO_PROJECT)
st.session_state.setdefault("question", "")
st.session_state.setdefault("detail_sources", False)

with st.sidebar:
    st.markdown("### Conversation")
    st.radio(
        "Projet",
        project_ids,
        key="project",
        format_func=lambda pid: f"{pid[-2:]} · {project_names[pid]}",
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        "L'agent répond en citant sa preuve, suit les décisions mises à jour, "
        "s'abstient quand les sources ne couvrent pas la question, et prépare des "
        "brouillons d'email sans jamais les envoyer."
    )

try:
    retriever = load_agent()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

project_id = st.session_state["project"]
st.title("Agent sémantique")
st.caption(f"Projet actif : **{project_names[project_id]}** (`{project_id}`)")

st.markdown(
    f"<div class='meta'>Parcours de démonstration — projet {project_names[DEMO_PROJECT]} "
    f"(<code>{DEMO_PROJECT}</code>, 3 réunions)</div>",
    unsafe_allow_html=True,
)
for i, (col, (label, question, detail)) in enumerate(zip(st.columns(len(DEMO_STEPS)), DEMO_STEPS)):
    col.button(label, key=f"demo_{i}", on_click=apply_scenario, args=(DEMO_PROJECT, question, detail), use_container_width=True)
with st.expander("Autres exemples (négation, email, clarification)"):
    for col, (label, pid, question) in zip(st.columns(len(OTHER_EXAMPLES)), OTHER_EXAMPLES):
        col.button(label, on_click=apply_scenario, args=(pid, question), use_container_width=True)

with st.form("ask", border=False):
    question = st.text_input("Question", key="question", placeholder="Pose une question ou demande un email…")
    submitted = st.form_submit_button("Envoyer", type="primary")

if submitted:
    if not question.strip():
        st.info("Saisis une question.")
    else:
        intent = detect_intent(question)
        st.markdown(f"<div class='meta'>Intention détectée : <code>{intent}</code></div>", unsafe_allow_html=True)
        with st.spinner("Recherche dans les sources…"):
            if intent == "draft_email":
                result = draft_email(question, project_id)
            else:
                result = answer(question, project_id, retriever=retriever)
        if isinstance(result, Clarification):
            render_clarification(result)
        elif isinstance(result, EmailDraft):
            render_draft(result)
        else:
            render_answer(result, retriever, detail_sources=st.session_state.get("detail_sources", False))

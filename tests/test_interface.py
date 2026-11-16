"""Finition de l'interface : tout en francais, phrase de mise a jour lisible, details techniques replies.

Les tests de libelles n'ont pas besoin du modele ; le rendu reel de app.py (Streamlit AppTest) en a besoin.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from src.agent.display import INTENT_LABELS, STATUS_LABELS, french_number, intent_label, resolution_sentences, status_label
from src.memory.corpus import load_corpus

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "e5-finetuned" / "config.json"
UNITS = {c.unit.unit_id: c.unit for c in load_corpus(ROOT / "data" / "raw")}
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="modele fine-tune absent (clone frais) : app.py ne peut pas tourner")


class Lookup:
    @staticmethod
    def unit(unit_id: str):  # noqa: ANN205
        return UNITS[unit_id]


def test_statuses_and_intents_are_shown_in_french() -> None:
    assert STATUS_LABELS == {"active": "en vigueur", "superseded": "remplacée", "cancelled": "annulée"}
    assert {u.status for u in UNITS.values()} <= set(STATUS_LABELS)
    assert INTENT_LABELS == {"answer": "Répondre", "draft_email": "Rédiger un email"}
    assert (status_label("active"), intent_label("draft_email")) == ("en vigueur", "Rédiger un email")


def test_numbers_use_the_french_decimal_comma() -> None:
    assert french_number(0.09) == "0,090"
    assert french_number(0.45, 2) == "0,45"
    assert french_number(0.032787, 4) == "0,0328"


def test_update_banner_is_a_readable_sentence() -> None:
    atlas = ("p01_u01 (superseded) -supersedes-> p01_u12 (superseded)", "p01_u12 (superseded) -supersedes-> p01_u20 (active)")
    assert resolution_sentences(atlas, Lookup) == [
        "La proposition du 03/03/2026 (14 avril) a été remplacée par la décision du 31/03/2026 (14 avril).",
        "La décision du 31/03/2026 (14 avril) a été remplacée par celle du 09/04/2026 (5 mai).",
    ]
    budget = ("p02_u01 (superseded) -contradicts-> p02_u08 (active)",)
    assert resolution_sentences(budget, Lookup) == ["La décision du 04/02/2026 a été contredite par celle du 18/02/2026."]


def test_no_middle_dot_or_long_dash_in_the_interface_code() -> None:
    for name in ("app.py", "src/agent/display.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        for char in "·—–":
            assert char not in source, (name, char)


def test_streamlit_config_hides_the_toolbar_and_sets_the_theme() -> None:
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["client"]["toolbarMode"] == "viewer"
    assert (config["theme"]["primaryColor"], config["theme"]["backgroundColor"].upper()) == ("#1E3A5F", "#FFFFFF")
    assert "#C8901A" in (ROOT / "app.py").read_text(encoding="utf-8")  # accent : preuve et surlignage


def run_scenario(key: str):  # noqa: ANN201
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600).run()
    at.button(key=key).click().run()
    next(b for b in at.button if b.label == "Envoyer").click().run()
    assert not at.exception
    return at


@needs_model
def test_atlas_replaced_decision_screen() -> None:
    at = run_scenario("demo_2")
    assert at.title[0].value == "Embedding Lab"
    page = "\n".join(m.value for m in at.markdown)
    assert "Mémoire de réunions : décisions, actions et sources" in page
    assert "Intention détectée : <b>Répondre</b>" in page and "<code>answer</code>" not in page
    assert "(statut <b>en vigueur</b>)" in page and "<b>active</b>" not in page and "<b>superseded</b>" not in page
    assert "statut : <b>remplacée</b> par <code>p01_u20</code>" in page
    banner = "\n".join(i.value for i in at.info)
    assert "La décision du 31/03/2026 (14 avril) a été remplacée par celle du 09/04/2026 (5 mai)." in banner
    assert "p01_u12" not in banner  # le chemin brut n'est plus dans le bandeau
    details = at.expander[-1]
    assert details.label == "Détails techniques"
    inside = "\n".join(m.value for m in details.markdown)
    assert "p01_u12 (superseded) -supersedes-&gt; p01_u20 (active)" in inside
    assert "Meilleur cosinus dense : 0," in inside and "(seuil d'abstention : 0,45)" in inside and "Score RRF : 0," in inside
    assert at.sidebar.markdown[0].value == "### Projet"
    assert "s'abstient quand le meilleur passage trouvé est trop éloigné de la question" in at.sidebar.caption[0].value
    for element in [*at.markdown, *at.info, *at.caption, *at.sidebar.caption]:
        assert not set("·—–") & set(element.value), element.value
    assert not any(set("·—–") & set(b.label) for b in at.button)


@needs_model
def test_off_sources_screen_shows_the_cosine_with_a_comma_in_the_details() -> None:
    at = run_scenario("demo_4")
    assert "Information absente ou incertaine dans les sources" in at.warning[0].value
    inside = "\n".join(m.value for m in at.expander[-1].markdown)
    assert "Meilleur cosinus dense : 0,090 (seuil d'abstention : 0,45)" in inside


@needs_model
def test_email_example_shows_recipient_draft_confirmation_and_is_never_sent() -> None:
    at = run_scenario("other_1")
    page = "\n".join(m.value for m in at.markdown)
    assert "Intention détectée : <b>Rédiger un email</b>" in page
    assert "Brouillon à confirmer, non envoyé" in page
    fields = {t.label: t for t in at.text_input}
    assert fields["Destinataire"].value == "Karim"
    assert fields["Adresse"].value == "" and "jamais inventée" in fields["Adresse"].placeholder
    assert fields["Objet"].value.startswith("Suivi : signer le devis")
    assert "Bonjour Karim," in page and "mise en attente" in page  # corps du brouillon et point d'attention
    assert "Confirmation requise avant tout envoi : <b>oui</b> (<code>needs_confirmation = True</code>)" in page
    assert "Email envoyé : <b>non</b> (<code>executed = False</code>)" in page
    assert "Adresse du destinataire : <b>à renseigner</b>" in page
    assert any("Rien n'est envoyé" in c.value for c in at.caption)

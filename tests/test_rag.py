"""RAG controle : controles des citations, repli extractif sans reseau, resultats enregistres.

Aucun appel reseau : conftest.py force RAG_CACHE_ONLY=1 ; les reponses du LLM viennent du cache versionne.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.rag import RAG_ABSTENTION_MESSAGE, check_synthesis

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "e5-finetuned" / "config.json"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="modele fine-tune absent (clone frais)")


def synthesis(sentences: list[tuple[str, list[str]]], answerable: bool = True, used: list[str] | None = None) -> str:
    items = [{"text": t, "unit_ids": ids} for t, ids in sentences]
    return json.dumps({"answerable": answerable, "sentences": items, "used_unit_ids": used if used is not None else sorted({i for _, ids in sentences for i in ids})})


def test_valid_synthesis_is_accepted() -> None:
    answerable, sentences, rejection = check_synthesis(synthesis([("La mise en production est reportée au 5 mai.", ["p01_u20"])]), ["p01_u20", "p01_u12"])
    assert (answerable, rejection) == (True, None) and sentences[0].unit_ids == ("p01_u20",)


@pytest.mark.parametrize(
    "raw, reason",
    [
        ("pas du json", "JSON illisible"),
        (json.dumps({"sentences": []}), "champ answerable absent ou invalide"),
        (synthesis([("Phrase sans source.", [])]), "phrase sans citation"),
        (synthesis([("Phrase inventée.", ["p01_u99"])]), "identifiant cite absent des unites fournies : ['p01_u99']"),
        (synthesis([("Première phrase. Deuxième phrase.", ["p01_u20"])]), "plusieurs phrases dans un meme element"),
        (synthesis([("Selon p01_u20, c'est le 5 mai.", ["p01_u20"])]), "identifiant ecrit dans le texte"),
        (synthesis([]), "answerable sans aucune phrase"),
        (synthesis([("Le 5 mai.", ["p01_u20"])], used=["p01_u99"]), "used_unit_ids contient un identifiant absent des unites fournies"),
    ],
)
def test_invalid_syntheses_are_rejected(raw: str, reason: str) -> None:
    assert check_synthesis(raw, ["p01_u20"])[2] == reason


def test_not_answerable_is_an_abstention() -> None:
    assert check_synthesis(synthesis([], answerable=False), ["p01_u20"]) == (False, (), None)
    assert RAG_ABSTENTION_MESSAGE == "Les sources trouvées ne contiennent pas cette information"


@needs_model
def test_without_network_or_cache_the_agent_is_exactly_extractive(tmp_path: Path) -> None:
    from src.agent.answer import answer
    from src.agent.rag import RAG_EXTRA, RAG_MAX_COMPLETION_TOKENS, RAG_MODEL, rag_answer
    from src.ingest.groq_client import CachedGroq

    empty = CachedGroq(RAG_MODEL, 0.0, tmp_path / "vide.jsonl", 0.0, json_mode=True, cache_only=True,
                       max_completion_tokens=RAG_MAX_COMPLETION_TOKENS, extra_body=RAG_EXTRA)
    question = "La mise en production d'Atlas est-elle toujours prévue le 14 avril ?"
    result = rag_answer(question, "projet_01", client=empty)
    assert result.status == "indisponible" and result.sentences == ()
    assert result.base == answer(question, "projet_01")  # reponse extractive inchangee
    assert empty.calls == 0


@needs_model
def test_demo_questions_are_in_the_versioned_cache() -> None:
    from src.agent.rag import rag_answer

    first = rag_answer("Quelle est la date de mise en production d'Atlas ?", "projet_01")
    assert first.status == "synthese" and "p01_u20" in first.cited_ids
    assert set(first.cited_ids) <= set(first.provided_ids)
    assert rag_answer("Quel est le salaire du nouveau directeur technique ?", "projet_01").status == "non_appelee"
    assert rag_answer("La facture de mars d'Orvane est-elle réglée ?", "projet_10").status == "conflit"  # aucun appel


def test_recorded_dev_results_are_consistent() -> None:
    data = json.loads((ROOT / "eval/results/rag_dev.json").read_text(encoding="utf-8"))
    r = data["results"]
    rows = r["per_query"]
    assert len(rows) == 30 and r["answerable"]["n"] == 25 and r["sans_reponse"]["n"] == 5
    assert r["answerable"]["correct"] == sum(x["correct"] for x in rows if x["relevant_unit_ids"] and x["piege"] != "sans_reponse")
    assert r["sans_reponse"]["abstained"] == sum(x["abstained"] for x in rows if x["piege"] == "sans_reponse")
    for x in rows:
        assert set(x["cited_ids"]) <= set(x["provided_ids"]), x["query_id"]  # aucune citation hors des unites fournies
        if x["status"] == "synthese":
            assert x["cited_ids"] and x["synthesis"]
    assert r["citations_valid"] == r["citations"] and r["citations_valid_rate"] == 1.0
    assert set(data["criteria"]) == {"C1_answered_with_expected_unit", "C2_unanswerable_abstained", "C3_citations_valid", "C4_atlas_identical"}
    assert data["criteria"]["C1_answered_with_expected_unit"]["min"] == 23 and data["criteria"]["C2_unanswerable_abstained"]["min"] == 4
    assert data["llm"]["temperature"] == 0.0 and data["llm"]["model"] == "openai/gpt-oss-120b"


def test_rag_is_on_by_default_only_because_the_criteria_are_met() -> None:
    from src.agent.rag import RAG_DEFAULT

    met = json.loads((ROOT / "eval/results/rag_dev.json").read_text(encoding="utf-8"))["results"]["criteria_met"]
    assert RAG_DEFAULT is all(v for k, v in met.items() if k != "C4_atlas_identical")


@needs_model
def test_app_shows_the_synthesis_above_the_unchanged_extractive_proof() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600).run()
    assert at.sidebar.toggle[0].value is True
    at.button(key="demo_2").click().run()
    next(b for b in at.button if b.label == "Envoyer").click().run()
    assert not at.exception
    page = "\n".join(m.value for m in at.markdown)
    assert "##### Synthèse" in page and "générée à partir des sources citées" in page
    assert page.index("##### Synthèse") < page.index("##### Réponse")  # la preuve extractive suit la synthese
    assert "<mark>La mise en production d&#x27;Atlas est reportée au 5 mai.</mark>" in page
    assert "<code>p01_u20</code>" in page


def test_sealed_test_is_a_single_recording() -> None:
    data = json.loads((ROOT / "eval/results/rag_test.json").read_text(encoding="utf-8"))
    assert data["split"] == "test" and "execution unique d'enregistrement sur le test scelle" in data["status"]
    assert "abstention_probes" not in data["results"] and len(data["results"]["per_query"]) == 6
    for x in data["results"]["per_query"]:
        assert set(x["cited_ids"]) <= set(x["provided_ids"])

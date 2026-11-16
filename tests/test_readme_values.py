"""Le README ne cite que des chiffres presents dans eval/results (ou comptes dans les donnees versionnees).

Chaque ligne de tableau est reconstruite a partir des JSON puis cherchee telle quelle dans le README ; tout
nombre a trois decimales du README doit exister dans l'un des fichiers de resultats.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
RESULTS = ROOT / "eval" / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def fr(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def all_numbers(node: object) -> set[float]:
    if isinstance(node, bool):
        return set()
    if isinstance(node, (int, float)):
        return {float(node)}
    if isinstance(node, dict):
        return set().union(*(all_numbers(v) for v in node.values())) if node else set()
    if isinstance(node, list):
        return set().union(*(all_numbers(v) for v in node)) if node else set()
    return set()


def test_every_three_decimal_number_of_the_readme_exists_in_a_results_json() -> None:
    known = set()
    for name in ("retrieval_dev.json", "retrieval_test.json", "extraction.json", "real_qmsum.json"):
        known |= {fr(v) for v in all_numbers(load(name))}
    cited = set(re.findall(r"(?<![\d,])\d,\d{3}(?![\d])", README))
    assert cited, "aucun chiffre trouve dans le README"
    assert cited <= known, f"chiffres du README absents des JSON : {sorted(cited - known)}"


def test_retrieval_tables_are_built_from_the_json() -> None:
    dev = load("retrieval_dev.json")["experiments"]
    test = load("retrieval_test.json")["experiments"]["compare_finetuned"]
    rows = [
        ("BM25", dev["run_baselines"]["results"]["bm25"]["global"]),
        ("e5-small de base", dev["compare_finetuned"]["results"]["e5_base"]["global"]),
        ("e5-small fine-tuné", dev["compare_finetuned"]["results"]["e5_finetuned"]["global"]),
        ("e5-small de base", test["results"]["e5_base"]["global"]),
        ("e5-small fine-tuné", test["results"]["e5_finetuned"]["global"]),
    ]
    for label, g in rows:
        assert f"| {label} | {fr(g['Recall@3'])} | {fr(g['Recall@5'])} | {fr(g['MRR'])} | {fr(g['nDCG@5'])} |" in README
    q_dev, q_test = dev["compare_finetuned"]["queries"], test["queries"]
    assert f"{q_dev['evaluated']} requêtes évaluées sur {q_dev['total']}" in README
    assert f"{q_test['evaluated']} requêtes évaluées sur {q_test['total']}" in README


def test_extraction_tables_and_figures_are_built_from_the_json() -> None:
    data = load("extraction.json")
    dev, test, glob = (data["groups"][g] for g in ("dev_04_10", "test_11_12", "global_04_12"))
    ud, ut = dev["units"], test["units"]
    rd, rt = dev["relations"], test["relations"]
    expected_rows = [
        f"| Unités annotées, extraites, appariées | {ud['annotated']}, {ud['extracted']}, {ud['matched']} | {ut['annotated']}, {ut['extracted']}, {ut['matched']} |",
        f"| Précision des unités | {fr(ud['precision'])} | {fr(ut['precision'])} |",
        f"| Rappel des unités | {fr(ud['recall'])} | {fr(ut['recall'])} |",
        f"| F1 des unités | {fr(ud['f1'])} | {fr(ut['f1'])} |",
        f"| Type exact | {ud['type_correct']}/{ud['accuracy_denominator']} ({fr(ud['type_accuracy'])}) | {ut['type_correct']}/{ut['accuracy_denominator']} ({fr(ut['type_accuracy'])}) |",
        f"| Statut exact | {ud['status_correct']}/{ud['accuracy_denominator']} ({fr(ud['status_accuracy'])}) | {ut['status_correct']}/{ut['accuracy_denominator']} ({fr(ut['status_accuracy'])}) |",
        f"| Relations annotées, extraites | {rd['annotated']}, {rd['extracted']} | {rt['annotated']}, {rt['extracted']} |",
        f"| Relations strictes, précision | {fr(rd['strict']['precision'])} | {fr(rt['strict']['precision'])} |",
        f"| Relations strictes, rappel | {fr(rd['strict']['recall'])} | {fr(rt['strict']['recall'])} |",
        f"| Relations strictes, F1 | {fr(rd['strict']['f1'])} | {fr(rt['strict']['f1'])} |",
    ]
    for row in expected_rows:
        assert row in README, row
    for rel, xd in rd["per_type"].items():
        xt = rt["per_type"][rel]
        test_cell = f"{xt['annotated']} | {fr(xt['f1'])}" if xt["annotated"] else "0 | sans objet"
        assert f"| {rel} | {xd['annotated']} | {fr(xd['f1'])} | {test_cell} |" in README, rel
    weakest = min(rd["per_type"], key=lambda r: rd["per_type"][r]["f1"] if rd["per_type"][r]["annotated"] > 1 else 9)
    assert weakest == "creates_action" and "La relation creates_action est la plus faible" in README
    # chiffres cites dans le texte
    global_f1 = f"{glob['relations']['strict']['f1']:.2f}".replace(".", ",")  # cite a deux decimales : 0,40
    assert f"le F1 strict des relations vaut {global_f1}" in README
    cost = glob["cost"]
    assert f"{cost['groq_calls']} appels Groq, pour {thousands(cost['prompt_tokens'])} tokens en entrée et {thousands(cost['completion_tokens'])} en sortie" in README
    idea = glob["units"]["per_type"]["idea"]
    assert f"{idea['extracted']} unités idea pour {idea['annotated']} annotées" in README
    conf = ud["type_confusions"]
    assert f"{conf['commitment->action']} engagements annotés sont extraits comme actions et {conf['action->commitment']} actions comme engagements" in README
    assert f"le rappel des actions n'est que de {fr(glob['units']['per_type']['action']['recall'])}" in README
    assert f"{ut['matched']} unités appariées pour l'extraction" in README


def test_qmsum_tables_are_built_from_the_json() -> None:
    data = load("real_qmsum.json")
    groups = {
        "Global hors Pays de Galles": "global hors Pays de Galles",
        "Product (AMI)": "Product",
        "Academic (ICSI)": "Academic",
        "Committee, Canada": "Committee/Canada",
        "Committee, Pays de Galles": "Committee/Pays de Galles",
    }
    systems = {"BM25": "bm25", "e5 de base": "e5_base", "e5 fine-tuné": "e5_finetuned", "Hybride RRF": "hybrid_rrf"}
    for label, group in groups.items():
        for sys_label, system in systems.items():
            m = data["results"][system][group]
            row = f"| {label} | {sys_label} | {m['n_queries']} | {fr(m['hit@5'])} | {fr(m['hit@10'])} | {fr(m['mrr'])} | {fr(m['ndcg@10'])} |"
            assert row in README, row
    assert data["reproducibility"]["rerun_identical"] is True
    assert "NON VERIFIEE" in data["licence_status"]["Committee/Pays de Galles"] and "licence de réutilisation n'a pas pu être vérifiée" in README
    assert f"plafonne en moyenne à {fr(data['results']['bm25']['Product']['recall@5_plafond'])} de Recall@5 sur Product" in README


def test_corpus_counts_come_from_the_versioned_data() -> None:
    projects = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/raw").glob("projet_*.json"))]
    meetings = sum(len(p["meetings"]) for p in projects)
    units = sum(len(p["units"]) for p in projects)
    relations = sum(len(p["relations"]) for p in projects)
    assert f"{len(projects)} projets, {meetings} réunions, {units} unités et {relations} relations" in README
    statement_project = {u["statement"]: p["project_id"] for p in projects for u in p["units"]}
    hand = [json.loads(line) for line in (ROOT / "data/processed/triplets_dev.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    llm = [json.loads(line) for line in (ROOT / "data/processed/triplets_llm.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    late = {"projet_09", "projet_10"}
    assert sum(statement_project.get(t["positive"]) in late for t in llm) == 0
    assert f"seuls {sum(statement_project.get(t['positive']) in late for t in hand)} triplets construits à la main en proviennent, aucun triplet enrichi" in README


def test_projects_meetings_triplets_and_loss_come_from_the_repository() -> None:
    """12 projets et 26 reunions (data/raw), comptes de triplets (data/processed), fonction de perte (code)."""
    projects = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/raw").glob("projet_*.json"))]
    assert (len(projects), sum(len(p["meetings"]) for p in projects)) == (12, 26)
    assert f"{len(projects)} projets, {sum(len(p['meetings']) for p in projects)} réunions" in README

    def count(name: str) -> list[dict]:
        text = (ROOT / "data/processed" / name).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    hand, llm = count("triplets_dev.jsonl"), count("triplets_llm.jsonl")
    assert f"{len(hand)} triplets construits à la main" in README
    assert f"{len(llm)} triplets enrichis" in README
    statement_project = {u["statement"]: p["project_id"] for p in projects for u in p["units"]}
    assert not any(statement_project.get(t["positive"]) in {"projet_11", "projet_12"} for t in hand + llm)
    assert "Aucun triplet ne provient des projets 11 et 12" in README

    source = (ROOT / "scripts/train_embeddings.py").read_text(encoding="utf-8")
    losses = set(re.findall(r"losses\.(\w+)\(", source))
    assert len(losses) == 1
    assert f"fonction de perte {losses.pop()}" in README


def test_reintegrated_limits_are_present_once() -> None:
    assert len(re.findall(r"^#+ .*[Ll]imites?\b", README, re.MULTILINE)) == 1  # une seule section de limites
    assert "il n'est plus vierge, et seul le test scellé reste indépendant du réglage" in README
    test_n = json.loads((RESULTS / "retrieval_test.json").read_text(encoding="utf-8"))["experiments"]["compare_finetuned"]["queries"]["evaluated"]
    assert f"ses résultats restent fragiles, car il ne comporte que {test_n} requêtes évaluées" in README
    assert "mesure non biaisée" not in README and "Test scellé minuscule" not in README
    assert README.count("triplets construits à la main (") == 1  # une seule puce pour les triplets faits a la main
    assert "non testée ici" in README  # l'explication de la faiblesse de BM25 reste une hypothese
    assert "premier participant nommé dans l'énoncé" in README and "pas généralisable telle quelle" in README


def test_relation_error_figures_come_from_the_analysis_json() -> None:
    data = load("relation_errors_dev.json")
    t = data["totals"]
    assert t["predicted_errors"]["mauvais_sens"] == 0 and "aucune relation prédite n'est fausse à cause de son seul sens" in README
    assert (
        f"Sur les {t['predicted']} relations prédites, {t['correct']} sont justes ; parmi les erreurs, "
        f"{t['predicted_errors']['extremite_non_annotee']} relient une unité absente de l'annotation et "
        f"{t['predicted_errors']['mauvaise_cible']} visent la mauvaise cible."
    ) in README
    assert (
        f"Sur les {t['annotated']} relations annotées, {t['missed_total']} sont manquées, dont "
        f"{t['missed']['jamais_proposee']} jamais proposées ({data['by_type']['supports']['missed']['jamais_proposee']} supports)."
    ) in README
    assert "`eval/results/relation_errors_dev.json`" in README and "uv run python eval/analyze_relation_errors.py" in README
    assert "non versionnée)" not in README.split("## Limites assumées")[1].split("\n- Abstention")[0]


def test_abstention_threshold_quoted_from_the_code() -> None:
    source = (ROOT / "src/agent/answer.py").read_text(encoding="utf-8")
    threshold = float(re.search(r"^ABSTENTION_THRESHOLD = ([0-9.]+)", source, re.MULTILINE).group(1))
    assert f"passe sous {threshold:.2f}".replace(".", ",") in README


def test_readme_typography() -> None:
    for char, name in (("—", "tiret long"), ("–", "tiret moyen"), ("·", "point median")):
        assert char not in README, f"{name} present dans le README"

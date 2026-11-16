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
    for name in ("retrieval_dev.json", "retrieval_test.json", "extraction.json", "real_qmsum.json", "agent_dev.json", "agent_test.json", "agent_extracted_dev.json", "rag_dev.json", "rag_test.json",
                 "architectures_dev.json", "architectures_test.json", "relations_v2.json", "test2.json", "stats.json", "rag_nli.json"):
        known |= {fr(abs(v)) for v in all_numbers(load(name))}  # le signe est lu a part (differences des tests apparies)
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
        ("Hybride RRF (agent)", dev["hybrid_rrf"]["results"]["hybrid_rrf"]["global"]),
        ("Hybride RRF (agent)", load("retrieval_test.json")["experiments"]["hybrid_rrf"]["results"]["hybrid_rrf"]["global"]),
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
    # « mesure non biaisee » n'est admis que dans la section relations v2, ou il a ete demande explicitement
    v2_title = "### Seconde passe de liaison des relations : résultat négatif"
    outside_v2 = README.split(v2_title)[0] + "\n### " + README.split(v2_title)[1].split("\n### ", 1)[1]
    assert "mesure non biaisée" not in outside_v2 and README.count("mesure non biaisée") == 1
    assert "Test scellé minuscule" not in README
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


def test_hybrid_and_final_answer_figures_come_from_the_json() -> None:
    dev_search = load("retrieval_dev.json")["experiments"]
    hybrid = dev_search["hybrid_rrf"]
    h, finetuned = hybrid["results"]["hybrid_rrf"], dev_search["compare_finetuned"]["results"]["e5_finetuned"]["global"]
    g = h["global"]
    assert f"RRF avec k = {hybrid['params']['rrf_k']}" in README
    assert f"le meilleur Recall@3 ({fr(g['Recall@3'])})" in README and g["Recall@3"] > finetuned["Recall@3"]
    assert f"(MRR de {fr(g['MRR'])} contre {fr(finetuned['MRR'])})" in README
    assert all(g[m] < finetuned[m] for m in ("Recall@5", "MRR", "nDCG@5"))  # "sous le e5 fine-tune sur les trois autres"
    assert f"Sur les décisions en vigueur, son MRR est de {fr(h['per_category']['decision_active']['MRR'])}" in README

    dev, test = load("agent_dev.json"), load("agent_test.json")
    final_dev, final_test = dev["experiments"]["final_answer"], test["experiments"]["final_answer"]
    gd, gt = final_dev["results"]["global"], final_test["results"]["global"]
    assert f"| Global | {gd['correct']}/{gd['n']} ({fr(gd['accuracy'])}) | {gt['correct']}/{gt['n']} ({fr(gt['accuracy'])}) |" in README
    for piege, bd in final_dev["results"]["per_category"].items():
        bt = final_test["results"]["per_category"][piege]
        assert f"| {piege} | {bd['correct']}/{bd['n']} | {bt['correct']}/{bt['n']} |" in README, piege
    resolved = [r for r in final_dev["results"]["per_query"] if r["resolved"]]
    assert all(r["correct"] for r in resolved)
    assert f"En dev, {len(resolved)} réponses passent par une résolution" in README
    assert f"et {gd['unresolved_conflict']} réponse est un conflit non tranché" in README
    categories = final_dev["results"]["per_category"]
    assert min(categories, key=lambda p: categories[p]["accuracy"]) == "reformulation"
    assert "La reformulation est la catégorie la plus faible" in README
    assert f"L'agent s'abstient à tort {gd['abstained']} fois sur ces {gd['n']} requêtes" in README
    sd, st = final_dev["sans_reponse"], final_test["sans_reponse"]
    assert f"l'agent s'abstient {sd['abstained']} fois sur {sd['n']} en dev et {st['abstained']} fois sur {st['n']} en test scellé" in README
    probes = dev["experiments"]["abstention_probes"]["results"]
    summary = dev["experiments"]["abstention_probes"]["summary"]
    assert {k: summary[k] for k in ("n", "abstained", "wrongly_answered")} == {k: probes[k] for k in ("n", "abstained", "wrongly_answered")}
    assert summary["acceptable_probe_ids"] == ["probe_02"] and "p06_u25" in summary["acceptable_notes"]["probe_02"]
    assert (
        f"{summary['n']} questions, {summary['abstained']} abstention ; {summary['wrongly_answered']} réponses à tort et "
        f"{summary['acceptable_on_substance']} acceptable sur le fond (`p06_u25` indique qu'aucune décision n'a été prise)"
    ) in README
    assert "sondes exploratoires, sur les projets de développement, jamais utilisées pour régler" in README
    same = sum("same_as_dev_query" in p for p in json.loads((ROOT / "data/eval/probes_abstention.json").read_text(encoding="utf-8"))["probes"])
    assert f"{same} des {summary['n']} sondes sont les requêtes sans_reponse du dev" in README
    assert "seule la question sur le budget publicitaire d'Atlas est nouvelle : ce n'est pas une mesure indépendante" in README
    breakdown = final_dev["resolution_breakdown"]
    assert (breakdown["resolved"], breakdown["resolved_correct"]) == (len(resolved), len(resolved))
    assert breakdown["unresolved_conflict_counted_correct"] == gd["unresolved_conflict"]
    assert breakdown["resolved_correct"] + breakdown["unresolved_conflict_counted_correct"] == gd["correct_thanks_to_resolution"]
    strict = breakdown["correct_if_unresolved_conflict_not_counted"]
    assert (
        f"{gd['correct']}/{gd['n']} ({strict}/{gd['n']} si le conflit non tranché de "
        f"{breakdown['unresolved_conflict_query_ids'][0]} n'est pas compté juste)"
    ) in README
    recorded_once = "execution unique d'enregistrement sur le test scelle"
    assert recorded_once in test["status"] and recorded_once in load("retrieval_test.json")["experiments"]["hybrid_rrf"]["status"]
    assert "d'une seule exécution d'enregistrement, sans aucun réglage" in README

    limits = README.split("## Limites assumées")[1].split("\n## ")[0]
    assert (
        f"{sd['abstained']} abstention sur les {sd['n']} requêtes sans réponse du dev, {st['abstained']} sur {st['n']} en test scellé ; "
        f"sur {summary['n']} sondes exploratoires, {summary['abstained']} abstention, {summary['wrongly_answered']} réponses à tort et "
        f"{summary['acceptable_on_substance']} acceptable sur le fond"
    ) in limits
    assert f"{same} sur {summary['n']} sont les requêtes sans_reponse du dev" in limits
    wrongly_abstained = [r for r in final_dev["results"]["per_query"] if r["abstained"]]
    assert [r["query_id"] for r in wrongly_abstained] == ["q_dev_002"]
    assert f"le seuil écarte à tort q_dev_002, dont la réponse existe (meilleur cosinus de {fr(wrongly_abstained[0]['best_dense_score'])})" in limits
    active = categories["decision_active"]
    assert active["correct"] == active["n"] and f"les {active['n']} requêtes de cette catégorie obtiennent une réponse juste" in limits
    ref = categories["reformulation"]
    assert (
        f"{gd['n'] - gd['correct']} réponses sur {gd['n']} sont fausses ({gd['n'] - strict} si le conflit non tranché n'est pas compté juste), "
        f"dont {ref['n'] - ref['correct']} sur {ref['n']} en reformulation"
    ) in limits
    assert f"la catégorie la plus faible en réponse finale, avec {ref['correct']}/{ref['n']} en dev. L'agent ne retient que le premier résultat" in limits
    assert (
        f"l'hybride est légèrement sous le e5 fine-tuné seul (MRR de {fr(g['MRR'])} contre {fr(finetuned['MRR'])}, et de même en nDCG@5 "
        f"et en Recall@5), bien qu'il soit meilleur en Recall@3 ({fr(g['Recall@3'])} contre {fr(finetuned['Recall@3'])})"
    ) in limits
    assert "Conflit non tranché : le corpus de développement n'en contient qu'un seul cas (projet_10)" in limits
    assert "conflit non tranché, avec les deux affirmations et leurs sources, sans en barrer aucune et sans conclure" in README
    assert "« proposition, non validée »" in README and "ajoute la date de la réunion sans rien calculer d'autre" in README


def test_extracted_chain_figures_come_from_the_json() -> None:
    data = load("agent_extracted_dev.json")
    r, mem = data["results"], data["extracted_memory"]
    ge, ga = r["extracted"]["global"], r["annotated"]["global"]
    assert f"| Global | {ge['correct']}/{ge['n']} ({fr(ge['accuracy'])}) | {ga['correct']}/{ga['n']} ({fr(ga['accuracy'])}) |" in README
    for piege, be in r["extracted"]["per_category"].items():
        ba = r["annotated"]["per_category"][piege]
        assert f"| {piege} | {be['correct']}/{be['n']} | {ba['correct']}/{ba['n']} |" in README, piege
    assert f"La mesure porte sur les {data['queries']['evaluated']} requêtes avec réponse du dev" in README
    assert mem["expected_units_with_extracted_match"] == mem["expected_units"]
    assert f"Les {mem['expected_units']} unités attendues ont toutes un équivalent dans la mémoire extraite" in README
    c = r["comparison"]
    assert (
        f"{c['correct_in_both']} requêtes sont justes dans les deux mémoires, {len(c['correct_only_annotated'])} seulement en "
        f"mémoire annotée et {len(c['correct_only_extracted'])} seulement en mémoire extraite"
    ) in README
    da, de = r["annotated"]["per_category"]["decision_active"], r["extracted"]["per_category"]["decision_active"]
    assert f"({de['correct']}/{de['n']} contre {da['correct']}/{da['n']})" in README
    glob_f1 = load("extraction.json")["groups"]["global_04_12"]["relations"]["strict"]["f1"]
    assert f"dont le F1 strict n'est que de {glob_f1:.2f}".replace(".", ",") + " sur les projets 04 à 12" in README
    assert f"L'agent s'abstient à tort {ge['abstained']} fois en mémoire extraite ({ga['abstained']} fois en mémoire annotée)" in README
    s = data["sans_reponse"]
    assert s["extracted"]["abstained"] == s["annotated"]["abstained"] == 0
    assert f"aucune des {s['extracted']['n']} requêtes sans réponse de ces projets" in README
    limits = README.split("## Limites assumées")[1].split("\n## ")[0]
    assert f"({ge['correct']}/{ge['n']} contre {ga['correct']}/{ga['n']} sur les mêmes requêtes)" in limits


def test_rag_figures_come_from_the_json() -> None:
    dev, test = load("rag_dev.json"), load("rag_test.json")
    rd, rt = dev["results"], test["results"]
    assert f"| Requêtes avec réponse, unité attendue citée | {rd['answerable']['correct']}/{rd['answerable']['n']} | {rt['answerable']['correct']}/{rt['answerable']['n']} |" in README
    assert f"| Requêtes sans réponse, abstention | {rd['sans_reponse']['abstained']}/{rd['sans_reponse']['n']} | {rt['sans_reponse']['abstained']}/{rt['sans_reponse']['n']} |" in README
    assert f"| Citations valides | {rd['citations_valid']}/{rd['citations']} | {rt['citations_valid']}/{rt['citations']} |" in README
    assert f"| Synthèses rejetées par les contrôles | {len(rd['rejections'])} | {len(rt['rejections'])} |" in README
    assert all(v for k, v in rd["criteria_met"].items() if k != "C4_atlas_identical") and "Ils sont atteints" in README
    per = ", ".join(f"{p} {v['correct']}/{v['n']}" for p, v in rd["answerable"]["per_category"].items())
    assert f"Par catégorie en dev : {per}." in README
    missed = [x for x in rd["per_query"] if x["relevant_unit_ids"] and x["piege"] != "sans_reponse" and not x["correct"]]
    assert [x["query_id"] for x in missed] == ["q_dev_002"] and missed[0]["status"] == "non_appelee"
    assert f"(meilleur cosinus de {fr(missed[0]['best_dense_score'])})" in README
    probes = rd["abstention_probes"]
    acceptable = probes["n"] - probes["abstained"] - probes["wrongly_answered"]
    assert f"{probes['abstained']} abstentions et {acceptable} réponse acceptable sur le fond" in README
    assert f"soit {probes['wrongly_answered']} réponse à tort, contre 5 sans RAG" in README
    calls, tokens = 0, 0
    for line in (ROOT / "data/processed/groq_rag_cache.jsonl").read_text(encoding="utf-8").splitlines():
        usage = json.loads(line).get("usage") or {}
        calls, tokens = calls + 1, tokens + usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    assert f"{thousands(tokens)} tokens Groq ({calls} appels)" in README
    limits = README.split("## Limites assumées")[1].split("\n## ")[0]
    assert (
        f"({rd['sans_reponse']['abstained']} abstentions sur {rd['sans_reponse']['n']} en dev, {rt['sans_reponse']['abstained']} sur "
        f"{rt['sans_reponse']['n']} en test scellé, {probes['wrongly_answered']} réponse à tort sur les {probes['n']} sondes)"
    ) in limits
    ref = rd["answerable"]["per_category"]["reformulation"]
    assert f"la catégorie passe à {ref['correct']}/{ref['n']}" in limits
    assert "Il n'intervient jamais dans la recherche, la résolution des relations ni l'abstention sur le seuil" in README


def test_demo_scope_is_stated() -> None:
    limits = README.split("## Limites assumées")[1].split("\n## ")[0]
    assert "Ingestion et agent non reliés" not in limits  # plus vrai depuis la chaine complete
    assert "Ingestion et agent reliés en partie seulement : le parcours de démonstration" in limits
    demo = README.split("## Démonstration")[1].split("\n## ")[0]
    assert "Elle s'appuie sur les unités et relations annotées du projet Atlas" in demo


def test_abstention_threshold_quoted_from_the_code() -> None:
    source = (ROOT / "src/agent/answer.py").read_text(encoding="utf-8")
    threshold = float(re.search(r"^ABSTENTION_THRESHOLD = ([0-9.]+)", source, re.MULTILINE).group(1))
    assert f"passe sous {threshold:.2f}".replace(".", ",") in README


def test_readme_typography() -> None:
    for char, name in (("—", "tiret long"), ("–", "tiret moyen"), ("·", "point median")):
        assert char not in README, f"{name} present dans le README"


def test_architecture_section_is_built_from_the_json() -> None:
    section = README.split("### Comparaison d'architectures")[1].split("\n### ")[0]
    dev, test = load("architectures_dev.json"), load("architectures_test.json")
    labels = {
        "e5_small_base": "e5-small de base", "e5_small_finetuned": "e5-small fine-tuné (production)",
        "e5_small_sans_negatifs_durs": "e5-small fine-tuné sans négatifs difficiles", "e5_base_base": "e5-base de base",
        "e5_base_finetuned": "e5-base fine-tuné", "minilm_l12_base": "MiniLM-L12 de base", "minilm_l12_finetuned": "MiniLM-L12 fine-tuné",
    }
    for data in (dev, test):
        for name, label in labels.items():
            g = data["results"][name]["global"]
            row = f"| {label} | " + " | ".join(fr(g[m]) for m in ("Recall@3", "Recall@5", "MRR", "nDCG@5")) + " |"
            assert row in section, row
    assert f"{dev['queries']['evaluated']} requêtes évaluées sur {dev['queries']['total']}" in section
    assert f"{test['queries']['evaluated']} requêtes évaluées sur {test['queries']['total']}" in section
    for name, label in (("e5_small_finetuned", "e5-small"), ("e5_small_sans_negatifs_durs", "e5-small sans négatifs difficiles"),
                        ("e5_base_finetuned", "e5-base"), ("minilm_l12_finetuned", "MiniLM-L12")):
        r = dev["results"][name]
        seconds = r["training"]["train_seconds"]
        cells = [label, thousands(r["size"]["parameters"]), f"{r['size']['weights_mb']:.1f} Mo".replace(".", ","),
                 "durée non enregistrée" if seconds is None else f"{seconds:.1f} s".replace(".", ","),
                 f"{r['encode_latency_ms_per_query']:.2f} ms".replace(".", ",")]
        assert "| " + " | ".join(cells) + " |" in section, cells
    choice = dev["choice"]
    assert choice["eligible_architectures"] == [] and "e5-small reste en production" in section
    for name, verdict in choice["candidates"].items():
        assert fr(verdict["mrr"]) in section, name
    assert fr(choice["candidates"]["e5_base_finetuned"]["max_category_mrr_drop"]) in section
    head = README.split("## Évaluer le projet en 5 minutes")[1].split("\n## ")[0]
    c = choice["candidates"]
    assert (f"(e5-base {fr(c['e5_base_finetuned']['mrr'])}, MiniLM-L12 {fr(c['minilm_l12_finetuned']['mrr'])}, "
            f"contre {fr(choice['reference']['mrr'])}), e5-small reste en production") in head


def test_relations_v2_section_is_built_from_the_json() -> None:
    section = README.split("### Seconde passe de liaison des relations : résultat négatif")[1].split("\n### ")[0]
    data = load("relations_v2.json")
    dev, test = data["dev"], data["test"]
    for v in dev["variants"].values():  # toutes les variantes essayees, executees et indicatives
        s = v["relations"]["strict"]
        row = (f"| {v['prompt']} | {v['k']} | {v['mode']} | {'exécutée' if v['executed'] else 'indicative'} | "
               f"{v['relations']['extracted']} | {fr(s['precision'])} | {fr(s['recall'])} | {fr(s['f1'])} |")
        assert row in section, row
    for label, get in (("Relations prédites", lambda b: str(b["extracted"])), ("Relations justes", lambda b: str(b["strict"]["correct"])),
                       ("Précision stricte", lambda b: fr(b["strict"]["precision"])), ("Rappel strict", lambda b: fr(b["strict"]["recall"])),
                       ("F1 strict", lambda b: fr(b["strict"]["f1"]))):
        row = f"| {label} | {get(dev['before'])} | {get(dev['after'])} | {get(test['before'])} | {get(test['after'])} |"
        assert row in section, row
    c = dev["ceiling"]
    assert f"{c['both_endpoints_extracted']} des {c['annotated_relations']} relations du dev" in section
    assert f"ne peut pas dépasser {fr(c['max_strict_recall'])}" in section
    assert "Seul k = 3 a été exécuté" in section and "6 configurations exécutées" in section and "optimiste" in section
    assert "seul le test scellé, exécuté une fois, sert de mesure non biaisée" in section
    assert "mode complement a lui aussi été introduit après l'analyse des résultats de la v1" in section
    assert not dev["criterion_met"] and "Le critère n'est pas atteint : rien n'est activé" in section
    sel = dev["selected"]
    assert f"v2, k = {sel['k']}, {sel['mode']}" in section and sel["prompt"] == "v2"
    cost = {v: c["prompt_tokens"] + c["completion_tokens"] for v, c in dev["cost"].items()}
    test_cost = sum(c["prompt_tokens"] + c["completion_tokens"] for c in test["cost"].values())
    b = data["budget"]
    assert b["lot_tokens_consumed"] == cost["v1"] + cost["v2"] + test_cost
    assert (f"{thousands(b['lot_tokens_consumed'])} tokens Groq en {b['lot_groq_calls']} appels, sous le budget de "
            f"{thousands(b['limit_tokens'])} : {thousands(cost['v1'])} pour la v1 sur le dev, {thousands(cost['v2'])} pour la v2 "
            f"sur le dev et {thousands(test_cost)} pour le test scellé") in section
    totals = dev["error_analysis"]["after"]["totals"]
    assert (f"{totals['predicted_errors']['extremite_non_annotee']} relations prédites dont une extrémité n'est pas annotée, "
            f"{totals['predicted_errors']['mauvaise_cible']} mauvaises cibles et {totals['predicted_errors']['mauvais_type']} mauvais types") in section
    assert f"{totals['missed']['jamais_proposee']} ne sont jamais proposées" in section


def test_test2_section_is_built_from_the_json() -> None:
    section = README.split("### Test scellé élargi (projets 13 à 16)")[1].split("\n## ")[0]
    data = load("test2.json")
    st = data["stages"]
    for key, label in (("bm25", "BM25"), ("e5_base", "e5-small de base"), ("e5_finetuned", "e5-small fine-tuné"), ("hybrid_rrf", "Hybride RRF (agent)")):
        g = st["offline"]["retrieval"]["results"][key]["global"]
        assert f"| {label} | " + " | ".join(fr(g[m]) for m in ("Recall@3", "Recall@5", "MRR", "nDCG@5")) + " |" in section, key
    fa, ab, cx, rag = st["offline"]["final_answer"]["global"], st["offline"]["abstention"], st["offline"]["complex"], st["rag"]
    assert f"| Requêtes simples, réponse juste | {fa['correct']}/{fa['n']} | {rag['answerable']['correct']}/{rag['answerable']['n']} |" in section
    assert f"| Requêtes sans réponse, abstention | {ab['abstained']}/{ab['n']} | {rag['sans_reponse']['abstained']}/{rag['sans_reponse']['n']} |" in section
    assert (f"| Questions complexes, unités attendues affichées | {cx['expected_units_shown']}/{cx['expected_units']} | "
            f"{rag['complex']['expected_units_shown']}/{rag['complex']['expected_units']} |") in section
    do = st["offline"]["dossiers"]
    assert f"{do['expected_units_cited']} des {do['expected_units']} unités attendues sont citées" in section and do["units_without_source"] == 0
    assert f"{thousands(rag['groq_tokens'])} tokens Groq ({rag['groq_calls']} appels" in section
    ex = st["extraction"]
    assert ex["projects_measured"] == ["projet_13", "projet_14", "projet_15", "projet_16"] and ex["projects_not_measured"] == []
    assert sum(run["tokens"] for run in ex["interrupted_run"]["runs"]) == ex["groq_tokens"]
    assert "reprise ensuite sur les seuls projets 15 et 16, sans modifier ni la consigne ni le jeu" in section
    assert "une réponse pour p15_m02, construite sans les unités de p15_m01 ; cette réponse reste dans le cache mais n'est pas utilisée" in section
    frozen = load("extraction.json")["extraction"]["system_prompt_sha256"]
    assert f"(empreinte SHA-256 {frozen[:8]})" in section
    u, r = ex["metrics"]["units"], ex["metrics"]["relations"]
    assert f"| Unités annotées, extraites, appariées | {u['annotated']}, {u['extracted']}, {u['matched']} |" in section
    assert f"| F1 des unités | {fr(u['f1'])} |" in section and f"| Relations strictes, F1 | {fr(r['strict']['f1'])} |" in section
    assert f"{thousands(ex['groq_tokens'])} tokens Groq" in section
    assert "après lecture des interfaces de l'agent, ce qui peut le rendre plus favorable au système" in section
    assert "relations" not in st  # seconde passe de liaison non lancee sur ce jeu


def test_stats_tables_are_built_from_the_json() -> None:
    section = README.split("### Intervalles de confiance et significativité")[1].split("\n## ")[0]
    s = load("stats.json")
    sg = lambda v: f"{v:+.3f}".replace(".", ",")  # noqa: E731
    for key, label in (("MRR e5_finetuned", "MRR e5-small fine-tuné"), ("MRR hybrid_rrf", "MRR hybride RRF"),
                       ("reponse finale extractive", "Réponse finale extractive"), ("reponse finale RAG", "Réponse finale RAG contrôlé"),
                       ("abstention RAG", "Abstention RAG contrôlé"), ("F1 unites", "F1 des unités extraites"), ("F1 relations strict", "F1 strict des relations")):
        cells = [f"{fr(v['estimate'])} [{fr(v['ci95'][0])} ; {fr(v['ci95'][1])}]" for v in (s["intervals"][sp][key] for sp in ("dev", "test", "test2"))]
        assert f"| {label} | " + " | ".join(cells) + " |" in section, label
    for key, label in (("MRR e5 fine-tune contre e5 de base", "e5 fine-tuné contre e5 de base (MRR)"),
                       ("reponse finale RAG contre extractive", "RAG contre extractif (réponse juste)"),
                       ("abstention RAG contre extractive", "RAG contre extractif (abstention)")):
        cells = []
        for sp in ("dev", "test", "test2"):
            t = s["paired_tests"][sp][key]
            cell = f"{sg(t['difference'])} [{sg(t['ci95_difference'][0])} ; {sg(t['ci95_difference'][1])}], p = {fr(t['p_value_permutation'])}"
            cells.append(cell + (", significatif" if t["significant"] else ""))
        assert f"| {label} | " + " | ".join(cells) + " |" in section, label
    head = README.split("## Évaluer le projet en 5 minutes")[1].split("\n## ")[0]
    ft = s["intervals"]["dev"]["MRR e5_finetuned"]["ci95"]
    assert f"intervalle à 95 % du MRR fine-tuné : [{fr(ft[0])} ; {fr(ft[1])}] en dev" in head


def test_nli_section_is_built_from_the_json() -> None:
    section = README.split("### Contrôle automatique de fidélité (exploratoire)")[1].split("\n### ")[0]
    data = load("rag_nli.json")
    for split, label in (("dev", "Développement"), ("test2", "Test scellé élargi")):
        s = data["splits"][split]
        c, r = s["counts"], s["rates"]
        row = (f"| {label} | {s['syntheses']} | {s['sentences']} | {c['entailment']} ({fr(r['entailment'])}) | "
               f"{c['neutral']} ({fr(r['neutral'])}) | {c['contradiction']} ({fr(r['contradiction'])}) |")
        assert row in section, row
        assert "rejetee" not in s["excluded_responses"]
    excluded = {sp: sum(data["splits"][sp]["excluded_responses"].values()) for sp in ("dev", "test2")}
    assert f"{excluded['dev']} en dev, {excluded['test2']} sur le test scellé élargi" in section
    assert data["protocol"]["model"] in section and data["protocol"]["revision"][:7] in section
    assert "ne remplace pas une évaluation humaine, non réalisée" in section
    inspection = load("rag_nli_inspection.json")
    c = inspection["counts"]
    not_entailed = sum(len(data["splits"][sp]["not_entailed"]) for sp in ("dev", "test2"))
    assert sum(c.values()) == not_entailed == len(inspection["items"])
    assert (f"les {not_entailed} phrases non impliquées se répartissent en {c['reformulation ou synthese fidele']} reformulations ou "
            f"synthèses fidèles, {c['extrapolation legere']} extrapolations légères et {c['affirmation non soutenue']} affirmation non soutenue") in section
    assert "non une évaluation humaine indépendante" in section and "non une evaluation humaine independante" in inspection["reviewer"]


def test_five_minute_section_figures_come_from_the_json() -> None:
    head = README.split("## Évaluer le projet en 5 minutes")[1].split("\n## ")[0]
    dev = load("retrieval_dev.json")["experiments"]["compare_finetuned"]["results"]
    test = load("retrieval_test.json")["experiments"]["compare_finetuned"]["results"]
    assert (
        f"MRR de {fr(dev['e5_base']['global']['MRR'])} à {fr(dev['e5_finetuned']['global']['MRR'])} en dev, de "
        f"{fr(test['e5_base']['global']['MRR'])} à {fr(test['e5_finetuned']['global']['MRR'])} en test scellé"
    ) in head
    rd, rt = load("rag_dev.json")["results"], load("rag_test.json")["results"]
    assert (
        f"unité attendue citée pour {rd['answerable']['correct']} requêtes sur {rd['answerable']['n']} en dev et "
        f"{rt['answerable']['correct']} sur {rt['answerable']['n']} en test scellé"
    ) in head
    probes = rd["abstention_probes"]
    assert (
        f"{rd['sans_reponse']['abstained']} requêtes sans réponse sur {rd['sans_reponse']['n']} en dev, "
        f"{rt['sans_reponse']['abstained']} sur {rt['sans_reponse']['n']} en test scellé, "
        f"{probes['wrongly_answered']} réponse à tort sur {probes['n']} sondes"
    ) in head
    ex = load("agent_extracted_dev.json")["results"]
    assert (
        f"{ex['extracted']['global']['correct']} réponses justes sur {ex['extracted']['global']['n']}, contre "
        f"{ex['annotated']['global']['correct']} sur {ex['annotated']['global']['n']} avec la mémoire annotée"
    ) in head
    assert "releases/download/v1.0-rendu/demo_atlas.webm" in head
    assert "uv run python scripts/train_embeddings.py" in head and "uv run streamlit run app.py" in head

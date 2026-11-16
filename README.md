# Embedding Lab : de la mémoire sémantique à l'agent intelligent

## Objectif et périmètre

Ce projet construit un agent qui lit des comptes rendus de réunions et répond à des questions en citant sa preuve. L'agent distingue une décision en vigueur d'une décision remplacée ou contredite, identifie la personne qui doit réaliser une tâche, prépare un brouillon d'email sans jamais l'envoyer et s'abstient quand le meilleur passage trouvé est trop éloigné de la question, selon un seuil de similarité dont les limites sont décrites plus bas.

Le projet comporte quatre parties :

1. un corpus de réunions en français, découpé en unités sémantiques typées et reliées entre elles ;
2. un moteur de recherche hybride (BM25 et embeddings e5 fine-tunés, fusionnés par Reciprocal Rank Fusion) ;
3. un pipeline d'ingestion qui extrait automatiquement unités et relations d'un transcript brut à l'aide d'un LLM ;
4. un agent et son interface de démonstration Streamlit.

Le corpus interne est fictif : 12 projets, 26 réunions, 225 unités et 146 relations, écrits pour le projet et contenant volontairement les cas difficiles (proposition contre décision, date ou responsable qui change, information contredite, question sans réponse). Une évaluation externe sur le jeu public QMSum complète ces mesures pour la seule recherche de passages.

Le LLM (openai/gpt-oss-120b via Groq) intervient à deux endroits, toujours en amont de l'agent : il reformule des questions pour construire les triplets d'entraînement, et il extrait les unités et les relations dans le pipeline d'ingestion. Il n'intervient jamais dans la recherche ni dans les réponses de l'agent.

## Structure du dépôt

    src/schemas.py               contrats de données (unité, relation, brouillon d'email)
    src/memory/corpus.py         chargement du corpus, des relations et des participants
    src/retrieval/               recherches de référence BM25 et dense (e5)
    src/ingest/                  extraction d'unités et de relations par LLM, client Groq avec cache
    src/agent/                   recherche hybride, réponse sourcée, historique, passages sources, intentions, email
    eval/                        métriques et scripts d'évaluation
    eval/results/                résultats enregistrés en JSON, avec métadonnées et empreintes SHA-256
    scripts/                     validation du corpus, triplets, entraînement, démo en ligne de commande,
                                 vérification et enregistrement vidéo de la démonstration
    data/raw/                    les 12 projets annotés
    data/eval/                   requêtes de développement et de test scellé
    data/processed/              triplets d'entraînement
    data/external/               documentation des corpus externes (les données ne sont pas versionnées)
    app.py                       interface de démonstration Streamlit
    tests/                       tests pytest

## Entraînement des embeddings

Le modèle intfloat/multilingual-e5-small est fine-tuné par apprentissage contrastif avec la fonction de perte MultipleNegativesRankingLoss (`scripts/train_embeddings.py`), sur des triplets issus des seuls projets de développement :

- 20 triplets construits à la main (`data/processed/triplets_dev.jsonl`) : questions écrites à partir de modèles, négatif miné dans le même projet, puis sélection manuelle ;
- 351 triplets enrichis (`data/processed/triplets_llm.jsonl`) : Groq reformule la question qui sert d'ancre, le positif reste l'énoncé réel du corpus et le négatif est miné dans le même projet.

Aucun triplet ne provient des projets 11 et 12. En revanche, le jeu de développement a aussi servi à régler la génération des triplets : il n'est plus vierge, et seul le test scellé reste indépendant du réglage ; ses résultats restent fragiles, car il ne comporte que 5 requêtes évaluées.

## Résultats

Tous les chiffres de cette section proviennent des fichiers JSON de `eval/results`. Le découpage se fait par projet : les projets 01 à 10 servent au développement, les projets 11 et 12 forment le test scellé, qui n'a jamais servi à régler quoi que ce soit.

### Recherche sur le corpus interne

Fichiers `retrieval_dev.json` et `retrieval_test.json`. Les requêtes sans réponse sont exclues de ces mesures.

Développement, 25 requêtes évaluées sur 30 :

| Système | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| BM25 | 0,800 | 0,880 | 0,631 | 0,695 |
| e5-small de base | 0,680 | 0,880 | 0,526 | 0,614 |
| e5-small fine-tuné | 0,840 | 1,000 | 0,716 | 0,786 |

Le fine-tuning contrastif fait passer le modèle dense devant BM25 sur les quatre métriques. Les deux faiblesses ciblées progressent le plus : sur la négation, le MRR passe de 0,207 à 0,507, et sur les engagements de 0,550 à 1,000.

Test scellé, 5 requêtes évaluées sur 6 :

| Système | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| e5-small de base | 0,600 | 0,800 | 0,640 | 0,677 |
| e5-small fine-tuné | 0,800 | 1,000 | 0,750 | 0,812 |

Avec seulement 5 requêtes, ce résultat est un signal fragile et non une preuve : une seule requête qui change de rang suffit à modifier fortement les moyennes.

La recherche hybride utilisée par l'agent est évaluée sur QMSum (voir plus bas), mais pas encore sur le corpus interne.

### Extraction automatique des unités et des relations

Fichier `extraction.json`. La consigne d'extraction a été réglée sur les projets 01 à 03 puis figée (empreinte SHA-256 enregistrée dans le JSON). Les mesures portent sur les projets 04 à 12, que le réglage n'a jamais vus. Une unité extraite est appariée à une unité annotée quand leurs preuves se recouvrent dans le transcript, ou à défaut quand leurs énoncés sont très proches.

| Mesure | Dev 04-10 | Test 11-12 |
|---|---|---|
| Unités annotées, extraites, appariées | 131, 160, 124 | 34, 48, 34 |
| Précision des unités | 0,775 | 0,708 |
| Rappel des unités | 0,947 | 1,000 |
| F1 des unités | 0,852 | 0,829 |
| Type exact | 100/124 (0,806) | 23/34 (0,676) |
| Statut exact | 120/124 (0,968) | 33/34 (0,971) |
| Relations annotées, extraites | 84, 54 | 23, 19 |
| Relations strictes, précision | 0,518 | 0,421 |
| Relations strictes, rappel | 0,333 | 0,348 |
| Relations strictes, F1 | 0,406 | 0,381 |

Une relation est comptée juste seulement si ses deux extrémités, son sens et son type correspondent à l'annotation. Sur l'ensemble des projets 04 à 12, le F1 strict des relations vaut 0,400. Par type de relation :

| Relation | Annotées dev | F1 dev | Annotées test | F1 test |
|---|---|---|---|---|
| answers | 16 | 0,710 | 3 | 0,444 |
| creates_action | 25 | 0,279 | 7 | 0,200 |
| supports | 35 | 0,353 | 12 | 0,455 |
| supersedes | 7 | 0,364 | 1 | 0,000 |
| contradicts | 1 | 0,000 | 0 | sans objet |

La relation creates_action est la plus faible. L'extraction des projets 04 à 12 a demandé 19 appels Groq, pour 37 815 tokens en entrée et 60 327 en sortie.

### Évaluation externe sur QMSum

Fichier `real_qmsum.json`. Cette évaluation mesure uniquement la recherche de passages dans des réunions en anglais : elle ne mesure ni la détection des décisions, ni celle des engagements, des contradictions ou des responsabilités. Chaque tour de parole est un passage, et un passage est pertinent s'il tombe dans un intervalle annoté par QMSum. Les quatre systèmes sont comparés sur exactement les mêmes requêtes du split test, et deux exécutions complètes donnent des valeurs identiques.

| Groupe | Système | n | Hit@5 | Hit@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| Global hors Pays de Galles | BM25 | 196 | 0,663 | 0,801 | 0,506 | 0,246 |
| Global hors Pays de Galles | e5 de base | 196 | 0,806 | 0,888 | 0,622 | 0,317 |
| Global hors Pays de Galles | e5 fine-tuné | 196 | 0,796 | 0,893 | 0,622 | 0,338 |
| Global hors Pays de Galles | Hybride RRF | 196 | 0,760 | 0,857 | 0,573 | 0,312 |
| Product (AMI) | BM25 | 129 | 0,705 | 0,876 | 0,524 | 0,257 |
| Product (AMI) | e5 de base | 129 | 0,814 | 0,891 | 0,618 | 0,315 |
| Product (AMI) | e5 fine-tuné | 129 | 0,806 | 0,891 | 0,653 | 0,353 |
| Product (AMI) | Hybride RRF | 129 | 0,791 | 0,922 | 0,588 | 0,327 |
| Academic (ICSI) | BM25 | 49 | 0,510 | 0,571 | 0,390 | 0,148 |
| Academic (ICSI) | e5 de base | 49 | 0,775 | 0,837 | 0,586 | 0,274 |
| Academic (ICSI) | e5 fine-tuné | 49 | 0,735 | 0,857 | 0,514 | 0,258 |
| Academic (ICSI) | Hybride RRF | 49 | 0,653 | 0,694 | 0,466 | 0,209 |
| Committee, Canada | BM25 | 18 | 0,778 | 0,889 | 0,695 | 0,432 |
| Committee, Canada | e5 de base | 18 | 0,833 | 1,000 | 0,749 | 0,444 |
| Committee, Canada | e5 fine-tuné | 18 | 0,889 | 1,000 | 0,691 | 0,442 |
| Committee, Canada | Hybride RRF | 18 | 0,833 | 0,833 | 0,754 | 0,488 |

Les comités du Parlement gallois sont isolés, car leur licence de réutilisation n'a pas pu être vérifiée (voir `data/external/README.md`). Ils ne sont pas inclus dans le global ci-dessus :

| Groupe | Système | n | Hit@5 | Hit@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| Committee, Pays de Galles | BM25 | 48 | 0,875 | 0,917 | 0,705 | 0,450 |
| Committee, Pays de Galles | e5 de base | 48 | 0,854 | 0,938 | 0,721 | 0,474 |
| Committee, Pays de Galles | e5 fine-tuné | 48 | 0,854 | 0,958 | 0,754 | 0,487 |
| Committee, Pays de Galles | Hybride RRF | 48 | 0,896 | 0,958 | 0,751 | 0,506 |

Le fine-tuning sur le corpus français s'applique à l'anglais sans dégradation mesurable sur le global hors Pays de Galles : le nDCG@10 passe de 0,317 à 0,338, le MRR reste à 0,622 et le Hit@5 varie à peine (0,806 puis 0,796). Il recule en revanche sur Academic (MRR de 0,586 à 0,514). Sur ce même global, l'hybride (nDCG@10 de 0,312) reste sous le e5 fine-tuné (0,338) ; il obtient le meilleur nDCG@10 sur les comités et le meilleur Hit@10 sur Product (0,922). BM25 est le plus faible des quatre systèmes sur Product et Academic ; une explication possible, non testée ici, est que la parole transcrite (hésitations, phrases inachevées, reformulations) se prête mal à une correspondance de mots exacts, ce qui pèserait aussi sur l'hybride. Le Recall@K, présent dans le JSON, est peu lisible ici : les intervalles couvrent souvent des dizaines de tours, si bien que même un classement parfait plafonne en moyenne à 0,163 de Recall@5 sur Product.

## Limites assumées

- Relations : c'est la partie la plus faible, avec un F1 strict de 0,406 en dev et 0,381 en test. L'analyse des erreurs du dev 04-10 (`eval/results/relation_errors_dev.json`) montre qu'aucune relation prédite n'est fausse à cause de son seul sens. Sur les 54 relations prédites, 28 sont justes ; parmi les erreurs, 16 relient une unité absente de l'annotation et 7 visent la mauvaise cible. Sur les 84 relations annotées, 56 sont manquées, dont 27 jamais proposées (15 supports).
- Abstention : l'agent s'abstient quand le meilleur cosinus dense passe sous 0,45 (seuil fixé dans `src/agent/answer.py`). Ce seuil écarte bien les questions hors sujet, mais il ne sépare pas une question sur le bon sujet dont la réponse est absente des sources : dans ce cas, l'agent peut répondre à tort. Un cross-encoder ou un modèle d'inférence textuelle serait la solution propre.
- Recul sur les décisions en vigueur : en dev, le fine-tuning fait baisser le MRR de 0,800 à 0,707 et le Recall@3 de 1,000 à 0,800 sur cette catégorie de requêtes. La résolution par les relations de l'agent est conçue pour compenser ce cas, sans que cela soit encore mesuré.
- Surextraction : sur les projets 04 à 12, l'extraction produit 19 unités idea pour 4 annotées.
- Confusion entre action et engagement : en dev, 4 engagements annotés sont extraits comme actions et 2 actions comme engagements ; sur les projets 04 à 12, le rappel des actions n'est que de 0,294. Une partie de ces écarts vient d'un désaccord de convention entre la consigne et l'annotation.
- Jeux d'évaluation petits : en dev, 25 requêtes évaluées pour la recherche ; en test scellé, 5 requêtes évaluées pour la recherche et 34 unités appariées pour l'extraction. Les écarts sont indicatifs, pas statistiquement solides.
- Dev non vierge : il a servi au réglage de la génération des triplets (voir plus haut).
- Projets 09 et 10 peu couverts : seuls 4 triplets construits à la main en proviennent, aucun triplet enrichi.
- Règle du responsable : la personne qui doit réaliser une tâche est le premier participant nommé dans l'énoncé de l'unité. Cette heuristique est vérifiée sur le corpus, mais elle n'est pas généralisable telle quelle.
- Routage et brouillon d'email : la détection d'intention et la préparation des brouillons sont vérifiées sur des exemples de démonstration, mais ne font encore l'objet ni d'une mesure chiffrée ni d'un test automatique.
- Corpus fictif : les 12 projets ont été écrits pour le projet. Sur QMSum, la majorité des réunions Product (AMI) sont elles aussi jouées selon un scénario.

## Démonstration

La démonstration porte sur le projet fictif Atlas (projet_01), dont trois comptes rendus successifs racontent une mise en production d'abord proposée puis validée pour le 14 avril, puis reportée au 5 mai. Le parcours se déroule en cinq temps :

1. retrouver la décision en vigueur, avec sa preuve et sa source ;
2. identifier l'action qui en découle, avec son responsable et son échéance ;
3. montrer la décision remplacée et le chemin suivi pour arriver à l'information à jour ;
4. afficher à l'écran les passages sources du transcript, pour la décision en vigueur et pour celles qu'elle remplace ;
5. déclencher l'abstention sur une question hors des sources.

Commandes :

    uv run streamlit run app.py --server.address localhost
    uv run python scripts/verify_demo.py

Le second script vérifie automatiquement les quatre points du parcours (remplacement de décision reproductible, affichage des sources, abstention, responsable et échéance de l'action). Une vidéo de la démonstration peut être enregistrée avec `scripts/record_demo.py` ; elle n'est pas versionnée.

## Audit qualitatif sur AMI

Un audit exploratoire applique l'extraction figée à trois réunions AMI du split test de QMSum, découpées en segments de 40 tours de parole. Il est partiel : 6 segments sur 23 ont été traités avant que le quota quotidien de Groq soit atteint. Il ne constitue pas une validation statistique, et son script n'est pas versionné tant qu'il n'est pas terminé.

## Reproductibilité

Environnement géré avec uv et Python 3.12 ; toutes les versions de `requirements.txt` sont épinglées.

    uv venv --python 3.12
    uv pip install -r requirements.txt
    uv run pytest -q
    uv run python scripts/validate_corpus.py

Le modèle fine-tuné n'est pas versionné : il faut l'entraîner une fois. Le premier lancement télécharge le modèle de base depuis Hugging Face.

    uv run python scripts/train_embeddings.py

Recherche sur le corpus interne (l'option `--out` réécrit le JSON correspondant) :

    uv run python eval/run_baselines.py --out eval/results/retrieval_dev.json
    uv run python eval/compare_finetuned.py --out eval/results/retrieval_dev.json
    uv run python eval/compare_finetuned.py --queries data/eval/queries_test.json

Extraction (se rejoue depuis le cache Groq local, sans aucun appel) :

    uv run python eval/eval_extraction.py --projects 4-12 --groups dev_04_10=4-10 test_11_12=11-12 --cache-only

Analyse des erreurs de relations sur le dev 04-10, depuis le même cache local (réécrit `eval/results/relation_errors_dev.json`) :

    uv run python eval/analyze_relation_errors.py

Évaluation QMSum, après avoir cloné https://github.com/Yale-LILY/QMSum dans `data/external/qmsum` (voir `data/external/README.md`). La commande lance deux exécutions complètes, vérifie qu'elles sont identiques et réécrit `eval/results/real_qmsum.json` :

    uv run python eval/eval_real.py --check-reproducible

Démonstration en ligne de commande :

    uv run python scripts/demo_agent.py "Quand a lieu la mise en production d'Atlas ?" projet_01

Points d'attention :

- Le cache des réponses Groq n'est pas versionné. `extraction.json` se rejoue donc à l'identique depuis un cache local, mais pas depuis un simple clone du dépôt ; relancer l'extraction sans cache demande une clé Groq, et Groq n'est pas strictement déterministe, même à température nulle.
- Chaque JSON de `eval/results` enregistre les empreintes SHA-256 des requêtes (calculées après normalisation des fins de ligne), des paramètres d'évaluation et d'entraînement, et des poids du modèle fine-tuné.
- Playwright n'est pas dans `requirements.txt` : il ne sert qu'à enregistrer la vidéo de démonstration (`uv pip install playwright==1.48.0`, puis `uv run playwright install ffmpeg`).

## Sécurité

La clé Groq se place uniquement dans un fichier `.env` à la racine (`GROQ_API_KEY=...`), ignoré par git. Elle sert à générer les triplets enrichis et à extraire les unités et relations ; elle n'est nécessaire ni pour entraîner le modèle, ni pour faire tourner l'agent, ni pour rejouer les évaluations depuis les caches. Le fichier `.env.example` versionné ne contient aucune valeur, et l'historique git ne contient aucune clé.

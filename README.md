# Embedding Lab : de la mémoire sémantique à l'agent intelligent

[![tests](https://github.com/Kadi2207/semantic-agent-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/Kadi2207/semantic-agent-engine/actions/workflows/tests.yml)

## Évaluer le projet en 5 minutes

Un agent lit des comptes rendus de réunions et répond en citant sa preuve : il distingue une décision en vigueur d'une décision remplacée, signale un conflit qu'il ne peut pas trancher, identifie le responsable d'une action, prépare un brouillon d'email sans l'envoyer, constitue un dossier documenté et s'abstient quand les sources ne contiennent pas la réponse.

Vidéo de la démonstration (parcours Atlas en sept étapes) : [demo_atlas.webm, Release v1.0-rendu](https://github.com/Kadi2207/semantic-agent-engine/releases/download/v1.0-rendu/demo_atlas.webm).

Cinq chiffres clés, tirés de `eval/results` (développement, puis test scellé quand il existe) :

- fine-tuning contrastif des embeddings : MRR de 0,526 à 0,716 en dev, de 0,640 à 0,750 en test scellé (`retrieval_dev.json`, `retrieval_test.json`) ; intervalle à 95 % du MRR fine-tuné : [0,586 ; 0,841] en dev ; gain significatif en dev, non significatif en test scellé et sur le test scellé élargi (`stats.json`) ;
- comparaison d'architectures : aucun autre modèle fine-tuné ne dépasse e5-small en dev (e5-base 0,693, MiniLM-L12 0,611, contre 0,716), e5-small reste en production (`architectures_dev.json`) ;
- réponse finale avec le RAG contrôlé : unité attendue citée pour 24 requêtes sur 25 en dev et 5 sur 5 en test scellé (`rag_dev.json`, `rag_test.json`) ; intervalle à 95 % en dev : [0,880 ; 1,000] ; gain sur la réponse extractive significatif en dev seulement ;
- abstention avec le RAG contrôlé : 4 requêtes sans réponse sur 5 en dev, 1 sur 1 en test scellé, 0 réponse à tort sur 6 sondes (`rag_dev.json`) ; intervalle à 95 % en dev : [0,400 ; 1,000] ; gain sur l'extractif significatif seulement sur le test scellé élargi (7 sur 8 contre 1 sur 8) ;
- chaîne complète sur la mémoire produite par l'extraction automatique : 10 réponses justes sur 20, contre 13 sur 20 avec la mémoire annotée, sur les mêmes requêtes (`agent_extracted_dev.json`).

Lancer la démonstration en une commande (vérifie l'environnement, entraîne le modèle s'il manque, lance `scripts/verify_demo.py` puis l'interface Streamlit en local) :

    uv venv --python 3.12
    uv pip install -r requirements.txt
    uv run python scripts/run_demo.py

Ou étape par étape (Python 3.12 et uv ; l'entraînement du modèle prend quelques minutes sur CPU et télécharge le modèle de base) :

    uv venv --python 3.12
    uv pip install -r requirements.txt
    uv run python scripts/train_embeddings.py
    uv run streamlit run app.py --server.address localhost
    uv run python scripts/verify_demo.py

Aucune clé n'est nécessaire : les synthèses du RAG de la démonstration sont rejouées depuis un cache versionné, et sans réseau l'agent répond de façon extractive.

## Objectif et périmètre

Ce projet construit un agent qui lit des comptes rendus de réunions et répond à des questions en citant sa preuve. L'agent distingue une décision en vigueur d'une décision remplacée ou contredite, identifie la personne qui doit réaliser une tâche, prépare un brouillon d'email sans jamais l'envoyer, constitue un dossier documenté sur un sujet ou un projet, et s'abstient quand le meilleur passage trouvé est trop éloigné de la question, selon un seuil de similarité dont les limites sont décrites plus bas.

Quand deux affirmations contradictoires sont toutes deux encore en vigueur, l'agent ne choisit pas : il affiche un conflit non tranché, avec les deux affirmations et leurs sources, sans en barrer aucune et sans conclure. Une contradiction ne vaut résolution que si l'unité contredite n'est plus active. L'interface affiche le type de chaque unité en français, présente une proposition comme « proposition, non validée » et, pour une échéance « le jour même », ajoute la date de la réunion sans rien calculer d'autre.

Le projet comporte quatre parties :

1. un corpus de réunions en français, découpé en unités sémantiques typées et reliées entre elles ;
2. un moteur de recherche hybride (BM25 et embeddings e5 fine-tunés, fusionnés par Reciprocal Rank Fusion) ;
3. un pipeline d'ingestion qui extrait automatiquement unités et relations d'un transcript brut à l'aide d'un LLM ;
4. un agent et son interface de démonstration Streamlit.

Le corpus interne est fictif : 12 projets, 26 réunions, 225 unités et 146 relations, écrits pour le projet et contenant volontairement les cas difficiles (proposition contre décision, date ou responsable qui change, information contredite, question sans réponse). Une évaluation externe sur le jeu public QMSum complète ces mesures pour la seule recherche de passages.

Le LLM (openai/gpt-oss-120b via Groq) intervient à trois endroits. En amont de l'agent, il reformule des questions pour construire les triplets d'entraînement, et il extrait les unités et les relations dans le pipeline d'ingestion. Dans l'agent, le RAG contrôlé lui fait rédiger une synthèse à partir des seules unités retrouvées, dont chaque citation est vérifiée par le code. Il n'intervient jamais dans la recherche, la résolution des relations ni l'abstention sur le seuil, et l'agent répond sans lui quand il est indisponible.

## Structure du dépôt

La chaîne complète et le rôle de chaque module sont décrits dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ; chaque choix technique, l'alternative écartée et sa preuve chiffrée, dans [docs/DECISIONS.md](docs/DECISIONS.md).

    src/schemas.py               contrats de données (unité, relation, brouillon d'email)
    src/memory/corpus.py         chargement du corpus, des relations et des participants
    src/retrieval/               recherches de référence BM25 et dense (e5)
    src/ingest/                  extraction d'unités et de relations par LLM, client Groq avec cache
    src/agent/                   recherche hybride, réponse sourcée, conflits, historique, passages sources, affichage, intentions, email
    eval/                        métriques et scripts d'évaluation
    eval/results/                résultats enregistrés en JSON, avec métadonnées et empreintes SHA-256
    scripts/                     validation du corpus, triplets, entraînement, démo en ligne de commande,
                                 vérification et enregistrement vidéo de la démonstration
    data/raw/                    les 12 projets annotés
    data/test2/                  les 4 projets du test scellé élargi (13 à 16), gelés avant toute exécution
    data/extracted/              mémoire produite par l'extraction automatique (projets 04 à 12), sans correction
    data/eval/                   requêtes de développement, de test scellé et de test scellé élargi, sondes d'abstention, manifeste du test élargi
    data/processed/              triplets d'entraînement et cache versionné du RAG de la démonstration
    data/external/               documentation des corpus externes (les données ne sont pas versionnées)
    app.py                       interface de démonstration Streamlit (thème dans .streamlit/config.toml)
    docs/                        architecture de la chaîne et décisions techniques
    tests/                       tests pytest
    .github/workflows/tests.yml  intégration continue : pytest à chaque push (Python 3.12, torch CPU)

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
| Hybride RRF (agent) | 0,920 | 0,960 | 0,708 | 0,772 |

Le fine-tuning contrastif fait passer le modèle dense devant BM25 sur les quatre métriques. Les deux faiblesses ciblées progressent le plus : sur la négation, le MRR passe de 0,207 à 0,507, et sur les engagements de 0,550 à 1,000.

Test scellé, 5 requêtes évaluées sur 6 :

| Système | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| e5-small de base | 0,600 | 0,800 | 0,640 | 0,677 |
| e5-small fine-tuné | 0,800 | 1,000 | 0,750 | 0,812 |
| Hybride RRF (agent) | 0,800 | 1,000 | 0,850 | 0,886 |

Avec seulement 5 requêtes, ce résultat est un signal fragile et non une preuve : une seule requête qui change de rang suffit à modifier fortement les moyennes.

La ligne Hybride RRF est la recherche utilisée par l'agent : BM25 et e5 fine-tuné fusionnés par RRF avec k = 60, mesurés par `eval/eval_hybrid.py` sur le classement de `HybridRetriever`. En dev, l'hybride obtient le meilleur Recall@3 (0,920) mais reste légèrement sous le e5 fine-tuné seul sur les trois autres métriques (MRR de 0,708 contre 0,716) : la fusion n'apporte pas de gain net sur ce corpus. Sur les décisions en vigueur, son MRR est de 0,840. Sur le test scellé, l'hybride et la réponse finale ci-dessous proviennent d'une seule exécution d'enregistrement, sans aucun réglage.

### Comparaison d'architectures

Fichiers `architectures_dev.json` et `architectures_test.json`, produits par `eval/eval_architectures.py` ; entraînements par `scripts/train_architectures.py`, avec exactement le protocole de `scripts/train_embeddings.py` (371 triplets, 3 époques, lot de 16, taux d'apprentissage 2e-5, graine 42, MultipleNegativesRankingLoss). Seuls changent le modèle de base et ses conventions d'entrée : préfixes « query: » et « passage: » pour e5, aucun préfixe pour paraphrase-multilingual-MiniLM-L12-v2. Recherche dense seule, mêmes requêtes et mêmes métriques que ci-dessus.

Développement, 25 requêtes évaluées sur 30 :

| Modèle | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| e5-small de base | 0,680 | 0,880 | 0,526 | 0,614 |
| e5-small fine-tuné (production) | 0,840 | 1,000 | 0,716 | 0,786 |
| e5-small fine-tuné sans négatifs difficiles | 0,840 | 0,920 | 0,689 | 0,747 |
| e5-base de base | 0,720 | 0,840 | 0,475 | 0,567 |
| e5-base fine-tuné | 0,880 | 0,960 | 0,693 | 0,761 |
| MiniLM-L12 de base | 0,680 | 0,840 | 0,431 | 0,533 |
| MiniLM-L12 fine-tuné | 0,840 | 1,000 | 0,611 | 0,708 |

Test scellé, 5 requêtes évaluées sur 6, une seule exécution d'enregistrement :

| Modèle | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| e5-small de base | 0,600 | 0,800 | 0,640 | 0,677 |
| e5-small fine-tuné (production) | 0,800 | 1,000 | 0,750 | 0,812 |
| e5-small fine-tuné sans négatifs difficiles | 0,800 | 1,000 | 0,717 | 0,786 |
| e5-base de base | 0,800 | 1,000 | 0,617 | 0,712 |
| e5-base fine-tuné | 1,000 | 1,000 | 0,767 | 0,826 |
| MiniLM-L12 de base | 0,600 | 0,600 | 0,367 | 0,426 |
| MiniLM-L12 fine-tuné | 0,800 | 0,800 | 0,567 | 0,626 |

Coût de chaque modèle (CPU ; latence = encodage d'une requête, moyenne sur le dev) :

| Modèle | Paramètres | Poids | Entraînement | Latence |
|---|---|---|---|---|
| e5-small | 117 653 760 | 448,8 Mo | durée non enregistrée | 48,23 ms |
| e5-small sans négatifs difficiles | 117 653 760 | 448,8 Mo | 217,1 s | 47,51 ms |
| e5-base | 278 043 648 | 1060,7 Mo | 1435,0 s | 79,15 ms |
| MiniLM-L12 | 117 653 760 | 448,8 Mo | 231,5 s | 40,94 ms |

Ablation : sans les négatifs difficiles (paires seules, les négatifs sont les autres exemples du lot), le MRR dev de e5-small fine-tuné passe de 0,716 à 0,689 ; le recul porte sur la reformulation (MRR de 0,600 à 0,507) et la négation (de 0,507 à 0,467). Les négatifs difficiles minés dans le même projet apportent donc un gain, modeste, sur 25 requêtes.

Choix : la règle, fixée avant les résultats et appliquée au seul dev, ne remplace e5-small que si un candidat fine-tuné bat son MRR dev sans recul de plus de 0,1 du MRR sur aucune catégorie, et si `verify_demo` et le parcours Atlas restent identiques. Aucun candidat ne bat le MRR dev de e5-small (0,716) : e5-base fine-tuné atteint 0,693, avec en plus un recul de 0,140 sur les décisions en vigueur, et MiniLM-L12 fine-tuné 0,611. e5-small reste en production. Sur le test scellé, e5-base fine-tuné obtient un MRR un peu plus élevé (0,767 contre 0,750) ; avec 5 requêtes, cet écart tient à une seule requête et ne sert pas au choix. e5-base coûte en outre 2,4 fois plus de paramètres et environ 1,6 fois la latence de e5-small.

### Réponse finale de l'agent

Fichiers `agent_dev.json` et `agent_test.json`, produits par `eval/eval_agent.py`. Ces chiffres sont ceux de la réponse extractive, sans le RAG contrôlé décrit plus bas. Une réponse est comptée juste si l'agent ne s'abstient pas et si l'unité affichée après résolution des relations est une unité pertinente ; pour un conflit non tranché, il suffit qu'une unité pertinente figure parmi les affirmations affichées.

| Catégorie | Dev | Test scellé |
|---|---|---|
| Global | 17/25 (0,680) | 4/5 (0,800) |
| decision_active | 5/5 | 1/1 |
| reformulation | 1/5 | 0/1 |
| responsable_reel | 3/5 | 1/1 |
| negation | 4/5 | 1/1 |
| engagement | 4/5 | 1/1 |

En dev, le compte global est de 17/25 (16/25 si le conflit non tranché de q_dev_026 n'est pas compté juste). En dev, 3 réponses passent par une résolution (l'unité retrouvée était périmée et l'agent a suivi les relations jusqu'à l'unité active), toutes justes, et 1 réponse est un conflit non tranché. La reformulation est la catégorie la plus faible. L'agent s'abstient à tort 1 fois sur ces 25 requêtes.

Requêtes sans réponse (abstention attendue) : l'agent s'abstient 0 fois sur 5 en dev et 0 fois sur 1 en test scellé. Six sondes complémentaires (`data/eval/probes_abstention.json`) posent une question dont le sujet existe dans le projet mais dont la réponse est absente des sources : 6 questions, 0 abstention ; 5 réponses à tort et 1 acceptable sur le fond (`p06_u25` indique qu'aucune décision n'a été prise), selon la règle écrite dans le fichier. Ce sont des sondes exploratoires, sur les projets de développement, jamais utilisées pour régler. 5 des 6 sondes sont les requêtes sans_reponse du dev et seule la question sur le budget publicitaire d'Atlas est nouvelle : ce n'est pas une mesure indépendante.

### RAG contrôlé

Fichiers `rag_dev.json` et `rag_test.json`, produits par `eval/eval_rag.py` ; module `src/agent/rag.py`. Après la recherche hybride, l'abstention et la résolution (inchangées), un appel à gpt-oss-120b (Groq, température 0, sortie JSON) reçoit la question et les 5 unités en vigueur retrouvées, avec leurs identifiants. Il répond si la question est couverte et, si oui, donne une synthèse dont chaque phrase cite les unités qui la justifient. Le code vérifie que chaque identifiant cité fait partie des unités fournies et qu'aucune phrase n'est sans citation ; sinon la synthèse est rejetée et l'agent garde la réponse extractive. Si le modèle juge la question non couverte, l'agent s'abstient avec la mention « Les sources trouvées ne contiennent pas cette information ». La preuve extractive (citation exacte et passage surligné) reste affichée sous la synthèse, marquée « générée à partir des sources citées ». Un conflit non tranché n'est jamais soumis au modèle. Sans réseau ni clé, l'agent répond exactement comme sans RAG ; les réponses du modèle sont mises en cache, et le cache est versionné pour rejouer la démonstration hors ligne.

Les critères d'acceptation ont été fixés avant l'exécution (`eval/eval_rag.py`) : au moins 23 des 25 requêtes avec réponse du dev répondues avec une unité attendue parmi les citations, au moins 4 des 5 requêtes sans réponse devenues des abstentions, 100 % de citations valides, parcours Atlas identique en preuve extractive. Ils sont atteints, et le mode est donc activé par défaut dans l'interface.

| Mesure | Dev | Test scellé |
|---|---|---|
| Requêtes avec réponse, unité attendue citée | 24/25 | 5/5 |
| Requêtes sans réponse, abstention | 4/5 | 1/1 |
| Citations valides | 28/28 | 5/5 |
| Synthèses rejetées par les contrôles | 0 | 0 |

Par catégorie en dev : decision_active 5/5, reformulation 4/5, responsable_reel 5/5, negation 5/5, engagement 5/5. La seule requête avec réponse manquée est q_dev_002, écartée avant tout appel par le seuil de similarité (meilleur cosinus de 0,440). Sur les 6 sondes d'abstention : 5 abstentions et 1 réponse acceptable sur le fond (`p06_u25`, aucune décision prise), soit 0 réponse à tort, contre 5 sans RAG. Le test scellé provient d'une seule exécution d'enregistrement. L'ensemble de la journée a consommé 25 895 tokens Groq (40 appels).

### Contrôle automatique de fidélité (exploratoire)

Fichier `rag_nli.json`, produit par `eval/nli_faithfulness.py`, sans aucun appel Groq. Protocole fixé avant l'exécution : chaque phrase d'une synthèse du RAG contrôlé en cache (dev et test scellé élargi) est soumise à un modèle NLI multilingue local (MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7, révision b5113eb) ; la prémisse est le texte intégral des unités citées par la phrase (énoncé puis preuve), l'hypothèse est la phrase sans ses citations, et la prédiction est l'étiquette de score maximal, sans seuil. Les réponses qui ne sont pas des synthèses acceptées sont exclues (abstentions du modèle ou du seuil de similarité, conflits non tranchés ; aucune synthèse n'a été rejetée par le code) : 6 en dev, 9 sur le test scellé élargi.

| Jeu | Synthèses | Phrases | Impliquées | Neutres | Contradictions |
|---|---|---|---|---|---|
| Développement | 24 | 25 | 19 (0,760) | 6 (0,240) | 0 (0,000) |
| Test scellé élargi | 27 | 33 | 26 (0,788) | 7 (0,212) | 0 (0,000) |

Aucune phrase n'est jugée contradictoire ; environ une phrase sur quatre ou cinq n'est pas jugée impliquée par les unités qu'elle cite. La liste complète de ces phrases, avec leurs unités, figure dans le JSON. Relues une à une par l'assistant de développement (une relecture, non une évaluation humaine indépendante ; `rag_nli_inspection.json`), les 13 phrases non impliquées se répartissent en 9 reformulations ou synthèses fidèles, 4 extrapolations légères et 0 affirmation non soutenue. Les 4 extrapolations ajoutent un élément absent des unités citées, tiré de la question ou d'une autre unité du projet, par exemple « avant la bascule » là où la source donne l'échéance du 10 juillet. Limite : un modèle NLI est imparfait, surtout en français et sur des phrases de synthèse ; ce contrôle complète le contrôle des citations et ne remplace pas une évaluation humaine, non réalisée.

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

### Seconde passe de liaison des relations : résultat négatif

Fichier `relations_v2.json`, produit par `eval/eval_relations_v2.py` ; module `src/ingest/link.py`. Cette passe part des unités et relations de l'extraction figée, rejouées depuis le cache (0 appel) ; la consigne d'extraction, `extraction.json` et `relation_errors_dev.json` ne sont pas modifiés. Pour chaque unité extraite, les candidats sont les k unités antérieures les plus proches du même projet (cosinus e5 fine-tuné), plus les paires déjà reliées par l'extraction. Un appel Groq par projet (gpt-oss-120b, sortie JSON, température 0, consigne séparée et versionnée) attribue à chaque paire candidate supersedes, contradicts, creates_action, answers, supports ou aucun, avec son sens ; le code rejette tout numéro de paire, type ou sens invalide.

Le critère a été fixé avant les résultats : sur le dev 04-10, F1 strict des relations d'au moins 0,456 (0,406 + 0,05), sans que la précision stricte descende sous 0,45. Plafond : 73 des 84 relations du dev ont leurs deux extrémités extraites, donc le rappel strict ne peut pas dépasser 0,869 (sur le test scellé, 23 relations sur 23).

Toutes les variantes essayées sur le dev. Seul k = 3 a été exécuté (un appel par projet et par consigne) : les lignes k = 0 à 2 sont indicatives, obtenues en filtrant a posteriori la réponse k = 3, et ne pouvaient pas être retenues. Les modes de fusion sont appliqués par le code à la sortie d'un même appel : remplacement (seules les relations de la passe), union (extraction + passe) et complement (extraction + passe sur les seules paires que l'extraction n'a pas reliées).

| Consigne | k | Fusion | Nature | Relations prédites | Précision | Rappel | F1 |
|---|---|---|---|---|---|---|---|
| v1 | 0 | remplacement | indicative | 28 | 0,429 | 0,143 | 0,214 |
| v1 | 0 | union | indicative | 59 | 0,491 | 0,345 | 0,406 |
| v1 | 0 | complement | indicative | 54 | 0,518 | 0,333 | 0,406 |
| v1 | 1 | remplacement | indicative | 42 | 0,429 | 0,214 | 0,286 |
| v1 | 1 | union | indicative | 73 | 0,479 | 0,417 | 0,446 |
| v1 | 1 | complement | indicative | 68 | 0,500 | 0,405 | 0,447 |
| v1 | 2 | remplacement | indicative | 54 | 0,333 | 0,214 | 0,261 |
| v1 | 2 | union | indicative | 85 | 0,412 | 0,417 | 0,414 |
| v1 | 2 | complement | indicative | 80 | 0,425 | 0,405 | 0,415 |
| v1 | 3 | remplacement | exécutée | 60 | 0,300 | 0,214 | 0,250 |
| v1 | 3 | union | exécutée | 91 | 0,385 | 0,417 | 0,400 |
| v1 | 3 | complement | exécutée | 86 | 0,395 | 0,405 | 0,400 |
| v2 | 0 | remplacement | indicative | 0 | 0,000 | 0,000 | 0,000 |
| v2 | 0 | union | indicative | 54 | 0,518 | 0,333 | 0,406 |
| v2 | 0 | complement | indicative | 54 | 0,518 | 0,333 | 0,406 |
| v2 | 1 | remplacement | indicative | 2 | 0,000 | 0,000 | 0,000 |
| v2 | 1 | union | indicative | 56 | 0,500 | 0,333 | 0,400 |
| v2 | 1 | complement | indicative | 56 | 0,500 | 0,333 | 0,400 |
| v2 | 2 | remplacement | indicative | 4 | 0,250 | 0,012 | 0,023 |
| v2 | 2 | union | indicative | 58 | 0,500 | 0,345 | 0,408 |
| v2 | 2 | complement | indicative | 58 | 0,500 | 0,345 | 0,408 |
| v2 | 3 | remplacement | exécutée | 6 | 0,167 | 0,012 | 0,022 |
| v2 | 3 | union | exécutée | 60 | 0,483 | 0,345 | 0,403 |
| v2 | 3 | complement | exécutée | 60 | 0,483 | 0,345 | 0,403 |

Conditions de ce choix : la consigne v2 est la seule et dernière révision, écrite après l'analyse des erreurs de la v1 sur le dev ; le mode complement a lui aussi été introduit après l'analyse des résultats de la v1. Le choix s'est fait parmi 6 configurations exécutées sur le dev : le score dev de la configuration retenue est donc optimiste, et seul le test scellé, exécuté une fois, sert de mesure non biaisée.

La v1 relie trop : avec la fusion la plus favorable (complement), son rappel monte à 0,405 mais sa précision tombe à 0,395 (F1 0,400). La v2, plus prudente, ne propose plus que 6 relations sur le dev (5 supersedes, 1 answers) et répond « aucun » aux 136 paires du test scellé ; 13 de ses réponses sur le projet 07 ont été rejetées par le code pour un sens invalide. Configuration retenue sur le dev : v2, k = 3, union (à égalité avec complement).

| Mesure | Dev avant | Dev après | Test avant | Test après |
|---|---|---|---|---|
| Relations prédites | 54 | 60 | 19 | 19 |
| Relations justes | 28 | 29 | 8 | 8 |
| Précision stricte | 0,518 | 0,483 | 0,421 | 0,421 |
| Rappel strict | 0,333 | 0,345 | 0,348 | 0,348 |
| F1 strict | 0,406 | 0,403 | 0,381 | 0,381 |

| Relation | F1 dev avant | F1 dev après | F1 test avant | F1 test après |
|---|---|---|---|---|
| answers | 0,710 | 0,688 | 0,444 | 0,444 |
| creates_action | 0,279 | 0,279 | 0,200 | 0,200 |
| supports | 0,353 | 0,353 | 0,455 | 0,455 |
| supersedes | 0,364 | 0,375 | 0,000 | 0,000 |
| contradicts | 0,000 | 0,000 | sans objet | sans objet |

Le critère n'est pas atteint : rien n'est activé, et la mémoire extraite garde les relations de l'extraction. Avec les mêmes catégories que l'analyse précédente, sur le dev, la configuration retenue compte 18 relations prédites dont une extrémité n'est pas annotée, 9 mauvaises cibles et 4 mauvais types ; parmi les relations annotées manquées, 25 ne sont jamais proposées (27 avant). Le lot a consommé 71 388 tokens Groq en 16 appels, sous le budget de 100 000 : 33 956 pour la v1 sur le dev, 29 402 pour la v2 sur le dev et 8 030 pour le test scellé.

### Chaîne complète en mode extrait

Fichier `agent_extracted_dev.json`, produit par `eval/eval_agent_extracted.py`. Le même agent interroge une autre mémoire : les unités et relations produites par le pipeline d'extraction pour les projets 04 à 12, sans correction manuelle (`data/extracted`, reconstruite depuis le cache Groq local par `scripts/build_extracted_memory.py`, sans aucun appel). Seul le chargeur de mémoire change ; l'interface propose ce mode à côté de la mémoire annotée, qui reste le mode par défaut. La mesure porte sur les 20 requêtes avec réponse du dev qui concernent les projets 04 à 10, sans aucun réglage ; une réponse en mémoire extraite est juste si l'unité affichée est appariée (même appariement que pour l'extraction) à une unité attendue.

| Catégorie | Mémoire extraite | Mémoire annotée |
|---|---|---|
| Global | 10/20 (0,500) | 13/20 (0,650) |
| decision_active | 2/4 | 4/4 |
| reformulation | 2/5 | 1/5 |
| responsable_reel | 1/4 | 2/4 |
| negation | 3/3 | 3/3 |
| engagement | 2/4 | 3/4 |

Les 20 unités attendues ont toutes un équivalent dans la mémoire extraite : l'écart vient du classement et de la résolution, pas d'unités manquantes. 9 requêtes sont justes dans les deux mémoires, 4 seulement en mémoire annotée et 1 seulement en mémoire extraite. Les décisions en vigueur reculent le plus (2/4 contre 4/4) : la résolution s'appuie sur les relations, dont le F1 strict n'est que de 0,40 sur les projets 04 à 12. L'agent s'abstient à tort 2 fois en mémoire extraite (1 fois en mémoire annotée) et ne s'abstient sur aucune des 5 requêtes sans réponse de ces projets, quelle que soit la mémoire.

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

### Test scellé élargi (projets 13 à 16)

Fichier `test2.json`, produit par `eval/eval_test2.py`. Quatre projets fictifs de domaines nouveaux (déménagement du siège, flotte de véhicules, rappel d'un lot de produits, rapport annuel), 3 réunions chacun, soit 12 réunions, 102 unités et 61 relations (`data/test2`), et 40 requêtes (`data/eval/queries_test2.json`) : 20 simples (4 par catégorie), 8 sans réponse, 8 questions complexes dont la réponse demande plusieurs unités, 4 demandes de dossier avec la liste attendue des unités. Ce jeu a été rédigé par l'assistant de développement avant toute exécution, et gelé avec l'empreinte SHA-256 de chaque fichier (`data/eval/test2_manifest.json`) ; il n'a jamais servi à un réglage. Il a été rédigé après lecture des interfaces de l'agent, ce qui peut le rendre plus favorable au système qu'un jeu rédigé par un tiers. Chaque mesure provient d'une seule exécution, avec la configuration actuelle et sans réentraîner le modèle.

Recherche, 28 requêtes avec réponse (simples et complexes) :

| Système | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| BM25 | 0,745 | 0,857 | 0,749 | 0,754 |
| e5-small de base | 0,714 | 0,834 | 0,663 | 0,689 |
| e5-small fine-tuné | 0,760 | 0,825 | 0,739 | 0,738 |
| Hybride RRF (agent) | 0,860 | 0,896 | 0,774 | 0,793 |

Réponse de l'agent :

| Mesure | Extractive | RAG contrôlé |
|---|---|---|
| Requêtes simples, réponse juste | 16/20 | 20/20 |
| Requêtes sans réponse, abstention | 1/8 | 7/8 |
| Questions complexes, unités attendues affichées | 7/24 | 13/24 |
| Questions complexes entièrement couvertes | 1/8 | 2/8 |

Sur ce jeu, BM25 dépasse e5-small fine-tuné en MRR (0,749 contre 0,739) ; l'hybride reste le meilleur des quatre. La réponse extractive n'affiche qu'une unité : elle couvre mal les questions complexes, que le RAG contrôlé couvre mieux sans les couvrir entièrement. Les 27 synthèses du RAG ont toutes des citations valides ; il a consommé 21 681 tokens Groq (33 appels, cache local non versionné).

Dossiers : 20 des 24 unités attendues sont citées, et aucune unité citée n'est sans source dans son transcript. Les trois dossiers qui répondent citent toutes leurs unités attendues ; le quatrième (« Prépare le dossier sur le choix du loueur ») s'abstient, son sujet passant sous le seuil de similarité.

Extraction figée, sur les 4 projets. La première exécution a été interrompue par une erreur d'appel Groq sur la première réunion du projet 15, après les projets 13 et 14 complets ; elle a été reprise ensuite sur les seuls projets 15 et 16, sans modifier ni la consigne ni le jeu (une première reprise a complété le projet 15 puis échoué deux fois sur le projet 16, une seconde a complété le projet 16). Après l'échec sur p15_m01, cette première exécution a tout de même obtenu une réponse pour p15_m02, construite sans les unités de p15_m01 ; cette réponse reste dans le cache mais n'est pas utilisée. Les 12 réponses utilisées ont été obtenues avec la consigne gelée (empreinte SHA-256 3f757d54) et les paramètres d'origine, ce que vérifie leur rejeu depuis le cache, sans aucun appel. Les 4 projets sont enregistrés depuis le cache. La seconde passe de liaison n'a pas été lancée sur ce jeu.

| Mesure | Projets 13 à 16 |
|---|---|
| Unités annotées, extraites, appariées | 102, 107, 100 |
| Précision des unités | 0,935 |
| Rappel des unités | 0,980 |
| F1 des unités | 0,957 |
| Type exact | 92/100 (0,920) |
| Relations annotées, extraites | 61, 33 |
| Relations strictes, précision | 0,697 |
| Relations strictes, rappel | 0,377 |
| Relations strictes, F1 | 0,489 |

L'extraction a consommé 56 435 tokens Groq sur les trois exécutions (13 appels réussis). Ces scores, plus élevés qu'au dev, portent sur des transcripts plus courts que ceux du corpus de départ.

### Intervalles de confiance et significativité

Fichier `stats.json`, produit par `eval/stats.py` : intervalle de confiance à 95 % par bootstrap (10 000 tirages, graine fixe), par requête pour les métriques par requête, par projet pour les F1 d'extraction ; tests appariés (intervalle bootstrap de la différence et test de permutation bilatéral). Un écart est dit significatif si p < 0,05 et si l'intervalle de la différence exclut 0. Les valeurs par requête de la recherche en dev et en test sont recalculées par le même code, qui vérifie qu'elles redonnent exactement les moyennes publiées.

| Mesure | Dev | Test scellé | Test scellé élargi |
|---|---|---|---|
| MRR e5-small fine-tuné | 0,716 [0,586 ; 0,841] | 0,750 [0,450 ; 1,000] | 0,739 [0,596 ; 0,875] |
| MRR hybride RRF | 0,708 [0,581 ; 0,828] | 0,850 [0,550 ; 1,000] | 0,774 [0,649 ; 0,887] |
| Réponse finale extractive | 0,680 [0,480 ; 0,840] | 0,800 [0,400 ; 1,000] | 0,800 [0,600 ; 0,950] |
| Réponse finale RAG contrôlé | 0,960 [0,880 ; 1,000] | 1,000 [1,000 ; 1,000] | 1,000 [1,000 ; 1,000] |
| Abstention RAG contrôlé | 0,800 [0,400 ; 1,000] | 1,000 [1,000 ; 1,000] | 0,875 [0,625 ; 1,000] |
| F1 des unités extraites | 0,852 [0,825 ; 0,877] | 0,829 [0,800 ; 0,857] | 0,957 [0,941 ; 0,972] |
| F1 strict des relations | 0,406 [0,338 ; 0,467] | 0,381 [0,250 ; 0,556] | 0,489 [0,468 ; 0,511] |

| Comparaison | Dev | Test scellé | Test scellé élargi |
|---|---|---|---|
| e5 fine-tuné contre e5 de base (MRR) | +0,190 [+0,047 ; +0,342], p = 0,019, significatif | +0,110 [+0,000 ; +0,310], p = 0,505 | +0,076 [-0,008 ; +0,167], p = 0,163 |
| hybride contre dense fine-tuné (MRR) | -0,008 [-0,099 ; +0,082], p = 0,928 | +0,100 [+0,000 ; +0,300], p = 1,000 | +0,035 [-0,043 ; +0,113], p = 0,478 |
| RAG contre extractif (réponse juste) | +0,280 [+0,120 ; +0,480], p = 0,015, significatif | +0,200 [+0,000 ; +0,600], p = 1,000 | +0,200 [+0,050 ; +0,400], p = 0,126 |
| RAG contre extractif (abstention) | +0,800 [+0,400 ; +1,000], p = 0,128 | +1,000 [+1,000 ; +1,000], p = 1,000 | +0,750 [+0,375 ; +1,000], p = 0,032, significatif |
| relations v2 contre extraction seule (F1 strict) | -0,003 [-0,015 ; +0,000], p = 1,000 | +0,000 [+0,000 ; +0,000], p = 1,000 | non mesuré |

Ce qui est significatif : le gain du fine-tuning et celui du RAG sur la réponse juste, en dev seulement (le dev a servi au réglage), et le gain du RAG sur l'abstention, sur le test scellé élargi seulement. Ce qui ne l'est pas : l'hybride contre le dense seul, partout ; tous les écarts du test scellé, qui ne compte que 5 requêtes avec réponse et 1 sans réponse, si bien qu'aucun test de permutation n'y peut conclure ; la seconde passe de liaison. Sur le test scellé élargi, le fine-tuning et le RAG vont dans le même sens qu'en dev sans que l'écart soit établi. Les intervalles par projet du test scellé (2 projets) et du test scellé élargi (4 projets) sont trop grossiers pour être interprétés. En test scellé, le BM25 n'avait jamais été mesuré : il est recalculé ici (MRR de 0,707).

## Limites assumées

- Relations : c'est la partie la plus faible, avec un F1 strict de 0,406 en dev et 0,381 en test. L'analyse des erreurs du dev 04-10 (`eval/results/relation_errors_dev.json`) montre qu'aucune relation prédite n'est fausse à cause de son seul sens. Sur les 54 relations prédites, 28 sont justes ; parmi les erreurs, 16 relient une unité absente de l'annotation et 7 visent la mauvaise cible. Sur les 84 relations annotées, 56 sont manquées, dont 27 jamais proposées (15 supports). Une seconde passe de liaison par LLM n'a pas amélioré ce score (F1 strict de 0,403 en dev et 0,381 en test scellé, `relations_v2.json`) : ce résultat négatif est décrit plus haut.
- Abstention : l'agent s'abstient quand le meilleur cosinus dense passe sous 0,45 (seuil fixé dans `src/agent/answer.py`). Ce seuil écarte bien les questions hors sujet, mais il ne sépare pas une question sur le bon sujet dont la réponse est absente des sources : dans ce cas, la réponse extractive est le plus souvent fausse. C'est mesuré sans RAG : 0 abstention sur les 5 requêtes sans réponse du dev, 0 sur 1 en test scellé ; sur 6 sondes exploratoires, 0 abstention, 5 réponses à tort et 1 acceptable sur le fond (`agent_dev.json`, `agent_test.json`). Ces sondes ne sont pas une mesure indépendante : 5 sur 6 sont les requêtes sans_reponse du dev. À l'inverse, le seuil écarte à tort q_dev_002, dont la réponse existe (meilleur cosinus de 0,440). Le RAG contrôlé, activé par défaut, corrige en grande partie ce cas (4 abstentions sur 5 en dev, 1 sur 1 en test scellé, 0 réponse à tort sur les 6 sondes), mais ces mesures restent petites et non indépendantes, et le seuil de similarité écarte toujours q_dev_002 avant tout appel.
- Recul sur les décisions en vigueur : en dev, le fine-tuning fait baisser le MRR de 0,800 à 0,707 et le Recall@3 de 1,000 à 0,800 sur cette catégorie de requêtes. La résolution par les relations de l'agent compense ce cas dans la réponse finale : en dev, les 5 requêtes de cette catégorie obtiennent une réponse juste (`agent_dev.json`), ce qui reste une mesure sur 5 requêtes.
- Reformulation : c'est la catégorie la plus faible en réponse finale, avec 1/5 en dev. L'agent ne retient que le premier résultat de la recherche : une unité pertinente classée deuxième ou troisième ne sert à rien. Ce chiffre vaut pour la réponse extractive ; avec le RAG contrôlé, qui lit les 5 premières unités, la catégorie passe à 4/5.
- Réponse finale extractive : en dev, 8 réponses sur 25 sont fausses (9 si le conflit non tranché n'est pas compté juste), dont 4 sur 5 en reformulation. La résolution ne corrige que les cas où la recherche tombe sur une unité périmée ; elle ne rattrape pas une recherche qui tombe sur une autre unité active.
- Conflit non tranché : le corpus de développement n'en contient qu'un seul cas (projet_10). Le comportement est testé sur ce cas, pas mesuré à plus grande échelle, et il dépend d'une relation contradicts annotée entre les deux unités.
- Recherche hybride : en dev, l'hybride est légèrement sous le e5 fine-tuné seul (MRR de 0,708 contre 0,716, et de même en nDCG@5 et en Recall@5), bien qu'il soit meilleur en Recall@3 (0,920 contre 0,840).
- RAG contrôlé : il dépend d'un service externe (Groq) et d'un modèle qui n'est pas strictement déterministe même à température nulle ; le cache versionné rend la démonstration rejouable, pas le modèle, et sans réseau ni clé l'agent se replie sur la réponse extractive. Les contrôles vérifient que chaque phrase cite une unité fournie, pas que la phrase dit exactement ce que dit l'unité : la preuve extractive affichée sous la synthèse reste la référence. Les critères ont été mesurés sur les 30 requêtes du dev, qui n'est pas vierge.
- Surextraction : sur les projets 04 à 12, l'extraction produit 19 unités idea pour 4 annotées.
- Confusion entre action et engagement : en dev, 4 engagements annotés sont extraits comme actions et 2 actions comme engagements ; sur les projets 04 à 12, le rappel des actions n'est que de 0,294. Une partie de ces écarts vient d'un désaccord de convention entre la consigne et l'annotation.
- Jeux d'évaluation petits : en dev, 25 requêtes évaluées pour la recherche ; en test scellé, 5 requêtes évaluées pour la recherche et 34 unités appariées pour l'extraction. Les écarts sont indicatifs, pas statistiquement solides.
- Dev non vierge : il a servi au réglage de la génération des triplets (voir plus haut).
- Projets 09 et 10 peu couverts : seuls 4 triplets construits à la main en proviennent, aucun triplet enrichi.
- Règle du responsable : la personne qui doit réaliser une tâche est le premier participant nommé dans l'énoncé de l'unité. Cette heuristique est vérifiée sur le corpus, mais elle n'est pas généralisable telle quelle.
- Routage et brouillon d'email : la détection d'intention et la préparation des brouillons sont couvertes par des tests automatiques sur des exemples des projets de développement (`tests/test_agent_logic.py`), mais ne font encore l'objet d'aucune mesure chiffrée.
- Ingestion et agent reliés en partie seulement : le parcours de démonstration, les mesures de l'agent et l'évaluation de la recherche lisent la mémoire annotée (`data/raw`). La mémoire extraite n'existe que pour les projets 04 à 12, n'est mesurée que sur 20 requêtes, et la réponse finale y est moins souvent juste (10/20 contre 13/20 sur les mêmes requêtes). Avec des relations à 0,40 de F1 strict, la résolution des décisions remplacées y est nettement moins fiable.
- Test scellé élargi : rédigé par l'assistant de développement après lecture des interfaces de l'agent, il peut être plus favorable au système qu'un jeu rédigé par un tiers .
- Audit qualitatif sur AMI partiel : 6 segments sur 23 traités, sans valeur statistique (voir plus haut).
- Corpus fictif : les 12 projets ont été écrits pour le projet. Sur QMSum, la majorité des réunions Product (AMI) sont elles aussi jouées selon un scénario.

## Démonstration

La démonstration porte sur le projet fictif Atlas (projet_01), dont trois comptes rendus successifs racontent une mise en production d'abord proposée puis validée pour le 14 avril, puis reportée au 5 mai. Elle s'appuie sur les unités et relations annotées du projet Atlas, et non sur celles de l'extraction automatique (mode par défaut de l'interface). Le parcours se déroule en sept temps :

1. retrouver la décision en vigueur, avec sa preuve et sa source ;
2. identifier l'action qui en découle, avec son responsable et son échéance ;
3. montrer la décision remplacée et le chemin suivi pour arriver à l'information à jour ;
4. afficher à l'écran les passages sources du transcript, pour la décision en vigueur et pour celles qu'elle remplace ;
5. déclencher l'abstention sur une question hors des sources ;
6. préparer un brouillon d'email à la personne chargée d'une action (Claire), à confirmer et jamais envoyé ;
7. constituer le dossier documenté de la mise en production : décisions en vigueur, historique des changements, conflits non tranchés, actions et engagements (responsable, échéance, statut), risques, questions ouvertes et sources citées, téléchargeable en Markdown.

Le dossier est constitué de façon déterministe, sans LLM : les unités du sujet sont retrouvées par la recherche hybride (ou toutes les unités du projet si la demande le nomme), puis complétées en suivant les relations du graphe sur deux liens. Chaque élément est une unité de la mémoire, citée avec sa réunion, sa date et sa preuve.

Commandes :

    uv run streamlit run app.py --server.address localhost
    uv run python scripts/verify_demo.py

Le second script vérifie automatiquement sept points du parcours (remplacement de décision reproductible, affichage des sources, abstention, responsable et échéance de l'action, libellés d'affichage : type en français, proposition non validée, échéance « le jour même » datée, brouillon d'email non envoyé, contenu du dossier). Une vidéo de la démonstration peut être enregistrée avec `scripts/record_demo.py` ; elle n'est pas versionnée.

## Audit qualitatif sur AMI

Un audit exploratoire applique l'extraction figée à trois réunions AMI du split test de QMSum, découpées en segments de 40 tours de parole. Il est partiel : 6 segments sur 23 ont été traités avant que le quota quotidien de Groq soit atteint. Il ne constitue pas une validation statistique. Son script (`eval/audit_real_extraction.py`) est versionné ; ses résultats ne le sont pas, puisque l'audit n'est pas terminé.

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
    uv run python eval/eval_hybrid.py --out eval/results/retrieval_dev.json

Comparaison d'architectures (entraîne les trois autres modèles dans `models/arch/`, puis les évalue ; environ 30 minutes sur CPU, dont 24 pour e5-base) :

    uv run python scripts/train_architectures.py e5_small_sans_negatifs_durs minilm_l12 e5_base
    uv run python eval/eval_architectures.py --out eval/results/architectures_dev.json
    uv run python eval/eval_architectures.py --queries data/eval/queries_test.json --out eval/results/architectures_test.json

Réponse finale de l'agent et sondes d'abstention (réécrit `eval/results/agent_dev.json`) :

    uv run python eval/eval_agent.py --out eval/results/agent_dev.json

RAG contrôlé (appels Groq mis en cache dans `data/processed/groq_rag_cache.jsonl`, versionné ; `--cache-only` rejoue sans réseau), puis préchauffage du cache pour la démonstration :

    uv run python eval/eval_rag.py --out eval/results/rag_dev.json --cache-only
    uv run python scripts/prewarm_rag.py

Chaîne complète en mode extrait (mémoire reconstruite depuis le cache local, puis mesure) :

    uv run python scripts/build_extracted_memory.py
    uv run python eval/eval_agent_extracted.py --out eval/results/agent_extracted_dev.json

Extraction (se rejoue depuis le cache Groq local, sans aucun appel) :

    uv run python eval/eval_extraction.py --projects 4-12 --groups dev_04_10=4-10 test_11_12=11-12 --cache-only

Analyse des erreurs de relations sur le dev 04-10, depuis le même cache local (réécrit `eval/results/relation_errors_dev.json`) :

    uv run python eval/analyze_relation_errors.py

Seconde passe de liaison des relations (rejoue l'extraction depuis le même cache local ; les réponses de liaison sont dans `data/processed/groq_link_cache.jsonl`, non versionné ; `--cache-only` rejoue sans réseau, l'option `--split test` refuse de relancer un test déjà enregistré) :

    uv run python eval/eval_relations_v2.py --split dev --cache-only

Test scellé élargi : validation du jeu gelé, puis une exécution d'enregistrement par étape (le script vérifie les empreintes du manifeste et refuse de relancer une étape déjà enregistrée ; `--cache-only` rejoue une étape depuis les caches locaux, sans réseau) :

    uv run python scripts/validate_corpus.py data/test2
    uv run python eval/eval_test2.py --stage offline --cache-only
    uv run python eval/eval_test2.py --stage rag --cache-only

Intervalles de confiance et tests appariés (réécrit `eval/results/stats.json`) :

    uv run python eval/stats.py

Contrôle automatique de fidélité du RAG (télécharge une fois le modèle NLI, environ 550 Mo ; lit les caches du RAG, aucun appel Groq) :

    uv run python eval/nli_faithfulness.py

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

## Licence

Tous droits réservés (voir [LICENSE](LICENSE)). Le dépôt est public uniquement pour l'évaluation du projet ; sa consultation est autorisée, mais toute autre utilisation, copie, modification, redistribution ou usage commercial du code, des données ou de la documentation demande l'autorisation écrite préalable de l'autrice. Les bibliothèques tierces restent soumises à leurs propres licences. Les données tierces (QMSum) ne sont pas redistribuées dans le dépôt : `eval/results/real_qmsum.json` ne contient que des identifiants de réunions et des scores, et le jeu se clone à part (voir `data/external/README.md`). Les 16 projets du corpus (12 dans `data/raw`, 4 dans `data/test2`) sont des jeux fictifs écrits pour ce projet.

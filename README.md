# EMBEDDING LAB — De la mémoire sémantique à l'agent intelligent

Moteur de compréhension sémantique pour un agent qui lit des comptes rendus de réunions, retrouve la bonne information en citant sa preuve, distingue une décision en vigueur d'une décision périmée, et prépare une action sans jamais inventer.

## Contexte

Une recherche vectorielle généraliste rapproche des phrases qui se ressemblent mais disent le contraire : « le budget est approuvé » et « le budget n'est pas approuvé » sont presque identiques pour un modèle d'embeddings standard. Ce projet construit le moteur de recherche d'un agent capable de lever cette ambiguïté : comprendre le statut d'une information, suivre l'évolution d'une décision, identifier la bonne personne, et dire « je ne sais pas » quand l'information est absente.

## Architecture

    Question de l'utilisateur
     -> détection d'intention (règles) : répondre, ou rédiger un email
     -> recherche hybride dans le projet : BM25 + embeddings fine-tunés, fusionnés par Reciprocal Rank Fusion
     -> résolution du statut : si l'unité trouvée est périmée, on suit les relations supersedes / contradicts jusqu'à l'unité active
     -> sortie : réponse sourcée, ou abstention ; brouillon d'email à confirmer, ou demande de clarification
        (destinataire ambigu ou introuvable)

Le LLM n'intervient pas dans la recherche. Il sert uniquement, en amont, à générer des reformulations de questions pour l'entraînement. La recherche dense est évaluée de façon chiffrée ; les relations et le routage sont vérifiés sur des scénarios de démonstration.

## Installation

Environnement géré avec uv (Python 3.12).

    uv venv --python 3.12
    uv pip install -r requirements.txt
    uv run pytest -q

Le modèle fine-tuné n'est pas versionné (`models/` est ignoré par git). Il faut l'entraîner une fois avant de lancer l'agent, l'interface ou la comparaison avant/après :

    uv run python scripts/train_embeddings.py        # produit models/e5-finetuned

Le premier lancement télécharge le modèle de base `intfloat/multilingual-e5-small` depuis Hugging Face.

Une clé Groq (fichier .env à la racine, GROQ_API_KEY=...) n'est nécessaire que pour régénérer les triplets d'entraînement. Elle n'est pas requise pour entraîner le modèle, faire tourner l'agent ni les évaluations.

## Structure du projet

    src/schemas.py            contrats de données (unité sémantique, relation, brouillon d'email)
    src/memory/corpus.py      chargement du corpus, des relations et des participants
    src/retrieval/            baselines BM25 et dense (e5)
    src/agent/                recherche hybride, réponse sourcée, intentions, email, clarification
    eval/                     métriques et scripts de comparaison
    scripts/                  validation du corpus, construction des triplets, entraînement, démo CLI
    data/raw/                 les 12 projets annotés
    data/eval/                jeux de requêtes (dev et test scellé)
    data/processed/           triplets d'entraînement
    app.py                    interface de démonstration Streamlit

## Données

Aucun jeu de données n'était fourni. J'ai construit un corpus de 12 projets fictifs en français (réunions d'entreprise), soit 225 unités sémantiques et 146 relations, en injectant volontairement les cas difficiles demandés : proposition contre décision, changement de date ou de responsable, personne citée contre personne réellement responsable, information périmée, contradiction, question sans réponse.

Séparation anti-fuite : le découpage se fait par projet, pas par phrase. Les projets 01 à 10 servent au développement (30 requêtes dev), les projets 11 et 12 forment un jeu de test scellé (6 requêtes), jamais vu à l'entraînement : aucun triplet ne provient de ces deux projets.

Triplets d'entraînement : 20 construits à la main (pièges supersedes et contradicts) et 351 enrichis. Pour ces derniers, le modèle Groq reformule la question (l'ancre), le positif reste l'énoncé réel du corpus, le négatif est miné dans le même projet. Les réponses du LLM sont mises en cache localement et le fichier de triplets est versionné : le jeu d'entraînement est figé et reproductible.

## Évaluation

Métriques : Recall@K, MRR, nDCG@K, calculées de façon identique pour tous les modèles.

Baselines sur le dev (25 requêtes, hors sans_reponse) : BM25 fait mieux que le modèle dense généraliste e5-small, qui échoue surtout sur la négation et les engagements.

Après fine-tuning contrastif (MultipleNegativesRankingLoss) :

    Dev (n=25)        Recall@3   Recall@5   MRR     nDCG@5
    e5 de base          0.680      0.880   0.526    0.614
    e5 fine-tuné        0.840      1.000   0.716    0.786
    BM25 (repère)       0.800      0.880   0.631    0.695

Le modèle fine-tuné passe devant BM25 sur les quatre métriques. Les deux faiblesses ciblées progressent le plus : sur la négation, le MRR passe de 0.207 à 0.507 ; sur les engagements, de 0.550 à 1.000. En contrepartie, les décisions en vigueur reculent légèrement (MRR de 0.800 à 0.707) ; l'étape de résolution du statut de l'agent est conçue pour compenser ce cas, sans que ce soit encore mesuré.

Test scellé (projets jamais vus, n=5) : le modèle fine-tuné fait mieux que la base sur les quatre métriques (MRR de 0.640 à 0.750). C'est un signal encourageant de généralisation, mais très fragile : l'essentiel du gain vient d'une seule requête (decision_active), et trois des cinq requêtes étaient déjà parfaites avant fine-tuning.

## Utilisation

    uv run python eval/run_baselines.py                                             baselines chiffrées
    uv run python eval/compare_finetuned.py                                         comparaison avant/après (dev)
    uv run python eval/compare_finetuned.py --queries data/eval/queries_test.json   test scellé
    uv run python scripts/demo_agent.py "Quand a lieu la mise en production d'Atlas ?" projet_01
    uv run streamlit run app.py --server.address localhost                          interface de démonstration

## Limites

- Les jeux d'évaluation sont petits (25 requêtes en dev, 5 en test). Les écarts sont indicatifs, pas statistiquement solides.
- Le dev a servi à régler la génération, il n'est donc plus vierge. Seul le test scellé donne une mesure non biaisée.
- Les chiffres ci-dessus mesurent le modèle dense seul. La recherche hybride (BM25 + dense, RRF), la résolution par les relations et le routage ne sont pas encore évalués de façon chiffrée.
- Les projets 09 et 10 ne sont pas couverts par les triplets enrichis (quota de génération atteint) : le modèle n'en a vu que 4 triplets construits à la main.
- L'abstention repose sur un seuil de similarité qui écarte de façon fiable les questions hors sujet, mais ne distingue pas une question sur le bon sujet dont la réponse est absente. Le cosinus mesure le sujet, pas la présence de la réponse. La solution propre est un cross-encoder ou un modèle d'inférence textuelle, identifiée comme prochaine étape.
- Le responsable d'une tâche est identifié par une règle simple (le premier participant nommé dans l'énoncé), vérifiée juste sur le corpus mais non généralisable telle quelle.

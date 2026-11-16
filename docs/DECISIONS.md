# Décisions techniques

Chaque choix majeur, l'alternative écartée, la raison et la preuve chiffrée. Les chiffres viennent des fichiers de `eval/results` cités ; le dev (projets 01-10) a servi au réglage, seuls le test scellé (11-12) et le test scellé élargi (13-16) sont indépendants, et tous ces jeux sont petits (voir les intervalles de confiance de `stats.json`).

## 1. Fine-tuner e5-small plutôt que l'utiliser tel quel

- Alternative écartée : e5-small de base, ou BM25 seul.
- Raison : le modèle de base classe mal les négations et les engagements du corpus ; un fine-tuning contrastif sur des triplets du dev corrige ces cas sans données du test.
- Preuve : MRR dev de 0,526 (base) à 0,716 (fine-tuné), devant BM25 (0,631) ; MRR test scellé de 0,640 à 0,750 (`retrieval_dev.json`, `retrieval_test.json`).

## 2. Garder e5-small plutôt qu'un modèle plus gros

- Alternative écartée : e5-base et paraphrase-multilingual-MiniLM-L12, entraînés avec le même protocole.
- Raison : règle fixée avant les résultats, appliquée au seul dev ; aucun candidat ne bat le MRR dev de e5-small.
- Preuve : MRR dev de 0,716 (e5-small), 0,693 (e5-base) et 0,611 (MiniLM-L12) ; e5-base coûte 2,4 fois plus de paramètres (`architectures_dev.json`).

## 3. Miner des négatifs difficiles dans le même projet

- Alternative écartée : paires (ancre, positif) seules, les négatifs étant les autres exemples du lot.
- Preuve : sans négatifs difficiles, le MRR dev de e5-small fine-tuné passe de 0,716 à 0,689 (`architectures_dev.json`). Gain modeste, sur 25 requêtes.

## 4. Recherche hybride BM25 + e5 fusionnée par RRF

- Alternative écartée : e5 fine-tuné seul.
- Raison : la fusion remonte plus souvent une unité pertinente dans les trois premiers résultats, que lit le RAG contrôlé.
- Preuve : Recall@3 dev de 0,920 contre 0,840 pour e5 fine-tuné seul ; en revanche, MRR dev de 0,708 contre 0,716 (`retrieval_dev.json`). Le gain net n'est pas établi sur ce corpus (voir `stats.json`).

## 5. Abstention sur un seuil de similarité

- Alternative écartée : répondre toujours avec la meilleure unité, ou un classifieur NLI (non construit).
- Raison : le seuil (0,45, `src/agent/answer.py`) écarte de façon fiable les questions hors sujet, sans modèle supplémentaire.
- Limite chiffrée : il ne sépare pas une question sur le bon sujet dont la réponse manque ; sans RAG, 0 abstention sur les 5 requêtes sans réponse du dev (`agent_dev.json`).

## 6. Résoudre par les relations plutôt que par le seul classement

- Alternative écartée : renvoyer l'unité la mieux classée, même périmée.
- Raison : une décision remplacée reste sémantiquement proche de la question ; seul le graphe supersedes / contradicts dit laquelle est en vigueur.
- Preuve : en dev, 3 réponses passent par une résolution, toutes justes, et les 5 requêtes sur les décisions en vigueur sont justes (`agent_dev.json`).

## 7. Ne jamais trancher un conflit entre deux unités actives

- Alternative écartée : retenir l'affirmation la plus récente.
- Raison : deux affirmations actives contradictoires sans remplacement explicite ne permettent pas de conclure ; l'agent montre les deux, avec leurs sources.
- Preuve : un seul cas au dev (projet 10), testé et non mesuré à grande échelle (`tests/test_conflicts.py`).

## 8. RAG contrôlé plutôt que réponse purement extractive ou génération libre

- Alternatives écartées : la seule unité extraite (pas de synthèse, pas d'abstention sur le fond) ; une génération libre (citations non vérifiables).
- Raison : le LLM ne reçoit que les unités retrouvées et le code rejette toute citation qui n'en fait pas partie ; la preuve extractive reste affichée.
- Preuve : critères fixés avant exécution et atteints au dev : 24 requêtes avec réponse sur 25 citent une unité attendue, 4 abstentions sur 5, 28 citations valides sur 28 (`rag_dev.json`).

## 9. Dossier déterministe, sans LLM

- Alternative écartée : un résumé rédigé par le LLM.
- Raison : chaque élément du dossier doit être une unité citée avec sa réunion, sa date et sa preuve.
- Preuve : sur le test scellé élargi, aucune unité citée sans source dans les 4 dossiers, dont l'un s'abstient ; 20 des 24 unités attendues sont citées (`test2.json`).

## 10. Consigne d'extraction réglée sur 01-03 puis gelée

- Alternative écartée : régler la consigne sur l'ensemble des projets.
- Raison : mesurer l'extraction sur des projets que le réglage n'a jamais vus ; l'empreinte SHA-256 de la consigne est enregistrée.
- Preuve : F1 des unités de 0,852 sur le dev 04-10 et 0,829 sur le test 11-12 ; F1 strict des relations de 0,406 seulement en dev (`extraction.json`).

## 11. Ne pas activer la seconde passe de liaison des relations

- Alternative écartée : brancher les relations v2 dans la mémoire extraite.
- Raison : le critère fixé avant les résultats (F1 strict dev d'au moins 0,456 avec une précision d'au moins 0,45) n'est pas atteint.
- Preuve : F1 strict dev de 0,403 pour la configuration retenue, contre 0,406 sans elle ; 0,381 au test scellé dans les deux cas (`relations_v2.json`).

## 12. Brouillon d'email jamais envoyé, sans adresse inventée

- Alternative écartée : envoi automatique, ou adresse déduite du prénom.
- Raison : le destinataire est la personne qui doit faire la tâche (premier participant nommé dans l'énoncé), et aucune adresse n'existe dans les sources ; l'utilisateur confirme.
- Preuve : comportement couvert par des tests (`tests/test_agent_logic.py`), sans mesure chiffrée.

## 13. Mettre en cache chaque réponse Groq

- Alternative écartée : rappeler le LLM à chaque exécution.
- Raison : Groq n'est pas strictement déterministe à température nulle ; le cache rend les mesures rejouables sans réseau.
- Preuve : le rejeu de l'extraction depuis le cache local redonne les métriques dev de `extraction.json` avec 0 appel Groq (champ `replay` de `relation_errors_dev.json`, vérifié par `tests/test_relation_errors.py` quand le cache est présent) ; le cache du RAG de la démonstration est versionné (`data/processed/groq_rag_cache.jsonl`).

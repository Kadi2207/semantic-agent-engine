# Données externes

Ce dossier contient des corpus publics téléchargés pour une **évaluation externe** de la recherche de passages.
Les données elles-mêmes ne sont **jamais versionnées** (`data/external/*` est ignoré par git) ; seul ce fichier l'est.

Licences vérifiées le 2026-11-15 sur les pages officielles indiquées ci-dessous. Un dépôt public ne prouve pas une
licence : chaque source d'origine est donc documentée séparément de QMSum, qui ne fait que les redistribuer.

Portée : ces données servent à mesurer la **recherche de passages dans des réunions en anglais**. Elles ne mesurent
pas la détection des décisions, engagements, contradictions ou responsabilités, qui est évaluée sur le corpus
interne (`data/raw/`).

---

## QMSum (jeu d'annotations requête / résumé / passages)

- **Source** : https://github.com/Yale-LILY/QMSum (dépôt officiel), cloné dans `data/external/qmsum/`.
- **Licence** : MIT, fichier `LICENSE` du dépôt (« Copyright (c) 2021 Yale-LILY »).
  Cette licence couvre le travail de Yale-LILY (requêtes, résumés, intervalles de passages, code), **pas** les
  transcriptions d'origine, qui restent soumises aux licences de leurs sources (sections suivantes). L'article
  (section 7.1) affirme que les transcriptions sont « public and open to use according to the regulation » :
  c'est une affirmation des auteurs, pas une licence.
- **Composition** (article, section 3.1) : Product = AMI (137 réunions), Academic = ICSI (59 réunions),
  Committee = 25 réunions du Parlement gallois + 11 du Parlement du Canada.
- **Citation** :
  Zhong, M., Yin, D., Yu, T., Zaidi, A., Mutuma, M., Jha, R., Hassan Awadallah, A., Celikyilmaz, A., Liu, Y.,
  Qiu, X., Radev, D. (2021). *QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization.* NAACL 2021,
  p. 5905-5921. DOI 10.18653/v1/2021.naacl-main.472.
- **Redistribution** : les annotations QMSum sont redistribuables sous MIT (avec la notice de copyright). Nous ne
  redistribuons rien : seules des métriques agrégées et des identifiants de requêtes sont publiés.
- **Usage fait** : `specific_query_list` et leurs `relevant_text_span`, splits officiels. `test` sert à la mesure
  finale ; `val` est réservé à un éventuel réglage (aucun n'a été fait). Le split `train` n'est pas utilisé.
  Les intervalles sont des indices de tours de parole (position dans `meeting_transcripts`, à partir de 0, bornes
  incluses) ; correspondance vérifiée à la main sur un exemple par domaine.

## AMI Meeting Corpus (domaine Product)

- **Source** : https://groups.inf.ed.ac.uk/ami/corpus/ (Université d'Édimbourg, projet AMI).
- **Licence** : Creative Commons Attribution 4.0 (CC BY 4.0), page officielle
  https://groups.inf.ed.ac.uk/ami/corpus/license.shtml (« released under the Creative Commons Attribution 4.0
  license agreement (also called CC BY 4.0) »).
- **Nature des réunions** : une grande partie du corpus (environ deux tiers d'après la présentation officielle,
  https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml) est constituée de **réunions de conception jouées selon un
  scénario** : des participants tiennent des rôles dans une équipe fictive qui conçoit une télécommande. Les réunions
  du domaine Product de QMSum en font partie (« 137 meetings about how to design a new remote control »). Ce ne sont
  donc pas des réunions d'entreprise réelles.
- **Citation / attribution demandée** : « AMI Project, http://www.amiproject.org ; AMI Meeting Corpus,
  https://www.idiap.ch/dataset/ami/ ». Référence : Carletta, J. (2006). Announcing the AMI Meeting Corpus.
  *The ELRA Newsletter* 11(1).
- **Redistribution** : autorisée, y compris modifiée, avec attribution, mention de la licence, lien et sans laisser
  entendre un soutien du concédant (CC BY 4.0, sections 2 et 3).
- **Usage fait** : passages des réunions Product du split test de QMSum (évaluation de la recherche). Le fichier
  `eval/results/real_extraction_sample.csv` (audit qualitatif) contient de **courts extraits de 3 réunions AMI**
  (champ `evidence`) ; ils sont redistribués sous CC BY 4.0 avec l'attribution ci-dessus. Les champs `statement` de
  ce fichier sont des reformulations produites par notre pipeline, pas des contenus AMI.

## ICSI Meeting Corpus (domaine Academic)

- **Source** : https://groups.inf.ed.ac.uk/ami/icsi/ (hébergement public par l'Université d'Édimbourg).
- **Licence** : CC BY 4.0, page officielle https://groups.inf.ed.ac.uk/ami/icsi/license.shtml (« released under the
  Creative Commons Attribution 4.0 license agreement (also called CC BY 4.0) »).
- **Distribution parallèle** : le LDC distribue aussi le corpus (LDC2004S02, transcriptions LDC2004T04) sous
  « LDC User Agreement for Non-Members », payant. Le texte de cet accord n'a pas pu être lu. Nous utilisons la copie
  incluse dans QMSum, dont la provenance exacte (hébergement CC BY 4.0 ou LDC) n'est pas précisée par QMSum.
- **Citation** : Janin, A. et al. *ICSI Meeting Transcripts* LDC2004T04. Philadelphia: Linguistic Data Consortium,
  2004. Copyright : « Portions © 2000-2003 International Computer Science Institute ».
- **Redistribution** : autorisée sous CC BY 4.0 avec attribution. Nous ne redistribuons aucun contenu ICSI.
- **Usage fait** : passages des réunions Academic du split test de QMSum (réunions hebdomadaires de groupes de
  recherche, réelles et non scénarisées).

## Sources parlementaires (domaine Committee)

### Parlement du Canada (Chambre des communes), réunions `covid_*`

- **Source** : comptes rendus des comités, https://www.ourcommons.ca/.
- **Licence / conditions** : « Speaker's Permission Regarding the Reproduction of the Proceedings of the House of
  Commons and its Committees », https://www.ourcommons.ca/en/important-notices : la reproduction « is hereby
  permitted provided that the reproduction is accurate and is not presented as official ». La permission ne couvre
  pas la reproduction ou l'usage commercial (« for commercial purpose of financial gain »).
- **Citation** : Chambre des communes du Canada, Comité spécial sur la pandémie de COVID-19, témoignages (via QMSum).
- **Redistribution** : possible si exacte, non présentée comme officielle et non commerciale. Nous ne redistribuons
  aucun contenu ; nos résultats ne sont pas une publication de la Chambre.
- **Usage fait** : 2 réunions du split test (18 requêtes spécifiques), évaluation non commerciale.

### Parlement gallois (Senedd), réunions `education_*`

- **Source** : Record of Proceedings des comités, https://record.senedd.wales/ (lien de l'article :
  https://record.assembly.wales/).
- **Licence / conditions, NON VÉRIFIÉE pour les comités** :
  - la page Open Data (https://senedd.wales/help/open-data/) place sous Open Government Licence uniquement le
    « Record of Proceedings (Plenary) XML library », pas les transcriptions de comités ;
  - la page « Re-using Public Sector Information »
    (https://senedd.wales/commission/access-to-information/re-using-public-sector-information/) autorise la
    réutilisation non commerciale des documents publiés « under its Open Licence (based on the Non-Commercial
    Government Licence) », sans dire explicitement que les transcriptions de comités sont couvertes ;
  - le lien de l'article vers la page Open Data (`senedd.wales/en/help/our-information/Pages/Open-data.aspx`)
    renvoie une erreur 404.
  Aucune page lue n'interdit l'usage non commercial ; aucune ne l'accorde explicitement pour les comités.
  Confirmation possible auprès de information-request@senedd.wales.
- **Citation** : Senedd Cymru / Welsh Parliament, Record of Proceedings, réunions de comité consacrées à l'éducation
  (fichiers `education_*` de QMSum).
- **Redistribution** : non vérifiée ; nous ne redistribuons aucun contenu.
- **Usage fait** : 4 réunions du split test (48 requêtes spécifiques). Elles sont **rapportées à part**
  (sous-groupe « Committee / Pays de Galles ») pour pouvoir être écartées si la licence se révélait incompatible.

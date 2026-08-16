# CineProfile 0.16.1 — figer l’ordre gagnant du vivier

## Décision issue de l’audit 0.16.0

Sur les 100 films 8+ éligibles à la période de trois ans, le dernier audit a
mesuré le même vivier à chaque étape :

- ordre équilibré brut : 61 films retrouvés à 300, 66 à 500 ;
- après ordre sémantique : 54 à 300, 65 à 500 ;
- après quotas : aucun gain supplémentaire.

Le plafond du vivier reste de 67 films retrouvables. La voie séparée des
classiques fonctionne et reste inchangée.

## Changement de production

Le moteur utilise directement l’ordre équilibré construit à partir des sources
TMDB. Il ne réordonne plus les candidats par proximité sémantique avant de
télécharger leurs fiches et n’applique plus les quotas de familles.

Ce changement ne supprime pas la compréhension sémantique : elle intervient
toujours après l’enrichissement dans le score personnel, le classement final et
les explications affichées.

## Audit 1.4 et condition de fin

L’audit reproduit maintenant exactement cet ordre de production. Les anciens
champs « avant sémantique » et « avant quotas » sont conservés pour rendre les
rapports 1.x comparables ; leurs valeurs sont désormais identiques au résultat
final.

Une dernière exécution doit confirmer environ 61 films sur 100 à 300 candidats
et 66 sur 100 à 500. Après cette vérification, l’ordre du vivier est considéré
comme terminé. Les améliorations futures devront porter sur la couverture des
films absents des sources, pas sur de nouvelles règles de tri.

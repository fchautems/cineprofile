# CineProfile 0.15.0 — vivier public durable

## Point de départ mesuré

L’audit réel 0.14.0 retrouvait 38 % des films 8+ admissibles dans les 100
premiers candidats, 61 % à 300 et 62 % à 500. Trente-six films admissibles
étaient absents de toutes les sources. Agrandir le budget ne pouvait donc pas
résoudre le principal problème.

## Changements de récupération

- La découverte publique est découpée par année pour les périodes courtes, puis
  par blocs de deux, cinq ou dix ans pour les périodes plus larges.
- TMDB Discover n’utilise plus `region=CH` par défaut : ce paramètre filtre les
  dates de sortie régionales. La disponibilité suisse reste traitée après la
  récupération.
- Un fonds de catalogue depuis 1920 complète, par défaut, la période récente.
- Les sources publiques utiles reçoivent davantage de places dans l’ordre de
  récupération.
- Les requêtes « films similaires » sont désactivées ; les recommandations par
  graines restent présentes avec deux fois moins de graines en mode normal.

## Persistance

Les réponses brutes sont stockées dans `candidate_catalog` et les scans dans
`candidate_catalog_scans`. Une combinaison identique de source, période,
filtres, langue et région est réutilisée pendant 45 jours. Le catalogue ancien
est valable 180 jours. Cela permet d’accumuler un vivier local sans confondre ce
cache de récupération avec le cache des fiches TMDB détaillées.

## Mesure

L’audit du vivier active le catalogue ancien et conserve le même protocole
chronologique. Lorsqu’un rapport antérieur existe, les écarts de rappel à 100,
300 et 500 sont affichés directement en points. Aucun score du classement final
n’est modifié dans cette version.

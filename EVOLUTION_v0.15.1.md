# CineProfile 0.15.1 — quotas de récupération

## Ce que le second audit a montré

La version 0.15 a agrandi le vivier et retrouve 84 films 8+ sur 182, contre 62
avec la 0.14. Sur les 100 films admissibles dans la période de trois ans, le
rappel atteint toutefois 36 % à 100, 58 % à 300 et 66 % à 500. Le gain global
est donc réel, mais l’ordre des candidats reste perfectible.

Le catalogue ancien apporte 15 films uniques dans les 500 premiers. À
l’inverse, la source fondée sur les films favoris a fourni 600 candidats sans
retrouver un seul film cible dans ce rapport.

## Changement appliqué

L’ordre de récupération est maintenant divisé en trois familles :

- 70 % de candidats publics récents ;
- 10 % de pistes personnelles complémentaires ;
- 20 % de catalogue public ancien.

Ces proportions organisent le budget d’analyse par blocs de dix. Une famille
vide ne bloque rien : ses places sont reprises par les autres. L’ordre interne
de chaque famille conserve le signal sémantique et l’équilibrage des sources
déjà calculés.

La source TMDB issue des films favoris est désactivée dans les trois
profondeurs. Elle pourra revenir uniquement si une nouvelle mesure montre un
apport utile.

## Audit 1.2

L’audit suit maintenant la même chaîne de récupération que l’application :
exclusion de l’historique, proximité sémantique, équilibrage des sources, puis
quotas. Il affiche aussi le nombre de films 8+ gagnés ou perdus par les quotas
par rapport au même vivier juste avant leur application.

Une comparaison avec un rapport 1.1 reste visible, mais elle est signalée
comme indicative puisque le protocole mesure désormais plus fidèlement le
comportement réel. Le prochain audit 1.2 constituera la nouvelle référence.

Le classement final et les probabilités affichées ne changent pas dans cette
version.

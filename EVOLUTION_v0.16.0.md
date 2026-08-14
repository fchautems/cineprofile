# CineProfile 0.16.0 — séparer le récent et les classiques

## Problème observé

Avec la période « 3 dernières années », CineProfile pouvait placer *Le
Dictateur* (1940) puis *Call Me by Your Name* (2017) dans les suggestions
principales. Le catalogue ancien, activé pour améliorer la couverture du
vivier, contournait volontairement la période et partageait le même budget
d’analyse que les films récents.

L’audit 0.15.1 a aussi montré que les quotas n’apportaient que deux films 8+
dans les 100 premiers et aucun à 300 ou 500. Le nouvel audit avait enfin révélé
que le signal sémantique pouvait modifier fortement l’ordre, sans mesurer son
effet séparément dans une même exécution.

## Trois listes, deux budgets

**Valeurs sûres** et **Découvertes pour toi** ne contiennent plus que des films
compris dans la période sélectionnée. Les films antérieurs sont envoyés vers
**Classiques à découvrir**.

Les deux voies sont enrichies avec des budgets indépendants :

- récent : 140, 300 ou 500 fiches selon la profondeur ;
- classiques : 50, 100 ou 150 fiches selon la profondeur.

Les classiques ne consomment donc plus une part du budget récent. Le marqueur
de voie est persisté avec chaque recommandation. Pour les résultats enregistrés
par une ancienne version, la source « Catalogue public plus ancien » permet de
les migrer automatiquement vers le nouvel onglet.

## Sémantique contrôlé

La proximité d’histoire reste calculée avant l’enrichissement. Elle ordonne les
candidats à l’intérieur de leurs véritables sources — popularité, qualité,
genres, thèmes ou personnes — mais n’est plus ajoutée à la liste des sources.
Elle ne peut donc plus obtenir des places supplémentaires par le mécanisme
d’équilibrage.

## Audit 1.3 et condition de fin

Pour chaque fenêtre chronologique, le rapport conserve maintenant :

1. le rappel de la voie récente avant ordre sémantique ;
2. le rappel après ordre sémantique ;
3. le rappel final après quotas ;
4. le rappel indépendant de la voie des classiques.

L’écran principal de l’audit affiche le rappel des seuls films 8+ sortis dans
la période, qui correspond au comportement des suggestions normales. L’objectif
fixé est au moins 60 films sur 100 à 300 et 64 sur 100 à 500, sans réduire les
67 films sur 100 présents quelque part. Si le prochain audit n’atteint pas ces
seuils, CineProfile arrête de régler l’ordre et traite séparément le problème
des films absents du vivier.

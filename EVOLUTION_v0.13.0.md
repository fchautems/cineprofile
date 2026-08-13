# CineProfile 0.13.0 — interface simplifiée et suggestions persistantes

## Objectif

Cette version commence la simplification de CineProfile avant les évolutions du
vivier. Elle réduit les recalculs inutiles et sépare l’usage quotidien des
outils de maintenance.

## Nouvelle navigation

Les six onglets historiques sont remplacés par quatre espaces :

1. **Suggestions** : les deux listes de recommandations et leur actualisation.
2. **Ma liste** : les films marqués ou envoyés à Radarr.
3. **Mon profil** : une synthèse lisible des goûts issus d’IMDb.
4. **Réglages** : connexions, mise à jour IMDb, personnalisation et maintenance.

La barre latérale n’affiche plus que l’état des connexions et la version.

## Suggestions persistantes

Chaque recherche conserve maintenant sa sélection, ses paramètres et son
diagnostic avec le profil qui l’a produite. Au démarrage, CineProfile recharge
immédiatement cette sélection au lieu de demander une nouvelle recherche TMDB.

Les films marqués **Pas intéressé** ou **Déjà vu** depuis la recherche sont
retirés de l’affichage restauré, sans effacer la trace historique de la
sélection.

Une nouvelle recherche reste nécessaire lorsqu’un nouvel import IMDb produit
un nouveau profil. Les prochaines étapes sépareront aussi le renouvellement du
vivier du simple reclassement local.

## Radarr : libellé honnête

Le libellé local **Downloaded** devient **Envoyé à Radarr**. Il signifie que
Radarr a accepté l’ajout et la recherche du film. CineProfile ne prétend donc
plus qu’un fichier est déjà arrivé sur le NAS.

La 0.13.1 rendra les actions de Ma liste directement cliquables. La 0.13.2
synchronisera les états détaillés de Radarr, notamment téléchargement et
fichier disponible.

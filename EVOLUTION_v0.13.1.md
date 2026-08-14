# CineProfile 0.13.1 — Ma liste, actions immédiates

## Objectif

Rendre les décisions quotidiennes rapides et cohérentes entre **Suggestions**
et **Ma liste**, sans formulaire de modification séparé.

## Interface

- Les filtres de Ma liste sont des boutons compacts avec icône et compteur.
- Ma liste affiche les mêmes cartes de film que Suggestions : affiche, résumé,
  métadonnées et lien IMDb/TMDB.
- Les actions sont identiques sur les deux écrans :
  - `thumb_up` : À voir ;
  - `thumb_down` : Pas pour moi ;
  - `visibility` : Déjà vu ;
  - `send` : envoyer la demande à Radarr.
- Une action personnelle active est mise en évidence. Cliquer à nouveau dessus
  retire l’état. Sélectionner une autre action personnelle remplace la première.

## Limite volontaire

Après une demande réussie, CineProfile montre l’icône `radar`. Elle signifie
uniquement « envoyé à Radarr ». La synchronisation des états réels (monitoré,
téléchargement, disponible, erreur) relève de la 0.13.2.

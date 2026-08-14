# CineProfile 0.13.3 — synchronisation Radarr à l’entrée

## Objectif

Supprimer l’affichage potentiellement provisoire observé en ouvrant **Ma
liste** après un téléchargement ou une importation Radarr.

## Comportement

- L’entrée dans **Ma liste** force une lecture immédiate de la bibliothèque et
  de la file Radarr, même si le relevé précédent date de moins de 25 secondes.
- Pendant cette lecture, l’interface affiche
  **« Actualisation des états Radarr… »**.
- Le rafraîchissement périodique de 30 secondes reste actif tant que l’onglet
  est ouvert, et le bouton de synchronisation manuelle reste disponible.
- Les interactions locales (filtres, recherche, 👍, 👎 et 👁) ne contactent
  toujours pas Radarr.

## Limite volontaire

Cette version ne présente pas encore un état Bazarr. CineProfile ne doit pas
inventer l’état des sous-titres : Bazarr doit d’abord être correctement relié à
Radarr et son comportement d’import validé. L’intégration éventuelle passera
par une connexion Bazarr explicite, séparée des clés Radarr/TMDB.

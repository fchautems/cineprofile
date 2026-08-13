# CineProfile 0.12.1 — connexions persistantes

Ce correctif remplace les réglages séparés de TMDB et Radarr par une seule
configuration locale cohérente.

## Changements

- Un panneau compact **Connexions** remplace les champs de clés constamment
  visibles dans la barre latérale.
- **Enregistrer** valide Radarr puis écrit TMDB, Radarr, le dossier des films et
  le profil de qualité en une seule opération atomique.
- Une erreur de validation ou d’écriture conserve intégralement l’ancienne
  configuration.
- Après enregistrement, l’interface affiche seulement **TMDB configuré** et
  **Radarr connecté** ; les secrets restent masqués.
- Radarr se reconnecte automatiquement au lancement, sans nouvelle saisie ni
  clic sur **Connecter**.
- Les variables d’environnement restent utilisables pour Docker.
- Un `.env` qui serait accidentellement un dossier produit désormais une erreur
  explicite et aucune donnée n’est écrasée.

# CineProfile 0.12.0 — Mes films et première intégration Radarr

Cette version ajoute une mémoire permanente des décisions prises sur les
suggestions et relie les films à Radarr.

## Mes films

- Un nouvel onglet rassemble tous les films marqués depuis les suggestions.
- Les filtres **Tous**, **À voir**, **Déjà vus**, **Pas intéressé** et
  **Downloaded** permettent de retrouver immédiatement une décision passée.
- Les statuts sont modifiables et réversibles sans attendre une nouvelle
  recherche de recommandations.
- Le statut Radarr reste indépendant des trois retours qui alimentent le
  moteur de recommandations.

## Connexion

- L’adresse et la clé API Radarr se configurent dans la barre latérale.
- Le bouton **Connecter** valide réellement l’accès à Radarr.
- CineProfile récupère ensuite les dossiers racines et profils de qualité
  disponibles afin de laisser choisir la destination.
- Les identifiants sont mémorisés uniquement dans le fichier `.env` local.

## Demande de film

- Chaque recommandation propose **Download** une fois Radarr connecté.
- CineProfile ajoute le film à Radarr, le surveille et lance sa recherche.
- Le statut **Downloaded** n’est enregistré qu’après confirmation de l’API, y
  compris lorsque le film existait déjà dans Radarr.
- Une erreur de connexion ou un refus de Radarr ne crée aucun faux succès.
- Chaque tentative est conservée avec sa date et son résultat : acceptée, déjà
  présente ou échouée.

## Choix de modèle

Dans cette version, **Downloaded** signifie précisément « envoyé à Radarr au
moins une fois — présence du fichier non vérifiée ». Annuler le statut retire
uniquement cette information de CineProfile. Cela ne supprime rien dans Radarr,
n’interrompt aucune recherche et aucun téléchargement, et rend le bouton
**Download** de nouveau disponible. L’historique des tentatives est conservé.

## Import IMDb incrémental

- Le résultat distingue les titres nouveaux, réellement modifiés, inchangés et
  ignorés.
- L’interface confirme combien d’enrichissements TMDB, de statuts de films et
  de corrections du profil ont été préservés.
- Aucun nouvel enrichissement global ni nouveau moteur d’IA n’est imposé par
  cette version.

# CineProfile 0.13.2 — Radarr réel, actions instantanées

## Objectifs

- Remplacer le simple marqueur local « envoyé » par le dernier état technique
  réellement observé dans Radarr.
- Garantir qu’un clic sur 👍, 👎 ou 👁 ne déclenche aucun calcul du moteur et
  aucun appel réseau.
- Ne rendre que l’onglet visible afin de réduire le coût de chaque interaction.

## États Radarr

CineProfile croise `GET /api/v3/movie` et `GET /api/v3/queue` :

- **Envoyé** : demande acceptée, première synchronisation encore à venir ;
- **Monitoré** : présent et surveillé dans Radarr ;
- **Aucun téléchargement** : recherche déjà effectuée, rien dans la file ;
- **Téléchargement** : élément présent dans la file, avec pourcentage si connu ;
- **Disponible** : `hasFile` est vrai et Radarr a importé un fichier ;
- **Erreur Radarr** : téléchargement ou import bloqué/échoué ;
- **Absent de Radarr** : la demande locale existe mais plus le film distant ;
- **Non monitoré** : le film existe mais sa surveillance a été désactivée.

Les états sont enregistrés avec leur détail, leur progression et l’heure de
vérification. Une panne de Radarr ne remplace jamais le dernier état fiable.

## Fluidité

- Les décisions personnelles sont écrites dans SQLite par callback avant le
  rerun automatique du widget ; l’ancien deuxième `st.rerun()` disparaît.
- Les quatre onglets principaux et les sous-onglets de Suggestions sont
  paresseux grâce à l’état suivi des onglets Streamlit.
- Ma liste est un fragment autonome actualisé toutes les trente secondes.
- La synchronisation est considérée fraîche pendant vingt-cinq secondes : un
  clic personnel ne contacte donc pas Radarr.

Cette version requiert Streamlit 1.61 ou plus ; le lanceur habituel met la
dépendance à niveau lors du démarrage.

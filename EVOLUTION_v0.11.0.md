# CineProfile 0.11.0 — apprentissage progressif

Cette version conserve les deux listes et le classement corrigé de la 0.10.1.
Elle ajoute une boucle d’apprentissage directement liée aux actions de
l’interface.

## Signaux utilisés

- Une nouvelle note IMDb modifie l’empreinte du profil et provoque le
  réentraînement automatique du modèle personnel au prochain calcul.
- « À voir » devient un signal d’envie faible. Il peut rapprocher des films
  similaires dans les découvertes, sans être traité comme une note positive.
- « Pas intéressé » reste un contre-exemple sémantique et exclut le film.
- « Déjà vu » exclut seulement le film et n’invente aucune préférence.

## Garde-fous

- Un film « À voir » n’entre pas dans le calcul de la note personnelle prévue
  ni dans la chance estimée d’attribuer 8/10 ou plus.
- Les valeurs sûres conservent le classement public dominant et le garde-fou
  de compatibilité de la 0.10.1.
- Si le modèle sémantique local est indisponible, le repli lexical continue de
  fonctionner.
- Les retours sont réversibles depuis les préférences.

## Diagnostic

Le diagnostic d’une recherche indique maintenant le nombre de signaux « À
voir », de refus et d’exclusions simples réellement disponibles pour
l’apprentissage.

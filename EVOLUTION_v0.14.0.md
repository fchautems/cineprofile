# CineProfile 0.14.0 — audit du vivier

## Question mesurée

Avant d’ajouter des sources ou de modifier leurs poids, CineProfile doit savoir
si le vivier actuel contient les films que l’utilisateur appréciera réellement.
La cible positive reste une note IMDb d’au moins 8/10.

## Protocole

- Trois ou cinq fenêtres chronologiques expansives et disjointes.
- Pour chaque fenêtre, une copie temporaire de la base ne conserve que les
  notes antérieures à la période test.
- Le profil de récupération est reconstruit depuis ce seul passé ; les retours,
  préférences et modèles postérieurs sont supprimés de la copie.
- Les sources TMDB du mode normal sont interrogées sur la période de sortie par
  défaut de trois ans.
- Le rappel des futurs films 8+ est calculé à 100, 300 et 500 candidats.
- Chaque source est retirée à son tour et l’ordre équilibré est reconstruit pour
  mesurer les films perdus.

## Diagnostic des absences

Un film cible manquant reçoit une cause vérifiable :

- absent de toutes les sources ;
- hors période de sortie ;
- nombre de votes insuffisant ;
- genre exclu ;
- présent, mais au-delà des 500 premières places.

Les exclusions de genre sont désactivées dans l’audit 0.14.0 afin de mesurer le
vivier sans injecter une règle de goût. Le classement final n’est pas évalué :
ce rapport isole volontairement la récupération des candidats.

## Limite honnête

TMDB ne fournit pas un instantané historique de ses pages de découverte. Les
notes personnelles sont correctement cachées dans le temps, mais les
métadonnées, votes et popularités sont ceux du jour de l’audit. Le résultat est
donc une mesure rétrospective du vivier actuel avec un profil passé, pas une
reconstitution parfaite de ce que TMDB aurait renvoyé à l’époque.

## Utilisation

Dans **Réglages**, activer **Afficher la maintenance et les diagnostics**, puis
ouvrir **Mesurer le vivier — films 8+ retrouvés**. Le mode trois fenêtres est le
premier diagnostic conseillé ; le mode cinq fenêtres sert à confirmer les
résultats avant toute modification des sources.

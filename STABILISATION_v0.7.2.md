# Stabilisation technique de CineProfile 0.7.2

## Périmètre

Le moteur reste fonctionnellement gelé pendant cette passe. Les changements de
score sont limités aux corrections vérifiées suivantes :

- choix de la source publique la mieux étayée entre IMDb et TMDB ;
- exclusion correcte des séries reconnues sous le type exact `tv` ;
- filtres de genres et de durée réellement appliqués ;
- audit v0.6/v0.7 alimenté par les mêmes données normalisées que
  l’apprentissage.

Aucun nouveau vote utilisateur n’est demandé. La base et les enrichissements
existants sont réutilisés.

## Architecture obtenue

`app.py` ne contient plus la logique de chaque écran. Il initialise
l’application, vérifie le protocole et appelle les modules suivants :

| Module | Responsabilité |
| --- | --- |
| `ui_import.py` | import IMDb, enrichissement et actualisation |
| `ui_profile.py` | profil, validation et exports |
| `ui_audit.py` | lancement et lecture du backtest |
| `ui_catalog.py` | requêtes et filtres de la vidéothèque |
| `ui_recommendations.py` | réglages, recherche, filtres et diagnostic |
| `ui_recommendation_cards.py` | affichage et retours sur les suggestions |
| `ui_preferences.py` | corrections manuelles et feedback |
| `ui_common.py` | petites fonctions communes |
| `genre_catalog.py` | identifiants et portée des genres TMDB |

L’application reste tout-en-un pour l’utilisateur : même dossier, même base,
même interface et même `Lancer CineProfile.bat`.

## Défauts corrigés

1. Les contextes SQLite validaient les transactions sans fermer le fichier.
   Cela bloquait le nettoyage de la copie d’audit sous Windows.
2. L’audit repassait des films déjà normalisés dans le convertisseur des
   candidats TMDB. Les entités, la durée et la référence publique étaient alors
   perdues pour v0.6/v0.7.
3. Le rapport d’audit n’était écrit qu’après le nettoyage temporaire : douze
   étapes terminées pouvaient donc ne produire aucun fichier.
4. Le sélecteur d’exclusion ne proposait que « Horreur ».
5. Les durées absentes passaient toutes les plages personnalisées.
6. Le type exact `tv` pouvait devenir une graine `/movie/.../recommendations`.
7. Des entités TMDB portant le même nom avec des identifiants différents
   pouvaient violer les clés étrangères.
8. Une erreur TMDB globale était parfois enregistrée comme une erreur locale
   sur chaque film.
9. Une petite note TMDB présente était toujours préférée à une note IMDb
   beaucoup plus fiable.
10. Le diagnostic contrôlait l’ordre interne plutôt que le tri visible.
11. Un échec de recherche affichait un traceback et faisait perdre le contexte
    au lieu de conserver les résultats précédents.
12. Le catalogue échouait ou vidait la liste lorsque toutes les années étaient
    absentes ou identiques.
13. Le lanceur pouvait vérifier les dépendances auprès du réseau à chaque
    démarrage.
14. La vérification locale relançait une construction éditable et échouait
    dans un ancien environnement ne contenant pas la commande `bdist_wheel`.
    Le lanceur vérifie désormais les imports et les dépendances sans
    reconstruire le paquet ; une réparation n’est lancée qu’en cas de besoin.

## Nettoyage

Les anciens points d’entrée internes `recommend_recent`, `export_profile` et
`PublicRating.as_dict`, sans appel dans l’interface, ont été retirés. Les
moteurs v0.5, v0.6 et v0.7 restent présents car ils sont encore nécessaires à
la sélection du champion et aux backtests ; ils ne sont donc pas du code mort.

## Vérifications effectuées

- compilation récursive de `app.py`, du paquet et des tests ;
- analyse statique des imports et fonctions orphelines ;
- 46 tests automatisés réussis sous Python 3.12 et Streamlit 1.60 ;
- couverture mesurée à 80 % pour l’ensemble du paquet, 90 % pour le modèle
  personnel, 84 % pour l’audit et 83 % pour le recommandateur ;
- contrôle manuel SQLite/import/encodage/déduplication/source
  publique/durée/tri/diagnostic ;
- construction d’un profil local à partir du CSV d’exemple ;
- backtest synthétique complet sur 160 films, deux découpages, quatre moteurs ;
- audit de charge sur 1’618 films avec 5 découpages, courbe d’apprentissage
  400/700/1’000/1’213 et contrôle chronologique : 12 étapes terminées, rapport
  de 68 Ko sauvegardé et base source inchangée ;
- comparaison de l’empreinte logique de la base avant et après l’audit ;
- vérification de la création du JSON d’audit avant le nettoyage temporaire.

Le fichier de tests contient en plus les régressions ciblées pour Windows,
TMDB, les collisions de genres/thèmes, le type `tv`, les genres exclus, la
durée inconnue et le diagnostic trié.

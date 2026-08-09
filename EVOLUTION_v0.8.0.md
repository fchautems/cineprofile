# CineProfile 0.8.0 — architecture et règles de décision

## Principe

La v0.6 linéaire reste le champion au démarrage. La 0.8.0 ajoute des
challengers et un protocole pour prouver leur valeur. Aucun score de deux
moteurs n’est mélangé.

L’historique IMDb existant fournit toutes les cibles. Les films jamais notés ne
sont jamais transformés en exemples négatifs.

## Chaîne de suggestion

1. **Récupération** : plusieurs sources TMDB construisent un vivier large.
2. **Préfiltrage** : période, fiabilité adaptative, types, films vus, feedback
   et genres exclus.
3. **Échantillonnage équilibré** : un tourniquet entre sources choisit les
   fiches à enrichir.
4. **Prédiction personnelle** : un seul moteur actif estime la note et la
   probabilité d’attribuer au moins 8/10.
5. **Prudence** : la probabilité revient vers le taux de base lorsque la fiche,
   la note publique ou la couverture du modèle sont faibles.
6. **Classement** : Valeurs sûres conserve l’ordre prudent ; les autres modes
   appliquent une diversité contrôlée.

Le genre est une variable parmi d’autres. Sa fréquence historique ne devient
jamais directement un poids de préférence.

## Challengers

| Variante | Blocs |
| --- | --- |
| Métadonnées | genres, thèmes, personnes, langue, pays, durée, sociétés |
| Syntaxe | TF‑IDF sur mots et groupes de deux mots |
| Métadonnées + syntaxe | les deux blocs précédents |
| Sémantique | vecteurs multilingues préentraînés locaux |
| Métadonnées + sémantique | signaux explicites et sens du texte |
| Métadonnées + syntaxe + sémantique | tous les blocs dans un seul modèle |

Le deep learning n’est pas entraîné sur les 1’618 notes. Il sert uniquement à
encoder le sens des textes avec un modèle préentraîné. Le modèle personnel qui
apprend les goûts reste une régression Ridge fortement régularisée, adaptée à
la taille de l’historique.

## Validation imbriquée

Pour chaque découpage extérieur :

- 75 % des notes construisent le modèle ;
- 25 % restent totalement cachées ;
- à l’intérieur des 75 %, une validation stratifiée choisit l’alpha parmi
  10, 25, 50, 100, 200, 400 et 800 ;
- le calibrateur n’utilise que des prédictions hors pli ;
- le groupe extérieur n’intervient jamais dans le vocabulaire TF‑IDF, les
  poids, l’alpha ou la calibration.

Le contrôle chronologique apprend ensuite sur les anciennes notes et teste les
20 % les plus récentes.

## Portes de promotion

Le meilleur challenger complet doit simultanément :

- gagner au moins 0,015 de NDCG@10 ;
- ne pas perdre plus de 2 points de précision@10 ;
- ne pas perdre plus de 0,01 d’AUC ;
- ne pas ajouter plus de 0,05 point d’erreur moyenne de note ;
- ne pas ajouter plus de 0,03 d’erreur de Brier ;
- gagner NDCG@10 sur au moins 60 % des découpages ;
- préserver NDCG@10 et précision@10 sur les notes récentes ;
- ne pas choisir la borne maximale de régularisation dans la moitié des essais.

Si une seule porte échoue, la v0.6 reste active. Le rapport conserve toutes les
mesures pour analyse.

## Reproductibilité et retour arrière

Le bouton d’activation enregistre dans SQLite :

- la variante exacte ;
- l’alpha retenu le plus souvent ;
- la date du rapport qui autorise l’activation.

Le modèle est reconstruit avec cette configuration, sans réoptimisation cachée.
Le bouton « Restaurer la v0.6 linéaire » supprime uniquement cette
configuration. Notes, enrichissements, préférences et caches restent intacts.

## Fichiers de retour

- `data/logs/cineprofile.log` : journal rotatif détaillé ;
- `data/logs/audit_backtest_*.json` : mesures, portes et configuration proposée ;
- `data/logs/diagnostic_*.json` : recherche, sources, exclusions, ordre et
  détail des suggestions.

Le bouton « Tester le moteur et télécharger le diagnostic » ajoute au JSON les
filtres, le tri et les films réellement affichés.

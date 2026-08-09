# Laboratoire v1 — étape 2 MovieLens

Cette étape ne change ni les suggestions ni le moteur de CineProfile. Elle
teste séparément si les goûts de personnes ayant noté les mêmes films apportent
un signal plus utile que les deux références de l’étape 1.

## Déroulement

Le laboratoire :

1. retrouve le dernier rapport complet `arena_baseline_*.json` ;
2. vérifie que les cinq fenêtres et leurs 798 notes cachées ont exactement les
   mêmes empreintes ;
3. télécharge le jeu stable MovieLens 32M depuis GroupLens ;
4. vérifie les sommes MD5 officielles de `links.csv`, `movies.csv` et
   `ratings.csv` ;
5. prépare localement les 32 millions de notes communautaires ;
6. relie l’historique CineProfile à MovieLens grâce aux identifiants IMDb ;
7. choisit le nombre de voisins, la régularisation et le poids collaboratif sur
   une validation interne située dans le passé de chaque fenêtre ;
8. mesure le classement de satisfaction sur les mêmes films cachés ;
9. mesure séparément la récupération des futurs films vus et des futurs films
   notés 8+, au milieu d’un vrai catalogue de films non vus.

Un film non regardé reste non étiqueté. Il n’est jamais transformé en rejet.

## Premier lancement

Le téléchargement représente environ 239 Mio. Après extraction et création du
cache, il faut prévoir environ 1,5 Gio libres. La préparation initiale analyse
32 millions de lignes et peut prendre plusieurs dizaines de minutes selon le
PC et le disque. Le jeu brut et son cache restent dans `data\movielens` ; les
lancements suivants les réutilisent.

Double-cliquer sur :

`Lancer le laboratoire v1 - etape 2 MovieLens.bat`

Le fichier à transmettre ensuite est :

`data\logs\arena_movielens_*.json`

L’application, les notes personnelles et la base SQLite restent inchangées.

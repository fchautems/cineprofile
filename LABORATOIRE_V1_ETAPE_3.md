# Laboratoire v1 — étape 3 : deep learning sémantique

Cette étape compare trois représentations profondes des films sans modifier le
moteur de CineProfile :

- MiniLM multilingue, qui sert de contrôle et doit reproduire l’étape 1 ;
- Multilingual E5 Large ;
- BGE-M3.

## Ce qui est contrôlé

Les cinq fenêtres, leurs 798 films cachés et leurs empreintes viennent du
rapport corrigé de l’étape 1. Pour chaque fenêtre, le petit modèle personnel est
réentraîné uniquement sur les notes antérieures. Sa structure et sa procédure
de sélection sont identiques pour les trois challengers : seule la
représentation sémantique change.

Deux mesures sont produites :

1. le classement des films réellement notés dans chaque période cachée ;
2. la récupération de ces futurs films dans le catalogue présent dans la base,
   complété par les fiches de films déjà enrichies dans le cache.

Les candidats non notés restent des données inconnues. Ils ne deviennent jamais
des exemples négatifs.

## Modèles locaux

Le premier lancement peut télécharger environ 4,5 Gio :

- MiniLM : environ 0,22 Gio, normalement déjà présent ;
- E5 Large : environ 2,24 Gio ;
- BGE-M3 : environ 2,27 Gio.

Les textes et les notes ne sont envoyés à aucun service d’inférence. Les modèles
ONNX s’exécutent localement et leurs fichiers sont réutilisés lors d’un nouveau
lancement.

## Lancement

Double-cliquer sur :

`Lancer le laboratoire v1 - etape 3 Deep Learning.bat`

Le rapport final est écrit dans :

`data\logs\arena_semantic_*.json`

La base source est copiée avant les calculs. Les vecteurs sont écrits seulement
dans cette copie temporaire et l’empreinte de la base source est vérifiée avant
et après le test.


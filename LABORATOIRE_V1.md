# Laboratoire v1 — étape 1

Cette étape n’ajoute pas un nouveau moteur à CineProfile. Elle construit le
banc d’essai qui empêchera d’intégrer encore une version simplement parce que
ses premières suggestions « ont l’air meilleures ».

## Ce qui est mesuré

L’historique est trié par date de notation. L’arène apprend d’abord sur la
première moitié du passé, puis teste le moteur sur la période immédiatement
suivante. Elle recommence jusqu’à cinq fois en ajoutant progressivement les
anciennes périodes de test à l’apprentissage.

- une note testée n’est utilisée que dans une seule fenêtre ;
- deux notes du même jour ne sont jamais séparées arbitrairement ;
- le modèle et ses hyperparamètres ne voient jamais la période test ;
- chaque fenêtre possède une empreinte stable pour comparer plus tard
  MovieLens, E5 et BGE-M3 sur exactement les mêmes cas ;
- la base SQLite complète est copiée et le calcul travaille sur cette copie ;
- si des vecteurs sémantiques manquent, le laboratoire les calcule dans cette
  copie et affiche leur progression ; la base CineProfile d’origine ne reçoit
  aucun vecteur ;
- le modèle MiniLM déjà présent dans `data/models` est réutilisé, sans
  téléchargement supplémentaire lorsqu’il est complet ;
- une empreinte avant/après vérifie que la base source n’a pas changé.

La mesure principale est le NDCG@20 : elle vérifie si les films réellement
appréciés remontent bien en tête. La précision, la note moyenne du top 20,
l’erreur de note et la calibration restent visibles séparément.

## Ce qui n’est volontairement pas encore mesuré

Cette première étape ne prétend pas savoir si le moteur retrouve de bons films
dans tout le catalogue. Les films jamais notés ne sont pas transformés en
rejets. Cette seconde évaluation deviendra possible lorsque MovieLens et son
mapping IMDb/TMDB seront branchés.

## Lancer l’arène

1. Fermer CineProfile si un enrichissement ou un import est en cours.
2. Double-cliquer sur `Lancer le laboratoire v1.bat`.
3. Attendre la préparation sémantique puis la fin des fenêtres chronologiques.
4. Envoyer le fichier `data\logs\arena_baseline_*.json`.

Il n’est pas nécessaire de réimporter le CSV, de relancer l’enrichissement ni
de noter quoi que ce soit. Le premier lancement peut durer plusieurs minutes
si le cache de vecteurs est incomplet. Un rapport n’est plus produit si la
référence personnelle ne peut pas être évaluée intégralement.

## Règle pour la suite

MovieLens, E5 et BGE-M3 seront des challengers séparés. Aucun ne remplacera le
moteur courant sans gain répété sur les mêmes fenêtres et sans une preuve
distincte sur la récupération dans un vrai catalogue.

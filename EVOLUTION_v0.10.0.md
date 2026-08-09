# CineProfile 0.10.0 — deux listes complémentaires

La version 0.10.0 transforme une même recherche TMDB en deux classements
visibles simultanément.

## Valeurs sûres

L’ordre repose uniquement sur des données publiques :

1. note TMDB corrigée par l’incertitude ;
2. fiabilité de cette note ;
3. nombre de votes.

L’affinité personnelle, la chance d’un 8+, l’indice d’envie, la proximité
sémantique et le modèle personnel ne sont jamais consultés pour déterminer cet
ordre. Un test dédié inverse tous les scores personnels et vérifie que le
classement reste strictement identique.

Les cartes mettent en avant la note publique corrigée, la fiabilité et le
nombre de votes. Le détail rappelle explicitement que l’ordre ne dépend pas du
profil personnel.

## Découvertes pour toi

La seconde liste réutilise les mêmes fiches déjà téléchargées et analysées.
Elle les ordonne avec :

- la chance personnelle d’apprécier le film ;
- l’indice d’envie ;
- la proximité des histoires ;
- un petit garde-fou de qualité publique ;
- la récence, la nouveauté et la diversité.

Les vingt premières valeurs sûres sont retirées de cette liste. Les deux
onglets ne doivent donc pas répéter les mêmes têtes d’affiche.

## Interface

Le sélecteur historique « Valeurs sûres / Équilibré / Découvertes » disparaît.
Après une recherche, deux onglets sont disponibles :

- **Valeurs sûres** ;
- **Découvertes pour toi**.

Changer d’onglet ne relance ni TMDB ni le modèle sémantique. Chaque onglet
conserve ses propres filtres, son tri et le nombre de cartes affichées.

Les actions **Pas intéressé** et **Déjà vu** retirent immédiatement le film des
deux listes. Les recherches suivantes continuent de l’exclure grâce au retour
enregistré dans la base.

## Mise à jour

1. Fermer complètement CineProfile.
2. Copier la version 0.10.0 par-dessus le dossier existant.
3. Conserver `.venv`, `.env` et `data`.
4. Relancer `Lancer CineProfile.bat`.
5. Effectuer une recherche normale.

Il n’est pas nécessaire de réimporter le CSV, de refaire l’enrichissement ou de
relancer les laboratoires.

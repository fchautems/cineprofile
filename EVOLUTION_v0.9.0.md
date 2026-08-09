# CineProfile 0.9.0 — recherche et score personnels

## Pourquoi cette version existe

Les diagnostics de la 0.8 ont montré que le problème principal n’était pas le
nombre de paramètres du modèle :

- 82 candidats sur 103 provenaient uniquement de la popularité ;
- aucun candidat « similaire aux favoris » ne survivait dans le vivier final ;
- 9 films sur 10 restaient identiques après l’activation du challenger ;
- 18 candidats sur 103 étaient des animations, dont quatre des cinq premiers ;
- la note publique corrigée restait le socle de la note prévue.

La 0.9 modifie donc la chaîne complète :

```text
sources TMDB larges
        ↓
récupération sémantique personnelle
        ↓
enrichissement des candidats retenus
        ↓
voisinage local aimé / moins aimé
        +
modèle personnel global
        ↓
chance prudente
        ↓
diversification limitée du classement
```

## Règles du score

### Chance de plaire

La chance de plaire estime la probabilité d’attribuer au moins 8/10.

- 65 % : voisinage local des histoires, thèmes et métadonnées proches ;
- 35 % : modèle global appris sur l’historique enrichi ;
- 0 % : note TMDB directe.

Le voisinage local compare chaque candidat à des films appréciés et à des
contre-exemples notés au plus 6/10. Un voisin moins aimé retranche plus qu’une
ressemblance positive faible n’ajoute. Lorsque la sémantique profonde locale
n’est pas disponible, un voisinage lexical TF-IDF assure le repli.

Le modèle global ne prédit plus un écart autour de la note publique. Il prédit
un écart autour de la moyenne personnelle, avec les genres, thèmes,
descriptions, réalisateurs, scénaristes, acteurs, langues, pays et durées. Les
colonnes de note publique et de volume de votes sont absentes de ses variables
de goût.

### Rôle de TMDB

TMDB conserve trois rôles utiles :

1. fournir les fiches, crédits, descriptions, thèmes et affiches ;
2. écarter, selon le réglage de fiabilité, les films dont le nombre de votes est
   trop faible ;
3. départager légèrement des candidats en mode Équilibré ou Découvertes.

La note publique ne change jamais le pourcentage d’affinité. Dans les modes
exploratoires, sa composante est plafonnée à 10 % du score d’ordre. En mode
Valeurs sûres, elle ne participe pas au classement personnel.

### Prudence et diversité

Une suggestion reçoit l’une des étiquettes suivantes :

- **Correspondance solide** : probabilité et confiance suffisantes ;
- **Piste plausible** : signal personnel positif mais encore incomplet ;
- **Solution de repli** : candidat conservé faute de meilleure correspondance.

La diversification ne peut promouvoir un film situé plus de huit points sous
le meilleur choix restant. Pour un genre appris comme négatif, les occurrences
au-delà des deux premières dans le top 10 reçoivent une pénalité, uniquement
lorsqu’une alternative de niveau comparable existe.

## Récupération des candidats

La profondeur choisie contrôle un budget multi-source :

- sorties et films bien établis dans la période ;
- recommandations et similitudes autour des favoris ;
- réalisateurs, scénaristes et acteurs suffisamment observés ;
- thèmes et genres personnels ;
- proximité sémantique avec l’historique.

Les candidats sont comparés aux textes déjà encodés avant les appels détaillés
TMDB. La source `Histoires proches de tes goûts` devient prioritaire, puis un
tourniquet conserve une diversité de provenances. Le diagnostic indique :

- le nombre de candidats récupérés par la sémantique ;
- la part de candidats provenant seulement de la popularité ;
- la source et le score de récupération de chaque résultat.

## Audit 0.9

L’audit continue de travailler sur une copie SQLite et vérifie que la base
d’origine reste inchangée. Il compare toujours les modèles sur des notes
cachées, mais ajoute une mesure indépendante du score :

- des films appréciés sont masqués ;
- seules les notes d’apprentissage servent d’ancres positives et négatives ;
- l’audit mesure combien de films appréciés réapparaissent dans les 20, 50 et
  100 premiers candidats ;
- il mesure aussi la part de films moins aimés introduite dans ces listes.

Un audit à cinq découpages suffit pour le premier contrôle. Dix découpages ne
servent qu’à confirmer une différence incertaine. L’audit ne demande aucune
nouvelle note et ne modifie jamais les préférences ou le moteur actif.

## Mise à jour depuis la 0.8

1. Fermer complètement la fenêtre noire CineProfile.
2. Extraire les fichiers 0.9 par-dessus le dossier existant.
3. Conserver `data/`, `.env` et `.venv`.
4. Relancer `Lancer CineProfile.bat`.
5. Lancer une recherche **Normale**, **Valeurs sûres**, sur la période voulue.

Il n’est pas nécessaire de réimporter le CSV ou de refaire l’enrichissement
historique. Le profil et le cache du modèle sont recalculés automatiquement une
fois. La première recherche peut enrichir davantage de candidats que la 0.8 et
être plus longue ; les fiches sont ensuite réutilisées.

## Contrôles de livraison

La livraison 0.9.0 a été vérifiée avec :

- analyse statique sans erreur ;
- 65 tests unitaires et d’intégration ;
- les mêmes 65 tests avec tous les avertissements traités comme erreurs ;
- un audit de charge de 1 618 notes, 5 découpages et 6 variantes personnelles ;
- contrôle que l’audit ne modifie pas la base source ;
- test du lancement Streamlit ;
- test de l’archive extraite avant livraison.

# CineProfile 0.9.1 — envie de regarder et satisfaction prévue

## Le problème corrigé

La 0.9 répondait correctement à la question :

> Si ce film est regardé, quelle est la probabilité de lui attribuer au moins
> 8/10 ?

Cette probabilité ne répond toutefois pas à une autre question indispensable :

> Est-ce que le sujet, l’affiche, la signature et les personnes donnent envie
> de lancer ce film maintenant ?

Dans l’historique actuel, environ 20,5 % des films reçoivent au moins 8/10.
Une probabilité de 30 % constitue donc une amélioration réelle de 9,5 points,
mais elle ne signifie ni « 30 % compatible » ni « seulement 30 % intéressant ».
La 0.9.1 supprime cette ambiguïté.

## Les deux axes

### 1. Indice d’envie

L’indice d’envie va de 0 à 100. Ce n’est pas une probabilité. Il part d’un
niveau neutre, puis explique chaque impact positif ou négatif.

Accroches conseillées par défaut :

| Facteur | Effet |
|---|---:|
| Réalisateur ou scénariste réellement apprécié | fort |
| Acteurs principaux familiers | modéré |
| Suite directe d’un film noté au moins 8/10 | fort |
| Comédie noire ou satire | fort |
| Néo-noir ou mystère criminel | modéré |
| Western | modéré |
| Thriller | faible |

Freins conseillés par défaut :

| Facteur | Effet |
|---|---:|
| Biographie ou biopic | modéré |
| Sujet principalement historique | modéré |
| Biographie musicale ou film de chanteur | fort |
| Sport, boxe, combat ou UFC | fort |
| Animation | modéré |
| Suite, surtout troisième épisode ou davantage | modéré à fort |
| Langue peu familière sans autre accroche | faible |
| Équipe principale sans repère connu | faible |

Une étiquette large ne suffit jamais à créer une forte envie. Par exemple,
« Thriller » apporte peu à lui seul ; une comédie noire de Coen ou une suite
d’un film noté 8/10 apporte un signal beaucoup plus précis. Les facteurs
restent modifiables dans **Ajuster le profil → Ce qui me donne envie de lancer
un film**. Choisir « Neutre » neutralise réellement le réglage par défaut.

### 2. Chance d’un 8+

Le moteur v0.9 continue d’estimer la probabilité d’attribuer au moins 8/10,
avec le voisinage local et le modèle global déjà validés. La carte affiche :

- la probabilité ;
- le taux personnel habituel ;
- le gain ou la perte en points ;
- la note personnelle prévue et sa fourchette.

Le libellé « chance prudente » est abandonné parce qu’il était interprété comme
un pourcentage global de compatibilité.

## Ordre conseillé

La satisfaction est d’abord convertie en indice relatif au taux personnel
habituel. L’ordre combine ensuite :

- 60 % d’envie ;
- 40 % de satisfaction relative.

La combinaison est une moyenne harmonique pondérée :

```text
ordre = 1 / (0,60 / envie + 0,40 / satisfaction_relative)
```

Cette formule est volontairement exigeante : un biopic musical prédit à 35 %
ne passe plus devant une comédie noire de signature connue à 30 % uniquement
à cause de cinq points de probabilité. À l’inverse, un film très attirant mais
fortement associé à des contre-exemples ne devient pas artificiellement une
valeur sûre.

## Panel de régression

Les tests reproduisent les cas décrits pendant l’évaluation :

- une comédie noire de Coen doit obtenir une envie élevée ;
- un drame historique générique reste sous le niveau neutre, même avec une
  chance d’un 8+ supérieure ;
- un sujet UFC/combat reçoit un frein fort ;
- un biopic musical reçoit deux freins distincts ;
- une suite directe d’un film noté 8/10 reçoit une accroche compensant
  partiellement l’usure des suites ;
- un réglage utilisateur neutre remplace réellement un défaut non neutre ;
- les traces de récupération survivent au cache des fiches.

Le panel synthétique donne notamment l’ordre attendu suivant :

1. comédie noire de signature connue ;
2. suite d’un épisode apprécié ;
3. animation dont le premier épisode a été apprécié ;
4. drame historique générique ;
5. sujet de sport/combat ;
6. biopic musical sans autre accroche.

## Diagnostic 0.9.1

Le JSON passe au schéma 4 et ajoute pour chaque candidat :

- l’indice, le niveau et la confiance d’envie ;
- les accroches et les freins avec leur impact ;
- la chance d’un 8+, son taux de base, son gain et son rapport au taux de base ;
- l’indice relatif de satisfaction ;
- le score combiné utilisé par l’ordre conseillé ;
- les scores de récupération, y compris après lecture du cache.

Les contrôles automatiques signalent notamment une chance d’un 8+ élevée
associée à une envie faible, l’absence de film réellement attirant ou la
répétition d’un même frein dans au moins cinq des dix premiers résultats.

## Mise à jour

1. Fermer complètement la fenêtre noire CineProfile.
2. Extraire la 0.9.1 par-dessus le dossier existant.
3. Conserver `data/`, `.env` et `.venv`.
4. Relancer `Lancer CineProfile.bat`.
5. Lancer une recherche **Normale · Valeurs sûres**.
6. Comparer d’abord l’ordre conseillé, puis essayer les tris **Indice d’envie**
   et **Chance d’un 8+** pour comprendre les écarts.
7. Télécharger le diagnostic si un résultat reste incohérent.

Aucun réimport, enrichissement historique, audit ou nouvelle notation n’est
nécessaire. Le modèle v0.9 existant est réutilisé ; la nouvelle couche d’envie
est calculée immédiatement sur les fiches déjà mises en cache.

## Contrôles de livraison

La livraison 0.9.1 est vérifiée avec :

- analyse statique sans erreur ;
- 68 tests unitaires et d’intégration ;
- contrôle des dépendances Python ;
- panel de régression des préférences réelles ;
- test de conservation des traces après cache ;
- compilation de tous les modules ;
- démarrage Streamlit ;
- test d’une archive fraîchement extraite.

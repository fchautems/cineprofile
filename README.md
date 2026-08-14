# CineProfile

Version actuelle : **0.13.1**. L’écran permanent **Ma liste** rassemble les
films marqués à voir, écartés, vus et envoyés à Radarr. Les décisions se font
directement sur les cartes, sont réversibles, et restent visibles dans la liste.

CineProfile transforme un export `ratings.csv` d’IMDb en trois éléments
réutilisables :

1. une base SQLite locale qui conserve les titres, genres, personnes, rôles,
   mots-clés et disponibilités ;
2. un rapport détaillé `profile_report.html` accompagné de sa version
   structurée `profile.json` ;
3. un moteur de suggestions récentes, explicable et capable d’exclure tout ce
   qui a déjà été noté.

Le CSV n’est donc pas le modèle : c’est l’historique initial. La base enrichie
devient la mémoire durable de l’outil.

## Mise à jour 0.13.1 — actions directes dans Ma liste

**Ma liste** abandonne le tableau et le formulaire « sélectionner puis
modifier ». Elle réutilise les cartes de **Suggestions**, avec les mêmes quatre
actions compactes : 👍 à voir, 👎 pas pour moi, 👁 déjà vu et ➤ envoyer à
Radarr. Un deuxième clic sur l’une des trois décisions personnelles l’annule ;
ces trois états sont mutuellement exclusifs.

Les filtres deviennent une rangée de boutons à icônes avec compteurs : liste
complète, à voir, déjà vus, pas pour moi et Radarr. L’icône Radar confirme
uniquement que la demande a été envoyée à Radarr ; elle ne prétend toujours pas
que le téléchargement est terminé.

## Mise à jour 0.13.0 — interface simplifiée et suggestions persistantes

L’application s’organise maintenant autour de quatre espaces : **Suggestions**,
**Ma liste**, **Mon profil** et **Réglages**. Les connexions, l’import IMDb,
l’enrichissement et les diagnostics ne prennent plus la place de l’usage
quotidien.

La dernière sélection calculée est conservée dans la base et réapparaît après
un redémarrage, avec ses paramètres et son diagnostic. Il devient donc possible
de consulter, filtrer et décider sans reconstruire inutilement le vivier TMDB.
Une ancienne sélection 0.12 est également restaurée lorsqu’elle est compatible
avec le profil courant.

Le libellé **Downloaded** est remplacé par **Envoyé à Radarr** : CineProfile
indique que Radarr a accepté la recherche, sans prétendre qu’un fichier a déjà
été importé. Les états techniques détaillés viendront dans la suite de la 0.13.

## Correctif 0.12.2 — recherche Radarr explicite

Après avoir ajouté ou retrouvé le film dans Radarr, CineProfile envoie
maintenant une commande `MoviesSearch` dédiée avec l’identifiant interne du
film. Le statut local **Envoyé à Radarr** est enregistré lorsque Radarr accepte
cette commande. En cas de refus, le message précise que le film est présent
dans Radarr mais que sa recherche n’a pas pu être lancée ; un nouveau clic
permet alors de retenter la recherche sans ajouter le film en double.

Radarr choisit ensuite une version conforme aux indexeurs, au profil de qualité
et au client de téléchargement configurés. Le lancement de la recherche ne
garantit donc pas qu’une version compatible existe immédiatement.

## Correctif 0.12.1 — connexions persistantes

TMDB et Radarr se configurent maintenant ensemble dans un panneau compact
**Connexions**. Un seul bouton enregistre atomiquement toutes les valeurs dans
le fichier local `.env` : la configuration précédente reste intacte si la
validation de Radarr ou l’écriture échoue. Une fois enregistrées, les clés ne
restent plus affichées ; CineProfile indique simplement **TMDB configuré** et
**Radarr connecté**, puis reconnecte automatiquement Radarr au lancement.

Le dossier de films et le profil de qualité Radarr sont également mémorisés.
Si `.env` est accidentellement un dossier au lieu d’un fichier, l’interface
l’explique explicitement au lieu de tenter une écriture incohérente.

## Mise à jour 0.12.0 — Mes films et Radarr

L’ancêtre de l’onglet **Ma liste**, alors nommé **Mes films**, conserve les décisions prises depuis les suggestions,
même lorsqu’un film disparaît de la liste courante. Les vues **Tous**, **À
voir**, **Déjà vus**, **Pas intéressé** et les envois à Radarr permettent de les
retrouver, puis de modifier ou d’annuler leur statut.

Dans **Réglages**, renseigner l’adresse locale de Radarr et sa clé API.
CineProfile valide l’accès et propose le dossier racine ainsi que le profil de
qualité récupérés depuis Radarr.

Le bouton **Envoyer à Radarr** d’une recommandation ajoute ensuite le film, active sa
surveillance et lance sa recherche. **Envoyé à Radarr** signifie ici « envoyé à
Radarr au moins une fois » : la présence du fichier n’est pas vérifiée. Annuler
ce statut agit uniquement dans CineProfile, sans supprimer le film de Radarr ni
interrompre une recherche ou un téléchargement. Le bouton redevient alors
utilisable. Les succès, doublons déjà présents et échecs restent horodatés dans
l’historique local.

Le rapport d’import IMDb distingue désormais les titres nouveaux, réellement
modifiés, inchangés et ignorés. Il indique aussi combien d’enrichissements TMDB,
de statuts et de corrections du profil ont été conservés.

## Mise à jour 0.10.1 — valeurs sûres réellement pertinentes

La liste **Valeurs sûres** ne confond plus « très bien noté par le public » et
« bon choix pour toi ». La note publique corrigée reste le signal dominant,
mais un garde-fou personnel empêche désormais les films sans envie ni
compatibilité suffisante d’occuper le haut de la liste.

Cette correction traite notamment le biais révélé par le diagnostic réel :
les validations historiques comparaient des films que l’utilisateur avait
déjà choisi de regarder, alors que la recherche en production analyse aussi
des films populaires sans rapport avec ses goûts.

La liste **Découvertes pour toi** conserve sa logique de la version 0.10.0.
Seules les dix premières valeurs sûres en sont retirées, au lieu de vingt, afin
de préserver davantage de bonnes pistes personnelles. Les détails et les
contrôles sont décrits dans `EVOLUTION_v0.10.1.md`.

## Mise à jour 0.10.0 — valeurs sûres et découvertes séparées

Une seule recherche produit désormais deux onglets indépendants :

- **Valeurs sûres** : classement par note publique corrigée, fiabilité et
  nombre de votes, sans aucune influence du profil personnel ;
- **Découvertes pour toi** : classement par affinité, envie, proximité des
  histoires et diversité.

Les dix premières valeurs sûres sont retirées des découvertes afin que les
deux listes soient réellement complémentaires. Le choix historique du mode de
classement disparaît ; passer d’un onglet à l’autre est instantané et ne
relance pas TMDB.

Il suffit de fermer CineProfile, remplacer les fichiers en conservant `.venv`,
`.env` et `data`, puis relancer `Lancer CineProfile.bat`. Aucun nouvel import ni
enrichissement complet n’est nécessaire.

Les règles exactes sont décrites dans `EVOLUTION_v0.10.0.md`.

## Mise à jour 0.9.1 — envie et satisfaction enfin séparées

La 0.9.1 corrige le problème révélé par la liste réelle : un film pouvait
avoir une chance raisonnable d’obtenir 8/10 **s’il était regardé**, tout en ne
donnant aucune envie de le lancer. Un seul pourcentage ne pouvait pas répondre
à ces deux questions.

- l’**indice d’envie** mesure désormais l’attrait avant visionnage ;
- la **chance d’un 8+** conserve son sens statistique : probabilité d’attribuer
  au moins 8/10, comparée au taux personnel habituel ;
- l’ordre conseillé combine les deux par une moyenne harmonique pondérée
  (60 % envie, 40 % satisfaction relative). Un score élevé sur un seul axe ne
  peut donc plus cacher un score faible sur l’autre ;
- les accroches utilisent les signatures réellement appréciées, les acteurs
  principaux familiers, les épisodes précédents, la comédie noire, le néo-noir,
  le western et le thriller ;
- les freins distinguent biopic, sujet historique, biographie musicale,
  sport/combat, animation, usure des suites, langue peu familière et équipe
  sans repère connu ;
- ces facteurs sont tous visibles et modifiables dans **Ajuster le profil**,
  sans changer les anciennes notes ;
- les cartes expliquent séparément les accroches, les réserves, l’envie, la
  chance d’un 8+ et le gain par rapport à l’habitude personnelle ;
- les tris proposent désormais l’ordre conseillé, l’envie et la chance d’un
  8+ comme trois choix distincts ;
- le diagnostic conserve les traces de récupération même lorsque la fiche
  provient du cache, et contrôle les freins réellement répétés plutôt que de
  tirer des conclusions hâtives d’un grand genre comme « Drame ».

Il ne faut **ni réimporter le CSV, ni refaire l’enrichissement, ni relancer
l’audit**. Fermer CineProfile, remplacer les fichiers, relancer puis effectuer
une recherche **Normale · Valeurs sûres**. Les réglages d’envie conseillés sont
déjà actifs ; l’onglet de préférences sert seulement à les corriger si besoin.

Les règles exactes et le panel de validation sont décrits dans
`EVOLUTION_v0.9.1.md`.

## Mise à jour 0.9.0 — recherche réellement personnelle

La 0.9 corrige la cause principale observée dans les diagnostics 0.8 : le
vivier provenait surtout des listes populaires TMDB et la note publique restait
le point de départ de la prédiction. Les nouveaux calculs ne se contentent pas
de changer quelques poids.

- les films proches des histoires aimées — et éloignés des contre-exemples —
  sont désormais récupérés avant l’enrichissement complet, puis mélangés aux
  autres sources ;
- la chance de plaire combine un voisinage local de tes films (65 %) et un
  modèle personnel global (35 %). Ces deux valeurs sont affichées dans le
  détail de chaque suggestion ;
- la note TMDB a **0 % d’influence directe sur l’affinité**. Elle sert de seuil
  de fiabilité et peut départager légèrement les modes exploratoires, sans
  dépasser 10 % de leur score d’ordre ;
- le modèle global apprend directement tes notes autour de ta moyenne, au lieu
  de corriger une note publique déjà élevée ;
- les notes basses comptent comme de vrais contre-exemples. En mode Valeurs
  sûres, un genre appris comme négatif ne peut plus saturer le haut de la liste
  lorsque des alternatives de niveau comparable existent ;
- les types IMDb français (`Série TV`, `Mini-série`, `Épisode`, etc.) sont
  normalisés et exclus d’un moteur consacré aux films ;
- les résultats indiquent `Correspondance solide`, `Piste plausible` ou
  `Solution de repli`. CineProfile avertit clairement lorsqu’aucun candidat
  solide n’a été trouvé au lieu de maquiller un repli en bonne recommandation ;
- l’audit mesure aussi la **récupération des candidats** : rappel des films
  appréciés masqués et part de films peu aimés introduite dans le haut du
  vivier.

Il ne faut ni réimporter `ratings.csv`, ni relancer l’enrichissement complet,
ni renoter quoi que ce soit. Après remplacement des fichiers, lancer
directement une recherche normale en mode **Valeurs sûres**. Le premier passage
peut être plus long, car le vivier est plus large ; les recherches suivantes
réutilisent le cache. L’audit à 5 découpages sert ensuite à mesurer la
récupération, pas à rendre la nouvelle formule active.

Les choix techniques et les règles exactes sont consignés dans
`EVOLUTION_v0.9.0.md`.

## Mise à jour 0.8.0 — challengers mesurés

Cette version améliore les suggestions sans demander une seule nouvelle note
et sans remplacer automatiquement le moteur stable.

- l’audit compare séparément six variantes : métadonnées, syntaxe TF‑IDF,
  sémantique multilingue profonde, puis leurs combinaisons ;
- chaque variante choisit sa régularisation dans une validation interne qui ne
  voit jamais les films du groupe test. La recherche couvre maintenant les
  valeurs 10 à 800 et signale un optimum collé à la borne ;
- le top 10 devient l’objectif principal, avec garde-fous sur la précision,
  le classement global, l’erreur de note, la calibration, la répétabilité et
  les notes les plus récentes ;
- l’audit n’active rien de lui-même. Un bouton n’apparaît que si un challenger
  passe tous les contrôles ; l’activation conserve exactement la variante et
  la régularisation validées. Le retour à la v0.6 reste immédiat ;
- les vecteurs profonds proviennent d’un modèle multilingue préentraîné local.
  Aucune note privée n’est envoyée à un service externe. Le cache est versionné
  par modèle et par empreinte du texte ;
- le vivier combine popularité, qualité publique, favoris personnels, films
  similaires, réalisateurs/scénaristes, acteurs confirmés, thèmes et genres.
  Un tourniquet entre sources empêche la popularité de consommer tout le budget
  d’analyse ;
- les graines sont choisies sur l’écart entre la note personnelle et la note
  publique fiabilisée, puis diversifiées par genres et créateurs ;
- **Valeurs sûres** est le mode par défaut et classe strictement par chance
  prudente. Équilibré et Découvertes ajoutent une diversité contrôlée sans
  promouvoir un film situé plus de huit points sous le meilleur choix restant ;
- le diagnostic JSON et le journal consignent le moteur unique, la variante,
  les sources, les scores de sûreté/classement et les contrôles automatiques.

Il ne faut ni réimporter `ratings.csv`, ni relancer l’enrichissement TMDB, ni
renoter des films. Après la mise à jour, lancer l’audit complet dans l’onglet
Profil. Le premier audit sémantique peut télécharger environ 220 Mo ; les
suivants réutilisent le cache.

## Mise à jour 0.7.2 — stabilisation

Cette version ne cherche pas à imposer un nouveau moteur. Elle fiabilise
l’application existante, son audit et ses outils de diagnostic.

- `app.py` est passé de plus de 2’200 lignes à un orchestrateur d’environ
  200 lignes. Import, profil, audit, catalogue, préférences, recherche,
  filtres et cartes de résultats vivent maintenant dans des modules séparés ;
- toutes les connexions SQLite sont fermées de façon déterministe. L’audit
  sauvegarde son rapport avant de supprimer sa copie temporaire, ce qui corrige
  l’échec Windows `[WinError 32]` / `[WinError 267]` après la dernière étape ;
- l’audit transmet désormais aux moteurs exactement la représentation enrichie
  utilisée pendant l’apprentissage. Les mesures v0.6/v0.7 produites par
  l’ancien audit 0.7.1 ne doivent pas servir à choisir des poids ;
- le sélecteur de genres exclus propose toute la liste TMDB et l’exclusion est
  contrôlée avant l’analyse détaillée puis avant le classement ;
- une durée inconnue ne traverse plus silencieusement une plage personnalisée ;
- les types IMDb `tv`, `tvSeries` et épisodes sont tous exclus des graines de
  recommandations de films ;
- les erreurs TMDB globales (authentification, quota, réseau) interrompent
  proprement le lot au lieu de marquer chaque titre comme absent. Les fiches
  réellement absentes (`404`) restent ignorées individuellement ;
- lorsqu’IMDb et TMDB sont disponibles, la référence publique la mieux étayée
  est choisie. Une note TMDB sur quelques votes ne remplace plus une note IMDb
  fondée sur des milliers de votes ;
- l’import refuse les notes hors de 1–10, déduplique les identifiants, accepte
  UTF‑8 et CP1252 et ne construit plus de requête SQLite trop grande ;
- le diagnostic téléchargé recalcule ses contrôles sur le tri et les résultats
  réellement affichés, pas sur l’ordre interne antérieur aux filtres ;
- le lanceur met à jour le code local sans rechercher de nouvelles dépendances
  à chaque démarrage. Il ne réinstalle les composants que si `pip check`
  détecte un manque réel.

La base, le jeton TMDB et les enrichissements sont conservés. Le profil et le
cache du modèle personnel sont recalculés une fois car leur version passe en
0.7.2. Il ne faut ni réimporter le CSV, ni relancer l’enrichissement, ni renoter
des films.

## Mise à jour 0.7.1

Cette version corrige la surévaluation observée sur des films TMDB très récents
ou encore peu notés :

- l’apprentissage et les nouveaux candidats utilisent désormais la même
  référence publique TMDB lorsque celle-ci est disponible ;
- la note TMDB brute est fiabilisée par une moyenne bayésienne : avec peu de
  votes, elle reste proche de la moyenne générale, puis gagne progressivement
  en poids ;
- la décennie et l’année de sortie ne sont plus considérées comme des goûts
  personnels ; la récence reste seulement un critère séparé de découverte ;
- le pourcentage affiché est rendu prudent selon la couverture des données, la
  présence du résumé, la fiabilité publique et l’erreur mesurée du modèle ;
- la calibration des pourcentages est contrôlée hors échantillon avec un score
  de Brier et un tableau « pourcentage annoncé / fréquence observée » ;
- le détail d’une suggestion distingue la note TMDB brute, la note publique
  fiabilisée, la correction personnelle et la note prévue ;
- chaque recherche écrit un journal local rotatif dans `data/logs` ;
- le bouton **Tester le moteur et télécharger le diagnostic** produit un JSON
  directement joignable à une conversation, sans jeton TMDB ni historique
  complet.

Le profil est recalculé automatiquement au premier lancement. La base enrichie
est réutilisée : aucun CSV à réimporter et aucun film à renoter.

### Outil d’audit de la 0.7.1

L’onglet **Profil** contient maintenant un panneau **Audit complet du moteur**.
Ce panneau n’est pas une nouvelle version du moteur : il mesure la version
installée avant toute modification supplémentaire.

- la base complète est copiée dans un fichier temporaire et le calcul ne
  travaille que sur cette copie ;
- une empreinte de toutes les tables vérifie à la fin que notes,
  enrichissements, préférences, feedback, caches et modèles sont inchangés ;
- 5 ou 10 découpages indépendants cachent 25 % des notes à chaque essai ;
- la note publique seule, la v0.5, la v0.6 et la v0.7 sont comparées
  séparément sur exactement les mêmes films cachés ;
- un contrôle chronologique apprend sur les anciennes notes et teste les 20 %
  les plus récentes ;
- une courbe d’apprentissage compare automatiquement plusieurs tailles
  d’historique, notamment autour de 400, 700 et 1’000 films, au lieu de fixer
  arbitrairement 1’000 ;
- le journal détaillé est écrit dans `data/logs/cineprofile.log` et le bouton
  final télécharge un rapport JSON partageable.

Il n’est pas nécessaire de réimporter le CSV, de relancer TMDB ou de renoter
quoi que ce soit. Pour un premier contrôle, choisir **Complet — 5
découpages**. Le mode 10 découpages sert à confirmer un résultat incertain.

## Mise à jour 0.7.0

- la v0.6 reste le moteur champion tandis qu’un challenger indépendant v0.7
  apprend des **îlots de goût** positifs et négatifs ;
- un îlot représente une combinaison récurrente d’histoires, de styles, de
  genres, de thèmes et de personnes, ce qui évite de confondre fréquence et
  préférence ;
- v0.6 et v0.7 prédisent séparément les mêmes notes masquées : leurs scores ne
  sont jamais mélangés ;
- la v0.7 n’est activée que si elle gagne clairement sur au moins deux mesures
  (classement global, qualité du top 50 et erreur de note), sans régression
  significative ni sur l’historique complet ni sur les notes récentes ;
- l’onglet Profil indique le moteur actif, montre le comparatif complet et
  permet d’inspecter les films représentatifs de chaque îlot ;
- chaque suggestion v0.7 explique de quel îlot apprécié et de quel îlot moins
  aimé elle est la plus proche ;
- aucun nouvel avis n’est demandé : la base, les enrichissements TMDB et les
  1’600 notes déjà importées sont réutilisés.

Au premier lancement de la 0.7, CineProfile recalcule automatiquement les deux
moteurs. Il ne faut ni réimporter le CSV, ni relancer l’enrichissement TMDB.
Si la v0.7 ne prouve pas son avantage, la v0.6 reste active automatiquement.

## Mise à jour 0.6.0

- un modèle personnel est appris exclusivement à partir des notes IMDb déjà
  présentes : aucune nouvelle notation n’est demandée ;
- la validation croisée masque successivement des films, apprend sur le reste
  et vérifie les prédictions sur les notes cachées ;
- la version 0.5 et le nouveau modèle sont comparés sur le même historique ;
- le nouveau moteur n’est activé que s’il passe le contrôle hors échantillon,
  afin d’éviter une dégradation silencieuse ;
- l’ancien indice arbitraire devient alors une probabilité calibrée
  d’attribuer au moins 8/10, accompagnée d’une note personnelle prévue, d’une
  fourchette et d’un niveau de confiance ;
- le modèle apprend l’écart entre la note personnelle et la note publique à
  partir des genres, thèmes, descriptions, personnes, langues, pays, durées et
  périodes ;
- un contrôle chronologique supplémentaire teste, lorsque les dates le
  permettent, les évaluations les plus récentes ;
- les films jamais vus ne sont jamais utilisés comme exemples négatifs.

Au premier lancement, CineProfile recalcule automatiquement le profil et
construit ce modèle local. Il réutilise la base et les enrichissements
existants : il ne faut ni réimporter le CSV, ni relancer TMDB, ni renoter les
films.

## Mise à jour 0.5.0

- l’horreur est exclue par défaut avant l’analyse détaillée, avec un réglage
  pour la réactiver ;
- l’affinité personnelle et l’ordre conseillé sont désormais affichés et
  expliqués séparément ; le tri par défaut utilise l’affinité ;
- la sémantique combine résumé, accroche, genres et thèmes, puis oppose les
  ressemblances avec les films aimés aux films mal notés ou refusés ;
- la proximité sémantique passe à 40 % du signal personnel ;
- l’équipe passe à 15 % : les acteurs ne comptent que parmi les cinq rôles
  principaux et lorsqu’ils ont été observés sur au moins cinq films ;
- réalisation et scénario dominent maintenant le signal humain ;
- le niveau d’exploration par défaut est ramené à 15 %.

La base, le jeton TMDB et les données enrichies sont conservés. Au premier
calcul de cette version, les textes déjà présents sont réencodés
automatiquement avec les genres et les thèmes ; aucun appel TMDB supplémentaire
n’est nécessaire pour cette opération.

## Mise à jour 0.4.1

La 0.4.1 empêche une série IMDb reliée à TMDB d'être interrogée comme un
film. Une fiche de recommandation TMDB absente est désormais ignorée sans
interrompre toute la recherche.

- recherche issue de plusieurs viviers : sorties, recommandations autour des
  films préférés, personnes appréciées et genres affinitaires ;
- analyse sémantique multilingue des résumés, comparée aux films aimés **et**
  rejetés ; les vecteurs restent locaux et sont mis en cache ;
- genre ramené à 10 % afin de distinguer exposition et préférence ;
- minimum de votes TMDB adaptatif à l’âge du film, avec trois niveaux de
  fiabilité dans les réglages avancés ;
- périodes de 1 à 20 ans, toutes les années ou intervalle personnalisé ;
- tri par recommandation, affinité, confiance, note corrigée, votes, date,
  durée ou titre ;
- tous les films analysés restent filtrables et l’affichage se fait par blocs
  de 20 ;
- affiche cliquable vers IMDb, réalisateur et acteurs principaux visibles ;
- onglet de correction du profil pour favoriser, réduire ou exclure genres,
  thèmes et personnes ;
- retours locaux « À voir », « Pas intéressé » et « Déjà vu » ;
- cache des fiches candidates et commande facultative de rafraîchissement des
  anciennes traductions en `fr-FR`.

Le premier calcul sémantique télécharge environ 220 Mo. Il fonctionne ensuite
localement et ne recalcule que les descriptions nouvelles ou modifiées. Si ce
téléchargement échoue, CineProfile conserve automatiquement un repli lexical
et reste utilisable.

La mise à jour ne demande **ni réimport du CSV, ni nouvel enrichissement** :
copier les nouveaux fichiers, fermer l’ancienne fenêtre noire et relancer
`Lancer CineProfile.bat` suffit.

## Mise à jour 0.3.0

- un même film garde désormais le même **indice d’affinité**, quels que soient
  le niveau d’exploration et les filtres de recherche ;
- l’exploration agit seulement sur le classement, pas sur l’indice affiché ;
- chaque fiche détaille les quatre signaux personnels et leurs poids ;
- les affiches et les résumés français de France (`fr-FR`) sont affichés ;
- le profil est recalculé automatiquement après import ou enrichissement, ce
  qui rend les thèmes immédiatement disponibles ;
- les options Streamlit réservées au développement sont masquées.

L’indice personnel utilise 25 % de genres, 25 % de thèmes, 30 % de personnes
(réalisation, scénario, interprétation et équipe technique) et 20 % de
proximité textuelle. La qualité publique, la récence et la nouveauté servent
uniquement à ordonner les suggestions. L’indice est un repère comparatif, pas
une probabilité.

## Correctifs 0.2

- la fenêtre de sortie est maintenant réellement transmise à TMDB ;
- une validation locale interdit tout résultat en dehors de cette fenêtre ;
- le vivier est trié par popularité dans la période choisie, afin de couvrir
  toute l’année plutôt que les quelques sorties les plus proches ;
- le score minimal n’élimine plus de suggestions par défaut ;
- le filtre de durée utilise une plage fixe de 30 à 300 minutes ;
- un panneau de diagnostic indique combien de titres ont été trouvés, exclus,
  enrichis et conservés.

### Mise à jour 0.2.1

Fermer complètement CineProfile avant de remplacer ses fichiers. Streamlit
conserve les modules Python en mémoire tant que sa fenêtre noire reste ouverte.
Le lanceur rattache désormais explicitement l’environnement Python au dossier
depuis lequel il est exécuté à chaque démarrage, et
l’interface détecte une éventuelle combinaison de fichiers de versions
différentes sans afficher de traceback technique.

## Ce que le profil mesure

- distribution et évolution des notes ;
- préférences et rejets par genre, décennie, langue et pays ;
- affinités avec réalisateurs, scénaristes et acteurs ;
- thèmes récurrents via les mots-clés et les descriptions ;
- écart entre la note personnelle et la note du public, afin de distinguer les
  vrais goûts personnels d’un simple attrait pour les films très bien notés ;
- fiabilité de chaque signal : un réalisateur vu huit fois pèse davantage
  qu’un acteur apparu une seule fois.

## Démarrage local

```bash
python -m venv .venv
```

Sous Windows :

Double-cliquer sur **`Lancer CineProfile.bat`**. Au premier lancement, le
script crée l’environnement Python et installe les composants. Les lancements
suivants ouvrent directement l’interface dans le navigateur.

Le démarrage manuel reste possible :

```powershell
.venv\Scripts\activate
pip install -e .
streamlit run app.py
```

Ouvrir ensuite `http://localhost:8501`.

## Démarrage sur un NAS avec Docker

Copier `.env.example` vers `.env`, ajouter le jeton TMDB, puis :

```bash
docker compose up -d --build
```

L’interface sera disponible sur `http://ADRESSE_DU_NAS:8501`. Le dossier
`data/` est monté comme volume persistant.

## Enrichissement TMDB

Créer gratuitement un compte TMDB et renseigner le **Read Access Token** dans
`.env`. CineProfile utilise l’identifiant IMDb présent dans le CSV pour éviter
les correspondances fragiles sur le titre. Chaque film est enrichi avec les
crédits, mots-clés, pays, sociétés de production, résumé, affiche et
disponibilités en Suisse. Le traitement est mis en cache et peut reprendre
après une interruption.

Sans jeton TMDB, l’import et un premier profil fondé sur les colonnes IMDb
restent disponibles.

Le bouton **Enregistrer** du panneau **Connexions** mémorise TMDB et Radarr dans
le fichier local `.env`. Ce fichier est ignoré par Git et exclu des archives de
mise à jour. Les commandes **Oublier TMDB** et **Oublier Radarr** suppriment
uniquement la connexion choisie.

## Données et limites

- Les données restent dans la base locale.
- Les jeux de données IMDb sont destinés à un usage personnel et non
  commercial selon leurs conditions.
- Les disponibilités proviennent de TMDB/JustWatch et nécessitent leur
  attribution.
- Aucun téléchargement de contenu commercial non autorisé n’est intégré.
  Une passerelle Transmission pourra être ajoutée ultérieurement pour les
  torrents légaux, libres de droits ou correspondant à des contenus possédés.

## Structure

```text
app.py                      interface Streamlit
cineprofile/imdb_import.py  lecture robuste du CSV IMDb
cineprofile/tmdb.py         enrichissement avec cache SQLite
cineprofile/profile.py      modèle de goûts et exports
cineprofile/personal_model.py anciens moteurs de comparaison v0.5 à v0.7
cineprofile/hybrid_model.py modèle global personnel v0.9 et variantes d’audit
cineprofile/semantic.py     voisinage local positif/négatif des histoires
cineprofile/audit.py        validation imbriquée et audit de récupération
cineprofile/candidate_pool.py récupération personnelle et multi-source
cineprofile/recommender.py  fusion locale/globale et suggestions expliquées
cineprofile/watch_interest.py attrait avant visionnage et freins explicites
cineprofile/ranking.py      prudence, diversité et pénalités de saturation
data/cineprofile.db         base créée au premier lancement
```

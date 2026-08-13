# CineProfile 0.12.2 — recherche Radarr explicite

## Problème corrigé

La version 0.12.1 transmettait `searchForMovie: true` uniquement dans les
options d’ajout. Selon la version ou la configuration de Radarr, le film était
bien créé et surveillé, mais aucune recherche n’était réellement mise en file.

## Nouveau déroulement

1. CineProfile retrouve le film existant ou l’ajoute à Radarr.
2. Il récupère l’identifiant interne attribué par Radarr.
3. Il envoie `POST /api/v3/command` avec la commande `MoviesSearch` et
   `movieIds: [identifiant]`.
4. Le statut local **Downloaded** n’est enregistré qu’après acceptation de
   cette commande par Radarr.

Le même mécanisme est appliqué aux films déjà présents : un second clic ne les
duplique pas, mais relance explicitement leur recherche.

Si Radarr refuse la commande, CineProfile conserve une tentative en échec et
affiche une erreur explicite. Le film peut déjà être présent côté Radarr ; il
suffit alors de corriger sa configuration (indexeurs ou client de
téléchargement, par exemple) puis de cliquer de nouveau sur **Download**.

## Portée du statut Downloaded

Le statut signifie toujours « demande acceptée par Radarr ». CineProfile ne
surveille pas encore l’arrivée effective d’un fichier. Une recherche acceptée
peut ne rien télécharger si aucune version ne respecte les règles de Radarr.

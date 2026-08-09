# CineProfile 0.10.1 — valeurs sûres réellement pertinentes

La version 0.10.1 corrige un biais de sélection dans la liste **Valeurs
sûres**.

## Cause

Les validations historiques avaient montré que la note publique classait bien
les films parmi ceux que l’utilisateur avait choisi de regarder. Cette
conclusion ne pouvait pas être appliquée directement au catalogue de recherche :
celui-ci contient aussi des films populaires, concerts, animations ou
franchises qui peuvent être très bien notés tout en ne suscitant aucune envie.

La version 0.10.0 classait pourtant tout ce vivier uniquement par qualité
publique. Le diagnostic réel pouvait ainsi placer en tête un film noté 8,37
par le public avec seulement 20/100 d’envie personnelle.

## Nouveau classement

Le classement devient public-dominant, mais personnellement contraint :

- 70 % de qualité publique corrigée ;
- 15 % de compatibilité globale ;
- 15 % d’envie avant visionnage.

Avant de comparer ce score, les films sont répartis dans trois niveaux :

1. **Solide pour toi** : qualité publique, compatibilité, envie et note prévue
   atteignent tous un niveau prudent ;
2. **Plausible pour toi** : les mêmes conditions sont légèrement assouplies ;
3. **Qualité publique seulement** : le film reste visible, mais ne peut plus
   passer devant les candidats réellement compatibles.

En l’absence de modèle personnel, CineProfile revient automatiquement au
classement public historique.

## Interface et diagnostic

Les cartes de valeurs sûres affichent maintenant le niveau de compatibilité.
Le détail conserve la note publique brute, la note corrigée et sa fiabilité,
puis expose la note personnelle prévue et l’indice d’envie utilisés comme
garde-fous.

Le diagnostic téléchargé reprend désormais le rang et les champs du véritable
onglet affiché. Il ne présente plus l’ancien ordre interne comme s’il s’agissait
de l’ordre visible.

## Découvertes

Le moteur **Découvertes pour toi** n’est pas modifié. Il conserve l’ordre
personnel et la diversité de la version 0.10.0. Seules les dix premières valeurs
sûres en sont retirées, au lieu de vingt, afin de conserver davantage de pistes
personnelles pertinentes tout en évitant que les deux premiers écrans soient
identiques.

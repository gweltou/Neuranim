# Neuranim
![preview](res/neuranim.gif)

Apprentissage automatique de la marche pour robots virtuels.

## Dépendances
 * Python3
 * Numpy
 * Pygame
 * Box2D

## Mise en marche
### Première étape: Modifier le fichier "parameters.py"
 * Activer ou désactiver le mode 'PRESENTATION'
 * START_POP : nombre d'individus pour la génération 0
 * WINNERS_PER_GENERATION : nombre d'individus selectionnés pour la génération suivante
 * MAX_STEPS : limite de durée de la simulation pour chaque individu
 * END_GEN : nombre de générations au bout duquel la simulation s'interrompra

### Evolution
`python3 evolve.py [genXXX.txt]`

## Types de mutations
Chaque exécution de la fonction Animatronic.mutate provoque la mutation de 2 gènes en moyenne.
Une mutation peut définir une nouvelle valeur (entre -1 et 1) à un gène ou bien le désactiver (valeur définie à 0). Un gène désactivé ne subit plus de mutations et il ne peut donc pas être réactivé. Les désactivations représentent 2% des mutations.

## TODO:
 * Implémenter la nouvelle morphologie (avec perception de l'horizontalité et de la hauteur par rapport au sol)
 * Utiliser des scenarios d'entrainement
 * Texturer le sol
 * Utiliser des arguments pour activer le mode présentation
 * Option de sol accidenté

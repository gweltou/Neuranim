# Neuranim
![preview](res/neuranim.gif)

## Mise en marche
### Première étape: Modifier le fichier "parameters.py"
 * Activer ou désactiver le mode 'PRESENTATION'
 * START_POP : nombre d'individus pour la génération 0
 * WINNERS_PER_GENERATION : nombre d'individus selectionnés pour la génération suivante
 * MAX_STEPS : limite de durée de la simulation pour chaque individu
 * END_GEN : nombre de générations au bout duquel la simulation s'interrompra

### Evolution
`python3 simple_mutate.py [genXXX.txt]`

## TODO:
 * Implémenter la nouvelle morphologie (avec perception de l'horizontalité et de la hauteur par rapport au sol)
 * Utiliser des scenarios d'entrainement
 * Texturer le sol

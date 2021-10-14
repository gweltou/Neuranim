# Neuranim
![preview](res/neuranim.gif)

Apprentissage automatique de la marche pour robots virtuels.

## Dépendances
 * Python3
 * Numpy
 * Matplotlib
 * Pygame
 * Box2D

## Installation
`pip3 install numpy matplotlib pygame box2d`

## Mise en marche
### Première étape: Modifier le fichier "parameters.py"
 * Choisir une morphologie par la variable "ANIMATRONIC"
 * Choisir le nombre de couches intermédiaires et le nombre de neurones par couche (hors biais) avec la variable "HIDDEN_LAYERS"
 * Choisir la fonction d'activation globale avec la variable "ACTIVATION"

### Evolution
Usage:
```
$ python3 evolve.py -h
usage: evolve.py [-h] [-v] [-m MUTATE] [-f FILE] [-t TERRAIN_ROUGHNESS]
                 [-s SAVE_INTERVAL] [-l LIMIT_STEPS] [-p POOL_SIZE]
                 [-w WINNERS_PERCENT] [-e END_GENERATION]

optional arguments:
  -h, --help            show this help message and exit
  -v, --view            enable presentation mode
  -m MUTATE, --mutate MUTATE
                        mutation frequency multiplier (defaults to 2)
  -f FILE, --file FILE  population file
  -t TERRAIN_ROUGHNESS, --terrain_roughness TERRAIN_ROUGHNESS
                        terrain variation in elevation (in percent)
  -s SAVE_INTERVAL, --save_interval SAVE_INTERVAL
                        save population to disk every X generations
  -l LIMIT_STEPS, --limit_steps LIMIT_STEPS
                        max number of steps for each individual trial
                        (defaults to 500)
  -p POOL_SIZE, --pool_size POOL_SIZE
                        size of creature population (defaults to 200)
  -w WINNERS_PERCENT, --winners_percent WINNERS_PERCENT
                        percent of selected individuals per generation
  -e END_GENERATION, --end_generation END_GENERATION
                        limit simulation to this number of generations
                        (defaults to 500)
```

Evolution d'une population d'après les paramètres par défaut

`python3 evolve.py [-v]`

Reprendre l'évolution d'une population depuis un fichier

`python3 evolve.py -f genXXX.txt`

Activer le mode présentation:

`python3 evolve.py -f genXXX.txt -v`

## Types de mutations
Chaque exécution de la fonction Animatronic.mutate provoque la mutation de 2 gènes en moyenne.
Une mutation peut définir une nouvelle valeur (entre -1 et 1) à un gène ou bien le désactiver (valeur définie à 0). Un gène désactivé ne subit plus de mutations et il ne peut donc pas être réactivé. Les désactivations représentent 2% des mutations.

## TODO:
 * Entrainer à sauter
 * Mode présentation : touche "screenshot", placer l'image dans le rép de la population
 * Utiliser des scenarios d'entrainement
 * Cumul du score pour chaque creature

## Quelques idées:
 * Tester réseau neuronal qui s'actualise d'une couche à la fois (pour un RN de x couches intermédiaires, il faudra donc x+1 pas pour le traverser entièrement)

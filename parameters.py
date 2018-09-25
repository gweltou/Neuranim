PRESENTATION = False	# Without display
PRESENTATION = True	# With display ON

DISPLAY = False
BREED = True
if PRESENTATION:
    DISPLAY = True
    BREED = False

START_GEN = 160
END_GEN = 350
STARTING_POPULATION = 400
WINNERS_PER_GENERATION = 15
MAX_STEPS = 500
TARGET = (40, 2)

STARTPOS = (14, 2)
TARGETS = [(-12, 2), (40, 2)]    # Run
#TARGETS = [(16.5, 4), (16.5, 4)]	# Jump
SCORE_MIN = False

NEURON_LAYERS = [8, 16, 16, 4] # including input and output layers
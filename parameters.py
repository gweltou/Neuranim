PRESENTATION = False	# Without display
#PRESENTATION = True	# With display ON

DISPLAY = False
BREED = True
if PRESENTATION:
    DISPLAY = True
    BREED = False

END_GEN = 350
START_POP = 200
WINNERS_PER_GENERATION = 15
MAX_STEPS = 500
TARGET = (35, 2)

STARTPOS = (0, 3)
TARGETS = [(-12, 2), (40, 2)]    # Run
#TARGETS = [(16.5, 4), (16.5, 4)]	# Jump
SCORE_MIN = False

# [10, 24, 24, 24, 24, 4]    # Cubotron1000
# [15, 30, 30, 30, 6]    # Boulotron2000
NEURON_LAYERS = [15, 30, 30, 12, 6] # including input and output layers
ACTIVATION = "tanh"

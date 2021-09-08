#PRESENTATION = False	# Without display but enables evolution
PRESENTATION = True	# With display ON and evolution OFF


DISPLAY = False
BREED = True
if PRESENTATION:
    DISPLAY = True
    BREED = False

PLOT_EVOLUTION = True


START_POP = 200
WINNERS_PER_GENERATION = 15
END_GEN = 350
MAX_STEPS = 500


STARTPOS = (-3, 3)
TARGET = (35, 2)
TARGETS = [(-12, 2), (40, 2)]    # Run
#TARGETS = [(16.5, 4), (16.5, 4)]	# Jump
SCORE_MIN = False


ACTIVATION = "tanh"
HIDDEN_LAYERS = [30, 30]
# Recommended
# [24, 24, 24, 24, 4]    # Cubotron1000
# [15, 30, 30, 30, 6]    # Boulotron2000

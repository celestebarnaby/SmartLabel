# LearnSy constants

NUM_LEARNSY_SAMPLES = 10
W_DEFAULT = .5

# Active learning constants 

NUM_INITIAL_EXAMPLES = 2
NUM_SAMPLES= 50

# Image editing constants

IMG_EDIT_DIR = "./image_edit_directories/{}.json"
MAX_PRED_SET_SIZE = 512
PARTIAL_AMT = .05
MAX_PARTIAL_SAMPLES = 10
MAX_AST_SIZE = 6
MAX_AST_DEPTH = 5
ATTRIBUTES = [
    "EyesOpen",
    "Smile",
    "MouthOpen",
]
MIN_IOU = .5
MIN_SAMPLES = 3

INDIST_INPS = []

# Conformal prediction constants

PERTURB_AMT = .1
GT_CONFIDENCE = 80
PRED_CONFIDENCE = 70

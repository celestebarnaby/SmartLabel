# LearnSy constants

NUM_LEARNSY_SAMPLES = 10
W_DEFAULT = .5

# Active learning constants 

NUM_INITIAL_EXAMPLES = 2
NUM_SAMPLES= 50

# Image editing constants

IMG_EDIT_DIR = "./image_edit_domain/image_edit_datasets/{}.json"
MAX_PRED_SET_SIZE = 512
PARTIAL_AMT = .05
IMG_EDIT_MAX_PARTIAL_SAMPLES = 10
IMG_EDIT_MIN_PARTIAL_SAMPLES = 3
IMAGE_EDIT_AST_SIZE = 6
IMAGE_EDIT_AST_DEPTH = 5
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

MNIST_DELTA = .0002

# MNIST constants

LIST_LENGTH = 3
DIGITS_PER_ITEM = 2
NUM_INPUTS = 100
MNIST_NOISE = 0.03
MNIST_AST_DEPTH = 4
START_SYMBOL = "int"
MAX_DIGIT = 10
MNIST_NUM_PARTIAL_SAMPLES = 1

MNIST_IMGS_DIR = "./mnist_domain/mnist_dataset/mnist.csv"

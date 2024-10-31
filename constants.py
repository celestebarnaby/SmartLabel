# LearnSy constants

NUM_LEARNSY_SAMPLES = 10
W_DEFAULT = .5

# Active learning constants 

NUM_INITIAL_EXAMPLES = 2

# Image editing constants

IMG_EDIT_DIR = "./image_edit_domain/image_edit_datasets/{}.json"
MAX_PRED_SET_SIZE = 512
PARTIAL_AMT = .05
IMAGE_EDIT_NUM_SAMPLES = 50
IMG_EDIT_MAX_PARTIAL_SAMPLES = 10
IMG_EDIT_MIN_PARTIAL_SAMPLES = 3
IMAGE_EDIT_NUM_SAMPLED_SUB_EXPRS = 6
IMAGE_EDIT_INIT_PROG_SPACE_SIZE = 50000
IMAGE_EDIT_AST_SIZE = 8
LEARNSY_IMAGE_EDIT_AST_SIZE = 6
ATTRIBUTES = [
    "EyesOpen",
    "Smile",
    "MouthOpen",
    # "Price",
    # "PhoneNumber"
]
MIN_IOU = .5
MIN_SAMPLES = 3

INDIST_INPS = []

# Conformal prediction constants

PERTURB_AMT = .1
GT_CONFIDENCE = 80
PRED_CONFIDENCE = 70

MNIST_DELTA = .005
# MNIST constants

LIST_LENGTH = 3
DIGITS_PER_ITEM = 1
NUM_INPUTS = 100
MNIST_NUM_SAMPLES= 200
MNIST_AST_DEPTH = 4
START_SYMBOL = "int"
MAX_DIGIT = 10
MNIST_NUM_PARTIAL_SAMPLES = 1

MNIST_IMGS_DIR = "./mnist_domain/mnist_dataset/svhn.csv"
MNIST_QUESTIONS_DIR = "./mnist_domain/input_space.json"

SEED = 35
NUM_SEEDS = 5
